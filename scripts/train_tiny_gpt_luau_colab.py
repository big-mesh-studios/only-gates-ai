#!/usr/bin/env python3
"""Train a Luau Ask/Write model on Colab and export an app bundle.

The model is the same tiny GPT architecture used by Integer GPT. Training is
PyTorch/float; the app quantizes the exported checkpoint for integer-only WASM
inference.

Data comes from ``scripts/luau_tasks.py`` plus the repository's Luau FAQ. Task
examples use a ``Task:`` prefix and answer with code. FAQ examples use a
``Question:`` prefix and answer with prose plus examples.

Examples
--------

First run, warm-starting from an existing Luau Q&A checkpoint::

    python scripts/train_tiny_gpt_luau_colab.py \
      --tasks data/luau-tasks \
      --out /content/drive/MyDrive/tiny-gpt-luau \
      --init-from /content/drive/MyDrive/tiny-gpt-luau-faq/latest.pt \
      --epochs 15 --batch 8 --lr 1e-4 \
      --save-every 200 --max-minutes 300 --export-final

Resume after a Colab disconnect::

    python scripts/train_tiny_gpt_luau_colab.py \
      --tasks data/luau-tasks \
      --out /content/drive/MyDrive/tiny-gpt-luau \
      --resume --epochs 15 --batch 8 --lr 1e-4 --max-minutes 300

CPU smoke test::

    python scripts/train_tiny_gpt_luau_colab.py \
      --tasks data/luau-tasks --out /tmp/luau-smoke --smoke
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import signal
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np
import torch

import model_bundle
import train_tiny_gpt_solid2_colab as base
from bpe_tokenizer import BYTE_DECODER, BpeTokenizer, load_tokenizer
from luau_sft import build_examples


CHECKPOINT_FORMAT = "only-gates-luau-train"
CHECKPOINT_VERSION = 1
CONFIG_KEYS = ("vocabSize", "context", "dim", "layers", "heads", "mlp")
DEFAULT_TOKENIZER = (
    REPO_ROOT
    / "apps"
    / "only-gates-ai"
    / "public"
    / "models"
    / "tiny-gpt-solid2"
    / "tokenizer.json"
)
DEFAULT_FAQ = REPO_ROOT / "apps" / "only-gates-ai" / "src" / "assets" / "luau-faq.md"
TASK_PRESETS = [
    "Write a for loop that sums the first 50 elements of the array `values` and prints the result in Luau. The array has at least 50 elements.",
    "Write a for loop that prints the numbers from 1 to 20 in Luau.",
    "Write a for loop that counts how many elements of `scores` are greater than 80, then prints the count in Luau.",
    "Write a local function named `add` in Luau that takes `a` and `b` and returns their sum. Then print `add(2, 3)`.",
    "Given the array `items = { \"sword\", \"shield\", \"potion\" }`, write a loop in Luau that prints each index and value.",
    "Use a while loop in Luau to add the numbers 1 through 10 and print the total.",
]
FAQ_PRESETS = [
    "How do you do a for loop in Luau?",
    "How do you create a table in Luau?",
    "How do you add type annotations in Luau?",
    "How do you concatenate strings in Luau?",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_task_rows(tasks_dir: Path) -> Tuple[List[Dict], List[Dict], str]:
    train_path = tasks_dir / "train.jsonl"
    val_path = tasks_dir / "val.jsonl"
    if not train_path.exists() or not val_path.exists():
        raise SystemExit(
            f"missing {train_path} or {val_path}; run scripts/luau_tasks.py first"
        )
    train = list(iter_jsonl(train_path))
    val = list(iter_jsonl(val_path))
    if not train or not val:
        raise SystemExit("task train/val files are empty")
    fingerprint = sha256_text(
        json.dumps(
            {
                "train": sha256_file(train_path),
                "val": sha256_file(val_path),
                "counts": [len(train), len(val)],
            },
            sort_keys=True,
        )
    )
    return train, val, fingerprint


def split_faq_pairs(path: Path) -> Tuple[List[Dict], List[Dict]]:
    """Split Markdown FAQ headings into train/validation Ask pairs."""
    if not path.exists():
        return [], []
    text = path.read_text(encoding="utf-8")
    pairs: List[Dict] = []
    title: Optional[str] = None
    body: List[str] = []

    def flush() -> None:
        nonlocal title, body
        if title is None:
            return
        answer = "\n".join(body).strip()
        question = title.strip()
        if not question.endswith("?"):
            question = question.rstrip(".") + "?"
        if answer:
            pairs.append({"instruction": question, "answer": answer, "mode": "ask"})
        title = None
        body = []

    for line in text.splitlines():
        heading = re.match(r"^#{2,6}[ \t]+(.+?)[ \t]*#*[ \t]*$", line)
        if heading:
            flush()
            title = heading.group(1).strip()
            continue
        if title is not None:
            body.append(line)
    flush()

    # Hold out every fifth pair for validation. The FAQ is small, so this is a
    # regression check rather than a statistically strong split.
    train = [pair for index, pair in enumerate(pairs) if index % 5 != 0]
    val = [pair for index, pair in enumerate(pairs) if index % 5 == 0]
    return train, val


def task_item(row: Mapping[str, object]) -> Dict[str, str]:
    return {
        "id": str(row.get("id", "")),
        "family": str(row.get("family", "")),
        "mode": "write",
        "instruction": str(row.get("instruction", "")),
        "answer": str(row.get("code", "")),
    }


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def capture_rng() -> Dict:
    return base.capture_rng()


def restore_rng(value: Optional[Dict]) -> None:
    base.restore_rng(value)


def atomic_save(path: Path, payload: Mapping[str, object]) -> None:
    base.atomic_torch_save(path, dict(payload))


def load_checkpoint(path: Path) -> Dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise SystemExit(f"{path} is not a {CHECKPOINT_FORMAT} checkpoint")
    if int(payload.get("version", -1)) != CHECKPOINT_VERSION:
        raise SystemExit(f"{path} has unsupported checkpoint version {payload.get('version')}")
    return payload


def model_state_cpu(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


# ---------------------------------------------------------------------------
# Init and tokenizer
# ---------------------------------------------------------------------------


def load_init_payload(path: Path) -> Optional[Dict]:
    if path.is_file() and path.suffix == ".pt":
        return torch.load(path, map_location="cpu", weights_only=False)
    if path.is_dir():
        checkpoint_path = path / "checkpoint.json"
        if checkpoint_path.exists():
            return {
                "config": json.loads(checkpoint_path.read_text(encoding="utf-8"))["config"],
                "model_dir": str(path),
                "tokenizer_model": (
                    json.loads((path / "tokenizer.json").read_text(encoding="utf-8"))
                    if (path / "tokenizer.json").exists()
                    else None
                ),
            }
    raise SystemExit(
        f"--init-from must be a .pt checkpoint or a directory with checkpoint.json: {path}"
    )


def unpack_weights(model: torch.nn.Module, packed: np.ndarray) -> None:
    vocab, context, dim, layers, mlp = (
        model.vocab_size,
        model.context,
        model.dim,
        model.layers_n,
        model.mlp_n,
    )
    offset = 0

    def take(count: int) -> torch.Tensor:
        nonlocal offset
        chunk = packed[offset : offset + count]
        if len(chunk) != count:
            raise ValueError("packed weights are truncated")
        offset += count
        return torch.from_numpy(chunk.copy())

    with torch.no_grad():
        model.token_emb.weight.copy_(take(vocab * dim).view(vocab, dim))
        model.pos_emb.weight.copy_(take(context * dim).view(context, dim))
        model.g_final.gain.copy_(take(dim))
        for block in model.blocks:
            block.g1.gain.copy_(take(dim))
            block.wqkv.weight.copy_(take(dim * dim * 3).view(dim, dim * 3).T)
            block.wo.weight.copy_(take(dim * dim).view(dim, dim).T)
            block.g2.gain.copy_(take(dim))
            block.w1.weight.copy_(take(dim * mlp).view(dim, mlp).T)
            block.w2.weight.copy_(take(mlp * dim).view(mlp, dim).T)
    if offset != packed.size:
        raise ValueError("packed weights have trailing values")


def resolve_config(
    args: argparse.Namespace, init_payload: Optional[Dict]
) -> Dict[str, int]:
    if init_payload and init_payload.get("config"):
        config = {key: int(init_payload["config"][key]) for key in CONFIG_KEYS}
        print(
            "using the init checkpoint config: "
            + ", ".join(f"{key}={value}" for key, value in config.items()),
            flush=True,
        )
        return config
    config = {
        "vocabSize": args.vocab,
        "context": args.context,
        "dim": args.dim,
        "layers": args.layers,
        "heads": args.heads,
        "mlp": args.mlp,
    }
    if config["dim"] % config["heads"] != 0:
        raise SystemExit("--dim must be divisible by --heads")
    return config


def resolve_tokenizer(
    args: argparse.Namespace,
    config: Mapping[str, int],
    init_payload: Optional[Dict],
) -> Tuple[BpeTokenizer, Dict]:
    model: Optional[Dict] = None
    if args.tokenizer is not None:
        model = json.loads(args.tokenizer.read_text(encoding="utf-8"))
    elif init_payload and init_payload.get("tokenizer_model"):
        model = init_payload["tokenizer_model"]
    elif init_payload and init_payload.get("model_dir"):
        path = Path(init_payload["model_dir"]) / "tokenizer.json"
        if path.exists():
            model = json.loads(path.read_text(encoding="utf-8"))
    elif DEFAULT_TOKENIZER.exists():
        model = json.loads(DEFAULT_TOKENIZER.read_text(encoding="utf-8"))
    else:
        raise SystemExit(
            "no tokenizer found; pass --tokenizer or place the deployed "
            f"tokenizer at {DEFAULT_TOKENIZER}"
        )
    tokenizer = BpeTokenizer(model)
    if len(tokenizer.vocab) != int(config["vocabSize"]):
        raise SystemExit(
            f"tokenizer has {len(tokenizer.vocab):,} tokens but config needs "
            f"{config['vocabSize']:,}"
        )
    for special in ("<|user|>", "<|assistant|>", "<|end|>", "<|pad|>"):
        if tokenizer.id_for(special) is None:
            raise SystemExit(f"tokenizer is missing {special}")
    return tokenizer, model


def warm_start(
    model: torch.nn.Module,
    config: Mapping[str, int],
    init_payload: Dict,
) -> None:
    if init_payload.get("model") is not None:
        model.load_state_dict(init_payload["model"])
        print(f"warm-started from checkpoint step {init_payload.get('step', '?')}", flush=True)
        return
    model_dir = Path(str(init_payload["model_dir"]))
    weights = model_dir / "weights.bin"
    if not weights.exists():
        raise SystemExit(f"missing {weights} for warm start")
    packed = np.fromfile(weights, dtype="<f4")
    expected = model_bundle.parameter_count(config)
    if packed.size != expected:
        raise SystemExit(
            f"{weights} has {packed.size:,} values but the config needs {expected:,}"
        )
    unpack_weights(model, packed)
    print(f"warm-started from {weights}", flush=True)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def make_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup: int):
    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def batch_tensors(
    examples: Sequence[Tuple[List[int], List[int]]],
    indices: Sequence[int],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    x = torch.tensor([examples[index][0] for index in indices], dtype=torch.long, device=device)
    y = torch.tensor([examples[index][1] for index in indices], dtype=torch.long, device=device)
    return x[:, :-1], y[:, 1:]


@torch.no_grad()
def eval_loss(
    model: torch.nn.Module,
    examples: Sequence[Tuple[List[int], List[int]]],
    batch: int,
    device: torch.device,
) -> float:
    if not examples:
        return float("nan")
    model.eval()
    total = 0.0
    seen = 0
    for start in range(0, len(examples), batch):
        chunk = examples[start : start + batch]
        x = torch.tensor([item[0] for item in chunk], dtype=torch.long, device=device)
        y = torch.tensor([item[1] for item in chunk], dtype=torch.long, device=device)
        _, loss = model(x[:, :-1], y[:, 1:])
        if math.isfinite(float(loss)):
            total += float(loss) * len(chunk)
            seen += len(chunk)
    model.train()
    return total / seen if seen else float("nan")


def decode_tokens(tokenizer: BpeTokenizer, ids: Sequence[int]) -> str:
    special_ids = {
        token_id
        for text, token_id in tokenizer.token_to_id.items()
        if text.startswith("<|")
    }
    pieces = [
        tokenizer.vocab[token_id]
        for token_id in ids
        if token_id not in special_ids and 0 <= token_id < len(tokenizer.vocab)
    ]
    encoded = bytearray()
    for char in "".join(pieces):
        byte = BYTE_DECODER.get(char)
        if byte is not None:
            encoded.append(byte)
    return encoded.decode("utf-8", errors="replace")


@torch.no_grad()
def generate_samples(
    model: torch.nn.Module,
    tokenizer: BpeTokenizer,
    prompts: Sequence[Tuple[str, str]],
    device: torch.device,
    max_new: int = 120,
) -> List[Dict[str, str]]:
    user_id = tokenizer.id_for("<|user|>")
    assistant_id = tokenizer.id_for("<|assistant|>")
    end_id = tokenizer.id_for("<|end|>")
    if user_id is None or assistant_id is None or end_id is None:
        return []
    model.eval()
    samples: List[Dict[str, str]] = []
    for mode, prompt in prompts:
        prefix = "Task: " if mode == "write" else "Question: "
        prompt_ids = [user_id] + tokenizer.encode_ordinary(prefix + prompt) + [assistant_id]
        tokens = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        generated = model.generate(
            tokens,
            max_new=max_new,
            temperature=0.3 if mode == "ask" else 0.1,
            top_k=40,
            eos_id=end_id,
        )[0].tolist()
        text = decode_tokens(tokenizer, generated[len(prompt_ids) :]).strip()
        samples.append({"prompt": prompt, "text": text})
        print(f"sample> [{mode}] {prompt} => {text[:160]!r}", flush=True)
    return samples


def export_bundle(
    *,
    out_dir: Path,
    bundle_path: Path,
    config: Mapping[str, int],
    tokenizer_model: Mapping[str, object],
    tokenizer: BpeTokenizer,
    model: torch.nn.Module,
    prompts: Sequence[Tuple[str, str]],
    history: Sequence[Mapping[str, object]],
    step: int,
    best_step: int,
    best_loss: float,
    elapsed_seconds: float,
    args: argparse.Namespace,
    corpus_hash: str,
) -> Path:
    packed = base.pack_weights(model)
    base.verify_packed(model, packed, dict(config))
    weights = packed.numpy().astype("<f4", copy=False).tobytes()
    samples = generate_samples(model, tokenizer, prompts, torch.device(args.device))
    prompt_texts = [prompt for mode, prompt in prompts if mode == "write"][:8]
    training = {
        "steps": int(step),
        "batch": args.batch,
        "learningRate": args.lr,
        "dropout": args.dropout,
        "seed": args.seed,
        "bestStep": int(best_step),
        "bestLoss": best_loss if math.isfinite(best_loss) else None,
        "elapsedMs": int(max(0.0, elapsed_seconds) * 1000),
        "corpusHash": corpus_hash,
        "sourceUrls": ["data/luau-tasks/train.jsonl", str(DEFAULT_FAQ)],
    }
    written = model_bundle.write_model_bundle(
        out_dir,
        bundle_path,
        config=config,
        tokenizer_model=tokenizer_model,
        weights=weights,
        name=args.name,
        source=(
            "Generated Luau beginner tasks from scripts/luau_tasks.py plus the "
            "repository's Luau quick-answer FAQ. Task answers are execution "
            "checked before training."
        ),
        license=(
            "Generated task data and model weights are produced by this "
            "repository. Luau documentation and the FAQ retain their own MIT "
            "licenses."
        ),
        prompts=prompt_texts,
        samples=samples,
        training=training,
        history=history,
        assistant=True,
        kind="custom",
        checkpoint_format="only-gates-tiny-gpt-luau",
        generator_feature="luau-training",
    )
    print(f"\nwrote {written} ({written.stat().st_size:,} bytes)", flush=True)
    print(
        "Copy checkpoint.json, tokenizer.json, and weights.bin into "
        "apps/only-gates-ai/public/models/tiny-gpt-luau/ to publish in the Luau tab.",
        flush=True,
    )
    return written


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--tasks", type=Path, default=REPO_ROOT / "data" / "luau-tasks")
    parser.add_argument("--out", type=Path, default=Path("tiny-gpt-luau-out"))
    parser.add_argument("--init-from", type=Path, default=None)
    parser.add_argument("--tokenizer", type=Path, default=None)
    parser.add_argument("--faq", type=Path, default=DEFAULT_FAQ)
    parser.add_argument("--faq-repeat", type=int, default=20)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-fraction", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--vocab", type=int, default=4096)
    parser.add_argument("--context", "--ctx", dest="context", type=int, default=256)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--mlp", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--name", default="tiny-gpt-luau")
    parser.add_argument("--bundle-out", type=Path, default=None)
    parser.add_argument("--max-minutes", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--export-final", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.context, args.dim, args.layers, args.heads, args.mlp = 128, 32, 1, 2, 64
        args.epochs, args.batch = 1, 2
        args.max_steps, args.save_every, args.val_every = 2, 1, 1
        args.init_from = None
        args.faq_repeat = 1
        print("SMOKE mode: tiny config, two optimizer steps", flush=True)
    if args.batch < 1:
        raise SystemExit("--batch must be at least 1")
    if args.epochs < 1 and not args.export_only:
        raise SystemExit("--epochs must be at least 1")
    if args.max_minutes < 0:
        raise SystemExit("--max-minutes must be non-negative")

    signal.signal(signal.SIGINT, base.request_stop)
    signal.signal(signal.SIGTERM, base.request_stop)

    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.out / "latest.pt"
    bundle_path = args.bundle_out or (args.out / "only-gates-luau-bundle.zip")
    device = torch.device(args.device)

    checkpoint: Optional[Dict] = None
    if not args.fresh and (args.resume or args.export_only) and checkpoint_path.exists():
        checkpoint = load_checkpoint(checkpoint_path)
        print(f"loaded {checkpoint_path} at step {checkpoint.get('step', 0)}", flush=True)
    elif args.export_only:
        raise SystemExit("--export-only needs an existing latest.pt")

    init_payload: Optional[Dict] = None
    if checkpoint is None and args.init_from is not None:
        init_payload = load_init_payload(args.init_from)

    if checkpoint is not None:
        config = {key: int(checkpoint["config"][key]) for key in CONFIG_KEYS}
        tokenizer_model = checkpoint["tokenizer_model"]
        tokenizer = BpeTokenizer(tokenizer_model)
    else:
        config = resolve_config(args, init_payload)
        tokenizer, tokenizer_model = resolve_tokenizer(args, config, init_payload)

    if args.export_only and checkpoint is not None:
        args.batch = int(checkpoint.get("batch", args.batch))
        args.lr = float(checkpoint.get("lr", args.lr))
        args.dropout = float(checkpoint.get("dropout", args.dropout))
        args.seed = int(checkpoint.get("seed", args.seed))
        args.name = str(checkpoint.get("name", args.name))
        model = base.TinyGPT(
            config["vocabSize"],
            config["context"],
            config["dim"],
            config["layers"],
            config["heads"],
            config["mlp"],
            args.dropout,
        ).to(device)
        model.load_state_dict(checkpoint["model"])
        if checkpoint.get("best_model") is not None and not args.export_final:
            model.load_state_dict(checkpoint["best_model"])
        export_bundle(
            out_dir=args.out,
            bundle_path=bundle_path,
            config=config,
            tokenizer_model=tokenizer_model,
            tokenizer=tokenizer,
            model=model,
            prompts=[("write", prompt) for prompt in TASK_PRESETS[:5]]
            + [("ask", prompt) for prompt in FAQ_PRESETS[:3]],
            history=checkpoint.get("history", []),
            step=int(checkpoint.get("step", 0)),
            best_step=int(checkpoint.get("best_step", 0)),
            best_loss=float(checkpoint.get("best_loss", float("inf"))),
            elapsed_seconds=float(checkpoint.get("elapsed_seconds", 0.0)),
            args=args,
            corpus_hash=str(checkpoint.get("corpus_hash", "")),
        )
        return

    train_rows, val_rows, task_hash = load_task_rows(args.tasks)
    faq_train, faq_val = split_faq_pairs(args.faq)
    train_items: List[Dict[str, str]] = [task_item(row) for row in train_rows]
    val_items: List[Dict[str, str]] = [task_item(row) for row in val_rows]
    for pair in faq_train:
        train_items.extend([dict(pair) for _ in range(max(1, args.faq_repeat))])
    val_items.extend(faq_val)
    if args.smoke:
        train_items = [task_item(row) for row in train_rows[:32]]
        val_items = [task_item(row) for row in val_rows[:8]]

    corpus_hash = sha256_text(
        json.dumps(
            {
                "tasks": task_hash,
                "faq": sha256_text(
                    args.faq.read_text(encoding="utf-8") if args.faq.exists() else ""
                ),
                "faq_repeat": args.faq_repeat,
                "seed": args.seed,
            },
            sort_keys=True,
        )
    )

    if checkpoint is not None:
        for key, value in (
            ("config", config),
            ("seed", args.seed),
            ("corpus_hash", corpus_hash),
        ):
            old = checkpoint.get(key)
            if key == "config":
                old = {name: int(old[name]) for name in CONFIG_KEYS} if old else old
            if old != value:
                raise SystemExit(
                    f"checkpoint {key} differs; use --fresh to start a new run "
                    f"(checkpoint={old!r}, current={value!r})"
                )

    train_examples, skipped_train = build_examples(tokenizer, train_items, config["context"])
    val_examples, skipped_val = build_examples(tokenizer, val_items, config["context"])
    if not train_examples:
        raise SystemExit("no training examples fit the context")
    if not val_examples:
        raise SystemExit("no validation examples fit the context")
    print(
        f"data: {len(train_examples):,} train, {len(val_examples):,} val, "
        f"{len(faq_train):,} FAQ train pairs, skipped {skipped_train:,}/{skipped_val:,}\n"
        f"config: {config['vocabSize']} vocab / {config['context']} ctx / "
        f"{config['dim']} dim / {config['layers']} layers / "
        f"{config['heads']} heads / {config['mlp']} mlp "
        f"({model_bundle.parameter_count(config):,} params)",
        flush=True,
    )

    model = base.TinyGPT(
        config["vocabSize"],
        config["context"],
        config["dim"],
        config["layers"],
        config["heads"],
        config["mlp"],
        args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay
    )
    steps_per_epoch = max(1, math.ceil(len(train_examples) / args.batch))
    planned_steps = steps_per_epoch * args.epochs
    if args.max_steps > 0:
        planned_steps = min(planned_steps, args.max_steps)
    scheduler = make_scheduler(
        optimizer, planned_steps, max(1, int(planned_steps * args.warmup_fraction))
    )

    start_epoch = 0
    start_cursor = 0
    start_perm: Optional[List[int]] = None
    step = 0
    history: List[Dict[str, object]] = []
    best_loss = float("inf")
    best_step = 0
    best_model: Optional[Dict[str, torch.Tensor]] = None
    elapsed_base = 0.0

    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        restore_rng(checkpoint.get("rng"))
        start_epoch = int(checkpoint.get("epoch", 0))
        start_cursor = int(checkpoint.get("cursor", 0))
        start_perm = checkpoint.get("perm")
        step = int(checkpoint.get("step", 0))
        history = list(checkpoint.get("history", []))
        best_loss = float(checkpoint.get("best_loss", float("inf")))
        best_step = int(checkpoint.get("best_step", 0))
        best_model = checkpoint.get("best_model")
        elapsed_base = float(checkpoint.get("elapsed_seconds", 0.0))
        print(
            f"resuming at epoch {start_epoch + 1}/{args.epochs}, step {step}, "
            f"cursor {start_cursor}",
            flush=True,
        )
    elif init_payload is not None:
        warm_start(model, config, init_payload)

    session_start = time.monotonic()
    deadline = session_start + args.max_minutes * 60.0 if args.max_minutes > 0 else None

    def elapsed_seconds() -> float:
        return elapsed_base + (time.monotonic() - session_start)

    def save(
        epoch: int,
        cursor: int,
        perm: Optional[List[int]],
        current_step: int,
    ) -> None:
        atomic_save(
            checkpoint_path,
            {
                "format": CHECKPOINT_FORMAT,
                "version": CHECKPOINT_VERSION,
                "config": config,
                "seed": args.seed,
                "batch": args.batch,
                "lr": args.lr,
                "dropout": args.dropout,
                "name": args.name,
                "corpus_hash": corpus_hash,
                "tokenizer_model": tokenizer_model,
                "epoch": epoch,
                "cursor": cursor,
                "perm": perm,
                "step": current_step,
                "model": model_state_cpu(model),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "rng": capture_rng(),
                "history": history,
                "best_loss": best_loss,
                "best_step": best_step,
                "best_model": best_model,
                "elapsed_seconds": elapsed_seconds(),
            },
        )

    if args.export_only:
        if best_model is not None and not args.export_final:
            model.load_state_dict(best_model)
        export_bundle(
            out_dir=args.out,
            bundle_path=bundle_path,
            config=config,
            tokenizer_model=tokenizer_model,
            tokenizer=tokenizer,
            model=model,
            prompts=[("write", prompt) for prompt in TASK_PRESETS[:5]]
            + [("ask", prompt) for prompt in FAQ_PRESETS[:3]],
            history=history,
            step=step,
            best_step=best_step,
            best_loss=best_loss,
            elapsed_seconds=elapsed_seconds(),
            args=args,
            corpus_hash=corpus_hash,
        )
        return

    model.train()
    stop_requested = False
    epoch = start_epoch
    current_step = step
    while epoch < args.epochs:
        if start_perm is not None:
            perm = list(start_perm)
        else:
            perm = list(range(len(train_examples)))
            random.shuffle(perm)
        begin = start_cursor if epoch == start_epoch else 0
        start_perm = None
        start_cursor = 0
        run_loss = 0.0
        run_count = 0
        index = begin
        while index < len(perm):
            if base.STOP_REQUESTED or (deadline is not None and time.monotonic() >= deadline):
                stop_requested = True
                break
            if args.max_steps > 0 and current_step >= args.max_steps:
                stop_requested = True
                break
            batch_indices = perm[index : index + args.batch]
            x, y = batch_tensors(train_examples, batch_indices, device)
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            scheduler.step()
            current_step += 1
            loss_value = float(loss.detach())
            if not math.isfinite(loss_value):
                raise SystemExit(f"non-finite loss at step {current_step}")
            run_loss += loss_value * len(batch_indices)
            run_count += len(batch_indices)
            index += len(batch_indices)
            if current_step % 10 == 0:
                print(
                    f"epoch {epoch + 1}/{args.epochs} step {current_step} "
                    f"loss {loss_value:.4f} lr {scheduler.get_last_lr()[0]:.2e} "
                    f"{elapsed_seconds():.1f}s",
                    flush=True,
                )
            if args.save_every > 0 and current_step % args.save_every == 0:
                save(epoch, index, perm, current_step)
        if stop_requested:
            save(epoch, index, perm, current_step)
            print(
                f"stopped at epoch {epoch + 1}/{args.epochs}, step {current_step}; "
                "rerun with --resume to continue",
                flush=True,
            )
            break

        if run_count:
            print(
                f"epoch {epoch + 1}/{args.epochs} train "
                f"{run_loss / run_count:.4f}",
                flush=True,
            )
        if (epoch + 1) % max(1, args.val_every) == 0:
            value = eval_loss(model, val_examples, args.batch, device)
            point = {
                "step": current_step,
                "epoch": epoch + 1,
                "loss": run_loss / max(1, run_count),
                "valLoss": value if math.isfinite(value) else None,
                "lr": scheduler.get_last_lr()[0],
                "elapsedMs": elapsed_seconds() * 1000.0,
            }
            history.append(point)
            print(
                f"epoch {epoch + 1}/{args.epochs} val "
                f"{value:.4f}" if math.isfinite(value) else "val n/a",
                flush=True,
            )
            if math.isfinite(value) and value < best_loss:
                best_loss = value
                best_step = current_step
                best_model = model_state_cpu(model)
        save(epoch + 1, 0, None, current_step)
        epoch += 1

    save(epoch, 0, None, current_step)
    if best_model is not None and not args.export_final:
        model.load_state_dict(best_model)
    export_bundle(
        out_dir=args.out,
        bundle_path=bundle_path,
        config=config,
        tokenizer_model=tokenizer_model,
        tokenizer=tokenizer,
        model=model,
        prompts=[("write", prompt) for prompt in TASK_PRESETS[:5]]
        + [("ask", prompt) for prompt in FAQ_PRESETS[:3]],
        history=history,
        step=current_step,
        best_step=best_step,
        best_loss=best_loss,
        elapsed_seconds=elapsed_seconds(),
        args=args,
        corpus_hash=corpus_hash,
    )


if __name__ == "__main__":
    main()
