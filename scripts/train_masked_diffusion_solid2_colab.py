#!/usr/bin/env python3
"""Train a Solid 2 masked-diffusion denoiser on Colab and export app artifacts.

This script trains a bidirectional Tiny GPT with an absorbing <|mask|> token
on Solid 2 documentation and Q&A data, targeting ~10 million parameters.

The model shares the packed-weight layout of the causal Tiny GPT, so the browser
loads it with `tiny_gpt_create` and runs one integer-only denoising pass per
animation frame.

Unlike the causal Solid 2 model, every answer position is predicted in parallel.
Training samples a mask ratio t ~ U(0, 1) (with a small full-mask
probability), masks that fraction of the answer region, and computes cross
entropy only on the masked positions.

Examples
--------

Train from scratch on a T4, with resumable checkpoints::

    python scripts/train_masked_diffusion_solid2_colab.py \\
      --out /content/drive/MyDrive/masked-diffusion-solid2 \\
      --epochs 40 --batch 16 --lr 3e-4 --max-minutes 240

CPU smoke test::

    python scripts/train_masked_diffusion_solid2_colab.py \\
      --out /tmp/masked-smoke --smoke
"""

from __future__ import annotations

import argparse
import json
import math
import random
import signal
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path("/content/only-gates-ai/scripts")
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import torch
import torch.nn as nn
import torch.nn.functional as F

import model_bundle
import train_tiny_gpt_solid2_colab as base
from bpe_tokenizer import BpeTokenizer

CHECKPOINT_FORMAT = "only-gates-masked-diffusion-solid2"
CHECKPOINT_VERSION = 1
CONFIG_KEYS = ("vocabSize", "context", "dim", "layers", "heads", "mlp")
MASK_TOKEN = "<|mask|>"
SAMPLE_PROMPTS: Sequence[Tuple[str, str]] = (
    ("ask", "What are split effects in Solid 2?"),
    ("ask", "How do I migrate createEffect from Solid 1?"),
    ("ask", "What is the new createEffect API?"),
    ("ask", "How do derive and computed signals differ in Solid 2?"),
    ("ask", "What are the rules for writing to signals under owned scope?"),
)
DEFAULT_TOKENIZER = (
    REPO_ROOT / "apps" / "only-gates-ai" / "public" / "models" / "tiny-gpt-solid2" / "tokenizer.json"
)


def load_masked_tokenizer(path: Path) -> Tuple[BpeTokenizer, Dict]:
    """Load the Solid 2 BPE model and append an absorbing mask token."""
    if not path.exists():
        raise SystemExit(
            f"missing tokenizer {path}; run scripts/train_tiny_gpt_solid2_colab.py first "
            "or pass --tokenizer"
        )
    model = json.loads(path.read_text(encoding="utf-8"))
    specials = list(model.get("specialTokens", []))
    vocab = list(model["vocab"])
    if MASK_TOKEN not in specials:
        specials.append(MASK_TOKEN)
    if MASK_TOKEN not in vocab:
        vocab.append(MASK_TOKEN)
    model["specialTokens"] = specials
    model["vocab"] = vocab
    tokenizer = BpeTokenizer(model)
    mask_id = tokenizer.id_for(MASK_TOKEN)
    if mask_id is None:
        raise SystemExit(f"tokenizer {path} is missing {MASK_TOKEN}")
    return tokenizer, model


class BidirectionalBlock(nn.Module):
    """Pre-norm transformer block with no attention mask (bidirectional)."""

    def __init__(self, dim: int, heads: int, mlp: int, dropout: float):
        super().__init__()
        if dim % heads:
            raise ValueError("dim % heads != 0")
        self.heads = heads
        self.head_dim = dim // heads
        self.g1 = base.RMSNorm(dim)
        self.g2 = base.RMSNorm(dim)
        self.wqkv = nn.Linear(dim, dim * 3, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.w1 = nn.Linear(dim, mlp, bias=False)
        self.w2 = nn.Linear(mlp, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, dim = x.shape
        h = self.g1(x)
        qkv = self.wqkv(h).view(batch, length, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attended = (torch.softmax(scores, -1) @ v).transpose(1, 2)
        x = x + self.dropout(self.wo(attended.reshape(batch, length, dim)))
        h = self.g2(x)
        x = x + self.dropout(self.w2(F.gelu(self.w1(h), approximate="tanh")))
        return x


class MaskedDiffusionGPT(base.TinyGPT):
    """Same weights and forward as TinyGPT, with bidirectional blocks."""

    def __init__(
        self,
        vocab: int,
        context: int,
        dim: int,
        layers: int,
        heads: int,
        mlp: int,
        dropout: float,
    ):
        super().__init__(vocab, context, dim, layers, heads, mlp, dropout)
        self.blocks = nn.ModuleList(
            [BidirectionalBlock(dim, heads, mlp, dropout) for _ in range(layers)]
        )
        self.apply(self._init)
        for block in self.blocks:
            nn.init.normal_(block.wo.weight, std=0.02 / math.sqrt(2 * max(1, layers)))
            nn.init.normal_(block.w2.weight, std=0.02 / math.sqrt(2 * max(1, layers)))


def model_state_cpu(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def mask_region(
    region: Sequence[int],
    mask_id: int,
    pad_id: int,
    full_mask_probability: float,
) -> Tuple[List[int], List[int]]:
    real = [i for i, token in enumerate(region) if token != pad_id]
    if not real:
        return list(region), [-100] * len(region)
    ratio = random.random()
    if random.random() < full_mask_probability:
        ratio = 1.0
    length = len(region)
    masked = [random.random() < ratio for _ in range(length)]
    if not any(masked[i] for i in real):
        masked[random.choice(real)] = True
    input_region = [mask_id if masked[i] else region[i] for i in range(length)]
    targets = [
        region[i] if masked[i] and region[i] != pad_id else -100
        for i in range(length)
    ]
    return input_region, targets


def build_batch(
    examples: Sequence[Tuple[List[int], List[int]]],
    indices: Sequence[int],
    context: int,
    pad_id: int,
    mask_id: int,
    device: torch.device,
    full_mask_probability: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.full((len(indices), context), pad_id, dtype=torch.long)
    targets = torch.full((len(indices), context), -100, dtype=torch.long)
    for row, index in enumerate(indices):
        prompt, region = examples[index]
        prompt_length = len(prompt)
        input_region, target_region = mask_region(
            region, mask_id, pad_id, full_mask_probability
        )
        inputs[row, :prompt_length] = torch.tensor(prompt, dtype=torch.long)
        inputs[row, prompt_length:] = torch.tensor(input_region, dtype=torch.long)
        targets[row, prompt_length:] = torch.tensor(target_region, dtype=torch.long)
    return inputs.to(device), targets.to(device)


def build_full_mask_batch(
    examples: Sequence[Tuple[List[int], List[int]]],
    context: int,
    mask_id: int,
    pad_id: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.full((len(examples), context), mask_id, dtype=torch.long)
    targets = torch.full((len(examples), context), -100, dtype=torch.long)
    for row, (prompt, region) in enumerate(examples):
        prompt_length = len(prompt)
        inputs[row, :prompt_length] = torch.tensor(prompt, dtype=torch.long)
        real = [token if token != pad_id else -100 for token in region]
        targets[row, prompt_length:] = torch.tensor(real, dtype=torch.long)
    return inputs.to(device), targets.to(device)


@torch.no_grad()
def eval_full_mask_loss(
    model: torch.nn.Module,
    examples: Sequence[Tuple[List[int], List[int]]],
    context: int,
    mask_id: int,
    pad_id: int,
    batch: int,
    device: torch.device,
) -> float:
    if not examples:
        return float("nan")
    model.eval()
    total = 0.0
    count = 0
    for start in range(0, len(examples), max(1, batch)):
        chunk = examples[start : start + batch]
        x, y = build_full_mask_batch(chunk, context, mask_id, pad_id, device)
        logits, _ = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        supervised = int((y != -100).sum().item())
        total += float(loss.item())
        count += supervised
    model.train()
    return total / count if count else float("nan")


def pack_diffusion_example(
    tokenizer: BpeTokenizer,
    item: Mapping[str, str],
    context: int,
    min_answer: int = 16,
) -> Optional[Tuple[List[int], List[int]]]:
    """Return (prompt_ids, answer_region) with region length context - prompt."""
    pad_id = tokenizer.id_for("<|pad|>")
    end_id = tokenizer.id_for(">")
    if pad_id is None or end_id is None:
        raise SystemExit("tokenizer must define <|pad|> and >")

    prefix = "Question: "
    prompt_ids = tokenizer.encode(f"<|user|>{prefix}{item['instruction']}<|assistant|>")
    answer_ids = tokenizer.encode_ordinary(item["answer"]) + [end_id]
    available = context - len(prompt_ids)
    if available < min_answer:
        return None
    if len(answer_ids) > available:
        answer_ids = answer_ids[: available - 1] + [end_id]
    region = answer_ids + [pad_id] * (available - len(answer_ids))
    if len(region) != available:
        return None
    return prompt_ids, region


def build_examples(
    tokenizer: BpeTokenizer,
    items: Sequence[Mapping[str, str]],
    context: int,
) -> Tuple[List[Tuple[List[int], List[int]]], int]:
    examples: List[Tuple[List[int], List[int]]] = []
    skipped = 0
    for item in items:
        packed = pack_diffusion_example(tokenizer, item, context)
        if packed is None:
            skipped += 1
        else:
            examples.append(packed)
    return examples, skipped


def export_artifacts(
    *,
    out_dir: Path,
    config: Mapping[str, int],
    tokenizer_model: Mapping[str, object],
    tokenizer: BpeTokenizer,
    model: torch.nn.Module,
    mask_id: int,
    history: Sequence[Mapping[str, object]],
    step: int,
    best_step: int,
    best_loss: float,
    elapsed_seconds: float,
    args: argparse.Namespace,
    corpus_hash: str,
) -> None:
    packed = base.pack_weights(model)
    base.verify_packed(model, packed, dict(config))
    weights = packed.numpy().astype("<f4", copy=False).tobytes()
    expected = model_bundle.parameter_count(dict(config)) * 4
    if len(weights) != expected:
        raise SystemExit(f"weights are {len(weights):,} bytes, expected {expected:,}")

    checkpoint = {
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        "config": {key: int(config[key]) for key in CONFIG_KEYS},
        "parameterCount": model_bundle.parameter_count(dict(config)),
        "tokenizer": "tokenizer.json",
        "diffusion": {
            "maskTokenId": mask_id,
            "maskToken": MASK_TOKEN,
            "steps": args.steps,
            "schedule": "left-to-right",
            "answerBudget": 0,
        },
        "meta": {
            "name": args.name,
            "source": (
                "Solid 2 documentation and curated Q&A, trained with "
                "an absorbing <|mask|> token."
            ),
            "license": (
                "Solid 2 documentation is MIT licensed. Model weights "
                "are produced by this repository."
            ),
            "training": {
                "steps": int(step),
                "batch": args.batch,
                "learningRate": args.lr,
                "dropout": args.dropout,
                "seed": args.seed,
                "bestStep": int(best_step),
                "bestLoss": best_loss if math.isfinite(best_loss) else None,
                "elapsedMs": int(max(0.0, elapsed_seconds) * 1000),
                "corpusHash": corpus_hash,
                "maskRatio": "uniform+t-full",
                "fullMaskProbability": args.full_mask_prob,
                "denoisingSteps": args.steps,
            },
        },
        "samples": [],
        "prompts": [p for _, p in SAMPLE_PROMPTS],
        "history": list(history),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "weights.bin").write_bytes(weights)
    (out_dir / "tokenizer.json").write_text(
        json.dumps(tokenizer_model, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "checkpoint.json").write_text(
        json.dumps(checkpoint, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    readme = (
        "# Masked-diffusion Solid 2 model\n\n"
        f"Config: {config['vocabSize']:,} vocab / {config['context']} context / "
        f"{config['dim']} dim / {config['layers']} layers / "
        f"{config['heads']} heads / {config['mlp']} mlp\n"
        f"Parameters: {model_bundle.parameter_count(dict(config)):,}\n"
        f"Mask token id: {mask_id}\n"
        f"Denoising steps: {args.steps}\n\n"
        "Files:\n"
        "  checkpoint.json  model metadata and diffusion schedule\n"
        "  tokenizer.json   byte-level BPE with <|mask|>\n"
        "  weights.bin      float32 little-endian packed weights\n\n"
        "Generated by scripts/train_masked_diffusion_solid2_colab.py.\n"
    )
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    print(f"\nwrote {out_dir / 'checkpoint.json'}", flush=True)
    print(f"wrote {out_dir / 'tokenizer.json'}", flush=True)
    print(f"wrote {out_dir / 'weights.bin'} ({len(weights):,} bytes)", flush=True)
    print(
        "Copy these three files into "
        "apps/only-gates-ai/public/models/solid2-masked-diffusion/ to publish "
        "the Solid 2 masked-diffusion tab.",
        flush=True,
    )


def atomic_save(path: Path, payload: Mapping[str, object]) -> None:
    base.atomic_torch_save(path, dict(payload))


def load_checkpoint(path: Path) -> Dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise SystemExit(f"{path} is not a {CHECKPOINT_FORMAT} checkpoint")
    if int(payload.get("version", -1)) != CHECKPOINT_VERSION:
        raise SystemExit(f"{path} has unsupported version {payload.get('version')}")
    return payload


def make_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup: int):
    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=Path("masked-diffusion-solid2-out"))
    parser.add_argument("--tokenizer", type=Path, default=None)
    parser.add_argument("--docs-dir", type=Path, default=None, help="Local dir with Solid 2 docs")
    parser.add_argument("--qa-json", type=Path, default=None, help="Path to qa.json or qa-compact list")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-fraction", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--context", "--ctx", dest="context", type=int, default=256)
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--mlp", type=int, default=1152)
    parser.add_argument("--steps", type=int, default=16, help="denoising steps at inference")
    parser.add_argument(
        "--full-mask-prob",
        type=float,
        default=0.15,
        help="probability of training on a fully masked answer region",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--name", default="masked-diffusion-solid2")
    parser.add_argument("--max-minutes", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--export-final", action="store_true")
    parser.add_argument("--early-stop-patience", type=int, default=6)
    parser.add_argument("--sample-count", type=int, default=6)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.context, args.dim, args.layers, args.heads, args.mlp = 128, 32, 1, 2, 64
        args.epochs, args.batch = 1, 2
        args.max_steps, args.save_every, args.val_every = 2, 1, 1
        args.steps, args.sample_count = 4, 2
        args.name = "masked-diffusion-solid2-smoke"
        print("SMOKE mode: tiny config, two optimizer steps", flush=True)
    if args.batch < 1:
        raise SystemExit("--batch must be at least 1")
    if args.steps < 1:
        raise SystemExit("--steps must be at least 1")
    if not 0.0 <= args.full_mask_prob <= 1.0:
        raise SystemExit("--full-mask-prob must be in [0, 1]")
    if args.epochs < 1 and not args.export_only:
        raise SystemExit("--epochs must be at least 1")
    if args.max_minutes < 0:
        raise SystemExit("--max-minutes must be non-negative")

    signal.signal(signal.SIGINT, base.request_stop)
    signal.signal(signal.SIGTERM, base.request_stop)

    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.out / "latest.pt"
    device = torch.device(args.device)

    checkpoint: Optional[Dict] = None
    if not args.fresh and (args.resume or args.export_only) and checkpoint_path.exists():
        checkpoint = load_checkpoint(checkpoint_path)
        print(f"loaded {checkpoint_path} at step {checkpoint.get('step', 0)}", flush=True)
    elif args.export_only:
        raise SystemExit("--export-only needs an existing latest.pt")

    tokenizer_path = (
        args.tokenizer
        or (Path(checkpoint["tokenizer_path"]) if checkpoint and checkpoint.get("tokenizer_path") else None)
        or DEFAULT_TOKENIZER
    )
    if checkpoint is not None:
        config = {key: int(checkpoint["config"][key]) for key in CONFIG_KEYS}
        tokenizer_model = checkpoint["tokenizer_model"]
        tokenizer = BpeTokenizer(tokenizer_model)
    else:
        tokenizer, tokenizer_model = load_masked_tokenizer(tokenizer_path)
        config = {
            "vocabSize": len(tokenizer.vocab),
            "context": args.context,
            "dim": args.dim,
            "layers": args.layers,
            "heads": args.heads,
            "mlp": args.mlp,
        }
        if config["dim"] % config["heads"] != 0:
            raise SystemExit("--dim must be divisible by --heads")
    mask_id = tokenizer.id_for(MASK_TOKEN)
    pad_id = tokenizer.id_for("<|pad|>")
    if mask_id is None or pad_id is None:
        raise SystemExit(f"tokenizer is missing {MASK_TOKEN} or <|pad|>")
    if args.export_only and checkpoint is not None:
        args.batch = int(checkpoint.get("batch", args.batch))
        args.lr = float(checkpoint.get("lr", args.lr))
        args.dropout = float(checkpoint.get("dropout", args.dropout))
        args.seed = int(checkpoint.get("seed", args.seed))
        args.name = str(checkpoint.get("name", args.name))
        args.full_mask_prob = float(
            checkpoint.get("full_mask_prob", args.full_mask_prob)
        )
        model = MaskedDiffusionGPT(
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
        export_artifacts(
            out_dir=args.out,
            config=config,
            tokenizer_model=tokenizer_model,
            tokenizer=tokenizer,
            model=model,
            mask_id=mask_id,
            history=checkpoint.get("history", []),
            step=int(checkpoint.get("step", 0)),
            best_step=int(checkpoint.get("best_step", 0)),
            best_loss=float(checkpoint.get("best_loss", float("inf"))),
            elapsed_seconds=float(checkpoint.get("elapsed_seconds", 0.0)),
            args=args,
            corpus_hash=str(checkpoint.get("corpus_hash", "")),
        )
        return

    # Load Solid 2 data
    docs_dir = args.docs_dir
    docs = None
    qa = None
    if docs_dir is None:
        # Try to find local data
        for p in [Path("data/solid2"), Path("/content/data/solid2"), REPO_ROOT / "data" / "solid2"]:
            if p.is_dir():
                docs_dir = p
                break
    if docs_dir is None:
        raise SystemExit("Could not find Solid 2 docs directory. Pass --docs-dir.")

    import train_tiny_gpt_solid2_colab as t2
    docs = t2.load_docs(docs_dir)
    qa = t2.load_qa(args.qa_json)

    # Build QA examples - oversample to balance with doc chunks
    qa_items = []
    for d in qa:
        for q in d["questions"]:
            qa_items.append({"instruction": q, "answer": d["answer"], "mode": "ask"})
    # Oversample QA 8x to match the integer GPT trainer
    qa_items = qa_items * 8

    # Build doc examples by splitting docs into sections by headings
    doc_items = []
    for name, text in docs:
        # Skip supplemental docs for training (they're too noisy)
        if name.startswith("[SUPPLEMENTAL"):
            continue
        # Split by headings to create smaller, focused chunks
        import re
        sections = re.split(r'\n(?=#{1,3} )', text)
        for section in sections:
            section = section.strip()
            if len(section) < 50:
                continue
            # Use first line as topic, rest as content
            lines = section.split('\n', 1)
            topic = lines[0].strip().lstrip('#').strip()
            content = lines[1].strip() if len(lines) > 1 else section
            if len(content) < 30:
                continue
            # Truncate content to fit context
            if len(content) > 400:
                content = content[:400]
            doc_items.append({
                "instruction": f"Explain {topic} in Solid 2",
                "answer": content,
                "mode": "ask",
            })

    # Build story examples for grammar/coherence
    # Load stories from the data directory
    story_items = []
    stories_path = None
    for p in [Path("data/solid2/tinystories.txt"), REPO_ROOT / "data" / "solid2" / "tinystories.txt", Path("/content/data/solid2/tinystories.txt")]:
        if p.exists():
            stories_path = p
            break
    if stories_path:
        stories_text = stories_path.read_text(encoding="utf-8", errors="replace")
        # Limit to 200K chars for manageable training
        if len(stories_text) > 200000:
            stories_text = stories_text[:200000]
        # Split into paragraphs and create training examples
        paragraphs = [p.strip() for p in stories_text.split("\n\n") if p.strip()]
        for para in paragraphs[:500]:  # Limit to 500 paragraphs
            if len(para) < 30 or len(para) > 400:
                continue
            story_items.append({
                "instruction": "Continue the story",
                "answer": para,
                "mode": "ask",
            })
        print(f"loaded {len(story_items)} story examples from {stories_path}", flush=True)
    else:
        print("WARNING: no stories file found, skipping story examples", flush=True)

    all_items = qa_items + doc_items + story_items

    # Build examples
    examples, skipped = build_examples(tokenizer, all_items, config["context"])
    if not examples:
        raise SystemExit("no training examples fit the context")

    print(
        f"data: {len(examples):,} examples, skipped {skipped:,}\n"
        f"config: {config['vocabSize']:,} vocab / {config['context']} ctx / "
        f"{config['dim']} dim / {config['layers']} layers / "
        f"{config['heads']} heads / {config['mlp']} mlp "
        f"({model_bundle.parameter_count(config):,} params)\n"
        f"mask id: {mask_id} · denoising steps: {args.steps} · "
        f"full-mask prob: {args.full_mask_prob}",
        flush=True,
    )

    corpus_hash = "solid2-masked-diffusion"

    model = MaskedDiffusionGPT(
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
    steps_per_epoch = max(1, math.ceil(len(examples) / args.batch))
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
    epochs_without_improvement = 0
    elapsed_base = 0.0

    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        base.restore_rng(checkpoint.get("rng"))
        start_epoch = int(checkpoint.get("epoch", 0))
        start_cursor = int(checkpoint.get("cursor", 0))
        start_perm = checkpoint.get("perm")
        step = int(checkpoint.get("step", 0))
        history = list(checkpoint.get("history", []))
        best_loss = float(checkpoint.get("best_loss", float("inf")))
        best_step = int(checkpoint.get("best_step", 0))
        best_model = checkpoint.get("best_model")
        epochs_without_improvement = int(
            checkpoint.get("epochs_without_improvement", 0)
        )
        elapsed_base = float(checkpoint.get("elapsed_seconds", 0.0))

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
                "tokenizer_path": str(tokenizer_path),
                "tokenizer_model": tokenizer_model,
                "diffusion_steps": args.steps,
                "full_mask_prob": args.full_mask_prob,
                "epoch": epoch,
                "cursor": cursor,
                "perm": perm,
                "step": current_step,
                "model": model_state_cpu(model),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "rng": base.capture_rng(),
                "history": history,
                "best_loss": best_loss,
                "best_step": best_step,
                "best_model": best_model,
                "epochs_without_improvement": epochs_without_improvement,
                "elapsed_seconds": elapsed_seconds(),
            },
        )

    model.train()
    stop_requested = False
    epoch = start_epoch
    current_step = step
    while epoch < args.epochs:
        if start_perm is not None:
            perm = list(start_perm)
        else:
            perm = list(range(len(examples)))
            random.shuffle(perm)
        begin = start_cursor if epoch == start_epoch else 0
        start_perm = None
        start_cursor = 0
        run_loss = 0.0
        run_count = 0
        index = begin
        while index < len(perm):
            if base.STOP_REQUESTED or (
                deadline is not None and time.monotonic() >= deadline
            ):
                stop_requested = True
                break
            if args.max_steps > 0 and current_step >= args.max_steps:
                stop_requested = True
                break
            batch_indices = perm[index : index + args.batch]
            x, y = build_batch(
                examples,
                batch_indices,
                config["context"],
                pad_id,
                mask_id,
                device,
                args.full_mask_prob,
            )
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
            supervised = int((y != -100).sum().item())
            run_loss += loss_value * max(1, supervised)
            run_count += max(1, supervised)
            index += len(batch_indices)
            if current_step % 10 == 0:
                print(
                    f"epoch {epoch + 1}/{args.epochs} step {current_step} "
                    f"loss {loss_value:.4f} masked {supervised}/{x.numel()} "
                    f"lr {scheduler.get_last_lr()[0]:.2e} "
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
            value = eval_full_mask_loss(
                model,
                examples,
                config["context"],
                mask_id,
                pad_id,
                args.batch,
                device,
            )
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
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
        save(epoch + 1, 0, None, current_step)
        epoch += 1
        if (
            args.early_stop_patience > 0
            and epochs_without_improvement >= args.early_stop_patience
        ):
            print(
                f"early stop at epoch {epoch}/{args.epochs}: no val improvement "
                f"for {epochs_without_improvement} epochs "
                f"(best {best_loss:.4f} at step {best_step})",
                flush=True,
            )
            break

    save(epoch, 0, None, current_step)
    if best_model is not None and not args.export_final:
        model.load_state_dict(best_model)
    export_artifacts(
        out_dir=args.out,
        config=config,
        tokenizer_model=tokenizer_model,
        tokenizer=tokenizer,
        model=model,
        mask_id=mask_id,
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
