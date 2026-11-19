#!/usr/bin/env python3
"""Train a Luau masked-diffusion denoiser on Colab and export app artifacts.

The model is a bidirectional Tiny GPT trained with an absorbing ``<|mask|>``
token. It shares the packed-weight layout of the causal Tiny GPT, so the browser
loads it with ``tiny_gpt_create`` and runs one integer-only denoising pass per
animation frame.

Unlike the causal Luau model, every answer position is predicted in parallel.
Training samples a mask ratio ``t ~ U(0, 1)`` (with a small full-mask
probability), masks that fraction of the answer region, and computes cross
entropy only on the masked positions. The answer region always fills the rest
of the context (``context - prompt_length``), which matches the browser
decoder: it starts with a full region of masks and fills the next left-to-right
block on each step.

Examples
--------

Train from scratch on a T4, with resumable checkpoints::

    python scripts/train_masked_diffusion_luau_colab.py \
      --tasks data/luau-tasks \
      --out /content/drive/MyDrive/masked-diffusion-luau \
      --epochs 40 --batch 16 --lr 3e-4 --max-minutes 240

Resume after a disconnect::

    python scripts/train_masked_diffusion_luau_colab.py \
      --tasks data/luau-tasks \
      --out /content/drive/MyDrive/masked-diffusion-luau \
      --resume --epochs 40 --batch 16 --lr 3e-4 --max-minutes 240

CPU smoke test::

    python scripts/train_masked_diffusion_luau_colab.py \
      --tasks data/luau-tasks --out /tmp/masked-smoke --smoke
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

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import torch
import torch.nn as nn
import torch.nn.functional as F

import model_bundle
import train_tiny_gpt_solid2_colab as base
from bpe_tokenizer import BpeTokenizer
from train_tiny_gpt_luau_colab import (
    load_task_rows,
    sha256_text,
    split_faq_pairs,
    task_item,
)


CHECKPOINT_FORMAT = "only-gates-masked-diffusion-luau"
CHECKPOINT_VERSION = 1
CONFIG_KEYS = ("vocabSize", "context", "dim", "layers", "heads", "mlp")
MASK_TOKEN = "<|mask|>"
# Prompts shown in the app's Masked diffusion tab. Keep the phrasing close to
# the generated task templates so the exported samples reflect the live demo.
SAMPLE_PROMPTS: Sequence[Tuple[str, str]] = (
    (
        "write",
        "Write a for loop in Luau that runs from 1 to 20 and prints every number.",
    ),
    (
        "write",
        "Write a local function named `add` in Luau that takes `a` and `b` and "
        "returns their sum. Then print `add(2, 3)`.",
    ),
    (
        "write",
        "Write a for loop that counts how many elements of `scores` are greater "
        "than 80, then prints the count in Luau.",
    ),
    (
        "write",
        "Use a while loop in Luau to add the numbers 1 through 10 and print the "
        "total.",
    ),
    ("ask", "How do you do a for loop in Luau?"),
    ("ask", "How do you create a table in Luau?"),
)
DEFAULT_TOKENIZER = (
    REPO_ROOT
    / "apps"
    / "only-gates-ai"
    / "public"
    / "models"
    / "tiny-gpt-luau"
    / "tokenizer.json"
)
DEFAULT_FAQ = REPO_ROOT / "apps" / "only-gates-ai" / "src" / "assets" / "luau-faq.md"
DEFAULT_TOKENS = ("<|pad|>", "<|user|>", "<|assistant|>", "<|end|>")


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


def load_masked_tokenizer(path: Path) -> Tuple[BpeTokenizer, Dict]:
    """Load the Luau BPE model and append an absorbing mask token.

    Appending keeps every existing token id stable, which matters because the
    deployed Luau tab and the masked tab share prompt encoding.
    """
    if not path.exists():
        raise SystemExit(
            f"missing tokenizer {path}; run scripts/train_luau_tokenizer.py first "
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
    for special in DEFAULT_TOKENS:
        if tokenizer.id_for(special) is None:
            raise SystemExit(f"tokenizer {path} is missing {special}")
    mask_id = tokenizer.id_for(MASK_TOKEN)
    if mask_id is None:
        raise SystemExit(f"tokenizer {path} is missing {MASK_TOKEN}")
    return tokenizer, model


# ---------------------------------------------------------------------------
# Data packing
# ---------------------------------------------------------------------------


def pack_diffusion_example(
    tokenizer: BpeTokenizer,
    item: Mapping[str, str],
    context: int,
    min_answer: int = 16,
) -> Optional[Tuple[List[int], List[int]]]:
    """Return ``(prompt_ids, answer_region)`` with region length context - prompt.

    The prompt matches the browser's ``encodeChatPrompt`` exactly. The answer
    region is ``answer_ids + <|end|>`` followed by ``<|pad|>`` filler. Training
    targets never include prompt positions.
    """
    pad_id = tokenizer.id_for("<|pad|>")
    end_id = tokenizer.id_for("<|end|>")
    if pad_id is None or end_id is None:
        raise SystemExit("tokenizer must define <|pad|> and <|end|>")

    prefix = "Task: " if item.get("mode") == "write" else "Question: "
    prompt_ids = tokenizer.encode(
        f"<|user|>{prefix}{item['instruction']}<|assistant|>"
    )
    answer_ids = tokenizer.encode_ordinary(item["answer"]) + [end_id]
    available = context - len(prompt_ids)
    if available < min_answer:
        return None
    if len(answer_ids) > available:
        if item.get("mode") == "write":
            # Truncated code teaches broken syntax. Skip it, as the causal
            # Luau trainer does.
            return None
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


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class BidirectionalBlock(nn.Module):
    """Pre-norm transformer block with no attention mask.

    Attribute names match ``base.Block`` so ``base.pack_weights`` and the C
    ``tiny_gpt_create`` layout stay byte-for-byte compatible.
    """

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


# ---------------------------------------------------------------------------
# Masking and batching
# ---------------------------------------------------------------------------


def mask_region(
    region: Sequence[int],
    mask_id: int,
    pad_id: int,
    full_mask_probability: float,
) -> Tuple[List[int], List[int]]:
    """Return ``(input_region, targets)`` for one answer region.

    Every position is a candidate input mask, but pad positions after
    ``<|end|>`` are never targets: the answer is short and the leftover region
    would otherwise dominate the loss. At least one real position is always
    masked.
    """
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


# ---------------------------------------------------------------------------
# PyTorch mirror of the C denoiser
# ---------------------------------------------------------------------------


def format_prompt(mode: str, prompt: str) -> str:
    prefix = "Task: " if mode == "write" else "Question: "
    return f"<|user|>{prefix}{prompt}<|assistant|>"


def decode_answer(tokenizer: BpeTokenizer, ids: Sequence[int]) -> str:
    end_id = tokenizer.id_for("<|end|>")
    pad_id = tokenizer.id_for("<|pad|>")
    mask_id = tokenizer.id_for(MASK_TOKEN)
    specials = {
        token_id
        for text, token_id in tokenizer.token_to_id.items()
        if text.startswith("<|")
    }
    kept: List[int] = []
    for token_id in ids:
        if token_id == end_id:
            break
        if token_id in (pad_id, mask_id):
            continue
        if token_id in specials:
            continue
        if 0 <= token_id < len(tokenizer.vocab):
            kept.append(token_id)
    return tokenizer.decode(kept).strip()


@torch.no_grad()
def denoise(
    model: torch.nn.Module,
    tokenizer: BpeTokenizer,
    prompt: str,
    context: int,
    mask_id: int,
    steps: int,
    temperature: float,
    device: torch.device,
) -> str:
    model.eval()
    prompt_ids = tokenizer.encode(prompt)
    if len(prompt_ids) >= context:
        return ""
    tokens = list(prompt_ids) + [mask_id] * (context - len(prompt_ids))
    prompt_length = len(prompt_ids)

    for step in range(max(1, steps)):
        tensor = torch.tensor([tokens], dtype=torch.long, device=device)
        logits, _ = model(tensor)
        masked = [i for i in range(prompt_length, context) if tokens[i] == mask_id]
        if not masked:
            break
        steps_left = max(1, max(1, steps) - step)
        to_unmask = max(1, math.ceil(len(masked) / steps_left))
        # Left to right, matching the C sampler: a later `print(i)` should see
        # the `i` chosen by the earlier `for`, not an independent guess.
        for position in masked[:to_unmask]:
            probs = torch.softmax(
                logits[0, position] / max(temperature, 1e-4), dim=-1
            )
            probs[0] = 0.0
            probs[mask_id] = 0.0
            values, ids = torch.topk(probs, min(32, probs.numel()))
            choice = int(torch.multinomial(values, 1).item())
            tokens[position] = int(ids[choice])

    return decode_answer(tokenizer, tokens[prompt_length:])


def generate_samples(
    model: torch.nn.Module,
    tokenizer: BpeTokenizer,
    context: int,
    mask_id: int,
    steps: int,
    device: torch.device,
    limit: int,
) -> List[Dict[str, str]]:
    prompts = list(SAMPLE_PROMPTS)
    samples: List[Dict[str, str]] = []
    for mode, prompt in prompts[: max(0, limit)]:
        text = denoise(
            model,
            tokenizer,
            format_prompt(mode, prompt),
            context,
            mask_id,
            steps,
            temperature=0.1 if mode == "ask" else 0.01,
            device=device,
        )
        samples.append({"prompt": prompt, "text": text})
        print(f"sample> [{mode}] {prompt} => {text[:160]!r}", flush=True)
    model.train()
    return samples


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


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

    samples = generate_samples(
        model,
        tokenizer,
        int(config["context"]),
        mask_id,
        args.steps,
        torch.device(args.device),
        args.sample_count,
    )
    prompts = [prompt for _, prompt in SAMPLE_PROMPTS]
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
        "maskRatio": "uniform+t-full",
        "fullMaskProbability": args.full_mask_prob,
        "denoisingSteps": args.steps,
        "sourceUrls": [
            "data/luau-tasks/train.jsonl",
            "apps/only-gates-ai/src/assets/luau-faq.md",
        ],
    }
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
            # 0 means "fill the rest of the context", matching training.
            "answerBudget": 0,
        },
        "meta": {
            "name": args.name,
            "source": (
                "Generated Luau beginner tasks from scripts/luau_tasks.py plus "
                "the repository's Luau quick-answer FAQ. The denoiser is a "
                "bidirectional Tiny GPT trained with an absorbing <|mask|>."
            ),
            "license": (
                "Generated task data and model weights are produced by this "
                "repository. Luau documentation and the FAQ retain their own "
                "MIT licenses."
            ),
            "training": training,
        },
        "samples": samples,
        "prompts": prompts,
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
        "# Masked-diffusion Luau model\n\n"
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
        "Generated by scripts/train_masked_diffusion_luau_colab.py.\n"
    )
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    print(f"\nwrote {out_dir / 'checkpoint.json'}", flush=True)
    print(f"wrote {out_dir / 'tokenizer.json'}", flush=True)
    print(f"wrote {out_dir / 'weights.bin'} ({len(weights):,} bytes)", flush=True)
    print(
        "Copy these three files into "
        "apps/only-gates-ai/public/models/masked-diffusion-luau/ to publish "
        "the masked-diffusion tab.",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--tasks", type=Path, default=REPO_ROOT / "data" / "luau-tasks")
    parser.add_argument("--out", type=Path, default=Path("masked-diffusion-luau-out"))
    parser.add_argument("--tokenizer", type=Path, default=None)
    parser.add_argument("--faq", type=Path, default=DEFAULT_FAQ)
    parser.add_argument("--faq-repeat", type=int, default=20)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-fraction", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--context", "--ctx", dest="context", type=int, default=128)
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
    parser.add_argument("--name", default="masked-diffusion-luau")
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
        args.faq_repeat, args.steps, args.sample_count = 1, 4, 2
        args.name = "masked-diffusion-luau-smoke"
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

    train_rows, val_rows, task_hash = load_task_rows(args.tasks)
    faq_train, faq_val = split_faq_pairs(args.faq)
    train_items: List[Dict[str, str]] = [task_item(row) for row in train_rows]
    val_items: List[Dict[str, str]] = [task_item(row) for row in val_rows]
    for pair in faq_train:
        train_items.extend([dict(pair) for _ in range(max(1, args.faq_repeat))])
    val_items.extend(faq_val)
    if args.smoke:
        train_items = [task_item(row) for row in train_rows[:48]]
        val_items = [task_item(row) for row in val_rows[:12]]

    corpus_hash = sha256_text(
        json.dumps(
            {
                "tasks": task_hash,
                "faq": sha256_text(
                    args.faq.read_text(encoding="utf-8") if args.faq.exists() else ""
                ),
                "faq_repeat": args.faq_repeat,
                "seed": args.seed,
                "mask_token": MASK_TOKEN,
                "full_mask_prob": args.full_mask_prob,
            },
            sort_keys=True,
        )
    )

    if checkpoint is not None:
        for key, value in (
            ("config", config),
            ("seed", args.seed),
            ("corpus_hash", corpus_hash),
            ("diffusion_steps", args.steps),
        ):
            old = checkpoint.get(key)
            if key == "config":
                old = {name: int(old[name]) for name in CONFIG_KEYS} if old else old
            if old != value:
                raise SystemExit(
                    f"checkpoint {key} differs; use --fresh to start a new run "
                    f"(checkpoint={old!r}, current={value!r})"
                )

    train_examples, skipped_train = build_examples(
        tokenizer, train_items, config["context"]
    )
    val_examples, skipped_val = build_examples(tokenizer, val_items, config["context"])
    if not train_examples:
        raise SystemExit(
            "no training examples fit the context; increase --context or shorten prompts"
        )
    if not val_examples:
        raise SystemExit("no validation examples fit the context")

    print(
        f"data: {len(train_examples):,} train, {len(val_examples):,} val, "
        f"skipped {skipped_train:,}/{skipped_val:,}\n"
        f"config: {config['vocabSize']:,} vocab / {config['context']} ctx / "
        f"{config['dim']} dim / {config['layers']} layers / "
        f"{config['heads']} heads / {config['mlp']} mlp "
        f"({model_bundle.parameter_count(config):,} params)\n"
        f"mask id: {mask_id} · denoising steps: {args.steps} · "
        f"full-mask prob: {args.full_mask_prob}",
        flush=True,
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
        print(
            f"resuming at epoch {start_epoch + 1}/{args.epochs}, step {step}, "
            f"cursor {start_cursor}",
            flush=True,
        )

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
            perm = list(range(len(train_examples)))
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
                train_examples,
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
                val_examples,
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
