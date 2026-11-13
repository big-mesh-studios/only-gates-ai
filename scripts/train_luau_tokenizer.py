#!/usr/bin/env python3
"""Train a Luau-focused byte-level BPE tokenizer.

The deployed 4,096-token tokenizer was trained on Solid 2 docs, where `local`
costs three tokens and `print` costs two. This script trains the same kind of
tokenizer on the generated task corpus plus the Luau FAQ, so keywords and
common identifiers collapse into single tokens.

Usage::

    python3 scripts/train_luau_tokenizer.py \
      --out data/luau-tokenizer/tokenizer.json

    python3 scripts/audit_luau_tokenizer.py \
      --tokenizer data/luau-tokenizer/tokenizer.json \
      --tasks data/luau-tasks/train.jsonl

The output keeps the five special tokens used by the app and the trainer, so
the model config (``vocabSize=4096``) does not change.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from bpe_tokenizer import BYTE_DECODER, BpeTokenizer, train_bpe

DEFAULT_TASKS = REPO_ROOT / "data" / "luau-tasks"
DEFAULT_FAQ = REPO_ROOT / "apps" / "only-gates-ai" / "src" / "assets" / "luau-faq.md"
SPECIAL_TOKENS = ("<|pad|>", "<|user|>", "<|assistant|>", "<|end|>", "<|facts|>")
SAMPLES = (
    "local sum = 0\nfor i = 1, #values do\n    sum += values[i]\nend\nprint(sum)",
    "local function add(a, b)\n    return a + b\nend\n\nprint(add(2, 3))",
    "if score >= 90 then\n    print(\"A\")\nelseif score >= 80 then\n    print(\"B\")\nelse\n    print(\"C\")\nend",
    "for index, value in items do\n    print(index, value)\nend",
)


def iter_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_corpus(tasks_dir: Path, faq: Path, code_repeat: int, faq_repeat: int) -> str:
    parts: List[str] = []
    for name in ("train.jsonl", "val.jsonl"):
        path = tasks_dir / name
        if not path.exists():
            raise SystemExit(f"missing task file: {path}")
        for row in iter_jsonl(path):
            code = str(row.get("code", ""))
            instruction = str(row.get("instruction", ""))
            # Repeat code so merges favor Luau syntax rather than prose.
            parts.extend([code] * max(1, code_repeat))
            parts.append(instruction)
    if faq.exists():
        text = faq.read_text(encoding="utf-8")
        parts.extend([text] * max(1, faq_repeat))
    if not parts:
        raise SystemExit("corpus is empty")
    return "\n\n".join(parts)


def round_trip(tokenizer: BpeTokenizer, text: str) -> bool:
    try:
        encoded = tokenizer.encode_ordinary(text)
    except ValueError:
        return False
    encoded_text = "".join(
        tokenizer.vocab[token_id]
        for token_id in encoded
        if 0 <= token_id < len(tokenizer.vocab)
    )
    raw = bytearray()
    for char in encoded_text:
        byte = BYTE_DECODER.get(char)
        if byte is not None:
            raw.append(byte)
    return raw.decode("utf-8", errors="replace") == text


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--faq", type=Path, default=DEFAULT_FAQ)
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "data" / "luau-tokenizer" / "tokenizer.json"
    )
    parser.add_argument("--vocab", type=int, default=4096)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--code-repeat", type=int, default=4)
    parser.add_argument("--faq-repeat", type=int, default=4)
    args = parser.parse_args()

    corpus = build_corpus(args.tasks, args.faq, args.code_repeat, args.faq_repeat)
    print(f"corpus: {len(corpus):,} characters", flush=True)
    model = train_bpe(
        corpus,
        vocab_size=args.vocab,
        min_frequency=args.min_frequency,
        special_tokens=SPECIAL_TOKENS,
    )
    tokenizer = BpeTokenizer(model)
    for sample in SAMPLES:
        if not round_trip(tokenizer, sample):
            raise SystemExit(f"tokenizer round-trip failed for {sample[:40]!r}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(model, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"wrote {args.out} ({args.out.stat().st_size:,} bytes; "
        f"vocab {len(model['vocab']):,}, merges {len(model['merges']):,})"
    )
    probes = ("local", "print", "for", "elseif", "continue", "function", "values", "scores")
    for probe in probes:
        ids = tokenizer.encode_ordinary(probe)
        pieces = " ".join(tokenizer.vocab[token_id] for token_id in ids)
        print(f"  {probe:10s} {len(ids):>2d} token(s): {pieces}")


if __name__ == "__main__":
    main()