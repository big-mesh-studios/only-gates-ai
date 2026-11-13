#!/usr/bin/env python3
"""Audit how the deployed BPE tokenizer handles Luau task data.

Answers the question "do we need a Luau-specific tokenizer?" with numbers:

- average instruction and code token counts
- how many rows would exceed the model context
- which common identifiers fragment into many tokens

No PyTorch required.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from bpe_tokenizer import BpeTokenizer, load_tokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_TOKENIZER = (
    REPO_ROOT
    / "apps"
    / "only-gates-ai"
    / "public"
    / "models"
    / "tiny-gpt-solid2"
    / "tokenizer.json"
)

PROBE_IDENTIFIERS = (
    "local",
    "print",
    "for",
    "while",
    "repeat",
    "until",
    "function",
    "return",
    "then",
    "else",
    "elseif",
    "end",
    "table",
    "values",
    "scores",
    "numbers",
    "items",
    "data",
    "points",
    "index",
    "value",
    "sum",
    "total",
    "count",
    "product",
    "string",
    "number",
    "boolean",
    "continue",
    "break",
)


def iter_rows(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def percentile(values: Sequence[int], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return float(ordered[index])


def identify_fragments(tokenizer: BpeTokenizer) -> List[Dict[str, object]]:
    results = []
    for identifier in PROBE_IDENTIFIERS:
        ids = tokenizer.encode_ordinary(identifier)
        results.append(
            {
                "identifier": identifier,
                "tokens": len(ids),
                "pieces": [tokenizer.vocab[token_id] for token_id in ids],
                "chars_per_token": len(identifier) / max(1, len(ids)),
            }
        )
    results.sort(key=lambda item: (item["chars_per_token"], item["identifier"]))
    return results


def audit_row(row: Dict, tokenizer: BpeTokenizer) -> Dict[str, object]:
    instruction = str(row.get("instruction", ""))
    code = str(row.get("code", ""))
    prompt_ids = tokenizer.encode_ordinary("Task: " + instruction)
    code_ids = tokenizer.encode_ordinary(code)
    answer_ids = code_ids + [tokenizer.id_for("<|end|>") or 0]
    return {
        "family": row.get("family", "unknown"),
        "prompt_tokens": len(prompt_ids),
        "code_tokens": len(code_ids),
        "total_tokens_with_specials": len(prompt_ids) + len(answer_ids) + 2,
        "code_chars": len(code),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=DEFAULT_TOKENIZER,
    )
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.tokenizer)
    rows = list(iter_rows(args.tasks))
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit(f"no rows in {args.tasks}")

    per_family: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    over_context = 0
    for row in rows:
        result = audit_row(row, tokenizer)
        per_family[str(result["family"])].append(result)
        if int(result["total_tokens_with_specials"]) > args.context:
            over_context += 1

    print(f"tokenizer: {args.tokenizer}")
    print(f"vocab:     {len(tokenizer.vocab):,}")
    print(f"tasks:     {len(rows):,} from {args.tasks}")
    print(f"context:   {args.context}")
    print(f"over context: {over_context:,} ({over_context / len(rows):.2%})")
    print()
    print(
        f"{'family':24s} {'rows':>6s} {'code avg':>9s} {'code p95':>9s} "
        f"{'prompt avg':>11s} {'total p95':>10s}"
    )
    print("-" * 78)
    for family in sorted(per_family):
        values = per_family[family]
        code_tokens = [int(item["code_tokens"]) for item in values]
        prompt_tokens = [int(item["prompt_tokens"]) for item in values]
        total_tokens = [int(item["total_tokens_with_specials"]) for item in values]
        print(
            f"{family:24s} {len(values):>6,} "
            f"{statistics.mean(code_tokens):>9.1f} "
            f"{percentile(code_tokens, 0.95):>9.0f} "
            f"{statistics.mean(prompt_tokens):>11.1f} "
            f"{percentile(total_tokens, 0.95):>10.0f}"
        )

    all_code = [int(item["code_tokens"]) for values in per_family.values() for item in values]
    all_total = [
        int(item["total_tokens_with_specials"])
        for values in per_family.values()
        for item in values
    ]
    print("-" * 78)
    print(
        f"{'all':24s} {len(rows):>6,} "
        f"{statistics.mean(all_code):>9.1f} "
        f"{percentile(all_code, 0.95):>9.0f} "
        f"{'':>11s} "
        f"{percentile(all_total, 0.95):>10.0f}"
    )

    print()
    print("Most fragmented probe identifiers:")
    print(f"{'identifier':20s} {'tokens':>6s} {'chars/token':>11s}  pieces")
    for item in identify_fragments(tokenizer)[:15]:
        pieces = " ".join(str(piece).replace("Ġ", "·") for piece in item["pieces"])
        print(
            f"{item['identifier']:20s} {item['tokens']:>6d} "
            f"{item['chars_per_token']:>11.2f}  {pieces}"
        )

    if args.json_out:
        payload = {
            "tokenizer": str(args.tokenizer),
            "tasks": str(args.tasks),
            "context": args.context,
            "rows": len(rows),
            "over_context": over_context,
            "families": {
                family: {
                    "rows": len(values),
                    "code_avg": statistics.mean(
                        int(item["code_tokens"]) for item in values
                    ),
                    "code_p95": percentile(
                        [int(item["code_tokens"]) for item in values], 0.95
                    ),
                    "prompt_avg": statistics.mean(
                        int(item["prompt_tokens"]) for item in values
                    ),
                    "total_p95": percentile(
                        [int(item["total_tokens_with_specials"]) for item in values],
                        0.95,
                    ),
                }
                for family, values in sorted(per_family.items())
            },
            "probe_identifiers": identify_fragments(tokenizer),
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
