#!/usr/bin/env python3
"""Verify every prompt mentions the literals its expected output depends on.

A row may carry ``tags.required``: a list of literals (identifiers, numbers,
strings) that must appear in the instruction. If the instruction omits one of
them, the row is under-specified: the model cannot know which value to use, so
exact-stdout evaluation marks a correct model as wrong.

This validates the dataset, not a model. Run it after changing
``scripts/luau_tasks.py`` and before starting a training job:

    python3 scripts/audit_luau_prompts.py --tasks data/luau-tasks/train.jsonl

Accepts a glob so one command covers every split:

    python3 scripts/audit_luau_prompts.py --tasks 'data/luau-tasks/*.jsonl'

Exits non-zero when any row fails.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TASKS = SCRIPT_DIR.parent / "data" / "luau-tasks" / "*.jsonl"


def expand_patterns(patterns: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise SystemExit(f"no task files matched: {pattern}")
        for match in matches:
            path = Path(match)
            if path not in paths:
                paths.append(path)
    return paths


def iter_rows(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def required_literals(row: Dict) -> List[str]:
    tags = row.get("tags") or {}
    if not isinstance(tags, dict):
        return []
    required = tags.get("required") or ()
    if isinstance(required, str):
        return [required]
    return [str(value) for value in required]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--tasks",
        action="append",
        default=None,
        help=(
            "JSONL file or glob (repeatable). "
            "Default: data/luau-tasks/*.jsonl"
        ),
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--show-failures", type=int, default=20)
    args = parser.parse_args()

    paths = expand_patterns(args.tasks or [str(DEFAULT_TASKS)])
    failures: List[Dict[str, object]] = []
    per_family: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"rows": 0, "checked": 0, "failed": 0}
    )
    total = 0
    checked = 0

    for path in paths:
        for row in iter_rows(path):
            if args.limit > 0 and total >= args.limit:
                break
            total += 1
            family = str(row.get("family", "unknown"))
            stats = per_family[family]
            stats["rows"] += 1
            required = required_literals(row)
            if not required:
                continue
            stats["checked"] += 1
            checked += 1
            instruction = str(row.get("instruction", ""))
            missing = [value for value in required if value not in instruction]
            if missing:
                stats["failed"] += 1
                failures.append(
                    {
                        "path": str(path),
                        "id": row.get("id"),
                        "family": family,
                        "style": row.get("prompt_style"),
                        "missing": missing,
                        "instruction": instruction,
                    }
                )

    print(f"audited {total} rows across {len(paths)} file(s)")
    print(f"rows with required literals: {checked}")
    print(f"under-specified rows: {len(failures)}")

    if per_family:
        print("\nfamily                 rows  checked  failed")
        for family in sorted(per_family):
            stats = per_family[family]
            print(
                f"{family:22s} {stats['rows']:5d} {stats['checked']:8d} "
                f"{stats['failed']:7d}"
            )

    if failures:
        reasons = Counter(
            item for failure in failures for item in failure["missing"]
        )
        print("\nmost common missing literals:")
        for value, count in reasons.most_common(10):
            print(f"  {value!r}: {count}")
        shown = failures[: max(0, args.show_failures)]
        if shown:
            print("\nfailures:")
            for failure in shown:
                print(
                    f"\n{failure['path']}: {failure['id']}\n"
                    f"  {failure['family']}/{failure['style']} missing "
                    f"{failure['missing']}\n"
                    f"  instruction: {failure['instruction']}"
                )
        if len(failures) > len(shown):
            print(
                f"\n... {len(failures) - len(shown)} more "
                f"(use --show-failures to see more)"
            )
        raise SystemExit(1)

    print("\nOK: every required literal appears in its instruction")


if __name__ == "__main__":
    main()