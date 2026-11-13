#!/usr/bin/env python3
"""Execute every generated Luau task and verify its harness.

This validates the dataset, not a model. Run it after changing
``scripts/luau_tasks.py`` and before starting a training job.

    python3 scripts/check_luau_tasks.py \
      --tasks data/luau-tasks/test-seen.jsonl \
      --luau-bin /usr/local/bin/luau

The Luau CLI is required. On Colab, install a pinned release or build
``Luau.Repl.CLI`` from source.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from luau_exec import execute_luau


DEFAULT_TASKS = (
    SCRIPT_DIR.parent / "data" / "luau-tasks" / "test-seen.jsonl",
)


def iter_rows(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        type=Path,
        action="append",
        default=None,
        help="JSONL file (repeatable). Default: data/luau-tasks/test-seen.jsonl",
    )
    parser.add_argument("--luau-bin", default="luau")
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--show-failures", type=int, default=10)
    args = parser.parse_args()

    paths: List[Path] = args.tasks or list(DEFAULT_TASKS)
    rows: List[Dict] = []
    for path in paths:
        if not path.exists():
            raise SystemExit(f"task file not found: {path}")
        rows.extend(iter_rows(path))
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("no tasks to check")

    failures: List[Dict[str, object]] = []
    per_family: Dict[str, Dict[str, int]] = defaultdict(lambda: {"rows": 0, "failed": 0})
    reasons: Counter[str] = Counter()
    for index, row in enumerate(rows, 1):
        family = str(row.get("family", "unknown"))
        per_family[family]["rows"] += 1
        result = execute_luau(
            str(row.get("code", "")),
            row.get("harness", {}),
            args.luau_bin,
            timeout=args.timeout,
        )
        if not result.ok:
            per_family[family]["failed"] += 1
            reasons[result.reason or "unknown"] += 1
            failures.append(
                {
                    "id": row.get("id"),
                    "family": family,
                    "instruction": row.get("instruction"),
                    "code": row.get("code"),
                    "reason": result.reason,
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "expected_stdout": result.expected_stdout,
                }
            )
        if index % 100 == 0:
            print(f"checked {index}/{len(rows)}", flush=True)

    print()
    print(f"{'family':24s} {'rows':>6s} {'failed':>7s}")
    print("-" * 40)
    for family in sorted(per_family):
        values = per_family[family]
        print(f"{family:24s} {values['rows']:>6d} {values['failed']:>7d}")
    print("-" * 40)
    failed = len(failures)
    print(f"checked {len(rows):,}; failed {failed:,} ({failed / len(rows):.2%})")
    if reasons:
        print("reasons: " + ", ".join(f"{name}={count}" for name, count in reasons.most_common()))

    if failures and args.show_failures > 0:
        print()
        print(f"First {min(args.show_failures, len(failures))} failure(s):")
        for failure in failures[: args.show_failures]:
            print()
            print(f"{failure['id']} ({failure['family']}) [{failure['reason']}]")
            print(f"  instruction: {failure['instruction']}")
            print("  code:")
            for line in str(failure["code"]).splitlines():
                print(f"    {line}")
            if failure["stderr"]:
                print("  stderr:")
                for line in str(failure["stderr"]).splitlines():
                    print(f"    {line}")
            if failure["expected_stdout"] is not None:
                print(f"  expected stdout: {failure['expected_stdout']!r}")
                print(f"  actual stdout:   {failure['stdout']!r}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(
                {
                    "tasks": [str(path) for path in paths],
                    "rows": len(rows),
                    "failed": failed,
                    "reasons": dict(reasons),
                    "families": dict(per_family),
                    "failures": failures,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"wrote {args.json_out}")

    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
