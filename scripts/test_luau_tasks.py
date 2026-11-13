#!/usr/bin/env python3
"""Unit tests for the Luau task generator.

Run with:

    python3 -m unittest scripts/test_luau_tasks.py

The tests do not need Luau or PyTorch. Execution validation lives in
``scripts/eval_luau_tasks.py`` and ``scripts/check_luau_tasks.py``.
"""

from __future__ import annotations

import json
import os
import random
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import luau_tasks


class LuauTaskGeneratorTest(unittest.TestCase):
    def generate(self, train: int = 80, seed: int = 11) -> dict[str, list[dict]]:
        families = luau_tasks.FAMILIES
        return {
            "train": luau_tasks.generate_split(
                families, train, seed, "train", luau_tasks.TRAIN_STYLES, luau_tasks.COMMON_NAMES
            ),
            "val": luau_tasks.generate_split(
                families, 20, seed + 1, "val", luau_tasks.TRAIN_STYLES, luau_tasks.COMMON_NAMES
            ),
            "test-seen": luau_tasks.generate_split(
                families, 20, seed + 2, "test-seen", ("canonical",), luau_tasks.COMMON_NAMES
            ),
            "test-paraphrase": luau_tasks.generate_split(
                families,
                20,
                seed + 3,
                "test-paraphrase",
                (luau_tasks.HELD_OUT_STYLE,),
                luau_tasks.COMMON_NAMES,
            ),
            "test-unseen": luau_tasks.generate_split(
                families, 20, seed + 4, "test-unseen", ("canonical",), luau_tasks.UNSEEN_NAMES
            ),
        }

    def test_generation_is_deterministic(self) -> None:
        first = self.generate()
        second = self.generate()
        self.assertEqual(
            json.dumps(first, sort_keys=True),
            json.dumps(second, sort_keys=True),
        )

    def test_every_family_is_present(self) -> None:
        splits = self.generate()
        train_families = {row["family"] for row in splits["train"]}
        self.assertEqual(train_families, set(luau_tasks.FAMILIES))

    def test_no_duplicate_instruction_code_pairs(self) -> None:
        splits = self.generate(train=200)
        keys = [(row["instruction"], row["code"]) for row in splits["train"]]
        self.assertEqual(len(keys), len(set(keys)))

    def test_rows_have_required_fields(self) -> None:
        splits = self.generate()
        for split, rows in splits.items():
            for row in rows:
                self.assertEqual(row["split"], split)
                self.assertEqual(row["mode"], "write")
                self.assertTrue(row["id"])
                self.assertTrue(row["family"])
                self.assertTrue(row["instruction"])
                self.assertTrue(row["code"])
                self.assertIn("harness", row)
                self.assertIsInstance(row["harness"], dict)
                self.assertTrue(
                    "expect_stdout" in row["harness"] or "check" in row["harness"],
                    f"{row['id']} has no execution check",
                )

    def test_no_answer_exceeds_context(self) -> None:
        splits = self.generate(train=200)
        for rows in splits.values():
            for row in rows:
                self.assertLessEqual(
                    len(row["code"]),
                    900,
                    f"{row['id']} is unexpectedly long",
                )

    def test_test_styles_are_held_out(self) -> None:
        splits = self.generate()
        train_styles = {row["prompt_style"] for row in splits["train"]}
        self.assertEqual(train_styles, set(luau_tasks.TRAIN_STYLES))
        self.assertEqual(
            {row["prompt_style"] for row in splits["test-paraphrase"]},
            {luau_tasks.HELD_OUT_STYLE},
        )

    def test_unseen_split_uses_other_identifiers(self) -> None:
        splits = self.generate()
        unseen_arrays = set(luau_tasks.UNSEEN_NAMES.arrays)
        for row in splits["test-unseen"]:
            if row["family"] in {"for_sum_all", "for_sum_first_n"}:
                # The prelude contains the identifier; ensure it comes from the
                # held-out pool rather than the common pool.
                prelude = row["harness"].get("prelude", "")
                self.assertTrue(
                    any(f"local {name} =" in prelude for name in unseen_arrays),
                    f"{row['id']} did not use an unseen array name",
                )

    def test_every_style_mentions_required_literals(self) -> None:
        required_families = {
            "function_add",
            "table_literal",
            "table_access",
            "string_concat",
            "string_interp",
            "function_typed",
            "if_elseif_grade",
            "if_else_health",
        }
        seen = set()
        for family, builder in luau_tasks.FAMILY_BUILDERS.items():
            for attempt in range(30):
                rng = random.Random(attempt * 7919)
                result = builder(rng, luau_tasks.COMMON_NAMES)
                required = result.tags.get("required", ())
                if required:
                    seen.add(family)
                for style, text in result.instructions.items():
                    for literal in required:
                        self.assertIn(
                            str(literal),
                            text,
                            f"{family}/{style} does not mention {literal!r}: {text}",
                        )
        self.assertEqual(required_families, seen)

    def test_family_allocation_sums_to_requested_count(self) -> None:
        families = luau_tasks.FAMILIES
        for count in (20, 200, 1200, 12000):
            counts = luau_tasks.allocate_family_counts(families, count)
            self.assertEqual(sum(counts.values()), count)
            self.assertEqual(set(counts), set(families))
            self.assertTrue(all(value >= 1 for value in counts.values()))

    def test_weighted_allocation_favours_weak_families(self) -> None:
        families = luau_tasks.FAMILIES
        first = luau_tasks.allocate_family_counts(
            families, 12000, luau_tasks.FAMILY_WEIGHTS
        )
        second = luau_tasks.allocate_family_counts(
            families, 12000, luau_tasks.FAMILY_WEIGHTS
        )
        self.assertEqual(first, second)
        self.assertEqual(sum(first.values()), 12000)
        self.assertGreater(first["function_add"], first["for_find_max"])
        self.assertGreater(first["nested_loop"], first["while_accumulate"])
        self.assertEqual(first["function_add"], first["table_access"])

    def test_weighted_generation_keeps_total_rows(self) -> None:
        rows = luau_tasks.generate_split(
            luau_tasks.FAMILIES,
            200,
            11,
            "train",
            luau_tasks.TRAIN_STYLES,
            luau_tasks.COMMON_NAMES,
            luau_tasks.FAMILY_WEIGHTS,
        )
        self.assertEqual(len(rows), 200)
        self.assertEqual({row["family"] for row in rows}, set(luau_tasks.FAMILIES))

    def test_generated_code_executes_when_luau_is_available(self) -> None:
        luau_bin = os.environ.get("LUAU_BIN")
        if not luau_bin:
            self.skipTest("set LUAU_BIN to a Luau CLI to execute generated tasks")
        from luau_exec import execute_luau

        splits = self.generate(train=40)
        checked = 0
        for split, rows in splits.items():
            for row in rows:
                result = execute_luau(row["code"], row["harness"], luau_bin)
                checked += 1
                self.assertTrue(
                    result.ok,
                    f"{row['id']} failed: {result.reason}\n"
                    f"stdout={result.stdout!r}\nstderr={result.stderr!r}",
                )
        self.assertGreater(checked, 0)

    def test_write_jsonl_round_trip(self) -> None:
        splits = self.generate()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tasks.jsonl"
            count = luau_tasks.write_jsonl(path, splits["train"])
            self.assertEqual(count, len(splits["train"]))
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line
            ]
            self.assertEqual(len(rows), count)
            self.assertEqual(rows[0]["id"], splits["train"][0]["id"])


if __name__ == "__main__":
    unittest.main()
