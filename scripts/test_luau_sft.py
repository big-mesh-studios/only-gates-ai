#!/usr/bin/env python3
"""Tests for Luau SFT example packing.

Run with:

    python3 -m unittest scripts/test_luau_sft.py

No PyTorch or Luau binary is required.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from bpe_tokenizer import load_tokenizer
from luau_sft import next_token_targets, pack_example


TOKENIZER = (
    SCRIPT_DIR.parent
    / "apps"
    / "only-gates-ai"
    / "public"
    / "models"
    / "tiny-gpt-solid2"
    / "tokenizer.json"
)


class PackExampleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tokenizer = load_tokenizer(TOKENIZER)
        cls.item = {
            "mode": "write",
            "instruction": "Write a for loop that prints the numbers from 1 to 3 in Luau.",
            "answer": "for i = 1, 3 do\n    print(i)\nend",
        }

    def prompt_length(self, item: dict[str, str]) -> int:
        prefix = "Task: " if item["mode"] == "write" else "Question: "
        return (
            2
            + len(self.tokenizer.encode_ordinary(prefix + item["instruction"]))
        )

    def test_labels_are_masked_on_the_prompt(self) -> None:
        tokens, labels = pack_example(self.tokenizer, self.item, 256)
        prompt_len = self.prompt_length(self.item)
        self.assertEqual(labels[:prompt_len], [-100] * prompt_len)
        self.assertNotEqual(labels[prompt_len], -100)

    def test_shifted_targets_start_at_the_first_answer_token(self) -> None:
        tokens, labels = pack_example(self.tokenizer, self.item, 256)
        prompt_len = self.prompt_length(self.item)
        answer_ids = self.tokenizer.encode_ordinary(self.item["answer"])
        pairs = next_token_targets(tokens, labels)
        # Position prompt_len - 1 predicts the first answer token.
        self.assertEqual(pairs[prompt_len - 1][1], answer_ids[0])
        # Nothing before it is supervised.
        self.assertTrue(all(target == -100 for _, target in pairs[: prompt_len - 1]))

    def test_the_end_token_is_supervised(self) -> None:
        tokens, labels = pack_example(self.tokenizer, self.item, 256)
        end_id = self.tokenizer.id_for("<|end|>")
        end_positions = [
            index for index, token_id in enumerate(tokens) if token_id == end_id
        ]
        self.assertTrue(end_positions)
        self.assertEqual(labels[end_positions[-1]], end_id)
        # The target after <|end|> is padding, which must be ignored.
        self.assertEqual(labels[end_positions[-1] + 1], -100)

    def test_padding_has_no_targets(self) -> None:
        tokens, labels = pack_example(self.tokenizer, self.item, 128)
        pad_id = self.tokenizer.id_for("<|pad|>")
        first_pad = tokens.index(pad_id)
        self.assertTrue(all(label == -100 for label in labels[first_pad:]))

    def test_code_that_does_not_fit_is_skipped_not_truncated(self) -> None:
        item = {
            "mode": "write",
            "instruction": "Write a loop.",
            "answer": "\n".join(f"print({index})" for index in range(200)),
        }
        self.assertIsNone(pack_example(self.tokenizer, item, 128))

    def test_question_answers_may_be_truncated(self) -> None:
        item = {
            "mode": "ask",
            "instruction": "How do you write a loop?",
            "answer": "A loop repeats code. " * 80,
        }
        packed = pack_example(self.tokenizer, item, 128)
        self.assertIsNotNone(packed)
        tokens, labels = packed
        self.assertEqual(len(tokens), 128)
        self.assertEqual(tokens[-1], self.tokenizer.id_for("<|end|>"))
        supervised = [label for label in labels if label != -100]
        self.assertTrue(supervised)
        self.assertEqual(supervised[-1], self.tokenizer.id_for("<|end|>"))


if __name__ == "__main__":
    unittest.main()