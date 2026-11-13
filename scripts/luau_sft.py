#!/usr/bin/env python3
"""Torch-free packing for Luau supervised fine-tuning.

Keeping this separate from ``train_tiny_gpt_luau_colab.py`` means the label
alignment can be unit tested without installing PyTorch.

Example layout, with ``P`` prompt tokens, ``A`` answer tokens, and ``E`` the
end token::

    seq    = [P0 P1 P2 A0 A1 E pad ...]
    labels = [-100 -100 -100 A0 A1 E -100 ...]

The trainer shifts once with ``y[:, 1:]``, so the logits at ``P2`` are trained
against ``A0`` and the logits at ``E`` are trained against the first pad
token, which the loss ignores.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from bpe_tokenizer import BpeTokenizer


def pack_example(
    tokenizer: BpeTokenizer,
    item: Mapping[str, str],
    context: int,
) -> Optional[Tuple[List[int], List[int]]]:
    user_id = tokenizer.id_for("<|user|>")
    assistant_id = tokenizer.id_for("<|assistant|>")
    end_id = tokenizer.id_for("<|end|>")
    pad_id = tokenizer.id_for("<|pad|>")
    if user_id is None or assistant_id is None or end_id is None:
        raise SystemExit("tokenizer is missing <|user|>, <|assistant|>, or <|end|>")
    prefix = "Task: " if item.get("mode") == "write" else "Question: "
    prompt_ids = (
        [user_id]
        + tokenizer.encode_ordinary(prefix + item["instruction"])
        + [assistant_id]
    )
    answer_ids = tokenizer.encode_ordinary(item["answer"]) + [end_id]
    if len(prompt_ids) >= context:
        return None
    budget = context - len(prompt_ids)
    if len(answer_ids) > budget:
        if item.get("mode") == "write":
            # A truncated code answer teaches broken syntax. Skip it instead.
            return None
        if budget < 8:
            return None
        # Keep the stop token so the model still learns to terminate.
        answer_ids = answer_ids[: budget - 1] + [end_id]
    seq = prompt_ids + answer_ids
    # Labels align with `seq` (masked on the prompt). The training loop shifts
    # them once with ``y[:, 1:]``, so position `prompt_len - 1` learns to emit
    # the first answer token and the last answer token learns to emit <|end|>.
    labels = [-100] * len(prompt_ids) + list(answer_ids)
    pad = context - len(seq)
    if pad > 0:
        seq = seq + [pad_id if pad_id is not None else 0] * pad
        labels = labels + [-100] * pad
    return seq[:context], labels[:context]


def build_examples(
    tokenizer: BpeTokenizer,
    items: Sequence[Mapping[str, str]],
    context: int,
) -> Tuple[List[Tuple[List[int], List[int]]], int]:
    examples: List[Tuple[List[int], List[int]]] = []
    skipped = 0
    for item in items:
        packed = pack_example(tokenizer, item, context)
        if packed is None:
            skipped += 1
        else:
            examples.append(packed)
    return examples, skipped


def next_token_targets(
    tokens: Sequence[int], labels: Sequence[int]
) -> List[Tuple[int, int]]:
    """Return ``(input_id, target_id)`` pairs for a packed example.

    This mirrors the trainer's ``x[:, :-1]`` / ``y[:, 1:]`` shift and is used
    by the tests to assert that no prompt token leaks into the loss.
    """
    pairs: List[Tuple[int, int]] = []
    for index in range(len(tokens) - 1):
        pairs.append((tokens[index], labels[index + 1]))
    return pairs


if __name__ == "__main__":  # pragma: no cover - debug helper
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tokenizer", type=Path)
    parser.add_argument("task")
    args = parser.parse_args()
    from bpe_tokenizer import load_tokenizer

    tokenizer = load_tokenizer(args.tokenizer)
    row: Dict = json.loads(args.task)
    packed = pack_example(
        tokenizer,
        {"mode": row.get("mode", "write"), "instruction": row["instruction"], "answer": row["code"]},
        256,
    )
    if packed is None:
        raise SystemExit("example does not fit the context")
    tokens, labels = packed
    supervised = [(i, t) for i, (i, t) in enumerate(next_token_targets(tokens, labels)) if t != -100]
    print(f"tokens={len(tokens)} supervised={len(supervised)}")
    for position, target in supervised[:5]:
        print(position, "->", repr(tokenizer.decode([target])))