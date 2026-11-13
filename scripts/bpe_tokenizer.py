#!/usr/bin/env python3
"""Byte-level BPE tokenizer shared by the Luau task scripts.

This mirrors ``scripts/train_tiny_gpt_solid2_colab.py`` exactly but has no
PyTorch dependency, so data generation, tokenizer audits, and evaluation
helpers can run on a laptop, Termux, or Colab without a GPU stack.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


def bytes_to_unicode() -> Dict[int, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


BYTE_ENCODER = bytes_to_unicode()
BYTE_DECODER = {v: k for k, v in BYTE_ENCODER.items()}

try:
    import regex as _regex

    PRE_TOKENIZE_RE = _regex.compile(
        r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
except Exception:  # pragma: no cover - stdlib fallback
    PRE_TOKENIZE_RE = re.compile(
        r"""'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+"""
    )

PAIR_STRIDE = 1 << 16


def get_pairs(symbols: Sequence[str]) -> Dict[Tuple[str, str], int]:
    pairs: Dict[Tuple[str, str], int] = {}
    for index in range(len(symbols) - 1):
        pair = (symbols[index], symbols[index + 1])
        pairs[pair] = pairs.get(pair, 0) + 1
    return pairs


def train_bpe(
    corpus: str,
    vocab_size: int = 4096,
    min_frequency: int = 2,
    special_tokens: Sequence[str] = (),
    log: bool = True,
) -> Dict:
    """Train a byte-level BPE model, matching the browser/Colab tokenizer."""
    vocab: List[str] = list(special_tokens)
    for byte in range(256):
        vocab.append(BYTE_ENCODER[byte])
    token_to_id: Dict[str, int] = {token: index for index, token in enumerate(vocab)}

    word_freq: Dict[Tuple[str, ...], int] = {}
    for piece in pre_tokenize(corpus):
        symbols = text_to_symbols(piece)
        word_freq[symbols] = word_freq.get(symbols, 0) + 1
    words = list(word_freq.keys())
    freqs = [word_freq[word] for word in words]
    if log:
        print(
            f"tokenizer: {len(words):,} unique pre-tokens, base {len(vocab):,}",
            flush=True,
        )

    merges: List[List[str]] = []
    while len(vocab) < vocab_size:
        pair_counts: Dict[Tuple[str, str], int] = {}
        for symbols, frequency in zip(words, freqs):
            if len(symbols) < 2 or frequency <= 0:
                continue
            for pair, count in get_pairs(symbols).items():
                pair_counts[pair] = pair_counts.get(pair, 0) + count * frequency
        if not pair_counts:
            break
        best = max(pair_counts.items(), key=lambda item: (item[1], item[0][0], item[0][1]))
        if best[1] < min_frequency:
            break
        left, right = best[0]
        merged = left + right
        merges.append([left, right])
        if merged not in token_to_id:
            token_to_id[merged] = len(vocab)
            vocab.append(merged)
        rebuilt: List[Tuple[str, ...]] = []
        for symbols in words:
            if len(symbols) < 2:
                rebuilt.append(symbols)
                continue
            output: List[str] = []
            index = 0
            while index < len(symbols):
                if (
                    index < len(symbols) - 1
                    and symbols[index] == left
                    and symbols[index + 1] == right
                ):
                    output.append(merged)
                    index += 2
                else:
                    output.append(symbols[index])
                    index += 1
            rebuilt.append(tuple(output))
        words = rebuilt
        if log and (len(vocab) % 512 == 0 or len(vocab) == vocab_size):
            print(f"  vocab {len(vocab):,} merges {len(merges):,}", flush=True)

    return {
        "version": 1,
        "specialTokens": list(special_tokens),
        "vocab": vocab,
        "merges": merges,
    }


def pre_tokenize(text: str) -> List[str]:
    return PRE_TOKENIZE_RE.findall(text)


def text_to_symbols(text: str) -> Tuple[int, ...]:
    return tuple(BYTE_ENCODER[b] for b in text.encode("utf-8"))


class BpeTokenizer:
    def __init__(self, model: Dict):
        self.vocab = model["vocab"]
        self.merges = [tuple(m) for m in model["merges"]]
        self.special_tokens = model.get("specialTokens", [])
        self.token_to_id: Dict[str, int] = {}
        for i, token in enumerate(self.vocab):
            if token not in self.token_to_id:
                self.token_to_id[token] = i
        self.merge_ranks = {"\0".join(m): i for i, m in enumerate(self.merges)}
        self.cache: Dict[str, List[int]] = {}

    def _bpe(self, token: str) -> List[str]:
        symbols = list(text_to_symbols(token))
        while len(symbols) > 1:
            best = None
            best_rank = 10**18
            for i in range(len(symbols) - 1):
                rank = self.merge_ranks.get(symbols[i] + "\0" + symbols[i + 1])
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best = i
            if best is None:
                break
            merged = symbols[best] + symbols[best + 1]
            symbols[best : best + 2] = [merged]
        return symbols

    def encode_ordinary(self, text: str) -> List[int]:
        ids: List[int] = []
        for piece in pre_tokenize(text):
            cached = self.cache.get(piece)
            if cached is not None:
                ids.extend(cached)
                continue
            piece_ids = []
            for sym in self._bpe(piece):
                tid = self.token_to_id.get(sym)
                if tid is None:
                    raise ValueError(f"missing symbol {sym!r}")
                piece_ids.append(tid)
            if len(self.cache) < 100000:
                self.cache[piece] = piece_ids
            ids.extend(piece_ids)
        return ids

    def encode(self, text: str) -> List[int]:
        """Encode while honoring special tokens."""
        if not self.special_tokens:
            return self.encode_ordinary(text)
        specials = sorted(self.special_tokens, key=len, reverse=True)
        pattern = "(" + "|".join(re.escape(token) for token in specials) + ")"
        ids: List[int] = []
        for part in re.split(pattern, text):
            if not part:
                continue
            if part in self.special_tokens:
                ids.append(self.token_to_id[part])
            else:
                ids.extend(self.encode_ordinary(part))
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        pieces: List[str] = []
        for token_id in ids:
            if 0 <= token_id < len(self.vocab):
                pieces.append(self.vocab[token_id])
        encoded = bytearray()
        for char in "".join(pieces):
            byte = BYTE_DECODER.get(char)
            if byte is not None:
                encoded.append(byte)
        return encoded.decode("utf-8", errors="replace")

    def id_for(self, token: str) -> Optional[int]:
        return self.token_to_id.get(token)

    def to_json(self) -> Dict:
        return {
            "version": 1,
            "specialTokens": list(self.special_tokens),
            "vocab": list(self.vocab),
            "merges": [list(pair) for pair in self.merges],
        }


def load_tokenizer(path: Path | str) -> BpeTokenizer:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("tokenizer.json must contain a JSON object")
    for key in ("vocab", "merges"):
        if key not in data:
            raise ValueError(f"tokenizer.json is missing {key!r}")
    return BpeTokenizer(data)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Encode or decode text with a BPE tokenizer.")
    parser.add_argument("tokenizer", type=Path)
    parser.add_argument("text", nargs="?")
    parser.add_argument("--decode", action="store_true")
    args = parser.parse_args()
    tokenizer = load_tokenizer(args.tokenizer)
    if args.text is None:
        print(f"vocab {len(tokenizer.vocab):,}; merges {len(tokenizer.merges):,}")
    elif args.decode:
        print(tokenizer.decode([int(part) for part in args.text.split(",")]))
    else:
        ids = tokenizer.encode(args.text)
        print(",".join(str(token_id) for token_id in ids))
