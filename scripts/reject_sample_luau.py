#!/usr/bin/env python3
"""Generate execution-verified Luau training rows by rejection sampling.

For every task in ``--tasks``, sample several candidate completions from a
trained checkpoint, execute each one with the Luau CLI against the row's
harness, and keep the candidates that pass. The output is a JSONL file in the
same schema as ``data/luau-tasks/*.jsonl`` and can be fed back to
``train_tiny_gpt_luau_colab.py`` (via ``--extra-tasks``) for a short
fine-tuning pass.

Example
-------
    python3 scripts/reject_sample_luau.py \\
        --model-dir /content/drive/MyDrive/tiny-gpt-luau \\
        --tasks data/luau-tasks/train.jsonl \\
        --luau-bin /usr/local/bin/luau \\
        --samples 8 --temperature 0.7 --out data/luau-tasks-verified/train.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np
import torch

from bpe_tokenizer import BYTE_DECODER, BpeTokenizer, load_tokenizer
from luau_exec import execute_luau, strip_code_fence

import train_tiny_gpt_luau_colab as luau_trainer
import train_tiny_gpt_solid2_colab as base


CONFIG_KEYS = ("vocabSize", "context", "dim", "layers", "heads", "mlp")
SPECIAL_TEXT = {"<|pad|>", "<|user|>", "<|assistant|>", "|", "<|facts|>"}


def iter_rows(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def prepare_model_dir(path: Path) -> Tuple[Path, Optional[tempfile.TemporaryDirectory]]:
    if path.is_dir():
        return path, None
    if path.suffix.lower() == ".zip":
        handle = tempfile.TemporaryDirectory(prefix="luau-bundle-")
        with zipfile.ZipFile(path) as archive:
            archive.extractall(handle.name)
        return Path(handle.name), handle
    raise SystemExit(f"model path is neither a directory nor a .zip bundle: {path}")


def load_model_and_tokenizer(model_dir: Path, device: torch.device):
    latest = model_dir / "latest.pt"
    embedded = None
    if latest.exists():
        payload = torch.load(latest, map_location="cpu", weights_only=False)
        config = {key: int(payload["config"][key]) for key in CONFIG_KEYS}
        tokenizer_model = payload.get("tokenizer_model")
        model = base.TinyGPT(
            config["vocabSize"],
            config["context"],
            config["dim"],
            config["layers"],
            config["heads"],
            config["mlp"],
            float(payload.get("dropout", 0.0)),
        ).to(device)
        model.load_state_dict(payload["model"])
        model.eval()
        tokenizer = BpeTokenizer(tokenizer_model) if tokenizer_model else None
        if tokenizer is None:
            tokenizer = load_tokenizer(model_dir / "tokenizer.json")
        return model, config, tokenizer, int(payload.get("seed", 1337))
    checkpoint_path = model_dir / "checkpoint.json"
    weights_path = model_dir / "weights.bin"
    if not checkpoint_path.exists() or not weights_path.exists():
        raise SystemExit(f"no model found in {model_dir}")
    checkpoint_json = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    config = {key: int(checkpoint_json["config"][key]) for key in CONFIG_KEYS}
    packed = np.fromfile(weights_path, dtype="<f4")
    model = base.TinyGPT(
        config["vocabSize"], config["context"], config["dim"],
        config["layers"], config["heads"], config["mlp"], 0.0,
    ).to(device)
    luau_trainer.unpack_weights(model, packed)
    model.eval()
    tokenizer = load_tokenizer(model_dir / "tokenizer.json")
    return model, config, tokenizer, 1337


def decode_generated(tokenizer: BpeTokenizer, ids: List[int]) -> str:
    special_ids = {
        token_id for text, token_id in tokenizer.token_to_id.items()
        if text in SPECIAL_TEXT
    }
    pieces = [
        tokenizer.vocab[token_id]
        for token_id in ids
        if token_id not in special_ids and 0 <= token_id < len(tokenizer.vocab)
    ]
    encoded = bytearray()
    for char in "".join(pieces):
        byte = BYTE_DECODER.get(char)
        if byte is not None:
            encoded.append(byte)
    return encoded.decode("utf-8", errors="replace")


@torch.no_grad()
def sample_candidates(
    model: torch.nn.Module,
    tokenizer: BpeTokenizer,
    prompt_ids: List[int],
    *,
    context: int,
    max_new: int,
    temperature: float,
    top_k: int,
    device: torch.device,
    end_id: int,
    count: int,
    seed: int,
) -> List[Tuple[List[int], bool]]:
    results: List[Tuple[List[int], bool]] = []
    for sample_index in range(count):
        torch.manual_seed(seed + sample_index * 9973)
        ids = list(prompt_ids)
        generated: List[int] = []
        stopped = False
        for _ in range(max_new):
            if len(ids) >= context:
                break
            x = torch.tensor([ids], dtype=torch.long, device=device)
            logits, _ = model(x)
            logits = logits[0, -1]
            if temperature > 0:
                logits = logits / max(temperature, 1e-4)
                if top_k > 0:
                    values, _ = torch.topk(logits, min(top_k, logits.numel()))
                    logits = logits.masked_fill(logits < values[-1], float("-inf"))
                probabilities = torch.softmax(logits, dim=-1)
                next_id = int(torch.multinomial(probabilities, 1).item())
            else:
                next_id = int(torch.argmax(logits).item())
            generated.append(next_id)
            ids.append(next_id)
            if next_id == end_id:
                stopped = True
                break
        results.append((generated, stopped))
    return results


def prompt_ids_for(tokenizer: BpeTokenizer, instruction: str, mode: str) -> List[int]:
    user_id = tokenizer.id_for("<|user|>")
    assistant_id = tokenizer.id_for("<|assistant|>")
    if user_id is None or assistant_id is None:
        raise SystemExit("tokenizer is missing <|user|> or <|assistant|>")
    prefix = "Task: " if mode == "write" else "Question: "
    return [user_id] + tokenizer.encode_ordinary(prefix + instruction) + [assistant_id]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--tasks", type=Path, action="append", default=None)
    parser.add_argument("--luau-bin", default="luau")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--max-new", type=int, default=160)
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    model_dir, temp_handle = prepare_model_dir(args.model_dir)
    _ = temp_handle
    device = torch.device(args.device)
    model, config, tokenizer, base_seed = load_model_and_tokenizer(model_dir, device)
    end_id = tokenizer.id_for("|")
    if end_id is None:
        raise SystemExit("tokenizer is missing |")
    context = int(config["context"])

    task_paths = args.tasks or [Path("data/luau-tasks/train.jsonl")]
    rows: List[Dict] = []
    for path in task_paths:
        if not path.exists():
            raise SystemExit(f"task file not found: {path}")
        rows.extend(iter_rows(path))
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("no tasks to process")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    seen_pairs = set()
    kept = 0
    total_candidates = 0
    with args.out.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows, 1):
            mode = str(row.get("mode", "write"))
            if mode != "write":
                continue
            instruction = str(row.get("instruction", ""))
            prompt_ids = prompt_ids_for(tokenizer, instruction, mode)
            if len(prompt_ids) >= context:
                continue
            harness = row.get("harness", {})
            candidates = sample_candidates(
                model, tokenizer, prompt_ids,
                context=context, max_new=args.max_new,
                temperature=args.temperature, top_k=args.top_k,
                device=device, end_id=end_id, count=args.samples,
                seed=args.seed + index * 1009,
            )
            for generated, stopped in candidates:
                total_candidates += 1
                code = strip_code_fence(decode_generated(tokenizer, generated))
                result = execute_luau(code, harness, args.luau_bin, timeout=args.timeout)
                if not result.ok:
                    continue
                key = (instruction, code)
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                out_row = dict(row)
                out_row["id"] = f"{row.get('family', 'unknown')}/rejected/{kept:06d}"
                out_row["code"] = code
                out_row["split"] = "train"
                out_row["prompt_style"] = str(row.get("prompt_style", "canonical"))
                handle.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                handle.flush()
                kept += 1
            if index % 20 == 0:
                print(f"processed {index}/{len(rows)}; kept {kept}", flush=True)

    print(f"candidates={total_candidates} kept={kept} -> {args.out}")


if __name__ == "__main__":
    main()
PY