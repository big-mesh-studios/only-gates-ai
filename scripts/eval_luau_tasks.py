#!/usr/bin/env python3
"""Evaluate an exported Luau task model with real Luau execution.

The model is loaded from a training checkpoint (``latest.pt``) or an exported
model directory (``checkpoint.json`` + ``weights.bin`` + ``tokenizer.json``).
Generated code is written to a temporary Luau file, run with a timeout, and
compared against the task's harness.

Examples
--------

Greedy pass@1 on the held-out canonical split::

    python3 scripts/eval_luau_tasks.py \
      --model-dir /content/drive/MyDrive/tiny-gpt-luau \
      --tasks data/luau-tasks/test-seen.jsonl \
      --luau-bin /usr/local/bin/luau

pass@5 with sampling::

    python3 scripts/eval_luau_tasks.py \
      --model-dir /content/drive/MyDrive/tiny-gpt-luau \
      --tasks data/luau-tasks/test-seen.jsonl \
      --pass-k 5 --temperature 0.7

Generate only, without executing (useful on a machine without Luau)::

    python3 scripts/eval_luau_tasks.py \
      --model-dir /content/drive/MyDrive/tiny-gpt-luau \
      --tasks data/luau-tasks/test-seen.jsonl \
      --limit 10 --no-exec
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from bpe_tokenizer import BYTE_DECODER, BpeTokenizer, load_tokenizer
from luau_exec import analyze_luau, execute_luau, strip_code_fence

import numpy as np
import torch

import train_tiny_gpt_solid2_colab as base


CONFIG_KEYS = ("vocabSize", "context", "dim", "layers", "heads", "mlp")
DEFAULT_TASKS = SCRIPT_DIR.parent / "data" / "luau-tasks" / "test-seen.jsonl"
SPECIAL_TEXT = {"<|pad|>", "<|user|>", "<|assistant|>", "<|end|>", "<|facts|>"}


def iter_rows(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def clean_config(value: Mapping[str, object]) -> Dict[str, int]:
    config = {key: int(value[key]) for key in CONFIG_KEYS}
    if config["dim"] % config["heads"] != 0:
        raise SystemExit("model config: dim must be divisible by heads")
    return config


def unpack_weights(model: torch.nn.Module, packed: np.ndarray) -> None:
    """Fill a TinyGPT model from packed ``tiny_gpt_create`` float32 weights."""
    vocab, context, dim, layers, mlp = (
        model.vocab_size,
        model.context,
        model.dim,
        model.layers_n,
        model.mlp_n,
    )
    offset = 0

    def take(count: int) -> torch.Tensor:
        nonlocal offset
        chunk = packed[offset : offset + count]
        if len(chunk) != count:
            raise ValueError("packed weights are truncated")
        offset += count
        return torch.from_numpy(chunk.copy())

    with torch.no_grad():
        model.token_emb.weight.copy_(take(vocab * dim).view(vocab, dim))
        model.pos_emb.weight.copy_(take(context * dim).view(context, dim))
        model.g_final.gain.copy_(take(dim))
        for block in model.blocks:
            block.g1.gain.copy_(take(dim))
            block.wqkv.weight.copy_(take(dim * dim * 3).view(dim, dim * 3).T)
            block.wo.weight.copy_(take(dim * dim).view(dim, dim).T)
            block.g2.gain.copy_(take(dim))
            block.w1.weight.copy_(take(dim * mlp).view(dim, mlp).T)
            block.w2.weight.copy_(take(mlp * dim).view(mlp, dim).T)
    if offset != packed.size:
        raise ValueError("packed weights have trailing values")


def prepare_model_dir(path: Path) -> Tuple[Path, Optional[tempfile.TemporaryDirectory]]:
    """Return a directory containing model artifacts, extracting a bundle zip."""
    if path.is_dir():
        return path, None
    if path.suffix.lower() == ".zip":
        handle = tempfile.TemporaryDirectory(prefix="luau-bundle-")
        with zipfile.ZipFile(path) as archive:
            archive.extractall(handle.name)
        return Path(handle.name), handle
    raise SystemExit(f"model path is neither a directory nor a .zip bundle: {path}")


def load_model(
    model_dir: Path,
    device: torch.device,
    checkpoint_path: Optional[Path] = None,
) -> Tuple[torch.nn.Module, Dict[str, int], Dict, Optional[Dict]]:
    """Load a TinyGPT model and return ``(model, config, checkpoint, payload)``."""
    latest = checkpoint_path or (model_dir / "latest.pt")
    payload: Optional[Dict] = None
    tokenizer_model: Optional[Dict] = None

    if latest.exists():
        payload = torch.load(latest, map_location="cpu", weights_only=False)
        config = clean_config(payload["config"])
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
        checkpoint = dict(payload.get("config", config))
        model.eval()
        return model, config, checkpoint, tokenizer_model

    checkpoint_path = model_dir / "checkpoint.json"
    weights_path = model_dir / "weights.bin"
    if not checkpoint_path.exists() or not weights_path.exists():
        raise SystemExit(
            f"no model found in {model_dir}; expected latest.pt or "
            "checkpoint.json + weights.bin + tokenizer.json"
        )
    checkpoint_json = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    config = clean_config(checkpoint_json["config"])
    packed = np.fromfile(weights_path, dtype="<f4")
    expected = base.parameter_count(**{
        "vocab": config["vocabSize"],
        "ctx": config["context"],
        "dim": config["dim"],
        "layers": config["layers"],
        "mlp": config["mlp"],
    })
    if packed.size != expected:
        raise SystemExit(
            f"weights.bin has {packed.size:,} values but the config needs {expected:,}"
        )
    model = base.TinyGPT(
        config["vocabSize"],
        config["context"],
        config["dim"],
        config["layers"],
        config["heads"],
        config["mlp"],
        0.0,
    ).to(device)
    unpack_weights(model, packed)
    model.eval()
    return model, config, checkpoint_json, tokenizer_model


def get_tokenizer(model_dir: Path, embedded: Optional[Dict]) -> BpeTokenizer:
    if embedded:
        return BpeTokenizer(embedded)
    path = model_dir / "tokenizer.json"
    if not path.exists():
        raise SystemExit(f"missing tokenizer: {path}")
    return load_tokenizer(path)


@torch.no_grad()
def generate_greedy(
    model: torch.nn.Module,
    prompt_ids: Sequence[int],
    *,
    context: int,
    max_new: int,
    device: torch.device,
    end_id: int,
) -> Tuple[List[int], bool]:
    ids = list(prompt_ids)
    generated: List[int] = []
    model.eval()
    for _ in range(max_new):
        if len(ids) >= context:
            break
        x = torch.tensor([ids], dtype=torch.long, device=device)
        logits, _ = model(x)
        next_id = int(torch.argmax(logits[0, -1]).item())
        generated.append(next_id)
        ids.append(next_id)
        if next_id == end_id:
            return generated, True
    return generated, False


@torch.no_grad()
def generate_sampled(
    model: torch.nn.Module,
    prompt_ids: Sequence[int],
    *,
    context: int,
    max_new: int,
    temperature: float,
    top_k: int,
    device: torch.device,
    end_id: int,
    seed: int,
) -> Tuple[List[int], bool]:
    torch.manual_seed(seed)
    ids = list(prompt_ids)
    generated: List[int] = []
    model.eval()
    for _ in range(max_new):
        if len(ids) >= context:
            break
        x = torch.tensor([ids], dtype=torch.long, device=device)
        logits, _ = model(x)
        logits = logits[0, -1] / max(temperature, 1e-4)
        if top_k > 0:
            values, _ = torch.topk(logits, min(top_k, logits.numel()))
            logits = logits.masked_fill(logits < values[-1], float("-inf"))
        probabilities = torch.softmax(logits, dim=-1)
        next_id = int(torch.multinomial(probabilities, 1).item())
        generated.append(next_id)
        ids.append(next_id)
        if next_id == end_id:
            return generated, True
    return generated, False


def decode_generated(tokenizer: BpeTokenizer, ids: Sequence[int]) -> str:
    special_ids = {
        token_id
        for text, token_id in tokenizer.token_to_id.items()
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


def prompt_ids_for(
    tokenizer: BpeTokenizer,
    instruction: str,
    mode: str,
) -> List[int]:
    user_id = tokenizer.id_for("<|user|>")
    assistant_id = tokenizer.id_for("<|assistant|>")
    if user_id is None or assistant_id is None:
        raise SystemExit("tokenizer is missing <|user|> or <|assistant|>")
    prefix = "Task: " if mode == "write" else "Question: "
    return [user_id] + tokenizer.encode_ordinary(prefix + instruction) + [assistant_id]


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return float(ordered[index])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--tasks",
        type=Path,
        action="append",
        default=None,
        help="JSONL file (repeatable). Default: data/luau-tasks/test-seen.jsonl",
    )
    parser.add_argument("--luau-bin", default="luau")
    parser.add_argument("--analyze-bin", default=None)
    parser.add_argument("--pass-k", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--max-new", type=int, default=160)
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-exec", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--show-failures", type=int, default=10)
    args = parser.parse_args()

    if args.pass_k < 1:
        raise SystemExit("--pass-k must be at least 1")
    if args.pass_k > 1 and args.temperature <= 0:
        args.temperature = 0.7
        print("pass@k > 1: using temperature 0.7 for sampling", flush=True)

    model_dir, temp_handle = prepare_model_dir(args.model_dir)
    # Keep ``temp_handle`` referenced for the rest of the process so an
    # extracted bundle directory is not cleaned up while the model is loaded.
    _ = temp_handle
    device = torch.device(args.device)
    model, config, checkpoint, embedded_tokenizer = load_model(
        model_dir, device, args.checkpoint
    )
    tokenizer = get_tokenizer(model_dir, embedded_tokenizer)
    end_id = tokenizer.id_for("<|end|>")
    if end_id is None:
        raise SystemExit("tokenizer is missing <|end|>")

    task_paths: List[Path] = args.tasks or [DEFAULT_TASKS]
    rows: List[Dict] = []
    for path in task_paths:
        if not path.exists():
            raise SystemExit(f"task file not found: {path}")
        rows.extend(iter_rows(path))
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("no tasks to evaluate")

    print(
        f"model: {model_dir} ({checkpoint.get('name', 'luau')})\n"
        f"config: {config['vocabSize']} vocab / {config['context']} ctx / "
        f"{config['dim']} dim / {config['layers']} layers / {config['heads']} heads / "
        f"{config['mlp']} mlp\n"
        f"device: {device}; tasks: {len(rows):,}; pass@{args.pass_k}; "
        f"temperature={args.temperature}",
        flush=True,
    )

    records: List[Dict] = []
    per_family: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: {
            "pass1": [],
            "passk": [],
            "syntax": [],
            "stop": [],
            "truncated": [],
            "tokens": [],
        }
    )
    failures: List[Dict] = []

    for index, row in enumerate(rows, 1):
        family = str(row.get("family", "unknown"))
        instruction = str(row.get("instruction", ""))
        mode = str(row.get("mode", "write"))
        prompt_ids = prompt_ids_for(tokenizer, instruction, mode)
        if len(prompt_ids) >= config["context"]:
            print(f"skip {row.get('id')}: prompt is {len(prompt_ids)} tokens", flush=True)
            continue

        samples: List[Dict] = []
        first_ok = False
        any_ok = False
        first_syntax = False
        first_stop = False
        first_truncated = False
        code_tokens: List[int] = []
        for sample_index in range(args.pass_k):
            if args.temperature <= 0:
                generated, stopped = generate_greedy(
                    model,
                    prompt_ids,
                    context=config["context"],
                    max_new=args.max_new,
                    device=device,
                    end_id=end_id,
                )
            else:
                generated, stopped = generate_sampled(
                    model,
                    prompt_ids,
                    context=config["context"],
                    max_new=args.max_new,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    device=device,
                    end_id=end_id,
                    seed=args.seed + index * 1009 + sample_index,
                )
            code = strip_code_fence(decode_generated(tokenizer, generated))
            code_tokens.append(len(generated))
            truncated = len(generated) >= args.max_new and not stopped
            syntax_ok = True
            if args.analyze_bin:
                analyzed = analyze_luau(code, args.analyze_bin, timeout=args.timeout)
                syntax_ok = bool(analyzed.get("ok"))
            execution = {
                "ok": None,
                "reason": "not-executed",
                "stdout": "",
                "stderr": "",
            }
            if not args.no_exec and mode == "write":
                result = execute_luau(
                    code,
                    row.get("harness", {}),
                    args.luau_bin,
                    timeout=args.timeout,
                )
                execution = {
                    "ok": result.ok,
                    "reason": result.reason,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "syntax_error": result.syntax_error,
                }
                if result.syntax_error:
                    syntax_ok = False
            sample = {
                "sample": sample_index,
                "code": code,
                "stopped": stopped,
                "truncated": truncated,
                "token_count": len(generated),
                "syntax_ok": syntax_ok,
                "execution": execution,
            }
            samples.append(sample)
            ok = bool(execution["ok"]) if execution["ok"] is not None else True
            any_ok = any_ok or ok
            if sample_index == 0:
                first_ok = ok
                first_syntax = syntax_ok
                first_stop = stopped
                first_truncated = truncated
                if not ok:
                    failures.append(
                        {
                            "id": row.get("id"),
                            "family": family,
                            "instruction": instruction,
                            "code": code,
                            "reason": execution["reason"],
                            "stderr": execution["stderr"],
                            "expected_stdout": row.get("harness", {}).get(
                                "expect_stdout"
                            ),
                            "stdout": execution["stdout"],
                        }
                    )

        values = per_family[family]
        values["pass1"].append(1.0 if first_ok else 0.0)
        values["passk"].append(1.0 if any_ok else 0.0)
        values["syntax"].append(1.0 if first_syntax else 0.0)
        values["stop"].append(1.0 if first_stop else 0.0)
        values["truncated"].append(1.0 if first_truncated else 0.0)
        values["tokens"].extend(float(item) for item in code_tokens)
        records.append(
            {
                "id": row.get("id"),
                "family": family,
                "mode": mode,
                "instruction": instruction,
                "sample_count": len(samples),
                "first_ok": first_ok,
                "any_ok": any_ok,
                "first_syntax_ok": first_syntax,
                "first_stopped": first_stop,
                "first_truncated": first_truncated,
                "samples": samples,
            }
        )
        if index % 20 == 0:
            print(f"evaluated {index}/{len(rows)}", flush=True)

    def mean(values: Sequence[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    # Group by held-out split so it is obvious whether the model generalises
    # to new wording (paraphrase) and new identifiers (unseen), not just the
    # canonical prompts.
    per_split: Dict[str, List[Dict]] = defaultdict(list)
    for record in records:
        parts = str(record.get("id", "")).split("/")
        per_split[parts[1] if len(parts) > 1 else "unknown"].append(record)

    def split_summary(items: Sequence[Dict]) -> Dict[str, float]:
        return {
            "rows": len(items),
            "pass@1": mean([1.0 if item["first_ok"] else 0.0 for item in items]),
            "pass@k": mean([1.0 if item["any_ok"] else 0.0 for item in items]),
            "syntax": mean(
                [1.0 if item["first_syntax_ok"] else 0.0 for item in items]
            ),
            "stop": mean([1.0 if item["first_stopped"] else 0.0 for item in items]),
            "truncated": mean(
                [1.0 if item["first_truncated"] else 0.0 for item in items]
            ),
        }

    print()
    print(
        f"{'family':24s} {'rows':>6s} {'pass@1':>8s} {'pass@k':>8s} "
        f"{'syntax':>8s} {'stop':>8s} {'trunc':>7s} {'tok p50':>8s}"
    )
    print("-" * 84)
    for family in sorted(per_family):
        values = per_family[family]
        print(
            f"{family:24s} {len(values['pass1']):>6d} "
            f"{mean(values['pass1']):>8.2%} {mean(values['passk']):>8.2%} "
            f"{mean(values['syntax']):>8.2%} {mean(values['stop']):>8.2%} "
            f"{mean(values['truncated']):>7.2%} "
            f"{percentile(values['tokens'], 0.5):>8.0f}"
        )

    all_values = {
        key: [value for values in per_family.values() for value in values[key]]
        for key in ("pass1", "passk", "syntax", "stop", "truncated")
    }
    print("-" * 84)
    print(
        f"{'all':24s} {len(all_values['pass1']):>6d} "
        f"{mean(all_values['pass1']):>8.2%} {mean(all_values['passk']):>8.2%} "
        f"{mean(all_values['syntax']):>8.2%} {mean(all_values['stop']):>8.2%} "
        f"{mean(all_values['truncated']):>7.2%}"
    )

    print()
    print(f"{'split':24s} {'rows':>6s} {'pass@1':>8s} {'pass@k':>8s} {'stop':>8s}")
    print("-" * 60)
    for split in sorted(per_split):
        stats = split_summary(per_split[split])
        print(
            f"{split:24s} {stats['rows']:>6d} "
            f"{stats['pass@1']:>8.2%} {stats['pass@k']:>8.2%} "
            f"{stats['stop']:>8.2%}"
        )

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

    report = {
        "modelDir": str(model_dir),
        "checkpoint": checkpoint,
        "config": config,
        "tasks": [str(path) for path in task_paths],
        "rows": len(records),
        "passK": args.pass_k,
        "temperature": args.temperature,
        "summary": {
            "pass@1": mean(all_values["pass1"]),
            f"pass@{args.pass_k}": mean(all_values["passk"]),
            "syntax": mean(all_values["syntax"]),
            "stop": mean(all_values["stop"]),
            "truncated": mean(all_values["truncated"]),
        },
        "families": {
            family: {
                "rows": len(values["pass1"]),
                "pass@1": mean(values["pass1"]),
                f"pass@{args.pass_k}": mean(values["passk"]),
                "syntax": mean(values["syntax"]),
                "stop": mean(values["stop"]),
                "truncated": mean(values["truncated"]),
                "token_p50": percentile(values["tokens"], 0.5),
            }
            for family, values in sorted(per_family.items())
        },
        "splits": {
            split: split_summary(items) for split, items in sorted(per_split.items())
        },
        "failures": failures,
        "records": records,
    }
    out = args.out or Path("eval-report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    raise SystemExit(0 if not failures else 1)


if __name__ == "__main__":
    main()
