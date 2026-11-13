#!/usr/bin/env python3
"""Write Only Gates AI model bundles from Python training jobs.

The browser trainer exports ``only-gates-model-bundle`` ZIPs from
``apps/only-gates-ai/src/lib/modelBundle.ts``.  This module produces the same
flat archive from a Python model so a Colab run can be imported with the
existing **Import model (.zip)** button.

The bundle is inference-only: it contains ``manifest.json``,
``checkpoint.json``, ``tokenizer.json``, ``weights.bin``, ``training.json``,
and ``README.txt``.  It deliberately omits ``training-state.json`` and sets
``training.resumable`` to false.

Keep this file dependency-free (standard library only) so it can be tested
without PyTorch or NumPy.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


CONFIG_KEYS = ("vocabSize", "context", "dim", "layers", "heads", "mlp")
MAX_SOURCE_URLS = 256
MAX_PROMPTS = 64
MAX_SAMPLES = 64
MAX_TEXT = 200_000


def parameter_count(config: Mapping[str, int]) -> int:
    """Mirror ``tinyGptParameterCount`` from the browser and C engine."""
    vocab = int(config["vocabSize"])
    context = int(config["context"])
    dim = int(config["dim"])
    layers = int(config["layers"])
    mlp = int(config["mlp"])
    return (
        vocab * dim
        + context * dim
        + dim
        + layers * (2 * dim + 4 * dim * dim + 2 * dim * mlp)
    )


def _finite(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _finite_or_zero(value: Any) -> float:
    number = _finite(value)
    return 0.0 if number is None else number


def _clean_config(value: Mapping[str, Any]) -> Dict[str, int]:
    config = {key: int(value[key]) for key in CONFIG_KEYS}
    if config["dim"] % config["heads"] != 0:
        raise ValueError("model.config.dim must be divisible by heads")
    if parameter_count(config) <= 0:
        raise ValueError("model config has no parameters")
    return config


def _clean_prompts(prompts: Sequence[str]) -> List[str]:
    cleaned: List[str] = []
    for prompt in prompts[:MAX_PROMPTS]:
        text = str(prompt).strip()
        if text:
            cleaned.append(text[:4000])
    return cleaned


def _clean_samples(samples: Sequence[Mapping[str, Any]]) -> List[Dict[str, str]]:
    cleaned: List[Dict[str, str]] = []
    for sample in samples[:MAX_SAMPLES]:
        prompt = str(sample.get("prompt", ""))[:4000]
        text = str(sample.get("text", ""))[:MAX_TEXT]
        cleaned.append({"prompt": prompt, "text": text})
    return cleaned


def _clean_history(history: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    points: List[Dict[str, Any]] = []
    for point in history:
        item: Dict[str, Any] = {
            "step": int(point.get("step", 0)),
            "loss": _finite(point.get("loss")),
            "smoothedLoss": _finite(point.get("smoothedLoss")),
            "lr": _finite_or_zero(point.get("lr")),
            "elapsedMs": _finite_or_zero(point.get("elapsedMs")),
        }
        val_loss = _finite(point.get("valLoss"))
        if val_loss is not None:
            item["valLoss"] = val_loss
        points.append(item)
    return points


def _json(value: Any) -> str:
    """Serialize JSON that the browser's JSON.parse will accept."""
    return json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _readme(manifest: Mapping[str, Any]) -> str:
    config = manifest["model"]["config"]
    lines = [
        "Only Gates AI - Integer GPT model bundle",
        "",
        f"Kind: {manifest['kind']}",
        f"Model: {manifest['model']['name']}",
        f"Created: {manifest['createdAt']}",
        "Architecture: tiny-gpt",
        (
            f"Config: {config['vocabSize']} vocab / {config['context']} context / "
            f"{config['dim']} dim / {config['layers']} layers / "
            f"{config['heads']} heads / {config['mlp']} mlp"
        ),
        f"Parameters: {manifest['model']['parameterCount']:,}",
        "",
        "Import this file from the Integer GPT training panel:",
        'choose "Import model (.zip)" and select this archive.',
        "",
        "This bundle is inference-only and cannot be resumed for further training.",
        "",
        "Files:",
        "  manifest.json        describes the model and where it loads",
        "  checkpoint.json      model metadata and prompt suggestions",
        "  tokenizer.json       BPE tokenizer definition",
        "  weights.bin          float32 little-endian weights",
        "  training.json        training summary and loss history",
        "",
    ]
    return "\n".join(lines)


def write_model_bundle(
    out_dir: os.PathLike[str] | str,
    bundle_path: Optional[os.PathLike[str] | str],
    *,
    config: Mapping[str, Any],
    tokenizer_model: Mapping[str, Any],
    weights: bytes | bytearray | memoryview,
    name: str,
    source: str,
    license: str,
    prompts: Sequence[str],
    samples: Sequence[Mapping[str, Any]],
    training: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    assistant: bool,
    kind: str = "custom",
    checkpoint_format: str = "only-gates-tiny-gpt-custom",
    generator_feature: str = "colab-training",
) -> Path:
    """Write the app-compatible artifacts and return the ZIP path.

    ``weights`` must already be packed in ``tiny_gpt_create`` order and
    little-endian float32.  ``training`` must contain the fields the browser
    importer validates (steps, batch, learningRate, dropout, seed, bestStep,
    bestLoss, elapsedMs, corpusHash, sourceUrls, assistant).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    bundle = Path(bundle_path) if bundle_path is not None else out / "only-gates-custom-bundle.zip"

    cleaned_config = _clean_config(config)
    params = parameter_count(cleaned_config)
    weight_bytes = bytes(weights)
    if len(weight_bytes) != params * 4:
        raise ValueError(
            f"weights.bin has {len(weight_bytes):,} bytes but the config needs "
            f"{params * 4:,}"
        )
    if cleaned_config["vocabSize"] != int(len(tokenizer_model.get("vocab", []))):
        raise ValueError("tokenizer vocabulary does not match model.config.vocabSize")

    source_urls = [
        str(url) for url in (training.get("sourceUrls") or [])
    ][:MAX_SOURCE_URLS]
    cleaned_prompts = _clean_prompts(prompts)
    cleaned_samples = _clean_samples(samples)
    best_loss = _finite(training.get("bestLoss"))
    training_info: Dict[str, Any] = {
        "steps": int(training.get("steps", 0)),
        "batch": int(training.get("batch", 1)),
        "learningRate": _finite_or_zero(training.get("learningRate")),
        "dropout": _finite_or_zero(training.get("dropout")),
        "seed": int(training.get("seed", 0)),
        "bestStep": int(training.get("bestStep", 0)),
        "bestLoss": best_loss,
        "elapsedMs": int(_finite_or_zero(training.get("elapsedMs"))),
        "corpusHash": str(training.get("corpusHash", "")),
        "resumable": False,
        "sourceUrls": source_urls,
        "assistant": bool(assistant),
        "backend": "colab",
    }

    model_meta: Dict[str, Any] = {
        "name": name[:400],
        "source": source[:4000],
        "license": license[:4000],
        "training": {
            "assistant": bool(assistant),
            "mode": "qa" if assistant else "docs",
            "steps": training_info["steps"],
            "batch": training_info["batch"],
            "learningRate": training_info["learningRate"],
            "dropout": training_info["dropout"],
            "seed": training_info["seed"],
            "bestStep": training_info["bestStep"],
            "bestLoss": best_loss,
            "sourceUrls": source_urls,
        },
    }

    manifest = {
        "format": "only-gates-model-bundle",
        "version": 1,
        "kind": kind,
        "createdAt": _timestamp(),
        "generator": {
            "app": "only-gates-ai",
            "feature": generator_feature,
        },
        "target": {
            "panel": "integer-gpt",
            "example": kind,
            "runtime": "integer-wasm",
            "promptFormat": "chat" if assistant else "text",
        },
        "model": {
            "name": name[:400],
            "architecture": "tiny-gpt",
            "config": cleaned_config,
            "parameterCount": params,
            "tokenizer": {"file": "tokenizer.json", "type": "bpe"},
            "weights": {
                "file": "weights.bin",
                "dtype": "float32",
                "endianness": "little",
                "layout": "tiny_gpt_create",
            },
            "prompts": cleaned_prompts,
            "samples": cleaned_samples,
            "meta": model_meta,
        },
        "training": training_info,
        "integrity": {"algorithm": "sha256", "weights": _sha256_bytes(weight_bytes)},
    }

    checkpoint = {
        "format": checkpoint_format,
        "version": 1,
        "config": cleaned_config,
        "parameterCount": params,
        "tokenizer": "tokenizer.json",
        "meta": model_meta,
        "samples": cleaned_samples,
        "prompts": cleaned_prompts,
    }

    summary = {
        "format": "only-gates-training-summary",
        "version": 1,
        "kind": kind,
        "training": {**training_info, "resumable": False},
        "history": _clean_history(history),
    }

    files: List[tuple[str, bytes]] = [
        ("manifest.json", _json(manifest).encode("utf-8")),
        ("checkpoint.json", _json(checkpoint).encode("utf-8")),
        ("tokenizer.json", _json(tokenizer_model).encode("utf-8")),
        ("weights.bin", weight_bytes),
        ("training.json", _json(summary).encode("utf-8")),
        ("README.txt", _readme(manifest).encode("utf-8")),
    ]

    # Keep loose artifacts next to the checkpoint for inspection and for users
    # who prefer to copy the model files into a public model directory.
    for filename, data in files:
        (out / filename).write_bytes(data)

    bundle.parent.mkdir(parents=True, exist_ok=True)
    tmp = bundle.with_name(bundle.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for filename, data in files:
            archive.writestr(filename, data)
    os.replace(tmp, bundle)
    return bundle
