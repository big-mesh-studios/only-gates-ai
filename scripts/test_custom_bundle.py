#!/usr/bin/env python3
"""Tests for the Python model-bundle writer.

Run with:
  python scripts/test_custom_bundle.py

This deliberately avoids PyTorch and NumPy: the Colab trainer imports those,
but the ZIP/JSON contract can be checked with the standard library alone.
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
import zipfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import model_bundle  # noqa: E402

REPO = SCRIPT_DIR.parent
TOKENIZER = REPO / "apps" / "only-gates-ai" / "public" / "models" / "tiny-gpt-solid2" / "tokenizer.json"


def tiny_config() -> dict:
    return {"vocabSize": 4096, "context": 8, "dim": 8, "layers": 1, "heads": 2, "mlp": 16}


def main() -> None:
    tokenizer_model = json.loads(TOKENIZER.read_text(encoding="utf-8"))
    config = tiny_config()
    params = model_bundle.parameter_count(config)
    assert params == 33_368, params

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out"
        bundle = Path(tmp) / "custom.zip"
        written = model_bundle.write_model_bundle(
            out,
            bundle,
            config=config,
            tokenizer_model=tokenizer_model,
            weights=b"\x00" * (params * 4),
            name="test-custom",
            source="fixture",
            license="fixture",
            prompts=["How do I use this?", "What is it?"],
            samples=[{"prompt": "How do I use this?", "text": "Use it."}],
            training={
                "steps": 12,
                "batch": 2,
                "learningRate": 0.0004,
                "dropout": 0.1,
                "seed": 1337,
                "bestStep": 10,
                "bestLoss": math.nan,
                "elapsedMs": 1234.5,
                "corpusHash": "abc123",
                "sourceUrls": ["https://example.com/docs/guide.md"],
            },
            history=[
                {"step": 1, "loss": 3.0, "smoothedLoss": 3.0, "lr": 0.0004, "elapsedMs": 100},
                {"step": 2, "loss": math.nan, "smoothedLoss": 2.5, "lr": 0.0003, "elapsedMs": 200},
            ],
            assistant=True,
        )

        assert written == bundle
        assert bundle.exists()
        for artifact in (
            "manifest.json",
            "checkpoint.json",
            "tokenizer.json",
            "weights.bin",
            "training.json",
            "README.txt",
        ):
            assert (out / artifact).exists(), artifact
        with zipfile.ZipFile(bundle) as archive:
            names = set(archive.namelist())
            assert names == {
                "manifest.json",
                "checkpoint.json",
                "tokenizer.json",
                "weights.bin",
                "training.json",
                "README.txt",
            }, names
            manifest_bytes = archive.read("manifest.json")
            checkpoint_bytes = archive.read("checkpoint.json")
            summary_bytes = archive.read("training.json")
            manifest = json.loads(manifest_bytes)
            checkpoint = json.loads(checkpoint_bytes)
            summary = json.loads(summary_bytes)
            weights = archive.read("weights.bin")

        assert manifest["format"] == "only-gates-model-bundle"
        assert manifest["kind"] == "custom"
        assert manifest["target"]["panel"] == "integer-gpt"
        assert manifest["target"]["example"] == "custom"
        assert manifest["target"]["promptFormat"] == "chat"
        assert manifest["model"]["parameterCount"] == params
        assert manifest["model"]["config"] == config
        assert manifest["model"]["tokenizer"]["type"] == "bpe"
        assert manifest["training"]["steps"] == 12
        assert manifest["training"]["bestLoss"] is None
        assert manifest["training"]["resumable"] is False
        assert "stateFile" not in manifest["training"]
        assert manifest["training"]["sourceUrls"] == ["https://example.com/docs/guide.md"]
        assert manifest["training"]["backend"] == "colab"
        assert manifest["integrity"]["algorithm"] == "sha256"
        assert len(manifest["integrity"]["weights"]) == 64
        assert len(weights) == params * 4

        assert checkpoint["format"] == "only-gates-tiny-gpt-custom"
        assert checkpoint["meta"]["training"]["assistant"] is True
        assert checkpoint["prompts"] == ["How do I use this?", "What is it?"]
        assert summary["history"][1]["loss"] is None
        assert summary["training"]["resumable"] is False

        # A non-finite history value must not leak ``NaN`` into the JSON.
        for payload in (manifest_bytes, checkpoint_bytes, summary_bytes):
            text = payload.decode("utf-8")
            assert "NaN" not in text
            assert "Infinity" not in text

    # Wrong weight length is rejected before a corrupt bundle is written.
    with tempfile.TemporaryDirectory() as tmp:
        try:
            model_bundle.write_model_bundle(
                tmp,
                Path(tmp) / "bad.zip",
                config=config,
                tokenizer_model=tokenizer_model,
                weights=b"\x00" * 4,
                name="bad",
                source="bad",
                license="bad",
                prompts=[],
                samples=[],
                training={
                    "steps": 1,
                    "batch": 1,
                    "learningRate": 0.001,
                    "dropout": 0,
                    "seed": 1,
                    "bestStep": 0,
                    "bestLoss": None,
                    "elapsedMs": 1,
                    "corpusHash": "x",
                    "sourceUrls": [],
                },
                history=[],
                assistant=False,
            )
        except ValueError as exc:
            assert "weights.bin" in str(exc)
        else:
            raise AssertionError("expected a weight-size error")

    # The bundled model's known parameter count.
    assert model_bundle.parameter_count(
        {"vocabSize": 4096, "context": 256, "dim": 128, "layers": 4, "heads": 4, "mlp": 512}
    ) == 1_344_640
    print("model bundle tests passed")


if __name__ == "__main__":
    main()
