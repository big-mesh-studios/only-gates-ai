#!/usr/bin/env python3
"""Rebuild notebooks/train_tiny_gpt_luau_colab.ipynb from scratch.

The notebook is generated so cell types cannot drift out of sync with the
contents the way a chain of index-based edits did.
"""

import json
from pathlib import Path


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(True),
    }


cells = [
    md(
        """# Train the Luau task model

This notebook builds the dedicated Luau model used by the **Luau** tab:
short beginner tasks (`Task:` prompts) plus the repository's Luau FAQ
(`Question:` prompts).

Runtime: **GPU (T4 is enough)**. The export uses the same architecture as
Integer GPT, so it plugs into the app's integer-only WASM path. The default
run trains a ~5.2M-parameter model on 12,000 generated rows with the
Luau-focused BPE tokenizer (1,721 tokens, so `local`, `print`, and `for` are
one token each).

Files are written to `MyDrive/tiny-gpt-luau` when Drive is mounted; otherwise
they stay under `/content`.
"""
    ),
    md("## 1. Check the runtime\n"),
    code(
        """import os, sys, subprocess
import torch

print(sys.version)
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
else:
    print("No GPU found. Runtime -> Change runtime type -> T4 GPU.")
    print("Training will still run on CPU, just much more slowly.")
"""
    ),
    md(
        """## 2. Get the repository

Set `REPO_URL` to a clone URL you can reach. If the repository is private,
upload a zip of it to `/content` instead and unzip it into
`/content/only-gates-ai` manually.
"""
    ),
    code(
        """REPO_URL = "https://github.com/clinuxrulz/only-gates-ai.git"
REPO_DIR = "/content/only-gates-ai"

if os.path.isdir(os.path.join(REPO_DIR, ".git")):
    subprocess.run(["git", "-C", REPO_DIR, "pull", "--ff-only"], check=True)
else:
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, REPO_DIR], check=True)

os.chdir(REPO_DIR)
print("working directory:", os.getcwd())
"""
    ),
    md("## 3. Generate and execute-check the task data\n"),
    code(
        """# The JSONL files are committed, so this is optional but keeps the run
# reproducible if the generator changed.
!python3 scripts/luau_tasks.py --out data/luau-tasks --seed 1337

# Luau is required for execution checks and evaluation.
LUAU_ZIP = "/content/luau-ubuntu.zip"
LUAU_DIR = "/content/luau-bin"
subprocess.run([
    "curl", "-L", "-o", LUAU_ZIP,
    "https://github.com/luau-lang/luau/releases/download/0.746/luau-ubuntu.zip",
], check=True)
os.makedirs(LUAU_DIR, exist_ok=True)
subprocess.run(["unzip", "-o", LUAU_ZIP, "-d", LUAU_DIR], check=True)
os.chmod(os.path.join(LUAU_DIR, "luau"), 0o755)
LUAU_BIN = os.path.join(LUAU_DIR, "luau")
print(subprocess.run([LUAU_BIN, "--help"], capture_output=True, text=True).stdout[:200])
"""
    ),
    code(
        """!python3 scripts/check_luau_tasks.py \\
  --tasks data/luau-tasks/train.jsonl \\
  --tasks data/luau-tasks/val.jsonl \\
  --tasks data/luau-tasks/test-seen.jsonl \\
  --tasks data/luau-tasks/test-paraphrase.jsonl \\
  --tasks data/luau-tasks/test-unseen.jsonl \\
  --luau-bin "{LUAU_BIN}" \\
  --show-failures 3
"""
    ),
    md(
        """## 4. Train the Luau tokenizer

The committed tokenizer is already trained on the task corpus. Re-run this
when the dataset changes; the vocabulary size is printed and must be passed to
the trainer as `--vocab`.
"""
    ),
    code(
        """!python3 scripts/train_luau_tokenizer.py \\
  --tasks data/luau-tasks \\
  --out data/luau-tokenizer/tokenizer.json
"""
    ),
    md(
        """## 5. Train

`--max-minutes` stops cleanly before Colab disconnects. Re-run this cell with
`--resume` to continue from `latest.pt`.

Mount Drive first if you want checkpoints to survive a disconnected runtime.
"""
    ),
    code(
        """from google.colab import drive
drive.mount("/content/drive")

OUT_DIR = "/content/drive/MyDrive/tiny-gpt-luau"
os.makedirs(OUT_DIR, exist_ok=True)

!python3 scripts/train_tiny_gpt_luau_colab.py \\
  --tasks data/luau-tasks \\
  --out "{OUT_DIR}" \\
  --tokenizer data/luau-tokenizer/tokenizer.json \\
  --vocab 1721 --context 256 \\
  --dim 256 --layers 6 --heads 8 --mlp 1024 \\
  --epochs 25 --batch 32 --lr 3e-4 \\
  --dropout 0.1 --faq-repeat 10 \\
  --save-every 500 --max-minutes 120 \\
  --export-final
"""
    ),
    md("## 6. Evaluate with real Luau execution\n"),
    code(
        """!python3 scripts/eval_luau_tasks.py \\
  --model-dir "{OUT_DIR}" \\
  --tasks data/luau-tasks/test-seen.jsonl \\
  --tasks data/luau-tasks/test-paraphrase.jsonl \\
  --tasks data/luau-tasks/test-unseen.jsonl \\
  --luau-bin "{LUAU_BIN}" \\
  --pass-k 1 \\
  --out "/content/luau-eval.json" \\
  --show-failures 10
"""
    ),
    md(
        """## 7. Publish

Copy `checkpoint.json`, `tokenizer.json`, and `weights.bin` into
`apps/only-gates-ai/public/models/tiny-gpt-luau/`, commit them, and rebuild the
app. The Luau tab loads that folder lazily.
"""
    ),
    code(
        """import shutil

target = os.path.join(REPO_DIR, "apps/only-gates-ai/public/models/tiny-gpt-luau")
os.makedirs(target, exist_ok=True)
for name in ("checkpoint.json", "tokenizer.json", "weights.bin"):
    source = os.path.join(OUT_DIR, name)
    if os.path.exists(source):
        shutil.copy2(source, target)
        print("copied", source)
    else:
        print("missing", source)

# Convenience download if you are not using Drive.
shutil.make_archive("/content/tiny-gpt-luau", "zip", OUT_DIR)
from google.colab import files
files.download("/content/tiny-gpt-luau.zip")
"""
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

path = (
    Path(__file__).resolve().parent.parent
    / "notebooks"
    / "train_tiny_gpt_luau_colab.ipynb"
)
path.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
print("wrote", path, "cells", len(cells))
