#!/usr/bin/env python3
"""Rebuild notebooks/train_masked_diffusion_luau_colab.ipynb from scratch.

The notebook is generated so its cells stay in sync with the trainer and with
the app's model directory.
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
        """# Train the masked-diffusion Luau model

This notebook builds the model used by the **Masked diffusion** tab. It is a
bidirectional Tiny GPT trained with an absorbing `<|mask|>` token: the whole
answer region is masked, then each denoising step fills the next block of
positions left to right. The browser runs one integer-only WASM pass per
animation frame.

Runtime: **GPU (T4 is enough)**. The default configuration is about 9.6M
parameters with a 128-token context, which fits the generated Luau answers
(median 28 tokens, max 42) and keeps the live animation fast. The committed
dataset has 30,000 training rows and 3,000 validation rows, with six training
prompt styles and three held-out styles for evaluation.

Files are written to `MyDrive/masked-diffusion-luau` when Drive is mounted;
otherwise they stay under `/content`.
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

Set `REPO_URL` to a clone URL you can reach. The masked trainer only needs the
new script, but the notebook also uses the committed tokenizer and task data.
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
    md(
        """## 3. Check the data and tokenizer

The task JSONL files and the 1,722-token Luau BPE model are committed. The
masked trainer appends `<|mask|>` at the end of the vocabulary (id 1,721), so
existing prompt ids do not move. Training uses canonical, `paraphrase_a`,
`paraphrase_c`, `paraphrase_d`, `paraphrase_e`, and `paraphrase_f`; the
`paraphrase_b`, `paraphrase_g`, and `paraphrase_h` splits are held out.
"""
    ),
    code(
        """print(open("data/luau-tasks/train.jsonl", encoding="utf-8").readline()[:200])
print(open("apps/only-gates-ai/public/models/tiny-gpt-luau/tokenizer.json", encoding="utf-8").read()[:200])
"""
    ),
    md(
        """## 4. Train

`--max-minutes` stops cleanly before Colab disconnects. Re-run the same cell
with `--resume` to continue from `latest.pt`.

The trainer exports the best-validation checkpoint by default. It stops early
after `--early-stop-patience` epochs without improvement (default 6; `0`
disables).
"""
    ),
    code(
        """from google.colab import drive
drive.mount("/content/drive")

OUT_DIR = "/content/drive/MyDrive/masked-diffusion-luau"
os.makedirs(OUT_DIR, exist_ok=True)

!python3 scripts/train_masked_diffusion_luau_colab.py \\
  --tasks data/luau-tasks \\
  --out "{OUT_DIR}" \\
  --tokenizer apps/only-gates-ai/public/models/tiny-gpt-luau/tokenizer.json \\
  --context 128 --dim 384 --layers 6 --heads 6 --mlp 1152 \\
  --epochs 40 --batch 24 --lr 3e-4 \\
  --dropout 0.1 --faq-repeat 20 --steps 16 \\
  --save-every 250 --max-minutes 180
"""
    ),
    md(
        """## 5. Inspect a few denoised samples

The checkpoint stores samples produced by a PyTorch mirror of the C denoiser:
start with a fully masked answer region and fill the next left-to-right block
on each step.
"""
    ),
    code(
        """import json, os

checkpoint = json.load(open(os.path.join(OUT_DIR, "checkpoint.json"), encoding="utf-8"))
print("config:", checkpoint["config"])
print("parameters:", f"{checkpoint['parameterCount']:,}")
print("mask id:", checkpoint["diffusion"]["maskTokenId"])
for sample in checkpoint.get("samples", []):
    print("\\nPROMPT:", sample["prompt"])
    print(sample["text"][:600])
"""
    ),
    md(
        """## 6. Publish

Copy `checkpoint.json`, `tokenizer.json`, and `weights.bin` into
`apps/only-gates-ai/public/models/masked-diffusion-luau/`, commit them, and
rebuild the app. The Masked diffusion tab loads that folder lazily.
"""
    ),
    code(
        """import shutil

target = os.path.join(REPO_DIR, "apps/only-gates-ai/public/models/masked-diffusion-luau")
os.makedirs(target, exist_ok=True)
for name in ("checkpoint.json", "tokenizer.json", "weights.bin"):
    source = os.path.join(OUT_DIR, name)
    if os.path.exists(source):
        shutil.copy2(source, target)
        print("copied", source)
    else:
        print("missing", source)

# Convenience download if you are not using Drive.
shutil.make_archive("/content/masked-diffusion-luau", "zip", OUT_DIR)
from google.colab import files
files.download("/content/masked-diffusion-luau.zip")
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
    / "train_masked_diffusion_luau_colab.ipynb"
)
path.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
print("wrote", path, "cells", len(cells))
