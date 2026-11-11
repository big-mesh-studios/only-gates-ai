# tiny-gpt-solid2 model artifacts

This directory is loaded by the **Integer GPT → Solid 2 Docs (BPE)** example.

Place these files here (produced by the Colab trainer):
- `checkpoint.json`
- `tokenizer.json`
- `weights.bin` (packed float32, `tiny_gpt_create` order)

## Train from scratch (Termux + Colab CLI)

The repo includes a self-contained trainer: `scripts/train_tiny_gpt_solid2_colab.py`.
It fetches pinned Solid 2 docs, includes supplemental Solid 2 HackMD notes (proposal/pattern
context), uses the curated Q&A corpus, and can mix in general-language short stories for
coherence. It trains a byte-level BPE tokenizer + decoder-only transformer (chat-formatted
LM + QA finetune) and exports the artifacts above.

Example (uses your installed `colab` CLI; GPU recommended):
```sh
# from repo root
colab run --gpu T4 --timeout 7200 scripts/train_tiny_gpt_solid2_colab.py -- \
  --out /content/tiny-gpt-solid2

# keep session and download:
# colab run --keep -s solid2train --gpu T4 --timeout 7200 scripts/train_tiny_gpt_solid2_colab.py -- --out /content/tiny-gpt-solid2
# colab download -s solid2train /content/tiny-gpt-solid2/weights.bin apps/only-gates-ai/public/models/tiny-gpt-solid2/weights.bin
# colab download -s solid2train /content/tiny-gpt-solid2/tokenizer.json apps/only-gates-ai/public/models/tiny-gpt-solid2/tokenizer.json
# colab download -s solid2train /content/tiny-gpt-solid2/checkpoint.json apps/only-gates-ai/public/models/tiny-gpt-solid2/checkpoint.json
# colab stop -s solid2train
```

Optional: place `data/solid2/tinystories.txt` (or pass `--stories /content/tinystories.txt`
on Colab) to add general-language stories; pass `--stories-url <url>` to fetch them.

## Resume a run

The trainer checkpoints the model, AdamW optimizer, scheduler, RNG, and QA
cursor to `<out>/latest.pt`. If a free Colab session ends, run the same command
with `--resume`; it continues from the saved optimizer step instead of
restarting. A time budget stops cleanly before the session limit:

```sh
python scripts/train_tiny_gpt_solid2_colab.py \
  --out /content/drive/MyDrive/tiny-gpt-solid2 \
  --lm-steps 5000 --qa-epochs 100 \
  --save-every 100 --max-minutes 300 --resume
```

The trainer verifies the model config, seed, tokenizer, and data fingerprint and
refuses to resume if they changed. Use `--fresh` to start a new run. See
`docs/COMPUTE.md` for the checkpoint fields, reporting protocol, and troubleshooting.

Then copy/replace the three exported files in this folder and reload the web app.
`latest.pt` and `best-qa.pt` are training state and do not need to be copied.

Trainer default (sized to the small corpus): `vocab=4096, ctx=256, dim=48,
layers=3, heads=4, mlp=96` (264,528 params). The bundled checkpoint uses
`vocab=4096, ctx=256, dim=128, layers=4, heads=4, mlp=512` (1,344,640 params)
for better answer quality. The integer-only WASM engine loads any config packed
in this format.
