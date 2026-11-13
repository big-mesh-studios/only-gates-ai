# tiny-gpt-luau model artifacts

This folder is served by the **Luau** tab. It needs three files:

| File | Source |
| --- | --- |
| `checkpoint.json` | `scripts/train_tiny_gpt_luau_colab.py` export |
| `tokenizer.json` | copied from `data/luau-tokenizer/tokenizer.json` |
| `weights.bin` | packed float32 weights from the export |

The app quantizes `weights.bin` to int8 for integer-only WASM inference, so the
files here are the float checkpoints, not browser-ready quantized weights.

The checkpoint carries the architecture (`vocabSize`, `context`, `dim`,
`layers`, `heads`, `mlp`), so `apps/only-gates-ai/src/lib/luauConfig.ts` only
needs the same values as the pre-load fallback. The current run uses:

```jsonc
{ "vocabSize": 1721, "context": 256, "dim": 256, "layers": 6, "heads": 8, "mlp": 1024 }
```

The committed bundle is the run-6 checkpoint: 5,228,032 parameters
(`weights.bin` is 20,912,128 bytes of float32). Execution-evaluated pass@1 on
the 800-row held-out suite is 84.38% overall, split as:

| Split | Run 4 | Run 6 |
| --- | --- | --- |
| `test-seen` (canonical prompts) | 99.67% | 100.00% |
| `test-question` (question prompts) | 94.50% | 99.50% |
| `test-unseen` (held-out identifiers) | 64.00% | 64.00% |
| `test-paraphrase` (held-out wording) | 59.50% | 56.00% |
| all rows | 83.88% | **84.38%** |

Run 6 keeps the run-4 architecture and tokenizer. It reallocates the 12,000
training rows toward families that failed held-out paraphrases (405–809 rows
per family instead of 600), and exports the best-validation checkpoint
(0.1446). Test splits stay equal per family, so scores remain comparable.
Syntax and stop-token rates are 100%, so failures are wrong answers, not
invalid Luau.

## Produce the files

```sh
# 1. Generate the task data (committed, but deterministic to regenerate)
python3 scripts/luau_tasks.py --out data/luau-tasks --seed 1337

# 2. Train the Luau-focused tokenizer (committed under data/luau-tokenizer)
python3 scripts/train_luau_tokenizer.py \
  --tasks data/luau-tasks \
  --out data/luau-tokenizer/tokenizer.json

# 3. Train on Colab (or any CUDA machine) and export
python3 scripts/train_tiny_gpt_luau_colab.py \
  --tasks data/luau-tasks \
  --out /content/drive/MyDrive/tiny-gpt-luau \
  --tokenizer data/luau-tokenizer/tokenizer.json \
  --vocab 1721 --context 256 \
  --dim 256 --layers 6 --heads 8 --mlp 1024 \
  --epochs 25 --batch 32 --lr 3e-4 \
  --dropout 0.1 --faq-repeat 10 \
  --save-every 500 --max-minutes 120

# 4. Copy checkpoint.json, tokenizer.json, and weights.bin here
cp /content/drive/MyDrive/tiny-gpt-luau/checkpoint.json .
cp /content/drive/MyDrive/tiny-gpt-luau/tokenizer.json .
cp /content/drive/MyDrive/tiny-gpt-luau/weights.bin .
```

`notebooks/train_tiny_gpt_luau_colab.ipynb` runs the same flow end to end,
including Luau execution checks and evaluation. `scripts/luau_tasks.py`
weights train and validation rows toward weak families; pass
`--equal-families` to regenerate the older equal allocation.

## Evaluate a checkpoint

```sh
python3 scripts/eval_luau_tasks.py \
  --model-dir apps/only-gates-ai/public/models/tiny-gpt-luau \
  --tasks data/luau-tasks/test-seen.jsonl \
  --tasks data/luau-tasks/test-paraphrase.jsonl \
  --tasks data/luau-tasks/test-question.jsonl \
  --tasks data/luau-tasks/test-unseen.jsonl \
  --luau-bin /path/to/luau
```

The report includes pass@1, pass@k, syntax rate, stop rate, a per-split
breakdown (canonical, question, held-out wording, unseen identifiers), and
first-token failure samples. Generated code is never executed in the browser.