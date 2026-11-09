# Sharing compute for Integer GPT

This guide is for anyone who wants to donate a training run. A free Google
Colab T4 session is enough to make progress because training is resumable: a
session can stop at a time budget and the next session continues from the same
optimizer step.

The goal is a better integer-friendly tiny GPT, then a gate-lowered one, then a
measured FPGA prototype. Training is the part that still needs floating point
and a lot of arithmetic; inference is the part the project is trying to make
small and deterministic.

## 1. Before you start

You need:

- a GitHub account, so results can be reported in an issue;
- a Google account for Colab;
- optional: a local GPU or Kaggle account for longer runs.

You do **not** need a GPU to help with evaluation, code, documentation, or RTL
review. The smoke test and the float-vs-integer comparison run on a CPU.

## 2. Smoke test first

The smoke test uses a tiny configuration and a couple of steps. It proves the
corpus, tokenizer, model, export, and checkpoint path all work before you spend
session time.

```sh
python scripts/train_tiny_gpt_solid2_colab.py \
  --out /content/tiny-gpt-smoke \
  --smoke --fresh
```

A successful run writes `weights.bin`, `tokenizer.json`, `checkpoint.json`, and
`latest.pt` into the output directory.

## 3. Free Colab workflow

Mount Drive so a disconnect does not lose the checkpoint:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Clone once per session:

```sh
git clone https://github.com/clinuxrulz/only-gates-ai.git
cd only-gates-ai
```

Start a real run with a budget shorter than the session limit:

```sh
python scripts/train_tiny_gpt_solid2_colab.py \
  --out /content/drive/MyDrive/tiny-gpt-solid2 \
  --lm-steps 5000 --qa-epochs 100 \
  --save-every 100 --max-minutes 300 --fresh
```

If the run reaches the budget, it saves and exits with status 0. In the next
session, run the same command with `--resume`:

```sh
python scripts/train_tiny_gpt_solid2_colab.py \
  --out /content/drive/MyDrive/tiny-gpt-solid2 \
  --lm-steps 5000 --qa-epochs 100 \
  --save-every 100 --max-minutes 300 --resume
```

The same `--out` directory, configuration, seed, and data are required. The
trainer refuses to resume if the fingerprint does not match.

### Colab etiquette

- Start with the smoke test.
- Keep one free account; do not work around usage limits.
- Save checkpoints to Drive, not just `/content`.
- Stop a session when you are done; do not leave idle runtimes.
- Follow Colab's terms of service.

Kaggle or a local GPU is a better fit if you need longer uninterrupted runs.

## 4. What resume restores

`latest.pt` contains:

| Field | Why it matters |
| --- | --- |
| `stage` | `lm`, `lm-done`, `qa`, or `done` |
| `model` | model weights |
| `optimizer` | AdamW moment estimates |
| `scheduler` | cosine learning-rate position |
| `rng` | Python, NumPy, PyTorch, and CUDA RNG state |
| `lm_step` / `lm_loss_*` | LM progress and running loss |
| `qa_epoch` / `qa_cursor` / `qa_perm` | exact QA example order |
| `qa_best` / `qa_best_epoch` | best validation checkpoint |
| `config` | full model and dropout configuration |
| `fingerprint` | docs, QA, stories, token stream, tokenizer, seed |
| `tokenizer_model` | the exact tokenizer, embedded |
| `wall_seconds` | compute already spent |
| `resume_count` | how many sessions contributed |

The tokenizer is saved before LM training and reused. It is never retrained on
resume, because a different BPE merge order would renumber every token.

Restoring state gives a correct continuation, not a guarantee of bit-identical
results across different GPUs or PyTorch versions. On the same hardware and
seed the trajectory is very close; across devices the schedule and data order
are the same even if kernel rounding differs.

Useful flags:

```text
--resume                 load <out>/latest.pt
--resume-from PATH       explicit checkpoint
--fresh                  ignore an existing run
--save-every N           checkpoint cadence (default 100 steps)
--max-minutes N          stop cleanly before the session ends
--max-hours N            same, in hours
--stage lm|qa|all        run only one stage
--keep-checkpoints N     rolling snapshots to keep (default 3)
--export-only            rebuild weights.bin from latest.pt
```

## 5. What to report

Open a **Training result** issue with:

- the checkpoint SHA-256:
  ```sh
  sha256sum /content/drive/MyDrive/tiny-gpt-solid2/latest.pt
  ```
- the final `run.json` and `metrics.jsonl`;
- `latest.pt`, `weights.bin`, `tokenizer.json`, and `checkpoint.json` if you can
  share them (a release asset, Hugging Face, or a Drive link with access);
- the command line, seed, configuration, and number of sessions;
- hardware (for example `Tesla T4`), PyTorch/CUDA version, and wall time;
- whether the run resumed, and at which step or epoch;
- the QA best validation loss and a few generated samples.

Do not paste large binaries into an issue. Link them.

## 6. Evaluation protocol

For a float-vs-integer report, use the same prompts and the same checkpoint:

1. Run the float model and the integer path with the same prompt and sampling
   parameters.
2. Record top-1 token match, top-5 overlap, and mean probability delta over a
   fixed prompt set.
3. Save the prompt set and the raw outputs, not just aggregate numbers.
4. Note the device and whether the comparison used the TypeScript int8 path or
   the fully integer WebAssembly path.

For a training result, report held-out QA loss and the exact data fingerprint
from `run.json`. A loss curve without a fingerprint is hard to compare.

## 7. Artifact naming

Suggested convention:

```text
tiny-gpt-solid2-<git-short-sha>-<seed>-<date>/
  latest.pt
  best-qa.pt
  checkpoint.json
  tokenizer.json
  weights.bin
  metrics.jsonl
  run.json
```

Keep `latest.pt` even after export: it is the only file that can resume the run.

## 8. Credit and licensing

Contributors are credited by GitHub handle and run in the release notes when
their checkpoint, evaluation, or RTL is used. State clearly whether a run is a
reproduction, an ablation, or a new configuration.

The repository is MIT licensed. The pinned Solid 2 documentation, supplemental
notes, and any general-language stories keep their own licenses and attribution;
do not relicense them by copying them into a derived artifact without checking.

## 9. Troubleshooting

**“checkpoint config differs”** — you changed the model size, dropout, QA
oversampling, or seed. Resume only works with the original configuration. Use a
new `--out` directory or `--fresh`.

**“checkpoint data mismatch”** — the docs, Q&A, stories, token stream, or
tokenizer changed. Use `--fresh` for a new run, or restore the original data.

**Out of memory** — lower `--batch` or `--context`. The checkpoint fingerprint
includes the model config, so start a new run if you change the model.

**CUDA device count changed** — the trainer warns and skips CUDA RNG restore.
Training still continues; exact randomness is not preserved across a different
GPU topology.

**Drive writes are slow** — checkpoints are written to a temporary file in the
same directory and then renamed. If Drive is unreliable, train with `--out
/content/run` and copy `latest.pt` to Drive periodically.

**The session died without a save** — rerun with `--resume`. Up to
`--save-every` steps of work may be lost; the checkpoint that exists is still
consistent.
