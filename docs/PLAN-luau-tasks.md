# Plan: Luau task model + dedicated Luau tab

Status: **draft, agreed direction, no implementation started.**
Scope: take the current Luau Q&A Integer GPT from "answers questions about
Luau" to "writes short Luau snippets for beginner tasks", and ship it as its own
tab in the web app.

## Decisions from the project owner

1. **Training target:** Google Colab is the main training environment. Local
   hardware is only for data generation, evaluation smoke tests, and app work.
2. **Product shape:** Luau gets its **own tab** and its **own model bundle**,
   separate from the Solid 2 model and the generic Custom Docs flow.
3. **Process:** write this plan to Markdown before implementation begins.

## 1. Objective

Build a small, integer-only Luau assistant that can do two things:

- **Ask mode:** answer Luau language questions in prose, with short examples
  (the behaviour that already works today).
- **Write mode:** turn a concrete beginner task into a short, runnable Luau
  snippet, copying variable names and numbers from the prompt.

Example Write tasks:

- "Write a for loop that sums the first 50 elements of the array `values`."
- "Print the numbers 1 to 20."
- "Count how many entries in `scores` are above 80."
- "Write a local function that adds two numbers."
- "Loop over a table and print each index and value."

The model should be useful as a **Luau pattern demonstrator** for a learner.
It is not expected to be a general programmer.

## 2. Non-goals

- General-purpose code generation, multi-file programs, or debugging.
- Arbitrary algorithm design from a natural-language description.
- Executing generated code in the browser. Evaluation runs code on Colab/local
  only, in a timeout-bounded sandbox.
- Replacing the Solid 2 model or the generic Custom Docs trainer.
- Adding new special tokens to the deployed tokenizer in the first version.

## 3. What already exists

| Piece | Location | Notes |
| --- | --- | --- |
| Current Luau Q&A data | `apps/only-gates-ai/src/assets/luau-faq.md` | ~25 concepts, near-question wording |
| Custom-docs Colab trainer | `scripts/train_tiny_gpt_custom_colab.py` | Fetches docs, builds Q&A from Markdown headings, resumable |
| Base model/trainer | `scripts/train_tiny_gpt_solid2_colab.py` | `TinyGPT`, BPE tokenizer, masked QA loss in `make_qa_examples` |
| Bundled model | `apps/only-gates-ai/public/models/tiny-gpt-solid2/` | `vocab=4096, ctx=256, dim=128, layers=4, heads=4, mlp=512`, 1,344,640 params |
| Integer engine | `src/tiny_gpt_infer.c`, `packages/only-gates-ai/src/tiny-gpt.ts` | Loads any packed config; integer inference only |
| App tabs | `apps/only-gates-ai/src/App.tsx` | Currently Symbol classifier, Mini language model, Integer GPT |
| Integer GPT panel | `apps/only-gates-ai/src/components/IntegerGptPanel.tsx` | Alice, Solid 2, and custom model UI |
| WASM model loader | `apps/only-gates-ai/src/lib/tinyGptSolid2.ts` | `loadSolid2Wasm(baseUrl)` is already parameterized |

Important: the base trainer already has `<|user|>`, `<|assistant|>`,
`<|end|>` tokens and masked answer loss. The custom-docs trainer does not mask
prompt loss. The real missing pieces are **task data**, **execution evaluation**,
and **a dedicated app surface**.

## 4. Product design: the Luau tab

### 4.1 Tab layout

Add a fourth top-level tab:

```text
Symbol classifier | Mini language model | Integer GPT | Luau
```

The Luau tab is a focused code assistant, not a copy of the Integer GPT panel.
It contains:

- **Mode toggle:** `Ask` / `Write code`.
- **Prompt input:** a textarea with a "Generate" button.
- **Example chips:** 6-10 preset beginner tasks.
- **Output:** streamed code or prose, rendered as text with a copy button.
- **Model status:** loaded model name, parameter count, context, integer lane
  (SIMD/scalar), tokens/sec.
- **Honesty banner:** "Trained on generated beginner Luau tasks. Check code
  before using it."
- **Training link:** "How this model was trained" and a Colab notebook link.

Do **not** add a Run button in the browser in v1. Running untrusted generated
code is a separate, larger project.

### 4.2 One model, two modes

Use one Luau bundle with a plain-text mode prefix in the user message:

```text
<|user|>Question: How do I write a for loop in Luau?<|assistant|>...
<|user|>Task: Write a for loop that sums the first 50 elements of values.<|assistant|>...
```

The prefix is ordinary text, not a new special token. This avoids tokenizer and
embedding changes while giving the model a clean signal for the expected answer
shape.

- `Question:` answers: short prose explanation plus a small code block.
- `Task:` answers: code only, canonical formatting, then `<|end|>`.

The app adds the prefix automatically based on the selected mode. The training
data contains the same prefixes.

### 4.3 App behaviour

- Lazy-load the Luau model when the tab is selected, similar to the Solid 2
  model load in `IntegerGptPanel`.
- Prompt template: `<|user|>{prefix}{prompt}<|assistant|>`.
- Stop generation on `<|end|>`.
- Defaults for Write mode: `temperature=0.15`, `top_k=40`,
  `repetition_penalty=1.05`, `max_new_tokens=160`.
- Defaults for Ask mode: `temperature=0.3`, `top_k=40`, `max_new_tokens=200`.
- If the model directory is missing, show a clear message with the training
  command instead of a broken tab.
- Reuse the existing styles (`card`, `chat`, code block styles) and add only the
  minimum Luau-specific CSS.

## 5. Data plan

### 5.1 Why a new dataset is required

The current Luau FAQ is small enough for a 1.3M-parameter model to memorise.
Asking "sum the first 50 elements" is a different task: the model must select a
loop pattern and copy values from the prompt. That requires many examples of
the same pattern with varied names, bounds, and wording.

### 5.2 Schema

One JSON object per line in `data/luau-tasks/*.jsonl`:

```json
{
  "id": "sum_first_n/00042",
  "family": "sum_first_n",
  "mode": "write",
  "prompt_style": "canonical",
  "instruction": "Write a for loop that sums the first 50 elements of the array `values` and prints the result. The array has at least 50 elements.",
  "code": "local sum = 0\nfor i = 1, 50 do\n    sum += values[i]\nend\nprint(sum)",
  "harness": {
    "prelude": "local values = {1, 2, 3, ...}",
    "check": "assert(sum == 1275)"
  },
  "split": "train"
}
```

- `harness` is used only by the evaluation script. It is never shown to the
  model.
- For print-based tasks, `check` compares stdout instead of a variable.
- For function tasks, `check` calls the generated function and asserts a return
  value.
- `prompt_style` distinguishes `canonical` from `paraphrase`.

### 5.3 Task families (first release)

Aim for 20 families, balanced at roughly 1,000-2,000 examples each.

| Family | What varies |
| --- | --- |
| `for_print_range` | start, finish, step, variable name |
| `for_sum_all` | array name, element values |
| `for_sum_first_n` | N, array name, element values |
| `for_count_above` | threshold, array name, comparison |
| `for_find_max` | array name, element values |
| `for_find_min` | array name, element values |
| `for_product` | array name, range |
| `while_accumulate` | limit, step, accumulator name |
| `repeat_until` | limit, count |
| `iterate_print` | table name, key/value names |
| `iterate_filter` | table name, threshold, output table |
| `if_elseif_grade` | score, grade thresholds |
| `if_else_health` | health value, threshold |
| `function_add` | parameter names, return type |
| `function_typed` | parameter types, return type |
| `table_literal` | field names, values |
| `table_access` | field name, new value |
| `string_concat` | variable names, separator |
| `string_interp` | variable names, message text |
| `nested_loop` | outer/inner bounds, print text |

Each family needs 3-6 instruction paraphrases, for example:

- canonical: "Write a for loop that sums the first 50 elements of `values`."
- paraphrase: "Show Luau code that adds up the first 50 entries in `values`."
- paraphrase: "How can I total the first 50 values of the `values` table?"

### 5.4 Generation rules

- Deterministic, seeded Python generator: `scripts/luau_tasks.py`.
- Canonical formatting: four spaces, `local` variables, one statement per line,
  `end` on its own line.
- Answers must fit the 256-token context with room for the prompt.
- Reject duplicates by a hash of `(instruction, code)`.
- Split by **generator seed and identifier pool**, not random rows, so the test
  set contains unseen names and numbers.
- Verify every generated answer by running it. A code generator that teaches
  wrong code is worse than too little data.
- Keep the first release code-only for Write mode. Comments and explanations
  are a later experiment.

### 5.5 Mixed Ask/Write data

The same bundle should still answer questions. Mix the datasets:

- ~70% Write tasks.
- ~30% Ask examples, derived from `luau-faq.md` and the Luau docs.
- Prefix all Ask prompts with `Question:` and all Write prompts with `Task:`.
- Keep a 40-question FAQ regression set to detect forgetting.

If mixing visibly degrades either mode, split into two bundles later. Do not
start with two models; one bundle is simpler for the tab and the user.

### 5.6 Target sizes

| Split | Rows |
| --- | ---: |
| `train.jsonl` | 20,000-40,000 |
| `val.jsonl` | 1,000-2,000 |
| `test-seen.jsonl` | 300-500 |
| `test-paraphrase.jsonl` | 200-300 |
| `test-unseen.jsonl` | 100-200 |
| `faq-regression.jsonl` | 40 |

`test-unseen` means held-out combinations or identifiers. A truly unseen task
family will score near zero on a 1.3M model; use it to measure the ceiling, not
as the release gate.

## 6. Tokenizer decision

### Current state

The deployed 4096-token BPE tokenizer was trained mostly on Solid 2 docs.
A five-line Luau sum loop tokenizes to 34 tokens. Common pieces fragment:

- `local` -> 3 tokens
- `print` -> 2 tokens
- `numbers` -> 4 tokens
- `sum` -> 2 tokens

It is workable for short snippets but wastes context and capacity.

### Decision gate

Write `scripts/audit_luau_tokenizer.py` (or a `--audit-tokenizer` mode in the
task generator). Report:

- average tokens per answer, by family
- how often answers exceed 80 / 120 / 160 tokens
- the most fragmented identifiers
- prompt token counts

**Use the existing tokenizer for v1 if** answers average under ~80 tokens and
fewer than ~1% of examples truncate. This keeps warm-starting from the current
Luau checkpoint possible and avoids a new model artifact.

**Retrain a Luau BPE if** answers routinely truncate or code fragments badly.
Train a 4096 or 8192 vocabulary on a mix of Luau docs, generated code, and task
instructions (so English prompts also tokenize well). This requires a new model
trained from scratch or with resized embeddings; treat it as a v2 project, not
part of the first slice.

## 7. Training plan

### 7.1 Model size ladder

| Config | Params | When |
| --- | ---: | --- |
| `vocab=4096, ctx=256, dim=128, layers=4, heads=4, mlp=512` | 1,344,640 | First run; warm-start from the current Luau model |
| `vocab=4096, ctx=256, dim=192, layers=6, heads=6, mlp=768` | ~3,492,000 | If the first run plateaus below target |
| `vocab=4096, ctx=256, dim=256, layers=6, heads=8, mlp=1024` | ~5,836,000 | Only if data and tokenizer are already good |

Keep `context=256` for v1. Move to 384 only if answers truncate. The integer
engine and bundle format support any of these configs; larger models make
browser inference slower, so measure on a phone before committing.

### 7.2 Stage A: continued language modelling (optional)

Warm-start from the best current Luau checkpoint. If no Luau checkpoint exists,
warm-start from the bundled Solid 2 model.

- Data: Luau docs + generated code corpus (unmasked)
- Steps: 2,000-5,000
- LR: `3e-4`
- Batch: 8
- Purpose: adapt to Luau syntax and formatting

If the current Luau Q&A model is already available, try skipping Stage A first.
It may be unnecessary.

### 7.3 Stage B: supervised fine-tuning on masked Ask/Write pairs

Pack each example as:

```text
[USER] + tokens(prefix + instruction) + [ASSISTANT] + tokens(answer) + [END]
```

Compute loss only on answer tokens and the final `[END]`. Reuse the masking
logic from `make_qa_examples` in
`scripts/train_tiny_gpt_solid2_colab.py`.

Starting hyperparameters for the 1.34M model:

| Setting | Value |
| --- | --- |
| LR | `1e-4` |
| Batch | `8-16` |
| Epochs | `10-20` |
| Dropout | `0.1` |
| Warmup | 5% of steps |
| Schedule | cosine decay |
| Gradient clip | `1.0` |
| Validation | every epoch, masked loss |
| Task pass@1 probe | every 5 epochs on 100 tasks |

Export the **best-validation** weights by default. Run 3 showed the opposite of
the earlier assumption: its best val loss was `0.1674` at epoch 10, but the
bundle exported the final epoch at `0.1949`. Keep `--export-final` as an
explicit flag for experiments, and compare both when in doubt. Training also
stops early after `--early-stop-patience` epochs without validation
improvement (default 4; `0` disables).

### 7.4 Optional Stage C: rejection sampling

Only after a solid SFT baseline:

1. Sample 4-8 candidates per training prompt at `T=0.7`.
2. Execute each candidate with `luau`.
3. Keep passing candidates.
4. Fine-tune 1-2 epochs on the verified set.

This is the useful part of `task01.md`'s Phase 4. Skip DPO and RLAIF for now;
they add complexity without evidence at this scale.

### 7.5 Resumability

Colab sessions are short. The new trainer must checkpoint:

- model, optimizer, scheduler, RNG
- step, epoch, example cursor and permutation
- config, data fingerprint, tokenizer hash, seed
- validation history

Follow the existing `latest.pt` + atomic replace pattern from
`train_tiny_gpt_custom_colab.py`. Support:

```text
--resume
--fresh
--save-every N
--max-minutes N
--export-only
```

Exit 0 on a time-budget stop and print the exact resume command.

## 8. Evaluation plan

### 8.1 Execution harness

`scripts/eval_luau_tasks.py`:

1. Load the trained float checkpoint and tokenizer.
2. Build prompts exactly as in training.
3. Generate with low temperature and stop on `<|end|>`.
4. Write a temporary Luau file per task:

   ```lua
   -- harness.prelude
   -- generated code
   -- harness.check
   ```

5. Run with the Luau CLI in a temporary directory, with a short timeout.
6. Compare stdout or assert result.
7. Record the generated code, exit status, and error.

Install Luau on Colab by pinning the current Linux release from the
`luau-lang/luau` GitHub releases page, or build `Luau.Repl.CLI` and
`Luau.Analyze.CLI` from source. Pin the version in the notebook and evaluation
reports.

### 8.2 Metrics

| Metric | Meaning |
| --- | --- |
| `syntax_pass` | `luau-analyze` accepts the snippet |
| `exec_pass@1` | the task's check passes on the first sample |
| `exec_pass@5` | at least one of five samples passes |
| `stop_rate` | generation ended with `<|end|>` |
| `truncation_rate` | output hit `max_new_tokens` |
| `avg_tokens` | average generated length |
| `faq_regression` | 40-question Ask-mode check |

Report results by family, by `prompt_style`, and by split
(`seen`, `paraphrase`, `unseen`).

### 8.3 Error analysis categories

- wrong loop bounds
- off-by-one index
- missing `end`
- hallucinated variable name
- wrong operator
- wrong table iteration form
- no stop token
- output truncation
- prose mixed into Write output

Fix data distribution based on these categories before adding training steps.

### 8.4 Integer-path verification

The app runs int8/Q16 inference, while training and most evaluation use float.
After float evaluation passes, verify the deployed path:

- Build `examples/integer-gpt` on Colab.
- Encode 20-30 prompts with Python, pass token IDs to the CLI, decode the
  output, and run the same execution checks.
- Compare pass rates against the float model. Quantization will change the
  distribution; the goal is to confirm there is no catastrophic drop.

## 9. Training script plan

Add `scripts/train_tiny_gpt_luau_colab.py`.

It should import the existing `TinyGPT`, `BpeTokenizer`, `pack_weights`,
`verify_packed`, and checkpoint helpers from
`scripts/train_tiny_gpt_solid2_colab.py`, and add:

- `load_jsonl_tasks(path)`
- `pack_sft_example(tokenizer, prompt, answer, context, prefix)`
- `build_sft_examples(tasks, tokenizer, context)`
- `eval_masked_loss(model, examples, batch, device)`
- `train_sft(...)`
- `export_bundle(...)` using `scripts/model_bundle.py`
- `--smoke` mode for a CPU-only one-step test

Do not refactor the existing trainers in the first slice. A small amount of
duplication is preferable to breaking the Solid 2 and custom-docs paths.

CLI sketch:

```bash
python scripts/train_tiny_gpt_luau_colab.py \
  --tasks data/luau-tasks \
  --out /content/drive/MyDrive/tiny-gpt-luau \
  --init-from /content/drive/MyDrive/tiny-gpt-luau-faq/latest.pt \
  --epochs 15 --batch 8 --lr 1e-4 \
  --save-every 200 --max-minutes 300
```

Outputs:

```text
<out>/latest.pt
<out>/tokenizer.json
<out>/checkpoint.json
<out>/weights.bin
<out>/only-gates-luau-bundle.zip
<out>/metrics.jsonl
<out>/eval-report.json
```

## 10. App implementation plan

### 10.1 Files to add

| File | Purpose |
| --- | --- |
| `apps/only-gates-ai/src/lib/luauConfig.ts` | Model base URL, default config, task presets, parameter count |
| `apps/only-gates-ai/src/components/LuauPanel.tsx` | The Luau tab UI |
| `apps/only-gates-ai/public/models/tiny-gpt-luau/README.md` | Artifact and training notes |
| `apps/only-gates-ai/public/models/tiny-gpt-luau/checkpoint.json` | Exported checkpoint |
| `apps/only-gates-ai/public/models/tiny-gpt-luau/tokenizer.json` | Exported tokenizer |
| `apps/only-gates-ai/public/models/tiny-gpt-luau/weights.bin` | Packed float32 weights |
| `notebooks/train_tiny_gpt_luau_colab.ipynb` | One-click Colab flow |

### 10.2 Files to change

| File | Change |
| --- | --- |
| `apps/only-gates-ai/src/App.tsx` | Add the `luau` tab and render `LuauPanel` |
| `apps/only-gates-ai/public/models/tiny-gpt-luau/.gitignore` | Ignore resumable training state, keep model artifacts tracked |
| `apps/only-gates-ai/src/styles.css` | Minimal Luau panel styles if existing classes are insufficient |
| `README.md` or model README | Document the new tab and training flow |

### 10.3 Reuse

- `loadSolid2Wasm(baseUrl)` for loading the bundle.
- `Solid2Decoder` with `promptFormat: "chat"`.
- `decodeGenerated`, `isStopToken`, `encodeChatPrompt` from `tinyGptBpe.ts`.
- The streaming timer pattern from `IntegerGptPanel.tsx`.
- The existing code block and copy-button styles where possible.

### 10.4 Accessibility and UX

- Tab uses `role="tab"` with `aria-selected`.
- Mode toggle uses `role="tablist"` or a fieldset with clear labels.
- Output code is selectable; copy button announces success.
- Preset chips are keyboard accessible.
- Loading and error states are explicit.
- On a missing model, link to the Colab notebook rather than failing silently.

## 11. Milestones

| Milestone | Deliverable | Exit criteria |
| --- | --- | --- |
| M0 | This plan | Approved |
| M1 | Task generator + eval harness | 5 families, 300 executable eval tasks, current-model baseline recorded |
| M2 | First 1.34M SFT run | Model beats baseline; syntax >= 85%, stop >= 95% on seen tasks |
| M3 | Luau tab | Tab loads model, streams Ask/Write output, copy works, existing tabs unaffected |
| M4 | Full first release | 20 families, 20k+ pairs, pass@1 >= 60% seen / >= 35% paraphrase, FAQ regression >= 80% |
| M5 | Optional scale/rejection | Only if M4 plateaus; evaluate 3.5M config or tokenizer retrain |

### First slice (recommended)

Do this before the full 20-family build:

1. Pick 5 families: `for_print_range`, `for_sum_first_n`, `for_count_above`,
   `if_elseif_grade`, `function_add`.
2. Generate ~3,000 training pairs and 300 eval tasks.
3. Train the 1.34M model for 10-15 epochs, warm-starting from the current Luau
   checkpoint if available.
4. Run execution evaluation.
5. Decide: more data, bigger model, or tokenizer work.

This gives a real signal in roughly 2-3 days instead of committing to a full
build first.

## 12. Acceptance criteria

### Data

- Generator is deterministic for a given seed.
- No duplicate `(instruction, code)` pairs.
- Every `test-*` row executes successfully with its harness before training.
- Training/validation/test splits share no generator seed or identifier pool.
- At least 20 families and 20,000 training rows for the first full release.

### Model

- `syntax_pass >= 90%` on `test-seen`.
- `exec_pass@1 >= 60%` on `test-seen`.
- `exec_pass@1 >= 35%` on `test-paraphrase`.
- `stop_rate >= 95%`.
- `truncation_rate <= 2%`.
- FAQ regression set scores at least 80% by the chosen rubric.
- Integer-path pass rate is within a small margin of the float path.

### App

- Luau tab loads the model and generates output on desktop and mobile.
- Write mode emits code only; Ask mode emits prose plus code.
- `<|end|>` stops generation; trailing special tokens are stripped.
- Copy button works and announces success.
- Missing-model state is actionable.
- `pnpm typecheck`, app tests, and the existing build still pass.
- No changes to the Solid 2 or Alice behaviour.

## 13. Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| 1.34M model cannot generalise enough | Start narrow; scale to ~3.5M only after data/eval are solid |
| Tokenizer fragments Luau | Audit first; retrain BPE as a separate v2 project |
| Train/test leakage | Split by seed and identifier pool; dedupe |
| Ask/Write modes blur | Plain-text prefixes; balanced data; per-mode eval |
| Overfitting the small dataset | Dropout, early stopping, dedupe, held-out paraphrases |
| Colab session limits | Resumable checkpoints, `--max-minutes`, `--save-every` |
| Quantization drops quality | Integer-path verification on 20-30 prompts |
| Generated code runs forever | Timeout every eval process; no in-app execution |
| Model bundle size | 1.34M float32 weights ~5.4 MB; ~14 MB at 3.5M; ~23 MB at 5.8M |
| App regression | Keep Luau code isolated; reuse existing loaders; run existing tests |

## 14. Commands

### Generate data

```bash
python3 scripts/luau_tasks.py \
  --out data/luau-tasks \
  --train 30000 --val 1500 \
  --seed 1337
```

### Audit tokenization

```bash
python3 scripts/audit_luau_tokenizer.py \
  --tokenizer apps/only-gates-ai/public/models/tiny-gpt-solid2/tokenizer.json \
  --tasks data/luau-tasks/train.jsonl
```

### Train on Colab

```bash
python scripts/train_tiny_gpt_luau_colab.py \
  --tasks data/luau-tasks \
  --out /content/drive/MyDrive/tiny-gpt-luau \
  --init-from /content/drive/MyDrive/tiny-gpt-luau-faq/latest.pt \
  --epochs 15 --batch 8 --lr 1e-4 \
  --save-every 200 --max-minutes 300
```

### Evaluate

```bash
python scripts/eval_luau_tasks.py \
  --model-dir /content/drive/MyDrive/tiny-gpt-luau \
  --tasks data/luau-tasks/test-seen.jsonl \
  --luau-bin /usr/local/bin/luau \
  --pass-k 5
```

### Publish artifacts

Copy the three inference files into the app:

```bash
cp /content/drive/MyDrive/tiny-gpt-luau/{checkpoint.json,tokenizer.json,weights.bin} \
  apps/only-gates-ai/public/models/tiny-gpt-luau/
```

Then rebuild the site and verify the Luau tab.

## 15. Review of `task01.md`

Keep:

- supervised fine-tuning on instruction/response pairs
- prompt loss masking
- lower learning rate than pretraining
- execution-guided rejection sampling

Drop or defer:

- adding `<|system|>` and resizing embeddings: the chat tokens already exist,
  and resizing invalidates the current checkpoint
- PyTorch-specific `-100` instructions: the base trainer already implements
  masked QA loss
- DPO and RLAIF: premature at this scale
- mixing Solid 2 docs into the Luau task model: irrelevant data and a capacity
  tax
- "5,000 examples" as the goal: diversity, execution checks, and held-out
  splits matter more than raw count
- treating the task as generic code generation: the model is a narrow pattern
  demonstrator

## 16. Open questions

1. Should the Luau tab expose a temperature/top-k control, or keep decoding
   fixed for reliability? Default plan: advanced controls hidden behind a
   details/summary element.
2. Should Ask mode answers include code blocks, or prose only? Default plan:
   prose plus one short code block, matching the current FAQ style.
3. Should the first release use the current Luau checkpoint for warm-start, or
   start from the bundled Solid 2 model? Default plan: use the best existing
   Luau checkpoint if it is available; otherwise Solid 2 plus Stage A.
4. Should the generated task data be committed to the repository? Default plan:
   commit the generator, a small sample, and the schema; generate the full
   dataset in Colab to keep the repo small.

---

## Implementation status

M1-M3 are implemented and a trained bundle is ready to publish:

| Piece | Path |
| --- | --- |
| Deterministic task generator | `scripts/luau_tasks.py` |
| Generated dataset (20 families, 12,000 train / 1,200 val / 800 eval) | `data/luau-tasks/` |
| Luau-focused BPE tokenizer (1,721 tokens) | `data/luau-tokenizer/tokenizer.json` |
| Tokenizer training | `scripts/train_luau_tokenizer.py` |
| Tokenizer audit | `scripts/audit_luau_tokenizer.py` |
| Luau execution helpers | `scripts/luau_exec.py` |
| Dataset execution check | `scripts/check_luau_tasks.py` |
| Execution-based evaluation | `scripts/eval_luau_tasks.py` |
| Torch-free SFT packing | `scripts/luau_sft.py` |
| Colab/GPU trainer and bundle export | `scripts/train_tiny_gpt_luau_colab.py` |
| Generator tests | `scripts/test_luau_tasks.py` |
| Packing tests | `scripts/test_luau_sft.py` |
| Colab notebook builder | `scripts/build_luau_notebook.py` |
| Colab notebook | `notebooks/train_tiny_gpt_luau_colab.ipynb` |
| Model artifact folder | `apps/only-gates-ai/public/models/tiny-gpt-luau/` |
| Tab config and UI | `apps/only-gates-ai/src/lib/luauConfig.ts`, `apps/only-gates-ai/src/components/LuauPanel.tsx` |

The committed slice is 12,000 training rows (600 per family), with 1,200
validation rows and an 800-row held-out suite split into canonical
(`test-seen`), held-out wording (`test-paraphrase`), question wording with
unseen values (`test-question`), and held-out identifiers (`test-unseen`). The
generator raises an error instead of silently duplicating rows when a family
saturates, and the low-diversity families were widened
(`for_print_range`, `repeat_until`, `function_typed`, `string_interp`,
`nested_loop`) to make the 600-per-family target possible.

Prompt styles matter more than expected. Training now covers four styles:

| Style | Split | Shape |
| --- | --- | --- |
| `canonical` | train + test-seen | "Write a for loop in Luau that runs from 1 to 5 and prints every number." |
| `paraphrase_a` | train | "Show Luau code that counts from 1 to 5 and prints each value." |
| `paraphrase_c` | train | imperative "Write Luau code that ..." |
| `paraphrase_d` | train | question "How do I ... in Luau?" |
| `paraphrase_b` | held out | "What Luau loop prints the values from 1 to 5?" |

Validation performed on this slice:

- All 14,000 generated snippets execute and produce the expected stdout with
  Luau 0.746.
- `python3 -m unittest scripts/test_luau_tasks.py scripts/test_luau_sft.py`
  passes (including the optional execution test when `LUAU_BIN` is set).
- No prompt plus answer exceeds the 256-token context (worst family p95 is 94
  tokens for `iterate_filter`).
- The Luau tokenizer keeps the app special tokens and collapses common
  keywords to one token each (`local`, `print`, `for`, `elseif`, `continue`,
  `function`).

## Training runs

| Run | Tokenizer | Rows | Parameters | Epochs | Held-out pass@1 |
| --- | --- | --- | --- | --- | --- |
| 1 | Solid 2 base, 4,096 | 3,000 | 1,344,640 | 20 | 30.67% |
| 2 | Luau BPE, 1,695 | 6,000 | 3,031,296 | 30 | 42.50% |
| 3 | Luau BPE, 1,721 | 12,000 | 5,228,032 | 25 | 72.88% |
| 4 | Luau BPE, 1,721 | 12,000 | 5,228,032 | 13 (early stop) | **83.88%** |

Run 2 details (`dim=192, layers=6, heads=6, mlp=768`):

| Split | pass@1 |
| --- | --- |
| `test-seen` | 70.67% |
| `test-unseen` | 43.00% |
| `test-paraphrase` | 0.00% |
| all 600 | 42.50% |

Syntax and stop-token rates were both 100%, so the failure mode was wrong
constants and cross-family drift, not malformed code. A probe showed that the
model answered "Write Luau code that loops from 6 to 1 with a step of -1 and
prints every number." correctly but answered "How do I loop from 6 to 1 with a
step of -1 in Luau and print every number?" with `for index = 1, 0, -1 do`.
That is why run 3 trains on both an imperative and a question style while
keeping `paraphrase_b` held out, and why three generator bugs were fixed
first:

- `function_add` no longer draws function names from the accumulator pool
  (`sum`, `total`, `accumulate`), which taught the model to emit
  `local function sum += ...`.
- `table_access` no longer emits an unmentioned `name = "Ada"` field.
- `iterate_filter` no longer reuses the `matches` name from the count family
  and now says "prints the size of `filtered`" instead of "number of matches".

A third of the run-2 failures came from families that need the model to copy
two or more literals from the prompt (`table_access`, `table_literal`,
`string_concat`), so run 3 also doubles the row count and grows the model from
3.0M to 5.2M parameters.

## Third training run

| Setting | Run 2 | Run 3 |
| --- | --- | --- |
| Tokenizer | 1,695 | 1,721 |
| Training rows | 6,000 | 12,000 |
| Parameters | 3,031,296 | 5,228,032 |
| `dim` / `layers` / `heads` / `mlp` | 192 / 6 / 6 / 768 | 256 / 6 / 8 / 1024 |
| Epochs | 30 | 25 |
| Prompt styles in training | 2 | 4 |
| Train loss (final) | — | 0.0847 |
| Val loss (final / best) | — | 0.1949 / 0.1674 (epoch 10) |

Run 3 details, 800-row held-out suite:

| Split | Run 2 | Run 3 |
| --- | --- | --- |
| `test-seen` (canonical) | 70.67% | **100.00%** |
| `test-question` (question style) | n/a | **75.50%** |
| `test-unseen` (held-out identifiers) | 43.00% | **63.00%** |
| `test-paraphrase` (held-out wording) | 0.00% | **34.50%** |
| all rows | 42.50% | **72.88%** |

Syntax and stop-token rates stayed at 100% with no truncation, so every
failure is a wrong answer rather than malformed code. Adding the question style
to training moved the real product prompt path from unusable to 75.5%, and
training on all four styles also lifted the fully held-out wording from 0% to
34.5%.

Per-family pass@1 across the 800 rows:

| Family | pass@1 | Family | pass@1 |
| --- | --- | --- | --- |
| `if_else_health` | 100% | `for_find_max` | 87.5% |
| `for_count_above` | 90% | `if_elseif_grade` | 87.5% |
| `for_find_min` | 90% | `for_product` | 85% |
| `iterate_filter` | 90% | `repeat_until` | 77.5% |
| `for_print_range` | 75% | `table_access` | 75% |
| `while_accumulate` | 75% | `for_sum_all` | 70% |
| `nested_loop` | 70% | `for_sum_first_n` | 67.5% |
| `iterate_print` | 65% | `function_typed` | 62.5% |
| `string_concat` | 50% | `string_interp` | 50% |
| `table_literal` | 50% | `function_add` | 40% |

`function_add` remains the weakest family even though the accumulator-name bug
was fixed, and the remaining failures cluster in families that require copying
two or more literals from the prompt or matching a specific string template.
The next step after shipping this bundle is to increase the number of
distinct prompts per family for those four regardless-of-style failures, not to
grow the model again.

Metrics come from `/content/luau-eval3.json` (Luau 0.746 execution checks).

## Fourth training run

Run 4 isolates the data fix. The architecture, tokenizer, row counts, and
hyperparameters are identical to run 3; only the under-specified prompts and
the exported checkpoint change.

| Setting | Run 3 | Run 4 |
| --- | --- | --- |
| Tokenizer | 1,721 | 1,721 |
| Training rows | 12,000 | 12,000 |
| Parameters | 5,228,032 | 5,228,032 |
| `dim` / `layers` / `heads` / `mlp` | 256 / 6 / 8 / 1024 | 256 / 6 / 8 / 1024 |
| Epochs run | 25 | 13 (early stop, patience 4) |
| Best val loss | 0.1674 (epoch 10) | **0.1438** (step 3,429) |
| Exported weights | final (val 0.1949) | best (val 0.1438) |

The data fix added every stdout-affecting literal to eight families' prompt
styles and tagged each row with `tags.required`. `scripts/audit_luau_prompts.py`
now fails if a generated instruction omits one. All six splits pass the audit
and 14,000/14,000 execution checks.

Run 4 details, 800-row held-out suite:

| Split | Run 3 | Run 4 |
| --- | --- | --- |
| `test-seen` (canonical) | 100.00% | 99.67% |
| `test-question` (question style) | 75.50% | **94.50%** |
| `test-unseen` (held-out identifiers) | 63.00% | 64.00% |
| `test-paraphrase` (held-out wording) | 34.50% | **59.50%** |
| all rows | 72.88% | **83.88%** |

Syntax and stop-token rates are 100% with no truncation. Against the run-4
targets: overall, seen, question, and paraphrase pass; `test-unseen` reached
64% versus a 70% target.

Per-family pass@1 across the 800 rows:

| Family | Run 3 | Run 4 | Family | Run 3 | Run 4 |
| --- | --- | --- | --- | --- | --- |
| `table_literal` | 50% | **100%** | `string_concat` | 50% | **97.5%** |
| `function_typed` | 62.5% | **100%** | `string_interp` | 50% | **92.5%** |
| `while_accumulate` | 75% | **100%** | `function_add` | 40% | **67.5%** |
| `if_else_health` | 100% | 100% | `for_print_range` | 75% | 67.5% |
| `if_elseif_grade` | 87.5% | **97.5%** | `for_count_above` | 90% | 65% |
| `for_sum_first_n` | 67.5% | **92.5%** | `iterate_print` | 65% | 65% |
| `iterate_filter` | 90% | 92.5% | `nested_loop` | 70% | 47.5% |
| `for_find_min` | 90% | 90% | `table_access` | 75% | 75% |
| `for_find_max` | 87.5% | 87.5% | `repeat_until` | 77.5% | 75% |
| `for_product` | 85% | 87.5% | `for_sum_all` | 70% | 77.5% |

The four families that previously sat at their under-specified ceilings moved
to near-ceiling, which confirms the diagnosis: the model was already solving
the answerable rows. The remaining failures concentrate in two places:

- Held-out paraphrase wording for structurally similar families
  (`for_print_range`, `for_count_above`, `iterate_print`, `repeat_until`,
  `table_access`, `function_add`), where the model falls back to a
  sum/accumulator loop.
- `nested_loop`, where question-style prompts make the model reuse one index
  variable in both loops (`for k = 1, 3 do for k = 1, 3 do print(k, k)`).

Phase 2 (family rebalancing) and Phase 5 (recipe ablation) target those two
failure modes. Run 3 remains reachable at commit `d7b6763` for rollback.

Metrics come from `/content/luau-eval4.json` (Luau 0.746 execution checks).
The committed bundle is run 4 and exports the best-validation checkpoint.
