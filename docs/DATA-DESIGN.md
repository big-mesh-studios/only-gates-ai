# Designing training data for small code-generation models

Rules of thumb for building synthetic, execution-checked instruction data, as
used by the Luau task model (`docs/PLAN-luau-tasks.md`). The goal is a dataset
where every row is unambiguous, verifiable, and measures one thing.

## The unit of data

Each row is a self-contained contract:

```json
{
  "id": "for_sum_all/train/000123-4f2a9c1b7e30",
  "family": "for_sum_all",
  "mode": "write",
  "prompt_style": "canonical",
  "instruction": "Write a for loop in Luau that adds up every element of `values` and prints the total.",
  "code": "local total = 0\nfor i = 1, #values do\n    total += values[i]\nend\nprint(total)",
  "harness": {
    "prelude": "local values = { 4, 9, 2, 7 }",
    "expect_stdout": "22"
  },
  "split": "train",
  "tags": {}
}
```

The model sees only `instruction`. Everything else exists to check the answer.
A good row has four properties: **one concept**, **one correct behavior**,
**no hidden inputs**, and **a check that can fail**.

## Rules of thumb

### 1. Every answer must be machine-verifiable

If you cannot execute it and compare output, it is not training data — it is a
guess. Prefer a deterministic stdout contract over "looks right".

- Run the reference code once at generation time to compute `expect_stdout`.
- Run the *generated* code at evaluation time against the same harness.
- Make wrong answers fail: write a deliberately wrong snippet and confirm the
  harness rejects it. A harness that passes everything measures nothing.

### 2. The prompt must uniquely determine the answer

If several programs are valid, a strict stdout check will mark correct answers
wrong.

- Either constrain the prompt (names, literals, output format), or
- accept semantic equivalence / multiple references.

For beginner tasks, constraining is cheaper and teaches copy fidelity, which is
most of the difficulty at small scale.

### 3. The prompt must contain everything the answer needs

Any symbol in the prelude must be named in the instruction.

- Good: "Given the array `values = { ... }`, write a loop that ..."
- Bad: a hidden prelude defining `values` that the prompt never mentions.

The model cannot see the harness, so hidden context becomes an impossible row.

### 4. Keep answers short, and measure the tail

Track prompt + answer tokens per family, not just the average. At least 20% of
context should remain free; the worst family sets your context budget.

- If p95 exceeds ~60–70% of context, simplify the family or raise context.
- Never truncate a code answer to fit: it teaches broken syntax. Skip the row
  instead (the Luau packer does this).
- Always keep the end token in the label budget, even when truncating prose.

### 5. Vary exactly one axis per split

Every split should isolate one generalization question. If a split changes
values *and* wording *and* names, a failure tells you nothing.

### 6. Split by generation axis, not by random row

Random row splits leak near-duplicates: the same template with a different
literal appears in both train and test.

Split by seed and by identifier pool:

| Split | What changes | Question it answers |
| --- | --- | --- |
| `train` | — | Can it learn the distribution? |
| `val` | new seed, same pools/styles | Is it overfitting? |
| `test-seen` | new values, canonical wording | Can it copy constants? |
| `test-question` | question phrasing, new values | Does the product prompt path work? |
| `test-paraphrase` | held-out wording | Does it generalize across phrasing? |
| `test-unseen` | held-out identifier pool | Does it copy names or memorize them? |

`test-seen` at 100% is expected once the model fits the templates. It is not
evidence of generalization — read it together with the other splits.

### 7. Deduplicate on the pair, not the code

Deduplicate `(instruction, code)` tuples. The same code with five different
instructions is five useful rows; the same instruction with the same code twice
is waste and skews the loss.

Track unique codes per family too — a family with 4 distinct programs and 600
rows is 600 near-duplicates, not 600 examples.

### 8. Balance families and detect saturation

- Generate a fixed number of rows per family by default (the Luau set used
  600). When one family fails a held-out split, weight *training* rows toward
  it instead of only adding more data.
- Weight train and validation, never the test splits: changing test-family
  counts makes scores incomparable across runs.
- Allocate with a largest-remainder method so the total stays exact, and keep a
  flag for the old equal allocation.
- Cap attempts per family; if it cannot produce enough unique pairs, the
  generator space is too small.
- When saturated, **widen the space** (more field names, value ranges, shapes)
  or **add a family**. Never silently duplicate to hit a count.
- A low-diversity family quietly teaches memorization; audit distinct codes
  before training.
- Watch the families you reduced: if a previously strong family drops more than
  ~3 points, the rebalance was too aggressive.

### 9. Train several prompt styles; hold out one

One canonical wording is not enough. The model should see:

- canonical / imperative ("Write a for loop that ...")
- a second imperative paraphrase ("Write Luau code that ...")
- a question form ("How do I ... in Luau?")

Hold out a *distinct* wording for `test-paraphrase`. Rotate styles per row with
`(produced + offset) % len(styles)` — indexing by attempt count can collapse
onto one style.

### 10. Avoid cross-family collisions and unmentioned distractors

- Do not reuse a name for two roles (`sum` as a function name and as an
  accumulator taught `local function sum += ...`).
- Do not emit fields or variables the prompt never mentions (`name = "Ada"` in
  an unrelated table-access task).
- Keep each row's identifiers drawn from the split's pool so unseen-name tests
  stay meaningful.

### 11. Make the harness boring and deterministic

- No randomness without a fixed seed, no clock, no filesystem, no network.
- Do not depend on hash-table iteration order. Prefer arrays, or sort keys.
- Pin number formatting (`tostring`, integer-only values) to avoid locale or
  float-format differences.
- Capture stdout; print a single sentinel value when only a result matters.
- Give every execution a timeout, and treat timeout as failure.

### 12. Mask the prompt in the loss

Supervise only the answer tokens plus the end token. Prompt tokens must be
`-100` (or equivalent). If the model is trained on its own prompt, it learns to
reproduce questions instead of answering them.

Test this without a GPU: assert that no prompt token contributes to the loss
(see `scripts/test_luau_sft.py`).

### 13. Co-design the tokenizer with the data

- Keep the app's special tokens (`<|user|>`, `<|assistant|>`, `<|end|>`,
  `<|pad|>`).
- Audit tokens per family; common keywords should be single tokens.
- Measure `tokens / character` and the p95 sequence length before training.
- A tokenizer that splits `elseif` into four pieces wastes context and makes the
  task look harder than it is.

### 14. Version and commit the data

- Record the generator seed and a content hash.
- Commit the generated JSONL so a run is reproducible.
- Regenerate only through the generator; never hand-edit rows.
- Keep the generator, the tokenizer, and the checkpoint in the same commit when
  they change together.

### 15. Keep a small human-reviewed golden set

Sample 20–50 rows and read them. Check for ambiguous prompts, leaked answers,
nonsense identifiers, and answers that are technically correct but bad style.
Automation catches syntax; only reading catches intent.

### 16. Report per-family results, never only the aggregate

An 80% average can hide one family at 0%. Track pass@1 per family per split, and
treat a family at 0% as a data or generator bug first, a model bug second.

## Anti-patterns

- Duplicating rows to reach a target count.
- Random row splits that leak templates.
- Hidden preludes the prompt does not mention.
- Prompts with multiple valid answers checked by exact stdout.
- Distractors that are never referenced.
- Truncating code answers to fit context.
- Training on prompt tokens.
- Changing the tokenizer without regenerating and re-auditing the data.
- Reading only the aggregate score.
- Calling a `test-seen` score of 100% "generalization".

## Volume heuristics

| Parameter | Rule of thumb |
| --- | --- |
| Rows per family | 200–600 distinct `(instruction, code)` pairs |
| Prompt styles per family | 3–5 in training, 1–2 held out |
| Validation rows | ~10% of train |
| Held-out eval rows | 20–40 per family per split |
| Duplicate rate | ~0; audit unique pairs |
| Epochs | 20–30 for small SFT sets; watch val loss, not train loss |

More rows per family help until the family's *distinct behavior* space is
exhausted. After that, add a new family or a new axis (values, names, shapes,
composition), not more duplicates.

## Pre-training checklist

- [ ] Every row executes and matches `expect_stdout`.
- [ ] A deliberately wrong answer fails the harness.
- [ ] Every instruction is unambiguous given its prelude.
- [ ] No duplicate `(instruction, code)` pairs; unique-code counts audited.
- [ ] Splits are separated by seed/identifier pool/style, not random row.
- [ ] Held-out wording differs structurally from every training style.
- [ ] p95 prompt+answer fits the context with headroom.
- [ ] Tokenizer keeps special tokens and single-token keywords.
- [ ] Prompt tokens are masked in the labels; the end token is supervised.
- [ ] Per-family metrics are part of the evaluation report.
- [ ] 20–50 rows have been read by a human.

## Worked example: Luau task data

| Property | Value |
| --- | --- |
| Families | 20 |
| Train / val | 12,000 / 1,200 rows (405-809 per family, weighted) |
| Eval | 800 rows across four splits |
| Prompt styles in training | 4 |
| Held-out style | 1 (`paraphrase_b`) |
| Tokenizer | 1,721-token BPE, 1,460 merges |
| Context | 256 tokens; worst family p95 = 94 |
| Execution checks | 14,000 / 14,000 pass under Luau 0.746 |

The two data bugs that mattered most were a cross-family name collision
(`function_add` reusing `sum`) and an unmentioned distractor field
(`table_access` emitting `name = "Ada"`). Both produced plausible-looking code
that the execution check correctly rejected — which is exactly why the harness
exists.