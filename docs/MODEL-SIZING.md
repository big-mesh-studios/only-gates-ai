# When does a small model need to be larger?

Rules of thumb for deciding whether a narrow, execution-evaluated model should
grow, or whether the bottleneck is really data, tokenization, or training.

These were developed while training the Luau task model
(`docs/PLAN-luau-tasks.md`), but the reasoning applies to any small
instruction-to-code model.

## The three-way split

Always read **training loss**, **validation loss**, and **task pass@1**
together. They point at different problems.

| Train | Val | pass@1 | Diagnosis | Fix |
| --- | --- | --- | --- | --- |
| High, still falling | High | Low | **Undertrained** | More steps / better LR before more parameters |
| High, flat | High | Low | **Capacity or optimization** | Bigger model, or fix LR/init/normalization |
| ~0 | High | Low | **Data-limited** | More/diverse data — size will not help |
| ~0 | Low | Low | **Objective mismatch** | Loss is not task success; fix labels, decoding, stop tokens |
| Low | Low | High seen, low paraphrase | **Coverage gap** | More prompt styles, not more parameters |

## The five rules of thumb

1. **If it cannot fit its own training set, it is too small.**
   Evaluate on the *train* split. Below ~95–98% there, you have a capacity or
   optimization problem. This is the cheapest decisive test and does not depend
   on generalization.

2. **A train/val gap above ~2–3× in loss means data-limited.**
   Closing that gap needs more varied data, not more parameters. Extra capacity
   usually just memorizes harder.

3. **Double the data for the worst family, same model.**
   - Accuracy moves → data-limited. Keep generating.
   - Accuracy flat → capacity-limited for that family. Size up (or add depth).

4. **Double the parameters, same data and steps.**
   If validation loss improves less than ~3–5% relative, scaling is not worth
   the inference cost. The threshold is deliberately conservative for a mobile
   target.

5. **Watch the regression matrix, not just the average.**
   Compare per-family pass@1 before and after adding new families. If a new
   family drops old ones by more than a few points on balanced data, you are at
   capacity pressure — that is what dilution looks like. If nothing regresses,
   there is slack and the model does not need to grow.

## Structural signals often mistaken for size problems

- **Longer prompts or answers fail, short ones pass** → context or depth issue,
  not width.
- **Errors are "right first step, wrong later step"** → composition/depth. Add
  layers before width.
- **Errors are lexical — wrong names, swapped constants** → data coverage or
  tokenizer, not capacity.
- **Common keywords split into 3+ tokens** → the tokenizer is wasting context
  and effective capacity. Retrain the vocabulary first.
- **A family is 100% canonical but ~0% paraphrased** → phrasing coverage, not
  parameter count.

## Cheap protocol

Run three trainings at the same budget, then compare validation loss *and* the
per-family pass@1 table:

- **A** — current size, current data (baseline)
- **B** — current size, 2× data for the 3–4 weakest families
- **C** — ~2× parameters (`dim=384, layers=6, heads=8, mlp=1536` for the 5.2M
  config), same data

Read the result:

| Outcome | Conclusion |
| --- | --- |
| B improves, C flat | Stay small; generate more data |
| C improves, B flat | Size up |
| Both improve | Do both; size moderately |
| Neither improves | Generator, tokenizer, or decoding — not the model |

For a narrow task model, this protocol usually points at data before
parameters. The exception is adding entire new *feature groups*, where the
thing you are really buying is depth.

## Capacity per behavior cluster

A quick sanity check before scaling: parameters divided by the number of
distinct behavior clusters (task families). Below roughly **100k–250k
parameters per cluster**, expect interference and dilution.

The relationship is sublinear because grammar is shared across clusters: 5× the
families usually needs ~2–3× the parameters, not 5×.

Rough targets for Luau-style coverage:

| Scope | Behavior clusters | Plausible parameters |
| --- | --- | --- |
| Narrow task model | 20 | 5M |
| Every language construct, template tasks | ~80–120 | 15–30M |
| Compositional semantics, execution-checked | hundreds | 50–120M |
| Open-ended assistant | unbounded | 500M–1B+ |

## Worked example: Luau run 3

| Metric | Value |
| --- | --- |
| Training loss (final) | 0.0847 |
| Validation loss (final / best) | 0.1949 / 0.1674 |
| `test-seen` (canonical) | 100.00% |
| `test-question` | 75.50% |
| `test-unseen` (identifiers) | 63.00% |
| `test-paraphrase` (held-out wording) | 34.50% |

Train/val loss ratio ≈ 2.3× with perfect seen accuracy: the model has fit the
distribution and is limited by **data diversity**, not capacity. Growing it
would mostly lower a training loss that is already low enough, so the next
experiment is more prompt diversity for the weak families — not a bigger
model.

## Constraints specific to this app

Parameter count is not free even when it helps accuracy:

- The browser downloads float32 weights and quantizes to int8.
  5.2M params → 20.9 MB download, ~5 MB in memory.
  25M → ~100 MB download. 100M → ~400 MB.
- WASM decode cost scales roughly linearly with parameters. On a phone, 50M
  params would likely be seconds per token.
- Luau also has far less public source code than Python, so a large model would
  be undertrained rather than well-fed. Synthetic, execution-checked data is
  the main lever.

Practical ceiling for the in-browser tab: **~20–30M parameters**. Beyond that,
the long tail belongs behind a server or in retrieval, not in the weights.