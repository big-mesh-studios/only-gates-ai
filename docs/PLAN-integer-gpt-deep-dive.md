# Plan: Integer GPT deep-dive page

Status: implemented. See "Implementation status" at the end for the verified result.

## Objective

Add a from-scratch tutorial at `/learn/deep-dive/` that teaches a developer how
the current Integer GPT works and how to implement it. The page is a companion
to `/learn/` (the field notes), but goes all the way down to fixed-point
arithmetic, int8 quantization, the integer forward pass, sampling, training,
export, and verification.

The page must be readable on its own: a developer should be able to rebuild the
engine from the page, the full source listings, and the runnable example.

## Decisions

- Extend `apps/only-gates-ai-learn`; prerender a second document.
- Show the real `src/*.c` and `src/*.h` source, not a hand-copied version.
- Add a runnable harness at `examples/integer-gpt/`.
- Center on the C/WASM inference engine and the PyTorch export path.

## Route and build

```text
/learn/                 existing field notes
/learn/deep-dive/       new tutorial
```

- `src/entry-server.tsx` exports `renderLearn()` and `renderDeepDive()`.
- `scripts/prerender.ts` writes `dist/index.html` and
  `dist/deep-dive/index.html`, inlining the shared styles.
- Add `deep-dive/index.html` and `src/deep-dive.tsx` for local dev.
- `scripts/assemble-learn.mjs` copies the whole `dist` tree; no deploy change.
- Add a "Build it yourself" link from the field notes.
- Add a deep-dive link in the app's Integer GPT panel.

## Page structure

The table below is the original 16-chapter plan. The beginner expansion adds
chapters 01–09 and renumbers the existing chapters to 10–24; see "Beginner
primer expansion" below for the new chapters.

| # | Chapter | Content |
| --- | --- | --- |
| 0 | Read this first | Float-to-integer contract, repository map, three inference lanes |
| 1 | The model on one page | Shapes, parameter formula, tensor inventory, tied head |
| 2 | Build the float reference | PyTorch model, forward pass, packing, why training is float |
| 3 | Fixed-point toolkit | Q16/Q15/Q12, rounding, integer sqrt, overflow budgets |
| 4 | Quantize at load | Per-column/per-row scales, gains, GELU LUT, packing |
| 5 | The integer matmul | Row activation quantization, int32 accumulation, int64 requantization |
| 6 | RMSNorm in integers | Sum of squares, Q32 epsilon, isqrt, Q16 gain |
| 7 | GELU without floats | 1025-point table, interpolation, clamping |
| 8 | Integer attention | Per-head Q/K/V quantization, Q12 scores, mask, DI-Exp, V sum |
| 9 | The full forward pass | Residual order, buffers, final norm, tied logits |
| 10 | Softmax, temperature, sampling | DI-Exp, Q15 normalization, top-k, xorshift32 |
| 11 | Tokenizer and generation | Word-level vs BPE, context window, chat template, WASM boundary |
| 12 | Train and export | Data, LM/QA stages, hyperparameters, `weights.bin` layout |
| 13 | Verify and debug | Top-1/top-5/probability deltas, test vectors, symptom table |
| 14 | Performance and gates | KV-cache decode schedule, WASM SIMD lane, FPGA lowering contract, conformance vectors |
| 15 | Appendices | Full source, runnable CLI, build commands, glossary, checklist |

Every chapter: why, math, code, checkpoint test, production source, common bug.

## Code integrity

- Import the real source with Vite `?raw`:
  - `src/int_math.h`
  - `src/tiny_gpt.h`
  - `src/tiny_gpt_quantize.c`
  - `src/tiny_gpt_infer.c`
- Inline chapter excerpts are extracted by function markers from the same raw
  imports. If a marker disappears, the build fails.
- Full files appear in collapsible appendices.
- The PyTorch listings come from the real trainer definitions.

## Runnable example

`examples/integer-gpt/`:

- `README.md`
- `Makefile`
- `main.c`

The Makefile compiles the real `src/tiny_gpt_infer.c` and
`src/tiny_gpt_quantize.c`. `main.c` is a CLI harness:

- `--selftest`: synthetic weights, forward pass, probability sum, sampler
  determinism, temperature monotonicity.
- `--weights`, `--config`, `--tokens`: load a real checkpoint and inspect logits,
  probabilities, and samples.
- `--steps`, `--temperature`, `--top-k`, `--seed`: generation loop.
- `--dump-buffer`: inspect intermediate Q16 buffers.

Tokenization remains a separate layer; the README explains how to obtain token
ids from the BPE tokenizer.

## Diagrams

1. End-to-end pipeline with precision labels.
2. Tensor shapes through the model.
3. Per-column vs per-row quantization.
4. int8 × int8 → int32 → int64 requantization.
5. Attention pipeline and causal mask.
6. Packed checkpoint order and PyTorch-to-C transpose.

All SVGs use `role="img"`, `<title>`, `<desc>`, and responsive `viewBox`.

## Numeric details to state explicitly

- Activation quantization has two regimes: raw Q16 when `max_abs <= 127`, and
  `ceil(max_abs/127)` rescaling otherwise.
- `int_div_round` rounds ties away from zero; `int_shift_round` adds half and
  arithmetic-shifts, so negative ties round toward +∞.
- Weight scales are Q16 fractions; activation scales are Q16-unit integers.
- Attention scores are Q12 (`>> 36`); V aggregation is Q16 (`>> 15`).
- GELU interpolation shifts by 11 (`1 << 11` step).
- `tiny_gpt_probabilities` doubles Q15 to Q16; the JS wrapper divides by 65536.
- The TypeScript `tinyGpt.ts` int8 path keeps RMSNorm/GELU/softmax in float.
  Only `src/tiny_gpt_infer.c` is fully integer.
- Two schedules share one arithmetic: the reference lane recomputes the full
  context, and the decode lane prefills once and then computes one context row
  per token against a per-layer KV cache. Tests assert bit-identical logits.
- `examples/integer-gpt` and the WASM SIMD lane must produce the same Q16 logits
  as the scalar lane; the SIMD dot product groups integer additions differently
  but the result is exact.

## Verification

- `pnpm --filter @only-gates-ai/learn typecheck`
- `pnpm --filter @only-gates-ai/learn build`
- `pnpm build:site` copies both pages into the main app output.
- Built deep-dive page contains no unexpected `<script>` tags.
- Local anchors and relative links resolve.
- SVG accessibility attributes present.
- Responsive at 320 / 768 / 1280 px; print styles readable.
- `make -C examples/integer-gpt check` compiles and runs the selftest.
- Existing app tests still pass if production code changes (it should not).

## Consistency fix

The field-notes page now states that the Solid 2 weights and tokenizer ship in
`apps/only-gates-ai/public/models/tiny-gpt-solid2/`, rather than leaving the
artifacts implicit. A stale note about missing artifacts was not present in the
current source.

## Beginner primer expansion

Status: implemented on top of the original deep dive.

The original page assumed the reader already knew what a GPT is and why a
transformer block contains attention, an MLP, norms, and residuals. The
expansion adds a nine-chapter foundations track before the existing model
chapters, so the page can be read by a developer with no prior GPT knowledge.

New chapters:

| # | Chapter | Purpose |
| --- | --- | --- |
| 01 | The math you need | Vectors, dot products, matrix multiply, softmax, RMS, logarithms, derivatives |
| 02 | What a GPT is | Next-token prediction, the generation loop, parameters, context, training vs inference |
| 03 | Tokens and embeddings | Tokenizers, vocabulary, ids, embedding lookup, position embeddings, the context window |
| 04 | The transformer block | Residual stream, RMSNorm, attention, MLP, residual adds, pre-norm, parameter budget |
| 05 | Attention from scratch | Q/K/V intuition, scaling, causal mask, multi-head shapes, a two-token worked example |
| 06 | MLP and GELU | Why expansion and nonlinearity are needed, GELU shape, parameter count |
| 07 | Logits and sampling | Tied head, softmax, temperature, top-k, greedy vs sampling, the generation loop |
| 08 | How training works | Windows and targets, cross-entropy, gradients, backpropagation, AdamW, epochs, overfitting |
| 09 | Implement the forward pass | A complete runnable NumPy reference, PyTorch weight loading, verification path to C |

The existing chapters were renumbered 10–24 and expanded with plain-language
explanations, line-by-line reading guides, and numeric examples. The navigation,
cross-references, glossary, appendix learning path, and prerender description
were updated to match.

Verified for the expansion:

```text
pnpm --filter @only-gates-ai/learn typecheck   pass
pnpm --filter @only-gates-ai/learn build       pass
```

The rendered deep-dive page contains 25 chapters, no client-side `<script>`
tags, and no duplicate element ids. The new SVG diagrams use `role="img"`,
`<title>`, and `<desc>`.

## Implementation status

Done:

- `src/entry-server.tsx` exports `renderLearn()` and `renderDeepDive()`.
- `scripts/prerender.ts` writes `dist/index.html` and
  `dist/deep-dive/index.html`, inlining the shared and deep-dive styles.
- `deep-dive/index.html` and `src/deep-dive.tsx` provide the dev entry.
- The field notes link to the deep dive from the masthead, hero, and footer.
- `IntegerGptPanel.tsx` links to `learn/deep-dive/`.
- `examples/integer-gpt/` builds the real engine and passes `make check`.
- The page imports the four production C files with `?raw`; a missing function
  marker fails the build.
- The deep-dive page renders six accessible SVG diagrams and contains no
  client-side `<script>`.
- Anchors, relative links, printer styles, and responsive breakpoints were
  checked in the assembled site.

Verified:

```text
pnpm --filter @only-gates-ai/learn typecheck   pass
pnpm --filter @only-gates-ai/learn build       pass
pnpm build:site                                pass
pnpm typecheck:app                             pass
ctest --test-dir build-tests                   7/7 pass
pnpm --filter @only-gates-ai/app test          12/12 pass
make -C examples/integer-gpt check             pass
```
