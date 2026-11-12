# Lowering Integer GPT to FPGA gates

This is the contract for turning the shipping browser engine into a
fixed-function accelerator. It exists so that the demo, the deep dive, and a
future bitstream all agree on what "Integer GPT on an FPGA" means.

The browser engine is the golden model. Nothing in this document invents a
second algorithm: it maps `src/tiny_gpt_infer.c` operation by operation onto
gates, memories, and a token schedule.

## 1. Architecture boundary

| Runs on the host CPU | Runs in the accelerator |
| --- | --- |
| BPE tokenization and detokenization | Token/position embedding lookup |
| Chat template (`<|user|>` … `<|assistant|>`) | RMSNorm, QKV projection, attention |
| Top-k sampling and RNG (for now) | MLP, output head, integer softmax |
| Weight quantization at load (float → int8/Q16) | KV cache management |

The boundary is the same one the WebAssembly module has today: JS moves token
ids in and Q16 logits out, and every arithmetic operation in between is integer.
A future build can move sampling on-chip (comparator network + LFSR); it is not
required for a first token.

## 2. Op-by-op lowering

Config: `V=4096, T=256, D=128, L=4, H=4, M=512`, 1,344,640 parameters.

| Operation (source) | Integer form | Hardware shape | Storage |
| --- | --- | --- | --- |
| Embedding add (`forward_range_impl`) | int8 × Q16 scale, int64 add, saturate to int32 | ROM read + multiply + adder | token 4096×128 = 512 KiB int8; pos 256×128 = 32 KiB int8; scales 17 KiB |
| RMSNorm (`rmsnorm_q16`) | int64 sum of squares, headroom shift, `int_isqrt_u64`, `int_div_round` by RMS | multiplier tree + isqrt pipeline + one reciprocal | gain 128 × int32 per norm |
| Linear projection (`matmul_q16`) | int8 × int8 → int32 accumulate; activation row quantized to int8 with Q16 scale; requantize `acc · act_scale · weight_scale >> 16` with rounding | MAC array / DSP slices; one contiguous `k` row per output column | 1.34 MB int8 + per-column Q16 scales |
| GELU (`gelu_q16`) | 1025-entry Q16 LUT, linear interpolation, shift 11 | BRAM/ROM + subtract + multiply + shift | 1025 × int32 = 4 KiB |
| Attention scores | int8 dot over `head_dim`, Q16 scales, `>> 36` to Q12, clip ±20 nats | MAC lanes + shifter + comparator | KV cache 2 × 128 KiB int8 |
| Attention probabilities | DI-Exp LUT (33 entries), Q15 normalize | LUT + reciprocal | 33 × int32 |
| Value aggregation | `sum(P_q15 · V_int8 · V_scale) >> 15`, int64 accumulate | MAC array with wider accumulator | reuses KV cache |
| Softmax (`softmax_last_q15`) | DI-Exp, 20-nat clip, Q15 normalize | LUT + reciprocal | 16 KiB |
| Sampling | top-k insertion sort, xorshift32 | comparator network + LFSR (optional on-chip) | — |

### Fixed widths to preserve

- Activations: Q16 `int32`.
- Weights: `int8`, one Q16 scale per output column.
- Matmul accumulator: `int32` (max 127·127·512 = 8,258,048).
- Requantization product: `int64`, then one rounded shift by 16.
- Attention scores: Q12, clipped to ±81,920.
- Probabilities: Q15.

Any hardware datapath that keeps these widths and the same shift/rounding
points reproduces the browser's logits exactly. That is what
[FPGA-CONFORMANCE.md](FPGA-CONFORMANCE.md) verifies.

## 3. The awkward operations (read this before writing RTL)

These are the parts that do not lower to a naive MAC array, and the honest
options:

1. **Dynamic per-row activation quantization.** The engine computes `max|row|`,
   then `q = round(x · 127 / max)` and stores `scale = ceil(max/127)`. In
   hardware this needs a max-reduce, a division, and a multiply per element.
   - Cheapest: replace the division with a reciprocal from a small LUT or a
     Newton iteration.
   - Most FPGA-friendly: static or per-tile scales chosen at synthesis time.
     This changes numerics, so it needs its own conformance vectors — do not
     mix it into the golden lane.
2. **RMSNorm and softmax normalization** divide by a runtime denominator.
   One divider pipeline (or reciprocal + multiply) serves all of them; the
   call sites are `int_div_round` in `rmsnorm_q16`, `attention_q16`, and
   `softmax_last_q15`.
3. **`int_isqrt_u64`** is a 64-bit square root. A 32-bit input (after the
   headroom shift) reduces it to a 6–8 stage pipeline, or use a LUT + Newton
   step.
4. **Top-k sampling** is data-dependent control flow. Keep it on the host, or
   implement a fixed-k sorting network.

## 4. Resource floor

Matmul work per generated token (one context row, KV cache warm), `L=4`:

- QKV + attention output projection: `L·4D²` = 262,144
- MLP up/down: `L·2DM` = 524,288
- Output head: `D·V` = 524,288
- Attention scores + value sum at `T=256`: `L·2TD` ≈ 262,144

Total ≈ **1.57 M MAC-equivalents per token** (≈ 1.31 M for the matmuls alone).

Examples, labelled as projections rather than measurements:

| Assumption | Estimate |
| --- | --- |
| 128 int8 MAC lanes @ 200 MHz | ~61 µs/token (~16,000 tok/s) |
| 256 lanes @ 200 MHz | ~31 µs/token |
| Weights streamed from 3.2 GB/s DRAM every token | ~420 µs/token (memory-bound) |
| Weights fully resident in BRAM | compute-bound, tens of µs/token |

The model is 1.34 MB of int8 weights (1.28 MiB). That fits on larger FPGAs (for
example ~13 Mb of BRAM) but is tight on small boards, so the realistic first
design is "weights in external memory, tiles cached on-chip". The KV cache is
`2 · L · T · D` int8 = 256 KiB plus 32 KiB of scales, which does fit on-chip.

One practical note: the browser allocates a 4 MiB `context × vocab` logits
buffer for debug/tests. Hardware never needs it — one row is 16 KiB — and a
real design should zero it out of the memory plan.

## 5. Where the gate compiler stops

The repository's compiler lowers small symbol-classifier circuits to the
16-gate vocabulary (`nor`, `xor`, `and`, …). A full transformer needs more than
that vocabulary:

- wide integer arithmetic (8/16/32-bit add, multiply, shift, compare),
- a matrix-tile schedule and on-chip memory plan,
- LUT/ROM generation for GELU and DI-Exp,
- token-by-token control and the KV cache.

These are engineering problems, not reasons the approach fails. The first
milestone should be one transformer block on gates, not the whole model.

## 6. Bring-up checklist

1. Freeze the checkpoint. Record `weights_fnv1a64` from
   `--dump-vectors` (see the conformance doc).
2. Build the golden vectors: one prompt window, one position, all Q16 logits.
3. Implement the embedding add and one `matmul_q16` tile in RTL; diff against
   the Q16 logits of a single linear layer exported from the C harness.
4. Add RMSNorm with the same rounding; diff after the norm.
5. Add one attention head with the KV cache; diff scores and probabilities at
   Q12/Q15.
6. Add the MLP and output head; diff the final logits bit-for-bit.
7. Then, and only then, replace the reference activation quantizer with static
   scales and generate a second set of vectors for that variant.
8. Measure tokens/second, latency, LUT/FF/BRAM/DSP use, and power; publish the
   number, not a claim.
