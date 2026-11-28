# Raw run log

Scratch log of completed runs so nothing is lost. The curated summary lives in
`README.md`.

Common setup unless noted: MNIST, 10,000 train / 2,000 test, 14×14 binary
(threshold 0.25), batch 128, Adam lr 0.1, temperature anneal 1.0 → 0.25,
forest of K circuits per label with cross-entropy over summed votes.

## K=1, no forest (early baseline)

| Model | Params | Gate cells | Soft | Hard |
| --- | --- | --- | --- | --- |
| dense | 71,072 | 202 | 81.2% | 27.3% |
| conv-shallow | 16,560 | 594 | 48.3% | 27.1% |
| conv-deep | 22,704 | 4,842 | 35.0% | 28.1% |
| float-cnn (reference) | 1,418 | 0 | 10.1% | – |

All hard accuracies pinned at 27–28% because the one-bit argmax ties.

## Forest K=4, forest CE, no STE

| Model | Params | Gate cells | Soft | Hard | All-zero |
| --- | --- | --- | --- | --- | --- |
| dense | 75,392 | 232 | 74.3% | 63.2% | 0.0% |
| conv-shallow | 20,880 | 624 | 52.5% | 25.0% | 0.0% |
| conv-deep | 27,024 | 4,872 | 46.0% | 31.8% | 0.0% |

## Forest K=4, spatially pooled conv

| Model | Params | Gate cells | Soft | Hard | Note |
| --- | --- | --- | --- | --- | --- |
| dense-wide | 182,912 | 424 | 75.5% | 70.8% | 256→128 head |
| conv-grid | 91,808 | 4,872 | ~70–72% | ~56–60% | plateau, 20 epochs |

`conv-grid` is 16@3×3 s1 → 2×2 OR → 32@3×3 s1 → 2×2 OR → 512 → 64 → 64 →
10×K. Therefore most of its cost is the dense head, not the shared detectors.

## STE fine-tune is a no-op

`dense`, 16 soft epochs then 4 STE epochs (`--ste-lr 0.02`, T held at 0.25):

| Phase | Soft | Hard |
| --- | --- | --- |
| before STE | 71.75% | 62.90% |
| after 4 STE epochs | 62.90% | 62.90% |

Hard accuracy is identical for all four STE epochs and the STE loss rises from
~1.03 to ~1.15 and then sits flat. The annealed argmax circuit is a stable
point of the hard cross-entropy: the STE gradient is too small (the T=0.25
softmax is saturated) to flip any wire or gate. The residual soft→hard gap
(~9 points here) is therefore a property of the graded-vs-binary output, not
something a short fine-tune recovers. Earlier STE "collapses" were the
`clamp` bug, not a real effect.

## Forest size sweep (dense)

20 epochs, forest CE over raw vote sums, `--score-scale 1.0`:

| K | Model | Params | Gate cells | Soft | Hard | Gap |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | dense | 71,072 | 202 | 62.3% | 49.2% | 13.2 |
| 1 | dense-small | 26,080 | 106 | 58.5% | 49.4% | 9.2 |
| 4 | dense | 75,392 | 232 | 76.8% | 68.5% | 8.3 |
| 4 | dense-small | 29,440 | 136 | 75.1% | 67.8% | 7.3 |
| 8 | dense | 81,152 | 272 | 83.7% | 79.2% | 4.5 |
| 8 | dense-small | 33,920 | 176 | 84.2% | 75.9% | 8.3 |
| 16 | dense | 92,672 | 352 | 84.9% | 77.8% | 7.1 |
| 16 | dense-small | 42,880 | 256 | 76.7% | 62.1% | 14.6 |

Circuit pruning (`--prune-circuits`) was a no-op: 100% of circuits were
non-constant on the training set at K ≤ 8, and 97.5–98.8% at K=16, where
pruning *lost* 0.75 points.

### `--score-scale` is a strong knob

`--score-scale` multiplies the vote scores before the cross-entropy, and the
loss now divides the summed votes by K first. So "raw sum" at K=4 is an
effective scale of 4, while `--score-scale 1.0` with the normalised loss is 1.
That difference is not cosmetic:

| Setting | Effective scale | Soft | Hard |
| --- | --- | --- | --- |
| raw sums, K=4 | 4 | 76.8% | 68.5% |
| normalised, `--score-scale 1.0`, K=4 | 1 | 49.4% | 45.7% |

All the K-sweep numbers above were collected with one consistent setting (raw
sums, before the divide-by-K change), so the K comparison is valid. The
normalised loss defaults to `--score-scale 8.0`; at that setting K=8 reaches
hard 77.4% and K=16 hard 77.5% (see the ensemble section), i.e. the same story
as the raw-sum sweep. Treat absolute accuracies from different scale settings
as not comparable.

## Ensemble compression (forest → fewer circuits)

Two selection rules, `dense`, `--score-scale 8`, 20 epochs.

**(a) Rank by individual balanced accuracy, keep the best m per label.**
K=8: 59.1 / 58.9 / 59.6 / 67.3 / 71.3 / 75.3 / 76.5 / 76.7% for m = 1…8.
K=16: 48.4 / 55.7 / 58.7 / 65.9 / 66.5 / 69.7 / 69.8 / 69.5 / 69.9 / 71.1 /
71.4 / 71.7 / 71.7 / 73.2 / 73.3 / 72.6%.

The top circuits are correlated (m=1…3 sits flat near 59%), then adding the
fourth unlocks the vote. Individual ranking is therefore not a good selection
rule, which is why (b) exists.

**(b) Greedy forward selection by marginal validation contribution.**

`dense` K=8, `--greedy-curve`, selected on 80% of train, scored on test:

| Circuits kept (of 80) | 1 | 10 | 26 | 36 | 49 | 55 | 71 | 80 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Hard accuracy | 20.5 | 47.0 | 58.3 | 73.7 | 77.1 | 79.0 | 78.8 | 77.4 |

49 of 80 circuits (61%) reach full accuracy; 55 circuits peak at 79.0%, above
the full forest's 77.4%, because greedy avoids stacking correlated circuits.
So ~30–40% of circuits are removable without loss when chosen by marginal
value, and the first ~26 circuits are near-useless in combination. Treat the
79.0% peak as "no loss" rather than a real gain: it is within ~1 point of the
full forest on 2,000 test samples.

## Gate-count scaling (dense, K=8)

`--width-sweep`, 20 epochs, `--score-scale 8`, single seed per row:

| Hidden | Params | Gate cells | Soft | Hard | Soft→hard gap |
| --- | --- | --- | --- | --- | --- |
| 128×64 | 81,152 | 272 | 82.5% | 78.7% | 3.8 |
| 256×128 | 193,792 | 464 | 84.8% | **83.5%** | 1.3 |
| 512×256 | 517,376 | 848 | 86.0% | 80.9% | 5.2 |
| 768×384 | 972,032 | 1,232 | 86.1% | 83.6% | 2.6 |
| 1024×512 | 1,557,760 | 1,616 | 86.3% | 81.8% | 4.5 |

Hard accuracy plateaus at ~83.5% from 464 cells onward; 3.5× more gates buys
nothing, while soft accuracy keeps rising. The extra capacity becomes graded
features that hardening discards. Single-seed spread is ~2 points, so the
80.9 row is noise around the same plateau.

### More data does not lift it either

`dense` 256×128, K=8, 50,000 train / 10,000 test:

| Train | Test | Soft | Hard | Peak hard |
| --- | --- | --- | --- | --- |
| 10,000 | 2,000 | 84.8% | 83.5% | 83.5% |
| 50,000 | 10,000 | 86.8% | 82.1% | 83.6% |

The soft model gains ~2–3 points from 5× data; the hard circuit gains none.
Combined with the flat width sweep, this rules out both gate budget and sample
count as the binding constraint. The next section shows what is left:
hardening plus **input coverage** — the hard wiring sees only a subset of the
196 pixels.

### Gate budget arithmetic

One FP32 multiply-add is roughly 5,000–10,000 two-input gates (24×24
multiplier ~2,700–5,000, alignment/normalization shifts ~1,500–2,200, adder
~300–500, rounding + exponent logic ~400–800). Against that:

| Circuit | Gate cells | Fraction of one FP32 FMA |
| --- | --- | --- |
| dense 128×64, K=8 | 272 | ~3–5% |
| dense 256×128, K=8 | 464 | ~5–9% |
| dense 512×256, frozen layer 0 | 848 | ~8–17% |
| conv-grid, K=4 | 4,872 | ~0.5–1 |

A 1,418-parameter float CNN is ~7–14 million gate equivalents; the soft dense
model is ~400–800 million. So the hard dense forest is 4–5 orders of magnitude
smaller than the float baselines in gate count, at 83.5% vs ~99% accuracy on
quantized 14×14 MNIST.

Storage for the hard circuit: two wire indices (≤8 bits each at fan-in 196)
plus a 4-bit gate per cell, ≈ 620 bytes for 272 cells, versus ~317 KB for the
81k-parameter soft model.

## Depth vs input coverage

A hard cell reads two wires, so an output's dependency set can at most double
per layer: depth L reaches ≤2^L input bits. With 196 input pixels, full reach
needs ≥8 layers.

Constant 464 cells, K=8, 20 epochs:

| Layers | Shape | Per-output mean/max | Inputs seen by any output | Soft | Hard |
| --- | --- | --- | --- | --- | --- |
| 3 | 256,128 | 7.1 / 8 | 85 | 86.5% | **80.7%** |
| 4 | 128,128,128 | 10.1 / 16 | 67 | 82.3% | 70.0% |
| 5 | 96×4 | 13.2 / 21 | 42 | 75.3% | 58.0% |
| 7 | 64×6 | 34.7 / 48 | 57 | 76.8% | 56.0% |
| 9 | 48×8 | 44.1 / 52 | 52 | 57.5% | 43.9% |
| 13 | 32×12 | 27.0 / 27 | 27 | 37.7% | 23.0% |

The per-output receptive field grows with depth exactly as the 2^L bound
predicts, but accuracy *falls*, because the first layer narrows and total
input coverage collapses with it.

Widening the first layer helps coverage but not enough — learned wires
concentrate on favourite pixels:

| Shape | Layer-0 coverage | Final coverage | Hard |
| --- | --- | --- | --- |
| 512,256 | 120 | 89 | 81.3% |
| 512,256,128 | 124 | 67 | 69.1% |
| 512,256,128,128 | 101 | 67 | 71.9% |
| 512,128×4 | 181 | 105 | 58.3% |
| 768,256,128 | 155 | 74 | 77.6% |

So depth prunes coverage layer by layer, and depth-3 remains the best. The
first layer is the gatekeeper: a pixel that no layer-0 wire selects is
invisible at any depth.

## Frozen wires (coverage held at 100%)

`--freeze-wires`: every wire is a shuffled round-robin cover, only gates train.

| Shape | Layers | Layer-0 → final coverage | Soft | Hard |
| --- | --- | --- | --- | --- |
| 256,128 | 3 | 196 → 179 | 68.5% | 65.0% |
| 512,256 | 3 | 196 → 187 | 67.3% | 65.4% |
| 512,256,128 | 4 | 196 → 194 | 76.5% | 73.4% |
| 512,256,128,128 | 5 | 196 → 194 | 76.7% | 74.4% |

With coverage fixed, depth becomes monotone (65.0 → 73.4 → 74.4), confirming
the 2^L dependency bound. But random wires are weaker features than learned
ones, so even at depth 5 this trails learned-wire models.

## Frozen first layer (coverage + learned deeper wiring)

`--freeze-first-wires`: layer 0 is a shuffled round-robin cover, deeper wires
train normally.

| Shape | Layers | Layer-0 → final coverage | Soft | Hard |
| --- | --- | --- | --- | --- |
| **512,256** | **3** | **196 → 154** | 86.9% | **84.2%** |
| 512,256,128 | 4 | 196 → 135 | 86.0% | 81.5% |
| 512,256,128,128 | 5 | 196 → 97 | 82.6% | 73.4% |

84.2% at 848 cells is the best hard result in the experiment. Deep layers still
prune coverage (154 → 135 → 97), which is why depth past 3 keeps losing.

## Pinned fraction per layer

`--pin-fraction f`: pin that share of every layer's wires to a round-robin
cover, re-imposed after each optimiser step; the rest learn. Depth 5,
512,256,128,128, 1,104 cells, K=8:

| Pinned | Soft | Hard | Inputs ever seen |
| --- | --- | --- | --- |
| 0 | 81.9% | 65.4% | 92 |
| 0.25 | 82.5% | 77.5% | 149 |
| 0.50 | 85.5% | 79.9% | 175 |
| 0.75 | 85.7% | 83.5% | 191 |
| 1.00 | 77.4% | 74.1% | 196 |

Same shape, same gate count: hard accuracy moves 65.4% → 83.5% purely with
coverage. Optimum ~0.75 (high coverage + 25% learned wires). Pinning all wires
loses feature quality; pinning none collapses reach.

### Depth at a fixed pin fraction (0.75), funnel

| Shape | Depth | Cells | Soft | Hard | Inputs ever seen |
| --- | --- | --- | --- | --- | --- |
| 256,128 | 3 | 464 | 85.5% | 82.1% | 176 |
| **512,256** | **3** | **848** | 85.6% | **84.5%** | 179 |
| 512,256,128 | 4 | 976 | 85.2% | 83.9% | 185 |
| 512,256,128,128 | 5 | 1,104 | 85.2% | 83.5% | 185 |

### Constant width: depth pays, then turns over

| Shape | Depth | Cells | Soft | Hard | Post-fold cells |
| --- | --- | --- | --- | --- | --- |
| 512,512 | 3 | 848 | 85.6% | 84.5% | 645 (76%) |
| 512×3 | 4 | 1,616 | 87.2% | 85.9% | 758 (47%) |
| **512×4** | **5** | **2,128** | 87.3% | **86.2%** | **877 (41%)** |
| 512×5 | 6 | 2,640 | 87.0% | 84.4% | 921 (35%) |
| 512×6 | 7 | 3,152 | 82.5% | 78.0% | 909 (29%) |

Constant width at depth 5 beats the funnel at the same depth by +2.7 points
(86.2% vs 83.5%) and keeps coverage at 196/196 in every layer. Depths 4–6 are
within ~2 points of each other (85.9 / 86.2 / 84.4), so the peak is broad;
depth 7 collapses to 78% as 3M parameters outrun 10k samples.

The best hard result in the experiment is 512×4 at **86.2%**, folding from
2,128 to 877 cells (41%). For gate efficiency, 512×3 is arguably better:
758 folded cells at 85.9%.

A funnel depth-8 run (512,256,128×5, 1,616 cells, pin 0.75) was aborted around
epoch 15 at **~63% hard**, consistent with the funnel decline.

### Width at constant depth (4 hidden layers, pin 0.75)

| Width | Cells | Params | Soft | Hard | Post-fold |
| --- | --- | --- | --- | --- | --- |
| 512 | 2,128 | 1.9M | 87.3% | 86.2% | 877 (41%) |
| **768** | **3,152** | **4.0M** | 88.1% | **87.0%** | **1,131 (36%)** |
| 1024 | 4,176 | 6.9M | 87.1% | 86.1% | 1,367 (33%) |

Width peaks at 768 and then overfits (1024 is 0.9 points worse and adds 236
post-fold cells). So widening buys +0.8 points at best, not the ~4 needed for
90%. Width alone is not the lever.

### Input resolution (768×4, pin 0.75)

| Input | Cells | Soft | Hard | Inputs ever seen |
| --- | --- | --- | --- | --- |
| 14×14 (196) | 3,152 | 88.1% | **87.0%** | 196/196 |
| 28×28 (784) | 3,152 | 87.2% | 85.8% | 650/784 |

28×28 is slightly *worse*: the same 768-wide first layer covers only 650 of
784 pixels, so most of the extra detail is unused. More pixels without a wider
pinned first layer just dilutes coverage.

### Augmentation

`--augment` applies random rotation (±10°), scale (±10%) and translation
(±10%) to the raw 28×28 image before pooling and thresholding, regenerated
every batch. The first attempt fed degrees straight into `cos`/`sin` (so ±10
*radians*) and collapsed both soft and hard accuracy to ~50%; fixed.

1024×4, pin 0.75:

| Augment | Soft | Hard |
| --- | --- | --- |
| off | 87.1% | 86.1% |
| on | 88.3% | 86.4% |

So augmentation raises soft accuracy by ~1.2 points and hard by only ~0.25.
It does not break the ~87% ceiling.

## Tri-logic cells

Signals in `{0, 0.5, 1}`, each cell an arbitrary 3×3 table, so 3^9 = 19,683
gates from 27 logits/cell (9 entries × 3-way softmax). Soft mode interpolates
the table bilinearly at the continuous pin values; hard mode argmaxes each
entry, each pin, then looks the gate up.

First result, `dense-tri` 256×128, K=8, pin 0.75, sum readout:

| Model | Soft | Hard (sum of 0/0.5/1) |
| --- | --- | --- |
| `dense-tri` | 83.4% | 71.1% |
| `dense` (binary), same shape | 82.1% | 82.1% |

Tri-logic raises soft accuracy slightly but the hardened circuit falls far
behind. The sum readout was a suspect — a pile of 0.5s contributes half a vote
and ties often — so a thresholded readout (count only entries ≥ 0.75) was
tried as well:

| Tri readout | Soft | Hard |
| --- | --- | --- |
| sum of {0, 0.5, 1} | 83.4% | 71.1% |
| count of entries ≥ 0.75 | 83.4% | 69.6% |

The readout is not the problem. Hardening a ternary circuit quantises at three
levels instead of one (pins, table entries, and every intermediate signal),
so the soft→hard gap is 12–20 points rather than the 1–3 the binary model
achieves with pinning. Tri-logic is a negative result as implemented: more
soft expressiveness, much worse hardening.

## Three-input LUTs

Binary signals, three input pins, arbitrary 3-input gate: 2^8 = 256 gates, and
only 8 Bernoulli logits per cell (a gate is its 8-bit truth table). Reach grows
3^L per layer instead of 2^L.

`dense-lut3` 256×128, K=8, pin 0.75:

| Model | Cells | Soft | Hard |
| --- | --- | --- | --- |
| `dense` (2-input), same shape | 464 | 82.1% | 82.1% |
| **`dense-lut3`** | **464** | **89.2%** | **88.2%** |
| `dense` (2-input), 768×4 | 3,152 | 88.1% | 87.0% |

That is +6 points over the 2-input cell at the same cell count, and it beats
the best 2-input model with 7× fewer cells. A 3-input LUT costs ~2–3 two-input
gates, so 464 LUT3 cells ≈ 1,100–1,400 gate equivalents — still well below the
3,152-cell binary circuit.

The LUT3 initialisation matters: `random_logits` returns `log(weights)`, which
is negative everywhere, so `table > 0` starts false and every gate is constant
0 (100% all-zero outputs). A small `randn` table fixes it.

### Lowering to two-input gates

A LUT3 is not a hardware commitment. Every 3-input function lowers to
two-input gates, then simplification applies. Breadth-first over all 256
functions gives the exact minimal cost:

| Two-input gates | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Functions | 5 | 33 | 58 | 136 | 9 | 10 | 5 |

Mean 2.63 gates per cell, worst case 6; the 0-cost functions are constants and
bare inputs, which fold away. So 464 LUT3 cells ≈ 1,220 two-input gates on
average, versus 3,152 cells for the best 2-input model at 87.0%.

The runner's `lut3_gate_cost` reports the actual compiled cost of the trained
tables (sum of per-table minima), not the uniform average.

### LUT3 depth scaling (constant width 512, K=8, pin 0.75)

| Shape | Depth | Cells | Soft | Hard |
| --- | --- | --- | --- | --- |
| 256×128 | 3 | 464 | 89.2% | 88.2% |
| 512×512 | 3 | 1,104 | 90.8% | 90.0% |
| **512×512×512** | **4** | **1,616** | **91.9%** | **91.6%** |
| 512×512×512×512 | 5 | 2,128 | 91.4% | 90.8% |

Peak 91.55% at 1,616 cells; depth 4 beats depth 5. Soft→hard gap only 0.3
points, the smallest of any model here.

## Four-input LUTs

Binary signals, four input pins, arbitrary 4-input gate: 2^16 = 65,536 gates
from a 16-bit truth table (16 Bernoulli logits, the same count as the 16-gate
two-input cell). Reach grows 4^L, so full 196-pixel coverage needs only 4
layers.

| Model | Cells | Soft | Hard | Est. 2-input gates |
| --- | --- | --- | --- | --- |
| `lut3` 512×512 | 1,104 | 90.8% | 90.0% | ~2,900 |
| `lut3` 512×3 | 1,616 | 91.9% | **91.6%** | ~3,800 |
| `lut4` 512×512 | 1,104 | 91.8% | **91.6%** | ~8,900 |
| `lut4` 512×3 | 1,616 | 92.5% | **92.25%** | ~8,150 |
| `lut4` 768×3 | 2,384 | 92.0% | 92.1% | 12,009 |

Widening LUT4 to 768 does not help: 92.1% at 2,384 cells versus 92.25% at
1,616. The measured compiled cost of the trained LUT4 tables is **5.04 gates
per cell**, not the 8.04 uniform bound — trained functions are simpler than
random ones. So LUT4 512×3 is ~8,150 two-input gates, still about 2× LUT3 for
+0.7 points.

## 2-bit ALU cells

Signals are 2-bit values in `{0,1,2,3}`, held as 4-way one-hot distributions.
Each cell picks two signals and one op from a cheap ALU set:

| Op | AND, OR, XOR, NAND, NOR, XNOR | add mod 4, sub mod 4 |
| --- | --- | --- |
| ~2-input gates | 2 | ~6 |

Mean ~3 gates/cell, close to LUT3. Soft mode mixes the op result distributions
under independent pins; hard mode argmaxes pins and op and emits a one-hot
2-bit result. The readout is `1 - p(value = 0)`, so the forest/vote machinery
is unchanged.

Why it might win: arithmetic gives a structured prior (cheap per gate) at
4 states per signal. Why it might not: 4-level signals harden worse, the same
failure mode that sank tri-logic.

10,000 train / 20 epochs, K=8, pin 0.75:

| Model | Cells | Soft | Hard |
| --- | --- | --- | --- |
| `dense-alut` 256×128 | 464 | 78.3% | 74.9% |
| `dense-lut3` 256×128 | 464 | 89.2% | 88.2% |

Negative result: the 2-bit cell is 13 points behind LUT3 at the same cell
count. Four-level signals harden badly (the tri-logic failure again), and the
op set is far smaller than a LUT's function space. The cheap-ALU prior does
not pay.

## Multi-scale input pyramid

`--pyramid` stacks every coarser scale of the same image as extra input
channels: at 14×14 the scales are 14, 7, 4, 2, 1, upsampled back to 14×14 and
concatenated, so the first layer sees 5 × 196 = 980 inputs instead of 196.

This costs input parameters and wiring (the first layer's fan-in is 5×), but
**not gates per cell** — a LUT3 cell still costs ~2.6 two-input gates. It gives
the first layer direct access to coarse shape statistics alongside fine detail,
which a purely local LUT3 cannot synthesise without depth.

First full result, `dense-lut3` 512×3, pin 0.75:

| Variant | Cells | Soft | Hard | Compiled 2-input gates |
| --- | --- | --- | --- | --- |
| no pyramid | 1,616 | 91.9% | **91.6%** | ~3,800 (est.) |
| pyramid | 1,616 | 91.3% | 90.4% | **2,768 (1.71/cell)** |

Accuracy is flat (within single-seed noise), but compiled gates drop 27%:
with coarse scales available directly, the trained tables get simpler (379
cells fold to 0 gates, 395 to 1 gate). So the pyramid is a gate-efficiency
tool, not an accuracy tool.

## Demo export

The LUT4 512×3 model is exported with `--export-hard` to
`apps/only-gates-ai/public/models/lut4-mnist/model.json` (33 KB: wire indices
plus 16-bit truth tables). Re-running the exported circuit in plain Python
scores **92.13% on the full 10,000-image test set**, matching training, so the
export is exact. The web app's Digit classifier tab runs that file in JS.

## Per-class accuracy

Mean accuracy hides weak digits. The exported LUT4 512×3 circuit on the full
10k test set:

| Digit | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Recall | 97.8 | 98.0 | 91.2 | 91.2 | 93.9 | 89.7 | 93.6 | 90.6 | **87.3** | **87.3** |

Overall 92.13%, worst class 87.27%, spread 10.7 points. The top confusions
are 9→4 (48), 5→3 (33), 7→2 (28), 8→5 (24).

With `--class-dro 0.2` the full-test numbers become:

| Digit | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Recall | 98.2 | 97.4 | 92.0 | 91.2 | 92.9 | 90.6 | 91.9 | 91.1 | **88.3** | **88.4** |

Overall 92.26%, worst 88.30%. The gain is concentrated on the weak digits
(8 +1.0, 9 +1.1, 5 +0.9) while 6 gives back 1.7, which is the intended
worst-class trade. This is the model bundled in the demo.

`--class-dro <eta>` adds a group-DRO step on the per-class training loss, and
the runner now reports `perClassAccuracy` for every model so a weak class
cannot hide behind the mean.

DRO on LUT3 256×128, step size matters a lot:

| Setting | Overall | Worst class | Weak classes (5 / 8 / 9) |
| --- | --- | --- | --- |
| baseline | 88.6% | 82.4% | 83.0 / 82.4 / 82.7 |
| `--class-dro 1.0` | 87.7% | 77.7% | 83.0 / 87.4 / 77.7 |
| `--class-dro 0.2` | **89.1%** | **85.3%** | 87.1 / 85.9 / 85.3 |

A gentle step improves both mean and worst-class: 5/8/9 gain 2.4–4.1 points
while 0/1/6 give back 1–4, so capacity is being rebalanced rather than
created. An aggressive step destabilises training and loses both metrics.
`--class-dro 0.2` is the useful setting.

### Demo preprocessing

The first demo centred drawings by bounding box on a 24×24 raster with a thin
brush, which shifted 7s and 9s relative to MNIST. It now follows the MNIST
pipeline: 20×20 box, centre of mass, ~2.3 px brush at 28×28, then 2×2
average-pool and threshold. This changes the demo input only; the exported
circuit is unchanged.

## One-vs-rest forests

The shipped symbol classifier trains one circuit per label, not one shared
trunk. `--one-vs-rest` reproduces that: ten independent LUT4 forests of the
same width and depth, each trained as `is this digit c?` with a positive-class
weight of 3, then the label with the most raw votes wins.

First run, LUT4 256×128, K=8, 20 epochs, 10k train / 2k test:

| | Overall | Worst | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OVR, raw votes | 89.45% | 83.51% | 97.4 | 98.7 | 84.0 | 86.1 | 84.4 | 83.5 | 91.9 | 91.0 | 91.0 | 84.8 |
| shared trunk 512×3 | 92.50% | 89.45% | 96.3 | 97.4 | 90.5 | 91.6 | 92.8 | 92.8 | 92.4 | 91.4 | 89.4 | 89.8 |

The per-class pattern is the tell. 0/1/6/7/8 improve, and 8 reaches 91.0 —
better than the shared model — while 2/3/4/5/9 lose 4–8 points. Each yes/no
model is over-positive because of the class weight, and classes whose models
fire on more negatives win the argmax. Raw vote totals are not comparable
across models.

`--ovr-calibrate` fixes this by choosing, per class, the vote level that makes
the class fire on 10% of the training set (MNIST is balanced), and scoring
`votes - offset`. The calibrated run is pending.

## Gate simplification

Taxonomy of a hardened cell by argmax gate: constant (`0`,`15`) reads nothing;
identity (`12`→a, `10`→b) folds to a wire; inverter (`3`,`5`) keeps a cell with
half the fan-in; everything else is a real 2-input gate.

Folding plus dead-cell elimination, trained dense forests, uniform
`--score-scale 8`:

| Model | Cells | Dead | Folded | Inverters | Remaining |
| --- | --- | --- | --- | --- | --- |
| dense K=1 | 202 | 52 | 8 | 5 | 142 (70%) |
| dense K=4 | 232 | 23 | 25 | 15 | 184 (79%) |
| dense K=8 | 272 | 17 | 29 | 25 | 226 (83%) |
| dense K=16 | 352 | 8 | 53 | 62 | 291 (83%) |

`run.mjs --simplify` implements the fold for dense circuits and bit-checks it
against the original wiring on every eighth test sample. Conv models are
skipped by `--simplify`: folding a conv filter means rewiring taps, which is
not implemented.

## JS runner bug fixed

The JS forest targets were built with `tf.tile(oneHot, [1, K])`, which repeats
the ten columns instead of producing label-major blocks of K. The readout
reshapes to `[batch, 10, K]` (label-major), so the JS runner was training
against the wrong labels: loss fell while accuracy sat at chance. Targets are
now `reshape(tile(oneHot[..., None], [1,1,K]), [N, 10K])`. After the fix, a
400-sample / 6-epoch `dense` K=8 run reaches soft 62.0% / hard 58.5% and folds
272 → 171 cells (80 gates, 40 not, 51 wires, 0 const) with identical accuracy,
bit-checked on 25 samples. Every JS number recorded before this fix was invalid
and has been removed from the tables.

## Conv vs forest

| Model | K | Params | Gate cells | Soft | Hard | Unary-or-constant gates |
| --- | --- | --- | --- | --- | --- | --- |
| conv-grid | 1 | 87,488 | 4,842 | 45.2% | 37.1% | 60.2% |
| conv-grid | 4 | 91,808 | 4,872 | ~70% | ~58% | 32–53% |
| conv-shallow | 1 | 16,560 | 594 | 48.3% | 27.1% | – |
| conv-shallow | 4 | 20,880 | 624 | 52.5% | 25.0% | – |

The forest lifts `conv-grid` by ~20 points of hard accuracy, so the readout is
not what limits the conv nets. `conv-shallow` barely moves, because its global
OR leaves only 16 features for the head.

## Pending

- `conv-grid-head` full 20-epoch run never completed (Colab preemption); its
  visible trajectory was soft ~58–62% / hard ~29–36%, so it is not competitive
  with the dense forest regardless.
