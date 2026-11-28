# Gate-wise CNN on MNIST

An isolated experiment testing whether the project's two-input Boolean gate
cell can be used as a **convolutional feature detector**: the same gate/wire
distribution is reused at every spatial position, and multiple gates are
stacked per detector.

This does not touch the shipped symbol classifier or its WASM core. If the
experiment produces a useful result, the next step would be a separate demo
tab rather than a change to the existing one.

## What is being compared

All models use the same soft-gate semantics as `src/main.c`:

- each cell has two input pins;
- each pin is a softmax over its inputs (a local patch for convolution, the
  whole previous layer for dense);
- the cell has a softmax over the 16 Boolean gates;
- the output is the gate-probability-weighted mixture of the 16 truth values.

| Model | Detector | Head |
| --- | --- | --- |
| `dense` | none (fully connected) | 128 → 64 → 10×K gates |
| `dense-wide` | none (fully connected) | 256 → 128 → 10×K gates |
| `conv-shallow` | 8 filters @ 3×3, stride 2, then 16 filters @ 3×3, stride 2 | global OR → 64 → 64 → 10×K gates |
| `conv-deep` | two stacked 3×3 gates per detector, at two resolutions, with 2×2 OR pooling | global OR → 64 → 64 → 10×K gates |
| `conv-grid` | 16 filters @ 3×3 stride 1, 2×2 OR pool, 32 filters @ 3×3 stride 1, 2×2 OR pool | 4×4×32 → 64 → 64 → 10×K gates |
| `conv-grid-head` | same, then 64 filters @ 1×1 (shared across the 4×4 grid) | global OR → 64 → 10×K gates |
| `conv-deep-grid` | two stacked 3×3 gates per detector (12 then 24 filters), 2×2 OR pooling | 4×4×24 → 64 → 64 → 10×K gates |
| `dense-small` | none (fully connected) | 48 → 48 → 10×K gates |
| `dense-lut3` | none, 3-input LUT cell | width sweep → 10×K gates |
| `dense-lut4` | none, 4-input LUT cell | width sweep → 10×K gates |
| `dense-lut5` | none, 5-input LUT cell | width sweep → 10×K gates |

`conv-deep` is the "multi-layer per feature detector" variant: each detector
is a two-gate circuit before pooling. `conv-shallow` is the control with one
gate per detector.

`conv-grid` exists because the first conv variants pooled everything into a
single global OR (16 features) before the head, which throws away spatial
layout and starves the classifier. It keeps a 4×4 grid instead.

## Readout: forest, not argmax

The shipped symbol classifier does not pick the highest of 10 independent
output bits. It trains `circuitsPerLabel` circuits per label (8 by default)
and picks the label with the most circuit votes. The experiment mirrors that
with `--circuits-per-label K`: the head emits `10 × K` bits, K per label, and
the prediction is the label whose K bits sum highest, in both soft and hard
mode.

An argmax over a single bit per label ties very often (all-zero input predicts
label 0 for every sample), which is a large part of why the first runs'
hardened accuracy collapsed. `--circuits-per-label 1` keeps the original
one-bit readout for comparison.

Training optimizes the readout directly: the K values for each label are
averaged (and scaled by `--score-scale`) and a plain cross-entropy is taken
over the ten scores. A one-vs-rest BCE over `10 × K` individual bits was tried
first and is a trap here — with 9×K negatives the mean loss falls to ~0.06
while the vote decision stays near chance, because the negatives dominate the
average and the positives barely contribute.

`--score-scale` is a strong knob, not a formality: it sets the logit scale the
cross-entropy sees, and the loss normalises the summed votes by K so that the
scale does not grow with the forest size. On `dense` at K=4, raw sums
(effective scale 4) reach hard 68.5%, while the same model at effective scale
1 reaches hard 45.7%. Compare accuracies only within one scale setting.

Each model is trained once with soft gates. At the end the experiment hardens
each cell to its argmax wires and gate and evaluates the resulting
deterministic Boolean circuit. So the reported `soft` accuracy is the
trainable relaxation and `hard` accuracy is what would actually be compiled.

## Does a bigger forest help?

Yes, dramatically, and it also closes the soft→hard gap. `dense` and
`dense-small` (48→48 hidden) swept over `--circuits-per-label 1,4,8,16`, 20
epochs, forest cross-entropy over the raw vote sums:

| K | Model | Params | Gate cells | Soft | Hard | Gap |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `dense` | 71,072 | 202 | 62.3% | 49.2% | 13.2 |
| 1 | `dense-small` | 26,080 | 106 | 58.5% | 49.4% | 9.2 |
| 4 | `dense` | 75,392 | 232 | 76.8% | 68.5% | 8.3 |
| 4 | `dense-small` | 29,440 | 136 | 75.1% | 67.8% | 7.3 |
| 8 | `dense` | 81,152 | 272 | 83.7% | **79.2%** | 4.5 |
| 8 | `dense-small` | 33,920 | 176 | 84.2% | 75.9% | 8.3 |
| 16 | `dense` | 92,672 | 352 | 84.9% | 77.8% | 7.1 |
| 16 | `dense-small` | 42,880 | 256 | 76.7% | 62.1% | 14.6 |

These used raw vote sums (before the divide-by-K loss change). To reproduce
that scale with the current loss, pass `--score-scale` equal to K.

Two things stand out:

1. **The forest is the single largest win in the whole experiment.** Going
   from one circuit to eight takes `dense` from hard 49% to hard 79% for
   10k extra parameters. It beats every architectural change tried here,
   including the convolutional detectors.
2. **The forest also shrinks the soft→hard gap** (13.2 → 4.5 points). That
   is the mechanism you would hope for: hardening flips individual bits, and
   a vote over eight circuits tolerates some flipped bits where a single
   argmax over ten bits does not. K=16 overfits and gives the gain back.

`dense-small` tracks `dense` closely at K=4–8 and degrades much harder at
K=16, so the forest and the head width are partly interchangeable — the win
is capacity per label, not the vote itself.

## Does gate simplification help?

Three different things could be meant, and they have different answers.

### 1. Compressing the forest to fewer circuits: yes, if selection is greedy

This is the most natural reading of "forest followed by simplification", and
the answer depends entirely on how circuits are chosen.

Ranking circuits per label by their own balanced accuracy and keeping the best
m is a poor rule. For K=8 it gives 59.1 / 58.9 / 59.6 / 67.3 / 71.3 / 75.3 /
76.5 / **76.7%** for m = 1…8: the top three circuits are correlated, so the
vote sits flat until the fourth is added.

Greedy forward selection (`--greedy-curve`) instead scores each candidate by
its *marginal* contribution to vote accuracy on a held-out slice of training
data:

| Circuits kept (of 80) | 1 | 10 | 26 | 36 | 49 | 55 | 71 | 80 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Hard accuracy | 20.5 | 47.0 | 58.3 | 73.7 | 77.1 | **79.0** | 78.8 | 77.4 |

So **49 of 80 circuits (61%) reach the full-model accuracy**, and 55 circuits
slightly beat it at 79.0%, because greedy selection avoids stacking correlated
circuits and effectively builds a better-conditioned forest. Beyond ~55 the
curve plateaus and then drifts back down.

This is a genuine simplification win: about a third of the circuits can be
dropped without losing accuracy, provided the subset is chosen by marginal
value rather than individual score. It also says the forest is not
*arbitrarily* redundant — the first ~26 circuits are near-useless in
combination, and accuracy only takes off once enough diverse voters are in.
The 79.0% peak is within ~1 point of the full forest on 2,000 test samples, so
read it as "no loss", not as a real improvement.

For the shipped classifier this suggests a concrete change: `circuitsPerLabel:
8` is doing its job, but roughly 5–6 well-chosen circuits per label carry the
same accuracy, and a greedy subset would be a better 5–6 than the first 5–6.

Pruning circuits that are constant on the training set is still a no-op: 100%
of circuits were non-constant at K ≤ 8, and pruning never improved accuracy —
greedy selection is a different (and much better) criterion than constancy.

### 2. Folding gates: yes, ~18–29% of cells, exactly

A hardened cell's gate fixes how much of its wiring matters:

| Argmax gate | Reads | After folding |
| --- | --- | --- |
| `0`, `15` (constant) | nothing | folds to a literal, cell disappears |
| `12` (a), `10` (b) | one tap | folds to a wire, cell disappears |
| `3` (¬a), `5` (¬b) | one tap | stays a cell, but one input instead of two |
| any other | two taps | unchanged |

Folding plus dead-cell elimination on trained dense forests:

| Model | Cells | Dead | Folded | Inverters kept | Remaining |
| --- | --- | --- | --- | --- | --- |
| `dense`, K=1 | 202 | 52 | 8 | 5 | 142 (70%) |
| `dense`, K=4 | 232 | 23 | 25 | 15 | 184 (79%) |
| `dense`, K=8 | 272 | 17 | 29 | 25 | 226 (83%) |
| `dense`, K=16 | 352 | 8 | 53 | 62 | 291 (83%) |

The reduction is semantics-preserving, so it costs no accuracy. `run.mjs
--simplify` implements the fold for dense circuits and cross-checks it
bit-for-bit against the original wiring on every eighth test sample; the
folded and unfolded circuits agree on every checked sample in every run.

The reachability column is the interesting part and it *shrinks with the
forest*: at K=1, 52 of 202 cells are unreachable from the output; at K=16 only
8 of 352 are. A hardened gate reads exactly two wires, so ten output cells can
reference at most twenty hidden cells — with a bigger forest the output layer
touches much more of the hidden layer, and less of the trained circuit is
wasted.

### 3. Gate choice is mostly degenerate

32–60% of cells use a gate that ignores at least one wire (`0, 3, 5, 10, 12,
15`). For the convolutional detectors this is the sharpest diagnostic: a
conv filter with a constant gate emits a constant plane, and one with an
identity gate is a shifted copy of a single input pixel, so a nominal 3×3
detector is effectively reading one or two of its nine taps. `conv-grid`
reports 60.2% at K=1.

## Gate budget: is a few hundred cells actually a lot?

No — it is remarkably little, and that reframes the comparison. A cell in the
hardened circuit is one two-input Boolean gate. An IEEE-754 binary32
multiply-add is not one gate; it is a small datapath:

| FP32 FMA component | Rough gate count |
| --- | --- |
| 24×24 significand multiplier (576 partial products + ~550 full adders) | 2,700–5,000 |
| alignment shift + exponent difference | 700–1,000 |
| 24-bit adder | 300–500 |
| leading-zero count + normalization shift | 800–1,200 |
| round-to-nearest-even (guard/round/sticky) | 200–400 |
| exponent/sign/exception logic | 200–400 |

So one FP32 multiply-add is on the order of **5,000–10,000 two-input gates**
(depending on the multiplier and rounding implementation; these are
order-of-magnitude figures, not a synthesis result). Against that:

| Circuit | Hard gate cells | vs one FP32 FMA |
| --- | --- | --- |
| `dense`, K=8 | 272 | ~3–5% |
| `dense`, K=16 | 352 | ~4–7% |
| `dense` 512×256, layer 0 pinned | 848 | ~8–17% |
| `conv-grid`, K=4 | 4,872 | ~0.5–1 FMA |

And the float baselines, counted the same way, are enormous:

| Model | Params | Gate equivalent at 5k–10k gates/MAC |
| --- | --- | --- |
| `float-cnn` reference | 1,418 | ~7–14 million |
| `dense` soft model | 81,152 | ~400–800 million |

That is the honest version of your point: **the hardened dense forest is
~25,000–50,000× smaller in gate count than even a 1,418-parameter float CNN**,
while reaching 83.5% at 464 cells (78.7% at 272) rather than ~99% on a
quantized 14×14 MNIST. The deployed artifact is also small in storage: 272
cells store two wire indices (8 bits max at fan-in 196) plus a 4-bit gate,
≈ 620 bytes, versus ~317 KB for the 81k-parameter soft model, a ~500×
reduction. The 81k parameters are *training-time* configuration; almost none
of that information survives hardening.

Three caveats keep this from being a free win:

1. **Accuracy is not a gate-count contest.** 79% at 272 gates and 99% at
   millions of gates are both "efficient" in their own regime; the curves do
   not cross in the data we have.
2. **Input bandwidth flatters the gate model.** It consumes 196 thresholded
   bits, while the float CNN consumes 784 float32 values (~25 kbit). Some of
   the efficiency is the input format, not the gates.
3. **Neither gates nor data is the binding constraint — hardening is.** At
   10k training samples, hard accuracy plateaus around 83.5% once the forest
   passes ~460 cells, and 3.5× more gates buys nothing:

| Hidden | Params | Gate cells | Soft | Hard |
| --- | --- | --- | --- | --- |
| 128×64 | 81,152 | 272 | 82.5% | 78.7% |
| **256×128** | 193,792 | **464** | 84.8% | **83.5%** |
| 512×256 | 517,376 | 848 | 86.0% | 80.9% |
| 768×384 | 972,032 | 1,232 | 86.1% | 83.6% |
| 1024×512 | 1,557,760 | 1,616 | 86.3% | 81.8% |

   Soft accuracy keeps climbing with width; hard accuracy does not. The extra
   capacity is spent on graded features that do not survive hardening, which
   is why the soft→hard gap widens from 1.3 to 4.5 points across the sweep.
   (Single seed per row; run-to-run spread is ~2 points.)

   More data does not lift the plateau either. The same 464-cell model trained
   on 50,000 samples instead of 10,000 reaches **soft 86.8% / hard 82.1%**
   (peak hard 83.6% during training), versus 84.8% / 83.5% on 10k. The soft
   model gains three points from the extra data; the hard circuit gains none.

   So the hard circuit is limited by the hardening step and by its restricted
   function class — two-input gates over two selected wires — not by the
   number of gates or the amount of data. The depth section below pins the
   mechanism down further: what actually caps accuracy is **input coverage**,
   how many of the 196 pixels the hard wiring can ever see. Pinning layer 0 to
   cover all of them lifts hard accuracy to 84.2% at that shape, and the
   constant-width depth-5 model below reaches 86.2%.

Fan-out and routing are a real but second-order cost: a gate array still has
to buffer and route each cell's two inputs, which in a naive layout can add
one to three gate equivalents per connection. Even at 3× that, 272 cells is
under a thousand gate equivalents — still a small fraction of one FP32 FMA.

### The part that argues against the conv variant

If gates are the scarce resource, weight sharing does not help, because a
shared filter still has to be *instantiated at every output position* to
produce a hard circuit. `conv-grid`'s first layer alone is 16 filters × 196
positions = 3,136 gate cells; the dense layer that replaces it is 128 gates
and sees all 196 inputs.

Sharing reduces the number of distinct gates you must *configure* (the 16-way
LUT per cell), not the number of gates you must *build*. A hard cell's fan-in
is nearly free in gate count: reading 2 of 196 inputs costs the same one gate
as reading 2 of 2, because the wiring is fixed at compile time. So the
gate-efficient topology is dense all-to-all, and convolution is
gate-expensive by construction.

This is the same conclusion the accuracy tables reached, but from the opposite
direction: the conv detectors were never going to win on gate budget either.

## Depth and the two-input bottleneck

A cell reads exactly two wires, so the set of inputs an output bit can depend
on at most **doubles per layer**. With 196 input pixels, depth 3 reaches at
most 8, and you need 8+ layers for the whole image. That bound is real and
measurable: I propagated the actual argmax wiring through each layer and
counted the distinct input bits reachable from each output.

Constant 464 cells, K=8, learned wires:

| Layers | Shape | Per-output mean / max | Inputs seen by any output | Soft | Hard |
| --- | --- | --- | --- | --- | --- |
| 3 | 256,128 | 7.1 / 8 | 85 | 86.5% | **80.7%** |
| 4 | 128,128,128 | 10.1 / 16 | 67 | 82.3% | 70.0% |
| 5 | 96×4 | 13.2 / 21 | 42 | 75.3% | 58.0% |
| 7 | 64×6 | 34.7 / 48 | 57 | 76.8% | 56.0% |
| 9 | 48×8 | 44.1 / 52 | 52 | 57.5% | 43.9% |
| 13 | 32×12 | 27.0 / 27 | 27 | 37.7% | 23.0% |

The per-output receptive field grows with depth exactly as the 2^L bound
predicts — 8, 16, 21, 48, 52 — yet accuracy *falls*. The reason is the last
column: depth is being bought by narrowing the first layer, and the first
layer is the gatekeeper. A pixel that no layer-0 wire selects is invisible at
every depth.

Widening the first layer helps coverage but not enough, because the learned
wire softmaxes collapse onto favourite pixels:

| Shape | Layer-0 coverage | Final coverage | Hard |
| --- | --- | --- | --- |
| 512,256 | 120 | 89 | 81.3% |
| 512,256,128 | 124 | 67 | 69.1% |
| 512,256,128,128 | 101 | 67 | 71.9% |
| 512,128×4 | 181 | 105 | 58.3% |
| 768,256,128 | 155 | 74 | 77.6% |

512 first-layer cells give 1,024 wire slots but cover only 120 of 196 pixels;
768 cells cover 155. Each further layer prunes coverage again. So depth-3 stays
best, not because depth is bad, but because coverage collapses faster than
depth adds reach.

### Fixing coverage changes the depth result

`--freeze-wires` pins every wire to a shuffled round-robin cover, so each layer
samples the whole previous layer and coverage cannot collapse:

| Shape | Layers | Layer-0 → final coverage | Soft | Hard |
| --- | --- | --- | --- | --- |
| 256,128 | 3 | 196 → 179 | 68.5% | 65.0% |
| 512,256 | 3 | 196 → 187 | 67.3% | 65.4% |
| 512,256,128 | 4 | 196 → 194 | 76.5% | 73.4% |
| 512,256,128,128 | 5 | 196 → 194 | 76.7% | **74.4%** |

Now depth is monotone: 65.0% → 73.4% → 74.4%. The 2-input-pin bottleneck is
real, and removing the coverage collapse lets depth do its job.

But random wires are weaker features than learned ones: 74.4% at depth 5 is
still below the 80.7% that learned wires reach at depth 3. So the tension is:

- **learned wires** → good features, but coverage collapses and caps depth;
- **random wires** → full coverage and depth scales, but weak features.

The natural next step is a hybrid or a coverage regulariser. `--freeze-first-wires`
pins only layer 0, which guarantees every pixel is reachable while deeper
wires stay trainable:

| Shape | Layers | Layer-0 → final coverage | Soft | Hard |
| --- | --- | --- | --- | --- |
| **512,256** | **3** | **196 → 154** | 86.9% | **84.2%** |
| 512,256,128 | 4 | 196 → 135 | 86.0% | 81.5% |
| 512,256,128,128 | 5 | 196 → 97 | 82.6% | 73.4% |

Forcing layer-0 coverage is worth roughly +3.5 points over learned layer-0
wiring at the same shape (80.7% → 84.2%).

But depth still loses past 3 layers, and the coverage column shows why: even
with layer 0 at 196, learned deeper layers prune the reachable set back down
(196 → 154 → 135 → 97). Depth only pays off when *every* layer keeps coverage,
which is why the all-frozen model improves monotonically and this one does not.

### Pinning a fraction of every layer

`--pin-fraction` pins that share of each layer's wires to a round-robin cover
and re-imposes them after every optimiser step, leaving the rest trainable.
At depth 5 (512,256,128,128, 1,104 cells):

| Pinned | Soft | Hard | Inputs ever seen |
| --- | --- | --- | --- |
| 0 | 81.9% | 65.4% | 92 |
| 0.25 | 82.5% | 77.6% | 149 |
| 0.50 | 85.5% | 79.9% | 175 |
| **0.75** | 85.7% | **83.5%** | 191 |
| 1.00 | 77.4% | 74.1% | 196 |

This is the cleanest single result in the depth work: at a fixed shape and
gate count, hard accuracy moves **65.4% → 83.5%** purely as a function of how
much of the wiring is pinned. The optimum is 0.75 — enough coverage to keep
reach high (191/196), but 25% free wires to learn useful features. Pinning
everything loses the learned features (74.1%); pinning nothing collapses reach.

With 75% pinned, depth 5 (83.5%) now matches the depth-3 best (84.2%), so the
2-input-pin limit is no longer what stops depth — wiring coverage is.

### Depth at a fixed pin fraction, funnel shapes

Holding the pin fraction at 0.75 and narrowing each layer:

| Shape | Depth | Cells | Soft | Hard | Inputs ever seen |
| --- | --- | --- | --- | --- | --- |
| 256,128 | 3 | 464 | 85.5% | 82.1% | 176 |
| **512,256** | **3** | **848** | 85.6% | **84.5%** | 179 |
| 512,256,128 | 4 | 976 | 85.2% | 83.9% | 185 |
| 512,256,128,128 | 5 | 1,104 | 85.2% | 83.5% | 185 |

With a funnel, depth is flat then slightly down. That is not because depth is
useless — it is because narrowing removes the candidates a 2-pin cell needs.

### Constant width: depth does pay

Keeping every hidden layer at 512 instead:

| Shape | Depth | Cells | Soft | Hard | Post-fold cells |
| --- | --- | --- | --- | --- | --- |
| 512,512 | 3 | 848 | 85.6% | 84.5% | 645 (76%) |
| 512×3 | 4 | 1,616 | 87.2% | 85.9% | 758 (47%) |
| **512×4** | **5** | **2,128** | 87.3% | **86.2%** | **877 (41%)** |
| 512×5 | 6 | 2,640 | 87.0% | 84.4% | 921 (35%) |
| 512×6 | 7 | 3,152 | 82.5% | 78.0% | 909 (29%) |

This is the answer to "does more depth buy accuracy": **yes, when the width is
held constant**, and it is the largest gain from architecture in the
experiment — 84.5% → 86.2%. Compare the funnel at the same depth
(512,256,128,128, 83.5%): constant width is worth +2.7 points. It also keeps
input coverage at 196/196 in every layer.

Depths 4–6 are within ~2 points (85.9 / 86.2 / 84.4), so the peak is broad
rather than a lucky depth. At depth 7 it collapses to 78.0%, and soft accuracy
falls too — that is overfitting (3M parameters on 10k samples), not coverage.

### Width at constant depth (four hidden layers, pin 0.75)

| Width | Cells | Params | Soft | Hard | Post-fold cells |
| --- | --- | --- | --- | --- | --- |
| 512 | 2,128 | 1.9M | 87.3% | 86.2% | 877 (41%) |
| **768** | **3,152** | **4.0M** | 88.1% | **87.0%** | **1,131 (36%)** |
| 1024 | 4,176 | 6.9M | 87.1% | 86.1% | 1,367 (33%) |

Width peaks at 768 and then overfits: 1024 is 0.9 points worse and adds 236
post-fold cells. Widening buys at most +0.8 points, so it is not the route to
90%. The input-resolution and data axes are tested separately below.

### Depth + simplification

Folding the hardened constant-width circuit is very effective, because a wide
layer leaves most cells unreachable when each cell reads only two wires:

| Shape | Cells | After folding/dead removal | Hard |
| --- | --- | --- | --- |
| 512,512 (d3) | 848 | 645 (76%) | 84.5% |
| **512×3 (d4)** | 1,616 | **758 (47%)** | **85.9%** |
| **512×4 (d5)** | 2,128 | **877 (41%)** | **86.2%** |
| 512×5 (d6) | 2,640 | 921 (35%) | 84.4% |
| 512,256,128,128 (funnel d5) | 1,104 | 712 (64%) | 83.5% |
| funnel d8 | 1,616 | ~60% | ~63% |

So "train deeper, then simplify" does work — but only at constant width, and
only to a point. The best compiled circuits found:

- **877 cells at 86.2%** (peak accuracy, depth 5);
- **758 cells at 85.9%** (best accuracy-per-gate, depth 4);
- **645 cells at 84.5%** (cheapest, depth 3).

The returns diminish sharply: +113 cells buys +1.4 points from d3→d4, but
+119 more buys only +0.3 from d4→d5, and beyond that it goes backwards.

Summary of the depth finding:

- the 2-input-pin limit is real — an output's reach can at most double per
  layer, so depth is necessary for anything beyond ~8 pixels;
- depth pays only if every layer keeps both coverage *and* width; a funnel
  starves the 2-pin mixers and erases the gain;
- constant-width depth 4–5 (three to four hidden layers) is the best
  configuration found: 85.9–86.2% hard, folding to 758–877 cells;
- past that, parameters outrun 10k samples and accuracy falls (78% at depth 7);
- pinning ~75% of wires keeps coverage at 196/196 while leaving enough free
  wires to learn features.

## Tri-logic cells

A different way to widen the per-cell function class without adding pins: let
signals take three values, `{0, 0.5, 1}`, and let each cell implement **any**
map `{0,0.5,1}² → {0,0.5,1}`. That is `3^9 = 19,683` functions instead of 16,
still with two pins.

The full categorical is not needed. A tri-gate is exactly its 3×3 table, so
storing 9 entries with a 3-way softmax each — **27 logits per cell**, versus 16
for a binary cell — covers all 19,683 gates exactly. Soft mode interpolates the
table bilinearly at the continuous pin values:

```
q[i,j] = Σ_v softmax(table[i,j,:])[v] · v        # expected table entry
a, b   = Σ_j wire_a[j]·x[j], Σ_j wire_b[j]·x[j]  # expected pin values
out    = Σ_{i,j} hat_i(a) · hat_j(b) · q[i,j]
```

Hard mode argmaxes each table entry and each pin, then looks the gate up at the
ternary pin values. The trade-off is that a ternary wire costs ~1.6 bits
(`log2 3`) rather than 1, so a tri-cell is roughly 1.6–2 binary gates in
hardware even though its soft parameter count is only 1.7× a binary cell.

Implemented as `dense-tri` (`--models dense-tri`), same depth/width/pinning
machinery. First result, 256×128, K=8, pin 0.75:

| Model | Soft | Hard |
| --- | --- | --- |
| `dense-tri`, sum readout | 83.4% | 71.1% |
| `dense-tri`, count of entries ≥ 0.75 | 83.4% | 69.6% |
| `dense` (binary), same shape | 82.1% | 82.1% |

Tri-logic buys soft expressiveness and loses hard accuracy. The readout is not
the cause — thresholding made it slightly worse. Hardening a ternary circuit
quantises at three levels instead of one (pins, table entries, and every
intermediate signal), so the soft→hard gap is 12–20 points instead of 1–3.
It is a negative result as implemented: the extra function class does not
survive hardening without a hardening-aware ternary training scheme.

## Three-input LUT cells

If the gate is allowed three input pins instead of two, the binary gate space
becomes `2^(2^3) = 256` functions, and an output's reach triples per layer
(`3^L`) instead of doubling. That is the FPGA `LUT3` primitive.

It is also *cheaper* to parameterise than the two-input cell: a 3-input gate is
exactly its 8-bit truth table, so 8 Bernoulli logits per cell span all 256
gates, versus 16 logits for the 16 two-input gates. Soft mode interpolates the
expected table trilinearly at the continuous pin values; hard mode argmaxes the
three pins, rounds the 8 bits, and looks the gate up.

Two implementation notes:

- the table must be initialised symmetrically. `random_logits` returns
  `log(weights)`, which is negative for every entry, so `table > 0` starts
  false and every gate is constant 0 (100% all-zero outputs). A small `randn`
  fixes it;
- a LUT3 is not a hardware commitment: every 3-input function lowers to
  two-input gates, and the simplification pass then applies as usual.

The lowering cost is exact and cheap to compute. Breadth-first over all 256
three-input functions, using the project's 16 two-input gates:

| Two-input gates needed | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Number of LUT3 functions | 5 | 33 | 58 | 136 | 9 | 10 | 5 |

Mean **2.63** gates per LUT3 cell (worst case 6; the five 0-cost functions are
the constants and the three bare inputs, which fold away entirely). The runner
reports the actual compiled cost of the trained tables rather than this
average, so the hardware story stays exact.

LUT4 is much more expensive to lower. A one-variable Shannon split
(`f = mux(a, g, h)` with `g,h` LUT3s) gives a mean upper bound of **8.04**
two-input gates per LUT4 cell (worst case 15); better synthesis might reach
~6.5–7, but the ratio holds:

| Gate | Functions | Mean 2-input gates | Worst |
| --- | --- | --- | --- |
| LUT3 | 256 | 2.63 (exact) | 6 |
| LUT4 | 65,536 | 8.04 (bound) | 15 |

So LUT4 wins on *cells* but loses on *compiled gates*. Measured on trained
tables, a LUT4 cell averages **5.04** two-input gates (not the 8.04 uniform
bound), so LUT4 512×3 is ~8,150 gates; LUT3 512×3 is ~3,800. If the currency
is gates, LUT3 is the sweet spot.

LUT4 does hold the accuracy record — **92.25% hard at 512×3, 1,616 cells** —
but the last 0.7 points over LUT3 cost about 2× the hardware. Widening to
768×3 gives 92.1% at 2,384 cells, so width is exhausted.

A LUT5 cell is the next and last step: 32 pins-worth of truth table
(`2^32` functions), reach `5^L`. It is implemented as `dense-lut5`, but the
lowering cost is high — a one-variable Shannon split needs a LUT4 mux per
level, so a cell is roughly 2–3× a LUT4 cell in two-input gates. It is only
worth it if the accuracy jump clears that cost.

Implemented as `dense-lut3`. Constant width 512, K=8, 75% pinned:

| Shape | Depth | Cells | Soft | Hard |
| --- | --- | --- | --- | --- |
| 256×128 | 3 | 464 | 89.2% | 88.2% |
| 512×512 | 3 | 1,104 | 90.8% | 90.0% |
| **512×512×512** | **4** | **1,616** | **91.9%** | **91.6%** |
| 512×512×512×512 | 5 | 2,128 | 91.4% | 90.8% |

This clears the 90% target: **91.6% hard at 1,616 LUT3 cells**, versus the best
2-input circuit's 87.0% at 3,152 cells. Depth 4 is the peak; depth 5 gives a
little back. The LUT3 model also hardens well — soft 91.9 → hard 91.6, a 0.3
point gap.

## Multi-scale input pyramid

`--pyramid` gives the first layer every coarser scale of the image, not just the
finest one: 14, 7, 4, 2, 1 upsampled back to 14×14 and stacked as channels, so
980 inputs instead of 196. This is cheap in gates (a LUT3 cell still costs ~2.6
two-input gates) and expensive only in first-layer parameters and wiring. It
lets a shallow LUT circuit see coarse blob statistics that local taps cannot
synthesise without depth.

## The soft-to-hard gap

The first runs exposed the central difficulty. A cell's output is the
gate-probability-weighted mixture of the 16 truth values. If the wires are
softmax mixtures over many inputs the output is graded, and the downstream
layers learn from those graded values. Argmax hardening replaces each of them
with one wire and one gate, so the Boolean circuit no longer computes the
function the soft model learned.

Observed on 10k train / 2k test, 14×14 binary inputs, with the original
one-bit-per-label readout (`--circuits-per-label 1`):

| Model | Soft | Hard (pre-STE) | All-zero outputs |
| --- | --- | --- | --- |
| `dense` | 88.8% | 12.8% | 95.5% |
| `conv-shallow` | 63.8% | 22.8% | 50.0% |

With temperature annealing to 0.25 (see below) the gap stayed open and became
suspiciously uniform across architectures:

| Model | Soft @ T=0.25 | Hard | All-zero outputs |
| --- | --- | --- | --- |
| `dense` | 81.2% | 27.3% | 75.1% |
| `conv-shallow` | 48.3% | 27.1% | 13.6% |
| `conv-deep` | 35.0% | 28.1% | 0.0% |

Every architecture lands at 27–28% hard accuracy even though their soft
accuracies and all-zero rates differ wildly. `conv-deep` never emits an
all-zero output, so its bits do carry information; the one-bit-per-label
argmax is what discards it. That is the readout problem described above, and
it is why the forest readout is the next thing to test.

With the forest readout (`--circuits-per-label 4`) and cross-entropy over the
summed votes, the collapse largely goes away for the dense forest:

| Model | Params | Gate cells | Soft | Hard |
| --- | --- | --- | --- | --- |
| `dense` | 75,392 | 232 | 74.3% | 63.2% |
| `conv-shallow` | 20,880 | 624 | 52.5% | 25.0% |
| `conv-deep` | 27,024 | 4,872 | 46.0% | 31.8% |

The collapse is gone, but the gap is still real: at K=4 `dense` loses about
11 points to hardening (74.3 → 63.2). Growing the forest closes most of it —
see the K sweep above, where K=8 cuts the gap to 4.5 points. The conv variants
are limited by their global-OR pooling (only 16 features reach the head), which
is what `conv-grid` addresses.

An abrupt straight-through fine-tune at lr 0.1 made things worse (`dense`
collapsed to 13.9%), because the hard forward pass produced mostly-constant
outputs and therefore zero gradient. A second bug compounded it: the original
BCE loss clamped Boolean 0/1 outputs to `[1e-6, 1-1e-6]`, and `clamp` has zero
derivative outside its interval, so the STE gradient was exactly zero. The
loss is now a cross-entropy over the per-label vote, which has no clamp and
optimizes the quantity the readout actually compares.

That is why the runner also supports training the hard circuit directly: pass
`--ste-epochs` equal to `--epochs` (or more generally a large value). Forward
is then the argmax circuit at every step, and gradients come from the
straight-through estimator. `soft` and `hard` accuracy should coincide exactly
in that mode, which makes it the honest measurement of what the gate circuit
can learn.

In practice, a short STE fine-tune at the annealed operating point does
nothing: `dense` went from hard 62.9% before STE to hard 62.9% after four
epochs, with the loss flat at ~1.15. At T=0.25 the softmax is saturated, so
the straight-through gradient is too small to flip any argmax wire or gate.
The remaining soft→hard gap is the cost of replacing graded values with
binary ones; it is not a training artefact. Earlier reports of STE
"collapsing" the model were caused by the `clamp` bug above.

Two distinct numbers are reported by the PyTorch runner:

- `soft` — accuracy at the annealed soft operating point (before STE),
- `hard-pre` — argmax circuit before STE,
- `hard-post` — argmax circuit after the STE fine-tune.

## Data

MNIST is downloaded from the standard `storage.googleapis.com` mirror and
cached under `data/`. Images are block-averaged from 28×28 to the configured
size (14×14 by default) and thresholded to Boolean inputs.

## Run

The JS runner is the reference implementation and runs entirely locally:

```bash
cd apps/only-gates-ai
node experiments/gate-cnn-mnist/run.mjs \
  --train 10000 --test 2000 --epochs 8 \
  --models dense,conv-shallow,conv-deep
```

For faster iteration, the PyTorch port runs on a Colab VM:

```bash
colab run --gpu T4 --timeout 1800 \
  apps/only-gates-ai/experiments/gate-cnn-mnist/run_colab.py \
  --train 10000 --test 2000 --epochs 20 --ste-epochs 5
```

`T4` is the correct Colab accelerator name. If the GPU session cannot be
created, omit `--gpu T4`; the CPU runtime is still fast for this model.

Useful options for both runners: `--size`, `--threshold`, `--batch`, `--lr`,
`--seed` (and `--data`, `--out` for the JS runner). The PyTorch runner adds:

- `--t-start`, `--t-min` — geometric temperature anneal of the gate and wire
  softmaxes (default 1.0 → 0.25);
- `--ste-epochs`, `--ste-lr` — straight-through fine-tune of the argmax
  circuit at the end of training. Set `--ste-epochs` equal to `--epochs` to
  train the hard circuit from the start instead.
- `--circuits-per-label` — forest size at the readout (default 1). Accepts a
  comma-separated sweep, e.g. `1,4,8,16`.
- `--dense-hidden` — hidden sizes for the `dense` model (default `128,64`).
- `--width-sweep` — semicolon-separated dense shapes to sweep, e.g.
  `128x64;256x128;512x256x128`, to trace accuracy against gate count and depth.
- `--freeze-wires` — pin every wire to a shuffled round-robin cover and train
  only the gates; keeps hard-circuit input coverage at 100%.
- `--freeze-first-wires` — pin only layer 0 (guarantees every input pixel is
  reachable) while deeper wires train normally.
- `--pin-fraction` / `--pin-sweep` — pin a fraction of every layer's wires to a
  round-robin cover (re-imposed after each step) and train the rest, e.g.
  `--pin-sweep 0,0.25,0.5,0.75,1`. This is the coverage-vs-features knob.
- `--augment` (PyTorch only) — random rotation/scale/translation on the raw
  28×28 image before pooling and thresholding, regenerated every batch.
  `--augment-rotation` (degrees, default 10), `--augment-scale` (default 0.1)
  and `--augment-translate` (default 0.1) control the ranges.
- `--one-vs-rest` — train ten independent yes/no forests (one per digit)
  instead of one shared-trunk 10-class forest, then vote. `--ovr-pos-weight`
  (default 3) weights the positive class. `--ovr-calibrate` scores each test
  example by its per-class log-likelihood ratio over the vote count, fitted on
  the training votes, because a raw vote total is not comparable across
  classes.
- `--class-dro <eta>` — group-DRO step on the per-class training loss. `0.2`
  rebalances toward weak digits and improved both mean and worst-class recall
  in testing; `1.0` destabilises training. Off by default.
- `--stacked-readout linear|gate` — after hardening, retrain a combiner on the
  `10×K` forest bits. `linear` is a weighted vote; `gate` is a small second
  gate layer (`--stacked-hidden`, default 32) that can learn non-additive
  combinations. Both stay Boolean at inference (the gate variant adds only
  `hidden + 10` cells).
- `--prune-circuits` — drop circuits that are constant on the training set
  before voting, and report the accuracy without them.
- `--prune-curve` — rank circuits per label by balanced accuracy and report
  hard accuracy for the forest truncated to the best 1..K circuits. This is
  the "can the forest be compressed?" measurement.
- `--greedy-curve` — greedy forward selection of circuits by marginal vote
  contribution on a held-out slice of training data, reporting accuracy as
  circuits are added.
- `--score-scale` — multiplier on the vote scores before the cross-entropy.
  The loss divides the summed votes by K first, so the logit range does not
  grow with the forest size.
- `--mask-constants` — forbid gates `0` (`zero`) and `15` (`one`), which
  otherwise dominate the argmax and turn layers into constants.

The JS runner accepts the same `--circuits-per-label`, `--mask-constants`,
`--t-start` and `--t-min` options, so the two implementations stay comparable.
It does not implement straight-through training; use the PyTorch runner for
`--ste-epochs`. It does add `--simplify`, which folds the hardened dense
circuit (constants to literals, identity gates to wires), drops unreachable
cells, reports the reduced cell count, and checks the folded circuit against
the original wiring on every eighth test sample.

Note: the JS forest targets were label-interleaved rather than label-major
until recently, so the JS runner trained against the wrong labels and every JS
result before that fix was invalid. Both runners now use the same label-major
target layout and the same `i // K` readout grouping.

Results are written to `out/results.json` and printed as a table. `data/` and
`out/` are ignored by git.

## What counts as a win

A useful result is not just higher accuracy than the dense gate baseline. The
interesting comparison is:

1. the convolutional detector beats the dense forest at comparable parameter
   count,
2. the advantage survives hardening, and
3. stacking a second gate per detector improves over the single-gate detector
   enough to justify the extra gate count.

If soft accuracy is good but hard accuracy collapses, the wiring/gate
distributions are not confident enough to harden, and the demo would need
hardening-aware training before it is worth building.

## Verdict so far

None of the convolutional detectors satisfy (1). At equal or larger cell
counts they lose to a plain dense forest:

| | Params | Gate cells | Soft | Hard | Post-fold |
| --- | --- | --- | --- | --- | --- |
| `lut4` 512×3, pin 0.75, K=8 | 2,688,256 | 1,616 | 92.5% | **92.3%** | – (~13k 2-in gates) |
| `lut3` 512×3, pin 0.75, K=8 | 2,009,728 | 1,616 | 91.9% | **91.6%** | – (~4.3k 2-in gates) |
| `dense` 768×4, pin 0.75, K=8 | 4,013,312 | 3,152 | 88.1% | 87.0% | 1,131 |
| `dense` 512×4, pin 0.75, K=8 | 1,889,536 | 2,128 | 87.3% | 86.2% | 877 |
| `dense` 512×3, pin 0.75, K=8 | 1,357,056 | 1,616 | 87.2% | 85.9% | 758 |
| `dense` 512×2, pin 0.75, K=8 | 517,376 | 848 | 85.6% | 84.5% | 645 |
| `dense` 256×128, K=8 | 193,792 | 464 | 84.8% | 83.5% | 380 |
| `dense` 128×64, K=8 | 81,152 | 272 | 82.5% | 78.7% | 224 |
| `dense-wide`, K=4 | 182,912 | 424 | 75.5% | 70.8% | – |
| `dense-small`, K=8 | 33,920 | 176 | 84.2% | 75.9% | – |
| `conv-grid`, K=1 | 87,488 | 4,842 | 45.2% | 37.1% |
| `conv-grid`, K=4 | 91,808 | 4,872 | ~70% | ~58% |
| `conv-grid-head`, K=4 | 30,368 | 5,832 | ~58% | ~34% |
| `conv-shallow`, K=4 | 20,880 | 624 | 52.5% | 25.0% |
| `conv-deep`, K=4 | 27,024 | 4,872 | 46.0% | 31.8% |

Rows come from different runs and score-scale settings, which is worth about
two points; the gaps shown are much larger than that.

The shared-detector idea *works* — `conv-grid` does learn a usable convolution
and its hard circuit keeps most of its soft accuracy — but it needs 18× the
cells to land 20 points lower. Two measured reasons:

- **The forest helps the conv nets too, but they start too far behind.**
  `conv-grid` goes from hard 37.1% at K=1 to ~58% at K=4 for the same
  parameter budget, so the readout is not the problem. What the forest cannot
  fix is a detector that is not carrying enough signal: `conv-shallow`, whose
  global OR leaves only 16 features for the head, barely moves (27.1% → 25.0%).
- **Most shared cells are routing, not detecting.** 60.2% of `conv-grid`'s
  cells at K=1 use a gate that ignores at least one wire (32–53% for the other
  models), so a nominal 3×3 detector reads one or two of its nine taps.
  Folding the trained dense forest, by comparison, only removes 18–29% of
  cells — the conv stack is markedly more degenerate than the dense one.

So: no separate demo tab for a gate-CNN. The honest result is that a
moderately sized dense gate forest is the better circuit on MNIST at this
input resolution, and the forest readout — not the convolution — is what
makes hardening survivable.

A dense-LUT3 tab is a different matter. At 91.6% hard from 1,616 cells
(~4,250 two-input gates after lowering), it is a genuinely interesting
artifact: eight circuits per label, one FP32-FMA's worth of logic, and a
0.3-point soft→hard gap. If a demo is wanted, that is the model to build, not
the convolutional one.

The gate-budget framing strengthens rather than weakens that conclusion. The
best accuracy found is **94.20% from 2,128 LUT4 cells** (512×4 at 28×28, 50k
training, augmentation, DRO 0.2, ~10,400 two-input gates); the gate-efficient
point is **91.6% from 1,616 LUT3 cells**. Lowered to
two-input logic, the LUT3 circuit is ~4,250 gates — less than one FP32
multiply-add (5,000–10,000) — and ~1,700–3,300× below a 1,418-parameter float
CNN. Raising the primitive from a 2-input cell to a 3-input LUT is what broke
the ~87% ceiling. The remaining levers were tested, not assumed: full 50k
training plus augmentation (+0.97), full-resolution 28×28 input (+0.97), a
gentle class-DRO (+0.5 mean, +2.9 worst class), one-vs-rest forests (−3), and
5-input LUT cells (−0.85). The final ceiling is a soft accuracy of 95.2%, so
97% would need a different architecture, not another tuning pass. And the budget is certainly not what limits the
conv variants: they spend 4,872 cells to reach 58%, i.e. 6× the gates for a
much worse error rate. Weight sharing saves *parameters you must configure*,
not *gates you must build*, so it is the wrong tool when gates are the
currency.

The follow-up question ("forest, then simplify the circuit") lands positive,
with a caveat. Greedy selection by marginal vote contribution keeps 49 of 80
circuits (61%) at full hard accuracy, and 55 circuits slightly beat the full
forest (79.0% vs 77.4%) by avoiding correlated voters. But choosing circuits
by their individual accuracy does almost nothing, and dropping circuits that
are merely constant is a no-op. Separately, folding constant and identity
gates plus unreachable cells is exact and removes 17–30% of a trained dense
forest — a compile-size win, not an accuracy win.

The simplification measurement is still worth keeping, because it quantifies
the compiled size of whatever circuit is shipped.

One retroactive result: this is a validation of the shipped classifier's
`circuitsPerLabel: 8` plus vote readout. Eight circuits was the best setting
tested here, and the vote — not any architectural trick — is what kept the
hardened circuit within ~5 points of the soft model. The experiment's failure
is specifically about using shared convolutional detectors instead of the
shipped dense circuit shape, not about the forest idea.
