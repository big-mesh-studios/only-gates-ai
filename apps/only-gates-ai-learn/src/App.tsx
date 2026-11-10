import { For } from "solid-js";

import {
  BlockDiagram,
  FpgaDiagram,
  LoweringDiagram,
  PrecisionDiagram,
  QuantizationDiagram,
  TrainingInferenceDiagram,
} from "./diagrams";
import {
  COLAB_FIRST,
  COLAB_NEXT,
  CONFIGS,
  FAQ,
  MISSIONS,
  NAV,
  PRECISION,
} from "./content";

const demoHref = import.meta.env.DEV ? "http://localhost:5174/" : "../";
const deepDiveHref = import.meta.env.DEV ? "/deep-dive/" : "deep-dive/";
const repo = "https://github.com/clinuxrulz/only-gates-ai";

function Nav() {
  return (
    <header class="masthead">
      <a class="brand" href="#top" aria-label="only-gates-ai, back to top">
        <span class="brand-mark" aria-hidden="true">⌁</span>
        <span>only-gates-ai</span>
      </a>
      <nav aria-label="Page sections">
        <For each={NAV}>
          {(item) => <a href={item.href}>{item.label}</a>}
        </For>
        <a href={deepDiveHref}>Deep dive</a>
      </nav>
      <a class="nav-cta" href={demoHref}>Open the demo</a>
    </header>
  );
}

function Hero() {
  return (
    <>
      <section class="hero" id="top">
        <p class="eyebrow">Integer GPT · field notes</p>
        <h1>
          A GPT that runs on integers —
          <span class="accent"> and is being lowered to gates.</span>
        </h1>
        <p class="lede">
          Training still needs floating point and gradients. Inference does not.
          This project keeps the model's arithmetic in int8 and Q16, turns the
          forward pass into a fixed integer dataflow, and is working toward the
          step after that: pure logic gates on an FPGA.
        </p>
        <div class="actions">
          <a class="button" href="#help">Help train the next model</a>
          <a class="button ghost" href={deepDiveHref}>Build it yourself</a>
          <a class="button ghost" href={demoHref}>Try the integer demo</a>
        </div>
        <ul class="chips" aria-label="Project status">
          <li class="chip ready">Integer inference · working</li>
          <li class="chip prototype">Gate compiler · prototype</li>
          <li class="chip roadmap">GPT on FPGA · roadmap</li>
        </ul>
      </section>

      <aside class="honesty" aria-label="What exists and what does not">
        <strong>Keep the claims honest.</strong> Integer-only inference in
        WebAssembly is shipping in the app. The gate compiler works on small
        symbol-classifier circuits. A gate-lowered GPT and an FPGA
        implementation are the roadmap. The page marks the difference
        throughout.
      </aside>
    </>
  );
}

function OneMinute() {
  return (
    <section id="overview" class="section">
      <div class="section-head">
        <p class="kicker">The one-minute version</p>
        <h2>Three facts explain the whole idea</h2>
      </div>
      <div class="cards three">
        <article class="card">
          <span class="card-index">01</span>
          <h3>Training is float</h3>
          <p>
            Gradients, AdamW moments, and loss need fractional values and
            dynamic range. A GPU does this part. The output is a checkpoint:
            a few million ordinary numbers.
          </p>
        </article>
        <article class="card">
          <span class="card-index">02</span>
          <h3>Inference is fixed arithmetic</h3>
          <p>
            Once the weights stop changing, the same matrix products, norms,
            exponentials, and comparisons run for every token. That is compatible
            with int8 storage, Q16 scaling, and integer accumulation.
          </p>
        </article>
        <article class="card">
          <span class="card-index">03</span>
          <h3>Gates are the next rung</h3>
          <p>
            Integer adds, compares, shifts, and table lookups are made of logic
            gates. Lower the graph far enough and it can become an FPGA
            bitstream instead of a program.
          </p>
        </article>
      </div>
      <figure class="figure">
        <TrainingInferenceDiagram />
        <figcaption>
          Training and inference are different machines. Only the left side
          needs a floating-point unit.
        </figcaption>
      </figure>
    </section>
  );
}

function Anatomy() {
  return (
    <section id="anatomy" class="section">
      <div class="section-head">
        <p class="kicker">What is inside</p>
        <h2>A decoder-only transformer, small enough to inspect</h2>
      </div>
      <p>
        The browser demo and the community training target are the same
        architecture at two sizes. Tokens become learned embeddings, pass through
        pre-norm transformer blocks, and are projected back onto the vocabulary
        with the same embedding matrix. There are no biases, no learned position
        table beyond a small embedding, and no exotic operators: every operation
        has an integer implementation.
      </p>
      <div class="table-wrap">
        <table>
          <caption>Model configurations used on this page</caption>
          <thead>
            <tr>
              <th scope="col">Model</th>
              <th scope="col">Role</th>
              <th scope="col">Vocab</th>
              <th scope="col">Context</th>
              <th scope="col">Dim</th>
              <th scope="col">Layers</th>
              <th scope="col">Heads</th>
              <th scope="col">MLP</th>
              <th scope="col">Params</th>
            </tr>
          </thead>
          <tbody>
            <For each={CONFIGS}>
              {(row) => (
                <tr>
                  <th scope="row">{row.name}</th>
                  <td>{row.role}</td>
                  <td>{row.vocab}</td>
                  <td>{row.context}</td>
                  <td>{row.dim}</td>
                  <td>{row.layers}</td>
                  <td>{row.heads}</td>
                  <td>{row.mlp}</td>
                  <td>{row.params}</td>
                </tr>
              )}
            </For>
          </tbody>
        </table>
      </div>
      <p class="note">
        Parameter count is <code>V·D + T·D + D + L·(2D + 4D² + 2D·M)</code>.
        For the Solid 2 target that is 264,528 values. At float32 that is about
        1 MB of weights; at int8 plus scales it is a few hundred kilobytes,
        small enough to sit in on-chip memory on a modest FPGA. The trained
        Solid 2 weights and tokenizer ship in{" "}
        <code>apps/only-gates-ai/public/models/tiny-gpt-solid2/</code>, so the
        demo runs the real checkpoint rather than a placeholder.
      </p>
      <figure class="figure">
        <BlockDiagram />
        <figcaption>
          The whole model is one token-embedding table, a stack of identical
          blocks, and a tied output projection.
        </figcaption>
      </figure>
    </section>
  );
}

function Integers() {
  return (
    <section id="integers" class="section">
      <div class="section-head">
        <p class="kicker">Why integers</p>
        <h2>Small values, wide accumulators, no floating point</h2>
      </div>
      <p>
        Quantization is not “round everything to whole numbers and hope.” The
        model keeps a scale beside each group of weights and each activation row.
        A value is stored as an int8 code and interpreted as{" "}
        <code>code × scale</code>. Products are accumulated in int32 because
        int8 × int8 fits comfortably; anything that multiplies a scaled value
        again moves to int64. Probabilities live in Q15, where 32768 means one.
      </p>
      <div class="cards two">
        <article class="card">
          <h3>What the integer path buys</h3>
          <ul class="ticks">
            <li>Roughly 4× smaller weights than float32, plus one small scale per channel.</li>
            <li>Deterministic arithmetic: the same bytes produce the same logits.</li>
            <li>No FPU, no denormals, no NaN policy differences between devices.</li>
            <li>Cache-friendly rows that map directly to adders and multipliers.</li>
          </ul>
        </article>
        <article class="card">
          <h3>What it costs</h3>
          <ul class="crosses">
            <li>Outputs differ from the float checkpoint; top-1 usually survives, exact text does not.</li>
            <li>Scales are chosen at load time, so a different checkpoint needs a fresh conversion.</li>
            <li>Very large activations need clipping and careful Q-format choices.</li>
            <li>Training cannot run on the integer path; gradients still need float.</li>
          </ul>
        </article>
      </div>
      <figure class="figure">
        <PrecisionDiagram />
        <figcaption>
          Storage stays narrow; arithmetic widens only where an intermediate
          product would overflow.
        </figcaption>
      </figure>
      <div class="table-wrap">
        <table>
          <caption>Precision used by the integer forward pass</caption>
          <thead>
            <tr>
              <th scope="col">Format</th>
              <th scope="col">Where</th>
              <th scope="col">Why</th>
            </tr>
          </thead>
          <tbody>
            <For each={PRECISION}>
              {(row) => (
                <tr>
                  <th scope="row"><code>{row.format}</code></th>
                  <td>{row.where}</td>
                  <td>{row.why}</td>
                </tr>
              )}
            </For>
          </tbody>
        </table>
      </div>
    </section>
  );
}

function Toolbox() {
  return (
    <section id="toolbox" class="section">
      <div class="section-head">
        <p class="kicker">The integer toolbox</p>
        <h2>Every operator has a fixed-point version</h2>
      </div>
      <div class="cards two">
        <article class="card code-card">
          <h3>RMSNorm without a square root instruction</h3>
          <p>
            Sum the squares, divide by the width, take an integer square root,
            then multiply by a Q16 reciprocal.
          </p>
          <pre><code>{`sum = Σ x[i] * x[i]
mean = sum / dim
inv = (1 << 16) / isqrt(mean)
y[i] = (x[i] * inv * gain[i]) >> 16`}</code></pre>
        </article>
        <article class="card code-card">
          <h3>GELU as a lookup table</h3>
          <p>
            The tanh approximation is smooth and bounded. A small Q12 table plus
            linear interpolation replaces the transcendental call.
          </p>
          <pre><code>{`index = clamp(x, -4, 4) mapped to Q12
y = lut[index] + slope * fraction
// one table shared by every MLP row`}</code></pre>
        </article>
        <article class="card code-card">
          <h3>Softmax with DI-Exp</h3>
          <p>
            Subtract the maximum logit, clip the negative difference, and look
            up <code>exp(-x)</code> in Q15. Normalize with a rounded integer
            division.
          </p>
          <pre><code>{`delta = max_logit - logit
p = exp_neg_q15_q12(min(delta, CLIP))
p = p * 32768 / Σp`}</code></pre>
        </article>
        <article class="card code-card">
          <h3>Attention with per-row scales</h3>
          <p>
            Q, K, and V are quantized per position. Scores accumulate in int64,
            softmax produces Q15 weights, and the value sum is requantized once.
          </p>
          <pre><code>{`score = Σ q8[d] * k8[d] * q_scale * k_scale
weight = softmax_q15(scores)
out[d] = Σ weight[j] * v8[j][d] * v_scale[j]`}</code></pre>
        </article>
        <article class="card code-card">
          <h3>Sampling with xorshift</h3>
          <p>
            Top-k keeps the candidate set small. A 32-bit xorshift generator
            supplies deterministic randomness without a platform RNG.
          </p>
          <pre><code>{`x ^= x << 13; x ^= x >> 17; x ^= x << 5;
pick = x % total_weight
// walk the Q15 candidates`}</code></pre>
        </article>
        <article class="card code-card">
          <h3>The rule that keeps it honest</h3>
          <p>
            Floating point is allowed in exactly one file: the load-time
            quantizer. The inference translation unit contains no float type,
            literal, or math library call.
          </p>
          <pre><code>{`tiny_gpt_quantize.c  // float allowed, once
tiny_gpt_infer.c     // integers only
int_math.h           // Q16 / Q15 / Q12 helpers`}</code></pre>
        </article>
      </div>
    </section>
  );
}

function Pipeline() {
  return (
    <section id="pipeline" class="section">
      <div class="section-head">
        <p class="kicker">Float to integer</p>
        <h2>Five steps from a checkpoint to a token</h2>
      </div>
      <ol class="steps">
        <li>
          <strong>Train in float.</strong> Backpropagation and AdamW need
          fractional state. This is the expensive part and the part that needs
          community compute.
        </li>
        <li>
          <strong>Export the checkpoint.</strong> The trainer writes{" "}
          <code>weights.bin</code>, <code>tokenizer.json</code>, and{" "}
          <code>checkpoint.json</code> in the exact packing order the C loader
          expects.
        </li>
        <li>
          <strong>Quantize at load.</strong> Per-channel scales turn weights
          into int8. Activations are quantized per row to int8 with a Q16 scale.
          This is the only place float appears.
        </li>
        <li>
          <strong>Run the integer forward pass.</strong> Embeddings, RMSNorm,
          attention, MLP, final norm, and softmax all stay in integers.
        </li>
        <li>
          <strong>Compare against float.</strong> Measure top-1 agreement,
          top-5 overlap, and mean probability delta. Preserve the behavior that
          matters; do not claim identical text.
        </li>
      </ol>
      <figure class="figure">
        <QuantizationDiagram />
        <figcaption>
          The exported checkpoint and the quantization boundary are the contract
          between training and inference.
        </figcaption>
      </figure>
      <div class="callout">
        <strong>Resumable by design.</strong> The community trainer checkpoints
        the model, AdamW moments, scheduler, RNG, and QA cursor to{" "}
        <code>latest.pt</code>. A free Colab session can stop at a time budget
        and the next session continues from the same optimizer step.
      </div>
    </section>
  );
}

function Gates() {
  return (
    <section id="gates" class="section">
      <div class="section-head">
        <p class="kicker">Integers to gates</p>
        <h2>Arithmetic is a circuit waiting to be written down</h2>
      </div>
      <p>
        An integer operation is not magic. An 8-bit adder is a handful of gates.
        A comparator is a tree of them. A shift by a constant is wiring. A
        multiply by a constant can become a shift-add network or a small ROM.
        A lookup table is a decoder plus a mux. The quantized model is already a
        fixed dataflow, so the lowering problem is “schedule this graph onto
        gates and memory,” not “invent a new model.”
      </p>
      <div class="cards two">
        <article class="card">
          <h3>What the gate compiler already does</h3>
          <p>
            The repository's compiler takes a trained hard-mode gate forest,
            builds a boolean AST, simplifies it with constant folding, identity,
            absorption, De Morgan, and common-subexpression elimination, then
            emits pure bitwise code. That proves the lowering pipeline on small
            symbol-classifier circuits.
          </p>
          <a class="text-link" href={`${repo}/tree/main/packages/only-gates-ai-compiler`}>
            Read the compiler source →
          </a>
        </article>
        <article class="card">
          <h3>What a GPT still needs</h3>
          <p>
            Wider arithmetic lowering, a schedule for matrix tiles, on-chip
            memory planning, control for token-by-token decoding, and a
            verification path against the integer reference. Those are the open
            engineering problems, not reasons the idea fails.
          </p>
          <a class="text-link" href={`${repo}/issues`}>Pick a task on the issue board →</a>
        </article>
      </div>
      <figure class="figure">
        <LoweringDiagram />
        <figcaption>
          The same model descends the stack. Today it stops at stage two; the
          work is to make stages three and four real.
        </figcaption>
      </figure>
      <div class="table-wrap">
        <table>
          <caption>Lowering vocabulary</caption>
          <thead>
            <tr><th scope="col">Integer operation</th><th scope="col">Hardware shape</th></tr>
          </thead>
          <tbody>
            <tr><td>add / subtract</td><td>carry-save or ripple-carry adder</td></tr>
            <tr><td>compare</td><td>subtractor plus sign/zero logic</td></tr>
            <tr><td>shift by constant</td><td>wire renumbering, no gates</td></tr>
            <tr><td>multiply by constant</td><td>shift-add network or ROM</td></tr>
            <tr><td>GELU / exp lookup</td><td>BRAM/ROM plus address mux</td></tr>
            <tr><td>clamp / requantize</td><td>comparator, shifter, rounding adder</td></tr>
            <tr><td>select / top-k</td><td>mux tree and comparators</td></tr>
          </tbody>
        </table>
      </div>
    </section>
  );
}

function Fpga() {
  return (
    <section id="fpga" class="section">
      <div class="section-head">
        <p class="kicker">Why an FPGA</p>
        <h2>Fast, offline, deterministic, and low power</h2>
      </div>
      <p>
        An FPGA is a grid of lookup tables, flip-flops, block RAM, and DSP
        slices that can be wired into a custom machine. It is not as fast as a
        GPU at training, but inference is not training: it is a fixed pipeline
        that can be deep, wide, and local. A gate-lowered GPT could answer
        without an operating system, a driver, a network, or a floating-point
        unit.
      </p>
      <div class="cards two">
        <article class="card">
          <h3>Benefits worth chasing</h3>
          <ul class="ticks">
            <li>Deterministic latency: a fixed pipeline, not a scheduler.</li>
            <li>Low power: no CUDA stack and no wasted floating-point work.</li>
            <li>Offline and private: the model and the prompt stay on the device.</li>
            <li>Deep pipelining: every layer can work on a different token at once.</li>
            <li>Long life: a bitstream can run unchanged for years.</li>
          </ul>
        </article>
        <article class="card">
          <h3>Trade-offs to state plainly</h3>
          <ul class="crosses">
            <li>The model is fixed at synthesis time; changing weights means resynthesis.</li>
            <li>On-chip memory is limited; larger models need external DRAM.</li>
            <li>The clock is far below a GPU; a tiny model fits, a large one does not.</li>
            <li>Quantization changes outputs; verification is part of the work.</li>
            <li>RTL, timing closure, and board bring-up are substantial engineering.</li>
          </ul>
        </article>
      </div>
      <figure class="figure">
        <FpgaDiagram />
        <figcaption>
          A possible floorplan, not a measured implementation. It shows the
          blocks the RTL work has to account for.
        </figcaption>
      </figure>
      <div class="callout">
        <strong>What success looks like:</strong> a gate-lowered transformer
        block that matches the integer reference bit for bit in simulation, then
        one measured board report with clock, tokens per second, latency, LUT/FF/
        BRAM/DSP use, and power. Publish the numbers even if they are modest.
      </div>
    </section>
  );
}

function Roadmap() {
  const items = [
    {
      tag: "M0",
      title: "Integer-only inference",
      status: "done",
      body: "int8 weights, Q16 activations, integer RMSNorm, GELU LUT, DI-Exp softmax, xorshift sampling, WebAssembly build.",
    },
    {
      tag: "M1",
      title: "Reproducible, resumable training",
      status: "now",
      body: "Checkpointed optimizer, scheduler, RNG, and QA cursor; free Colab sessions that add up; public seeds, hashes, and metrics.",
    },
    {
      tag: "M2",
      title: "One block lowered to gates",
      status: "next",
      body: "A bit-level or RTL implementation of one transformer block, verified against the integer reference on fixed vectors.",
    },
    {
      tag: "M3",
      title: "Full tiny GPT netlist",
      status: "later",
      body: "Schedule the whole model, plan memory, and generate a netlist that can be simulated token by token.",
    },
    {
      tag: "M4",
      title: "FPGA prototype",
      status: "later",
      body: "Synthesize and place on a real board; measure tokens per second, latency, area, and power; publish the report.",
    },
  ];
  return (
    <section id="roadmap" class="section">
      <div class="section-head">
        <p class="kicker">Roadmap</p>
        <h2>Where the project is, and what comes next</h2>
      </div>
      <ol class="roadmap">
        <For each={items}>
          {(item) => (
            <li class={`milestone ${item.status}`}>
              <span class="milestone-tag">{item.tag}</span>
              <div>
                <h3>{item.title}</h3>
                <p>{item.body}</p>
              </div>
              <span class="milestone-status">{item.status}</span>
            </li>
          )}
        </For>
      </ol>
    </section>
  );
}

function Help() {
  return (
    <section id="help" class="section">
      <div class="section-head">
        <p class="kicker">Share compute</p>
        <h2>Help train the next checkpoint</h2>
      </div>
      <p class="lede small">
        We have a working integer-only tiny GPT. We want a better one, then a
        gate-lowered one, then a measured FPGA prototype. That needs more
        training runs, more evaluation, and people who know RTL.
      </p>
      <div class="cards three">
        <article class="card track">
          <h3>Compute</h3>
          <p>
            A free Colab T4, a local GPU, or a few CPU hours for evaluation.
            Training is resumable, so short sessions are useful.
          </p>
        </article>
        <article class="card track">
          <h3>Code</h3>
          <p>
            Quantized kernels, gate lowering, RTL, evaluation harnesses,
            tokenizer work, and reproducible training notes.
          </p>
        </article>
        <article class="card track">
          <h3>Hardware</h3>
          <p>
            An FPGA board, synthesis and timing help, area/power measurement,
            and verification against the integer reference.
          </p>
        </article>
      </div>

      <h3 class="subhead">Free Google Colab: first session</h3>
      <p class="actions">
        <a class="button" href="https://colab.research.google.com/github/clinuxrulz/only-gates-ai/blob/main/notebooks/train_tiny_gpt_solid2_colab.ipynb">
          Open the training notebook in Colab
        </a>
        <a class="button ghost" href={`${repo}/blob/main/docs/COMPUTE.md`}>
          Read the compute guide
        </a>
      </p>
      <ol class="steps compact">
        <li>Open the repository's training notebook in Colab.</li>
        <li>Runtime → Change runtime type → T4 GPU.</li>
        <li>Mount Google Drive so a disconnect does not lose the run.</li>
        <li>Run the smoke test first; it proves the pipeline in a minute.</li>
        <li>Start a real run with a time budget. The checkpoint goes to Drive.</li>
      </ol>
      <pre class="code-block"><code>{COLAB_FIRST}</code></pre>

      <h3 class="subhead">Next session: continue, do not restart</h3>
      <p>
        Run the same command with <code>--resume</code>. The trainer verifies the
        configuration, data, tokenizer, and seed, then restores the model,
        optimizer, scheduler, RNG, and QA cursor from{" "}
        <code>latest.pt</code>. It also refuses to resume if the data changed.
      </p>
      <pre class="code-block"><code>{COLAB_NEXT}</code></pre>

      <div class="callout">
        <strong>Be a good citizen of the free tier.</strong> Sessions are
        time-limited, can disconnect, and have usage limits. Use one account,
        start with the smoke test, save to Drive, and follow Colab's terms. If
        you need longer sessions, Kaggle or a local GPU is a better fit.
      </div>

      <h3 class="subhead">Mission board</h3>
      <div class="table-wrap">
        <table>
          <caption>Open work. Claim one in a GitHub issue and say what you can run.</caption>
          <thead>
            <tr>
              <th scope="col">Track</th>
              <th scope="col">Mission</th>
              <th scope="col">Compute</th>
              <th scope="col">What to report</th>
              <th scope="col">Status</th>
            </tr>
          </thead>
          <tbody>
            <For each={MISSIONS}>
              {(mission) => (
                <tr>
                  <td><span class={`tag ${mission.track.toLowerCase()}`}>{mission.track}</span></td>
                  <th scope="row">{mission.title}</th>
                  <td>{mission.compute}</td>
                  <td>{mission.output}</td>
                  <td>{mission.status}</td>
                </tr>
              )}
            </For>
          </tbody>
        </table>
      </div>
      <p class="actions">
        <a class="button" href={`${repo}/issues/new?template=compute-pledge.yml`}>
          Pledge compute
        </a>
        <a class="button ghost" href={`${repo}/issues/new?template=training-result.yml`}>
          Report a training run
        </a>
      </p>
    </section>
  );
}

function Faq() {
  return (
    <section id="faq" class="section">
      <div class="section-head">
        <p class="kicker">Questions</p>
        <h2>Answers before you ask</h2>
      </div>
      <div class="faq">
        <For each={FAQ}>
          {(item) => (
            <details>
              <summary>{item.question}</summary>
              <p>{item.answer}</p>
            </details>
          )}
        </For>
      </div>
    </section>
  );
}

function Footer() {
  return (
    <footer class="footer">
      <p>
        <strong>only-gates-ai</strong> — integer inference, gate lowering, and a
        long road to an FPGA. The code is MIT licensed; the training corpora keep
        their own licenses and attribution.
      </p>
      <p>
        <a href={repo}>Source</a>
        <span class="dot">·</span>
        <a href={`${repo}/blob/main/docs/COMPUTE.md`}>Compute guide</a>
        <span class="dot">·</span>
        <a href={demoHref}>Demo</a>
        <span class="dot">·</span>
        <a href={deepDiveHref}>Deep dive</a>
        <span class="dot">·</span>
        <a href="#top">Back to top</a>
      </p>
    </footer>
  );
}

export function App() {
  return (
    <>
      <Nav />
      <main>
        <Hero />
        <OneMinute />
        <Anatomy />
        <Integers />
        <Toolbox />
        <Pipeline />
        <Gates />
        <Fpga />
        <Roadmap />
        <Help />
        <Faq />
      </main>
      <Footer />
    </>
  );
}
