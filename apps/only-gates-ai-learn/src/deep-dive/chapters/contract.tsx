import { Chapter, Checkpoint, DataTable, Note } from "../components";
import { lineCount, sources } from "../code";
import { PipelineDiagram } from "../diagrams";

const repo = "https://github.com/clinuxrulz/only-gates-ai";

export function Contract() {
  return (
    <Chapter
      id="contract"
      number="00"
      kicker="Read this first"
      title="One float boundary, then integers all the way"
    >
      <p>
        Integer GPT is not a new architecture. It is the same decoder-only
        transformer with a different arithmetic contract. Training still needs
        floating point and gradients. Inference does not: once the weights stop
        changing, every operation is a fixed sequence of adds, multiplies,
        shifts, comparisons, and table lookups. This page teaches that sequence
        from the checkpoint down to the sampled token.
      </p>
      <p>
        If you have never built a GPT before, do not skip the next nine
        chapters. They answer three questions in order: what a GPT computes,
        what each part is for, and how to implement the float version yourself.
        The integer chapters only make sense once that dataflow is familiar.
      </p>

      <figure class="figure">
        <PipelineDiagram />
        <figcaption>
          One float boundary at load, then a fixed integer pipeline. The labels
          on the upper row are the storage formats; the lower row is the
          per-token forward pass.
        </figcaption>
      </figure>

      <div class="contract-grid">
        <div class="contract-card">
          <span class="contract-tag">before</span>
          <h3>Float checkpoint</h3>
          <p>
            <code>weights.bin</code> is a packed little-endian float32 array. A
            trainer produced it. It is the only place a fractional value is
            allowed to enter the inference engine.
          </p>
        </div>
        <div class="contract-arrow" aria-hidden="true">→</div>
        <div class="contract-card">
          <span class="contract-tag">boundary</span>
          <h3>Quantize once</h3>
          <p>
            At load, each matrix becomes int8 plus a Q16 scale. Gains become Q16
            integers. The GELU table is built. After this point the engine never
            includes <code>math.h</code>.
          </p>
        </div>
        <div class="contract-arrow" aria-hidden="true">→</div>
        <div class="contract-card">
          <span class="contract-tag">after</span>
          <h3>Integer inference</h3>
          <p>
            Tokens are integers. Activations are Q16 int32. Dot products
            accumulate in int32 and requantize in int64. Softmax is Q15. The
            sampler is a 32-bit xorshift.
          </p>
        </div>
      </div>

      <h3 class="subhead">Three inference lanes, and only one is fully integer</h3>
      <DataTable
        caption="The app ships three ways to compare the same checkpoint"
        head={["Lane", "Where", "What is integer", "What is still float"]}
        rows={[
          [
            "Float32",
            "apps/only-gates-ai/src/lib/tinyGpt.ts",
            "Nothing",
            "Everything: matmul, norm, GELU, softmax",
          ],
          [
            "TS int8",
            "apps/only-gates-ai/src/lib/tinyGpt.ts",
            "int8 weights, int8 activations, int32 dot accumulation",
            "RMSNorm, GELU, softmax, and the final scale multiplication",
          ],
          [
            "C/WASM integer-only",
            "src/tiny_gpt_infer.c",
            "The whole forward pass and sampler",
            "Nothing after load",
          ],
        ]}
      />
      <Note>
        <strong>Do not confuse the lanes.</strong> The TypeScript “int8” path is
        a useful visual comparison, but it keeps RMSNorm, GELU, and softmax in
        float. The reference this page builds is the C/WASM path in{" "}
        <code>src/tiny_gpt_infer.c</code>. That file contains no float type,
        float literal, or <code>math.h</code> call.
      </Note>

      <h3 class="subhead">Why integer inference at all?</h3>
      <ul>
        <li>
          <strong>Size.</strong> An int8 weight is one byte instead of four. The
          model fits in smaller memory and moves less data.
        </li>
        <li>
          <strong>Determinism.</strong> Integer operations have exact results.
          The same input and seed produce the same output on every machine,
          which is useful for tests and for hardware.
        </li>
        <li>
          <strong>Portability.</strong> The engine needs only integer add,
          multiply, shift, compare, and table lookup. Those exist on a CPU, a
          GPU, a microcontroller, an FPGA, and eventually a gate-level circuit.
        </li>
        <li>
          <strong>Understanding.</strong> Quantization forces you to name every
          scale and every shift. That makes the arithmetic auditable instead of
          hidden inside a float library.
        </li>
      </ul>
      <p>
        The cost is accuracy and engineering effort. Quantization changes the
        probabilities, and every operation needs a scale and overflow argument.
        This page treats those costs honestly: the goal is the same decisions,
        not bit-identical output.
      </p>

      <h3 class="subhead">The files you will read</h3>
      <DataTable
        caption="Production map"
        head={["File", "Lines", "Role"]}
        rows={[
          ["src/int_math.h", String(lineCount(sources.intMath)), "Q16/Q15/Q12 helpers, rounded division and shifts, integer sqrt, DI-Exp table"],
          ["src/tiny_gpt.h", String(lineCount(sources.header)), "Storage contract, structs, scratch buffers, public API"],
          ["src/tiny_gpt_quantize.c", String(lineCount(sources.quantize)), "The only float file: checkpoint conversion and allocation"],
          ["src/tiny_gpt_infer.c", String(lineCount(sources.infer)), "Integer-only forward pass, softmax, sampling"],
          ["tests/test_tiny_gpt.c", "207", "Integer math, create/forward, probability sum, sampler determinism"],
          ["apps/only-gates-ai/src/lib/tinyGptWasm.ts", "261", "Packs the checkpoint and calls the WASM API"],
          ["apps/only-gates-ai/src/lib/tinyGptSolid2.ts", "169", "Solid 2 loader, chat wrapper, generation loop"],
          ["scripts/train_tiny_gpt_solid2_colab.py", "~1260", "PyTorch training and exact export packing"],
          ["examples/integer-gpt/main.c", "new", "Standalone CLI harness around the real engine"],
        ]}
      />

      <h3 class="subhead">What you need to know</h3>
      <p>
        You should be comfortable reading C and a little linear algebra. You do
        not need to know fixed-point arithmetic, quantization, or transformer
        internals in advance; chapters 01–09 build those up from scratch,
        chapter 11 is the float reference, and chapter 12 introduces fixed
        point. The code is C17, but every idea ports to Rust, Go, C#, or
        TypeScript with 64-bit integers.
      </p>

      <h3 class="subhead">Choose your path</h3>
      <DataTable
        caption="Three ways through the page"
        head={["If you…", "Start at", "Then"]}
        rows={[
          [
            "have never seen a GPT",
            "01 The math you need",
            "Read 01–09 in order, run the NumPy reference, then continue to 10",
          ],
          [
            "know transformers but not quantization",
            "10 The model",
            "Read 10–14, then follow one function through 15–19",
          ],
          [
            "only want to rebuild the engine",
            "09 Implement the forward pass",
            "Read 09, 12–19, and use the appendix as the source index",
          ],
        ]}
      />

      <h3 class="subhead">The ten operations you are going to build</h3>
      <ol class="steps compact">
        <li>Quantize a float matrix to int8 with one Q16 scale per output column.</li>
        <li>Quantize an embedding table to int8 with one Q16 scale per row.</li>
        <li>Quantize one Q16 activation row dynamically to int8.</li>
        <li>Multiply int8 × int8 into int32, then requantize in int64.</li>
        <li>Compute RMSNorm with an integer sum of squares and integer sqrt.</li>
        <li>Evaluate GELU with a Q16 lookup table and linear interpolation.</li>
        <li>Compute causal attention with Q12 scores and Q15 probabilities.</li>
        <li>Approximate <code>exp(-x)</code> with a 33-entry table and a shift.</li>
        <li>Normalize logits to Q15, apply temperature, and keep top-k.</li>
        <li>Sample deterministically with xorshift32 and a cumulative sum.</li>
      </ol>

      <Checkpoint>
        <p>
          Before moving on, open <code>src/tiny_gpt_infer.c</code> and confirm
          the rule for yourself: search for <code>float</code>,{" "}
          <code>double</code>, and <code>math.h</code>. You will find the word in
          comments, but no type or library call. The rule is the whole design.
        </p>
      </Checkpoint>

      <p class="source-line">
        Source:{" "}
        <a href={`${repo}/blob/main/src/tiny_gpt_infer.c`}>
          src/tiny_gpt_infer.c
        </a>
      </p>
    </Chapter>
  );
}
