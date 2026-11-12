import {
  Chapter,
  Checkpoint,
  Code,
  DataTable,
  Note,
} from "../components";
import { GLOSSARY } from "../content";
import { lineCount, sources } from "../code";

const repo = "https://github.com/clinuxrulz/only-gates-ai";

export function Appendix() {
  return (
    <Chapter
      id="appendix"
      number="24"
      kicker="Appendix"
      title="The full source, the runnable harness, and the checklist"
    >
      <p>
        Everything on this page is grounded in the shipping files below. The
        chapter listings are extracted from these exact sources; if a function
        changes, the page build fails rather than showing old code.
      </p>

      <DataTable
        caption="Full source files on this page"
        head={["File", "Lines", "What it contains"]}
        rows={[
          ["src/int_math.h", String(lineCount(sources.intMath)), "Q formats, rounding, sqrt, DI-Exp"],
          ["src/tiny_gpt.h", String(lineCount(sources.header)), "Storage contract, structs, API"],
          ["src/tiny_gpt_quantize.c", String(lineCount(sources.quantize)), "Float boundary and allocation"],
          ["src/tiny_gpt_infer.c", String(lineCount(sources.infer)), "Integer-only forward pass and sampling"],
        ]}
      />

      <h3 class="subhead">A. int_math.h</h3>
      <Code
        code={sources.intMath}
        lang="c"
        file="src/int_math.h"
        collapsible
      />

      <h3 class="subhead">B. tiny_gpt.h</h3>
      <Code
        code={sources.header}
        lang="c"
        file="src/tiny_gpt.h"
        collapsible
      />

      <h3 class="subhead">C. tiny_gpt_quantize.c</h3>
      <Code
        code={sources.quantize}
        lang="c"
        file="src/tiny_gpt_quantize.c"
        collapsible
      />

      <h3 class="subhead">D. tiny_gpt_infer.c</h3>
      <Code
        code={sources.infer}
        lang="c"
        file="src/tiny_gpt_infer.c"
        collapsible
      />

      <h3 class="subhead">E. Build and run the real engine</h3>
      <p>
        The companion harness in <code>examples/integer-gpt/</code> compiles the
        production C files directly. It is not a second implementation.
      </p>
      <Code
        code={`cd examples/integer-gpt

# Build, check for float in the inference file, and run the selftest
make check

# Inspect a real checkpoint. Config order is V,T,D,L,H,M.
./integer-gpt \\
  --weights ../../apps/only-gates-ai/public/models/tiny-gpt-solid2/weights.bin \\
  --config 4096,256,48,3,4,96 \\
  --tokens 1,2,3,4,5,6,7,8 \\
  --steps 8 --temperature 0.7 --top-k 40 --seed 12345`}
        lang="sh"
        label="Companion commands"
      />

      <p>
        The repository’s own tests use the same engine through CMake:
      </p>
      <Code
        code={`pnpm test                 # C tests, including tiny GPT
pnpm typecheck:app        # TypeScript app
pnpm build:site           # SPA + field notes + deep dive`}
        lang="sh"
        label="Repository checks"
      />

      <h3 class="subhead">F. Glossary</h3>
      <DataTable
        caption="The vocabulary used in this tutorial"
        head={["Term", "Meaning"]}
        rows={GLOSSARY.map((g) => [g.term, g.meaning])}
      />

      <h3 class="subhead">G. How the chapters fit together</h3>
      <DataTable
        caption="The three layers of the tutorial"
        head={["Chapters", "Question answered", "What you produce"]}
        rows={[
          [
            "01–09",
            "What is a GPT, why does each part exist, and how do I build the float version?",
            "A working NumPy forward pass and a mental model of the dataflow",
          ],
          [
            "10–14",
            "What are the exact shapes, the float checkpoint, fixed point, quantization, and the integer matmul?",
            "The contract between the trainer and the C engine, plus the core kernel",
          ],
          [
            "15–24",
            "How do RMSNorm, GELU, attention, the forward pass, and sampling work in integers?",
            "The shipping engine, a way to verify it, and the path to gates",
          ],
        ]}
      />

      <h3 class="subhead">H. The integer contract at a glance</h3>
      <DataTable
        caption="Every operation and the exact integer rule it uses"
        head={["Operation", "Float idea", "Integer rule", "Scale"]}
        rows={[
          ["Embedding", "lookup token row and add position", "code × Q16 scale, add, saturate", "Q16 output"],
          ["Linear layer", "matrix multiply", "int8 × int8 → int32 accumulator", "int32"],
          ["Requantize", "apply both scales", "(int64)acc × act_scale × weight_scale", "shift 16 → Q16"],
          ["RMSNorm", "x / sqrt(mean(x²)+ε) · gain", "Q32 sum of squares, isqrt, rounded divide", "Q16 output"],
          ["GELU", "tanh approximation", "1025-entry Q16 table, linear interpolation", "step 2048, shift 11"],
          ["Attention scores", "q·k / sqrt(d)", "int8 dot, scales, inv_sqrt", "shift 36 → Q12"],
          ["Attention softmax", "exp / sum", "DI-Exp table, max subtraction, Q15 normalize", "Q15, sum 32768"],
          ["Value sum", "Σ p·v", "int64 sum of Q15 × int8 × scale", "shift 15 → Q16"],
          ["Output probabilities", "softmax over logits / temperature", "same DI-Exp path, then double", "Q15 → Q16"],
          ["Sampling", "draw from cumulative distribution", "xorshift32, target = (r × total) >> 32", "integer"],
          ["Generation", "one forward pass per token", "prefill once, then one context row per token against the KV cache", "n/a"],
        ]}
      />

      <h3 class="subhead">I. The fifteen-point checklist</h3>
      <ol class="steps compact">
        <li>Text becomes ids with the exact tokenizer that was trained with the checkpoint.</li>
        <li>Token embeddings and position embeddings are added, not concatenated.</li>
        <li>Every block has the same shape: norm, attention, add, norm, MLP, add.</li>
        <li>The causal mask only allows <code>j ≤ t</code>; the future is never computed.</li>
        <li>Weights are packed in the exact order, with every linear matrix transposed.</li>
        <li>Weight scales are Q16 fractions; activation scales are Q16-unit integers.</li>
        <li>Every dot product accumulates in int32 and requantizes in int64.</li>
        <li>Matmul shifts by 16; attention scores shift by 36; V sums shift by 15.</li>
        <li>RMSNorm uses a Q32 epsilon and an integer square root.</li>
        <li>GELU uses 1025 Q16 entries over [-16, 16] with a shift of 11.</li>
        <li>Attention only computes <code>j ≤ t</code>.</li>
        <li>Softmax subtracts the max and clamps deltas to 20 nats.</li>
        <li>Q15 probabilities sum to 32768; the host API converts to Q16 by doubling.</li>
        <li>Sampling is seeded, deterministic, and independent of the host RNG.</li>
        <li>The float reference and the integer engine agree on top-1 and top-5 for your test prompts.</li>
      </ol>

      <Note>
        <strong>The claim to keep.</strong> This page teaches integer-only
        inference. It does not claim bit-identical output to float or an FPGA
        implementation. The decode schedule now uses a KV cache, and the integer
        engine is still the golden model; the field notes and{" "}
        <a href={`${repo}/blob/main/docs/FPGA-LOWERING.md`}>the lowering
        contract</a> track the hardware milestone honestly.
      </Note>

      <Checkpoint title="You are done when…">
        <p>
          You can change one weight, requantize, run the forward pass, and
          explain why the logits moved. You can point at every shift in the
          engine and say what scale it converts. And you can reproduce the
          selftest from a clean checkout.
        </p>
      </Checkpoint>

      <p class="source-line">
        Repository:{" "}
        <a href={repo}>{repo.replace("https://", "")}</a>
      </p>
    </Chapter>
  );
}
