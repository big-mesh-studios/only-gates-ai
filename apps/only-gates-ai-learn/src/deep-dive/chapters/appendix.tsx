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
      number="15"
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

      <h3 class="subhead">G. The ten-point checklist</h3>
      <ol class="steps compact">
        <li>Weights are packed in the exact order, with every linear matrix transposed.</li>
        <li>Weight scales are Q16 fractions; activation scales are Q16-unit integers.</li>
        <li>Every dot product accumulates in int32 and requantizes in int64.</li>
        <li>Matmul shifts by 16; attention scores shift by 36; V sums shift by 15.</li>
        <li>RMSNorm uses a Q32 epsilon and an integer square root.</li>
        <li>GELU uses 1025 Q16 entries over [-16, 16] with a shift of 11.</li>
        <li>Attention only computes <code>j &lt;= t</code>.</li>
        <li>Softmax subtracts the max and clamps deltas to 20 nats.</li>
        <li>Q15 probabilities sum to 32768; the host API converts to Q16 by doubling.</li>
        <li>Sampling is seeded, deterministic, and independent of the host RNG.</li>
      </ol>

      <Note>
        <strong>The claim to keep.</strong> This page teaches integer-only
        inference. It does not claim bit-identical output to float, a KV cache,
        or an FPGA implementation. Those are separate milestones, and the field
        notes track them honestly.
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
