import {
  Chapter,
  Checkpoint,
  Bug,
  Code,
  DataTable,
  Note,
} from "../components";
import { AttentionDiagram } from "../diagrams";
import { snippets } from "../code";

const repo = "https://github.com/clinuxrulz/only-gates-ai";

export function AttentionChapter() {
  return (
    <Chapter
      id="attention"
      number="17"
      kicker="Where the precision changes"
      title="Causal attention with Q12 scores and Q15 probabilities"
    >
      <p>
        Attention is the part that looks hardest to quantize because it contains
        an exponential and a division. The engine handles both with tables and
        fixed-point normalization. The data flow is still the familiar one:
        project Q, K, and V; score every allowed key against the query; softmax
        the scores; mix the values.
      </p>
      <p>
        If any of those words are unfamiliar, read{" "}
        <a href="#attention-primer">chapter 05</a> first. It builds attention
        with ordinary floats and a two-token example. This chapter assumes that
        picture and explains how each operation is represented in integers.
      </p>

      <figure class="figure">
        <AttentionDiagram />
        <figcaption>
          Q, K, and V are quantized per head and per token. Scores are Q12
          nats, probabilities are Q15, and the value sum is requantized to Q16.
        </figcaption>
      </figure>

      <h3 class="subhead">Quantizing Q, K, and V</h3>
      <p>
        <code>wqkv</code> produces a <code>T × 3D</code> Q16 matrix. For each
        head, the engine slices the head’s <code>head_dim</code> values and
        quantizes each key and value row with{" "}
        <code>quantize_activation_row</code>. Keys and values are quantized for
        every position once per head; the query is quantized per position inside
        the loop. The resulting codes are int8 and the scales are Q16-unit
        integers.
      </p>

      <h3 class="subhead">The score rescale</h3>
      <p>
        A dot product of int8 query and key codes is an int32. To turn it into a
        real score, multiply by the query scale, the key scale, and{" "}
        <code>1/sqrt(head_dim)</code>, all Q16. The real score is the product
        divided by <code>2⁴⁸</code>. Softmax wants Q12 nats, so multiply by{" "}
        <code>2¹²</code>; the engine implements that as a single rounded shift
        by 36:
      </p>
      <pre class="math-block"><code>{`real_score = dot · qscale · kscale · inv_sqrt / 2⁴⁸
score_q12  = real_score · 2¹²
           = dot · qscale · kscale · inv_sqrt / 2³⁶`}</code></pre>

      <p>
        The result is clamped to ±81920 Q12, which is ±20 nats. That is the
        DI-Exp clipping range. The causal mask is the loop bound{" "}
        <code>j &lt;= t</code>; positions after the query are never computed and
        their probabilities are zeroed.
      </p>

      <h3 class="subhead">DI-Exp softmax</h3>
      <p>
        Subtracting the row maximum before exponentiating keeps the arguments
        non-positive. The engine computes <code>exp(-delta)</code> in Q15 with a
        33-entry table of <code>2^(-k/32)</code>, linear interpolation, and a
        shift by the integer part. The constants are in{" "}
        <code>int_math.h</code>: <code>5909/4096 ≈ log₂(e)</code> converts nats
        to bits. The result is normalized with a rounded integer divide so the
        weights sum to about 32768.
      </p>

      <h3 class="subhead">The value sum</h3>
      <p>
        Each value code is int8 with a per-token Q16-unit scale. The probability
        is Q15. The real output is{" "}
        <code>Σ p·v / 2³¹</code>; storing it as Q16 means shifting the int64
        accumulator by 15. That is the second shift that is easy to get wrong.
      </p>

      <Code code={snippets.attention} lang="c" file="attention_q16" collapsible />

      <Bug>
        <p>
          The three most common attention bugs are: using <code>&gt;&gt; 32</code>{" "}
          instead of <code>&gt;&gt; 36</code> for scores, forgetting{" "}
          <code>j &lt;= t</code> and attending to the future, and mixing the
          Q12 score scale with the Q15 probability scale. Print one row’s
          scores and probabilities and check that they sum to about 32768.
        </p>
      </Bug>

      <Checkpoint>
        <p>
          For a fixed random QKV buffer, compare each head’s output with a float
          reference. Then check the mask by changing a future value and
          confirming the output for an earlier position does not move. Finally,
          verify that a one-token row produces probability 32768 on itself.
        </p>
      </Checkpoint>

      <p class="source-line">
        Source:{" "}
        <a href={`${repo}/blob/main/src/tiny_gpt_infer.c`}>
          src/tiny_gpt_infer.c · attention_q16
        </a>
      </p>
    </Chapter>
  );
}

export function ForwardChapter() {
  return (
    <Chapter
      id="forward"
      number="18"
      kicker="Putting it together"
      title="The full integer forward pass"
    >
      <p>
        The forward pass is deliberately boring once the pieces exist. Embed,
        then repeat a pre-norm attention block and a pre-norm MLP block, then
        normalize one last time and project onto the vocabulary.
      </p>
      <p>
        If you ran the NumPy reference in chapter 09, you have already written
        this function. The C version is the same sequence with three changes:
        every matrix multiply is quantized, every norm uses an integer square
        root, and GELU is a table lookup. The order of operations and the
        residual adds are identical, which is exactly why the integer version
        can be verified against the float one.
      </p>

      <DataTable
        caption="The buffers reused by one forward pass"
        head={["Buffer", "Shape", "Contents"]}
        rows={[
          ["x", "T × D", "Residual stream, Q16"],
          ["x1", "T × D", "Residual after attention"],
          ["h", "T × D", "Normalized activations"],
          ["qkv", "T × 3D", "Q, K, V projections, Q16"],
          ["attn", "T × D", "Attention output, Q16"],
          ["y", "T × D", "Wo output"],
          ["z", "T × M", "MLP up-projection"],
          ["gz", "T × M", "GELU output"],
          ["m", "T × D", "MLP down-projection"],
          ["logits", "T × V", "Q16 logits for every position"],
        ]}
      />

      <h3 class="subhead">Embedding</h3>
      <p>
        The token row and position row are both int8. Each is multiplied by its
        own Q16 scale and added. The result is Q16 and is saturated to the int32
        range. There is no separate scale on the sum because both sides already
        share the Q16 unit.
      </p>

      <h3 class="subhead">The block</h3>
      <ol class="steps compact">
        <li><code>h = RMSNorm(x, g1)</code></li>
        <li><code>qkv = h · Wqkv</code></li>
        <li><code>attn = attention(qkv)</code></li>
        <li><code>y = attn · Wo</code></li>
        <li><code>x1 = x + y</code></li>
        <li><code>h = RMSNorm(x1, g2)</code></li>
        <li><code>z = h · W1</code></li>
        <li><code>gz = GELU(z)</code></li>
        <li><code>m = gz · W2</code></li>
        <li><code>x = x1 + m</code></li>
      </ol>

      <Code code={snippets.forward} lang="c" file="tiny_gpt_forward" collapsible />

      <h3 class="subhead">The final projection</h3>
      <p>
        The final norm uses <code>g_final</code>, then the head matmul projects{" "}
        <code>T × D</code> onto <code>T × V</code>. Only the last row is needed
        for generation, but the engine computes all positions because the same
        function is used for evaluation, probability inspection, and next-token
        comparisons. A later KV-cache implementation would compute the last row
        only during decode.
      </p>

      <Note>
        <strong>Aliasing rules.</strong> The scratch buffers are reused across
        projections, but no matmul writes its output into its input. If you
        refactor for memory, preserve that rule or add an explicit temporary.
        The sampling path uses <code>probs</code>; attention uses{" "}
        <code>scores</code> and <code>attn_prob</code>. Each buffer is sized for
        the largest use.
      </Note>

      <Bug>
        <p>
          A missing final norm produces plausible-looking but wrong logits,
          because the head expects a normalized residual stream. A missing
          residual produces a model that still runs but cannot represent the
          identity path. Check the two additions before debugging the
          quantization.
        </p>
      </Bug>

      <Checkpoint>
        <p>
          Run the forward pass with synthetic weights and assert that logits are
          non-degenerate, that every buffer is populated, and that the same
          tokens produce identical logits on a second call. Then compare the
          final row with the NumPy reference from chapter 09 or the float
          reference from chapter 11.
        </p>
      </Checkpoint>
    </Chapter>
  );
}

export function SamplingChapter() {
  return (
    <Chapter
      id="sampling"
      number="19"
      kicker="From logits to a token"
      title="Softmax, temperature, top-k, and xorshift32"
    >
      <p>
        The head produces Q16 logits. Generation needs a probability
        distribution over the vocabulary, a temperature, a top-k cut, and a
        random draw. Every step is integer-only.
      </p>
      <p>
        Chapter 07 explains what softmax, temperature, top-k, and sampling mean
        with ordinary floats. This chapter is the same pipeline with Q15
        probabilities, a DI-Exp table, and an integer random generator.
      </p>

      <h3 class="subhead">Logits to Q15</h3>
      <p>
        The last position’s logits are divided by a Q16 temperature. The code
        forms <code>(logit_q16 · 65536) / temperature_q16</code>, which is the
        real ratio stored in Q16, then shifts right by 4 to produce Q12 nats for
        the same DI-Exp table used by attention. The maximum is subtracted, each
        delta is clipped to 20 nats, and the weights are normalized to Q15.
      </p>

      <Code code={snippets.softmax} lang="c" file="softmax_last_q15" collapsible />

      <p>
        <code>tiny_gpt_probabilities</code> calls the same softmax and then
        doubles each Q15 value to produce Q16 for the host API. The WASM wrapper
        divides by 65536. If you forget the doubling, every probability is half
        of what it should be and the sum is 32768 instead of 65536.
      </p>

      <Code code={snippets.probabilities} lang="c" file="tiny_gpt_probabilities" />

      <h3 class="subhead">Top-k</h3>
      <p>
        The sampler keeps the k highest Q15 weights with a partial insertion
        sort. It skips zero weights, maintains a descending list, and stops at{" "}
        <code>count = k</code>. For a 4096-token vocabulary and k=40 this is
        trivial compared with the matmuls.
      </p>

      <h3 class="subhead">The random draw</h3>
      <p>
        A 32-bit xorshift generator supplies the only randomness in the engine.
        A zero seed is replaced by the golden-ratio constant so the state can
        never get stuck at zero. The target is{" "}
        <code>(random · total) &gt;&gt; 32</code>, which maps the 32-bit value
        onto the integer weight range without floating point. The sampler walks
        the cumulative sum and returns the first candidate whose cumulative
        weight exceeds the target.
      </p>

      <Code code={snippets.nextRandom} lang="c" file="next_random" />
      <Code code={snippets.sample} lang="c" file="tiny_gpt_sample" collapsible />

      <Bug>
        <p>
          Temperature has a floor of 0.01 in Q16. If you pass 0, the engine
          clamps it rather than dividing by zero. If top_k is greater than the
          vocabulary, it clamps to the vocabulary. If every probability is zero,
          the sampler returns token 0; do not treat that as a valid distribution.
        </p>
      </Bug>

      <Checkpoint>
        <p>
          With a fixed seed, call the sampler twice and assert the same token.
          Then lower the temperature and assert that the peak probability grows.
          Verify that the Q15 probabilities sum to within a few units of 32768
          and the Q16 output to within a few units of 65536.
        </p>
      </Checkpoint>

      <p class="source-line">
        Source:{" "}
        <a href={`${repo}/blob/main/src/tiny_gpt_infer.c`}>
          src/tiny_gpt_infer.c · softmax_last_q15 · tiny_gpt_sample
        </a>
      </p>
    </Chapter>
  );
}
