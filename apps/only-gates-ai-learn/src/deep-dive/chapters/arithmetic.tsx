import {
  Chapter,
  Checkpoint,
  Bug,
  Code,
  DataTable,
  Note,
} from "../components";
import { PRECISION } from "../content";
import { QuantizationDiagram, MatmulDiagram } from "../diagrams";
import { snippets, sources } from "../code";

const repo = "https://github.com/clinuxrulz/only-gates-ai";

export function FixedChapter() {
  return (
    <Chapter
      id="fixed"
      number="03"
      kicker="Fixed-point foundations"
      title="Q formats, rounding, and the overflow budget"
    >
      <p>
        A fixed-point number is an integer plus a contract about where the
        binary point sits. Q16 means “divide by 65536 to get the real value.”
        Q15 means “divide by 32768.” Q12 means “divide by 4096.” There is no
        exponent, no NaN, and no denormal. The contract is the whole type.
      </p>

      <DataTable
        caption="The formats used by the integer engine"
        head={["Format", "Representable real range", "Where", "Why"]}
        rows={PRECISION.map((p) => [p.format, p.range, p.where, p.why])}
      />

      <p>
        The activation stream is Q16 because residuals get small and sixteen
        fractional bits keep them useful. Probabilities are Q15 because they are
        non-negative and one is exactly 32768. The exponential input is Q12
        because a 20-nat clipped range at 1/4096 resolution is enough for
        softmax. Weights are int8 because a byte-sized code with a per-column
        scale is a good trade between size and error.
      </p>

      <h3 class="subhead">The four helpers you will use everywhere</h3>
      <p>
        The complete header is short. Read it once, then keep it open while
        reading the forward pass.
      </p>

      <Code
        code={sources.intMath}
        lang="c"
        file="src/int_math.h"
        collapsible
      />

      <h3 class="subhead">Rounding is not an implementation detail</h3>
      <p>
        Two different rounding helpers are used, and they do not have the same
        tie behavior:
      </p>
      <ul>
        <li>
          <code>int_div_round(a, b)</code> computes{" "}
          <code>round(a / b)</code> with ties away from zero. It works for
          negative numerators because it negates, adds half, divides, and
          negates back.
        </li>
        <li>
          <code>int_shift_round(a, shift)</code> adds half of the divisor and
          then performs an arithmetic right shift. For negative exact halves,
          the result rounds toward +∞, not away from zero. That is acceptable
          for model arithmetic, but a port must copy the behavior if it wants
          identical bytes.
        </li>
      </ul>

      <Code
        code={`/* int_div_round: ties away from zero */
int_div_round( 7, 2) ==  4
int_div_round(-7, 2) == -4

/* int_shift_round: add half, then arithmetic shift */
int_shift_round( 6, 2) == 2     /*  6/4 = 1.5 -> 2 */
int_shift_round(-6, 2) == -1    /* -6/4 = -1.5 -> -1, not -2 */`}
        lang="c"
        label="The rounding edge case"
      />

      <h3 class="subhead">Integer square root</h3>
      <p>
        RMSNorm needs <code>sqrt(mean of squares)</code>. The engine uses the
        classic bit-by-bit integer square root: start with the highest power of
        four below the input, then subtract and shift. It returns the floor of
        the square root, uses only integer arithmetic, and works for the full
        64-bit range.
      </p>

      <Code
        code={`uint64_t int_isqrt_u64(uint64_t n) {
  uint64_t result = 0;
  uint64_t bit = (uint64_t)1 << 62;
  while (bit > n) bit >>= 2;
  while (bit != 0) {
    if (n >= result + bit) {
      n -= result + bit;
      result = (result >> 1) + bit;
    } else {
      result >>= 1;
    }
    bit >>= 2;
  }
  return result;
}`}
        lang="c"
        label="The square-root loop, isolated"
      />

      <p>
        Why Q32 in RMSNorm? An activation is Q16. Squaring it gives Q32. Summing
        and dividing by the width stays Q32. The square root of a Q32 value is
        Q16, which is exactly the scale of the activation stream. The epsilon is
        also stored in Q32: <code>round(1e-5 · 2³²) = 42950</code>. That is why
        the code adds <code>42950</code> to the mean before taking the root.
      </p>

      <h3 class="subhead">Overflow budget</h3>
      <DataTable
        caption="Worst-case values for the two shipping configs"
        head={["Expression", "Worst case", "Limit", "Headroom"]}
        rows={[
          ["int8 × int8 product", "127 × 127 = 16,129", "int32", "enormous"],
          ["D=48 dot product", "48 × 16,129 = 774,192", "2,147,483,647", "~2,700×"],
          ["M=96 dot product", "96 × 16,129 = 1,548,384", "2,147,483,647", "~1,300×"],
          ["Q15 softmax sum, V=4096", "4096 × 32,768 = 134,217,728", "int32", "16×"],
          ["Q15 attention sum, T=256", "256 × 32,768 = 8,388,608", "int32", "256×"],
          ["Q16² sum of squares at |x|=1.0, D=48", "48 × 65536² ≈ 2.06e11", "int64", "~4.5e7×"],
        ]}
      />
      <Note>
        The int32 accumulator is safe for the current model. If you double the
        width or switch to int16 weights, redo this table before assuming it is
        still true. Overflow is silent in C, so the proof is not optional.
      </Note>

      <Checkpoint>
        <p>
          Write four assertions: <code>isqrt(144) == 12</code>,{" "}
          <code>isqrt(145) == 12</code>, <code>div_round(-7, 2) == -4</code>,
          and <code>shift_round(-6, 2) == -1</code>. Then test{" "}
          <code>exp_neg_q15_q12(4096)</code> against
          <code>0.3679 · 32768</code>.
        </p>
      </Checkpoint>
    </Chapter>
  );
}

export function QuantizeChapter() {
  return (
    <Chapter
      id="quantize"
      number="04"
      kicker="The one float file"
      title="Turn a checkpoint into int8 codes and Q16 scales"
    >
      <p>
        Quantization is not “round everything to a whole number.” Each value is
        stored as a signed int8 code and interpreted through a scale:{" "}
        <code>real ≈ code · scale / 65536</code>. The scale is chosen from the
        largest magnitude in a group, so the code uses the full{" "}
        <code>-127 … 127</code> range without overflowing.
      </p>

      <figure class="figure">
        <QuantizationDiagram />
        <figcaption>
          Linear weights get one scale per output column. Embedding tables get
          one scale per row. The group is chosen so that an outlier affects one
          column or row, not the whole matrix.
        </figcaption>
      </figure>

      <h3 class="subhead">The scalar helpers</h3>
      <Code code={snippets.q16FromFloat} lang="c" file="q16_from_float" />
      <Code code={snippets.scaleQ16FromFloat} lang="c" file="scale_q16_from_float" />
      <Code code={snippets.quantizeValue} lang="c" file="quantize_value" />

      <p>
        <code>q16_from_float</code> clamps to the int32 range after multiplying
        by 65536. <code>scale_q16_from_float</code> rejects non-positive scales
        and never returns zero, because a zero scale would make dequantization
        meaningless. <code>quantize_value</code> rounds and clamps to{" "}
        <code>-127 … 127</code>; the code deliberately does not use{" "}
        <code>-128</code>, which keeps symmetric multiplication and negation
        simple.
      </p>

      <h3 class="subhead">Per-column weights</h3>
      <Code
        code={snippets.quantizeColumns}
        lang="c"
        file="quantize_columns"
      />
      <p>
        A linear layer computes one output per column. Giving every column its
        own scale means a single large weight in column 7 does not force every
        other column to use a tiny scale. The transpose in the exporter ensures
        that “column” in the C matrix is “output feature” in PyTorch.
      </p>

      <h3 class="subhead">Per-row embeddings</h3>
      <Code code={snippets.quantizeRows} lang="c" file="quantize_rows" />
      <p>
        An embedding lookup returns one entire row, so the row is the natural
        quantization group. The tied output head reuses those exact row scales:
        when the embedding row is copied into the transposed head column, its
        scale becomes the column scale.
      </p>

      <h3 class="subhead">Gains and the head transpose</h3>
      <Code code={snippets.quantizeGain} lang="c" file="quantize_gain" />
      <Code
        code={`for (int token = 0; token < vocab_size; ++token) {
  model->head_scale[token] = model->token_emb_scale[token];
  for (int d = 0; d < dim; ++d) {
    model->head[(size_t)d * vocab_size + token] =
        model->token_emb[(size_t)token * dim + d];
  }
}`}
        lang="c"
        label="The tied head"
      />

      <h3 class="subhead">The GELU table and the head-dimension reciprocal</h3>
      <Code code={snippets.buildGeluLut} lang="c" file="build_gelu_lut" />
      <Code code={snippets.inverseSqrtQ16} lang="c" file="inverse_sqrt_q16" />

      <p>
        The GELU table has 1025 entries from -16 to 16 in steps of 1/32. Each
        entry is a Q16 value. That is 4,100 bytes, small enough to live in
        on-chip memory and cheap enough to build at load. The attention
        reciprocal is also computed once:{" "}
        <code>inverse_sqrt_q16(head_dim)</code> returns{" "}
        <code>round(65536 / sqrt(head_dim))</code> using the integer square root.
      </p>

      <h3 class="subhead">Creation and allocation</h3>
      <Code
        code={snippets.create}
        lang="c"
        file="tiny_gpt_create"
        collapsible
      />

      <p>
        <code>tiny_gpt_create</code> walks the packed float32 buffer in order,
        quantizes each group, allocates every scratch buffer once, and builds the
        GELU table. After it returns, the float buffer is no longer needed. The
        model struct owns only integers.
      </p>

      <Bug>
        <p>
          The two scale units are easy to mix up. A weight scale is a Q16{" "}
          <em>fraction</em> of the real value. An activation scale is a count of
          Q16 <em>units</em> per code. The matmul multiplies both and shifts by
          16; if you treat the activation scale as a fraction you will be off by
          a factor of 65536.
        </p>
      </Bug>

      <Checkpoint>
        <p>
          Quantize a matrix with one outlier and compare the reconstruction
          error with per-tensor quantization. Then assert{" "}
          <code>abs(restored - original) ≤ scale/2</code> for every value, and
          confirm a zero row produces scale 0 and all-zero codes.
        </p>
      </Checkpoint>

      <p class="source-line">
        Source:{" "}
        <a href={`${repo}/blob/main/src/tiny_gpt_quantize.c`}>
          src/tiny_gpt_quantize.c
        </a>
      </p>
    </Chapter>
  );
}

export function MatmulChapter() {
  return (
    <Chapter
      id="matmul"
      number="05"
      kicker="The heart of the engine"
      title="int8 dot products, int32 accumulation, int64 requantization"
    >
      <p>
        Every linear layer is the same operation: quantize one Q16 activation
        row to int8, dot it against every int8 weight column, accumulate in
        int32, then multiply by the two scales in int64 and shift back to Q16.
      </p>

      <figure class="figure">
        <MatmulDiagram />
        <figcaption>
          The dot product stays narrow; only the scale multiplication needs 64
          bits. The final shift is what turns an accumulator back into a Q16
          activation.
        </figcaption>
      </figure>

      <h3 class="subhead">The derivation</h3>
      <p>
        Let <code>x = qx · sx / 2¹⁶</code> and{" "}
        <code>w = qw · sw / 2¹⁶</code>. Then
      </p>
      <pre class="math-block"><code>{`y = Σ x·w
  = Σ (qx·qw) · sx·sw / 2³²
  = acc · sx·sw / 2³²

The engine stores y in Q16, so:
y_q16 = y · 2¹⁶ = acc · sx·sw / 2¹⁶`}</code></pre>
      <p>
        That is the entire justification for the single <code>&gt;&gt; 16</code>{" "}
        at the end of <code>matmul_q16</code>. The activation scale{" "}
        <code>sx</code> is in Q16 units, so it is not itself divided by 65536;
        the weight scale <code>sw</code> is a Q16 fraction, so it is.
      </p>

      <h3 class="subhead">Dynamic row quantization</h3>
      <Code
        code={snippets.quantizeActivationRow}
        lang="c"
        file="quantize_activation_row"
      />
      <p>
        There are two regimes. If every value fits in <code>-127 … 127</code>,
        the row is already small enough to use directly and the scale is{" "}
        <code>1</code>. Otherwise the scale is{" "}
        <code>ceil(max_abs / 127)</code>, and each code is{" "}
        <code>round(value · 127 / max_abs)</code>. The first regime is not a
        rounding shortcut: the source values are Q16 integers, so a code of{" "}
        <code>100</code> with scale 1 means “100 Q16 units,” and the final shift
        handles the rest.
      </p>

      <h3 class="subhead">The matmul</h3>
      <Code code={snippets.matmul} lang="c" file="matmul_q16" />

      <h3 class="subhead">A worked example</h3>
      <p>
        Take a Q16 activation row <code>[65536, 32768]</code>, which is{" "}
        <code>[1.0, 0.5]</code>. The maximum is 65536, so the activation scale is{" "}
        <code>(65536 + 63) / 127 = 516</code> and the codes are{" "}
        <code>[127, 64]</code>. Take an int8 weight column{" "}
        <code>[64, -32]</code> with Q16 scale <code>1024</code>, which is{" "}
        <code>[1.0, -0.5]</code>.
      </p>
      <pre class="math-block"><code>{`acc = 127·64 + 64·(-32) = 8128 - 2048 = 6080
product = 6080 · 516 · 1024 = 3,212,574,720
y_q16 = round(product / 65536) = 49,020
y = 49,020 / 65536 ≈ 0.748`}</code></pre>
      <p>
        The exact float answer is <code>1.0·1.0 + 0.5·(-0.5) = 0.75</code>. The
        difference is the activation and weight quantization error, not a shift
        bug.
      </p>

      <Bug>
        <p>
          If the output is consistently 65536 times too large, the final shift
          is missing. If it is 65536 times too small, an extra shift was added.
          Check the shift before changing the scales. In attention the
          corresponding shift is 36, not 16, because scores are Q12.
        </p>
      </Bug>

      <Checkpoint>
        <p>
          Implement a scalar reference in double precision and compare it with{" "}
          <code>matmul_q16</code> on random rows with a known maximum. The mean
          error should be on the order of <code>max_abs / 127</code>. Then test
          a row whose maximum is below 127 and confirm the scale-1 path.
        </p>
      </Checkpoint>
    </Chapter>
  );
}

export function NormChapter() {
  return (
    <Chapter
      id="norm"
      number="06"
      kicker="Normalization"
      title="RMSNorm without a square-root instruction"
    >
      <p>
        RMSNorm scales a row by the reciprocal of its root mean square, then
        multiplies by a learned gain. In float it is{" "}
        <code>x / sqrt(mean(x²) + ε) · gain</code>. The integer version replaces
        the square root with a bitwise loop and the division with a rounded
        integer divide.
      </p>

      <Code code={snippets.rmsnorm} lang="c" file="rmsnorm_q16" />

      <h3 class="subhead">The scale bookkeeping</h3>
      <ul>
        <li>
          <code>x</code> is Q16, so <code>x²</code> is Q32. The sum of squares
          and the mean stay Q32 in an int64.
        </li>
        <li>
          The epsilon is <code>round(1e-5 · 2³²) = 42950</code>, also Q32.
        </li>
        <li>
          <code>int_isqrt_u64(mean_q32)</code> returns Q16, the same scale as
          the input.
        </li>
        <li>
          The gain is Q16. Dividing <code>x · gain</code> by the Q16 RMS cancels
          the gain’s scale and returns Q16.
        </li>
      </ul>

      <h3 class="subhead">A worked row</h3>
      <pre class="math-block"><code>{`x = [65536, 65536]              // [1.0, 1.0] in Q16
sum_squares = 2 · 65536² = 8,589,934,592
mean_q32 = sum_squares / 2 + 42950 = 4,295,010,246
rms_q16 = isqrt(4,295,010,246) = 65536
y[i] = round(x[i] · gain[i] / rms_q16)
     = round(65536 · 65536 / 65536) = 65536  // [1.0, 1.0]`}</code></pre>

      <Bug>
        <p>
          Do not divide by <code>cols</code> after adding epsilon, and do not
          use a Q16 epsilon. The epsilon belongs in the same Q32 units as the
          mean. A Q16 epsilon would be far too small after squaring and the norm
          would behave differently from the float model.
        </p>
      </Bug>

      <Checkpoint>
        <p>
          Compare <code>rmsnorm_q16</code> with the float formula on random rows.
          Test a zero row (the code forces <code>rms_q16 = 1</code>), a row with
          one large value, and a row with negative values. The error should be
          within a few Q16 units, not a few thousand.
        </p>
      </Checkpoint>
    </Chapter>
  );
}

export function GeluChapter() {
  return (
    <Chapter
      id="gelu"
      number="07"
      kicker="The nonlinearity"
      title="GELU as a 4 KB lookup table"
    >
      <p>
        The MLP uses the tanh approximation of GELU:
      </p>
      <pre class="math-block"><code>{`GELU(x) = 0.5 · x · (1 + tanh(0.79788456 · (x + 0.044715 · x³)))`}</code></pre>
      <p>
        The function is smooth and bounded, so the engine samples it once at
        load into a Q16 table and interpolates between points at runtime. No
        transcendental call is needed in the forward pass.
      </p>

      <Code code={snippets.buildGeluLut} lang="c" file="build_gelu_lut" />
      <Code code={snippets.gelu} lang="c" file="gelu_q16" />

      <h3 class="subhead">The table geometry</h3>
      <ul>
        <li>Range: -16.0 to +16.0 in Q16.</li>
        <li>Step: 1/32, or 2048 in Q16 units.</li>
        <li>Points: 1025, because 32 units per side times 32 steps plus one.</li>
        <li>Storage: 1025 int32 values, about 4.1 KB.</li>
      </ul>
      <p>
        Below -16 the function is effectively zero, so the lookup returns 0.
        Above +16 it is effectively the identity, so it returns the input. In
        between, the index is <code>offset / 2048</code> and the remainder is{" "}
        <code>offset &amp; 2047</code>. The interpolation shift is 11 because{" "}
        <code>2¹¹ = 2048</code>.
      </p>

      <Bug>
        <p>
          Two off-by-ones live here. The table has 1025 entries, so the last
          valid index is 1024; the code guards the upper range before indexing.
          And the interpolation shift is 11, not 12: the step is 1/32 of a Q16
          unit, not 1/16.
        </p>
      </Bug>

      <Checkpoint>
        <p>
          Build the table in a test, evaluate it at every grid point, and
          compare with the float formula. Then evaluate halfway between two grid
          points and check that the result is between the neighbors. Measure the
          maximum error; it should be far below the Q16 resolution of the
          activation stream.
        </p>
      </Checkpoint>
    </Chapter>
  );
}
