import {
  Chapter,
  Checkpoint,
  Code,
  DataTable,
  Note,
} from "../components";
import { AttentionPrimerDiagram, BlockPurposeDiagram } from "../diagrams";

const repo = "https://github.com/clinuxrulz/only-gates-ai";

export function BlockChapter() {
  return (
    <Chapter
      id="block"
      number="04"
      kicker="Foundations · the parts"
      title="The transformer block: what each part is for"
    >
      <p>
        A GPT is a stack of identical <strong>transformer blocks</strong>. Each
        block reads a matrix of shape <code>[T, D]</code> and writes a matrix of
        the same shape. Nothing is added or removed: the block only changes the
        numbers in place. That fixed shape is why blocks can be stacked{" "}
        <code>L</code> times without special cases.
      </p>
      <p>
        The shared matrix is called the <strong>residual stream</strong>. Think
        of it as a shared notebook with one row per token position. Each part of
        the block reads the notebook, computes something useful, and writes an
        update back into the same row. A part never replaces the notebook; it
        adds to it.
      </p>

      <figure class="figure">
        <BlockPurposeDiagram />
        <figcaption>
          The residual stream runs top to bottom. RMSNorm prepares a readable
          copy, attention moves information between positions, and the MLP
          transforms each position independently. Both sublayers add their
          result back to the stream.
        </figcaption>
      </figure>

      <h3 class="subhead">The four parts and their jobs</h3>
      <DataTable
        caption="Why a transformer block contains exactly these pieces"
        head={["Part", "Input → output", "Job", "If you removed it"]}
        rows={[
          [
            "RMSNorm",
            "[T, D] → [T, D]",
            "Rescale each row to a stable size before a learned operation",
            "Activations drift in scale; training becomes unstable",
          ],
          [
            "Attention",
            "normalized [T, D] → [T, D]",
            "Move information between token positions",
            "Each position is isolated; pronouns and questions lose context",
          ],
          [
            "MLP",
            "normalized [T, D] → [T, D]",
            "Compute nonlinear features within each position",
            "The model is only a weighted average; much less expressive",
          ],
          [
            "Residual add",
            "stream + sublayer → stream",
            "Keep the original information and learn an incremental change",
            "Deep stacks are hard to train; the identity path disappears",
          ],
        ]}
      />

      <h3 class="subhead">RMSNorm: keep the numbers in a useful range</h3>
      <p>
        A vector’s values can become very large or very small as they pass
        through many layers. Large values saturate nonlinearities and make
        gradients unstable. <strong>RMSNorm</strong> divides each row by its
        root mean square, then multiplies by a learned per-dimension{" "}
        <strong>gain</strong>:
      </p>
      <pre class="math-block"><code>{`rms(x)     = sqrt(mean(x[i]²) + ε)
rmsnorm(x) = x / rms(x) · gain

The gain starts at 1 and is learned. ε is a tiny number
that prevents division by zero.`}</code></pre>
      <p>
        Unlike LayerNorm, which subtracts the mean and divides by the standard
        deviation, RMSNorm does not subtract the mean first. That saves an
        operation per row and works well in transformer language models. The
        integer implementation uses a Q32 epsilon and an integer square root;
        chapter 15 derives every scale.
      </p>

      <h3 class="subhead">Attention: move information between positions</h3>
      <p>
        After embedding, each position knows only its own token and position.
        Attention lets position <code>t</code> read from positions{" "}
        <code>0 … t</code>. It does this by projecting every row into three
        roles: a <strong>query</strong>, a <strong>key</strong>, and a{" "}
        <strong>value</strong>. The query of position <code>t</code> is
        compared with the key of every allowed position; the resulting scores
        decide how much of each value to add to the output. Chapter 05 builds
        this from scratch.
      </p>
      <p>
        The causal part is essential. Because the model is trained to predict
        the next token, position <code>t</code> must not see position{" "}
        <code>t + 1</code>. The implementation enforces this by only computing
        scores for <code>j ≤ t</code>, which is both correct and cheaper than
        computing the future and masking it later.
      </p>

      <h3 class="subhead">The MLP: compute within a position</h3>
      <p>
        Attention mixes information across positions, but it is a weighted
        average of value vectors, which is a mostly linear operation. The{" "}
        <strong>MLP</strong> (multi-layer perceptron, also called the
        feed-forward network) gives each position a nonlinear computation of its
        own. It expands the vector from <code>D</code> to a wider hidden size{" "}
        <code>M</code>, applies GELU, and projects back to <code>D</code>:
      </p>
      <pre class="math-block"><code>{`z = h · W1          # [T, D]  → [T, M]
g = GELU(z)         # elementwise, [T, M] → [T, M]
m = g · W2          # [T, M]  → [T, D]`}</code></pre>
      <p>
        The same MLP is applied to every row independently. It is where much of
        the model’s per-token feature computation happens. Chapter 06 explains
        the expansion, the nonlinearity, and the GELU approximation.
      </p>

      <h3 class="subhead">Residual connections: learn a change, not a replacement</h3>
      <p>
        Every sublayer is wrapped in a residual add:
      </p>
      <pre class="math-block"><code>{`x = x + attention(rmsnorm(x))
x = x + mlp(rmsnorm(x))`}</code></pre>
      <p>
        The addition has three effects. It preserves the original signal, so a
        later layer can always recover what an earlier layer knew. It gives the
        gradient a short path back through the network during training. And it
        means each sublayer only has to learn a small adjustment, which is
        easier than learning a complete replacement from scratch.
      </p>

      <h3 class="subhead">Pre-norm: normalize before, not after</h3>
      <p>
        The order above is <strong>pre-norm</strong>: normalize, apply the
        sublayer, add to the stream. Older transformers normalized after the
        add. Pre-norm is easier to train deeply, and it is cheaper in the integer
        engine because the residual stream itself is never rewritten by a norm.
        The engine’s block is literally the two lines above.
      </p>

      <h3 class="subhead">What one block costs</h3>
      <p>
        With <code>D = 48</code> and <code>M = 96</code>, one block contains:
      </p>
      <ul>
        <li>
          <strong>Attention:</strong> <code>D×3D</code> for QKV plus{" "}
          <code>D×D</code> for the output projection = <code>4D² = 9,216</code>{" "}
          weights.
        </li>
        <li>
          <strong>MLP:</strong> <code>D×M</code> up plus <code>M×D</code> down ={" "}
          <code>2DM = 9,216</code> weights.
        </li>
        <li>
          <strong>Norms:</strong> two gains of <code>D</code> ={" "}
          <code>2D = 96</code> weights.
        </li>
      </ul>
      <p>
        So one block is 18,528 parameters. Three blocks plus the embedding
        tables and the final gain give the 264,528-parameter Solid 2 model. A
        “bigger model” mostly means more blocks and wider matrices, not a
        different kind of computation.
      </p>

      <Note>
        <strong>Depth is iterative refinement.</strong> Early blocks tend to
        build local features; later blocks can combine them. That is a useful
        intuition, not a strict rule. What matters for this page is that every
        block has the same shape and the same set of operations, which is what
        makes both the C engine and a hardware pipeline regular.
      </Note>

      <Checkpoint>
        <p>
          Draw the residual stream as a vertical line. Mark where RMSNorm,
          attention, the first add, RMSNorm, the MLP, and the second add sit.
          Then remove attention from the drawing and describe what the model can
          no longer do. Finally, compute the per-block parameter count for{" "}
          <code>D = 48</code>, <code>M = 96</code> and confirm it is 18,528.
        </p>
      </Checkpoint>

      <p class="source-line">
        Source:{" "}
        <a href={`${repo}/blob/main/src/tiny_gpt_infer.c`}>
          src/tiny_gpt_infer.c · tiny_gpt_forward
        </a>
      </p>
    </Chapter>
  );
}

const attentionToy = `# One head, head_dim = 2. For this toy, Wq = Wk = Wv = identity.
# Real models learn these matrices from data.
x0 = [1, 0]     x1 = [0, 1]

q0 = [1, 0]     k0 = [1, 0]     v0 = [1, 0]
q1 = [0, 1]     k1 = [0, 1]     v1 = [0, 1]

# Position 0 may only see key 0.
score00 = (q0 · k0) / sqrt(2) = 0.7071
p0      = softmax([0.7071]) = [1.0]
out0    = 1.0 · v0 = [1.0, 0.0]

# Position 1 may see keys 0 and 1.
score10 = (q1 · k0) / sqrt(2) = 0.0
score11 = (q1 · k1) / sqrt(2) = 0.7071
p1      = softmax([0.0, 0.7071]) = [0.330, 0.670]
out1    = 0.330 · v0 + 0.670 · v1 = [0.330, 0.670]`;

export function AttentionPrimerChapter() {
  return (
    <Chapter
      id="attention-primer"
      number="05"
      kicker="Foundations · attention"
      title="Attention from scratch: queries, keys, and values"
    >
      <p>
        Attention answers a simple question: <em>which earlier positions should
        this position listen to?</em> Consider “The animal did not cross the
        street because it was too tired.” To interpret “it”, the model needs to
        connect that position to “animal”. Attention is the mechanism that
        makes that connection.
      </p>

      <figure class="figure">
        <AttentionPrimerDiagram />
        <figcaption>
          Each position produces a query, a key, and a value. Queries are
          compared with keys to produce one score per allowed pair. Softmax
          turns each row of scores into weights, and the weights mix the value
          vectors.
        </figcaption>
      </figure>

      <h3 class="subhead">The library analogy</h3>
      <p>
        Imagine a library. Your <strong>query</strong> is what you are looking
        for. Each book has a <strong>key</strong> on its spine describing what it
        contains. You compare your query with every key to see which books are
        relevant. Then you read the <strong>value</strong>, the actual content,
        and combine the relevant books in proportion to their relevance.
      </p>
      <p>
        In a transformer, every position emits all three:
      </p>
      <ul>
        <li>
          <strong>Query (Q):</strong> what this position is looking for.
        </li>
        <li>
          <strong>Key (K):</strong> what this position can offer to others.
        </li>
        <li>
          <strong>Value (V):</strong> the information this position passes on
          when selected.
        </li>
      </ul>
      <p>
        Q, K, and V are not hand-written. They are three learned linear
        projections of the same normalized row. The model learns which
        directions in the vector space should act as questions, which as
        labels, and which as content.
      </p>

      <h3 class="subhead">The formula, one piece at a time</h3>
      <pre class="math-block"><code>{`Q = h · Wq        # [T, D]
K = h · Wk        # [T, D]
V = h · Wv        # [T, D]

scores = Q · Kᵀ / sqrt(head_dim)     # [T, T]
scores[j > t] = -∞                   # causal mask
P = softmax(scores, axis=keys)       # [T, T]
out = P · V                          # [T, D]`}</code></pre>
      <ul>
        <li>
          <strong>Why divide by <code>sqrt(head_dim)</code>?</strong> A dot
          product over <code>d</code> dimensions has variance that grows with{" "}
          <code>d</code>. Without the scale, scores become large, softmax
          saturates, and gradients vanish. Dividing by{" "}
          <code>sqrt(d)</code> keeps the scores in a useful range.
        </li>
        <li>
          <strong>Why softmax over keys?</strong> Each query produces one score
          per allowed key. Softmax turns that row into non-negative weights that
          sum to one, so the output is a weighted average of values.
        </li>
        <li>
          <strong>Why the causal mask?</strong> During training the model sees
          the whole sequence but must predict each next token using only the
          past. Setting future scores to negative infinity gives them zero
          probability. At generation time those positions do not exist yet, so
          the same rule is both necessary and natural.
        </li>
      </ul>

      <h3 class="subhead">A two-token worked example</h3>
      <Code code={attentionToy} lang="text" label="Attention by hand" />
      <p>
        Position 0 keeps only its own value because it has no past. Position 1
        mixes both values, with more weight on the key that matches its query.
        With learned weights the pattern is richer, but the mechanics are
        exactly this.
      </p>

      <h3 class="subhead">Multi-head attention</h3>
      <p>
        One attention pattern can only express one kind of relationship. A{" "}
        <strong>head</strong> is an independent attention computation on a slice
        of the vector. With <code>H</code> heads, each head has width{" "}
        <code>head_dim = D / H</code>. Every head has its own Q, K, V
        projections, computes its own scores and weighted sum, and the results
        are concatenated back to width <code>D</code>. A final output matrix{" "}
        <code>Wo</code> mixes the heads.
      </p>
      <p>
        Different heads can learn different jobs: one may track nearby tokens,
        another may connect a pronoun to a noun, another may copy a delimiter.
        The model is not told which is which; the split is learned. In this
        project Solid 2 has <code>D = 48</code> and <code>H = 4</code>, so each
        head is 12 numbers wide.
      </p>

      <h3 class="subhead">Shapes through the attention block</h3>
      <DataTable
        caption="Every tensor attention touches"
        head={["Tensor", "Shape", "Meaning"]}
        rows={[
          ["h", "[T, D]", "Normalized input, one row per position"],
          ["Wqkv", "[D, 3D]", "One fused matrix producing Q, K, and V"],
          ["QKV", "[T, 3D]", "Three projections concatenated"],
          ["Q, K, V", "[T, D] each", "Split and reshaped to [H, T, D/H]"],
          ["scores", "[H, T, T]", "One score matrix per head"],
          ["P", "[H, T, T]", "Softmax weights, zero above the diagonal"],
          ["head output", "[H, T, D/H]", "Weighted values, then concatenated"],
          ["attention output", "[T, D]", "After the Wo projection"],
        ]}
      />

      <h3 class="subhead">What is learned here</h3>
      <p>
        The only learned parameters are <code>Wq</code>, <code>Wk</code>,{" "}
        <code>Wv</code>, and <code>Wo</code>. In the C engine the three input
        projections are fused into one <code>[D, 3D]</code> matrix called{" "}
        <code>wqkv</code>. That saves two matrix multiplies and one pass over
        the input. The math is exactly the same; only the storage layout changes.
      </p>

      <Note>
        <strong>Attention is not a database lookup.</strong> The weights are
        computed from the current vectors, so the same key can be relevant in
        one context and irrelevant in another. The dot product is a learned
        similarity measure, not an exact match.
      </Note>

      <Checkpoint>
        <p>
          Redo the two-token example with <code>q1 = [1, 1]</code>, keeping{" "}
          <code>k0 = [1, 0]</code> and <code>k1 = [0, 1]</code>. Compute the
          scores, the softmax weights, and the new <code>out1</code>. Then
          explain why the causal mask is a loop bound (<code>j ≤ t</code>) in
          the C engine and not a separate multiplication by zero.
        </p>
      </Checkpoint>

      <p class="source-line">
        Integer version:{" "}
        <a href="#attention">chapter 17, attention with Q12 scores</a> ·{" "}
        <a href={`${repo}/blob/main/src/tiny_gpt_infer.c`}>
          src/tiny_gpt_infer.c
        </a>
      </p>
    </Chapter>
  );
}

export function MlpPrimerChapter() {
  return (
    <Chapter
      id="mlp"
      number="06"
      kicker="Foundations · the MLP"
      title="The MLP and GELU: where each token is transformed"
    >
      <p>
        Attention is good at moving information between positions. It is less
        good at computing with that information, because its output is a
        weighted average of value vectors, which is close to linear. The{" "}
        <strong>MLP</strong> is the nonlinear computation that follows. It runs
        independently on every position, so it never mixes tokens.
      </p>

      <pre class="math-block"><code>{`z = h · W1          # expand:  [T, D] → [T, M]
g = GELU(z)         # nonlinear: [T, M] → [T, M]
m = g · W2          # contract: [T, M] → [T, D]

x = x + m           # residual add, back in the stream`}</code></pre>

      <p>
        The same matrix <code>W1</code> is applied to every row, then the same
        GELU, then the same <code>W2</code>. A position’s row never sees another
        position’s row inside the MLP. That is the clean division of labour:
        attention communicates, the MLP computes.
      </p>

      <h3 class="subhead">Why expand the width?</h3>
      <p>
        The up-projection gives the model a wider space in which to combine
        features before compressing back. The original transformer used{" "}
        <code>M = 4D</code>; this project uses <code>M = 2D</code> (96 for{" "}
        <code>D = 48</code>) to keep the tiny model small. A wider hidden layer
        means more parameters and more compute, but also more capacity to
        represent intermediate combinations.
      </p>
      <p>
        The cost is <code>2DM</code> weights per block: <code>D×M</code> for the
        up-projection and <code>M×D</code> for the down-projection. For Solid 2
        that is 9,216 parameters per block, the same as the entire attention
        block.
      </p>

      <h3 class="subhead">Why the nonlinearity is not optional</h3>
      <p>
        Suppose GELU were removed. Then the whole MLP would be:
      </p>
      <pre class="math-block"><code>{`m = (h · W1) · W2 = h · (W1 · W2)`}</code></pre>
      <p>
        Two linear maps in a row are just one linear map. No matter how many
        layers you stack, the model could only represent a single linear
        transformation of the input. The nonlinearity breaks that collapse and
        lets the network represent curved, conditional relationships. Attention
        needs its own nonlinearity for a different reason: without softmax, the
        weights would not be normalized, so the mix of values would not be a
        stable, bounded average. GELU and softmax are the two nonlinearities
        that make a deep transformer more than a stack of matrix multiplies.
      </p>

      <h3 class="subhead">What GELU looks like</h3>
      <p>
        GELU stands for Gaussian error linear unit. This project uses the tanh
        approximation:
      </p>
      <pre class="math-block"><code>{`GELU(x) = 0.5 · x · (1 + tanh(0.79788456 · (x + 0.044715 · x³)))`}</code></pre>
      <p>
        Its shape is simple to describe:
      </p>
      <ul>
        <li>
          For large positive <code>x</code>, it is almost the identity:{" "}
          <code>GELU(3) ≈ 2.996</code>.
        </li>
        <li>
          At zero it is zero: <code>GELU(0) = 0</code>.
        </li>
        <li>
          For small negative <code>x</code>, it becomes slightly negative:{" "}
          <code>GELU(-1) ≈ -0.159</code>.
        </li>
        <li>
          For large negative <code>x</code>, it approaches zero:{" "}
          <code>GELU(-3) ≈ -0.004</code>.
        </li>
      </ul>
      <p>
        So GELU acts like a smooth gate: it lets positive evidence through,
        suppresses strong negative evidence, and keeps a small negative channel
        near zero. It is smooth, which makes it easy to approximate with a
        lookup table. Chapter 16 builds the 1,025-entry integer table.
      </p>

      <h3 class="subhead">A tiny numeric picture</h3>
      <p>
        Take one position with <code>D = 2</code>, <code>M = 3</code>, and a
        hidden row <code>h = [1, -1]</code>. Let the up-projection produce{" "}
        <code>z = [1, -1, 0]</code>. After GELU:
      </p>
      <pre class="math-block"><code>{`z      = [ 1.000, -1.000,  0.000]
GELU(z)= [ 0.841, -0.159,  0.000]`}</code></pre>
      <p>
        The down-projection then mixes those three numbers back into two. The
        exact result depends on <code>W2</code>, but the important part is that
        the negative value was not simply deleted; it was scaled to{" "}
        <code>-0.159</code> and can still contribute.
      </p>

      <h3 class="subhead">Why this is easy to quantize</h3>
      <p>
        The MLP is two matrix multiplies and one elementwise function. The
        matrix multiplies use the same integer kernel as every other linear
        layer. GELU is the only difficult part, and because its input is bounded
        in practice, the engine can precompute it at load time into a Q16 table
        and interpolate. There is no <code>tanh</code> call in the forward pass.
      </p>

      <Checkpoint>
        <p>
          Show algebraically that <code>(h · W1) · W2 = h · (W1 · W2)</code> and
          explain why that makes a two-layer linear network no stronger than
          one layer. Then compute <code>GELU(1)</code>, <code>GELU(-1)</code>,
          and <code>GELU(0)</code> to three decimal places and describe the
          shape in one sentence. Finally, count the MLP parameters for{" "}
          <code>D = 48</code>, <code>M = 96</code>.
        </p>
      </Checkpoint>

      <p class="source-line">
        Integer version:{" "}
        <a href="#gelu">chapter 16, GELU as a lookup table</a> ·{" "}
        <a href={`${repo}/blob/main/src/tiny_gpt_infer.c`}>
          src/tiny_gpt_infer.c
        </a>
      </p>
    </Chapter>
  );
}

export function OutputPrimerChapter() {
  return (
    <Chapter
      id="output-primer"
      number="07"
      kicker="Foundations · the output"
      title="Logits, softmax, temperature, and sampling"
    >
      <p>
        After the last block, the model has a <code>[T, D]</code> matrix. Only
        the last row matters for generation, because it is the position that
        predicts what comes next. The final steps turn that row into a token
        id.
      </p>

      <ol class="steps compact">
        <li>Normalize the last row with the final RMSNorm gain.</li>
        <li>Multiply by the output head to get one score per vocabulary token.</li>
        <li>Divide by temperature and apply softmax to get probabilities.</li>
        <li>Optionally keep only the top-k tokens and renormalize.</li>
        <li>Draw a token from the distribution (or take the argmax).</li>
      </ol>

      <h3 class="subhead">The output head and tied embeddings</h3>
      <p>
        The output head is a matrix of shape <code>[D, V]</code>. Multiplying
        the normalized row <code>[1, D]</code> by it produces{" "}
        <code>[1, V]</code> <strong>logits</strong>: one unbounded score per
        vocabulary token.
      </p>
      <pre class="math-block"><code>{`logits = final_norm(x) · head`}</code></pre>
      <p>
        In this project the head is not a separate learned matrix. It is the
        token embedding matrix transposed, so the logit for token{" "}
        <code>id</code> is the dot product between the final hidden vector and{" "}
        that token’s embedding row. The model is asking: “how compatible is
        this position with token <code>id</code>?” This is called{" "}
        <strong>tied embeddings</strong>. It saves <code>V·D</code> parameters
        and keeps the input and output token spaces aligned.
      </p>

      <h3 class="subhead">Softmax turns scores into probabilities</h3>
      <p>
        Logits can be negative, large, or small. Softmax exponentiates them and
        divides by the sum, producing positive probabilities that sum to one:
      </p>
      <pre class="math-block"><code>{`p[i] = exp(logit[i]) / Σ exp(logit[j])

logits = [2, 1, 0]
exp    = [7.389, 2.718, 1.000]     sum = 11.107
p      = [0.665, 0.245, 0.090]`}</code></pre>
      <p>
        In practice the implementation subtracts the maximum logit before
        exponentiating. That changes nothing mathematically, because the same
        factor cancels in the division, but it prevents <code>exp</code> from
        overflowing for large logits. The integer engine does the same thing
        with a Q12 delta and a table.
      </p>

      <h3 class="subhead">Temperature changes the shape, not the order</h3>
      <p>
        Temperature <code>τ</code> divides the logits before softmax:
      </p>
      <pre class="math-block"><code>{`p[i] = exp(logit[i] / τ) / Σ exp(logit[j] / τ)

same logits [2, 1, 0]

τ = 1.0  → [0.665, 0.245, 0.090]
τ = 0.5  → [0.867, 0.117, 0.016]
τ = 2.0  → [0.506, 0.307, 0.186]`}</code></pre>
      <p>
        Lower temperature sharpens the distribution toward the top choice.
        Higher temperature flattens it toward uniform. Temperature never changes
        which token has the highest probability; it only changes how often the
        others are chosen. At <code>τ → 0</code> the result approaches greedy
        decoding. The engine clamps the temperature to a small positive floor so
        it never divides by zero.
      </p>

      <h3 class="subhead">Top-k</h3>
      <p>
        <strong>Top-k</strong> keeps only the <code>k</code> highest-probability
        tokens, sets the rest to zero, and renormalizes. It prevents the sampler
        from occasionally choosing a token from the long tail. For a 4,096-token
        vocabulary, <code>k = 40</code> is a common choice. The sampler in this
        engine uses a small insertion sort to find those 40 candidates.
      </p>

      <h3 class="subhead">Sampling versus greedy decoding</h3>
      <p>
        <strong>Greedy</strong> decoding always takes the argmax. It is
        deterministic and often repetitive. <strong>Sampling</strong> draws a
        random number in <code>[0, 1)</code> and walks the cumulative
        probabilities until it crosses that number. A token with probability
        0.6 occupies 60% of the interval, so it is chosen about 60% of the time.
      </p>
      <pre class="math-block"><code>{`p = [0.6, 0.3, 0.1]
cumulative = [0.6, 0.9, 1.0]

r = 0.42  → token 0
r = 0.75  → token 1
r = 0.97  → token 2`}</code></pre>
      <p>
        The engine implements this with integer weights and a 32-bit xorshift
        generator. Given the same seed and the same probabilities, it returns
        the same token, which makes tests reproducible. Chapter 19 covers the
        exact integer arithmetic.
      </p>

      <h3 class="subhead">Putting the loop together</h3>
      <pre class="math-block"><code>{`ids = tokenize(prompt)
while not done:
    window = last_T_ids(ids, pad=pad_id)
    logits = model(window)[-1]          # last position only
    probs  = softmax(logits / temperature)
    probs  = top_k(probs, k)
    next   = sample(probs, seed)
    if next is stop_token: break
    ids.append(next)
return detokenize(ids)`}</code></pre>
      <p>
        That is the whole generation algorithm. The model is called once per
        output token in this project, because there is no KV cache. The next
        chapters replace each line with the exact integer implementation.
      </p>

      <Note>
        <strong>Why the model does not “decide” to stop.</strong> The stop token
        is just another token in the vocabulary. If it has the highest
        probability, or is sampled, generation stops. The host enforces the
        stop condition by checking the returned id, not by asking the model.
      </Note>

      <Checkpoint>
        <p>
          Compute the softmax of <code>[2, 1, 0]</code> at{" "}
          <code>τ = 1</code> and <code>τ = 0.5</code> and confirm the values
          above. Then explain why subtracting the maximum before{" "}
          <code>exp</code> does not change the result. Finally, simulate a
          sample for <code>p = [0.5, 0.3, 0.2]</code> with{" "}
          <code>r = 0.71</code> and say which token is chosen.
        </p>
      </Checkpoint>

      <p class="source-line">
        Integer version:{" "}
        <a href="#sampling">chapter 19, integer sampling</a> ·{" "}
        <a href={`${repo}/blob/main/src/tiny_gpt_infer.c`}>
          src/tiny_gpt_infer.c
        </a>
      </p>
    </Chapter>
  );
}
