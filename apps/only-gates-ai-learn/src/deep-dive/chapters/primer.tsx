import {
  Chapter,
  Checkpoint,
  Code,
  DataTable,
  Note,
} from "../components";
import { NextTokenDiagram } from "../diagrams";

const repo = "https://github.com/clinuxrulz/only-gates-ai";

export function MathChapter() {
  return (
    <Chapter
      id="math"
      number="01"
      kicker="Foundations · the math"
      title="The small amount of math a GPT actually uses"
    >
      <p>
        You do not need calculus, statistics, or a machine-learning course to
        read this page. A GPT is built from five ideas: vectors, dot products,
        matrix multiplication, a normalizing function called softmax, and
        derivatives used during training. Everything else is bookkeeping. This
        chapter defines those ideas in the order you will meet them.
      </p>

      <h3 class="subhead">Numbers, vectors, and matrices</h3>
      <p>
        A <strong>scalar</strong> is one number, such as <code>3.5</code>. A{" "}
        <strong>vector</strong> is an ordered list of numbers, such as{" "}
        <code>[1.0, -2.0, 0.5]</code>. A <strong>matrix</strong> is a
        rectangular grid of numbers. In this tutorial, a vector is a row (or a
        column) and a matrix is a table of rows.
      </p>
      <p>
        The model constantly uses vectors to represent one token at one
        position. If a vector has <code>D</code> numbers, we say the{" "}
        <strong>model dimension</strong> is <code>D</code>. The numbers inside
        the vector are not individually labelled “noun” or “colour”; training
        chooses them so that useful distinctions become easy for later
        operations to read.
      </p>

      <DataTable
        caption="The notation used in every later chapter"
        head={["Notation", "Meaning", "Example"]}
        rows={[
          ["V", "Vocabulary size: how many token ids exist", "512 or 4096"],
          ["T", "Context length: how many token positions the model reads", "16 or 256"],
          ["D", "Model width: numbers per token vector", "48"],
          ["L", "Number of transformer blocks stacked together", "2 or 3"],
          ["H", "Number of attention heads", "2 or 4"],
          ["M", "Hidden width inside the MLP", "96"],
          ["[T, D]", "A matrix with T rows and D columns", "one vector per position"],
          ["W[in, out]", "A weight matrix with in rows and out columns", "D × 3D for QKV"],
        ]}
      />

      <p>
        A matrix is stored in memory as a flat list, row by row. The value at row{" "}
        <code>i</code> and column <code>j</code> of a matrix with{" "}
        <code>cols</code> columns lives at index <code>i * cols + j</code>. That
        single fact explains most pointer arithmetic in the C engine.
      </p>

      <h3 class="subhead">The dot product measures agreement</h3>
      <p>
        A dot product multiplies two equal-length vectors element by element and
        adds the results:
      </p>
      <pre class="math-block"><code>{`u · v = Σ u[i] · v[i]

[1, 2, 3] · [4, 5, 6]
  = 1·4 + 2·5 + 3·6
  = 4 + 10 + 18
  = 32`}</code></pre>
      <p>
        If two vectors point in similar directions, their dot product is large
        and positive. If they are unrelated, the positive and negative products
        cancel and the result is near zero. If they point in opposite
        directions, the result is negative. Attention and the output head both
        use the dot product as a <em>similarity score</em>.
      </p>

      <h3 class="subhead">Matrix multiplication is many dot products</h3>
      <p>
        Multiplying matrix <code>A</code> (shape <code>[m, k]</code>) by matrix{" "}
        <code>B</code> (shape <code>[k, n]</code>) produces{" "}
        <code>C</code> (shape <code>[m, n]</code>). Each output cell is one dot
        product: row <code>i</code> of <code>A</code> against column{" "}
        <code>j</code> of <code>B</code>.
      </p>
      <pre class="math-block"><code>{`C[i][j] = Σ A[i][k] · B[k][j]

A = [1 2 3]      B = [ 7  8]
    [4 5 6]          [ 9 10]
                     [11 12]

C = [1·7 + 2·9 + 3·11   1·8 + 2·10 + 3·12]  = [ 58   64]
    [4·7 + 5·9 + 6·11   4·8 + 5·10 + 6·12]    [139  154]`}</code></pre>
      <p>
        One matrix multiply can compute thousands of dot products at once. That
        is why nearly all of a GPT’s compute is matrix multiplication. The
        integer engine replaces the float multiply-add with an int8 multiply and
        an int32 accumulate, but the shape of the operation is identical.
      </p>

      <h3 class="subhead">Exponentials and softmax</h3>
      <p>
        <code>exp(x)</code>, written <code>eˣ</code>, turns any real number into
        a positive number. It is 1 at <code>x = 0</code>, grows quickly for
        positive inputs, and approaches 0 for negative inputs. <strong>Softmax</strong>{" "}
        uses it to turn a list of arbitrary scores into a probability
        distribution:
      </p>
      <pre class="math-block"><code>{`softmax(z)[i] = exp(z[i]) / Σ exp(z[j])

softmax([2, 1, 0])
  = [7.389, 2.718, 1.000] / 11.107
  = [0.665, 0.245, 0.090]

The outputs are positive and sum to 1.0.`}</code></pre>
      <p>
        The largest score gets the most probability mass, but the others still
        receive some. A <strong>temperature</strong> divides the scores before
        softmax and changes how sharp the result is; chapter 07 explains that in
        detail. In the integer engine, the exponential becomes a small lookup
        table and the division becomes an integer normalization.
      </p>

      <h3 class="subhead">Averages, square roots, and logarithms</h3>
      <p>
        Three more functions appear often:
      </p>
      <ul>
        <li>
          <strong>Mean</strong>: add the values and divide by the count. The
          average of <code>[1, 3, 5]</code> is <code>3</code>.
        </li>
        <li>
          <strong>Root mean square</strong>: square each value, take the mean,
          then take the square root. For <code>[3, 4]</code> the squares are{" "}
          <code>[9, 16]</code>, the mean is <code>12.5</code>, and the RMS is
          about <code>3.536</code>. RMSNorm uses this to measure the size of a
          vector.
        </li>
        <li>
          <strong>Natural logarithm</strong>: the inverse of <code>exp</code>.{" "}
          <code>log(1) = 0</code>, <code>log(e) = 1</code>, and{" "}
          <code>log(0.1) ≈ -2.303</code>. Training uses{" "}
          <code>-log(probability)</code> as the loss; a confident correct
          prediction has a small loss, and an unlikely correct prediction has a
          large loss.
        </li>
      </ul>

      <h3 class="subhead">Slopes, gradients, and the chain rule</h3>
      <p>
        A <strong>derivative</strong> answers: if I nudge this input a little,
        how much does the output move? For <code>f(x) = x²</code>, the
        derivative is <code>2x</code>. At <code>x = 3</code>, a tiny increase in{" "}
        <code>x</code> increases <code>f</code> about six times as fast.
      </p>
      <p>
        Training adjusts every weight in the direction that lowers the loss. The{" "}
        <strong>gradient</strong> is the vector of derivatives with respect to
        all weights. The <strong>chain rule</strong> says that for composed
        functions you multiply the local slopes. For example, if{" "}
        <code>y = w·x</code> and <code>loss = (y - target)²</code>, then
      </p>
      <pre class="math-block"><code>{`d loss / d w = 2 · (w·x - target) · x`}</code></pre>
      <p>
        Backpropagation is just the chain rule applied through every operation
        in the network, from the loss backwards to each weight. You do not have
        to derive it by hand to use the trainer, but knowing what a gradient is
        makes the training chapter readable.
      </p>

      <Note>
        <strong>Why this matters for integers.</strong> Every operation above is
        either exact integer arithmetic (add, multiply, compare, shift) or a
        function that can be approximated by a table (exp, square root, GELU).
        The model’s behaviour is fixed once training ends, so inference can
        replace the float versions with integer versions that are close enough
        for the same decisions to survive.
      </Note>

      <Checkpoint>
        <p>
          By hand: compute <code>[1, 2, 3] · [4, 5, 6]</code>; compute the
          matrix product above; compute <code>softmax([1, 0])</code>; and compute{" "}
          <code>d/dw (w·2 - 3)²</code> at <code>w = 2</code>. You should get{" "}
          <code>32</code>, the 2×2 matrix <code>[[58, 64], [139, 154]]</code>,{" "}
          <code>[0.731, 0.269]</code>, and <code>4</code>. If those are
          comfortable, you have all the math this page requires.
        </p>
      </Checkpoint>
    </Chapter>
  );
}

export function WhatGptChapter() {
  return (
    <Chapter
      id="what"
      number="02"
      kicker="Foundations · the idea"
      title="A GPT is a next-token probability machine"
    >
      <p>
        Strip away the marketing and a GPT is one function. You give it a
        sequence of tokens, and it returns a score for every token in its
        vocabulary. The score says how likely that token is to come next. That
        is the entire model. Generation is a loop around this function:
      </p>

      <figure class="figure">
        <NextTokenDiagram />
        <figcaption>
          One forward pass turns the current token window into a probability
          distribution over the vocabulary. The host picks one token, appends
          it, and calls the model again. Nothing else is hidden in the loop.
        </figcaption>
      </figure>

      <ol class="steps compact">
        <li>Turn the prompt into token ids.</li>
        <li>Run the model on the current window of ids.</li>
        <li>Get one probability for every token id in the vocabulary.</li>
        <li>Choose a token, usually by sampling from that distribution.</li>
        <li>Append the token and repeat until a stop condition.</li>
      </ol>

      <h3 class="subhead">What the letters mean</h3>
      <DataTable
        caption="GPT is an acronym plus an architecture"
        head={["Word", "Meaning", "In this project"]}
        rows={[
          ["Generative", "It produces new tokens one at a time", "A generation loop in the host"],
          ["Pre-trained", "Weights were learned from text before any fine-tuning", "An LM stage, then a Q&A stage"],
          ["Transformer", "The block stack that processes the tokens", "Pre-norm, attention, and an MLP per block"],
          ["Decoder-only", "Each position may only look at itself and earlier positions", "The causal mask in attention"],
        ]}
      />

      <p>
        “Decoder-only” matters for generation. The model is trained to predict
        the next token from the tokens before it. It is never allowed to look at
        the future during training, because at generation time there is no
        future yet. The rest of the architecture exists to make that prediction
        accurate and trainable.
      </p>

      <h3 class="subhead">A concrete example</h3>
      <p>
        Suppose the vocabulary is only four tokens:{" "}
        <code>sky</code>, <code>blue</code>, <code>dark</code>, and{" "}
        <code>cloud</code>. After reading the prompt “the sky is”, the model
        might return:
      </p>
      <pre class="math-block"><code>{`blue   0.62
dark   0.18
cloud  0.14
sky    0.06`}</code></pre>
      <p>
        The model has not looked up a fact. It has computed a vector for the
        current position and compared it with every token’s embedding. The
        comparison produces four logits, softmax turns them into probabilities,
        and the host samples one. A different sample can produce a different
        sentence, which is why temperature and top-k exist.
      </p>

      <Note>
        <strong>The model has no memory between calls.</strong> A chat appears to
        remember because the host sends the whole conversation back every time.
        The context window is the model’s entire working memory. Once the
        conversation is longer than the window, the oldest tokens are dropped.
      </Note>

      <h3 class="subhead">Parameters are the learned numbers</h3>
      <p>
        A <strong>parameter</strong> (or weight) is a number that training
        adjusts. A GPT is mostly a pile of matrices full of parameters, plus a
        few vectors. The tiny models on this page have 62,448 and 264,528
        parameters. Large language models have billions, but the operations are
        the same; only the shapes and the amount of compute change.
      </p>
      <p>
        The weights encode statistical regularities, not a database of facts.
        That is why a model can be fluent and confidently wrong at the same
        time. It is also why changing the tokenizer without changing the
        weights produces valid arithmetic and meaningless text: the ids no
        longer mean the same thing.
      </p>

      <h3 class="subhead">Why a transformer?</h3>
      <p>
        A simpler model could predict the next token from a fixed window using
        only a matrix multiply. The transformer adds two mechanisms that make
        long-range prediction much better:
      </p>
      <ul>
        <li>
          <strong>Attention</strong> lets each position pull information from
          earlier positions, so a pronoun can use the noun it refers to and a
          question can use the whole prompt.
        </li>
        <li>
          <strong>The MLP</strong> gives each position a nonlinear computation
          of its own, so the model can combine features instead of only
          averaging them.
        </li>
      </ul>
      <p>
        Stack those two mechanisms in a residual stream, repeat the block a few
        times, and normalize the result. That is the entire architecture.
        Chapters 04 to 07 explain each part and what it is for.
      </p>

      <h3 class="subhead">Training versus inference</h3>
      <DataTable
        caption="The two phases, and why only one needs floats"
        head={["Phase", "What happens", "Arithmetic", "Result"]}
        rows={[
          ["Training", "Run the model, measure the error, compute gradients, update weights", "Float32 and gradients", "A checkpoint"],
          ["Inference", "Run the fixed model on new tokens and sample", "Can be integer-only", "Generated text"],
        ]}
      />
      <p>
        Training needs a wide dynamic range because gradients are tiny,
        fractional, and change every step. Inference only needs to reproduce a
        fixed computation. That is the opening for integer arithmetic, and it
        is the subject of this entire deep dive.
      </p>

      <Checkpoint>
        <p>
          Explain in one paragraph why a GPT that predicts one token at a time
          can produce a whole paragraph. Then list the five steps of the
          generation loop and say which step contains all the model arithmetic.
          Finally, explain why the same checkpoint with a different tokenizer
          would produce garbage even though the integers are valid.
        </p>
      </Checkpoint>
    </Chapter>
  );
}

const embedExample = `vocab = 4, dim = 3

token_emb:
  id 0: [ 0.1, -0.2,  0.3]
  id 1: [ 0.0,  0.4, -0.1]
  id 2: [ 0.5,  0.1, -0.3]
  id 3: [-0.2,  0.2,  0.2]

pos_emb:
  pos 0: [0.0, 0.0, 0.0]
  pos 1: [0.1, 0.1, 0.1]

tokens = [2, 0]

x[0] = token_emb[2] + pos_emb[0] = [0.5, 0.1, -0.3]
x[1] = token_emb[0] + pos_emb[1] = [0.2, -0.1, 0.4]

X = [[0.5, 0.1, -0.3],
     [0.2, -0.1, 0.4]]        # shape [T=2, D=3]`;

export function TokensChapter() {
  return (
    <Chapter
      id="tokens"
      number="03"
      kicker="Foundations · the input"
      title="Tokens, token ids, and the embedding lookup table"
    >
      <p>
        A neural network cannot read characters. It reads integers. A{" "}
        <strong>tokenizer</strong> is the separate program that turns text into
        a sequence of token ids. A <strong>token</strong> is a chunk of text:
        a word, part of a word, a byte, or a special marker. The set of all
        tokens the model knows is its <strong>vocabulary</strong>, of size{" "}
        <code>V</code>.
      </p>

      <h3 class="subhead">Three common tokenizers</h3>
      <DataTable
        caption="The tokenizer choice is a modeling decision"
        head={["Kind", "How it splits text", "Vocabulary", "Used here"]}
        rows={[
          ["Character", "One token per character", "Small", "Rarely"],
          ["Word-level", "One token per word, unknown words share a token", "Small to medium", "Alice, V=512"],
          ["Byte-level BPE", "Starts from bytes and merges common pairs", "Medium", "Solid 2, V=4096"],
        ]}
      />
      <p>
        BPE stands for byte-pair encoding. It starts with individual bytes and
        repeatedly merges the most frequent adjacent pair into a new token. The
        result is a vocabulary that can represent any byte sequence while using
        longer tokens for common words. The details do not matter for the
        forward pass; the engine only ever sees ids.
      </p>
      <p>
        Special tokens are ordinary vocabulary entries with reserved ids. This
        project uses <code>&lt;pad&gt;</code> for padding,{" "}
        <code>&lt;bos&gt;</code> for the beginning of a sequence,{" "}
        <code>&lt;eos&gt;</code> for the end, and chat markers such as{" "}
        <code>&lt;|user|&gt;</code> and <code>&lt;|assistant|&gt;</code>. They
        let the host tell the model where a turn starts and stops.
      </p>

      <h3 class="subhead">An id is an index, not a quantity</h3>
      <p>
        Token id 400 is not “twice as much” as token id 200. The id is only a
        row number in a table. The table is called the{" "}
        <strong>token embedding</strong>. It has one row per vocabulary entry,
        and each row has <code>D</code> numbers:
      </p>
      <pre class="math-block"><code>{`token_emb: shape [V, D]
row for token id 2: [0.5, 0.1, -0.3]

x = token_emb[2]`}</code></pre>
      <p>
        The embedding row is learned. At the start of training it is random. By
        the end, tokens that need similar treatment tend to have rows that
        produce similar scores in later operations. Nobody assigns the meaning
        of each coordinate by hand.
      </p>

      <h3 class="subhead">Position embeddings tell the model where it is</h3>
      <p>
        Attention and the MLP are otherwise blind to order: they see a set of
        token vectors. To distinguish “dog bites man” from “man bites dog”, the
        model adds a second learned vector that depends on the position in the
        window:
      </p>
      <pre class="math-block"><code>{`pos_emb: shape [T, D]

x[t] = token_emb[tokens[t]] + pos_emb[t]`}</code></pre>
      <p>
        The addition works because both vectors have the same width and the
        model can learn to separate the two signals. This project uses a learned
        absolute position table. Other models use rotary or relative positions;
        the integer engine keeps the simpler table because it is easier to
        quantize and cheaper to run.
      </p>

      <h3 class="subhead">A complete embedding example</h3>
      <Code code={embedExample} lang="text" label="Embedding a two-token prompt" />
      <p>
        The output <code>X</code> is the input to the first transformer block.
        It has one row per position and one column per model dimension. Every
        later operation reads and writes matrices with the same{" "}
        <code>[T, D]</code> shape, with a few detours into <code>[T, 3D]</code>{" "}
        for attention and <code>[T, M]</code> for the MLP.
      </p>

      <h3 class="subhead">The context window</h3>
      <p>
        <code>T</code> is fixed at training time. The model has a position
        embedding for each slot and is trained on windows of exactly that
        length. At inference the host must therefore supply exactly{" "}
        <code>T</code> ids: it pads short prompts and keeps only the last{" "}
        <code>T</code> ids of long ones. This project left-pads so the newest
        token sits in the last slot.
      </p>
      <Note>
        <strong>A checkpoint and a tokenizer are a pair.</strong>{" "}
        <code>weights.bin</code> does not contain the tokenizer. If you load a
        checkpoint with the wrong vocabulary, every id maps to a different
        embedding row, and the model produces confident nonsense. Always ship
        the tokenizer next to the weights.
      </Note>

      <h3 class="subhead">What the integer engine stores</h3>
      <p>
        The C engine stores <code>token_emb</code> as int8 codes with one Q16
        scale per row, and <code>pos_emb</code> as int8 codes with one Q16 scale
        per row. Embedding a token is a table lookup, a multiply by the row
        scale, and an add. The scale is what turns a small integer code back
        into a useful real number. Chapter 13 explains how those scales are
        chosen.
      </p>

      <Checkpoint>
        <p>
          Using the table above, compute <code>X</code> for the token sequence{" "}
          <code>[3, 1]</code>. Then explain why token id 1 cannot be treated as
          “half of token id 2”. Finally, decide what the host should do with a
          prompt of 300 tokens when <code>T = 256</code>.
        </p>
      </Checkpoint>

      <p class="source-line">
        Host-side tokenizers:{" "}
        <a href={`${repo}/blob/main/apps/only-gates-ai/src/lib/tinyGptBpe.ts`}>
          tinyGptBpe.ts
        </a>{" "}
        ·{" "}
        <a href={`${repo}/blob/main/apps/only-gates-ai/src/lib/tinyGptSolid2.ts`}>
          tinyGptSolid2.ts
        </a>
      </p>
    </Chapter>
  );
}
