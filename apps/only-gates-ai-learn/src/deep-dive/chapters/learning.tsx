import {
  Chapter,
  Checkpoint,
  Code,
  DataTable,
  Note,
} from "../components";
import { TrainingLoopDiagram } from "../diagrams";

const repo = "https://github.com/clinuxrulz/only-gates-ai";

const numpyReference = `import numpy as np

def rmsnorm(x, gain, eps=1e-5):
    """x: [T, D] -> [T, D]"""
    return x / np.sqrt((x * x).mean(axis=-1, keepdims=True) + eps) * gain

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(
        0.7978845608028654 * (x + 0.044715 * x ** 3)
    ))

def attention(h, layer, heads):
    """h: [T, D]; weights use the C engine's [in, out] layout."""
    T, D = h.shape
    head_dim = D // heads

    qkv = h @ layer["wqkv"]                 # [T, 3D]
    q, k, v = np.split(qkv, 3, axis=-1)     # each [T, D]

    # Split the model dimension into heads: [T, D] -> [H, T, head_dim]
    q = q.reshape(T, heads, head_dim).transpose(1, 0, 2)
    k = k.reshape(T, heads, head_dim).transpose(1, 0, 2)
    v = v.reshape(T, heads, head_dim).transpose(1, 0, 2)

    scores = q @ k.transpose(0, 2, 1) / np.sqrt(head_dim)  # [H, T, T]

    # Causal mask: position t may only see keys j <= t.
    mask = np.triu(np.ones((T, T), dtype=bool), k=1)
    scores = np.where(mask, -np.inf, scores)

    probs = softmax(scores, axis=-1)         # [H, T, T]
    out = probs @ v                          # [H, T, head_dim]
    out = out.transpose(1, 0, 2).reshape(T, D)
    return out @ layer["wo"]                 # [T, D]

def forward(tokens, p):
    """tokens: list of token ids; p: dict of [in, out] weights."""
    T = len(tokens)
    x = p["token_emb"][tokens] + p["pos_emb"][:T]   # [T, D]

    for layer in p["blocks"]:
        h = rmsnorm(x, layer["g1"])
        x = x + attention(h, layer, p["heads"])

        h = rmsnorm(x, layer["g2"])
        x = x + gelu(h @ layer["w1"]) @ layer["w2"]

    h = rmsnorm(x, p["g_final"])
    return h @ p["token_emb"].T              # [T, V] logits

# A random model with the Alice shapes (V=512, T=16, D=48, L=2, H=2, M=96).
rng = np.random.default_rng(0)
D, M, V, C, H, L = 48, 96, 512, 16, 2, 2
p = {
    "heads": H,
    "token_emb": rng.normal(0, 0.02, (V, D)),
    "pos_emb": rng.normal(0, 0.02, (C, D)),
    "g_final": np.ones(D),
    "blocks": [
        {
            "g1": np.ones(D),
            "wqkv": rng.normal(0, 0.02, (D, 3 * D)),
            "wo": rng.normal(0, 0.02, (D, D)),
            "g2": np.ones(D),
            "w1": rng.normal(0, 0.02, (D, M)),
            "w2": rng.normal(0, 0.02, (M, D)),
        }
        for _ in range(L)
    ],
}

logits = forward([1, 2, 3, 4], p)
print(logits.shape)                         # (4, 512)
print(np.argsort(logits[-1])[-5:][::-1])    # top-5 ids for the last position`;

const torchPacking = `# Convert a trained PyTorch TinyGPT into the reference's [in, out] layout.
# PyTorch nn.Linear stores [out_features, in_features], so transpose each one.
def to_reference(model):
    p = {
        "heads": model.heads,
        "token_emb": model.token_emb.weight.detach().cpu().numpy(),
        "pos_emb": model.pos_emb.weight.detach().cpu().numpy(),
        "g_final": model.g_final.gain.detach().cpu().numpy(),
        "blocks": [],
    }
    for b in model.blocks:
        p["blocks"].append({
            "g1": b.g1.gain.detach().cpu().numpy(),
            "wqkv": b.wqkv.weight.detach().cpu().numpy().T,
            "wo": b.wo.weight.detach().cpu().numpy().T,
            "g2": b.g2.gain.detach().cpu().numpy(),
            "w1": b.w1.weight.detach().cpu().numpy().T,
            "w2": b.w2.weight.detach().cpu().numpy().T,
        })
    return p`;

export function TrainingPrimerChapter() {
  return (
    <Chapter
      id="training-primer"
      number="08"
      kicker="Foundations · learning"
      title="How training turns random numbers into a language model"
    >
      <p>
        At the start of training, every weight is random. The model produces
        nonsense. Training is the process that repeatedly measures how wrong the
        model is and nudges every weight in the direction that makes it slightly
        less wrong. That is the whole idea; the rest is engineering.
      </p>

      <figure class="figure">
        <TrainingLoopDiagram />
        <figcaption>
          Training repeats five steps: sample a batch, run the float forward
          pass, compare the prediction with the true next token, backpropagate
          the error into gradients, and update the weights. Inference throws
          away steps three to five.
        </figcaption>
      </figure>

      <h3 class="subhead">Step 1: turn text into training examples</h3>
      <p>
        The corpus is tokenized into one long stream of ids. A training example
        is a window of <code>T</code> ids plus the same window shifted by one:
      </p>
      <pre class="math-block"><code>{`stream = [the, cat, sat, on, the, mat, ...]

input  = [the, cat, sat, on]
target = [cat, sat, on, the]`}</code></pre>
      <p>
        At every position, the target is the token that came next. One window
        therefore provides <code>T</code> prediction problems at once. The model
        is not trained on a separate “what is a noun” task; next-token
        prediction is the only objective, and useful internal representations
        emerge because they help predict the next token.
      </p>

      <h3 class="subhead">Step 2: measure the error with cross-entropy</h3>
      <p>
        The forward pass produces a probability for every vocabulary token at
        every position. The loss looks only at the probability assigned to the
        correct next token:
      </p>
      <pre class="math-block"><code>{`loss = -log( p[target] )

p[target] = 0.90  → loss = 0.105
p[target] = 0.10  → loss = 2.303
p[target] = 1/4096 → loss = 8.317  (random guessing)`}</code></pre>
      <p>
        This is called <strong>cross-entropy loss</strong>. It punishes
        confident wrong answers much more than uncertain ones. The average loss
        over many examples is a single number that training tries to minimize.
        <strong>Perplexity</strong> is <code>exp(loss)</code>; it is roughly “how
        many equally likely tokens is the model choosing between?”
      </p>

      <h3 class="subhead">Step 3: compute gradients</h3>
      <p>
        For every parameter, the <strong>gradient</strong> says how the loss
        would change if that parameter increased slightly. Backpropagation
        computes all of them efficiently by applying the chain rule from the
        loss backwards through the network. In PyTorch this is one call:
      </p>
      <pre class="math-block"><code>{`loss.backward()   # fills param.grad for every parameter`}</code></pre>
      <p>
        You do not write a separate gradient formula for attention, RMSNorm, and
        GELU. The autograd system records the forward operations and applies the
        chain rule automatically. This is why the training code can be short
        even though the model is deep.
      </p>

      <h3 class="subhead">Step 4: update the weights</h3>
      <p>
        Plain gradient descent moves each weight against its gradient:
      </p>
      <pre class="math-block"><code>{`w = w - learning_rate · gradient`}</code></pre>
      <p>
        The <strong>learning rate</strong> controls the step size. Too large and
        training diverges; too small and it takes forever. This project uses{" "}
        <strong>AdamW</strong>, which keeps a running average of the gradient
        and of the squared gradient for each parameter, and scales the update
        accordingly. AdamW also applies a small weight decay that discourages
        unnecessarily large weights.
      </p>
      <p>
        A <strong>step</strong> is one optimizer update. A <strong>batch</strong>{" "}
        is the group of windows used to compute one gradient. An{" "}
        <strong>epoch</strong> is one pass over the training set. The learning
        rate usually follows a warmup and cosine schedule: it rises briefly at
        the start, then decays smoothly.
      </p>

      <h3 class="subhead">Training versus validation</h3>
      <p>
        If you only measure loss on the training data, a small model can memorize
        it and look perfect while failing on new text. That is{" "}
        <strong>overfitting</strong>. The trainer therefore holds out a
        validation set. Training loss should fall; validation loss should fall
        too, at least for a while. When validation loss starts rising while
        training loss keeps falling, the model is memorizing.
      </p>
      <p>
        <strong>Dropout</strong> randomly zeros some activations during training
        to make memorization harder. It is disabled at inference, which is why
        the C engine has no dropout code at all. The exported model is the
        deterministic version.
      </p>

      <h3 class="subhead">The whole loop in one listing</h3>
      <p>
        With a model definition and a batch sampler, training is only a few
        lines. This is the shape of the real trainer, minus checkpointing and
        the two-stage data pipeline:
      </p>
      <Code
        code={`model = TinyGPT(vocab, context, dim, layers, heads, mlp, dropout=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

for step in range(total_steps):
    inputs, targets = sample_batch(token_stream, context, batch_size)
    logits, loss = model(inputs, targets)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if step % eval_every == 0:
        print(step, loss.item(), estimate_validation_loss(model))`}
        lang="python"
        label="The essential training loop"
      />
      <p>
        Everything else in the trainer supports this loop: the tokenizer and
        data cache, the two stages, the learning-rate schedule, checkpoint
        resume, and the export. None of it changes the five core operations.
      </p>

      <h3 class="subhead">Why this project trains in two stages</h3>
      <DataTable
        caption="The Solid 2 recipe"
        head={["Stage", "Data", "Objective", "Result"]}
        rows={[
          [
            "Language-model (LM) stage",
            "Documentation, notes, and optional stories",
            "Predict the next token across the corpus",
            "General language and domain fluency",
          ],
          [
            "Question-answer (QA) stage",
            "Chat-formatted question/answer pairs",
            "Predict the answer tokens only; prompt tokens are masked",
            "Follows the chat template and answers questions",
          ],
        ]}
      />
      <p>
        The QA stage uses the same next-token loss, but sets the loss mask to{" "}
        <code>-100</code> on the prompt tokens so PyTorch ignores them. The model
        is not rewarded for predicting the user’s question; it is rewarded for
        producing the assistant’s answer and the end token.
      </p>

      <h3 class="subhead">Why training cannot be integer-only</h3>
      <p>
        Gradients are tiny, signed, and change magnitude constantly. AdamW keeps
        running averages that also need a wide dynamic range. A fixed-point
        format with the right range for activations would lose the small
        gradient updates entirely. Training and inference have different
        numerical requirements, so the boundary is the exported checkpoint.
      </p>

      <Note>
        <strong>Training does not “store” the text.</strong> It adjusts
        parameters to reduce prediction error on the text. A small model trained
        on a few documents will reproduce patterns from those documents, but it
        is not a searchable database. This is also why evaluation needs held-out
        prompts, not just prompts copied from the training set.
      </Note>

      <h3 class="subhead">The minimum experiment to run yourself</h3>
      <ol class="steps compact">
        <li>Collect a few hundred lines of text in one style.</li>
        <li>Train a byte-level tokenizer and save it with the model.</li>
        <li>Train a one-layer, one-head model until the loss stops falling.</li>
        <li>Export the weights and run the NumPy reference in chapter 09.</li>
        <li>Only then move to the integer engine and compare top-1 tokens.</li>
      </ol>
      <p>
        Starting tiny is not a shortcut; it is the fastest way to separate a
        modeling bug from a quantization bug. If the float reference is wrong,
        no amount of integer debugging will fix it.
      </p>

      <Checkpoint>
        <p>
          Compute the cross-entropy loss for a correct-token probability of{" "}
          <code>0.5</code> and <code>0.01</code>. Explain the difference between
          a step and an epoch. Then explain why dropout is absent from the C
          engine and why the QA stage masks the prompt tokens from the loss.
        </p>
      </Checkpoint>

      <p class="source-line">
        Source:{" "}
        <a href={`${repo}/blob/main/scripts/train_tiny_gpt_solid2_colab.py`}>
          scripts/train_tiny_gpt_solid2_colab.py
        </a>{" "}
        · <a href={`${repo}/blob/main/docs/COMPUTE.md`}>docs/COMPUTE.md</a>
      </p>
    </Chapter>
  );
}

export function ReferenceChapter() {
  return (
    <Chapter
      id="reference"
      number="09"
      kicker="Foundations · implement it"
      title="A complete forward pass you can run yourself"
    >
      <p>
        The fastest way to understand the architecture is to implement it once
        without a framework. The listing below is a complete, runnable NumPy
        forward pass for the same model the C engine implements. It has no
        training, no tokenizer, and no optimizations. Its only job is to make
        the shapes and the operations concrete.
      </p>

      <Code
        code={numpyReference}
        lang="python"
        file="reference.py"
        collapsible
        open
      />

      <h3 class="subhead">Read it in the same order as the model</h3>
      <ol class="steps compact">
        <li>
          <strong>Embed.</strong> Look up one row per token, add one position
          row, and get <code>[T, D]</code>.
        </li>
        <li>
          <strong>For each block:</strong> normalize, project to Q/K/V, split
          into heads, score, mask, softmax, mix values, project out, add to the
          residual.
        </li>
        <li>
          <strong>Then:</strong> normalize again, expand to <code>M</code>,
          apply GELU, contract back to <code>D</code>, add to the residual.
        </li>
        <li>
          <strong>Finally:</strong> normalize with the last gain and multiply by
          the transposed embedding table to get <code>[T, V]</code> logits.
        </li>
      </ol>

      <h3 class="subhead">Weight orientation matters</h3>
      <p>
        This reference stores weights as <code>[in, out]</code>, matching the C
        engine and the packed checkpoint. PyTorch stores{" "}
        <code>nn.Linear.weight</code> as <code>[out, in]</code>. If you load a
        trained PyTorch model directly, transpose every linear matrix:
      </p>
      <Code
        code={torchPacking}
        lang="python"
        label="Loading a trained model into the reference"
      />
      <p>
        This is the same transpose the exporter performs. Getting it wrong is
        the most common reason a model produces garbage with correct-looking
        shapes. The shape check passes because the dimensions are often equal;
        the numbers are wrong because the inner dot products pair the wrong
        elements.
      </p>

      <h3 class="subhead">What to verify before touching C</h3>
      <DataTable
        caption="A reliable path from floats to integers"
        head={["Check", "How", "Why"]}
        rows={[
          [
            "Shapes",
            "Print every intermediate shape against the table in chapter 10",
            "Catches a missed reshape or head split",
          ],
          [
            "Causal mask",
            "Change a future token and confirm earlier logits do not move",
            "Proves position t cannot see the future",
          ],
          [
            "Float parity",
            "Compare the NumPy logits with the PyTorch logits for the same weights",
            "Proves the reference implements the architecture",
          ],
          [
            "Tied head",
            "Compare a logit with the dot product of the final row and the token embedding",
            "Catches a separate or incorrectly transposed head",
          ],
          [
            "Integer parity",
            "Compare the C top-1 and top-5 with the NumPy float top-1 and top-5",
            "Quantization should not change the ranking wildly",
          ],
        ]}
      />

      <h3 class="subhead">A manual trace you can do on paper</h3>
      <p>
        Take <code>T = 2</code>, <code>D = 2</code>, <code>H = 1</code>,{" "}
        <code>M = 2</code>, and the identity attention from chapter 05. Set the
        MLP weights to the identity and the norm gains to ones. Then:
      </p>
      <pre class="math-block"><code>{`tokens = [0, 1]
x      = token_emb[[0, 1]] + pos_emb[:2]     # [2, 2]
h      = rmsnorm(x, g1)                      # [2, 2]
attn   = attention(h)                        # [2, 2]
x      = x + attn                            # [2, 2]
h      = rmsnorm(x, g2)                      # [2, 2]
m      = gelu(h @ W1) @ W2                   # [2, 2]
x      = x + m                               # [2, 2]
logits = rmsnorm(x, g_final) @ token_emb.T   # [2, V]`}</code></pre>
      <p>
        If you replace every float matrix multiply with a quantized one, you get
        the C engine. The structure does not change at all. That is the key
        insight of the rest of this page: quantization changes the number
        representation, not the dataflow.
      </p>

      <h3 class="subhead">Porting the reference to C</h3>
      <ul>
        <li>
          Replace each matrix with a flat array and a shape. Row-major indexing
          is <code>row * cols + col</code>.
        </li>
        <li>
          Replace each float multiply-add with the integer matmul: quantize the
          activation row, accumulate int8 products in int32, requantize in
          int64.
        </li>
        <li>
          Replace <code>sqrt</code> with <code>int_isqrt_u64</code>,{" "}
          <code>exp</code> with <code>exp_neg_q15_q12</code>, and GELU with the
          lookup table.
        </li>
        <li>
          Keep the same loop order and the same residual adds. If a result
          differs, compare the first buffer that diverges, not the final logits.
        </li>
        <li>
          Test the tiny identity model first. A model with known outputs is
          easier to debug than a trained one.
        </li>
      </ul>

      <Note>
        <strong>Why start from a reference?</strong> The C engine has no
        automatic differentiation, no shape checks, and no exceptions. A float
        reference gives you a trusted set of expected numbers. Without it, every
        integer bug looks like a model bug and every model bug looks like a
        quantization bug.
      </Note>

      <Checkpoint>
        <p>
          Run the reference with a fixed seed, print the shape of every
          intermediate, and verify the causal property: changing the last input
          token must change the last logits row, while changing a later token
          must not change an earlier row. Then load the same weights into the C
          harness and compare the top five tokens. If they differ, find the
          first buffer where the values diverge.
        </p>
      </Checkpoint>
    </Chapter>
  );
}
