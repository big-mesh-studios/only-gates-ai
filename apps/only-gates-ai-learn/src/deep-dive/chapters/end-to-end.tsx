import {
  Chapter,
  Checkpoint,
  Bug,
  Code,
  DataTable,
  Note,
} from "../components";
import { SYMPTOMS } from "../content";

const repo = "https://github.com/clinuxrulz/only-gates-ai";

const generationLoop = `function windowTokens(ids, context, padId) {
  const window = ids.slice(-context);
  while (window.length < context) window.unshift(padId);
  return window;
}

for (let step = 0; step < maxTokens; step++) {
  model.forward(windowTokens(ids, config.context, padId));
  const next = repetitionPenalty > 1
    ? sampleWithRepetitionPenalty(...)
    : model.sample(temperature, topK);
  if (isStopToken(tokenizer, next)) break;
  ids.push(next);
  output.push(next);
}`;

const generationNote = `// The engine only sees ids. Tokenization is a separate layer.
//
// Alice: word-level vocabulary, <unk>=0, <bos>=1, <eos>=2.
// Solid 2: byte-level BPE, special tokens:
//   <|pad|> <|user|> <|assistant|> <|end|> <|facts|>
//
// The Solid 2 prompt is a chat template:
//   <|user|>{question}<|assistant|>
// and generation stops on <|end|> or <|pad|>.`;

export function GenerationChapter() {
  return (
    <Chapter
      id="generation"
      number="11"
      kicker="End-to-end decode"
      title="A tokenizer, a sliding window, and one integer forward pass per token"
    >
      <p>
        The engine does not know what a word is. It receives exactly{" "}
        <code>config.context</code> token ids, runs the forward pass, and
        returns logits. Everything about prompts, chat templates, and stop
        tokens lives in the host wrapper. Keeping that boundary clean is what
        lets the same C engine run Alice, Solid 2, or a future checkpoint.
      </p>

      <h3 class="subhead">Two tokenizers, one engine</h3>
      <DataTable
        caption="What the wrapper hands to the engine"
        head={["Model", "Tokenizer", "Prompt shape", "Stop condition"]}
        rows={[
          [
            "Alice",
            "Word-level, 512 entries",
            "<bos> + words mapped through the vocabulary",
            "<eos> token 2",
          ],
          [
            "Solid 2",
            "Byte-level BPE, 4096 entries",
            "<|user|>question<|assistant|>",
            "<|end|> or <|pad|>",
          ],
        ]}
      />

      <Code code={generationNote} lang="text" label="Tokenizer boundary" />

      <h3 class="subhead">The context window</h3>
      <p>
        The model has a fixed context length. A prompt shorter than the window is
        left-padded; a prompt longer than the window keeps only the last{" "}
        <code>T</code> tokens. The position embedding is indexed by the position
        inside the window, not by the absolute position in the original prompt.
        This is an inference simplification, and it does not exactly match the
        trainer: training right-pads fixed-length examples, while generation
        left-pads so the newest token sits at the last position. A stronger
        design would use one convention on both sides, or an attention mask that
        ignores pad tokens.
      </p>

      <Code code={generationLoop} lang="ts" file="tinyGptSolid2.ts" />

      <h3 class="subhead">What crosses the WASM boundary</h3>
      <ul>
        <li>
          <code>forward</code>: a pointer to <code>T</code> int32 token ids.
        </li>
        <li>
          <code>logits</code>: a pointer to <code>T × V</code> Q16 int32 values.
        </li>
        <li>
          <code>probabilities</code>: a pointer to <code>V</code> Q16 int32
          values, converted to float only for display.
        </li>
        <li>
          <code>sample</code>: temperature in Q16, top-k as an integer; the
          return value is a token id.
        </li>
      </ul>
      <p>
        The only float value that crosses the core WASM API is the temperature,
        which the wrapper rounds to Q16 (<code>round(temperature · 65536)</code>).
        The model arithmetic never crosses back into float. The optional
        repetition-penalty path is different: it re-scores logits in JavaScript,
        as the next section explains.
      </p>

      <h3 class="subhead">Repetition penalty is a host feature</h3>
      <p>
        The Solid 2 wrapper optionally applies a repetition penalty before
        sampling. That path is implemented in JavaScript because it needs the
        recent token set and a float logit adjustment; it is not part of the
        integer engine. If you want a fully integer generation loop, port the
        penalty into the C sampler or leave it out.
      </p>

      <Bug>
        <p>
          If generation repeats the same token forever, check the tokenizer and
          stop-token ids before blaming the model. A one-off BOS/pad id or a
          missing <code>&lt;|end|&gt;</code> will look like a model failure.
          Print the ids, not just the decoded text.
        </p>
      </Bug>

      <Checkpoint>
        <p>
          Encode a prompt, print the ids, run one forward pass, and print the
          top five tokens with probabilities. Then generate ten tokens with a
          fixed seed. Repeat and confirm the same ten ids. Finally, feed a
          prompt longer than the context and confirm the window keeps the last
          <code>T</code> ids.
        </p>
      </Checkpoint>

      <p class="source-line">
        Source:{" "}
        <a href={`${repo}/blob/main/apps/only-gates-ai/src/lib/tinyGptSolid2.ts`}>
          tinyGptSolid2.ts
        </a>{" "}
        ·{" "}
        <a href={`${repo}/blob/main/apps/only-gates-ai/src/lib/tinyGptBpe.ts`}>
          tinyGptBpe.ts
        </a>
      </p>
    </Chapter>
  );
}

export function TrainingChapter() {
  return (
    <Chapter
      id="training"
      number="12"
      kicker="Where the weights come from"
      title="Train the float model, export the contract, then quantize"
    >
      <p>
        The integer engine is only as good as the checkpoint. This chapter is the
        minimum you need to produce one that the engine can load: the data
        pipeline, the two training stages, the tokenizer artifact, and the exact
        export format.
      </p>

      <h3 class="subhead">The Solid 2 recipe</h3>
      <DataTable
        caption="Default configuration and training budget"
        head={["Setting", "Value", "Why"]}
        rows={[
          ["vocab", "4,096", "Byte-level BPE over docs, Q&A, and stories"],
          ["context", "256", "Long enough for a doc section; small enough for the demo"],
          ["dim / layers / heads / mlp", "48 / 3 / 4 / 96", "264,528 parameters"],
          ["LM stage", "5,000 steps", "Learn the corpus distribution"],
          ["QA stage", "100 epochs", "Fit chat-formatted question/answer pairs"],
          ["optimizer", "AdamW + warmup cosine", "Stable small-model training"],
          ["dropout", "0.1 during training, 0 at inference", "The C engine has no dropout"],
        ]}
      />

      <h3 class="subhead">Data and tokenizer</h3>
      <p>
        The corpus is pinned Solid 2 documentation, supplemental Solid 2 notes,
        a curated Q&amp;A set, and optional short stories for general-language
        coherence. The byte-level BPE tokenizer is trained once and written to{" "}
        <code>tokenizer.json</code> before the LM stage. On resume, the trainer
        reuses that file and refuses to continue if the data or tokenizer hash
        changed.
      </p>

      <Note>
        <strong>Tokenization is part of the model contract.</strong> A checkpoint
        and a tokenizer are a pair. Loading Solid 2 weights with the Alice
        word-level vocabulary would produce valid integers and meaningless text.
      </Note>

      <h3 class="subhead">Two stages</h3>
      <p>
        The LM stage samples fixed-length windows from the token stream and
        predicts the next token at every position. The QA stage formats each
        example as a chat sequence, masks the prompt tokens from the loss, and
        trains on the answer plus the end token. Both stages use the same model;
        only the data and the loss mask change.
      </p>

      <h3 class="subhead">Export</h3>
      <p>
        Training ends with <code>pack_weights(model)</code>, which transposes
        every linear matrix, concatenates the buffer in the order from chapter
        1, and writes little-endian float32. <code>checkpoint.json</code> records
        the config, parameter count, tokenizer filename, and training metadata.
        The integer engine reads <code>weights.bin</code> and the config; the
        tokenizer stays on the host.
      </p>

      <Code
        code={`# first session: mount Drive, clone, smoke test, then train
python scripts/train_tiny_gpt_solid2_colab.py \\
  --out /content/drive/MyDrive/tiny-gpt-solid2 \\
  --lm-steps 5000 --qa-epochs 100 \\
  --save-every 100 --max-minutes 300 --fresh

# later session: continue instead of restarting
python scripts/train_tiny_gpt_solid2_colab.py \\
  --out /content/drive/MyDrive/tiny-gpt-solid2 \\
  --lm-steps 5000 --qa-epochs 100 \\
  --save-every 100 --max-minutes 300 --resume`}
        lang="sh"
        label="Training commands"
      />

      <p>
        The checkpoint stores the model, optimizer, scheduler, RNG state, LM
        step, QA epoch and batch cursor, and the exact QA permutation. A free
        Colab session can stop at a time budget and the next session continues
        from the same optimizer step. See <code>docs/COMPUTE.md</code> for the
        reporting protocol.
      </p>

      <Bug>
        <p>
          The most expensive training mistake is a shape-compatible export with a
          missing transpose. The second most expensive is training with dropout
          left on at evaluation. Both produce a model that looks trained and
          fails in the integer engine. Run <code>verify_packed</code> and a
          no-dropout sample before exporting.
        </p>
      </Bug>

      <Checkpoint>
        <p>
          Train a one-layer, one-head toy model on a few hundred lines of text.
          Export it, load it in the example CLI, and confirm the float PyTorch
          logits and the integer logits have the same top-1 token for a fixed
          prompt. Then intentionally remove one transpose and watch it fail; that
          is the debugging skill worth having.
        </p>
      </Checkpoint>

      <p class="source-line">
        Source:{" "}
        <a href={`${repo}/blob/main/scripts/train_tiny_gpt_solid2_colab.py`}>
          scripts/train_tiny_gpt_solid2_colab.py
        </a>{" "}
        ·{" "}
        <a href={`${repo}/blob/main/docs/COMPUTE.md`}>docs/COMPUTE.md</a>
      </p>
    </Chapter>
  );
}

export function VerifyChapter() {
  return (
    <Chapter
      id="verify"
      number="13"
      kicker="Make the difference visible"
      title="Verify against float, then debug by buffer, not by vibe"
    >
      <p>
        Integer inference is not bit-identical to float inference. Quantization
        changes every number. The goal is to preserve the behavior that matters:
        the top prediction, the top-five set, and the character of the
        distribution. Verification is how you prove that instead of hoping.
      </p>

      <h3 class="subhead">The metrics</h3>
      <DataTable
        caption="What to measure between the float and integer paths"
        head={["Metric", "What it tells you", "Healthy sign"]}
        rows={[
          [
            "Top-1 agreement",
            "Does the argmax survive quantization?",
            "High on a working model; not guaranteed on every prompt",
          ],
          [
            "Top-5 overlap",
            "Is the near-tie set stable?",
            "Usually higher than top-1",
          ],
          [
            "Mean absolute probability delta",
            "How much mass moved across the vocabulary",
            "Small compared with the peak probability",
          ],
          [
            "Max probability delta",
            "Worst single-token change",
            "Inspect the token when it is large",
          ],
          [
            "Perplexity / QA validation",
            "Did the model learn the task?",
            "Compare before and after quantization",
          ],
        ]}
      />

      <p>
        The TS comparison helper{" "}
        <code>compareTinyGpt</code> computes top-1, top-5 overlap, and mean/max
        delta between the float and TS int8 paths. The WASM comparison uses the
        same metrics against the C engine. Neither is a proof of correctness,
        but a large regression points at a specific stage.
      </p>

      <h3 class="subhead">Debug by buffer</h3>
      <p>
        The engine exposes its internal Q16 buffers through{" "}
        <code>tiny_gpt_buffer</code>: <code>x</code>, <code>x1</code>,{" "}
        <code>h</code>, <code>qkv</code>, <code>attn</code>, <code>y</code>,{" "}
        <code>z</code>, <code>gz</code>, <code>m</code>, and{" "}
        <code>logits</code>. Compare the first buffer where the integer and float
        paths diverge. Debugging the final logits without looking at the
        intermediates is guesswork.
      </p>

      <Code
        code={`/* The test suite already checks these invariants. */
tiny_gpt_forward(model, tokens);
assert(tiny_gpt_last_error() == TINY_GPT_OK);

const int32_t *logits = tiny_gpt_logits(model);
assert(max(logits) > min(logits));       /* non-degenerate */

tiny_gpt_probabilities(model, INT_Q16_ONE, probabilities);
assert(sum(probabilities) > 65536 - vocab);
assert(sum(probabilities) < 65536 + vocab);

tiny_gpt_set_seed(model, 12345);
const int32_t a = tiny_gpt_sample(model, 0.8 * 65536, 40);
tiny_gpt_set_seed(model, 12345);
const int32_t b = tiny_gpt_sample(model, 0.8 * 65536, 40);
assert(a == b);`}
        lang="c"
        file="tests/test_tiny_gpt.c"
      />

      <h3 class="subhead">A symptom table</h3>
      <DataTable
        caption="Start here when the output is wrong"
        head={["Symptom", "Likely cause", "Check"]}
        rows={SYMPTOMS.map((s) => [s.symptom, s.likely, s.check])}
      />

      <h3 class="subhead">The honesty rule</h3>
      <p>
        Do not claim the integer model is identical to the float model. Do claim
        what you measured: top-1 agreement, top-5 overlap, validation loss, and
        the exact prompt set. The app ships the comparison because the
        difference is part of the result, not a bug to hide.
      </p>

      <Checkpoint>
        <p>
          Build a table of ten prompts. For each, record the float top-1, the
          integer top-1, the top-5 overlap, and the mean delta. Then find the
          prompt with the largest delta and trace it to the first diverging
          buffer. That is the debugging loop.
        </p>
      </Checkpoint>

      <p class="source-line">
        Source:{" "}
        <a href={`${repo}/blob/main/tests/test_tiny_gpt.c`}>
          tests/test_tiny_gpt.c
        </a>{" "}
        ·{" "}
        <a href={`${repo}/blob/main/apps/only-gates-ai/src/lib/tinyGpt.ts`}>
          tinyGpt.ts
        </a>
      </p>
    </Chapter>
  );
}

export function BeyondChapter() {
  return (
    <Chapter
      id="beyond"
      number="14"
      kicker="What comes next"
      title="Performance, KV cache, and the road to gates"
    >
      <p>
        The shipping engine is intentionally simple. It recomputes the full
        context for every generated token, quantizes activations dynamically,
        and keeps every intermediate buffer in Q16. That is enough to prove the
        integer path; it is not the final performance story.
      </p>

      <h3 class="subhead">The current cost</h3>
      <p>
        For each generated token, the engine runs <code>L</code> blocks over{" "}
        <code>T</code> positions. The matrix work is roughly{" "}
        <code>L·T·(4D² + 2DM)</code> multiply-accumulates, and attention is{" "}
        <code>L·T²·D</code>. With the Solid 2 config that is small on a modern
        CPU, but it grows linearly in <code>T</code> for the matmuls and
        quadratically for attention. The implementation has no KV cache, so
        generating 100 tokens does 100 full passes.
      </p>

      <h3 class="subhead">The first optimization: KV cache</h3>
      <p>
        During decode, the keys and values for past positions do not change.
        Quantize them once, store <code>k8</code>, <code>v8</code>, and their
        scales per layer, and compute only the new query. The block still needs
        the current token’s residual, but the QKV projection can be reduced to
        one row, and attention becomes a dot product against the cache. The
        output is mathematically identical if the cache stores the same
        quantized values.
      </p>
      <Note>
        A KV cache changes the memory profile: each layer stores{" "}
        <code>T × D/H × H</code> int8 keys and values plus scales. For Solid 2
        that is small; for a larger model it becomes the dominant memory cost.
        This is why the current simple version is a good teaching reference
        first.
      </Note>

      <h3 class="subhead">From integers to gates</h3>
      <p>
        Every operation in this page has a hardware shape. An 8-bit adder is a
        handful of gates. A comparator is a tree of them. A shift by a constant
        is wiring. A multiply by a constant can become a shift-add network or a
        ROM. A GELU or exp lookup is BRAM plus an address mux. The quantized
        model is already a fixed dataflow; the lowering problem is scheduling
        that graph onto gates and memory, not inventing a new model.
      </p>
      <p>
        The repository’s gate compiler already lowers small symbol-classifier
        circuits to a 16-gate vocabulary. Lowering a full transformer adds wider
        arithmetic, a matrix-tile schedule, on-chip memory planning, and
        token-by-token control. Those are engineering problems, not reasons the
        idea fails.
      </p>

      <h3 class="subhead">What changes for an FPGA</h3>
      <ul>
        <li>
          Dynamic per-row activation quantization needs a max-reduction and a
          reciprocal at runtime. An FPGA implementation may prefer static or
          per-tile scales, or a small divider pipeline.
        </li>
        <li>
          The GELU and DI-Exp tables map naturally to block RAM; the address
          generation and interpolation are the work.
        </li>
        <li>
          Attention is a systolic or tiled array problem; the causal mask lets
          you skip work, but the schedule must not stall.
        </li>
        <li>
          The model is fixed at synthesis time. Changing weights means
          resynthesis or a memory-load path.
        </li>
      </ul>

      <p>
        The honest status is: integer-only inference in WebAssembly is working;
        the gate compiler is a prototype for small circuits; a gate-lowered GPT
        and an FPGA implementation are the roadmap. The deep dive stops at the
        integer engine because that is the part that ships.
      </p>

      <Checkpoint>
        <p>
          Measure tokens per second for a fixed prompt, then sketch a KV-cache
          layout with <code>k8</code>, <code>v8</code>, <code>k_scale</code>,
          and <code>v_scale</code> per layer. Estimate the memory for Solid 2 at
          <code>T=256</code>. Then pick one integer operation and write down its
          gate-level shape.
        </p>
      </Checkpoint>

      <p class="source-line">
        Related:{" "}
        <a href="../#gates">the field notes on the gate path</a> ·{" "}
        <a href={`${repo}/tree/main/packages/only-gates-ai-compiler`}>
          the gate compiler
        </a>
      </p>
    </Chapter>
  );
}
