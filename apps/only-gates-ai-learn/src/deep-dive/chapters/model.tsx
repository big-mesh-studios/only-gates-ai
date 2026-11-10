import { For } from "solid-js";

import {
  Chapter,
  Checkpoint,
  Bug,
  Code,
  DataTable,
  Note,
} from "../components";
import { CONFIGS, TENSORS } from "../content";
import { PackingDiagram, ShapeDiagram } from "../diagrams";

const repo = "https://github.com/clinuxrulz/only-gates-ai";

const PYTorchModel = `class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-5) * self.gain

class Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp: int, dropout: float):
        super().__init__()
        if dim % heads:
            raise ValueError("dim % heads != 0")
        self.heads, self.head_dim = heads, dim // heads
        self.g1, self.g2 = RMSNorm(dim), RMSNorm(dim)
        self.wqkv = nn.Linear(dim, dim * 3, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.w1 = nn.Linear(dim, mlp, bias=False)
        self.w2 = nn.Linear(mlp, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        h = self.g1(x)
        qkv = self.wqkv(h).view(B, T, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
        scores = scores.masked_fill(mask, float("-inf"))
        x = x + self.dropout(
            self.wo((torch.softmax(scores, -1) @ v).transpose(1, 2).reshape(B, T, D))
        )
        h = self.g2(x)
        x = x + self.dropout(self.w2(F.gelu(self.w1(h), approximate="tanh")))
        return x

class TinyGPT(nn.Module):
    def __init__(self, vocab, ctx, dim, layers, heads, mlp, dropout):
        super().__init__()
        self.vocab_size, self.context = vocab, ctx
        self.dim, self.layers_n, self.heads, self.mlp_n = dim, layers, heads, mlp
        self.token_emb = nn.Embedding(vocab, dim)
        self.pos_emb = nn.Embedding(ctx, dim)
        self.blocks = nn.ModuleList(
            [Block(dim, heads, mlp, dropout) for _ in range(layers)]
        )
        self.g_final = RMSNorm(dim)
        self.apply(self._init)
        for b in self.blocks:
            nn.init.normal_(b.wo.weight, std=0.02 / math.sqrt(2 * max(1, layers)))
            nn.init.normal_(b.w2.weight, std=0.02 / math.sqrt(2 * max(1, layers)))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, tokens, targets=None):
        B, T = tokens.shape
        if T > self.context:
            raise ValueError("seq > context")
        x = self.token_emb(tokens) + self.pos_emb(torch.arange(T, device=tokens.device))
        for b in self.blocks:
            x = b(x)
        logits = self.g_final(x) @ self.token_emb.weight.T
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
                ignore_index=-100,
            )
        return logits, loss`;

const PYTorchPack = `def parameter_count(vocab, ctx, dim, layers, mlp):
    return vocab*dim + ctx*dim + dim + layers*(2*dim + 4*dim*dim + 2*dim*mlp)

def pack_weights(model: TinyGPT) -> torch.Tensor:
    parts = [
        model.token_emb.weight.detach().cpu().flatten(),
        model.pos_emb.weight.detach().cpu().flatten(),
        model.g_final.gain.detach().cpu().flatten(),
    ]
    for b in model.blocks:
        parts += [
            b.g1.gain.detach().cpu().flatten(),
            b.wqkv.weight.detach().cpu().T.flatten(),
            b.wo.weight.detach().cpu().T.flatten(),
            b.g2.gain.detach().cpu().flatten(),
            b.w1.weight.detach().cpu().T.flatten(),
            b.w2.weight.detach().cpu().T.flatten(),
        ]
    return torch.cat(parts).to(torch.float32)

def verify_packed(model, packed, cfg):
    V, C, D, L, M = (cfg["vocabSize"], cfg["context"], cfg["dim"],
                     cfg["layers"], cfg["mlp"])
    if packed.numel() != parameter_count(V, C, D, L, M):
        raise ValueError("packed length does not match config")
    off = 0
    def take(n):
        nonlocal off
        value = packed[off:off+n]
        off += n
        return value
    assert torch.equal(take(V*D).view(V, D), model.token_emb.weight.cpu())
    assert torch.equal(take(C*D).view(C, D), model.pos_emb.weight.cpu())
    assert torch.equal(take(D), model.g_final.gain.cpu())
    for b in model.blocks:
        assert torch.equal(take(D), b.g1.gain.cpu())
        assert torch.equal(take(D*D*3).view(D, D*3), b.wqkv.weight.cpu().T)
        assert torch.equal(take(D*D).view(D, D), b.wo.weight.cpu().T)
        assert torch.equal(take(D), b.g2.gain.cpu())
        assert torch.equal(take(D*M).view(D, M), b.w1.weight.cpu().T)
        assert torch.equal(take(M*D).view(M, D), b.w2.weight.cpu().T)
    assert off == packed.numel()`;

export function ModelChapter() {
  return (
    <Chapter
      id="model"
      number="01"
      kicker="The shape of the thing"
      title="A decoder-only transformer small enough to hold in your head"
    >
      <p>
        Both models on this page are the same architecture at two sizes. Tokens
        become learned embeddings, pass through pre-norm blocks, and are
        projected back onto the vocabulary with the same embedding matrix. There
        are no biases, no LayerNorm, no learned position table beyond the small
        embedding, and no operator that lacks an integer implementation.
      </p>

      <div class="table-wrap">
        <table>
          <caption>The two real configurations</caption>
          <thead>
            <tr>
              <th scope="col">Model</th>
              <th scope="col">Role</th>
              <th scope="col">V</th>
              <th scope="col">T</th>
              <th scope="col">D</th>
              <th scope="col">L</th>
              <th scope="col">H</th>
              <th scope="col">M</th>
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
                  <td>{row.params.toLocaleString()}</td>
                </tr>
              )}
            </For>
          </tbody>
        </table>
      </div>

      <p>
        The parameter count is{" "}
        <code>V·D + T·D + D + L·(2D + 4D² + 2D·M)</code>. The first three terms
        are the token table, position table, and final norm gain. Each block has
        two norm gains of size <code>D</code>, a <code>D×3D</code> QKV matrix, a{" "}
        <code>D×D</code> output matrix, a <code>D×M</code> up-projection, and an{" "}
        <code>M×D</code> down-projection. For Solid 2 that is 264,528 values, or
        about 1.06 MB as float32.
      </p>

      <figure class="figure">
        <ShapeDiagram />
        <figcaption>
          The whole model is one embedding sum, a stack of identical blocks, one
          final norm, and a tied output head. The horizontal axis is the token
          position; the vertical axis is the model dimension.
        </figcaption>
      </figure>

      <h3 class="subhead">The tensor inventory</h3>
      <DataTable
        caption="Storage layout and scale granularity"
        head={["Tensor", "Shape", "Storage", "Scale", "Why"]}
        rows={TENSORS.map((t) => [t.name, t.shape, t.storage, t.scale, t.note])}
      />

      <p>
        Two details matter more than they look. First, the linear weights are
        stored as <code>[in, out]</code>, not PyTorch’s <code>[out, in]</code>.
        The C inner loop reads contiguous input rows and writes contiguous output
        columns, so the exporter transposes every linear weight. Second, the
        output head is not a separate learned matrix: it is the token embedding
        transposed to <code>[D, V]</code>, and it reuses the embedding’s row
        scales as its column scales. This is the “tied embeddings” trick.
      </p>

      <figure class="figure">
        <PackingDiagram />
        <figcaption>
          The checkpoint is a flat float32 buffer. The order is the contract
          between the trainer and the C loader.
        </figcaption>
      </figure>

      <h3 class="subhead">What one token costs</h3>
      <p>
        For each layer and each position, the engine does a <code>D×3D</code>{" "}
        matmul, a <code>D×D</code> matmul, a <code>D×M</code> matmul, and an{" "}
        <code>M×D</code> matmul, plus attention. With <code>D=48</code>,{" "}
        <code>M=96</code>, and <code>T=256</code>, the matrix work dominates.
        Attention is <code>T²·D/H</code> per head, but the small model keeps the
        whole context in cache and the matmuls still dominate.
      </p>
      <Note>
        The current engine recomputes the entire context for every generated
        token. That is deliberately simple. Chapter 14 sketches the KV-cache
        change and explains why it is not in the shipping path yet.
      </Note>

      <Checkpoint>
        <p>
          Compute the parameter count for Alice by hand:{" "}
          <code>512·48 + 16·48 + 48 + 2·(2·48 + 4·48² + 2·48·96)</code>. You
          should get 62,448. Then confirm the packed buffer length is exactly
          that many float32 values.
        </p>
      </Checkpoint>
    </Chapter>
  );
}

export function FloatChapter() {
  return (
    <Chapter
      id="float"
      kicker="Build the reference first"
      number="02"
      title="Train in float, then export a buffer the integer engine can read"
    >
      <p>
        You cannot verify an integer engine without a float reference. The
        trainer is also where the model shape is defined, so the two sides must
        agree on every detail: pre-norm order, no biases, GELU tanh, tied
        embeddings, learned positions, and the exact packing order.
      </p>

      <p>
        This is the real model definition from{" "}
        <code>scripts/train_tiny_gpt_solid2_colab.py</code>, abbreviated only in
        the data-loading code. It is the same architecture the C engine
        implements.
      </p>

      <Code
        code={PYTorchModel}
        lang="python"
        file="scripts/train_tiny_gpt_solid2_colab.py"
        collapsible
      />

      <h3 class="subhead">Why the forward pass is shaped this way</h3>
      <ul>
        <li>
          <strong>Pre-norm.</strong> <code>h = RMSNorm(x)</code>, then attention
          or MLP, then <code>x = x + sublayer(h)</code>. The integer engine can
          reuse the same residual buffer without a separate normalization pass
          at the end of the block.
        </li>
        <li>
          <strong>No biases.</strong> Every linear layer is a matrix multiply.
          That removes an add and a second scale from every output.
        </li>
        <li>
          <strong>GELU tanh.</strong> The tanh approximation is smooth and
          bounded, which makes it a good lookup table. The exact erf form would
          need a wider table for the same error.
        </li>
        <li>
          <strong>Tied embeddings.</strong> The final projection is{" "}
          <code>g_final(x) @ token_emb.weight.T</code>. The C loader can build
          the transposed head once and reuse the embedding scales.
        </li>
        <li>
          <strong>Learned absolute positions.</strong> A <code>T×D</code> table
          is added to the token embeddings. No rotary or ALiBi path is needed,
          which keeps the integer engine smaller.
        </li>
      </ul>

      <h3 class="subhead">The export contract</h3>
      <p>
        PyTorch stores <code>nn.Linear.weight</code> as{" "}
        <code>[out_features, in_features]</code>. The C engine wants{" "}
        <code>[in_features, out_features]</code> so the inner loop can walk a
        contiguous input row. The exporter transposes each matrix and then
        flattens. The resulting order is:
      </p>
      <ol class="steps compact">
        <li><code>token_emb</code> · <code>V×D</code></li>
        <li><code>pos_emb</code> · <code>T×D</code></li>
        <li><code>g_final</code> · <code>D</code></li>
        <li>
          for each layer: <code>g1</code>, <code>wqkv</code>, <code>wo</code>,{" "}
          <code>g2</code>, <code>w1</code>, <code>w2</code>
        </li>
      </ol>

      <Code
        code={PYTorchPack}
        lang="python"
        label="Packing and verification"
        collapsible
      />

      <Bug>
        <p>
          The most common checkpoint bug is a missing transpose. The shapes can
          still line up, the C code runs, and the output is nonsense. Always
          assert <code>packed[k*out_cols + j] == W[j][k]</code> on a small
          matrix before training a real model.
        </p>
      </Bug>

      <h3 class="subhead">What training does, in one paragraph</h3>
      <p>
        The trainer builds a token stream, samples fixed-length windows, runs the
        float forward pass, computes cross-entropy against the next token, and
        backpropagates through every matrix and gain. AdamW updates the weights.
        The Solid 2 recipe has two stages: a language-model stage over the docs
        and stories, then a Q&amp;A fine-tune over chat-formatted examples. The
        exported artifact is only the final weights; optimizer state stays
        behind.
      </p>

      <Note>
        <strong>Why not train in integers?</strong> Gradients need dynamic range
        and fractional updates. The integer engine is an inference format, not a
        training format. The boundary is the checkpoint.
      </Note>

      <Checkpoint>
        <p>
          Train a one-layer toy model on a tiny corpus, export{" "}
          <code>weights.bin</code> and <code>checkpoint.json</code>, and load it
          with the example CLI. Before touching the integer path, confirm the
          float model’s logits match the Python forward pass for the same token
          window.
        </p>
      </Checkpoint>

      <p class="source-line">
        Source:{" "}
        <a href={`${repo}/blob/main/scripts/train_tiny_gpt_solid2_colab.py`}>
          scripts/train_tiny_gpt_solid2_colab.py
        </a>{" "}
        ·{" "}
        <a href={`${repo}/blob/main/scripts/train-tiny-gpt.mjs`}>
          scripts/train-tiny-gpt.mjs
        </a>{" "}
        (dependency-free reference trainer)
      </p>
    </Chapter>
  );
}
