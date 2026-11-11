export function PipelineDiagram() {
  return (
    <svg class="diagram" viewBox="0 0 920 330" role="img" aria-labelledby="dd1t dd1d">
      <title id="dd1t">The integer inference pipeline</title>
      <desc id="dd1d">
        A float32 checkpoint is quantized once at load into int8 weights and Q16
        scales. The integer forward pass keeps activations in Q16, accumulates
        dot products in int32, requantizes in int64, produces Q16 logits, and
        normalizes them to Q15 probabilities before sampling.
      </desc>

      <rect class="d-panel" x="16" y="30" width="164" height="86" rx="12" />
      <text class="d-title" x="98" y="62" text-anchor="middle">checkpoint</text>
      <text class="d-muted" x="98" y="84" text-anchor="middle">float32 weights</text>
      <text class="d-muted" x="98" y="102" text-anchor="middle">tokenizer.json</text>

      <rect class="d-box d-blue-box" x="208" y="30" width="164" height="86" rx="12" />
      <text class="d-title" x="290" y="62" text-anchor="middle">quantize</text>
      <text class="d-muted" x="290" y="84" text-anchor="middle">float allowed here</text>
      <text class="d-muted" x="290" y="102" text-anchor="middle">once, at load</text>

      <rect class="d-box" x="400" y="30" width="164" height="86" rx="12" />
      <text class="d-title" x="482" y="62" text-anchor="middle">int8 + Q16</text>
      <text class="d-muted" x="482" y="84" text-anchor="middle">weights · scales</text>
      <text class="d-muted" x="482" y="102" text-anchor="middle">gains · GELU LUT</text>

      <rect class="d-box d-accent-box" x="592" y="30" width="150" height="86" rx="12" />
      <text class="d-title" x="667" y="62" text-anchor="middle">forward</text>
      <text class="d-muted" x="667" y="84" text-anchor="middle">int8 · Q16</text>
      <text class="d-muted" x="667" y="102" text-anchor="middle">int32 · int64</text>

      <rect class="d-box" x="770" y="30" width="134" height="86" rx="12" />
      <text class="d-title" x="837" y="62" text-anchor="middle">Q16 logits</text>
      <text class="d-muted" x="837" y="84" text-anchor="middle">one row per</text>
      <text class="d-muted" x="837" y="102" text-anchor="middle">position</text>

      <path class="d-line" d="M182 73 L204 73" />
      <path class="d-arrow" d="M198 67 L208 73 L198 79 Z" />
      <path class="d-line" d="M374 73 L396 73" />
      <path class="d-arrow" d="M390 67 L400 73 L390 79 Z" />
      <path class="d-line" d="M566 73 L588 73" />
      <path class="d-arrow" d="M582 67 L592 73 L582 79 Z" />
      <path class="d-line" d="M744 73 L766 73" />
      <path class="d-arrow" d="M760 67 L770 73 L760 79 Z" />

      <rect class="d-panel" x="16" y="154" width="888" height="150" rx="14" />
      <text class="d-title" x="40" y="184">integer forward pass</text>

      <rect class="d-box" x="40" y="202" width="132" height="56" rx="8" />
      <text class="d-text" x="106" y="226" text-anchor="middle">embed</text>
      <text class="d-muted" x="106" y="244" text-anchor="middle">int8 × Q16 → Q16</text>

      <rect class="d-box" x="196" y="202" width="132" height="56" rx="8" />
      <text class="d-text" x="262" y="226" text-anchor="middle">RMSNorm</text>
      <text class="d-muted" x="262" y="244" text-anchor="middle">isqrt + Q16</text>

      <rect class="d-box" x="352" y="202" width="132" height="56" rx="8" />
      <text class="d-text" x="418" y="226" text-anchor="middle">attention</text>
      <text class="d-muted" x="418" y="244" text-anchor="middle">Q12 scores · Q15</text>

      <rect class="d-box" x="508" y="202" width="132" height="56" rx="8" />
      <text class="d-text" x="574" y="226" text-anchor="middle">MLP</text>
      <text class="d-muted" x="574" y="244" text-anchor="middle">GELU LUT · Q16</text>

      <rect class="d-box" x="664" y="202" width="110" height="56" rx="8" />
      <text class="d-text" x="719" y="226" text-anchor="middle">head</text>
      <text class="d-muted" x="719" y="244" text-anchor="middle">D × V</text>

      <rect class="d-box d-blue-box" x="798" y="202" width="82" height="56" rx="8" />
      <text class="d-text" x="839" y="226" text-anchor="middle">softmax</text>
      <text class="d-muted" x="839" y="244" text-anchor="middle">Q15</text>

      <path class="d-line" d="M172 230 L192 230" />
      <path class="d-arrow" d="M186 224 L196 230 L186 236 Z" />
      <path class="d-line" d="M328 230 L348 230" />
      <path class="d-arrow" d="M342 224 L352 230 L342 236 Z" />
      <path class="d-line" d="M484 230 L504 230" />
      <path class="d-arrow" d="M498 224 L508 230 L498 236 Z" />
      <path class="d-line" d="M640 230 L660 230" />
      <path class="d-arrow" d="M654 224 L664 230 L654 236 Z" />
      <path class="d-line" d="M774 230 L794 230" />
      <path class="d-arrow" d="M788 224 L798 230 L788 236 Z" />

      <text class="d-muted" x="460" y="320" text-anchor="middle">
        no float type, float literal, or math.h call after this boundary
      </text>
    </svg>
  );
}

export function ShapeDiagram() {
  return (
    <svg class="diagram" viewBox="0 0 920 360" role="img" aria-labelledby="dd2t dd2d">
      <title id="dd2t">Tensor shapes through the tiny GPT</title>
      <desc id="dd2d">
        Token ids of length T become a T by D embedding. Each pre-norm block
        holds attention and an MLP; the final norm feeds a tied D by V head,
        producing T by V logits.
      </desc>

      <rect class="d-box d-blue-box" x="20" y="142" width="112" height="76" rx="10" />
      <text class="d-text" x="76" y="172" text-anchor="middle">token ids</text>
      <text class="d-muted" x="76" y="194" text-anchor="middle">T</text>

      <rect class="d-box" x="158" y="142" width="146" height="76" rx="10" />
      <text class="d-text" x="231" y="172" text-anchor="middle">embedding</text>
      <text class="d-muted" x="231" y="194" text-anchor="middle">T × D</text>

      <rect class="d-panel" x="330" y="44" width="342" height="272" rx="14" />
      <text class="d-title" x="501" y="74" text-anchor="middle">transformer block × L</text>

      <rect class="d-box" x="352" y="94" width="298" height="46" rx="8" />
      <text class="d-text" x="501" y="114" text-anchor="middle">RMSNorm → Wqkv → attention</text>
      <text class="d-muted" x="501" y="132" text-anchor="middle">T × 3D → T × D</text>

      <rect class="d-box" x="352" y="152" width="298" height="46" rx="8" />
      <text class="d-text" x="501" y="172" text-anchor="middle">+ residual → RMSNorm → MLP</text>
      <text class="d-muted" x="501" y="190" text-anchor="middle">T × M → T × D</text>

      <rect class="d-box" x="352" y="210" width="298" height="46" rx="8" />
      <text class="d-text" x="501" y="230" text-anchor="middle">+ residual</text>
      <text class="d-muted" x="501" y="248" text-anchor="middle">T × D stays Q16</text>

      <text class="d-muted" x="501" y="288" text-anchor="middle">
        pre-norm · no biases · learned positions
      </text>

      <rect class="d-box" x="698" y="142" width="96" height="76" rx="10" />
      <text class="d-text" x="746" y="172" text-anchor="middle">final</text>
      <text class="d-text" x="746" y="192" text-anchor="middle">RMSNorm</text>

      <rect class="d-box d-accent-box" x="820" y="142" width="80" height="76" rx="10" />
      <text class="d-text" x="860" y="172" text-anchor="middle">logits</text>
      <text class="d-muted" x="860" y="194" text-anchor="middle">T × V</text>

      <path class="d-line" d="M132 180 L154 180" />
      <path class="d-arrow" d="M148 174 L158 180 L148 186 Z" />
      <path class="d-line" d="M304 180 L326 180" />
      <path class="d-arrow" d="M320 174 L330 180 L320 186 Z" />
      <path class="d-line" d="M672 180 L694 180" />
      <path class="d-arrow" d="M688 174 L698 180 L688 186 Z" />
      <path class="d-line" d="M794 180 L816 180" />
      <path class="d-arrow" d="M810 174 L820 180 L810 186 Z" />

      <text class="d-muted" x="460" y="342" text-anchor="middle">
        Alice: V=512, T=16, D=48, L=2, H=2, M=96. Solid 2: V=4096, T=256, D=48, L=3, H=4, M=96.
      </text>
    </svg>
  );
}

export function QuantizationDiagram() {
  return (
    <svg class="diagram" viewBox="0 0 920 340" role="img" aria-labelledby="dd3t dd3d">
      <title id="dd3t">Per-column and per-row quantization</title>
      <desc id="dd3d">
        Linear weights use one scale per output column. Embedding tables use one
        scale per row. A value is stored as an int8 code multiplied by its Q16
        scale.
      </desc>

      <rect class="d-panel" x="18" y="24" width="430" height="292" rx="14" />
      <text class="d-title" x="42" y="54">linear weights · per column</text>
      <text class="d-muted" x="42" y="76">W is [in, out]; every output has its own scale</text>

      <g transform="translate(46 100)">
        <rect class="d-box" x="0" y="0" width="330" height="20" rx="3" />
        <rect class="d-box" x="0" y="28" width="330" height="20" rx="3" />
        <rect class="d-box" x="0" y="56" width="330" height="20" rx="3" />
        <text class="d-muted" x="10" y="14">in = 0</text>
        <text class="d-muted" x="10" y="42">in = 1</text>
        <text class="d-muted" x="10" y="70">in = 2</text>
        <path class="d-line" d="M70 0 L70 76" />
        <path class="d-line" d="M140 0 L140 76" />
        <path class="d-line" d="M210 0 L210 76" />
        <path class="d-line" d="M280 0 L280 76" />
      </g>
      <text class="d-muted" x="46" y="206">scale[j] = max_abs(column j) / 127</text>
      <text class="d-muted" x="46" y="226">q[k][j] = round(W[k][j] / scale[j])</text>
      <text class="d-muted" x="46" y="250">one outlier cannot ruin the whole matrix</text>

      <rect class="d-panel" x="472" y="24" width="430" height="292" rx="14" />
      <text class="d-title" x="496" y="54">embeddings · per row</text>
      <text class="d-muted" x="496" y="76">token_emb is [vocab, dim]; each token has its own scale</text>

      <g transform="translate(500 100)">
        <rect class="d-box" x="0" y="0" width="330" height="24" rx="3" />
        <rect class="d-box" x="0" y="32" width="330" height="24" rx="3" />
        <rect class="d-box" x="0" y="64" width="330" height="24" rx="3" />
        <text class="d-muted" x="10" y="16">token 0 · scale 0</text>
        <text class="d-muted" x="10" y="48">token 1 · scale 1</text>
        <text class="d-muted" x="10" y="80">token 2 · scale 2</text>
      </g>
      <text class="d-muted" x="500" y="206">scale[row] = max_abs(row) / 127</text>
      <text class="d-muted" x="500" y="226">q[row][d] = round(E[row][d] / scale[row])</text>
      <text class="d-muted" x="500" y="250">the tied head reuses these exact row scales</text>

      <text class="d-muted" x="460" y="334" text-anchor="middle">
        dequantize: real ≈ int8_code × Q16_scale / 65536
      </text>
    </svg>
  );
}

export function MatmulDiagram() {
  return (
    <svg class="diagram" viewBox="0 0 920 330" role="img" aria-labelledby="dd4t dd4d">
      <title id="dd4t">Integer matrix multiplication and requantization</title>
      <desc id="dd4d">
        A Q16 activation row is quantized to int8 with one scale. It is dotted
        against int8 weight columns and accumulated in int32. The accumulator is
        multiplied by both scales in int64 and shifted by 16 to produce Q16.
      </desc>

      <rect class="d-box d-blue-box" x="20" y="42" width="146" height="76" rx="10" />
      <text class="d-text" x="93" y="70" text-anchor="middle">Q16 x row</text>
      <text class="d-muted" x="93" y="92" text-anchor="middle">1 × K</text>

      <rect class="d-box" x="196" y="42" width="146" height="76" rx="10" />
      <text class="d-text" x="269" y="70" text-anchor="middle">quantize row</text>
      <text class="d-muted" x="269" y="92" text-anchor="middle">int8 + act_scale</text>

      <rect class="d-box" x="372" y="42" width="146" height="76" rx="10" />
      <text class="d-text" x="445" y="70" text-anchor="middle">int8 dot</text>
      <text class="d-muted" x="445" y="92" text-anchor="middle">K terms → int32</text>

      <rect class="d-box" x="548" y="42" width="146" height="76" rx="10" />
      <text class="d-text" x="621" y="70" text-anchor="middle">requantize</text>
      <text class="d-muted" x="621" y="92" text-anchor="middle">int64 · shift 16</text>

      <rect class="d-box d-accent-box" x="724" y="42" width="176" height="76" rx="10" />
      <text class="d-text" x="812" y="70" text-anchor="middle">Q16 output row</text>
      <text class="d-muted" x="812" y="92" text-anchor="middle">1 × N</text>

      <path class="d-line" d="M166 80 L192 80" />
      <path class="d-arrow" d="M186 74 L196 80 L186 86 Z" />
      <path class="d-line" d="M342 80 L368 80" />
      <path class="d-arrow" d="M362 74 L372 80 L362 86 Z" />
      <path class="d-line" d="M518 80 L544 80" />
      <path class="d-arrow" d="M538 74 L548 80 L538 86 Z" />
      <path class="d-line" d="M694 80 L720 80" />
      <path class="d-arrow" d="M714 74 L724 80 L714 86 Z" />

      <rect class="d-panel" x="20" y="150" width="880" height="150" rx="14" />
      <text class="d-title" x="44" y="180">the arithmetic, in order</text>
      <text class="d-text" x="44" y="210">acc = Σ q8_x[k] · q8_w[k][j]</text>
      <text class="d-text" x="44" y="236">product = (int64)acc · act_scale · weight_scale[j]</text>
      <text class="d-text" x="44" y="262">y_q16[j] = round(product / 2^16)</text>
      <text class="d-muted" x="44" y="284">act_scale is in Q16 units; weight_scale is a Q16 fraction</text>
      <text class="d-muted" x="620" y="210">max |acc| here: 96 · 127 · 127 = 1,548,384</text>
      <text class="d-muted" x="620" y="236">int32 limit: 2,147,483,647</text>
      <text class="d-muted" x="620" y="262">the accumulator has ~1,300× headroom</text>
    </svg>
  );
}

export function AttentionDiagram() {
  return (
    <svg class="diagram" viewBox="0 0 920 380" role="img" aria-labelledby="dd5t dd5d">
      <title id="dd5t">Integer attention</title>
      <desc id="dd5d">
        Q, K, and V are quantized per head and per token. Scores become Q12
        nats, future positions are masked, DI-Exp produces Q15 probabilities,
        and the weighted values are requantized to Q16.
      </desc>

      <rect class="d-box d-blue-box" x="18" y="34" width="130" height="70" rx="9" />
      <text class="d-text" x="83" y="62" text-anchor="middle">QKV</text>
      <text class="d-muted" x="83" y="82" text-anchor="middle">T × 3D · Q16</text>

      <rect class="d-box" x="172" y="34" width="146" height="70" rx="9" />
      <text class="d-text" x="245" y="62" text-anchor="middle">per-head quant</text>
      <text class="d-muted" x="245" y="82" text-anchor="middle">q8 · k8 · v8</text>

      <rect class="d-box" x="342" y="34" width="146" height="70" rx="9" />
      <text class="d-text" x="415" y="62" text-anchor="middle">Q · Kᵀ</text>
      <text class="d-muted" x="415" y="82" text-anchor="middle">int32 dot</text>

      <rect class="d-box" x="512" y="34" width="146" height="70" rx="9" />
      <text class="d-text" x="585" y="62" text-anchor="middle">scale + mask</text>
      <text class="d-muted" x="585" y="82" text-anchor="middle">Q12 · j ≤ t</text>

      <rect class="d-box d-accent-box" x="682" y="34" width="100" height="70" rx="9" />
      <text class="d-text" x="732" y="62" text-anchor="middle">DI-Exp</text>
      <text class="d-muted" x="732" y="82" text-anchor="middle">Q15</text>

      <rect class="d-box" x="806" y="34" width="96" height="70" rx="9" />
      <text class="d-text" x="854" y="62" text-anchor="middle">P · V</text>
      <text class="d-muted" x="854" y="82" text-anchor="middle">Q16</text>

      <path class="d-line" d="M148 69 L168 69" />
      <path class="d-arrow" d="M162 63 L172 69 L162 75 Z" />
      <path class="d-line" d="M318 69 L338 69" />
      <path class="d-arrow" d="M332 63 L342 69 L332 75 Z" />
      <path class="d-line" d="M488 69 L508 69" />
      <path class="d-arrow" d="M502 63 L512 69 L502 75 Z" />
      <path class="d-line" d="M658 69 L678 69" />
      <path class="d-arrow" d="M672 63 L682 69 L672 75 Z" />
      <path class="d-line" d="M782 69 L802 69" />
      <path class="d-arrow" d="M796 63 L806 69 L796 75 Z" />

      <rect class="d-panel" x="18" y="136" width="884" height="216" rx="14" />
      <text class="d-title" x="42" y="166">causal softmax, one query at a time</text>

      <g transform="translate(46 188)">
        <rect class="d-box" x="0" y="0" width="330" height="24" rx="3" />
        <text class="d-muted" x="10" y="16">score[j] = dot(q8, k8[j]) · qscale · kscale · inv_sqrt &gt;&gt; 36</text>
      </g>
      <g transform="translate(46 226)">
        <rect class="d-box d-accent-box" x="0" y="0" width="330" height="24" rx="3" />
        <text class="d-muted" x="10" y="16">delta = max(score) - score[j], clamped to 20 nats</text>
      </g>
      <g transform="translate(46 264)">
        <rect class="d-box" x="0" y="0" width="330" height="24" rx="3" />
        <text class="d-muted" x="10" y="16">p[j] = exp_neg_q15_q12(delta) / Σ p</text>
      </g>
      <g transform="translate(46 302)">
        <rect class="d-box" x="0" y="0" width="330" height="24" rx="3" />
        <text class="d-muted" x="10" y="16">out[d] = Σ p15[j] · v8[j][d] · vscale[j] &gt;&gt; 15</text>
      </g>

      <text class="d-muted" x="430" y="204">only j ≤ t is computed; the rest is zero</text>
      <text class="d-muted" x="430" y="228">scores are Q12 nats, not Q16</text>
      <text class="d-muted" x="430" y="252">probabilities are Q15, sum ≈ 32768</text>
      <text class="d-muted" x="430" y="276">V is quantized per token, per head</text>
      <text class="d-muted" x="430" y="312">the output is Q16, ready for Wo</text>

      <text class="d-muted" x="460" y="372" text-anchor="middle">
        head_dim = D / H; inv_sqrt_head_dim is a Q16 integer
      </text>
    </svg>
  );
}

export function PackingDiagram() {
  return (
    <svg class="diagram" viewBox="0 0 920 330" role="img" aria-labelledby="dd6t dd6d">
      <title id="dd6t">Checkpoint packing order</title>
      <desc id="dd6d">
        PyTorch linear weights are output by input. The exporter transposes them
        so the C engine reads input by output. The packed buffer is token
        embeddings, position embeddings, the final gain, then per-layer gains
        and transposed matrices.
      </desc>

      <rect class="d-panel" x="18" y="24" width="410" height="282" rx="14" />
      <text class="d-title" x="42" y="54">PyTorch</text>
      <text class="d-muted" x="42" y="76">nn.Linear.weight · [out, in]</text>
      <g transform="translate(46 100)">
        <rect class="d-box" x="0" y="0" width="330" height="26" rx="3" />
        <text class="d-muted" x="10" y="18">row = output neuron</text>
        <text class="d-muted" x="190" y="18">columns = inputs</text>
      </g>
      <text class="d-muted" x="46" y="160">flat memory: out0 in0, out0 in1, …</text>
      <text class="d-muted" x="46" y="184">a C dot product wants contiguous inputs</text>

      <rect class="d-panel" x="492" y="24" width="410" height="282" rx="14" />
      <text class="d-title" x="516" y="54">C engine</text>
      <text class="d-muted" x="516" y="76">weights · [in, out]</text>
      <g transform="translate(520 100)">
        <rect class="d-box" x="0" y="0" width="330" height="26" rx="3" />
        <text class="d-muted" x="10" y="18">row = input</text>
        <text class="d-muted" x="190" y="18">columns = outputs</text>
      </g>
      <text class="d-muted" x="520" y="160">flat memory: in0 out0, in0 out1, …</text>
      <text class="d-muted" x="520" y="184">pack_weights uses .T.flatten()</text>

      <path class="d-line d-line-wide" d="M432 165 L484 165" />
      <path class="d-arrow" d="M478 159 L488 165 L478 171 Z" />
      <text class="d-muted" x="460" y="152" text-anchor="middle">transpose</text>

      <text class="d-title" x="42" y="236">packed order</text>
      <text class="d-muted" x="42" y="260">token_emb · pos_emb · g_final</text>
      <text class="d-muted" x="42" y="282">then per layer: g1 · wqkv · wo · g2 · w1 · w2</text>
    </svg>
  );
}

export function NextTokenDiagram() {
  return (
    <svg class="diagram" viewBox="0 0 920 300" role="img" aria-labelledby="dd7t dd7d">
      <title id="dd7t">The autoregressive generation loop</title>
      <desc id="dd7d">
        A prompt is tokenized into ids. The model reads a fixed window of ids and
        produces one probability per vocabulary token. The host samples one
        token, appends it, and calls the model again until a stop token appears.
      </desc>

      <rect class="d-box d-blue-box" x="16" y="28" width="150" height="84" rx="10" />
      <text class="d-text" x="91" y="58" text-anchor="middle">prompt</text>
      <text class="d-muted" x="91" y="80" text-anchor="middle">raw text</text>
      <text class="d-muted" x="91" y="98" text-anchor="middle">“the sky is”</text>

      <rect class="d-box" x="196" y="28" width="140" height="84" rx="10" />
      <text class="d-text" x="266" y="58" text-anchor="middle">tokenizer</text>
      <text class="d-muted" x="266" y="80" text-anchor="middle">text → ids</text>
      <text class="d-muted" x="266" y="98" text-anchor="middle">host side</text>

      <rect class="d-box" x="366" y="28" width="120" height="84" rx="10" />
      <text class="d-text" x="426" y="58" text-anchor="middle">ids</text>
      <text class="d-muted" x="426" y="80" text-anchor="middle">[17, 42, 91]</text>
      <text class="d-muted" x="426" y="98" text-anchor="middle">length T</text>

      <rect class="d-box d-accent-box" x="516" y="28" width="160" height="84" rx="10" />
      <text class="d-text" x="596" y="58" text-anchor="middle">the model</text>
      <text class="d-muted" x="596" y="80" text-anchor="middle">L transformer</text>
      <text class="d-muted" x="596" y="98" text-anchor="middle">blocks</text>

      <rect class="d-box" x="706" y="28" width="198" height="84" rx="10" />
      <text class="d-text" x="805" y="58" text-anchor="middle">next-token probabilities</text>
      <text class="d-muted" x="805" y="80" text-anchor="middle">one value per id</text>
      <text class="d-muted" x="805" y="98" text-anchor="middle">sums to 1.0</text>

      <path class="d-line" d="M166 70 L192 70" />
      <path class="d-arrow" d="M186 64 L196 70 L186 76 Z" />
      <path class="d-line" d="M336 70 L362 70" />
      <path class="d-arrow" d="M356 64 L366 70 L356 76 Z" />
      <path class="d-line" d="M486 70 L512 70" />
      <path class="d-arrow" d="M506 64 L516 70 L506 76 Z" />
      <path class="d-line" d="M676 70 L702 70" />
      <path class="d-arrow" d="M696 64 L706 70 L696 76 Z" />

      <rect class="d-panel" x="16" y="146" width="888" height="130" rx="14" />
      <text class="d-title" x="40" y="176">the host closes the loop</text>

      <rect class="d-box" x="44" y="196" width="210" height="56" rx="8" />
      <text class="d-text" x="149" y="220" text-anchor="middle">sample or argmax</text>
      <text class="d-muted" x="149" y="238" text-anchor="middle">temperature · top-k · seed</text>

      <rect class="d-box" x="318" y="196" width="210" height="56" rx="8" />
      <text class="d-text" x="423" y="220" text-anchor="middle">append the new id</text>
      <text class="d-muted" x="423" y="238" text-anchor="middle">keep the last T ids</text>

      <rect class="d-box d-blue-box" x="592" y="196" width="284" height="56" rx="8" />
      <text class="d-text" x="734" y="220" text-anchor="middle">call the model again</text>
      <text class="d-muted" x="734" y="238" text-anchor="middle">stop on &lt;end&gt; or &lt;pad&gt;</text>

      <path class="d-line" d="M254 224 L314 224" />
      <path class="d-arrow" d="M308 218 L318 224 L308 230 Z" />
      <path class="d-line" d="M528 224 L588 224" />
      <path class="d-arrow" d="M582 218 L592 224 L582 230 Z" />

      <path class="d-line" d="M734 196 C734 166 426 166 426 196" />
      <path class="d-arrow" d="M420 186 L426 196 L432 186 Z" />
      <text class="d-muted" x="580" y="160" text-anchor="middle">repeat</text>
    </svg>
  );
}

export function BlockPurposeDiagram() {
  return (
    <svg class="diagram" viewBox="0 0 920 400" role="img" aria-labelledby="dd8t dd8d">
      <title id="dd8t">The purpose of each part of a transformer block</title>
      <desc id="dd8d">
        A residual stream runs through the block. RMSNorm produces a stable copy,
        attention mixes information between positions, and an add writes the
        result back. A second RMSNorm feeds an MLP that expands, applies GELU,
        and contracts before another add.
      </desc>

      <path class="d-line d-line-wide" d="M108 34 L108 366" />
      <text class="d-title" x="24" y="30">residual stream</text>
      <text class="d-muted" x="24" y="52">x · [T, D]</text>
      <text class="d-muted" x="24" y="70">shared by every part</text>

      <path class="d-line" d="M108 92 L170 92" />
      <path class="d-arrow" d="M164 86 L174 92 L164 98 Z" />
      <rect class="d-box" x="180" y="66" width="150" height="52" rx="8" />
      <text class="d-text" x="255" y="88" text-anchor="middle">RMSNorm</text>
      <text class="d-muted" x="255" y="106" text-anchor="middle">stable scale</text>

      <path class="d-line" d="M330 92 L386 92" />
      <path class="d-arrow" d="M380 86 L390 92 L380 98 Z" />
      <rect class="d-box d-accent-box" x="396" y="66" width="210" height="52" rx="8" />
      <text class="d-text" x="501" y="88" text-anchor="middle">attention</text>
      <text class="d-muted" x="501" y="106" text-anchor="middle">move info between positions</text>

      <path class="d-line" d="M606 92 L686 92" />
      <path class="d-arrow" d="M680 86 L690 92 L680 98 Z" />
      <circle class="d-plus" cx="716" cy="92" r="20" />
      <text class="d-text" x="716" y="98" text-anchor="middle">+</text>
      <path class="d-line" d="M736 92 L776 92 L776 180 L108 180" />
      <path class="d-arrow" d="M118 174 L108 180 L118 186 Z" />
      <text class="d-muted" x="430" y="174" text-anchor="middle">write the update back</text>

      <path class="d-line" d="M108 244 L170 244" />
      <path class="d-arrow" d="M164 238 L174 244 L164 250 Z" />
      <rect class="d-box" x="180" y="218" width="150" height="52" rx="8" />
      <text class="d-text" x="255" y="240" text-anchor="middle">RMSNorm</text>
      <text class="d-muted" x="255" y="258" text-anchor="middle">stable scale</text>

      <path class="d-line" d="M330 244 L386 244" />
      <path class="d-arrow" d="M380 238 L390 244 L380 250 Z" />
      <rect class="d-box" x="396" y="218" width="210" height="52" rx="8" />
      <text class="d-text" x="501" y="240" text-anchor="middle">MLP · D → M → D</text>
      <text class="d-muted" x="501" y="258" text-anchor="middle">GELU computes within a position</text>

      <path class="d-line" d="M606 244 L686 244" />
      <path class="d-arrow" d="M680 238 L690 244 L680 250 Z" />
      <circle class="d-plus" cx="716" cy="244" r="20" />
      <text class="d-text" x="716" y="250" text-anchor="middle">+</text>
      <path class="d-line" d="M736 244 L776 244 L776 330 L108 330" />
      <path class="d-arrow" d="M118 324 L108 330 L118 336 Z" />
      <text class="d-muted" x="430" y="324" text-anchor="middle">next block, or the final norm</text>

      <text class="d-muted" x="108" y="382" text-anchor="middle">one block · same shapes in and out</text>
    </svg>
  );
}

export function AttentionPrimerDiagram() {
  return (
    <svg class="diagram" viewBox="0 0 920 390" role="img" aria-labelledby="dd9t dd9d">
      <title id="dd9t">Attention from queries, keys, and values</title>
      <desc id="dd9d">
        A normalized input row is projected into queries, keys, and values. Each
        query is compared with allowed keys, softmax turns the scores into
        weights, and the weights mix the values into one output per position.
      </desc>

      <rect class="d-box d-blue-box" x="16" y="30" width="130" height="74" rx="9" />
      <text class="d-text" x="81" y="58" text-anchor="middle">h</text>
      <text class="d-muted" x="81" y="78" text-anchor="middle">[T, D]</text>
      <text class="d-muted" x="81" y="96" text-anchor="middle">normalized</text>

      <rect class="d-box" x="178" y="30" width="150" height="74" rx="9" />
      <text class="d-text" x="253" y="58" text-anchor="middle">Wqkv</text>
      <text class="d-muted" x="253" y="78" text-anchor="middle">[D, 3D]</text>
      <text class="d-muted" x="253" y="96" text-anchor="middle">learned</text>

      <rect class="d-box" x="360" y="30" width="150" height="74" rx="9" />
      <text class="d-text" x="435" y="52" text-anchor="middle">Q · K · V</text>
      <text class="d-muted" x="435" y="72" text-anchor="middle">[T, D] each</text>
      <text class="d-muted" x="435" y="92" text-anchor="middle">query · key · value</text>

      <rect class="d-box" x="542" y="30" width="150" height="74" rx="9" />
      <text class="d-text" x="617" y="52" text-anchor="middle">scores</text>
      <text class="d-muted" x="617" y="72" text-anchor="middle">Q · Kᵀ / √d</text>
      <text class="d-muted" x="617" y="92" text-anchor="middle">[H, T, T]</text>

      <rect class="d-box d-accent-box" x="724" y="30" width="180" height="74" rx="9" />
      <text class="d-text" x="814" y="52" text-anchor="middle">softmax + P · V</text>
      <text class="d-muted" x="814" y="72" text-anchor="middle">weights · [H, T, T]</text>
      <text class="d-muted" x="814" y="92" text-anchor="middle">output · [T, D]</text>

      <path class="d-line" d="M146 67 L174 67" />
      <path class="d-arrow" d="M168 61 L178 67 L168 73 Z" />
      <path class="d-line" d="M328 67 L356 67" />
      <path class="d-arrow" d="M350 61 L360 67 L350 73 Z" />
      <path class="d-line" d="M510 67 L538 67" />
      <path class="d-arrow" d="M532 61 L542 67 L532 73 Z" />
      <path class="d-line" d="M692 67 L720 67" />
      <path class="d-arrow" d="M714 61 L724 67 L714 73 Z" />

      <rect class="d-panel" x="16" y="136" width="888" height="238" rx="14" />
      <text class="d-title" x="40" y="166">causal attention for three positions</text>

      <g transform="translate(44 190)">
        <rect class="d-box d-accent-box" x="0" y="0" width="30" height="30" rx="3" />
        <rect class="d-box d-panel" x="32" y="0" width="30" height="30" rx="3" />
        <rect class="d-box d-panel" x="64" y="0" width="30" height="30" rx="3" />
        <rect class="d-box d-accent-box" x="0" y="32" width="30" height="30" rx="3" />
        <rect class="d-box d-accent-box" x="32" y="32" width="30" height="30" rx="3" />
        <rect class="d-box d-panel" x="64" y="32" width="30" height="30" rx="3" />
        <rect class="d-box d-accent-box" x="0" y="64" width="30" height="30" rx="3" />
        <rect class="d-box d-accent-box" x="32" y="64" width="30" height="30" rx="3" />
        <rect class="d-box d-accent-box" x="64" y="64" width="30" height="30" rx="3" />
        <text class="d-muted" x="47" y="118" text-anchor="middle">allowed</text>
        <text class="d-muted" x="79" y="118" text-anchor="middle">future</text>
      </g>

      <text class="d-muted" x="230" y="206">row t is the query; columns are keys.</text>
      <text class="d-muted" x="230" y="228">only j ≤ t contributes; future weights are zero.</text>
      <text class="d-muted" x="230" y="250">softmax is applied per row, not over the whole matrix.</text>
      <text class="d-muted" x="230" y="272">each head has its own Wq, Wk, Wv slice.</text>
      <text class="d-muted" x="230" y="294">output rows are concatenated and passed through Wo.</text>

      <text class="d-muted" x="460" y="352" text-anchor="middle">
        attention moves information · the MLP transforms it
      </text>
    </svg>
  );
}

export function TrainingLoopDiagram() {
  return (
    <svg class="diagram" viewBox="0 0 920 320" role="img" aria-labelledby="dd10t dd10d">
      <title id="dd10t">The training loop</title>
      <desc id="dd10d">
        A batch of token windows goes through the float forward pass. The loss
        compares the predicted next-token probabilities with the true tokens.
        Backpropagation produces gradients, the optimizer updates the weights,
        and the loop repeats.
      </desc>

      <rect class="d-box d-blue-box" x="18" y="36" width="150" height="84" rx="10" />
      <text class="d-text" x="93" y="66" text-anchor="middle">corpus</text>
      <text class="d-muted" x="93" y="88" text-anchor="middle">token windows</text>
      <text class="d-muted" x="93" y="106" text-anchor="middle">input + target</text>

      <rect class="d-box" x="198" y="36" width="150" height="84" rx="10" />
      <text class="d-text" x="273" y="66" text-anchor="middle">forward</text>
      <text class="d-muted" x="273" y="88" text-anchor="middle">float32</text>
      <text class="d-muted" x="273" y="106" text-anchor="middle">T × V logits</text>

      <rect class="d-box" x="378" y="36" width="150" height="84" rx="10" />
      <text class="d-text" x="453" y="66" text-anchor="middle">loss</text>
      <text class="d-muted" x="453" y="88" text-anchor="middle">cross-entropy</text>
      <text class="d-muted" x="453" y="106" text-anchor="middle">−log p[target]</text>

      <rect class="d-box" x="558" y="36" width="150" height="84" rx="10" />
      <text class="d-text" x="633" y="66" text-anchor="middle">backward</text>
      <text class="d-muted" x="633" y="88" text-anchor="middle">chain rule</text>
      <text class="d-muted" x="633" y="106" text-anchor="middle">one gradient per weight</text>

      <rect class="d-box d-accent-box" x="738" y="36" width="164" height="84" rx="10" />
      <text class="d-text" x="820" y="66" text-anchor="middle">optimizer</text>
      <text class="d-muted" x="820" y="88" text-anchor="middle">AdamW · lr</text>
      <text class="d-muted" x="820" y="106" text-anchor="middle">update weights</text>

      <path class="d-line" d="M168 78 L194 78" />
      <path class="d-arrow" d="M188 72 L198 78 L188 84 Z" />
      <path class="d-line" d="M348 78 L374 78" />
      <path class="d-arrow" d="M368 72 L378 78 L368 84 Z" />
      <path class="d-line" d="M528 78 L554 78" />
      <path class="d-arrow" d="M548 72 L558 78 L548 84 Z" />
      <path class="d-line" d="M708 78 L734 78" />
      <path class="d-arrow" d="M728 72 L738 78 L728 84 Z" />

      <rect class="d-panel" x="18" y="164" width="884" height="128" rx="14" />
      <text class="d-title" x="42" y="194">the loop</text>
      <text class="d-muted" x="42" y="222">
        one step = one gradient update. one epoch = one pass over the training set.
      </text>
      <text class="d-muted" x="42" y="246">
        inference keeps the forward pass and the trained weights; it discards the loss, backward pass, and optimizer.
      </text>
      <text class="d-muted" x="42" y="270">
        the exported checkpoint is the only artifact the integer engine needs.
      </text>

      <path class="d-line" d="M820 120 L820 152 L273 152 L273 124" />
      <path class="d-arrow" d="M267 134 L273 124 L279 134 Z" />
      <text class="d-muted" x="546" y="146" text-anchor="middle">next batch</text>
    </svg>
  );
}
