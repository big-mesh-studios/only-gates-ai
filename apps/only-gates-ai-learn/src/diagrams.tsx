/*
 * Inline SVG diagrams. They use the page's CSS tokens, so a theme change reaches
 * them without duplicating colours in markup. Every informative diagram carries
 * a title and a description for screen readers.
 */

export function TrainingInferenceDiagram() {
  return (
    <svg class="diagram" viewBox="0 0 820 340" role="img" aria-labelledby="d1t d1d">
      <title id="d1t">Training uses floating point; inference uses integers</title>
      <desc id="d1d">
        Training runs a float forward pass, backpropagation, and an AdamW update
        on a GPU. Inference runs a fixed integer dataflow and never needs
        floating point after the checkpoint is quantized.
      </desc>

      <rect class="d-panel" x="16" y="18" width="382" height="304" rx="16" />
      <text class="d-title" x="40" y="50">Training · float32</text>
      <text class="d-muted" x="40" y="72">gradients and optimizer state</text>

      <rect class="d-box" x="40" y="92" width="334" height="42" rx="8" />
      <text class="d-text" x="56" y="118">corpus → tokens → batches</text>
      <rect class="d-box" x="40" y="148" width="334" height="42" rx="8" />
      <text class="d-text" x="56" y="174">float forward pass + loss</text>
      <rect class="d-box" x="40" y="204" width="334" height="42" rx="8" />
      <text class="d-text" x="56" y="230">backprop → gradients</text>
      <rect class="d-box d-accent-box" x="40" y="260" width="334" height="42" rx="8" />
      <text class="d-text" x="56" y="286">AdamW update · GPU</text>

      <path class="d-line" d="M390 170 L430 170" />
      <path class="d-arrow" d="M424 164 L434 170 L424 176 Z" />
      <text class="d-muted" x="410" y="156" text-anchor="middle">export</text>

      <rect class="d-panel" x="438" y="18" width="366" height="304" rx="16" />
      <text class="d-title" x="462" y="50">Inference · integers only</text>
      <text class="d-muted" x="462" y="72">fixed dataflow, no FPU</text>

      <rect class="d-box d-blue-box" x="462" y="92" width="318" height="42" rx="8" />
      <text class="d-text" x="478" y="118">int8 weights + Q16 scales</text>
      <rect class="d-box d-blue-box" x="462" y="148" width="318" height="42" rx="8" />
      <text class="d-text" x="478" y="174">integer matmul · norm · exp LUT</text>
      <rect class="d-box d-blue-box" x="462" y="204" width="318" height="42" rx="8" />
      <text class="d-text" x="478" y="230">Q15 probabilities → token</text>
      <rect class="d-box d-accent-box" x="462" y="260" width="318" height="42" rx="8" />
      <text class="d-text" x="478" y="286">same answer, bit for bit, every run</text>

      <text class="d-muted" x="410" y="334" text-anchor="middle">
        float is a training detail, not an inference requirement
      </text>
    </svg>
  );
}

export function BlockDiagram() {
  return (
    <svg class="diagram" viewBox="0 0 900 380" role="img" aria-labelledby="d2t d2d">
      <title id="d2t">A decoder-only transformer block</title>
      <desc id="d2d">
        Token ids become embeddings, pass through repeated pre-norm transformer
        blocks with attention and an MLP, then a final norm and a tied
        embedding matrix produce logits for sampling.
      </desc>

      <rect class="d-box d-blue-box" x="18" y="142" width="112" height="72" rx="10" />
      <text class="d-text" x="74" y="172" text-anchor="middle">token ids</text>
      <text class="d-muted" x="74" y="194" text-anchor="middle">B × T</text>

      <rect class="d-box" x="158" y="142" width="126" height="72" rx="10" />
      <text class="d-text" x="221" y="172" text-anchor="middle">embedding</text>
      <text class="d-muted" x="221" y="194" text-anchor="middle">+ positions</text>

      <rect class="d-panel" x="312" y="54" width="356" height="248" rx="14" />
      <text class="d-title" x="490" y="82" text-anchor="middle">transformer block × L</text>

      <rect class="d-box" x="334" y="98" width="312" height="38" rx="7" />
      <text class="d-text" x="490" y="122" text-anchor="middle">RMSNorm → attention → + residual</text>
      <rect class="d-box" x="334" y="148" width="312" height="38" rx="7" />
      <text class="d-text" x="490" y="172" text-anchor="middle">RMSNorm → MLP (4×) → + residual</text>
      <text class="d-muted" x="490" y="216" text-anchor="middle">
        pre-norm, no biases, tied embeddings
      </text>
      <text class="d-muted" x="490" y="238" text-anchor="middle">
        attention: B × H × T × T
      </text>
      <text class="d-muted" x="490" y="260" text-anchor="middle">
        hidden: B × T × dim
      </text>

      <rect class="d-box" x="696" y="142" width="90" height="72" rx="10" />
      <text class="d-text" x="741" y="172" text-anchor="middle">final</text>
      <text class="d-text" x="741" y="192" text-anchor="middle">RMSNorm</text>

      <rect class="d-box d-accent-box" x="810" y="142" width="74" height="72" rx="10" />
      <text class="d-text" x="847" y="172" text-anchor="middle">logits</text>
      <text class="d-muted" x="847" y="194" text-anchor="middle">B×T×V</text>

      <path class="d-line" d="M130 178 L156 178" />
      <path class="d-line" d="M284 178 L310 178" />
      <path class="d-line" d="M668 178 L694 178" />
      <path class="d-line" d="M786 178 L808 178" />
      <path class="d-arrow" d="M800 172 L810 178 L800 184 Z" />

      <rect class="d-box" x="696" y="272" width="188" height="52" rx="10" />
      <text class="d-text" x="790" y="294" text-anchor="middle">sample next token</text>
      <text class="d-muted" x="790" y="314" text-anchor="middle">temperature · top-k</text>
      <path class="d-line" d="M847 216 L847 270" />
      <path class="d-arrow" d="M841 262 L847 272 L853 262 Z" />

      <text class="d-muted" x="450" y="356" text-anchor="middle">
        Alice: V=512, T=16, dim=48, L=2. Solid 2 target: V=4096, T=256, dim=48, L=3.
      </text>
    </svg>
  );
}

export function PrecisionDiagram() {
  return (
    <svg class="diagram" viewBox="0 0 900 250" role="img" aria-labelledby="d3t d3d">
      <title id="d3t">Where each integer format is used</title>
      <desc id="d3d">
        Weights and activations are int8. Activations and norms also use Q16.
        Matrix products accumulate in int32, requantization and attention sums
        use int64, probabilities use Q15, and the exponential lookup uses Q12.
      </desc>

      <path class="d-line d-line-wide" d="M40 120 L860 120" />
      <path class="d-arrow" d="M852 114 L862 120 L852 126 Z" />

      {[
        ["int8", "weights · activations", 84],
        ["Q16", "norms · scales", 230],
        ["int32", "dot products", 376],
        ["int64", "requantize · sums", 522],
        ["Q15", "softmax probs", 668],
        ["Q12", "exp(-x) LUT", 808],
      ].map(([tag, label, x]) => (
        <g>
          <circle class="d-dot" cx={x} cy="120" r="11" />
          <rect class="d-box" x={Number(x) - 66} y="44" width="132" height="42" rx="8" />
          <text class="d-text" x={x} y="70" text-anchor="middle">{tag}</text>
          <text class="d-muted" x={x} y="150" text-anchor="middle">{label}</text>
        </g>
      ))}

      <text class="d-muted" x="450" y="216" text-anchor="middle">
        wider accumulators protect the value while the storage stays small
      </text>
    </svg>
  );
}

export function QuantizationDiagram() {
  return (
    <svg class="diagram" viewBox="0 0 900 300" role="img" aria-labelledby="d4t d4d">
      <title id="d4t">From a float checkpoint to integer inference</title>
      <desc id="d4d">
        A float32 checkpoint is read once. Per-channel scales turn weights into
        int8. Activations are quantized to int8 rows with Q16 scales, products
        accumulate in int32, and results are requantized with int64 before a Q15
        probability is sampled.
      </desc>

      <rect class="d-box" x="18" y="96" width="132" height="88" rx="10" />
      <text class="d-text" x="84" y="126" text-anchor="middle">float32</text>
      <text class="d-text" x="84" y="146" text-anchor="middle">checkpoint</text>
      <text class="d-muted" x="84" y="168" text-anchor="middle">weights.bin</text>

      <rect class="d-box" x="178" y="96" width="132" height="88" rx="10" />
      <text class="d-text" x="244" y="126" text-anchor="middle">per-channel</text>
      <text class="d-text" x="244" y="146" text-anchor="middle">scales</text>
      <text class="d-muted" x="244" y="168" text-anchor="middle">load time only</text>

      <rect class="d-box d-blue-box" x="338" y="96" width="132" height="88" rx="10" />
      <text class="d-text" x="404" y="126" text-anchor="middle">int8 weights</text>
      <text class="d-text" x="404" y="146" text-anchor="middle">Q16 act.</text>
      <text class="d-muted" x="404" y="168" text-anchor="middle">int32 accum.</text>

      <rect class="d-box d-blue-box" x="498" y="96" width="132" height="88" rx="10" />
      <text class="d-text" x="564" y="126" text-anchor="middle">int64</text>
      <text class="d-text" x="564" y="146" text-anchor="middle">requantize</text>
      <text class="d-muted" x="564" y="168" text-anchor="middle">round to nearest</text>

      <rect class="d-box" x="658" y="96" width="102" height="88" rx="10" />
      <text class="d-text" x="709" y="126" text-anchor="middle">Q15</text>
      <text class="d-text" x="709" y="146" text-anchor="middle">softmax</text>
      <text class="d-muted" x="709" y="168" text-anchor="middle">probabilities</text>

      <rect class="d-box d-accent-box" x="788" y="96" width="94" height="88" rx="10" />
      <text class="d-text" x="835" y="126" text-anchor="middle">sample</text>
      <text class="d-text" x="835" y="146" text-anchor="middle">token</text>
      <text class="d-muted" x="835" y="168" text-anchor="middle">xorshift</text>

      {[150, 310, 470, 630, 760].map((x) => (
        <g>
          <path class="d-line" d={`M${x} 140 L${x + 26} 140`} />
          <path class="d-arrow" d={`M${x + 18} 134 L${x + 28} 140 L${x + 18} 146 Z`} />
        </g>
      ))}

      <text class="d-muted" x="450" y="232" text-anchor="middle">
        after load, the integer path contains no float type, float literal, or math.h call
      </text>
      <text class="d-muted" x="450" y="256" text-anchor="middle">
        this is the boundary the gate compiler will eventually consume
      </text>
    </svg>
  );
}

export function LoweringDiagram() {
  return (
    <svg class="diagram" viewBox="0 0 880 340" role="img" aria-labelledby="d5t d5d">
      <title id="d5t">The lowering stack from a float model to an FPGA bitstream</title>
      <desc id="d5d">
        A trained float model is exported, quantized into a fixed integer graph,
        lowered into an RTL or gate netlist, and mapped to an FPGA bitstream.
        Integer WASM inference exists today, the gate compiler is a prototype,
        and full GPT lowering is still ahead.
      </desc>

      <rect class="d-box d-blue-box" x="40" y="26" width="800" height="54" rx="10" />
      <text class="d-text" x="64" y="59">1 · trained float model</text>
      <text class="d-muted" x="810" y="59" text-anchor="end">GPU training · changes often</text>

      <rect class="d-box d-accent-box" x="40" y="96" width="800" height="54" rx="10" />
      <text class="d-text" x="64" y="129">2 · fixed integer graph</text>
      <text class="d-muted" x="810" y="129" text-anchor="end">today · WASM integer inference</text>

      <rect class="d-box" x="40" y="166" width="800" height="54" rx="10" />
      <text class="d-text" x="64" y="199">3 · RTL / gate netlist</text>
      <text class="d-muted" x="810" y="199" text-anchor="end">prototype · symbol classifier</text>

      <rect class="d-box" x="40" y="236" width="800" height="54" rx="10" />
      <text class="d-text" x="64" y="269">4 · FPGA bitstream</text>
      <text class="d-muted" x="810" y="269" text-anchor="end">roadmap · no working GPT yet</text>

      {[80, 150, 220].map((y) => (
        <g>
          <path class="d-line" d={`M440 ${y} L440 ${y + 14}`} />
          <path class="d-arrow" d={`M434 ${y + 8} L440 ${y + 18} L446 ${y + 8} Z`} />
        </g>
      ))}
    </svg>
  );
}

export function FpgaDiagram() {
  return (
    <svg class="diagram" viewBox="0 0 880 400" role="img" aria-labelledby="d6t d6d">
      <title id="d6t">A possible FPGA floorplan for a tiny GPT</title>
      <desc id="d6d">
        A host interface feeds a tokenizer and controller. Embedding tables live
        in block RAM. Attention and MLP engines use LUTs and DSP slices, with
        scratch block RAM between them. A sampler returns tokens to the host.
      </desc>

      <rect class="d-panel" x="20" y="20" width="840" height="356" rx="18" />
      <text class="d-title" x="44" y="52">FPGA</text>
      <text class="d-muted" x="836" y="52" text-anchor="end">fixed function · no OS · no CUDA</text>

      <rect class="d-box" x="44" y="78" width="150" height="70" rx="10" />
      <text class="d-text" x="119" y="106" text-anchor="middle">host I/O</text>
      <text class="d-muted" x="119" y="128" text-anchor="middle">UART · USB · SPI</text>

      <rect class="d-box d-blue-box" x="218" y="78" width="190" height="70" rx="10" />
      <text class="d-text" x="313" y="106" text-anchor="middle">tokenizer + control</text>
      <text class="d-muted" x="313" y="128" text-anchor="middle">tokens · position · schedule</text>

      <rect class="d-box" x="432" y="78" width="190" height="70" rx="10" />
      <text class="d-text" x="527" y="106" text-anchor="middle">embedding ROM</text>
      <text class="d-muted" x="527" y="128" text-anchor="middle">BRAM · int8 + scales</text>

      <rect class="d-box d-accent-box" x="646" y="78" width="190" height="70" rx="10" />
      <text class="d-text" x="741" y="106" text-anchor="middle">sampler</text>
      <text class="d-muted" x="741" y="128" text-anchor="middle">Q15 · top-k · xorshift</text>

      <rect class="d-box d-blue-box" x="218" y="174" width="300" height="92" rx="10" />
      <text class="d-text" x="368" y="204" text-anchor="middle">attention engine</text>
      <text class="d-muted" x="368" y="226" text-anchor="middle">QKV matmuls · scores · softmax</text>
      <text class="d-muted" x="368" y="248" text-anchor="middle">LUTs · DSP slices · accumulators</text>

      <rect class="d-box d-blue-box" x="542" y="174" width="294" height="92" rx="10" />
      <text class="d-text" x="689" y="204" text-anchor="middle">MLP engine</text>
      <text class="d-muted" x="689" y="226" text-anchor="middle">expand 4× · GELU LUT · project</text>
      <text class="d-muted" x="689" y="248" text-anchor="middle">pipelined int8 × int8 → int32</text>

      <rect class="d-box" x="218" y="292" width="220" height="62" rx="10" />
      <text class="d-text" x="328" y="318" text-anchor="middle">scratch BRAM</text>
      <text class="d-muted" x="328" y="338" text-anchor="middle">activations · KV cache</text>

      <rect class="d-box" x="462" y="292" width="374" height="62" rx="10" />
      <text class="d-text" x="649" y="318" text-anchor="middle">weights in BRAM or external memory</text>
      <text class="d-muted" x="649" y="338" text-anchor="middle">on-chip budget decides the model size</text>

      <path class="d-line" d="M194 113 L216 113" />
      <path class="d-line" d="M408 113 L430 113" />
      <path class="d-line" d="M622 113 L644 113" />
      <path class="d-line" d="M741 148 L741 172" />
      <path class="d-line" d="M368 148 L368 172" />
      <path class="d-line" d="M368 266 L368 290" />
      <path class="d-line" d="M689 266 L689 290" />
    </svg>
  );
}
