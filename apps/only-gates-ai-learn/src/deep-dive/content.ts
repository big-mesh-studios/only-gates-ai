export interface NavItem {
  href: string;
  label: string;
  number: string;
}

export const NAV: NavItem[] = [
  { href: "#contract", label: "Read this first", number: "00" },
  { href: "#model", label: "The model", number: "01" },
  { href: "#float", label: "Float reference", number: "02" },
  { href: "#fixed", label: "Fixed point", number: "03" },
  { href: "#quantize", label: "Quantize at load", number: "04" },
  { href: "#matmul", label: "Integer matmul", number: "05" },
  { href: "#norm", label: "RMSNorm", number: "06" },
  { href: "#gelu", label: "GELU table", number: "07" },
  { href: "#attention", label: "Attention", number: "08" },
  { href: "#forward", label: "Forward pass", number: "09" },
  { href: "#sampling", label: "Sampling", number: "10" },
  { href: "#generation", label: "Generation", number: "11" },
  { href: "#training", label: "Train and export", number: "12" },
  { href: "#verify", label: "Verify and debug", number: "13" },
  { href: "#beyond", label: "Performance and gates", number: "14" },
  { href: "#appendix", label: "Appendix", number: "15" },
];

export interface ModelConfig {
  name: string;
  role: string;
  vocab: number;
  context: number;
  dim: number;
  layers: number;
  heads: number;
  mlp: number;
  params: number;
}

export const CONFIGS: ModelConfig[] = [
  {
    name: "Alice",
    role: "Built into the app, word-level",
    vocab: 512,
    context: 16,
    dim: 48,
    layers: 2,
    heads: 2,
    mlp: 96,
    params: 62448,
  },
  {
    name: "Solid 2",
    role: "Community model, byte-level BPE",
    vocab: 4096,
    context: 256,
    dim: 48,
    layers: 3,
    heads: 4,
    mlp: 96,
    params: 264528,
  },
];

export interface TensorRow {
  name: string;
  shape: string;
  storage: string;
  scale: string;
  note: string;
}

export const TENSORS: TensorRow[] = [
  {
    name: "token_emb",
    shape: "V × D",
    storage: "int8",
    scale: "one Q16 per row (token)",
    note: "Embedding lookup table; also the tied output head.",
  },
  {
    name: "pos_emb",
    shape: "T × D",
    storage: "int8",
    scale: "one Q16 per row (position)",
    note: "Learned absolute position table.",
  },
  {
    name: "g1, g2, g_final",
    shape: "D",
    storage: "int32 Q16",
    scale: "none",
    note: "RMSNorm gains.",
  },
  {
    name: "wqkv",
    shape: "D × 3D",
    storage: "int8",
    scale: "one Q16 per output column",
    note: "Q, K, V projections fused; [in, out] layout.",
  },
  {
    name: "wo",
    shape: "D × D",
    storage: "int8",
    scale: "one Q16 per output column",
    note: "Attention output projection.",
  },
  {
    name: "w1",
    shape: "D × M",
    storage: "int8",
    scale: "one Q16 per output column",
    note: "MLP up-projection.",
  },
  {
    name: "w2",
    shape: "M × D",
    storage: "int8",
    scale: "one Q16 per output column",
    note: "MLP down-projection.",
  },
  {
    name: "head",
    shape: "D × V",
    storage: "int8",
    scale: "one Q16 per token",
    note: "Transposed token_emb, reusing its row scales.",
  },
];

export interface PrecisionRow {
  format: string;
  range: string;
  where: string;
  why: string;
}

export const PRECISION: PrecisionRow[] = [
  {
    format: "int8",
    range: "-127 … 127",
    where: "Weights, activations, embeddings",
    why: "One byte per value. int8 × int8 fits easily in int32.",
  },
  {
    format: "Q16",
    range: "-32768 … 32767.999…",
    where: "Activation stream, norm gains, scales, logits",
    why: "16 fractional bits preserve small residuals.",
  },
  {
    format: "int32",
    range: "-2^31 … 2^31-1",
    where: "Dot-product accumulators, buffers",
    why: "A 96-term int8 dot product cannot overflow it.",
  },
  {
    format: "int64",
    range: "-2^63 … 2^63-1",
    where: "Requantization, attention products",
    why: "Products of scaled values need more headroom.",
  },
  {
    format: "Q15",
    range: "-1 … 0.99997",
    where: "Softmax and attention probabilities",
    why: "32768 = 1.0; probabilities stay non-negative.",
  },
  {
    format: "Q12",
    range: "-2048 … 2047.999…",
    where: "DI-Exp input",
    why: "A 20-nat clipped range with 1/4096 resolution.",
  },
];

export interface SymptomRow {
  symptom: string;
  likely: string;
  check: string;
}

export const SYMPTOMS: SymptomRow[] = [
  {
    symptom: "Logits are all identical or zero",
    likely: "Weights not loaded, wrong packing order, or zero scales",
    check: "Dump the first embedding row and the first matmul output.",
  },
  {
    symptom: "Output is garbage but shapes are right",
    likely: "PyTorch [out, in] weights were not transposed",
    check: "Compare packed index [k*out_cols + j] with W[out=j, in=k].",
  },
  {
    symptom: "Model outputs are 65536× too large",
    likely: "One requantization shift is missing",
    check: "Matmul should shift 16; attention scores shift 36; V sum shift 15.",
  },
  {
    symptom: "Attention attends to the future",
    likely: "Causal mask or loop bound is wrong",
    check: "For row t, only j <= t may contribute.",
  },
  {
    symptom: "Probabilities sum to 32768, not 65536",
    likely: "Q15 output was not converted to Q16",
    check: "tiny_gpt_probabilities doubles the Q15 result.",
  },
  {
    symptom: "Float and integer top-1 differ",
    likely: "Expected: quantization changes the distribution",
    check: "Measure top-1 agreement, top-5 overlap, and mean delta.",
  },
  {
    symptom: "Generation repeats or stops immediately",
    likely: "Tokenizer, BOS/pad, or stop-token handling",
    check: "Print the token ids, not just the decoded text.",
  },
];

export interface GlossaryRow {
  term: string;
  meaning: string;
}

export const GLOSSARY: GlossaryRow[] = [
  {
    term: "Q16",
    meaning: "Fixed-point value with 16 fractional bits; 65536 means 1.0.",
  },
  {
    term: "Q15",
    meaning: "Fixed-point value with 15 fractional bits; 32768 means 1.0.",
  },
  {
    term: "Q12",
    meaning: "Fixed-point value with 12 fractional bits; 4096 means 1.0.",
  },
  {
    term: "Requantize",
    meaning: "Convert an accumulator back to a fixed-point scale with a rounded shift.",
  },
  {
    term: "Accumulator",
    meaning: "The integer sum of many products before scaling.",
  },
  {
    term: "Tied embeddings",
    meaning: "The output head is the token embedding matrix, transposed.",
  },
  {
    term: "Causal mask",
    meaning: "Position t may only attend to positions j <= t.",
  },
  {
    term: "Logits",
    meaning: "Unnormalized scores over the vocabulary.",
  },
  {
    term: "Top-k",
    meaning: "Keep only the k highest-probability candidates before sampling.",
  },
  {
    term: "DI-Exp",
    meaning: "A table-driven integer approximation of exp(-x).",
  },
  {
    term: "KV cache",
    meaning: "Reusing past keys and values instead of recomputing the whole context.",
  },
  {
    term: "GELU",
    meaning: "The Gaussian error linear unit; here the tanh approximation.",
  },
];
