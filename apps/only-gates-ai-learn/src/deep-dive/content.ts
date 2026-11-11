export interface NavItem {
  href: string;
  label: string;
  number: string;
}

export const NAV: NavItem[] = [
  { href: "#contract", label: "Read this first", number: "00" },
  { href: "#math", label: "The math you need", number: "01" },
  { href: "#what", label: "What a GPT is", number: "02" },
  { href: "#tokens", label: "Tokens and embeddings", number: "03" },
  { href: "#block", label: "The transformer block", number: "04" },
  { href: "#attention-primer", label: "Attention from scratch", number: "05" },
  { href: "#mlp", label: "MLP and GELU", number: "06" },
  { href: "#output-primer", label: "Logits and sampling", number: "07" },
  { href: "#training-primer", label: "How training works", number: "08" },
  { href: "#reference", label: "Implement the forward pass", number: "09" },
  { href: "#model", label: "The model", number: "10" },
  { href: "#float", label: "Float reference", number: "11" },
  { href: "#fixed", label: "Fixed point", number: "12" },
  { href: "#quantize", label: "Quantize at load", number: "13" },
  { href: "#matmul", label: "Integer matmul", number: "14" },
  { href: "#norm", label: "RMSNorm", number: "15" },
  { href: "#gelu", label: "GELU table", number: "16" },
  { href: "#attention", label: "Attention", number: "17" },
  { href: "#forward", label: "Forward pass", number: "18" },
  { href: "#sampling", label: "Sampling", number: "19" },
  { href: "#generation", label: "Generation", number: "20" },
  { href: "#training", label: "Train and export", number: "21" },
  { href: "#verify", label: "Verify and debug", number: "22" },
  { href: "#beyond", label: "Performance and gates", number: "23" },
  { href: "#appendix", label: "Appendix", number: "24" },
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
    term: "Autoregressive",
    meaning:
      "Generating one token at a time and feeding each new token back into the model as part of the next input.",
  },
  {
    term: "Checkpoint",
    meaning:
      "A saved set of trained weights plus the configuration needed to load them. The integer engine reads a packed checkpoint.",
  },
  {
    term: "Bias",
    meaning:
      "A learned offset added after a matrix multiply. This model deliberately has no biases.",
  },
  {
    term: "Dropout",
    meaning:
      "Randomly zeroing activations during training to reduce overfitting. It is disabled at inference.",
  },
  {
    term: "AdamW",
    meaning:
      "The optimizer used by the trainer. It adapts the step size per parameter and applies weight decay.",
  },
  {
    term: "Validation set",
    meaning:
      "Held-out examples used to detect overfitting. Validation loss should fall alongside training loss, at least for a while.",
  },
  {
    term: "GPT",
    meaning:
      "Generative pre-trained transformer. A decoder-only model trained to predict the next token, then used to generate text one token at a time.",
  },
  {
    term: "Token",
    meaning:
      "A chunk of text (word, subword, or byte) represented by an integer id. The model never sees raw characters.",
  },
  {
    term: "Vocabulary",
    meaning:
      "The fixed set of tokens a model knows. Its size is V. A checkpoint and its tokenizer are a pair.",
  },
  {
    term: "Tokenizer",
    meaning:
      "The separate program that converts text to token ids and back. It is not part of the C inference engine.",
  },
  {
    term: "BPE",
    meaning:
      "Byte-pair encoding. A tokenizer that starts from bytes and repeatedly merges frequent adjacent pairs.",
  },
  {
    term: "Token id",
    meaning:
      "An index into the embedding table. Id 400 is not 'twice' id 200; the number has no arithmetic meaning.",
  },
  {
    term: "Embedding",
    meaning:
      "A learned vector that represents a token (or a position). The embedding table is a lookup table of shape [V, D].",
  },
  {
    term: "Position embedding",
    meaning:
      "A learned vector added to each token embedding so the model can tell positions apart.",
  },
  {
    term: "Model dimension (D)",
    meaning:
      "The number of values in each token's vector. Also called the model width or hidden size.",
  },
  {
    term: "Context window (T)",
    meaning:
      "The fixed number of token positions the model reads in one forward pass. Older tokens fall out of the window.",
  },
  {
    term: "Logits",
    meaning:
      "Unnormalized scores over the vocabulary. Softmax turns logits into probabilities.",
  },
  {
    term: "Softmax",
    meaning:
      "The function exp(z[i]) / sum(exp(z[j])). It turns arbitrary scores into positive values that sum to one.",
  },
  {
    term: "Temperature",
    meaning:
      "A divisor applied to logits before softmax. Lower values sharpen the distribution; higher values flatten it.",
  },
  {
    term: "Top-k",
    meaning:
      "Keep only the k highest-probability candidates before sampling, then renormalize.",
  },
  {
    term: "Greedy decoding",
    meaning:
      "Always choose the highest-probability token. Deterministic, but often repetitive.",
  },
  {
    term: "Attention",
    meaning:
      "A mechanism that lets each position read information from allowed earlier positions using query/key similarity and a weighted sum of values.",
  },
  {
    term: "Query, key, value (Q/K/V)",
    meaning:
      "Three learned projections of a token vector. The query asks what is needed, the key advertises what is available, and the value carries the content.",
  },
  {
    term: "Head",
    meaning:
      "One independent attention computation on a slice of the model dimension. Multi-head attention runs several in parallel.",
  },
  {
    term: "Causal mask",
    meaning:
      "The rule that position t may only attend to positions j <= t. It prevents the model from seeing the future during training.",
  },
  {
    term: "Residual stream",
    meaning:
      "The [T, D] matrix that runs through the whole model. Each sublayer adds its output to it instead of replacing it.",
  },
  {
    term: "Residual connection",
    meaning:
      "The x = x + sublayer(x) pattern. It preserves information and gives gradients a short path during training.",
  },
  {
    term: "RMSNorm",
    meaning:
      "Root-mean-square normalization. Divides a row by the RMS of its values and multiplies by a learned gain.",
  },
  {
    term: "MLP",
    meaning:
      "Multi-layer perceptron, also called the feed-forward network. Expands each position to width M, applies GELU, and projects back to D.",
  },
  {
    term: "GELU",
    meaning:
      "Gaussian error linear unit. A smooth nonlinearity; here the tanh approximation, implemented as a lookup table.",
  },
  {
    term: "Parameter",
    meaning:
      "A learned number in a weight matrix, gain, or embedding table. Training adjusts parameters; inference only reads them.",
  },
  {
    term: "Tied embeddings",
    meaning:
      "Using the token embedding matrix, transposed, as the output head. It saves parameters and aligns input and output token spaces.",
  },
  {
    term: "Training",
    meaning:
      "The phase that measures prediction error and adjusts every parameter with gradients. It needs floating point.",
  },
  {
    term: "Inference",
    meaning:
      "Running a fixed trained model on new input. It is deterministic and can be integer-only.",
  },
  {
    term: "Loss",
    meaning:
      "A single number measuring how wrong the model is. Language models use cross-entropy on the correct next token.",
  },
  {
    term: "Cross-entropy",
    meaning:
      "The loss -log(p[target]). Confident correct predictions have low loss; unlikely correct predictions have high loss.",
  },
  {
    term: "Perplexity",
    meaning:
      "exp(loss). Roughly the number of equally likely tokens the model is choosing among.",
  },
  {
    term: "Gradient",
    meaning:
      "The derivative of the loss with respect to a parameter: which way, and how strongly, to change it.",
  },
  {
    term: "Backpropagation",
    meaning:
      "Applying the chain rule backwards through the network to compute every gradient efficiently.",
  },
  {
    term: "Optimizer",
    meaning:
      "The rule that turns gradients into weight updates. This project uses AdamW.",
  },
  {
    term: "Learning rate",
    meaning:
      "The size of an optimizer step. Too large diverges; too small learns slowly.",
  },
  {
    term: "Step / batch / epoch",
    meaning:
      "One update / the examples behind one update / one pass over the training set.",
  },
  {
    term: "Overfitting",
    meaning:
      "Memorizing training data so training loss falls while validation loss rises.",
  },
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
    term: "Quantization",
    meaning:
      "Storing a real value as a small integer code plus a scale. Real ≈ code × scale / 65536 for Q16 scales.",
  },
  {
    term: "Scale",
    meaning:
      "The multiplier that maps an int8 code back to a real value. Weights use per-column scales; embeddings use per-row scales.",
  },
  {
    term: "Requantize",
    meaning:
      "Convert an accumulator back to a fixed-point scale with a rounded shift.",
  },
  {
    term: "Accumulator",
    meaning:
      "The integer sum of many products before scaling. Here it is int32 for int8 dot products.",
  },
  {
    term: "DI-Exp",
    meaning:
      "A table-driven integer approximation of exp(-x), used for attention and softmax.",
  },
  {
    term: "KV cache",
    meaning:
      "Reusing past keys and values instead of recomputing the whole context for every generated token.",
  },
  {
    term: "Deterministic",
    meaning:
      "Given the same input, weights, and seed, the engine returns exactly the same output. The sampler is deterministic in this sense.",
  },
];
