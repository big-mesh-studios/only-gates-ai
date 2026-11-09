/*
 * The page's editable facts. Prose that changes often (missions, FAQ answers,
 * model configurations) lives here so the layout in App.tsx can stay stable.
 */

export interface NavItem {
  href: string;
  label: string;
}

export const NAV: NavItem[] = [
  { href: "#overview", label: "Overview" },
  { href: "#anatomy", label: "Anatomy" },
  { href: "#integers", label: "Integers" },
  { href: "#toolbox", label: "Toolbox" },
  { href: "#pipeline", label: "Pipeline" },
  { href: "#gates", label: "Gates" },
  { href: "#fpga", label: "FPGA" },
  { href: "#roadmap", label: "Roadmap" },
  { href: "#help", label: "Help train" },
];

export interface ModelConfig {
  name: string;
  role: string;
  vocab: string;
  context: string;
  dim: string;
  layers: string;
  heads: string;
  mlp: string;
  params: string;
}

export const CONFIGS: ModelConfig[] = [
  {
    name: "Alice",
    role: "Built into the app",
    vocab: "512",
    context: "16",
    dim: "48",
    layers: "2",
    heads: "2",
    mlp: "96",
    params: "62,448",
  },
  {
    name: "Solid 2",
    role: "Community training target",
    vocab: "4,096",
    context: "256",
    dim: "48",
    layers: "3",
    heads: "4",
    mlp: "96",
    params: "264,528",
  },
];

export interface PrecisionRow {
  format: string;
  where: string;
  why: string;
}

export const PRECISION: PrecisionRow[] = [
  {
    format: "int8",
    where: "Weights, activations",
    why: "One byte per value; dot products fit int32 accumulators.",
  },
  {
    format: "Q16",
    where: "Activations, norms, scales",
    why: "16 fractional bits keep small residuals useful after each layer.",
  },
  {
    format: "int32",
    where: "Matrix accumulation",
    why: "256 int8 × int8 products fit without overflow.",
  },
  {
    format: "int64",
    where: "Requantization, attention sums",
    why: "Intermediate products of scaled values need headroom.",
  },
  {
    format: "Q15",
    where: "Softmax probabilities",
    why: "32768 is one; attention and output probabilities stay positive.",
  },
  {
    format: "Q12",
    where: "Exponential lookup input",
    why: "A small clipped range is enough for exp(-x) in softmax.",
  },
];

export interface Mission {
  id: string;
  track: "Compute" | "Code" | "Hardware";
  title: string;
  compute: string;
  output: string;
  status: "open" | "in progress" | "done";
}

export const MISSIONS: Mission[] = [
  {
    id: "m1",
    track: "Compute",
    title: "Reproduce the Solid 2 checkpoint",
    compute: "1 free Colab T4 session, resumed as needed",
    output: "latest.pt, weights.bin, metrics.jsonl, checkpoint SHA-256",
    status: "open",
  },
  {
    id: "m2",
    track: "Compute",
    title: "Three-seed stability run",
    compute: "3 short sessions on the current config",
    output: "Three loss curves and final QA validation scores",
    status: "open",
  },
  {
    id: "m3",
    track: "Compute",
    title: "Docs-only vs docs + stories ablation",
    compute: "2 short runs with the same seed budget",
    output: "Held-out loss and sample transcripts for both arms",
    status: "open",
  },
  {
    id: "m4",
    track: "Code",
    title: "Float vs integer evaluation report",
    compute: "CPU is enough",
    output: "Top-1 match, top-5 overlap, mean probability delta",
    status: "open",
  },
  {
    id: "m5",
    track: "Code",
    title: "Gate-lower one transformer block",
    compute: "CPU, simulator",
    output: "Bit-level netlist or generated boolean module for one block",
    status: "in progress",
  },
  {
    id: "m6",
    track: "Hardware",
    title: "First FPGA bring-up of the MLP",
    compute: "A small FPGA board and synthesis time",
    output: "LUT/FF/BRAM/DSP counts, clock, tokens/s, power",
    status: "open",
  },
  {
    id: "m7",
    track: "Hardware",
    title: "Verify the RTL against the integer reference",
    compute: "CPU plus simulator",
    output: "Cycle-accurate traces and a mismatch report",
    status: "open",
  },
  {
    id: "m8",
    track: "Code",
    title: "Write a reproducible training note",
    compute: "No compute required",
    output: "Commands, seeds, hashes, hardware, and what changed",
    status: "open",
  },
];

export interface FaqItem {
  question: string;
  answer: string;
}

export const FAQ: FaqItem[] = [
  {
    question: "Do I need a GPU to help?",
    answer:
      "No. The smoke test and the float-vs-integer evaluation run on a CPU. Real training runs are much faster on a free Colab T4, but code, evaluation, documentation, and RTL review need no GPU.",
  },
  {
    question: "Is integer inference bit-identical to the float model?",
    answer:
      "No. Quantization changes the numbers. The goal is to stay close enough that the top prediction and the character of the output are preserved. The app ships a float-vs-int8 comparison so the difference is visible rather than hidden.",
  },
  {
    question: "Is the whole GPT running on an FPGA today?",
    answer:
      "Not yet. Integer-only inference in WebAssembly is working. The gate compiler is a prototype for small symbol-classifier circuits. Lowering a full transformer to gates and running it on an FPGA is the roadmap, not a finished feature.",
  },
  {
    question: "What does resume actually restore?",
    answer:
      "The model weights, AdamW optimizer moments, learning-rate scheduler position, Python/NumPy/PyTorch RNG state, the LM step, the QA epoch and batch cursor, and the exact QA example permutation. A resumed run continues from the same place instead of restarting.",
  },
  {
    question: "Why not just use an existing CPU inference engine?",
    answer:
      "Speed alone is not the point. The project wants a model whose inference is a fixed integer dataflow that can be lowered to logic gates and mapped to an FPGA. That constrains the operators, the precision, and the memory layout from the start.",
  },
  {
    question: "Can I use a free Colab account?",
    answer:
      "Yes, within Colab's terms. Free sessions are time-limited and can disconnect, which is why the trainer checkpoints to Google Drive and resumes. Start with the smoke test, keep one account, and do not try to work around usage limits.",
  },
  {
    question: "How is contributed work credited?",
    answer:
      "Compute runs report the checkpoint hash, configuration, seed, hardware, and wall time in a GitHub issue. Results used in a release are listed with the contributor and the run. The repository is MIT licensed; upstream corpora keep their own licenses.",
  },
];

export const COLAB_FIRST = `# First session: mount Drive, clone, smoke test, then start.
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/clinuxrulz/only-gates-ai.git
%cd only-gates-ai

!python scripts/train_tiny_gpt_solid2_colab.py \\
  --out /content/tiny-gpt-smoke --smoke --fresh

!python scripts/train_tiny_gpt_solid2_colab.py \\
  --out /content/drive/MyDrive/tiny-gpt-solid2 \\
  --lm-steps 5000 --qa-epochs 100 \\
  --save-every 100 --max-minutes 300 --fresh`;

export const COLAB_NEXT = `# Next session: the same checkpoint in Drive picks up where it stopped.
!git clone https://github.com/clinuxrulz/only-gates-ai.git
%cd only-gates-ai

!python scripts/train_tiny_gpt_solid2_colab.py \\
  --out /content/drive/MyDrive/tiny-gpt-solid2 \\
  --lm-steps 5000 --qa-epochs 100 \\
  --save-every 100 --max-minutes 300 --resume`;
