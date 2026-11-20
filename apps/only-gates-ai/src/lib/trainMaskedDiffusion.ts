import type { BpeModel } from "@only-gates-ai/core";
import type { Scalar, Tensor } from "@tensorflow/tfjs";

import luauFaq from "../assets/luau-faq.md?raw";
import { LUAU_PRESETS } from "./luauConfig";
import {
  createMaskedDiffusionWasmFromData,
  maskedDiffusionParameterCount,
  type MaskedDiffusionCheckpoint,
  type MaskedDiffusionModel,
} from "./maskedDiffusion";
import {
  type MaskedDiffusionTrainingState,
} from "./maskedDiffusionStorage";
import type { TinyGptConfig } from "./tinyGpt";
import {
  BPE_SPECIAL,
  loadBpeTokenizer,
  parseBpeTokenizer,
  serializeBpeTokenizer,
  type BpeTokenizer,
} from "./tinyGptBpe";
import {
  applyAdam,
  clipGradients,
  createModel,
  createOptimizer,
  disposeModel,
  disposeOptimizer,
  forward,
  hashString,
  packModelWeights,
  packOptimizerWeights,
  restoreModelFromPacked,
  restoreOptimizerFromPacked,
  selectTensorFlowBackend,
  yieldToBrowser,
  type BrowserTrainingBackend,
  type BrowserTrainingProgress,
  type ModelVars,
  type OptimizerVars,
  type TfModule,
  type TrainingPoint,
} from "./trainTinyGpt";
import { createZip, readZip, type ZipFileInput } from "./zipArchive";

/*
 * Browser-side masked-diffusion training.
 *
 * The deployed Luau denoiser is a bidirectional Tiny GPT trained with an
 * absorbing <|mask|> token. This module mirrors
 * `scripts/train_masked_diffusion_luau_colab.py` closely enough that a browser
 * run is a real fine-tune, not a synthetic loss curve:
 *
 *   - data: generated Luau tasks + the repository FAQ, packed as
 *     (prompt ids, answer region) pairs;
 *   - masking: t ~ U(0, 1), with a small full-mask probability; only masked,
 *     non-pad answer positions contribute to the loss;
 *   - model: the same packed weight layout as the C `tiny_gpt_create`, with
 *     bidirectional attention (no causal mask);
 *   - export: `checkpoint.json`, `tokenizer.json`, and `weights.bin`, exactly
 *     the files the Masked diffusion tab loads.
 */

export const MASKED_DIFFUSION_MODEL_BASE = `${import.meta.env.BASE_URL}models/masked-diffusion-luau/`;

/** Full task dataset, served from the repository (about 14 MB). */
export const MASKED_DIFFUSION_DATA_BASE =
  "https://raw.githubusercontent.com/big-mesh-studios/only-gates-ai/main/data/luau-tasks/";

/** Small bundled slice, copied from the same dataset at build time. */
export const MASKED_DIFFUSION_SMALL_DATA_BASE = `${import.meta.env.BASE_URL}data/luau-tasks/`;

/**
 * Fallback architecture when the user starts from random weights. The deployed
 * checkpoint is 9.56M parameters; this smaller config is friendlier to a first
 * in-browser run while keeping the same tokenizer and context.
 */
export const MASKED_BROWSER_TRAIN_CONFIG: TinyGptConfig = {
  vocabSize: 1722,
  context: 128,
  dim: 192,
  layers: 4,
  heads: 4,
  mlp: 512,
};

const MIN_ANSWER_TOKENS = 16;
const DEFAULT_FULL_MASK_PROBABILITY = 0.15;
const DEFAULT_VALIDATION_EXAMPLES = 64;
const DEFAULT_DENOISING_STEPS = 16;

export interface MaskedDiffusionDataInfo {
  readonly fullData: boolean;
  readonly taskRows: number;
  readonly valRows: number;
  readonly faqPairs: number;
  readonly trainExamples: number;
  readonly valExamples: number;
  readonly skippedTrain: number;
  readonly skippedVal: number;
}

export interface MaskedDiffusionDataOptions {
  readonly fullData?: boolean;
  readonly modelBaseUrl?: string;
  readonly context: number;
  readonly tokenizer?: BpeTokenizer;
  readonly onStatus?: (status: string) => void;
  readonly shouldCancel?: () => boolean;
}

export interface MaskedDiffusionTrainingData {
  readonly tokenizer: BpeTokenizer;
  readonly train: readonly PackedExample[];
  readonly val: readonly PackedExample[];
  readonly info: MaskedDiffusionDataInfo;
  readonly corpusHash: string;
  readonly prompts: readonly string[];
}

export interface MaskedDiffusionTrainingOptions
  extends Omit<MaskedDiffusionDataOptions, "context" | "tokenizer"> {
  readonly steps?: number;
  readonly batch?: number;
  readonly lr?: number;
  readonly seed?: number;
  readonly dropout?: number;
  readonly maxMinutes?: number;
  readonly backend?: "auto" | BrowserTrainingBackend;
  readonly initialState?: MaskedDiffusionTrainingState | null;
  /**
   * Initializes from the deployed checkpoint's weights instead of random
   * initialization. Ignored when resuming a saved run.
   */
  readonly warmStart?: boolean;
  readonly fullMaskProbability?: number;
  readonly validationEvery?: number;
  readonly maxValidationExamples?: number;
  readonly onProgress?: (progress: BrowserTrainingProgress) => void;
  readonly onCheckpoint?: (state: MaskedDiffusionTrainingState) => void;
  readonly yieldEvery?: number;
  readonly saveEvery?: number;
  /** Export the last step instead of the best-validation step. */
  readonly exportFinal?: boolean;
}

export interface MaskedDiffusionTrainingResult {
  readonly state: MaskedDiffusionTrainingState;
  readonly checkpoint: MaskedDiffusionCheckpoint;
  readonly tokenizer: BpeTokenizer;
  /** Packed in `tiny_gpt_create` order, ready for the WASM quantizer. */
  readonly weights: Float32Array;
  readonly backend: BrowserTrainingBackend;
  readonly stopped: boolean;
  readonly elapsedMs: number;
  readonly history: readonly TrainingPoint[];
  readonly data: MaskedDiffusionDataInfo;
  readonly prompts: readonly string[];
}

interface TrainingItem {
  readonly instruction: string;
  readonly answer: string;
  readonly mode: "write" | "ask";
}

export interface PackedExample {
  readonly prompt: readonly number[];
  readonly region: readonly number[];
}

interface MaskedBatch {
  readonly input: Int32Array;
  readonly target: Int32Array;
  readonly supervised: Float32Array;
}

/* ------------------------------------------------------------------ */
/* Data loading                                                       */
/* ------------------------------------------------------------------ */

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

async function fetchText(url: string): Promise<string> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`fetch failed: ${response.status} ${url}`);
  }
  return response.text();
}

function parseTaskRows(text: string, label: string): TrainingItem[] {
  const items: TrainingItem[] = [];
  const lines = text.split(/\r?\n/);
  for (let index = 0; index < lines.length; index++) {
    const line = lines[index]!.trim();
    if (line.length === 0) continue;
    let value: unknown;
    try {
      value = JSON.parse(line);
    } catch {
      throw new Error(`${label} line ${index + 1} is not valid JSON.`);
    }
    if (!isRecord(value)) {
      throw new Error(`${label} line ${index + 1} is not an object.`);
    }
    if (
      typeof value.instruction !== "string" ||
      typeof value.code !== "string"
    ) {
      throw new Error(
        `${label} line ${index + 1} is missing instruction/code.`,
      );
    }
    items.push({
      instruction: value.instruction,
      answer: value.code,
      mode: "write",
    });
  }
  if (items.length === 0) {
    throw new Error(`${label} is empty.`);
  }
  return items;
}

function parseFaqPairs(text: string): TrainingItem[] {
  const pairs: TrainingItem[] = [];
  let title: string | null = null;
  let body: string[] = [];

  const flush = (): void => {
    if (title === null) return;
    const answer = body.join("\n").trim();
    let question = title.trim();
    if (!question.endsWith("?")) {
      question = `${question.replace(/\.+$/, "")}?`;
    }
    if (answer.length > 0) {
      pairs.push({ instruction: question, answer, mode: "ask" });
    }
    title = null;
    body = [];
  };

  for (const line of text.split(/\r?\n/)) {
    const heading = /^#{2,6}[ \t]+(.+?)[ \t]*#*[ \t]*$/.exec(line);
    if (heading) {
      flush();
      title = heading[1]!.trim();
      continue;
    }
    if (title !== null) {
      body.push(line);
    }
  }
  flush();
  return pairs;
}

function packExample(
  tokenizer: BpeTokenizer,
  item: TrainingItem,
  context: number,
): PackedExample | null {
  const padId = tokenizer.idFor(BPE_SPECIAL.pad);
  const endId = tokenizer.idFor(BPE_SPECIAL.end);
  if (padId === undefined || endId === undefined) {
    throw new Error("The tokenizer must define <|pad|> and <|end|>.");
  }

  const prefix = item.mode === "write" ? "Task: " : "Question: ";
  const promptIds = tokenizer.encode(
    `${BPE_SPECIAL.user}${prefix}${item.instruction}${BPE_SPECIAL.assistant}`,
  );
  let answerIds = [
    ...tokenizer.encodeOrdinary(item.answer),
    endId,
  ];
  const available = context - promptIds.length;
  if (available < MIN_ANSWER_TOKENS) {
    return null;
  }
  if (answerIds.length > available) {
    if (item.mode === "write") {
      // Truncated code teaches broken syntax, so skip it.
      return null;
    }
    answerIds = [...answerIds.slice(0, available - 1), endId];
  }
  const region = answerIds.concat(
    new Array<number>(available - answerIds.length).fill(padId),
  );
  return { prompt: promptIds, region };
}

async function buildExamples(
  tokenizer: BpeTokenizer,
  items: readonly TrainingItem[],
  context: number,
  options: {
    readonly shouldCancel?: () => boolean;
    readonly onProgress?: (fraction: number) => void;
  } = {},
): Promise<{ examples: PackedExample[]; skipped: number }> {
  const examples: PackedExample[] = [];
  let skipped = 0;
  for (let index = 0; index < items.length; index++) {
    const packed = packExample(tokenizer, items[index]!, context);
    if (packed === null) {
      skipped += 1;
    } else {
      examples.push(packed);
    }
    if (index > 0 && index % 500 === 0) {
      options.onProgress?.(index / items.length);
      await yieldToBrowser();
      if (options.shouldCancel?.()) {
        throw new Error("Training was cancelled.");
      }
    }
  }
  options.onProgress?.(1);
  return { examples, skipped };
}

export async function loadMaskedDiffusionTrainingData(
  options: MaskedDiffusionDataOptions,
): Promise<MaskedDiffusionTrainingData> {
  const fullData = options.fullData ?? false;
  const modelBase = (options.modelBaseUrl ?? MASKED_DIFFUSION_MODEL_BASE).replace(
    /\/?$/,
    "/",
  );

  let tokenizer = options.tokenizer;
  if (!tokenizer) {
    options.onStatus?.("Loading the masked-diffusion BPE tokenizer...");
    tokenizer = await loadBpeTokenizer(`${modelBase}tokenizer.json`);
  }

  const dataBase = fullData
    ? MASKED_DIFFUSION_DATA_BASE
    : MASKED_DIFFUSION_SMALL_DATA_BASE;
  options.onStatus?.(
    fullData
      ? "Downloading the full Luau task dataset (about 14 MB)..."
      : "Loading the bundled Luau task subset...",
  );
  const [trainText, valText] = await Promise.all([
    fetchText(`${dataBase}train${fullData ? "" : "-small"}.jsonl`),
    fetchText(`${dataBase}val${fullData ? "" : "-small"}.jsonl`),
  ]);
  if (options.shouldCancel?.()) {
    throw new Error("Training was cancelled.");
  }

  options.onStatus?.("Parsing Luau tasks and FAQ examples...");
  const trainTasks = parseTaskRows(
    trainText,
    fullData ? "train.jsonl" : "train-small.jsonl",
  );
  const valTasks = parseTaskRows(
    valText,
    fullData ? "val.jsonl" : "val-small.jsonl",
  );
  const faqPairs = parseFaqPairs(luauFaq);
  const faqTrain = faqPairs.filter((_, index) => index % 5 !== 0);
  const faqVal = faqPairs.filter((_, index) => index % 5 === 0);

  // The Colab job oversamples the small FAQ so the Ask format is not drowned
  // out by 30k generated code tasks. A short browser run needs the same help.
  const faqRepeat = fullData ? 20 : 10;
  const trainItems: TrainingItem[] = [...trainTasks];
  for (const item of faqTrain) {
    for (let repeat = 0; repeat < faqRepeat; repeat++) {
      trainItems.push(item);
    }
  }
  const valItems: TrainingItem[] = [...valTasks, ...faqVal];

  options.onStatus?.(
    `Packing ${trainItems.length.toLocaleString()} training examples for ` +
      `${options.context}-token context...`,
  );
  const train = await buildExamples(tokenizer, trainItems, options.context, {
    shouldCancel: options.shouldCancel,
    onProgress: (fraction) =>
      options.onStatus?.(
        `Packing training examples (${Math.round(fraction * 100)}%)...`,
      ),
  });
  const val = await buildExamples(tokenizer, valItems, options.context, {
    shouldCancel: options.shouldCancel,
    onProgress: (fraction) =>
      options.onStatus?.(
        `Packing validation examples (${Math.round(fraction * 100)}%)...`,
      ),
  });
  if (train.examples.length === 0) {
    throw new Error(
      "No training examples fit the context. Use the full dataset or increase the context.",
    );
  }
  if (val.examples.length === 0) {
    throw new Error("No validation examples fit the context.");
  }

  const corpusHash = hashString(
    `${fullData ? "full" : "small"}\u0000${trainText.length}\u0000${valText.length}` +
      `\u0000${faqPairs.length}\u0000${tokenizer.vocab.length}\u0000${options.context}`,
  );
  const prompts = [...new Set(LUAU_PRESETS.map((preset) => preset.prompt))];

  const info: MaskedDiffusionDataInfo = {
    fullData,
    taskRows: trainTasks.length,
    valRows: valTasks.length,
    faqPairs: faqPairs.length,
    trainExamples: train.examples.length,
    valExamples: val.examples.length,
    skippedTrain: train.skipped,
    skippedVal: val.skipped,
  };
  options.onStatus?.(
    `Ready: ${info.trainExamples.toLocaleString()} training examples, ` +
      `${info.valExamples.toLocaleString()} validation examples` +
      `${fullData ? "" : " (bundled subset)"}.`,
  );

  return {
    tokenizer,
    train: train.examples,
    val: val.examples,
    info,
    corpusHash,
    prompts,
  };
}

/* ------------------------------------------------------------------ */
/* Seeded random and masking                                          */
/* ------------------------------------------------------------------ */

function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function sampleIndices(
  count: number,
  total: number,
  random: () => number,
): number[] {
  const indices: number[] = new Array(count);
  for (let index = 0; index < count; index++) {
    indices[index] = Math.floor(random() * total);
  }
  return indices;
}

function maskRegion(
  region: readonly number[],
  maskId: number,
  padId: number,
  fullMaskProbability: number,
  random: () => number,
): MaskedBatch {
  const length = region.length;
  const input = new Int32Array(length);
  const target = new Int32Array(length);
  const supervised = new Float32Array(length);
  const real: number[] = [];
  for (let index = 0; index < length; index++) {
    if (region[index] !== padId) real.push(index);
  }
  if (real.length === 0) {
    input.set(region);
    return { input, target, supervised };
  }

  let ratio = random();
  if (random() < fullMaskProbability) {
    ratio = 1;
  }
  const masked = new Array<boolean>(length);
  for (let index = 0; index < length; index++) {
    masked[index] = random() < ratio;
  }
  if (!real.some((index) => masked[index])) {
    masked[real[Math.floor(random() * real.length)]!] = true;
  }
  for (let index = 0; index < length; index++) {
    if (masked[index]) {
      input[index] = maskId;
      if (region[index] !== padId) {
        target[index] = region[index]!;
        supervised[index] = 1;
      }
    } else {
      input[index] = region[index]!;
    }
  }
  return { input, target, supervised };
}

function buildMaskedBatch(
  examples: readonly PackedExample[],
  indices: readonly number[],
  context: number,
  maskId: number,
  padId: number,
  fullMaskProbability: number,
  random: () => number,
): MaskedBatch {
  const rows = indices.length;
  const input = new Int32Array(rows * context);
  const target = new Int32Array(rows * context);
  const supervised = new Float32Array(rows * context);
  input.fill(padId);
  for (let row = 0; row < rows; row++) {
    const example = examples[indices[row]!]!;
    const promptLength = example.prompt.length;
    for (let column = 0; column < promptLength; column++) {
      input[row * context + column] = example.prompt[column]!;
    }
    const region = maskRegion(
      example.region,
      maskId,
      padId,
      fullMaskProbability,
      random,
    );
    input.set(region.input, row * context + promptLength);
    target.set(region.target, row * context + promptLength);
    supervised.set(region.supervised, row * context + promptLength);
  }
  return { input, target, supervised };
}

function buildFullMaskBatch(
  examples: readonly PackedExample[],
  context: number,
  maskId: number,
  padId: number,
): MaskedBatch {
  const rows = examples.length;
  const input = new Int32Array(rows * context);
  const target = new Int32Array(rows * context);
  const supervised = new Float32Array(rows * context);
  input.fill(maskId);
  for (let row = 0; row < rows; row++) {
    const example = examples[row]!;
    const promptLength = example.prompt.length;
    for (let column = 0; column < promptLength; column++) {
      input[row * context + column] = example.prompt[column]!;
    }
    for (let offset = 0; offset < example.region.length; offset++) {
      const token = example.region[offset]!;
      if (token !== padId) {
        target[row * context + promptLength + offset] = token;
        supervised[row * context + promptLength + offset] = 1;
      }
    }
  }
  return { input, target, supervised };
}

/* ------------------------------------------------------------------ */
/* Loss                                                               */
/* ------------------------------------------------------------------ */

function maskedLoss(
  tf: TfModule,
  model: ModelVars,
  config: TinyGptConfig,
  input: Tensor,
  target: Tensor,
  supervised: Tensor,
  training: boolean,
  dropoutSeed: number,
  dropoutRate: number,
): Scalar {
  return tf.tidy(() => {
    const logits = forward(
      tf,
      model,
      input,
      config,
      training,
      dropoutSeed,
      dropoutRate,
      true,
    );
    const batch = input.shape[0]!;
    const context = input.shape[1]!;
    const flatLogits = tf.reshape(logits, [batch * context, config.vocabSize]);
    const flatTargets = tf.reshape(target, [batch * context]);
    const flatMask = tf.reshape(supervised, [batch * context]);
    const oneHot = tf.oneHot(
      flatTargets,
      config.vocabSize,
      1,
      0,
      "float32",
    );
    const logProbs = tf.logSoftmax(flatLogits, -1);
    const tokenLogProb = tf.sum(tf.mul(oneHot, logProbs), -1);
    const total = tf.sum(tf.mul(tokenLogProb, flatMask));
    const count = tf.sum(flatMask);
    return tf.neg(tf.div(total, tf.add(count, 1e-8))) as Scalar;
  });
}

async function computeFullMaskLoss(
  tf: TfModule,
  model: ModelVars,
  config: TinyGptConfig,
  examples: readonly PackedExample[],
  batch: number,
  maskId: number,
  padId: number,
): Promise<number> {
  if (examples.length === 0) {
    return Number.NaN;
  }
  let total = 0;
  let count = 0;
  for (let start = 0; start < examples.length; start += batch) {
    const chunk = examples.slice(start, start + batch);
    const data = buildFullMaskBatch(chunk, config.context, maskId, padId);
    const input = tf.tensor2d(
      data.input,
      [chunk.length, config.context],
      "int32",
    );
    const target = tf.tensor2d(
      data.target,
      [chunk.length, config.context],
      "int32",
    );
    const supervised = tf.tensor2d(
      data.supervised,
      [chunk.length, config.context],
      "float32",
    );
    const loss = maskedLoss(
      tf,
      model,
      config,
      input,
      target,
      supervised,
      false,
      0,
      0,
    );
    try {
      const value = await loss.data();
      let supervisedCount = 0;
      for (let index = 0; index < data.supervised.length; index++) {
        supervisedCount += data.supervised[index]!;
      }
      total += Number(value[0] ?? 0) * supervisedCount;
      count += supervisedCount;
    } finally {
      loss.dispose();
      input.dispose();
      target.dispose();
      supervised.dispose();
    }
  }
  return count > 0 ? total / count : Number.NaN;
}

/* ------------------------------------------------------------------ */
/* Checkpoint and state helpers                                       */
/* ------------------------------------------------------------------ */

interface BundledMaskedModel {
  readonly checkpoint: MaskedDiffusionCheckpoint;
  readonly weights: Float32Array;
}

async function loadBundledMaskedModel(
  baseUrl: string,
): Promise<BundledMaskedModel> {
  const base = baseUrl.replace(/\/?$/, "/");
  const [checkpointResponse, weightsResponse] = await Promise.all([
    fetch(`${base}checkpoint.json`),
    fetch(`${base}weights.bin`),
  ]);
  if (!checkpointResponse.ok) {
    throw new Error(
      `fetch failed: ${checkpointResponse.status} ${base}checkpoint.json`,
    );
  }
  if (!weightsResponse.ok) {
    throw new Error(
      `fetch failed: ${weightsResponse.status} ${base}weights.bin`,
    );
  }
  return {
    checkpoint: (await checkpointResponse.json()) as MaskedDiffusionCheckpoint,
    weights: new Float32Array(await weightsResponse.arrayBuffer()),
  };
}

function makeState(
  config: TinyGptConfig,
  step: number,
  steps: number,
  batch: number,
  lr: number,
  dropout: number,
  seed: number,
  bestStep: number,
  bestLoss: number,
  history: readonly TrainingPoint[],
  weights: Float32Array,
  optimizer: { readonly m: Float32Array; readonly v: Float32Array } | null,
  bestWeights: Float32Array | null,
  corpusHash: string,
  elapsedMs: number,
  fullData: boolean,
  warmStart: boolean,
): MaskedDiffusionTrainingState {
  return {
    format: "only-gates-masked-diffusion-train",
    version: 1,
    config,
    step,
    steps,
    batch,
    lr,
    dropout,
    seed,
    bestStep,
    bestLoss,
    history: [...history],
    weights,
    optimizer,
    bestWeights,
    corpusHash,
    elapsedMs,
    savedAt: Date.now(),
    fullData,
    warmStart,
  };
}

function makeCheckpoint(
  config: TinyGptConfig,
  maskId: number,
  meta: {
    readonly steps: number;
    readonly batch: number;
    readonly learningRate: number;
    readonly dropout: number;
    readonly seed: number;
    readonly bestStep: number;
    readonly bestLoss: number;
    readonly elapsedMs: number;
    readonly corpusHash: string;
    readonly fullMaskProbability: number;
    readonly fullData: boolean;
  },
  prompts: readonly string[],
): MaskedDiffusionCheckpoint {
  return {
    format: "only-gates-masked-diffusion-luau",
    version: 1,
    config,
    parameterCount: maskedDiffusionParameterCount(config),
    tokenizer: "tokenizer.json",
    diffusion: {
      maskTokenId: maskId,
      steps: DEFAULT_DENOISING_STEPS,
      schedule: "left-to-right",
      // 0 means "fill the rest of the context", matching training.
      answerBudget: 0,
    },
    meta: {
      name: "masked-diffusion-luau-browser",
      source:
        "Luau task data and the repository FAQ, trained in the browser with " +
        "TensorFlow.js. The denoiser is a bidirectional Tiny GPT trained with " +
        "an absorbing <|mask|> token.",
      license:
        "Generated task data and model weights are produced by this repository. " +
        "Luau documentation and the FAQ retain their own MIT licenses.",
      training: {
        steps: meta.steps,
        batch: meta.batch,
        learningRate: meta.learningRate,
        dropout: meta.dropout,
        seed: meta.seed,
        bestStep: meta.bestStep,
        bestLoss: Number.isFinite(meta.bestLoss) ? meta.bestLoss : null,
        elapsedMs: Math.max(0, Math.round(meta.elapsedMs)),
        corpusHash: meta.corpusHash,
        maskRatio: "uniform+t-full",
        fullMaskProbability: meta.fullMaskProbability,
        denoisingSteps: DEFAULT_DENOISING_STEPS,
        sourceUrls: [
          meta.fullData
            ? "data/luau-tasks/train.jsonl"
            : "data/luau-tasks/train-small.jsonl",
          "apps/only-gates-ai/src/assets/luau-faq.md",
        ],
        backend: "tensorflow.js",
      },
    },
    samples: [],
    prompts: [...prompts],
  };
}

/* ------------------------------------------------------------------ */
/* Training                                                           */
/* ------------------------------------------------------------------ */

export async function trainMaskedDiffusion(
  options: MaskedDiffusionTrainingOptions = {},
): Promise<MaskedDiffusionTrainingResult> {
  const initial = options.initialState ?? null;
  const modelBase = (options.modelBaseUrl ?? MASKED_DIFFUSION_MODEL_BASE).replace(
    /\/?$/,
    "/",
  );

  options.onStatus?.("Loading the masked-diffusion tokenizer...");
  const tokenizer = await loadBpeTokenizer(`${modelBase}tokenizer.json`);

  let config: TinyGptConfig;
  let bundled: BundledMaskedModel | null = null;
  if (initial) {
    config = initial.config;
  } else if (options.warmStart !== false) {
    options.onStatus?.("Loading the deployed checkpoint for a warm start...");
    bundled = await loadBundledMaskedModel(modelBase);
    config = bundled.checkpoint.config;
  } else {
    config = {
      ...MASKED_BROWSER_TRAIN_CONFIG,
      vocabSize: tokenizer.vocab.length,
    };
  }
  if (config.vocabSize !== tokenizer.vocab.length) {
    throw new Error(
      `The model config has ${config.vocabSize.toLocaleString()} vocabulary entries ` +
        `but the tokenizer has ${tokenizer.vocab.length.toLocaleString()}.`,
    );
  }
  if (config.dim % config.heads !== 0) {
    throw new Error("dim must be divisible by heads.");
  }

  const data = await loadMaskedDiffusionTrainingData({
    fullData: options.fullData,
    modelBaseUrl: modelBase,
    context: config.context,
    tokenizer,
    onStatus: options.onStatus,
    shouldCancel: options.shouldCancel,
  });
  if (initial) {
    if (initial.corpusHash !== data.corpusHash) {
      throw new Error(
        "The saved run was trained on different data. Clear it before starting again.",
      );
    }
    if (initial.fullData !== data.info.fullData) {
      throw new Error(
        "The saved run used a different dataset size. Clear it or select the matching data option.",
      );
    }
  }

  const maskId = tokenizer.idFor(BPE_SPECIAL.mask);
  const padId = tokenizer.idFor(BPE_SPECIAL.pad);
  if (maskId === undefined || padId === undefined) {
    throw new Error("The tokenizer must define <|mask|> and <|pad|>.");
  }

  const batch = Math.max(1, initial?.batch ?? options.batch ?? 4);
  const learningRate = initial?.lr ?? options.lr ?? 4e-4;
  const dropoutRate = initial?.dropout ?? options.dropout ?? 0.1;
  const seed = initial?.seed ?? options.seed ?? 1337;
  const totalSteps = Math.max(
    options.steps ?? initial?.steps ?? 400,
    initial?.step ?? 0,
  );
  const validationEvery = Math.max(1, options.validationEvery ?? 20);
  const maxValidationExamples = Math.max(
    1,
    options.maxValidationExamples ?? DEFAULT_VALIDATION_EXAMPLES,
  );
  const saveEvery = Math.max(1, options.saveEvery ?? 200);
  const yieldEvery = Math.max(1, options.yieldEvery ?? 2);
  const maxMinutes = options.maxMinutes ?? 0;
  const fullMaskProbability =
    options.fullMaskProbability ?? DEFAULT_FULL_MASK_PROBABILITY;
  const startedAt = performance.now();
  const baseElapsed = initial?.elapsedMs ?? 0;
  const elapsed = (): number =>
    baseElapsed + (performance.now() - startedAt);

  options.onStatus?.("Selecting the TensorFlow.js backend...");
  const { tf, backend } = await selectTensorFlowBackend(
    options.backend ?? "auto",
  );

  let model: ModelVars | null = null;
  let optimizer: OptimizerVars | null = null;
  try {
    if (initial) {
      model = restoreModelFromPacked(tf, config, initial.weights);
      optimizer = initial.optimizer
        ? restoreOptimizerFromPacked(
            tf,
            config,
            initial.optimizer.m,
            initial.optimizer.v,
          )
        : createOptimizer(tf, config, model);
    } else if (bundled) {
      model = restoreModelFromPacked(tf, config, bundled.weights);
      optimizer = createOptimizer(tf, config, model);
    } else {
      model = createModel(tf, config, seed);
      optimizer = createOptimizer(tf, config, model);
    }

    const params = [
      model.tokenEmb,
      model.posEmb,
      model.gFinal,
      ...model.layers.flatMap((layer) => [
        layer.g1,
        layer.wqkv,
        layer.wo,
        layer.g2,
        layer.w1,
        layer.w2,
      ]),
    ];
    const history: TrainingPoint[] = initial ? [...initial.history] : [];
    let smooth: number | null =
      history.length > 0 ? history[history.length - 1]!.smoothedLoss : null;
    let bestLoss = initial?.bestLoss ?? Number.POSITIVE_INFINITY;
    let bestStep = initial?.bestStep ?? 0;
    let bestWeights = initial?.bestWeights ?? null;
    let stopped = false;
    let lastStep = initial?.step ?? 0;

    const warmup = Math.max(1, Math.floor(totalSteps * 0.05));
    const lrAt = (step: number): number => {
      if (step < warmup) {
        return learningRate * ((step + 1) / warmup);
      }
      const progress = Math.min(
        1,
        (step - warmup) / Math.max(1, totalSteps - warmup),
      );
      return learningRate * 0.5 * (1 + Math.cos(Math.PI * progress));
    };

    const validationExamples = data.val.slice(0, maxValidationExamples);
    const validationBatch = Math.max(1, Math.min(batch, 8));

    for (let step = lastStep + 1; step <= totalSteps; step++) {
      if (options.shouldCancel?.()) {
        stopped = true;
        break;
      }
      if (maxMinutes > 0 && performance.now() - startedAt > maxMinutes * 60_000) {
        stopped = true;
        break;
      }

      const random = mulberry32(
        (seed ^ Math.imul(step, 0x9e3779b9)) >>> 0,
      );
      const indices = sampleIndices(batch, data.train.length, random);
      const batchData = buildMaskedBatch(
        data.train,
        indices,
        config.context,
        maskId,
        padId,
        fullMaskProbability,
        random,
      );
      const input = tf.tensor2d(
        batchData.input,
        [batch, config.context],
        "int32",
      );
      const target = tf.tensor2d(
        batchData.target,
        [batch, config.context],
        "int32",
      );
      const supervised = tf.tensor2d(
        batchData.supervised,
        [batch, config.context],
        "float32",
      );
      const lr = lrAt(step);

      let lossValue = 0;
      let lossTensor: Tensor | null = null;
      let rawGrads: Tensor[] | null = null;
      try {
        const result = tf.variableGrads(
          () =>
            maskedLoss(
              tf,
              model!,
              config,
              input,
              target,
              supervised,
              true,
              (seed ^ step) >>> 0,
              dropoutRate,
            ),
          params,
        );
        lossTensor = result.value;
        const value = await lossTensor.data();
        lossValue = Number(value[0] ?? 0);
        if (!Number.isFinite(lossValue)) {
          throw new Error(
            "Training diverged with a non-finite loss. Try a smaller learning rate.",
          );
        }
        rawGrads = params.map((param) => {
          const grad = result.grads[param.name];
          if (!grad) {
            throw new Error(`Missing gradient for ${param.name}.`);
          }
          return grad;
        });
        const clipped = clipGradients(tf, rawGrads, 1);
        tf.dispose(rawGrads);
        rawGrads = null;
        applyAdam(tf, params, clipped, optimizer, step, lr, {
          beta1: 0.9,
          beta2: 0.95,
          epsilon: 1e-8,
          weightDecay: 0.01,
        });
        tf.dispose(clipped);
      } finally {
        lossTensor?.dispose();
        if (rawGrads) {
          tf.dispose(rawGrads);
        }
        input.dispose();
        target.dispose();
        supervised.dispose();
      }

      smooth =
        smooth === null ? lossValue : smooth * 0.95 + lossValue * 0.05;

      let valLoss: number | undefined;
      if (step % validationEvery === 0 || step === totalSteps) {
        const value = await computeFullMaskLoss(
          tf,
          model,
          config,
          validationExamples,
          validationBatch,
          maskId,
          padId,
        );
        if (Number.isFinite(value)) {
          valLoss = value;
          if (value < bestLoss) {
            bestLoss = value;
            bestStep = step;
            bestWeights = packModelWeights(config, model);
          }
        }
      }

      lastStep = step;
      const point: TrainingPoint = {
        step,
        loss: lossValue,
        smoothedLoss: smooth,
        valLoss,
        lr,
        elapsedMs: elapsed(),
      };
      history.push(point);
      options.onProgress?.({
        ...point,
        totalSteps,
        bestLoss: Number.isFinite(bestLoss) ? bestLoss : null,
        bestStep,
        backend,
        stepsPerSecond:
          (step - (initial?.step ?? 0)) /
          Math.max(0.001, (performance.now() - startedAt) / 1000),
      });

      if (step % saveEvery === 0 && step < totalSteps) {
        options.onCheckpoint?.(
          makeState(
            config,
            step,
            totalSteps,
            batch,
            learningRate,
            dropoutRate,
            seed,
            bestStep,
            bestLoss,
            history,
            packModelWeights(config, model),
            packOptimizerWeights(config, optimizer),
            bestWeights,
            data.corpusHash,
            elapsed(),
            data.info.fullData,
            options.warmStart !== false,
          ),
        );
      }

      if (step % yieldEvery === 0) {
        await yieldToBrowser();
      }
    }

    const currentWeights = packModelWeights(config, model);
    const finalWeights =
      options.exportFinal === true ? currentWeights : bestWeights ?? currentWeights;
    const state = makeState(
      config,
      lastStep,
      totalSteps,
      batch,
      learningRate,
      dropoutRate,
      seed,
      bestStep,
      bestLoss,
      history,
      currentWeights,
      packOptimizerWeights(config, optimizer),
      bestWeights,
      data.corpusHash,
      elapsed(),
      data.info.fullData,
      options.warmStart !== false,
    );
    options.onCheckpoint?.(state);

    return {
      state,
      checkpoint: makeCheckpoint(
        config,
        maskId,
        {
          steps: lastStep,
          batch,
          learningRate,
          dropout: dropoutRate,
          seed,
          bestStep,
          bestLoss,
          elapsedMs: elapsed(),
          corpusHash: data.corpusHash,
          fullMaskProbability,
          fullData: data.info.fullData,
        },
        data.prompts,
      ),
      tokenizer,
      weights: finalWeights,
      backend,
      stopped,
      elapsedMs: elapsed(),
      history,
      data: data.info,
      prompts: data.prompts,
    };
  } finally {
    if (optimizer) {
      disposeOptimizer(optimizer);
    }
    if (model) {
      disposeModel(model);
    }
  }
}

/** Builds the integer-only WASM model from a completed browser training run. */
export function buildMaskedDiffusionWasmFromTraining(
  result: MaskedDiffusionTrainingResult,
): Promise<MaskedDiffusionModel> {
  return createMaskedDiffusionWasmFromData(
    result.checkpoint,
    result.tokenizer,
    result.weights,
  );
}

/* ------------------------------------------------------------------ */
/* Model zip export / import                                          */
/* ------------------------------------------------------------------ */

export interface MaskedDiffusionBundle {
  readonly zip: Uint8Array;
  readonly filename: string;
}

function weightBytes(weights: Float32Array): Uint8Array {
  return new Uint8Array(
    weights.buffer,
    weights.byteOffset,
    weights.byteLength,
  ).slice();
}

export async function exportMaskedDiffusionBundle(
  result: MaskedDiffusionTrainingResult,
): Promise<MaskedDiffusionBundle> {
  const files: ZipFileInput[] = [
    {
      name: "checkpoint.json",
      data: JSON.stringify(result.checkpoint, null, 2),
    },
    {
      name: "tokenizer.json",
      data: JSON.stringify(serializeBpeTokenizer(result.tokenizer), null, 2),
    },
    { name: "weights.bin", data: weightBytes(result.weights) },
    {
      name: "training.json",
      data: JSON.stringify(
        {
          format: "only-gates-masked-diffusion-training-summary",
          version: 1,
          training: result.checkpoint.meta?.training,
          history: result.history,
        },
        null,
        2,
      ),
    },
    {
      name: "README.txt",
      data:
        "Only Gates AI — masked-diffusion Luau model\n\n" +
        "Files:\n" +
        "  checkpoint.json  model config and diffusion schedule\n" +
        "  tokenizer.json   byte-level BPE with <|mask|>\n" +
        "  weights.bin      float32 little-endian packed weights\n\n" +
        "Import this zip from the Masked diffusion training panel, or copy the\n" +
        "three model files into apps/only-gates-ai/public/models/masked-diffusion-luau/.\n",
    },
  ];
  const zip = await createZip(files);
  const stamp = new Date()
    .toISOString()
    .replace(/\.\d{3}Z$/, "Z")
    .replace(/:/g, "-");
  return {
    zip,
    filename: `only-gates-masked-diffusion-${stamp}.zip`,
  };
}

export function downloadMaskedDiffusionBundle(
  bundle: MaskedDiffusionBundle,
): void {
  const blob = new Blob([new Uint8Array(bundle.zip)], {
    type: "application/zip",
  });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = bundle.filename;
  anchor.rel = "noopener";
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

export interface ImportedMaskedDiffusionBundle {
  readonly checkpoint: MaskedDiffusionCheckpoint;
  readonly tokenizer: BpeTokenizer;
  readonly weights: Float32Array;
}

export async function importMaskedDiffusionBundle(
  file: File | Blob | ArrayBuffer | Uint8Array,
): Promise<ImportedMaskedDiffusionBundle> {
  const files = await readZip(file);
  const checkpointBytes = files.get("checkpoint.json");
  const tokenizerBytes = files.get("tokenizer.json");
  const weightsBytes = files.get("weights.bin");
  if (!checkpointBytes || !tokenizerBytes || !weightsBytes) {
    throw new Error(
      "The zip must contain checkpoint.json, tokenizer.json, and weights.bin.",
    );
  }
  if (weightsBytes.byteLength % 4 !== 0) {
    throw new Error("weights.bin length is not a multiple of 4 bytes.");
  }

  const decoder = new TextDecoder();
  const checkpoint = JSON.parse(
    decoder.decode(checkpointBytes),
  ) as MaskedDiffusionCheckpoint;
  if (checkpoint.format !== "only-gates-masked-diffusion-luau") {
    throw new Error("checkpoint.json is not a masked-diffusion Luau model.");
  }
  const tokenizerModel = JSON.parse(
    decoder.decode(tokenizerBytes),
  ) as BpeModel;
  const tokenizer = parseBpeTokenizer(tokenizerModel);
  const weights = new Float32Array(weightsBytes.slice().buffer);
  const expected = maskedDiffusionParameterCount(checkpoint.config);
  if (weights.length !== expected) {
    throw new Error(
      `weights.bin has ${weights.length.toLocaleString()} values but the checkpoint needs ` +
        `${expected.toLocaleString()}.`,
    );
  }
  return { checkpoint, tokenizer, weights };
}
