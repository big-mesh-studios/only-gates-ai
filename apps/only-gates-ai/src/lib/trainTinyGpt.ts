import type { Scalar, Tensor, Variable } from "@tensorflow/tfjs";

import { decodeFloat32, encodeFloat32 } from "./base64Float32";
import type { TinyGptCheckpoint, TinyGptConfig } from "./tinyGpt";

/*
 * Browser-side training for the Alice-sized Tiny GPT.
 *
 * Training is float32. The integer path starts after export: the existing
 * quantizer and WASM loader consume the same checkpoint format as the bundled
 * model.
 *
 * Backend order is explicit: WebGPU, then WebGL, then CPU. TensorFlow.js ships
 * WebGL in the main package; the WebGPU backend is a separate package.
 */

export type BrowserTrainingBackend = "webgpu" | "webgl" | "cpu";

export const DEFAULT_BROWSER_TRAIN_CONFIG: TinyGptConfig = {
  vocabSize: 512,
  context: 16,
  dim: 48,
  layers: 2,
  heads: 2,
  mlp: 96,
};

export const BROWSER_TRAIN_SPECIALS = ["<unk>", "<bos>", "<eos>"] as const;
export const BROWSER_TRAIN_UNK = 0;
export const BROWSER_TRAIN_EOS = 2;

export interface TrainingPoint {
  readonly step: number;
  readonly loss: number;
  readonly smoothedLoss: number;
  readonly valLoss?: number;
  readonly lr: number;
  readonly elapsedMs: number;
}

export interface BrowserTrainingProgress extends TrainingPoint {
  readonly totalSteps: number;
  readonly bestLoss: number | null;
  readonly bestStep: number;
  readonly backend: BrowserTrainingBackend;
  readonly stepsPerSecond: number;
}

export type SerializedTinyGptWeights = TinyGptCheckpoint["weights"];

export type BrowserTrainingKind = "alice" | "solid2" | "custom";

export interface BrowserTrainingState {
  readonly format: "only-gates-browser-train";
  readonly version: 1;
  readonly kind: BrowserTrainingKind;
  readonly config: TinyGptConfig;
  readonly vocab: readonly string[];
  readonly step: number;
  readonly steps: number;
  readonly batch: number;
  readonly lr: number;
  readonly dropout: number;
  readonly seed: number;
  readonly bestStep: number;
  readonly bestLoss: number;
  readonly history: readonly TrainingPoint[];
  readonly weights: SerializedTinyGptWeights;
  readonly optimizer: {
    readonly m: SerializedTinyGptWeights;
    readonly v: SerializedTinyGptWeights;
  };
  readonly bestWeights: SerializedTinyGptWeights | null;
  readonly corpusHash: string;
  readonly elapsedMs: number;
  readonly savedAt: number;
  /** Source URLs for a custom-docs run, so a saved run can be resumed. */
  readonly sourceUrls?: readonly string[];
  /** True when a custom-docs run used assistant (Q&A) mode. */
  readonly assistant?: boolean;
}

export interface BrowserTrainingOptions {
  readonly corpus: string;
  readonly config?: Partial<TinyGptConfig>;
  readonly vocabLimit?: number;
  readonly steps?: number;
  readonly batch?: number;
  readonly lr?: number;
  readonly seed?: number;
  readonly dropout?: number;
  readonly maxMinutes?: number;
  readonly backend?: "auto" | BrowserTrainingBackend;
  readonly initialState?: BrowserTrainingState | null;
  readonly onProgress?: (progress: BrowserTrainingProgress) => void;
  readonly onCheckpoint?: (state: BrowserTrainingState) => void;
  readonly shouldCancel?: () => boolean;
  readonly yieldEvery?: number;
  readonly saveEvery?: number;
}

export interface BrowserTrainingResult {
  readonly checkpoint: TinyGptCheckpoint;
  readonly state: BrowserTrainingState;
  readonly backend: BrowserTrainingBackend;
  readonly stopped: boolean;
  readonly elapsedMs: number;
  readonly history: readonly TrainingPoint[];
}

/**
 * A pre-tokenized corpus. The generic trainer never looks at text, so callers
 * can supply word-level ids, byte-level BPE ids, or any other token stream.
 */
export interface TokenTrainingData {
  readonly vocab: readonly string[];
  readonly trainingTokens: ArrayLike<number>;
  readonly validationTokens: ArrayLike<number>;
  readonly corpusHash: string;
  /** Recorded in checkpoint metadata. Defaults to the training token count. */
  readonly corpusSize?: number;
  /** Metadata key for `corpusSize`. Defaults to "corpusTokens". */
  readonly corpusLabel?: string;
}

export interface TokenTrainingModelMeta {
  readonly name: string;
  readonly source: string;
  readonly license: string;
}

export interface TokenTrainingOptions {
  readonly data: TokenTrainingData;
  readonly config?: Partial<TinyGptConfig>;
  readonly defaultConfig?: TinyGptConfig;
  readonly steps?: number;
  readonly batch?: number;
  readonly lr?: number;
  readonly seed?: number;
  readonly dropout?: number;
  readonly maxMinutes?: number;
  readonly backend?: "auto" | BrowserTrainingBackend;
  readonly initialState?: BrowserTrainingState | null;
  /**
   * Packed float32 weights in `tiny_gpt_create` order. Only used when there is
   * no `initialState`; this warm-starts the model while keeping a fresh
   * optimizer, history, and step counter.
   */
  readonly initialWeights?: Float32Array;
  readonly onProgress?: (progress: BrowserTrainingProgress) => void;
  readonly onCheckpoint?: (state: BrowserTrainingState) => void;
  readonly shouldCancel?: () => boolean;
  readonly yieldEvery?: number;
  readonly saveEvery?: number;
  readonly kind?: BrowserTrainingKind;
  readonly modelMeta: TokenTrainingModelMeta;
  /**
   * Export the final training weights instead of the best-validation
   * checkpoint. Small custom Q&A models often overfit their instruction data,
   * so the best-validation step can be far too early.
   */
  readonly exportFinal?: boolean;
}

export interface WordCorpusData {
  readonly vocab: readonly string[];
  readonly words: readonly string[];
  readonly trainingTokens: readonly number[];
  readonly validationTokens: readonly number[];
}

interface ParamSpec {
  readonly name: string;
  readonly shape: number[];
}

interface ModelVars {
  readonly tokenEmb: Variable;
  readonly posEmb: Variable;
  readonly gFinal: Variable;
  readonly layers: readonly {
    readonly g1: Variable;
    readonly wqkv: Variable;
    readonly wo: Variable;
    readonly g2: Variable;
    readonly w1: Variable;
    readonly w2: Variable;
  }[];
}

interface OptimizerVars {
  readonly m: readonly Variable[];
  readonly v: readonly Variable[];
}

interface BatchData {
  readonly input: Int32Array;
  readonly target: Int32Array;
}

interface WindowData extends BatchData {
  readonly count: number;
}

type TfModule = typeof import("@tensorflow/tfjs");

let tfPromise: Promise<TfModule> | null = null;

/** Loads TensorFlow.js lazily so the main app bundle stays small. */
export async function loadTensorFlow(): Promise<TfModule> {
  if (tfPromise === null) {
    tfPromise = (async () => {
      const tf = await import("@tensorflow/tfjs");
      if (typeof navigator !== "undefined" && "gpu" in navigator) {
        try {
          await import("@tensorflow/tfjs-backend-webgpu");
        } catch {
          // The WebGL and CPU backends still work without the WebGPU package.
        }
      }
      return tf;
    })();
  }
  return tfPromise;
}

/**
 * Selects the best available backend. WebGL is a first-class fallback, not an
 * error path: `@tensorflow/tfjs` registers it in every supported browser.
 */
export async function selectTensorFlowBackend(
  preferred: "auto" | BrowserTrainingBackend = "auto",
): Promise<{ readonly tf: TfModule; readonly backend: BrowserTrainingBackend }> {
  const tf = await loadTensorFlow();
  const candidates: BrowserTrainingBackend[] =
    preferred === "auto"
      ? typeof navigator !== "undefined" && "gpu" in navigator
        ? ["webgpu", "webgl", "cpu"]
        : ["webgl", "cpu"]
      : [preferred];

  for (const backend of candidates) {
    try {
      await tf.setBackend(backend);
      await tf.ready();
      if (tf.getBackend() === backend) {
        return { tf, backend };
      }
    } catch {
      // Try the next backend.
    }
  }

  await tf.ready();
  const actual = tf.getBackend();
  if (actual === "webgpu" || actual === "webgl" || actual === "cpu") {
    return { tf, backend: actual };
  }
  throw new Error("No usable TensorFlow.js backend was found.");
}

/* ------------------------------------------------------------------ */
/* Corpus and tokenizer                                               */
/* ------------------------------------------------------------------ */

export function tokenizeWords(text: string): string[] {
  return text.toLowerCase().match(/[a-z0-9']+/g) ?? [];
}

export function buildWordVocabulary(
  tokens: readonly string[],
  limit: number,
): string[] {
  const counts = new Map<string, number>();
  for (const token of tokens) {
    counts.set(token, (counts.get(token) ?? 0) + 1);
  }
  const words = [...counts.entries()]
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .slice(0, Math.max(0, limit - BROWSER_TRAIN_SPECIALS.length))
    .map(([word]) => word);
  return [...BROWSER_TRAIN_SPECIALS, ...words];
}

export function buildWordCorpus(
  corpus: string,
  vocabLimit = 512,
  existingVocab?: readonly string[],
): WordCorpusData {
  const words = tokenizeWords(corpus);
  const vocab = existingVocab
    ? [...existingVocab]
    : buildWordVocabulary(words, vocabLimit);
  const index = new Map(vocab.map((word, id) => [word, id]));

  const withEos: number[] = [];
  for (const paragraph of corpus.split(/\n\s*\n/)) {
    for (const word of tokenizeWords(paragraph)) {
      withEos.push(index.get(word) ?? BROWSER_TRAIN_UNK);
    }
    withEos.push(BROWSER_TRAIN_EOS);
  }

  const split = Math.floor(withEos.length * 0.85);
  return {
    vocab,
    words,
    trainingTokens: withEos.slice(0, split),
    validationTokens: withEos.slice(split),
  };
}

export function hashString(value: string): string {
  let hash = 0x811c9dc5;
  for (let index = 0; index < value.length; index++) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193);
  }
  return (hash >>> 0).toString(16).padStart(8, "0");
}

/* ------------------------------------------------------------------ */
/* Seeded random and batches                                          */
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

function sampleBatch(
  tokens: ArrayLike<number>,
  context: number,
  batch: number,
  random: () => number,
): BatchData {
  const maxStart = tokens.length - context - 1;
  if (maxStart < 0) {
    throw new Error("The training corpus is too short for this context size.");
  }
  const input = new Int32Array(batch * context);
  const target = new Int32Array(batch * context);
  for (let row = 0; row < batch; row++) {
    const start = Math.floor(random() * (maxStart + 1));
    for (let column = 0; column < context; column++) {
      input[row * context + column] = tokens[start + column]!;
      target[row * context + column] = tokens[start + column + 1]!;
    }
  }
  return { input, target };
}

function buildWindows(
  tokens: ArrayLike<number>,
  context: number,
): WindowData {
  const count = Math.floor((tokens.length - 1) / context);
  if (count <= 0) {
    return {
      input: new Int32Array(0),
      target: new Int32Array(0),
      count: 0,
    };
  }
  const input = new Int32Array(count * context);
  const target = new Int32Array(count * context);
  for (let window = 0; window < count; window++) {
    const start = window * context;
    for (let column = 0; column < context; column++) {
      input[window * context + column] = tokens[start + column]!;
      target[window * context + column] = tokens[start + column + 1]!;
    }
  }
  return { input, target, count };
}

export function yieldToBrowser(): Promise<void> {
  if (typeof requestAnimationFrame === "function") {
    return new Promise((resolve) => requestAnimationFrame(() => resolve()));
  }
  return new Promise((resolve) => setTimeout(resolve, 0));
}

/* ------------------------------------------------------------------ */
/* Model                                                              */
/* ------------------------------------------------------------------ */

function paramSpecs(config: TinyGptConfig): ParamSpec[] {
  const { vocabSize, context, dim, layers, mlp } = config;
  const specs: ParamSpec[] = [
    { name: "tokenEmb", shape: [vocabSize, dim] },
    { name: "posEmb", shape: [context, dim] },
    { name: "gFinal", shape: [dim] },
  ];
  for (let layer = 0; layer < layers; layer++) {
    specs.push(
      { name: `layer_${layer}_g1`, shape: [dim] },
      { name: `layer_${layer}_wqkv`, shape: [dim, dim * 3] },
      { name: `layer_${layer}_wo`, shape: [dim, dim] },
      { name: `layer_${layer}_g2`, shape: [dim] },
      { name: `layer_${layer}_w1`, shape: [dim, mlp] },
      { name: `layer_${layer}_w2`, shape: [mlp, dim] },
    );
  }
  return specs;
}

function flattenModel(model: ModelVars): Variable[] {
  const values: Variable[] = [model.tokenEmb, model.posEmb, model.gFinal];
  for (const layer of model.layers) {
    values.push(layer.g1, layer.wqkv, layer.wo, layer.g2, layer.w1, layer.w2);
  }
  return values;
}

function modelFromVariables(
  config: TinyGptConfig,
  variables: readonly Variable[],
): ModelVars {
  let index = 0;
  const tokenEmb = variables[index++]!;
  const posEmb = variables[index++]!;
  const gFinal = variables[index++]!;
  const layers = Array.from({ length: config.layers }, () => ({
    g1: variables[index++]!,
    wqkv: variables[index++]!,
    wo: variables[index++]!,
    g2: variables[index++]!,
    w1: variables[index++]!,
    w2: variables[index++]!,
  }));
  return { tokenEmb, posEmb, gFinal, layers };
}

function makeVariable(
  tf: TfModule,
  name: string,
  value: Tensor,
  trainable = true,
): Variable {
  const variable = tf.variable(value, trainable, name);
  value.dispose();
  return variable;
}

function createModel(
  tf: TfModule,
  config: TinyGptConfig,
  seed: number,
): ModelVars {
  let nextSeed = seed >>> 0;
  const takeSeed = (): number => {
    nextSeed = (nextSeed + 0x9e3779b9) >>> 0;
    return nextSeed;
  };
  const residualStd = 0.02 / Math.sqrt(2 * Math.max(1, config.layers));
  const specs = paramSpecs(config);
  const variables = specs.map((spec) => {
    if (spec.name.endsWith("_g1") || spec.name.endsWith("_g2") || spec.name === "gFinal") {
      return makeVariable(tf, spec.name, tf.ones(spec.shape, "float32"));
    }
    const std = spec.name === "posEmb"
      ? 0.01
      : spec.name.endsWith("_wo") || spec.name.endsWith("_w2")
        ? residualStd
        : 0.02;
    return makeVariable(
      tf,
      spec.name,
      tf.randomNormal(spec.shape, 0, std, "float32", takeSeed()),
    );
  });
  return modelFromVariables(config, variables);
}

function restoreModel(
  tf: TfModule,
  config: TinyGptConfig,
  weights: SerializedTinyGptWeights,
): ModelVars {
  const flat = flattenSerializedWeights(config, weights);
  const variables = paramSpecs(config).map((spec, index) => {
    const values = decodeFloat32(flat[index]!);
    const tensor = tf.tensor(values, spec.shape, "float32");
    return makeVariable(tf, spec.name, tensor);
  });
  return modelFromVariables(config, variables);
}

function restoreModelFromPacked(
  tf: TfModule,
  config: TinyGptConfig,
  packed: Float32Array,
): ModelVars {
  const specs = paramSpecs(config);
  const expected = specs.reduce(
    (total, spec) => total + spec.shape.reduce((a, b) => a * b, 1),
    0,
  );
  if (packed.length !== expected) {
    throw new Error(
      `Warm-start weights have ${packed.length.toLocaleString()} values but the config needs ` +
        `${expected.toLocaleString()} (config mismatch or corrupt weights).`,
    );
  }
  let offset = 0;
  const variables = specs.map((spec) => {
    const length = spec.shape.reduce((a, b) => a * b, 1);
    // Copy each slice: some backends do not honour the byteOffset of a view.
    const values = packed.slice(offset, offset + length);
    offset += length;
    return makeVariable(tf, spec.name, tf.tensor(values, spec.shape, "float32"));
  });
  return modelFromVariables(config, variables);
}

function serializeWeights(
  tf: TfModule,
  config: TinyGptConfig,
  model: ModelVars,
): SerializedTinyGptWeights {
  const specs = paramSpecs(config);
  const flat = flattenModel(model).map((variable) => {
    const data = variable.dataSync();
    return encodeFloat32(
      data instanceof Float32Array ? data : Float32Array.from(data),
    );
  });
  return serializedWeightsFromFlat(config, flat, specs);
}

function createOptimizer(
  tf: TfModule,
  config: TinyGptConfig,
  model: ModelVars,
): OptimizerVars {
  const specs = paramSpecs(config);
  const params = flattenModel(model);
  return {
    m: params.map((param, index) =>
      makeVariable(tf, `m_${specs[index]!.name}`, tf.zerosLike(param), false),
    ),
    v: params.map((param, index) =>
      makeVariable(tf, `v_${specs[index]!.name}`, tf.zerosLike(param), false),
    ),
  };
}

function restoreOptimizer(
  tf: TfModule,
  config: TinyGptConfig,
  model: ModelVars,
  state: BrowserTrainingState["optimizer"],
): OptimizerVars {
  const specs = paramSpecs(config);
  const flatM = flattenSerializedWeights(config, state.m);
  const flatV = flattenSerializedWeights(config, state.v);
  return {
    m: specs.map((spec, index) => {
      const values = decodeFloat32(flatM[index]!);
      const tensor = tf.tensor(values, spec.shape, "float32");
      return makeVariable(tf, `m_${spec.name}`, tensor, false);
    }),
    v: specs.map((spec, index) => {
      const values = decodeFloat32(flatV[index]!);
      const tensor = tf.tensor(values, spec.shape, "float32");
      return makeVariable(tf, `v_${spec.name}`, tensor, false);
    }),
  };
}

function serializeOptimizer(
  tf: TfModule,
  config: TinyGptConfig,
  optimizer: OptimizerVars,
): BrowserTrainingState["optimizer"] {
  const specs = paramSpecs(config);
  const read = (variable: Variable): string => {
    const data = variable.dataSync();
    return encodeFloat32(
      data instanceof Float32Array ? data : Float32Array.from(data),
    );
  };
  return {
    m: serializedWeightsFromFlat(config, optimizer.m.map(read), specs),
    v: serializedWeightsFromFlat(config, optimizer.v.map(read), specs),
  };
}

function serializedWeightsFromFlat(
  config: TinyGptConfig,
  flat: readonly string[],
  specs = paramSpecs(config),
): SerializedTinyGptWeights {
  const byName = new Map(specs.map((spec, index) => [spec.name, flat[index]!]));
  return {
    tokenEmb: byName.get("tokenEmb")!,
    posEmb: byName.get("posEmb")!,
    gFinal: byName.get("gFinal")!,
    layers: Array.from({ length: config.layers }, (_, layer) => ({
      g1: byName.get(`layer_${layer}_g1`)!,
      wqkv: byName.get(`layer_${layer}_wqkv`)!,
      wo: byName.get(`layer_${layer}_wo`)!,
      g2: byName.get(`layer_${layer}_g2`)!,
      w1: byName.get(`layer_${layer}_w1`)!,
      w2: byName.get(`layer_${layer}_w2`)!,
    })),
  };
}

export function flattenSerializedWeights(
  config: TinyGptConfig,
  weights: SerializedTinyGptWeights,
): string[] {
  return [
    weights.tokenEmb,
    weights.posEmb,
    weights.gFinal,
    ...weights.layers.flatMap((layer) => [
      layer.g1,
      layer.wqkv,
      layer.wo,
      layer.g2,
      layer.w1,
      layer.w2,
    ]),
  ];
}

function disposeModel(model: ModelVars): void {
  for (const variable of flattenModel(model)) {
    variable.dispose();
  }
}

function disposeOptimizer(optimizer: OptimizerVars): void {
  for (const variable of [...optimizer.m, ...optimizer.v]) {
    variable.dispose();
  }
}

/* ------------------------------------------------------------------ */
/* Forward pass and loss                                              */
/* ------------------------------------------------------------------ */

function rmsNorm(
  tf: TfModule,
  input: Tensor,
  gain: Variable,
): Tensor {
  const meanSquare = tf.mean(tf.square(input), -1, true);
  return tf.mul(tf.mul(input, tf.rsqrt(tf.add(meanSquare, 1e-5))), gain);
}

function gelu(tf: TfModule, input: Tensor): Tensor {
  const inner = tf.mul(
    0.7978845608028654,
    tf.add(input, tf.mul(0.044715, tf.pow(input, 3))),
  );
  return tf.mul(tf.mul(0.5, input), tf.add(1, tf.tanh(inner)));
}

/**
 * Rank-3 x rank-2 matrix product.
 *
 * TensorFlow.js does not reduce the batch dimension in the gradient of a
 * rank-2 right operand for BatchMatMul. Flattening the token axis keeps every
 * matmul rank-2, which is both portable and correct on the CPU/WebGL/WebGPU
 * backends.
 */
function linear(tf: TfModule, input: Tensor, weight: Variable): Tensor {
  const batch = input.shape[0]!;
  const context = input.shape[1]!;
  const dim = input.shape[2]!;
  const outDim = weight.shape[1]!;
  const flat = tf.reshape(input, [batch * context, dim]);
  const projected = tf.matMul(flat, weight);
  return tf.reshape(projected, [batch, context, outDim]);
}

const causalMaskCache = new Map<number, Float32Array>();

/**
 * Builds the additive causal mask as a constant tensor.
 *
 * Keeping the mask as a leaf avoids sending comparison/where ops through the
 * gradient tape, which is both faster and more portable across backends.
 */
function causalMask(tf: TfModule, context: number): Tensor {
  let values = causalMaskCache.get(context);
  if (!values) {
    values = new Float32Array(context * context);
    for (let row = 0; row < context; row++) {
      for (let column = 0; column < context; column++) {
        values[row * context + column] = column <= row ? 0 : -1e9;
      }
    }
    causalMaskCache.set(context, values);
  }
  return tf.tensor2d(values, [context, context], "float32");
}

function forward(
  tf: TfModule,
  model: ModelVars,
  tokens: Tensor,
  config: TinyGptConfig,
  training: boolean,
  dropoutSeed: number,
  dropoutRate: number,
): Tensor {
  const batch = tokens.shape[0]!;
  const context = tokens.shape[1]!;
  const { dim, heads } = config;
  const headDim = dim / heads;
  if (!Number.isInteger(headDim)) {
    throw new Error("dim must be divisible by heads");
  }

  const positions = tf.range(0, context, 1, "int32");
  let hidden = tf.add(
    tf.gather(model.tokenEmb, tokens),
    tf.gather(model.posEmb, positions),
  );

  const negativeMask = causalMask(tf, context);

  for (const layer of model.layers) {
    const normalized = rmsNorm(tf, hidden, layer.g1);
    const qkv = linear(tf, normalized, layer.wqkv);
    const qkvHeads = tf.reshape(qkv, [batch, context, 3, heads, headDim]);
    const qkvParts = tf.unstack(qkvHeads, 2);
    const q = qkvParts[0];
    const k = qkvParts[1];
    const v = qkvParts[2];
    if (!q || !k || !v) {
      throw new Error("Could not split the attention projection.");
    }
    const query = tf.transpose(q, [0, 2, 1, 3]);
    const key = tf.transpose(k, [0, 2, 1, 3]);
    const value = tf.transpose(v, [0, 2, 1, 3]);
    const scores = tf.div(
      tf.matMul(query, key, false, true),
      Math.sqrt(headDim),
    );
    const probabilities = tf.softmax(tf.add(scores, negativeMask), -1);
    const attended = tf.matMul(probabilities, value);
    const merged = tf.reshape(
      tf.transpose(attended, [0, 2, 1, 3]),
      [batch, context, dim],
    );
    const attentionOut = linear(tf, merged, layer.wo);
    hidden = tf.add(
      hidden,
      training && dropoutRate > 0
        ? tf.dropout(attentionOut, dropoutRate, undefined, dropoutSeed)
        : attentionOut,
    );

    const normalized2 = rmsNorm(tf, hidden, layer.g2);
    const expanded = gelu(tf, linear(tf, normalized2, layer.w1));
    const projected = linear(tf, expanded, layer.w2);
    hidden = tf.add(
      hidden,
      training && dropoutRate > 0
        ? tf.dropout(projected, dropoutRate, undefined, dropoutSeed + 1)
        : projected,
    );
  }

  const final = rmsNorm(tf, hidden, model.gFinal);
  const flatFinal = tf.reshape(final, [batch * context, dim]);
  const logits = tf.matMul(flatFinal, model.tokenEmb, false, true);
  return tf.reshape(logits, [batch, context, config.vocabSize]);
}

function lossForBatch(
  tf: TfModule,
  model: ModelVars,
  input: Tensor,
  target: Tensor,
  config: TinyGptConfig,
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
    );
    const batch = input.shape[0]!;
    const context = input.shape[1]!;
    const flatLogits = tf.reshape(logits, [batch * context, config.vocabSize]);
    const flatTargets = tf.reshape(target, [batch * context]);
    const oneHot = tf.oneHot(
      flatTargets,
      config.vocabSize,
      1,
      0,
      "float32",
    );
    return tf.losses.softmaxCrossEntropy(oneHot, flatLogits) as Scalar;
  });
}

function clipGradients(
  tf: TfModule,
  grads: readonly Tensor[],
  maxNorm: number,
): Tensor[] {
  return tf.tidy(() => {
    const norm = tf.sqrt(tf.addN(grads.map((grad) => tf.sum(tf.square(grad)))));
    const scale = tf.minimum(tf.div(maxNorm, tf.add(norm, 1e-6)), 1);
    return grads.map((grad) => tf.mul(grad, scale));
  });
}

function applyAdam(
  tf: TfModule,
  params: readonly Variable[],
  grads: readonly Tensor[],
  optimizer: OptimizerVars,
  step: number,
  learningRate: number,
): void {
  const beta1 = 0.9;
  const beta2 = 0.999;
  const epsilon = 1e-8;
  const bias1 = 1 - Math.pow(beta1, step);
  const bias2 = 1 - Math.pow(beta2, step);

  for (let index = 0; index < params.length; index++) {
    const param = params[index]!;
    const grad = grads[index]!;
    const m = optimizer.m[index]!;
    const v = optimizer.v[index]!;
    tf.tidy(() => {
      const nextM = tf.add(tf.mul(beta1, m), tf.mul(1 - beta1, grad));
      const nextV = tf.add(
        tf.mul(beta2, v),
        tf.mul(1 - beta2, tf.square(grad)),
      );
      m.assign(nextM);
      v.assign(nextV);
      const mHat = tf.div(nextM, bias1);
      const vHat = tf.div(nextV, bias2);
      const update = tf.div(
        tf.mul(learningRate, mHat),
        tf.add(tf.sqrt(vHat), epsilon),
      );
      param.assign(tf.sub(param, update));
    });
  }
}

async function computeValidationLoss(
  tf: TfModule,
  model: ModelVars,
  config: TinyGptConfig,
  windows: WindowData,
  batchSize = 32,
): Promise<number> {
  if (windows.count === 0) {
    return Number.NaN;
  }
  let total = 0;
  let count = 0;
  for (let start = 0; start < windows.count; start += batchSize) {
    const rows = Math.min(batchSize, windows.count - start);
    const input = tf.tensor2d(
      windows.input.subarray(start * config.context, (start + rows) * config.context),
      [rows, config.context],
      "int32",
    );
    const target = tf.tensor2d(
      windows.target.subarray(start * config.context, (start + rows) * config.context),
      [rows, config.context],
      "int32",
    );
    const loss = lossForBatch(tf, model, input, target, config, false, 0, 0);
    try {
      const data = await loss.data();
      total += Number(data[0] ?? 0) * rows;
      count += rows;
    } finally {
      loss.dispose();
      input.dispose();
      target.dispose();
    }
  }
  return count > 0 ? total / count : Number.NaN;
}

/* ------------------------------------------------------------------ */
/* Checkpoint helpers                                                 */
/* ------------------------------------------------------------------ */

function makeCheckpoint(
  config: TinyGptConfig,
  vocab: readonly string[],
  weights: SerializedTinyGptWeights,
  meta: TokenTrainingModelMeta & {
    readonly steps: number;
    readonly batch: number;
    readonly loss: number;
    readonly corpusSize: number;
    readonly corpusLabel: string;
  },
): TinyGptCheckpoint {
  return {
    config,
    meta: {
      name: meta.name,
      source: meta.source,
      license: meta.license,
      training: {
        [meta.corpusLabel]: meta.corpusSize,
        steps: meta.steps,
        batch: meta.batch,
        context: config.context,
        dim: config.dim,
        layers: config.layers,
        heads: config.heads,
        mlp: config.mlp,
        loss: Number.isFinite(meta.loss) ? meta.loss : 0,
      },
    },
    vocab: [...vocab],
    weights,
    samples: [],
  };
}

function makeState(
  config: TinyGptConfig,
  vocab: readonly string[],
  kind: BrowserTrainingKind,
  step: number,
  steps: number,
  batch: number,
  lr: number,
  dropout: number,
  seed: number,
  bestStep: number,
  bestLoss: number,
  history: readonly TrainingPoint[],
  weights: SerializedTinyGptWeights,
  optimizer: BrowserTrainingState["optimizer"],
  bestWeights: SerializedTinyGptWeights | null,
  corpusHash: string,
  elapsedMs: number,
): BrowserTrainingState {
  return {
    format: "only-gates-browser-train",
    version: 1,
    kind,
    config,
    vocab: [...vocab],
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
  };
}

/* ------------------------------------------------------------------ */
/* Training                                                           */
/* ------------------------------------------------------------------ */

export async function trainTokenGpt(
  options: TokenTrainingOptions,
): Promise<BrowserTrainingResult> {
  const { data } = options;
  const corpusHash = data.corpusHash;
  const initial = options.initialState ?? null;
  const kind = options.kind ?? "alice";
  if (initial && initial.corpusHash !== corpusHash) {
    throw new Error(
      "The saved run was trained on a different corpus. Clear it before starting again.",
    );
  }
  if (initial && (initial.kind ?? "alice") !== kind) {
    throw new Error("The saved run belongs to a different browser trainer.");
  }

  const config: TinyGptConfig = initial
    ? { ...initial.config }
    : {
        ...(options.defaultConfig ?? DEFAULT_BROWSER_TRAIN_CONFIG),
        ...options.config,
        vocabSize: data.vocab.length,
      };
  if (config.vocabSize !== data.vocab.length) {
    throw new Error("The saved vocabulary does not match the saved model config.");
  }
  if (config.dim % config.heads !== 0) {
    throw new Error("dim must be divisible by heads.");
  }
  if (data.trainingTokens.length < config.context + 2) {
    throw new Error("The training corpus is too short for this context size.");
  }

  const batch = initial?.batch ?? options.batch ?? 8;
  const learningRate = initial?.lr ?? options.lr ?? 0.003;
  const dropoutRate = initial?.dropout ?? options.dropout ?? 0;
  const seed = initial?.seed ?? options.seed ?? 1337;
  const totalSteps = Math.max(
    options.steps ?? initial?.steps ?? 1800,
    initial?.step ?? 0,
  );
  const saveEvery = Math.max(1, options.saveEvery ?? 100);
  const yieldEvery = Math.max(1, options.yieldEvery ?? 5);
  const maxMinutes = options.maxMinutes ?? 0;
  const startedAt = performance.now();
  const baseElapsed = initial?.elapsedMs ?? 0;
  const elapsed = (): number => baseElapsed + (performance.now() - startedAt);

  const { tf, backend } = await selectTensorFlowBackend(
    options.backend ?? "auto",
  );

  let model: ModelVars | null = null;
  let optimizer: OptimizerVars | null = null;
  try {
    model = initial
      ? restoreModel(tf, config, initial.weights)
      : options.initialWeights
        ? restoreModelFromPacked(tf, config, options.initialWeights)
        : createModel(tf, config, seed);
    optimizer = initial
      ? restoreOptimizer(tf, config, model, initial.optimizer)
      : createOptimizer(tf, config, model);

    const params = flattenModel(model);
    const history: TrainingPoint[] = initial ? [...initial.history] : [];
    const validationWindows = buildWindows(
      data.validationTokens,
      config.context,
    );
    let smooth: number | null =
      history.length > 0 ? history[history.length - 1]!.smoothedLoss : null;
    let bestLoss = initial?.bestLoss ?? Number.POSITIVE_INFINITY;
    let bestStep = initial?.bestStep ?? 0;
    let bestWeights = initial?.bestWeights ?? null;
    let stopped = false;
    let lastStep = initial?.step ?? 0;

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
      const batchData = sampleBatch(
        data.trainingTokens,
        config.context,
        batch,
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
      // Cosine decay from the configured rate to 10% of it, matching
      // scripts/train-tiny-gpt.mjs. The scheduled rate is applied to the
      // optimizer, not just reported.
      const lr =
        learningRate *
        (0.1 + 0.9 * 0.5 * (1 + Math.cos((Math.PI * step) / totalSteps)));

      let lossValue = 0;
      let lossTensor: Tensor | null = null;
      let rawGrads: Tensor[] | null = null;
      try {
        const result = tf.variableGrads(
          () =>
            lossForBatch(
              tf,
              model!,
              input,
              target,
              config,
              true,
              (seed ^ step) >>> 0,
              dropoutRate,
            ),
          params,
        );
        lossTensor = result.value;
        const data = await lossTensor.data();
        lossValue = Number(data[0] ?? 0);
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
        applyAdam(tf, params, clipped, optimizer, step, lr);
        tf.dispose(clipped);
      } finally {
        lossTensor?.dispose();
        if (rawGrads) {
          tf.dispose(rawGrads);
        }
        input.dispose();
        target.dispose();
      }

      smooth = smooth === null ? lossValue : smooth * 0.95 + lossValue * 0.05;
      let valLoss: number | undefined;
      if (step % 100 === 0 || step === totalSteps) {
        const value = await computeValidationLoss(
          tf,
          model,
          config,
          validationWindows,
        );
        if (Number.isFinite(value)) {
          valLoss = value;
          if (value < bestLoss) {
            bestLoss = value;
            bestStep = step;
            bestWeights = serializeWeights(tf, config, model);
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
          (step - (initial?.step ?? 0)) / Math.max(0.001, (performance.now() - startedAt) / 1000),
      });

      if (step % saveEvery === 0 && step < totalSteps) {
        options.onCheckpoint?.(
          makeState(
            config,
            data.vocab,
            kind,
            step,
            totalSteps,
            batch,
            learningRate,
            dropoutRate,
            seed,
            bestStep,
            bestLoss,
            history,
            serializeWeights(tf, config, model),
            serializeOptimizer(tf, config, optimizer),
            bestWeights,
            corpusHash,
            elapsed(),
          ),
        );
      }

      if (step % yieldEvery === 0) {
        await yieldToBrowser();
      }
    }

    const currentWeights = serializeWeights(tf, config, model);
    const finalWeights = options.exportFinal
      ? currentWeights
      : bestWeights ?? currentWeights;
    const state = makeState(
      config,
      data.vocab,
      kind,
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
      serializeOptimizer(tf, config, optimizer),
      bestWeights,
      corpusHash,
      elapsed(),
    );
    options.onCheckpoint?.(state);
    const finalLoss = Number.isFinite(bestLoss)
      ? bestLoss
      : history.length > 0
        ? history[history.length - 1]!.loss
        : Number.NaN;
    return {
      checkpoint: makeCheckpoint(config, data.vocab, finalWeights, {
        ...options.modelMeta,
        steps: lastStep,
        batch,
        loss: finalLoss,
        corpusSize: data.corpusSize ?? data.trainingTokens.length,
        corpusLabel: data.corpusLabel ?? "corpusTokens",
      }),
      state,
      backend,
      stopped,
      elapsedMs: elapsed(),
      history,
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

/** Trains the word-level Alice model on the bundled public-domain corpus. */
export async function trainTinyGpt(
  options: BrowserTrainingOptions,
): Promise<BrowserTrainingResult> {
  const corpusData = buildWordCorpus(
    options.corpus,
    options.vocabLimit ?? options.initialState?.config.vocabSize ?? 512,
    options.initialState?.vocab,
  );
  return trainTokenGpt({
    data: {
      vocab: corpusData.vocab,
      trainingTokens: corpusData.trainingTokens,
      validationTokens: corpusData.validationTokens,
      corpusHash: hashString(options.corpus),
      corpusSize: corpusData.words.length,
      corpusLabel: "corpusWords",
    },
    config: options.config,
    defaultConfig: DEFAULT_BROWSER_TRAIN_CONFIG,
    steps: options.steps,
    batch: options.batch,
    lr: options.lr,
    seed: options.seed,
    dropout: options.dropout,
    maxMinutes: options.maxMinutes,
    backend: options.backend,
    initialState: options.initialState,
    onProgress: options.onProgress,
    onCheckpoint: options.onCheckpoint,
    shouldCancel: options.shouldCancel,
    yieldEvery: options.yieldEvery,
    saveEvery: options.saveEvery,
    kind: "alice",
    modelMeta: {
      name: "tiny-gpt-alice-browser",
      source:
        "Lewis Carroll, Alice's Adventures in Wonderland (1865), Project Gutenberg #11. Public domain.",
      license:
        "Public domain text; model weights trained in the browser by this repository.",
    },
  });
}
