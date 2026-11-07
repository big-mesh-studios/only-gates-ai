/*
 * Tiny GPT inference with two paths:
 *
 *   float  - the checkpoint's float32 weights.
 *   int8   - weights stored as int8 with one scale per output channel, and
 *            activations quantized to int8 per token. Matrix products are
 *            integer dot products accumulated in int32, then dequantized.
 *
 * RMSNorm, GELU and softmax stay in float32. That is a deliberate compromise:
 * keeping those in integer arithmetic changes the model enough to be a
 * different demo, and they are a small part of the compute. The UI says so.
 */

export interface TinyGptConfig {
  readonly vocabSize: number;
  readonly context: number;
  readonly dim: number;
  readonly layers: number;
  readonly heads: number;
  readonly mlp: number;
}

export interface TinyGptCheckpoint {
  readonly config: TinyGptConfig;
  readonly meta: {
    readonly name: string;
    readonly source: string;
    readonly license: string;
    readonly training: Readonly<Record<string, number>>;
  };
  readonly vocab: readonly string[];
  readonly weights: {
    readonly tokenEmb: string;
    readonly posEmb: string;
    readonly layers: readonly {
      readonly g1: string;
      readonly wqkv: string;
      readonly wo: string;
      readonly g2: string;
      readonly w1: string;
      readonly w2: string;
    }[];
    readonly gFinal: string;
  };
  readonly samples: readonly { readonly prompt: string; readonly text: string }[];
}

export type TinyGptMode = "float" | "int8";

export interface FloatLayerWeights {
  readonly g1: Float32Array;
  readonly wqkv: Float32Array;
  readonly wo: Float32Array;
  readonly g2: Float32Array;
  readonly w1: Float32Array;
  readonly w2: Float32Array;
}

export interface FloatWeights {
  readonly tokenEmb: Float32Array;
  readonly posEmb: Float32Array;
  readonly layers: readonly FloatLayerWeights[];
  readonly gFinal: Float32Array;
}

/** Row-major matrix of int8 values with one scale per output column. */
export interface QuantizedMatrix {
  readonly rows: number;
  readonly cols: number;
  readonly values: Int8Array;
  /** Dequantization scale for each output column. */
  readonly scales: Float32Array;
}

/** Row-major int8 table with one scale per row (embedding lookup). */
export interface QuantizedRows {
  readonly rows: number;
  readonly cols: number;
  readonly values: Int8Array;
  readonly scales: Float32Array;
}

export interface QuantizedLayerWeights {
  readonly wqkv: QuantizedMatrix;
  readonly wo: QuantizedMatrix;
  readonly w1: QuantizedMatrix;
  readonly w2: QuantizedMatrix;
}

export interface TinyGptStats {
  readonly parameters: number;
  readonly buffers: number;
  readonly floatBytes: number;
  readonly int8Bytes: number;
  readonly scaleBytes: number;
  readonly compression: number;
  /** Root-mean-square weight error introduced by int8 quantization. */
  readonly weightRmsError: number;
  readonly maxWeightError: number;
  readonly maxWeightMagnitude: number;
}

export interface TinyGptModel {
  readonly config: TinyGptConfig;
  readonly meta: TinyGptCheckpoint["meta"];
  readonly vocab: readonly string[];
  readonly index: ReadonlyMap<string, number>;
  readonly float: FloatWeights;
  readonly int8: {
    /** Embedding table, one int8 row per token. */
    readonly tokenEmb: QuantizedRows;
    readonly posEmb: QuantizedRows;
    /** Output head: transposed token embedding, one int8 column per token. */
    readonly head: QuantizedMatrix;
    readonly layers: readonly QuantizedLayerWeights[];
  };
  readonly stats: TinyGptStats;
}

export interface TinyGptCandidate {
  readonly id: number;
  readonly token: string;
  readonly probability: number;
}

export interface TinyGptForward {
  readonly logits: Float32Array;
  readonly mode: TinyGptMode;
}

export interface TinyGptGeneration {
  readonly mode: TinyGptMode;
  readonly promptTokens: readonly string[];
  readonly tokenIds: readonly number[];
  readonly newTokens: readonly string[];
  readonly text: string;
  readonly steps: number;
  readonly stopped: boolean;
  readonly elapsedMs: number;
}

export interface TinyGptComparison {
  readonly promptTokens: readonly string[];
  readonly floatTop: readonly TinyGptCandidate[];
  readonly int8Top: readonly TinyGptCandidate[];
  readonly topOneMatches: boolean;
  readonly overlapAtFive: number;
  readonly maxProbabilityDelta: number;
  /** Mean absolute difference over the full next-token distribution. */
  readonly meanProbabilityDelta: number;
}

export interface TinyGptGenerationOptions {
  readonly maxTokens?: number;
  readonly temperature?: number;
  readonly topK?: number;
  readonly seed?: number;
}

const UNK = 0;
const BOS = 1;
const EOS = 2;
const EPS = 1e-5;

/* ------------------------------------------------------------------ */
/* Checkpoint decoding                                                 */
/* ------------------------------------------------------------------ */

function decodeFloat32(base64: string): Float32Array {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return new Float32Array(bytes.buffer, 0, bytes.length / 4);
}

export function tokenizePrompt(text: string): string[] {
  return text.toLowerCase().match(/[a-z0-9']+/g) ?? [];
}

/* ------------------------------------------------------------------ */
/* Quantization                                                        */
/* ------------------------------------------------------------------ */

export function quantizeByColumns(
  values: Float32Array,
  rows: number,
  cols: number,
): QuantizedMatrix {
  if (values.length !== rows * cols) {
    throw new Error("matrix size mismatch");
  }
  const quantized = new Int8Array(values.length);
  const scales = new Float32Array(cols);
  for (let col = 0; col < cols; col++) {
    let maxAbs = 0;
    for (let row = 0; row < rows; row++) {
      const magnitude = Math.abs(values[row * cols + col]!);
      if (magnitude > maxAbs) maxAbs = magnitude;
    }
    const scale = maxAbs > 0 ? maxAbs / 127 : 1;
    scales[col] = scale;
    const inverse = maxAbs > 0 ? 127 / maxAbs : 0;
    for (let row = 0; row < rows; row++) {
      const index = row * cols + col;
      quantized[index] = Math.max(
        -127,
        Math.min(127, Math.round(values[index]! * inverse)),
      );
    }
  }
  return { rows, cols, values: quantized, scales };
}

export function quantizeByRows(
  values: Float32Array,
  rows: number,
  cols: number,
): QuantizedRows {
  if (values.length !== rows * cols) {
    throw new Error("matrix size mismatch");
  }
  const quantized = new Int8Array(values.length);
  const scales = new Float32Array(rows);
  for (let row = 0; row < rows; row++) {
    const offset = row * cols;
    let maxAbs = 0;
    for (let col = 0; col < cols; col++) {
      const magnitude = Math.abs(values[offset + col]!);
      if (magnitude > maxAbs) maxAbs = magnitude;
    }
    const scale = maxAbs > 0 ? maxAbs / 127 : 1;
    scales[row] = scale;
    const inverse = maxAbs > 0 ? 127 / maxAbs : 0;
    for (let col = 0; col < cols; col++) {
      quantized[offset + col] = Math.max(
        -127,
        Math.min(127, Math.round(values[offset + col]! * inverse)),
      );
    }
  }
  return { rows, cols, values: quantized, scales };
}

function dequantizeRow(
  row: QuantizedRows,
  index: number,
  out: Float32Array,
  offset: number,
): void {
  const src = index * row.cols;
  const scale = row.scales[index];
  for (let i = 0; i < row.cols; i++) {
    out[offset + i] = row.values[src + i]! * scale!;
  }
}

function transposedMatrix(
  values: Float32Array,
  rows: number,
  cols: number,
): QuantizedMatrix {
  // values is rows x cols; return the quantized cols x rows transpose.
  const transposed = new Float32Array(cols * rows);
  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols; col++) {
      transposed[col * rows + row] = values[row * cols + col]!;
    }
  }
  return quantizeByColumns(transposed, cols, rows);
}

/* ------------------------------------------------------------------ */
/* Loading                                                             */
/* ------------------------------------------------------------------ */

export function loadTinyGpt(checkpoint: TinyGptCheckpoint): TinyGptModel {
  const { config } = checkpoint;
  const { vocabSize, context, dim, layers, mlp } = config;

  const tokenEmb = decodeFloat32(checkpoint.weights.tokenEmb);
  const posEmb = decodeFloat32(checkpoint.weights.posEmb);
  const gFinal = decodeFloat32(checkpoint.weights.gFinal);
  const floatLayers: FloatLayerWeights[] = checkpoint.weights.layers.map(
    (layer) => ({
      g1: decodeFloat32(layer.g1),
      wqkv: decodeFloat32(layer.wqkv),
      wo: decodeFloat32(layer.wo),
      g2: decodeFloat32(layer.g2),
      w1: decodeFloat32(layer.w1),
      w2: decodeFloat32(layer.w2),
    }),
  );

  const expected = {
    tokenEmb: vocabSize * dim,
    posEmb: context * dim,
    gFinal: dim,
    g1: dim,
    g2: dim,
    wqkv: dim * dim * 3,
    wo: dim * dim,
    w1: dim * mlp,
    w2: mlp * dim,
  };
  if (tokenEmb.length !== expected.tokenEmb) {
    throw new Error("checkpoint token embedding has the wrong size");
  }
  if (floatLayers.length !== layers) {
    throw new Error("checkpoint layer count does not match its config");
  }

  const model: TinyGptModel = {
    config,
    meta: checkpoint.meta,
    vocab: checkpoint.vocab,
    index: new Map(checkpoint.vocab.map((word, id) => [word, id])),
    float: {
      tokenEmb,
      posEmb,
      layers: floatLayers,
      gFinal,
    },
    int8: {
      tokenEmb: quantizeByRows(tokenEmb, vocabSize, dim),
      posEmb: quantizeByRows(posEmb, context, dim),
      head: transposedMatrix(tokenEmb, vocabSize, dim),
      layers: floatLayers.map((layer) => ({
        wqkv: quantizeByColumns(layer.wqkv, dim, dim * 3),
        wo: quantizeByColumns(layer.wo, dim, dim),
        w1: quantizeByColumns(layer.w1, dim, mlp),
        w2: quantizeByColumns(layer.w2, mlp, dim),
      })),
    },
    stats: {
      parameters: 0,
      buffers: 0,
      floatBytes: 0,
      int8Bytes: 0,
      scaleBytes: 0,
      compression: 1,
      weightRmsError: 0,
      maxWeightError: 0,
      maxWeightMagnitude: 0,
    },
  };

  const stats = measureQuantization(model);
  return { ...model, stats };
}

function measureQuantization(model: TinyGptModel): TinyGptStats {
  const uniqueMatrices: {
    float: Float32Array;
    rows: number;
    cols: number;
    quantized: Int8Array;
    scales: Float32Array;
    rowScales: boolean;
  }[] = [
    {
      float: model.float.tokenEmb,
      rows: model.config.vocabSize,
      cols: model.config.dim,
      quantized: model.int8.tokenEmb.values,
      scales: model.int8.tokenEmb.scales,
      rowScales: true,
    },
    {
      float: model.float.posEmb,
      rows: model.config.context,
      cols: model.config.dim,
      quantized: model.int8.posEmb.values,
      scales: model.int8.posEmb.scales,
      rowScales: true,
    },
  ];

  for (let layer = 0; layer < model.config.layers; layer++) {
    const params = model.float.layers[layer]!;
    const quantized = model.int8.layers[layer]!;
    uniqueMatrices.push(
      {
        float: params.wqkv,
        rows: model.config.dim,
        cols: model.config.dim * 3,
        quantized: quantized.wqkv.values,
        scales: quantized.wqkv.scales,
        rowScales: false,
      },
      {
        float: params.wo,
        rows: model.config.dim,
        cols: model.config.dim,
        quantized: quantized.wo.values,
        scales: quantized.wo.scales,
        rowScales: false,
      },
      {
        float: params.w1,
        rows: model.config.dim,
        cols: model.config.mlp,
        quantized: quantized.w1.values,
        scales: quantized.w1.scales,
        rowScales: false,
      },
      {
        float: params.w2,
        rows: model.config.mlp,
        cols: model.config.dim,
        quantized: quantized.w2.values,
        scales: quantized.w2.scales,
        rowScales: false,
      },
    );
  }

  let parameters = 0;
  let int8Bytes = 0;
  let scaleBytes = 0;
  let squaredError = 0;
  let maxError = 0;
  let maxMagnitude = 0;

  for (const entry of uniqueMatrices) {
    const { rows, cols, quantized, scales, rowScales } = entry;
    parameters += rows * cols;
    int8Bytes += quantized.length;
    scaleBytes += scales.length * 4;

    for (let row = 0; row < rows; row++) {
      const scale = rowScales ? scales[row]! : 0;
      for (let col = 0; col < cols; col++) {
        const index = row * cols + col;
        const original = entry.float[index]!;
        const restored = quantized[index]! * (rowScales ? scale : scales[col]!);
        const error = Math.abs(restored - original);
        squaredError += error * error;
        if (error > maxError) maxError = error;
        const magnitude = Math.abs(original);
        if (magnitude > maxMagnitude) maxMagnitude = magnitude;
      }
    }
  }

  // RMSNorm gains stay in float32. They are parameters, but not int8 data.
  const gains = model.config.layers * model.config.dim * 2 + model.config.dim;
  parameters += gains;
  int8Bytes += gains * 4;

  const floatBytes = parameters * 4;
  return {
    parameters,
    buffers: uniqueMatrices.length,
    floatBytes,
    int8Bytes,
    scaleBytes,
    compression: floatBytes / Math.max(1, int8Bytes + scaleBytes),
    weightRmsError: Math.sqrt(squaredError / Math.max(1, parameters - gains)),
    maxWeightError: maxError,
    maxWeightMagnitude: maxMagnitude,
  };
}

/* ------------------------------------------------------------------ */
/* Float kernels                                                       */
/* ------------------------------------------------------------------ */

function rmsnorm(
  x: Float32Array,
  rows: number,
  cols: number,
  gain: Float32Array,
  out: Float32Array,
): void {
  for (let row = 0; row < rows; row++) {
    const offset = row * cols;
    let sumSquares = 0;
    for (let i = 0; i < cols; i++) {
      const value = x[offset + i]!;
      sumSquares += value * value;
    }
    const inverse = 1 / Math.sqrt(sumSquares / cols + EPS);
    for (let i = 0; i < cols; i++) {
      out[offset + i] = x[offset + i]! * inverse * gain[i]!;
    }
  }
}

function gelu(x: number): number {
  const inner = 0.7978845608028654 * (x + 0.044715 * x * x * x);
  return 0.5 * x * (1 + Math.tanh(inner));
}

function softmaxRow(
  values: Float32Array,
  offset: number,
  length: number,
  out: Float32Array,
  outOffset: number,
): void {
  let max = -Infinity;
  for (let i = 0; i < length; i++) {
    const value = values[offset + i]!;
    if (value > max) max = value;
  }
  let sum = 0;
  for (let i = 0; i < length; i++) {
    const value = Math.exp(values[offset + i]! - max);
    out[outOffset + i] = value;
    sum += value;
  }
  const inverse = sum > 0 ? 1 / sum : 0;
  for (let i = 0; i < length; i++) {
    out[outOffset + i] = out[outOffset + i]! * inverse;
  }
}

function matmulFloat(
  x: Float32Array,
  rows: number,
  cols: number,
  w: Float32Array,
  outCols: number,
  out: Float32Array,
): void {
  for (let row = 0; row < rows; row++) {
    const xOffset = row * cols;
    const outOffset = row * outCols;
    for (let j = 0; j < outCols; j++) {
      let sum = 0;
      for (let k = 0; k < cols; k++) {
        sum += x[xOffset + k]! * w[k * outCols + j]!;
      }
      out[outOffset + j] = sum;
    }
  }
}

/* ------------------------------------------------------------------ */
/* Integer kernels                                                     */
/* ------------------------------------------------------------------ */

function matmulInt8(
  x: Float32Array,
  rows: number,
  cols: number,
  weights: QuantizedMatrix,
  out: Float32Array,
): void {
  const outCols = weights.cols;
  const quantized = new Int8Array(cols);
  const accumulators = new Int32Array(outCols);

  for (let row = 0; row < rows; row++) {
    const xOffset = row * cols;
    let maxAbs = 0;
    for (let k = 0; k < cols; k++) {
      const magnitude = Math.abs(x[xOffset + k]!);
      if (magnitude > maxAbs) maxAbs = magnitude;
    }
    const inverse = maxAbs > 0 ? 127 / maxAbs : 0;
    const activationScale = maxAbs > 0 ? maxAbs / 127 : 1;
    for (let k = 0; k < cols; k++) {
      quantized[k] = Math.max(
        -127,
        Math.min(127, Math.round(x[xOffset + k]! * inverse)),
      );
    }

    accumulators.fill(0);
    for (let k = 0; k < cols; k++) {
      const activation = quantized[k]!;
      if (activation === 0) continue;
      const weightOffset = k * outCols;
      for (let j = 0; j < outCols; j++) {
        accumulators[j] = accumulators[j]! + activation * weights.values[weightOffset + j]!;
      }
    }

    const outOffset = row * outCols;
    for (let j = 0; j < outCols; j++) {
      out[outOffset + j] =
        accumulators[j]! * activationScale * weights.scales[j]!;
    }
  }
}

function quantizeVector(
  source: Float32Array,
  offset: number,
  length: number,
  out: Int8Array,
  outOffset: number,
): number {
  let maxAbs = 0;
  for (let i = 0; i < length; i++) {
    const magnitude = Math.abs(source[offset + i]!);
    if (magnitude > maxAbs) maxAbs = magnitude;
  }
  const inverse = maxAbs > 0 ? 127 / maxAbs : 0;
  for (let i = 0; i < length; i++) {
    out[outOffset + i] = Math.max(
      -127,
      Math.min(127, Math.round(source[offset + i]! * inverse)),
    );
  }
  return maxAbs > 0 ? maxAbs / 127 : 1;
}

function dotInt8(
  a: Int8Array,
  aOffset: number,
  b: Int8Array,
  bOffset: number,
  length: number,
): number {
  let sum = 0;
  for (let i = 0; i < length; i++) {
    sum += a[aOffset + i]! * b[bOffset + i]!;
  }
  return sum;
}

function dotUint8Int8(
  a: Uint8Array,
  b: Int8Array,
  bOffset: number,
  length: number,
): number {
  let sum = 0;
  for (let i = 0; i < length; i++) {
    sum += a[i]! * b[bOffset + i]!;
  }
  return sum;
}

/* ------------------------------------------------------------------ */
/* Forward pass                                                        */
/* ------------------------------------------------------------------ */

function embeddingInput(
  model: TinyGptModel,
  tokenIds: readonly number[],
  mode: TinyGptMode,
): Float32Array {
  const { context, dim } = model.config;
  const out = new Float32Array(context * dim);
  for (let t = 0; t < context; t++) {
    const token = tokenIds[t]!;
    const dst = t * dim;
    if (mode === "float") {
      const tokenOffset = token * dim;
      const positionOffset = t * dim;
      for (let d = 0; d < dim; d++) {
        out[dst + d] =
          model.float.tokenEmb[tokenOffset + d]! +
          model.float.posEmb[positionOffset + d]!;
      }
    } else {
      const tokenOffset = token * dim;
      const tokenScale = model.int8.tokenEmb.scales[token]!;
      const positionScale = model.int8.posEmb.scales[t]!;
      const positionOffset = t * dim;
      for (let d = 0; d < dim; d++) {
        out[dst + d] =
          model.int8.tokenEmb.values[tokenOffset + d]! * tokenScale +
          model.int8.posEmb.values[positionOffset + d]! * positionScale;
      }
    }
  }
  return out;
}

function attentionInt8(
  qkv: Float32Array,
  output: Float32Array,
  config: TinyGptConfig,
): void {
  const { context, dim, heads } = config;
  const headDim = dim / heads;
  const scale = 1 / Math.sqrt(headDim);
  const q = new Int8Array(headDim);
  const keys = new Int8Array(context * headDim);
  const keyScales = new Float32Array(context);
  const values = new Int8Array(context * headDim);
  const valueScales = new Float32Array(context);
  const scores = new Float32Array(context);
  const probabilities = new Float32Array(context);
  const quantizedProbabilities = new Uint8Array(context);

  for (let h = 0; h < heads; h++) {
    const headOffset = h * headDim;
    for (let j = 0; j < context; j++) {
      const keyBase = j * dim * 3 + dim + headOffset;
      const valueBase = j * dim * 3 + dim * 2 + headOffset;
      keyScales[j] = quantizeVector(
        qkv,
        keyBase,
        headDim,
        keys,
        j * headDim,
      );
      valueScales[j] = quantizeVector(
        qkv,
        valueBase,
        headDim,
        values,
        j * headDim,
      );
    }

    for (let t = 0; t < context; t++) {
      const qBase = t * dim * 3 + headOffset;
      const qScale = quantizeVector(qkv, qBase, headDim, q, 0);

      for (let j = 0; j <= t; j++) {
        scores[j] =
          dotInt8(q, 0, keys, j * headDim, headDim) *
          qScale *
          keyScales[j]! *
          scale;
      }
      for (let j = t + 1; j < context; j++) scores[j] = -Infinity;
      softmaxRow(scores, 0, context, probabilities, 0);

      for (let j = 0; j <= t; j++) {
        quantizedProbabilities[j] = Math.max(
          0,
          Math.min(255, Math.round(probabilities[j]! * 255)),
        );
      }

      const outBase = t * dim + headOffset;
      for (let d = 0; d < headDim; d++) {
        let sum = 0;
        for (let j = 0; j <= t; j++) {
          sum +=
            dotUint8Int8(
              quantizedProbabilities,
              values,
              j * headDim,
              headDim,
            ) *
            (1 / 255) *
            valueScales[j]!;
        }
        output[outBase + d] = sum;
      }
    }
  }
}

export function forwardTinyGpt(
  model: TinyGptModel,
  tokenIds: readonly number[],
  mode: TinyGptMode,
): TinyGptForward {
  const { context, dim, layers, heads, mlp, vocabSize } = model.config;
  const headDim = dim / heads;
  if (tokenIds.length !== context) {
    throw new Error(`expected ${context} tokens, got ${tokenIds.length}`);
  }

  let x = embeddingInput(model, tokenIds, mode);
  const attentionScale = 1 / Math.sqrt(headDim);

  for (let layer = 0; layer < layers; layer++) {
    const params = model.float.layers[layer]!;
    const quantized = model.int8.layers[layer]!;
    const h1 = new Float32Array(context * dim);
    rmsnorm(x, context, dim, params.g1, h1);

    const qkv = new Float32Array(context * dim * 3);
    if (mode === "float") {
      matmulFloat(h1, context, dim, params.wqkv, dim * 3, qkv);
    } else {
      matmulInt8(h1, context, dim, quantized.wqkv, qkv);
    }

    const attention = new Float32Array(context * context * heads);
    const attentionOut = new Float32Array(context * dim);

    if (mode === "float") {
      for (let h = 0; h < heads; h++) {
        const headOffset = h * headDim;
        for (let t = 0; t < context; t++) {
          const row = (h * context + t) * context;
          for (let j = 0; j <= t; j++) {
            let sum = 0;
            const qBase = t * dim * 3 + headOffset;
            const kBase = j * dim * 3 + dim + headOffset;
            for (let d = 0; d < headDim; d++) {
              sum += qkv[qBase + d]! * qkv[kBase + d]!;
            }
            attention[row + j] = sum * attentionScale;
          }
          for (let j = t + 1; j < context; j++) attention[row + j] = -Infinity;
          const rowScores = attention.subarray(row, row + context);
          const rowProbs = new Float32Array(context);
          softmaxRow(rowScores, 0, context, rowProbs, 0);
          for (let j = 0; j < context; j++) attention[row + j] = rowProbs[j]!;

          const outBase = t * dim + headOffset;
          for (let d = 0; d < headDim; d++) {
            let sum = 0;
            for (let j = 0; j <= t; j++) {
              const vBase = j * dim * 3 + dim * 2 + headOffset;
              sum += attention[row + j]! * qkv[vBase + d]!;
            }
            attentionOut[outBase + d] = sum;
          }
        }
      }
    } else {
      attentionInt8(qkv, attentionOut, model.config);
    }

    const y = new Float32Array(context * dim);
    if (mode === "float") {
      matmulFloat(attentionOut, context, dim, params.wo, dim, y);
    } else {
      matmulInt8(attentionOut, context, dim, quantized.wo, y);
    }

    const x1 = new Float32Array(context * dim);
    for (let i = 0; i < x1.length; i++) x1[i] = x[i]! + y[i]!;

    const h2 = new Float32Array(context * dim);
    rmsnorm(x1, context, dim, params.g2, h2);

    const z = new Float32Array(context * mlp);
    if (mode === "float") {
      matmulFloat(h2, context, dim, params.w1, mlp, z);
    } else {
      matmulInt8(h2, context, dim, quantized.w1, z);
    }

    const gz = new Float32Array(context * mlp);
    for (let i = 0; i < gz.length; i++) gz[i] = gelu(z[i]!);

    const m = new Float32Array(context * dim);
    if (mode === "float") {
      matmulFloat(gz, context, mlp, params.w2, dim, m);
    } else {
      matmulInt8(gz, context, mlp, quantized.w2, m);
    }

    const x2 = new Float32Array(context * dim);
    for (let i = 0; i < x2.length; i++) x2[i] = x1[i]! + m[i]!;
    x = x2;
  }

  const hf = new Float32Array(context * dim);
  rmsnorm(x, context, dim, model.float.gFinal, hf);

  const logits = new Float32Array(context * vocabSize);
  if (mode === "float") {
    // Tied head: logits = hf @ tokenEmb^T.
    for (let t = 0; t < context; t++) {
      const hOffset = t * dim;
      const outOffset = t * vocabSize;
      for (let token = 0; token < vocabSize; token++) {
        let sum = 0;
        const eOffset = token * dim;
        for (let d = 0; d < dim; d++) {
          sum += hf[hOffset + d]! * model.float.tokenEmb[eOffset + d]!;
        }
        logits[outOffset + token] = sum;
      }
    }
  } else {
    matmulInt8(hf, context, dim, model.int8.head, logits);
  }

  return { logits, mode };
}

/* ------------------------------------------------------------------ */
/* Distribution and generation                                         */
/* ------------------------------------------------------------------ */

function windowFor(model: TinyGptModel, ids: readonly number[]): number[] {
  const window = ids.slice(-model.config.context);
  while (window.length < model.config.context) window.unshift(BOS);
  return window;
}

function nextTokenDistribution(
  model: TinyGptModel,
  logits: Float32Array,
  temperature: number,
): Float32Array {
  const { vocabSize, context } = model.config;
  const offset = (context - 1) * vocabSize;
  const probabilities = new Float32Array(vocabSize);
  const scaled = new Float32Array(vocabSize);
  const safeTemperature = Math.max(0.01, temperature);
  for (let i = 0; i < vocabSize; i++) {
    scaled[i] = logits[offset + i]! / safeTemperature;
  }
  softmaxRow(scaled, 0, vocabSize, probabilities, 0);
  return probabilities;
}

export function nextTokens(
  model: TinyGptModel,
  prompt: string,
  mode: TinyGptMode,
  limit = 6,
): TinyGptCandidate[] {
  const ids = [BOS, ...promptTokensToIds(model, prompt)];
  const window = windowFor(model, ids);
  const { logits } = forwardTinyGpt(model, window, mode);
  const probabilities = nextTokenDistribution(model, logits, 1);
  const candidates: TinyGptCandidate[] = [];
  for (let id = 0; id < model.config.vocabSize; id++) {
    candidates.push({
      id,
      token: model.vocab[id] ?? "<unk>",
      probability: probabilities[id]!,
    });
  }
  candidates.sort((a, b) => b.probability - a.probability);
  return candidates.slice(0, limit);
}

export function promptTokensToIds(
  model: TinyGptModel,
  prompt: string,
): number[] {
  return tokenizePrompt(prompt).map((word) => model.index.get(word) ?? UNK);
}

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

function sampleToken(
  probabilities: Float32Array,
  topK: number,
  random: () => number,
): number {
  const candidates: { id: number; probability: number }[] = [];
  for (let id = 0; id < probabilities.length; id++) {
    candidates.push({ id, probability: probabilities[id]! });
  }
  candidates.sort((a, b) => b.probability - a.probability);
  const top = candidates.slice(0, Math.max(1, Math.min(topK, candidates.length)));
  let sum = 0;
  for (const candidate of top) sum += candidate.probability;
  if (sum <= 0) return top[0]!.id;
  let target = random() * sum;
  for (const candidate of top) {
    target -= candidate.probability;
    if (target <= 0) return candidate.id;
  }
  return top[top.length - 1]!.id;
}

export function generateTinyGpt(
  model: TinyGptModel,
  prompt: string,
  mode: TinyGptMode,
  options: TinyGptGenerationOptions = {},
): TinyGptGeneration {
  const started = performance.now();
  const promptWords = tokenizePrompt(prompt);
  const ids = [BOS, ...promptWords.map((word) => model.index.get(word) ?? UNK)];
  const random = mulberry32(options.seed ?? 12345);
  const maxTokens = options.maxTokens ?? 24;
  const temperature = options.temperature ?? 0.8;
  const topK = options.topK ?? 40;
  const generated: number[] = [];
  let stopped = false;

  for (let step = 0; step < maxTokens; step++) {
    const window = windowFor(model, ids);
    const { logits } = forwardTinyGpt(model, window, mode);
    const probabilities = nextTokenDistribution(model, logits, temperature);
    const next = sampleToken(probabilities, topK, random);
    if (next === EOS) {
      stopped = true;
      break;
    }
    generated.push(next);
    ids.push(next);
  }

  const newTokens = generated.map((id) => model.vocab[id] ?? "<unk>");
  return {
    mode,
    promptTokens: promptWords.map((word) => word),
    tokenIds: ids,
    newTokens,
    text: newTokens.join(" "),
    steps: generated.length,
    stopped,
    elapsedMs: performance.now() - started,
  };
}

export function compareTinyGpt(
  model: TinyGptModel,
  prompt: string,
): TinyGptComparison {
  const promptWords = tokenizePrompt(prompt);
  const ids = [
    BOS,
    ...promptWords.map((word) => model.index.get(word) ?? UNK),
  ];
  const window = windowFor(model, ids);
  const floatForward = forwardTinyGpt(model, window, "float");
  const int8Forward = forwardTinyGpt(model, window, "int8");
  const floatProbabilities = nextTokenDistribution(
    model,
    floatForward.logits,
    1,
  );
  const int8Probabilities = nextTokenDistribution(model, int8Forward.logits, 1);
  const { vocabSize } = model.config;

  const toCandidates = (probabilities: Float32Array, limit: number) => {
    const list: TinyGptCandidate[] = [];
    for (let id = 0; id < vocabSize; id++) {
      list.push({
        id,
        token: model.vocab[id] ?? "<unk>",
        probability: probabilities[id]!,
      });
    }
    list.sort((a, b) => b.probability - a.probability);
    return list.slice(0, limit);
  };

  const floatTop = toCandidates(floatProbabilities, 8);
  const int8Top = toCandidates(int8Probabilities, 8);
  const floatIds = new Set(floatTop.slice(0, 5).map((item) => item.id));
  const overlap = int8Top
    .slice(0, 5)
    .filter((item) => floatIds.has(item.id)).length;

  let maxDelta = 0;
  let totalDelta = 0;
  for (let id = 0; id < vocabSize; id++) {
    const delta = Math.abs(
      floatProbabilities[id]! - int8Probabilities[id]!,
    );
    totalDelta += delta;
    if (delta > maxDelta) maxDelta = delta;
  }

  return {
    promptTokens: promptWords,
    floatTop,
    int8Top,
    topOneMatches: floatTop[0]!.id === int8Top[0]!.id,
    overlapAtFive: overlap,
    maxProbabilityDelta: maxDelta,
    meanProbabilityDelta: totalDelta / vocabSize,
  };
}

/** Small example of the actual int8 storage, for the explainer panel. */
export function sampleQuantizedWeights(
  model: TinyGptModel,
  count = 12,
): {
  readonly float: number[];
  readonly int8: number[];
  readonly scale: number;
  readonly restored: number[];
} {
  const layer = model.int8.layers[0]!;
  const matrix = layer.w1;
  const scale = matrix.scales[0]!;
  const float: number[] = [];
  const int8: number[] = [];
  const restored: number[] = [];
  for (let k = 0; k < Math.min(count, matrix.rows); k++) {
    const value = model.float.layers[0]!.w1[k * matrix.cols]!;
    const integer = matrix.values[k * matrix.cols]!;
    float.push(value);
    int8.push(integer);
    restored.push(integer * scale);
  }
  return { float, int8, scale, restored };
}
