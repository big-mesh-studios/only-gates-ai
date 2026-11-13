import {
  createTinyGpt,
  loadOnlyGatesModuleAuto,
  type OnlyGatesBackend,
  type OnlyGatesWasmModule,
  type TinyGpt,
  type TinyGptBuffer,
} from "@only-gates-ai/core";
import {
  tokenizePrompt,
  type TinyGptCheckpoint,
  type TinyGptConfig,
  type TinyGptModel,
  type TinyGptCandidate,
  nextTokens,
} from "./tinyGpt";
import { encodeFloat32 } from "./base64Float32";

/*
 * Integer-only WASM inference for the Tiny GPT demo.
 *
 * Everything numeric runs in the C module: int8 weights, Q16 activations,
 * integer RMSNorm/GELU and DI-Exp softmax. This wrapper only packs the float32
 * checkpoint once at load time, then moves token ids across the boundary.
 */

const BOS = 1;
const EOS = 2;
const UNK = 0;

export interface TinyGptWasmModel {
  readonly config: TinyGptConfig;
  readonly vocab: readonly string[];
  readonly index: ReadonlyMap<string, number>;
  readonly module: OnlyGatesWasmModule;
  readonly model: TinyGpt;
  /** Resolved integer lane: SIMD when the browser supports it, else scalar. */
  readonly backend: OnlyGatesBackend;
}

export interface TinyGptWasmGeneration {
  readonly promptTokens: readonly string[];
  readonly tokenIds: readonly number[];
  readonly newTokens: readonly string[];
  readonly text: string;
  readonly steps: number;
  readonly stopped: boolean;
  readonly elapsedMs: number;
}

export interface TinyGptWasmOptions {
  readonly maxTokens?: number;
  readonly temperature?: number;
  readonly topK?: number;
  readonly seed?: number;
}

function decodeBase64(base64: string): Float32Array {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return new Float32Array(bytes.buffer, 0, bytes.length / 4);
}

/**
 * Parameter lengths in `tiny_gpt_create` order:
 * tokenEmb, posEmb, gFinal, then per layer g1, wqkv, wo, g2, w1, w2.
 */
export function tinyGptParameterLengths(
  config: TinyGptConfig,
): readonly number[] {
  const { vocabSize, context, dim, layers, mlp } = config;
  const lengths = [vocabSize * dim, context * dim, dim];
  for (let layer = 0; layer < layers; layer++) {
    lengths.push(dim, dim * dim * 3, dim * dim, dim, dim * mlp, mlp * dim);
  }
  return lengths;
}

export function tinyGptParameterCount(config: TinyGptConfig): number {
  return tinyGptParameterLengths(config).reduce(
    (total, length) => total + length,
    0,
  );
}

/**
 * Packs the checkpoint in the order `tiny_gpt_create` expects:
 * tokenEmb, posEmb, gFinal, then per layer g1, wqkv, wo, g2, w1, w2.
 */
export function packTinyGptWeights(
  checkpoint: TinyGptCheckpoint,
): Float32Array {
  const parts: Float32Array[] = [
    decodeBase64(checkpoint.weights.tokenEmb),
    decodeBase64(checkpoint.weights.posEmb),
    decodeBase64(checkpoint.weights.gFinal),
  ];
  for (const layer of checkpoint.weights.layers) {
    parts.push(
      decodeBase64(layer.g1),
      decodeBase64(layer.wqkv),
      decodeBase64(layer.wo),
      decodeBase64(layer.g2),
      decodeBase64(layer.w1),
      decodeBase64(layer.w2),
    );
  }
  const total = parts.reduce((sum, part) => sum + part.length, 0);
  const packed = new Float32Array(total);
  let offset = 0;
  for (const part of parts) {
    packed.set(part, offset);
    offset += part.length;
  }
  return packed;
}

/**
 * Rebuilds the serialized checkpoint weight fields from packed float32 values.
 *
 * This is the inverse of `packTinyGptWeights`; imported word-level bundles use
 * it so the existing `loadTinyGpt` / WASM loaders can consume the result.
 */
export function unpackTinyGptWeights(
  config: TinyGptConfig,
  packed: Float32Array,
): TinyGptCheckpoint["weights"] {
  const lengths = tinyGptParameterLengths(config);
  const expected = lengths.reduce((total, length) => total + length, 0);
  if (packed.length !== expected) {
    throw new Error(
      `Packed weights have ${packed.length.toLocaleString()} values but the config needs ` +
        `${expected.toLocaleString()} (config mismatch or corrupt weights).`,
    );
  }
  let offset = 0;
  const take = (length: number): string => {
    const values = packed.slice(offset, offset + length);
    offset += length;
    return encodeFloat32(values);
  };

  const tokenEmb = take(lengths[0]!);
  const posEmb = take(lengths[1]!);
  const gFinal = take(lengths[2]!);
  let index = 3;
  const layers = Array.from({ length: config.layers }, () => ({
    g1: take(lengths[index++]!),
    wqkv: take(lengths[index++]!),
    wo: take(lengths[index++]!),
    g2: take(lengths[index++]!),
    w1: take(lengths[index++]!),
    w2: take(lengths[index++]!),
  }));
  return { tokenEmb, posEmb, layers, gFinal };
}

/** Loads the WASM module and builds the integer-only model. */
export async function loadTinyGptWasm(
  checkpoint: TinyGptCheckpoint,
): Promise<TinyGptWasmModel> {
  const { module, backend } = await loadOnlyGatesModuleAuto();
  const model = createTinyGpt(module, {
    config: checkpoint.config,
    weights: packTinyGptWeights(checkpoint),
  });
  const index = new Map<string, number>();
  checkpoint.vocab.forEach((token, id) => index.set(token, id));
  return {
    config: checkpoint.config,
    vocab: checkpoint.vocab,
    index,
    module,
    model,
    backend,
  };
}

function windowFor(model: TinyGptWasmModel, ids: readonly number[]): number[] {
  const window = ids.slice(-model.config.context);
  while (window.length < model.config.context) window.unshift(BOS);
  return window;
}

function promptIds(model: TinyGptWasmModel, prompt: string): number[] {
  return [
    BOS,
    ...tokenizePrompt(prompt).map((word) => model.index.get(word) ?? UNK),
  ];
}

/** Runs one integer forward pass and returns normalized probabilities. */
export function wasmProbabilities(
  model: TinyGptWasmModel,
  prompt: string,
  temperature = 1,
): Float32Array {
  const window = windowFor(model, promptIds(model, prompt));
  model.model.forward(window);
  return model.model.probabilities(temperature);
}

/** Top candidate tokens from the integer model, for the comparison panel. */
export function wasmTopTokens(
  model: TinyGptWasmModel,
  prompt: string,
  limit = 5,
): { id: number; token: string; probability: number }[] {
  const probabilities = wasmProbabilities(model, prompt, 1);
  const candidates = Array.from(probabilities, (probability, id) => ({
    id,
    token: model.vocab[id] ?? "<unk>",
    probability,
  }));
  candidates.sort((a, b) => b.probability - a.probability);
  return candidates.slice(0, limit);
}

/** Generates tokens using the C sampler (xorshift32 + integer cumsum). */
export function generateTinyGptWasm(
  model: TinyGptWasmModel,
  prompt: string,
  options: TinyGptWasmOptions = {},
): TinyGptWasmGeneration {
  const started = performance.now();
  const words = tokenizePrompt(prompt);
  const ids = [BOS, ...words.map((word) => model.index.get(word) ?? UNK)];
  const maxTokens = options.maxTokens ?? 24;
  const temperature = options.temperature ?? 0.8;
  const topK = options.topK ?? 40;
  model.model.setSeed(options.seed ?? 12345);

  const generated: number[] = [];
  let stopped = false;
  for (let step = 0; step < maxTokens; step++) {
    model.model.forward(windowFor(model, ids));
    const next = model.model.sample(temperature, topK);
    if (next === EOS) {
      stopped = true;
      break;
    }
    generated.push(next);
    ids.push(next);
  }

  const newTokens = generated.map((id) => model.vocab[id] ?? "<unk>");
  return {
    promptTokens: words,
    tokenIds: ids,
    newTokens,
    text: newTokens.join(" "),
    steps: generated.length,
    stopped,
    elapsedMs: performance.now() - started,
  };
}

/** Debug access to an internal Q16 buffer (tests and diagnostics). */
export function readWasmBuffer(
  model: TinyGptWasmModel,
  which: TinyGptBuffer,
  length: number,
): Int32Array {
  return model.model.readBuffer(which, length);
}

export interface TinyGptWasmComparison {
  readonly wasmTop: readonly TinyGptCandidate[];
  readonly floatTop: readonly TinyGptCandidate[];
  readonly topOneMatches: boolean;
  readonly overlapAtFive: number;
  readonly meanProbabilityDelta: number;
  readonly maxProbabilityDelta: number;
}

/**
 * Compares the integer WASM distribution with the float32 TS baseline for the
 * last context position. Both tokenizers are driven by the same checkpoint, so
 * the token ids line up.
 */
export function compareTinyGptWasm(
  wasmModel: TinyGptWasmModel,
  tsModel: TinyGptModel,
  prompt: string,
): TinyGptWasmComparison {
  const vocabSize = wasmModel.config.vocabSize;
  const floatAll = nextTokens(tsModel, prompt, "float", vocabSize);
  const floatProbability = new Float32Array(vocabSize);
  for (const candidate of floatAll) {
    floatProbability[candidate.id] = candidate.probability;
  }

  const wasmProbability = wasmProbabilities(wasmModel, prompt, 1);
  const wasmAll: TinyGptCandidate[] = Array.from(
    wasmProbability,
    (probability, id) => ({
      id,
      token: wasmModel.vocab[id] ?? "<unk>",
      probability,
    }),
  );
  wasmAll.sort((a, b) => b.probability - a.probability);

  const wasmTop = wasmAll.slice(0, 5);
  const floatTop = floatAll.slice(0, 5);

  let meanDelta = 0;
  let maxDelta = 0;
  for (let id = 0; id < vocabSize; id++) {
    const delta = Math.abs(floatProbability[id]! - wasmProbability[id]!);
    meanDelta += delta;
    if (delta > maxDelta) maxDelta = delta;
  }

  const floatIds = new Set(floatTop.map((candidate) => candidate.id));
  return {
    wasmTop,
    floatTop,
    topOneMatches: wasmTop[0]?.id === floatTop[0]?.id,
    overlapAtFive: wasmTop.filter((candidate) => floatIds.has(candidate.id))
      .length,
    meanProbabilityDelta: meanDelta / vocabSize,
    maxProbabilityDelta: maxDelta,
  };
}
