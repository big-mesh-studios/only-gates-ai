import {
  createTinyGpt,
  loadOnlyGatesModuleAuto,
  type OnlyGatesBackend,
  type OnlyGatesWasmModule,
  type TinyGpt as WasmTinyGpt,
} from "@only-gates-ai/core";
import { decodeGenerated, encodeChatPrompt, isStopToken, loadBpeTokenizer, type BpeTokenizer } from "./tinyGptBpe";

export interface Solid2Checkpoint {
  format: string;
  version: number;
  config: {
    vocabSize: number;
    context: number;
    dim: number;
    layers: number;
    heads: number;
    mlp: number;
  };
  parameterCount: number;
  tokenizer: string;
  meta?: { name?: string; source?: string; license?: string; training?: Record<string, unknown> };
  samples?: Array<{ prompt: string; text: string }>;
  prompts?: string[];
}

export interface Solid2WasmModel {
  config: Solid2Checkpoint["config"];
  checkpoint: Solid2Checkpoint;
  tokenizer: BpeTokenizer;
  module: OnlyGatesWasmModule;
  model: WasmTinyGpt;
  parameterCount: number;
  /** Resolved integer lane: SIMD when the browser supports it, else scalar. */
  backend: OnlyGatesBackend;
}

export interface Solid2Generation {
  prompt: string;
  text: string;
  tokenIds: number[];
  newTokenCount: number;
  stopped: boolean;
  /** Wall time since the decoder was created, including cooperative yields. */
  elapsedMs: number;
  /** Time spent inside the integer sampler/forward pass (prefill + decode). */
  computeMs: number;
  /** Decode rate excluding UI scheduling. */
  computeTokensPerSecond: number;
}

const PAD = 0;

/** Options for incremental decoding. */
export interface Solid2DecodeOptions {
  maxTokens?: number;
  temperature?: number;
  topK?: number;
  seed?: number;
  repetitionPenalty?: number;
  repWindow?: number;
}

/**
 * One token-at-a-time decoder over the integer WASM model.
 *
 * The prompt is prefilled once, then every `step()` computes a single context
 * row and reuses the cached keys and values. Positions are never padded and
 * never slid: generation stops at the context limit instead of re-basing the
 * window, because the KV cache is addressed by absolute position.
 */
export class Solid2Decoder {
  readonly promptIds: readonly number[];
  readonly opts: Required<Solid2DecodeOptions>;
  readonly promptText: string;
  private readonly ids: number[];
  private readonly generated: number[] = [];
  private readonly startedAt = performance.now();
  private stopped = false;
  private finished = false;
  private steps = 0;
  private prefillMs = 0;
  private computeMs = 0;

  constructor(
    private readonly model: Solid2WasmModel,
    prompt: string,
    options: Solid2DecodeOptions = {},
  ) {
    this.opts = {
      maxTokens: options.maxTokens ?? 160,
      temperature: options.temperature ?? 0.7,
      topK: options.topK ?? 40,
      seed: options.seed ?? (Date.now() >>> 0),
      repetitionPenalty: options.repetitionPenalty ?? 1.15,
      repWindow: options.repWindow ?? 64,
    };
    let ids = encodeChatPrompt(model.tokenizer, prompt);
    if (ids.length === 0) {
      ids = [
        model.tokenizer.idFor("<|user|>") ?? 1,
        model.tokenizer.idFor("<|assistant|>") ?? 2,
      ];
    }
    // A KV cache is addressed by absolute position, so a prompt longer than the
    // context cannot be slid; keep the newest window and generate nothing.
    this.ids = ids.slice(-model.config.context);
    this.promptIds = this.ids.slice();
    this.promptText = prompt;
    model.model.setSeed(this.opts.seed);
    const prefillStarted = performance.now();
    model.model.resetCache();
    if (this.ids.length > 0) {
      model.model.forwardRange(this.ids, 0, this.ids.length);
    } else {
      this.finished = true;
    }
    this.prefillMs = performance.now() - prefillStarted;
  }

  /** Absolute position whose logits produce the next token. */
  get position(): number {
    return this.ids.length - 1;
  }

  get tokenIds(): readonly number[] {
    return this.ids;
  }

  get newTokenCount(): number {
    return this.generated.length;
  }

  get elapsedMs(): number {
    return performance.now() - this.startedAt;
  }

  /** Samples one token, then computes its context row for the next step. */
  step(): { token: number | null; done: boolean } {
    if (this.finished) return { token: null, done: true };

    const maxTokens = this.opts.maxTokens;
    const context = this.model.config.context;
    if (this.steps >= maxTokens || this.stopped) {
      this.finished = true;
      return { token: null, done: true };
    }
    // A full window can still emit one token from its last position, but it
    // cannot then compute that token's row, so the next step finishes.
    const atCapacity = this.ids.length >= context;
    if (atCapacity && this.steps > 0) {
      this.finished = true;
      return { token: null, done: true };
    }

    const computeStarted = performance.now();
    // The integer sampler runs inside WASM against exactly the row that
    // `forwardRange` just wrote. Keeping the vocab-sized softmax and top-k out
    // of JavaScript removes the dominant per-token overhead.
    const next = this.model.model.sampleAtPenalized(
      this.position,
      this.ids,
      this.opts.temperature,
      this.opts.topK,
      this.opts.repetitionPenalty,
      this.opts.repWindow,
    );

    if (isStopToken(this.model.tokenizer, next) || next === PAD) {
      this.computeMs += performance.now() - computeStarted;
      this.stopped = true;
      this.finished = true;
      return { token: null, done: true };
    }

    this.ids.push(next);
    this.generated.push(next);
    this.steps += 1;
    // Only pay for the next row if another step can actually use it.
    if (this.steps < maxTokens && this.ids.length - 1 < context) {
      this.model.model.forwardRange(this.ids, this.ids.length - 1, 1);
    }
    this.computeMs += performance.now() - computeStarted;
    return { token: next, done: false };
  }

  /** Runs to completion. Use `step()` for a time-sliced UI loop. */
  run(): Solid2Generation {
    while (!this.step().done) {
      // decode
    }
    return this.result();
  }

  result(): Solid2Generation {
    const elapsedMs = performance.now() - this.startedAt;
    const computeMs = this.prefillMs + this.computeMs;
    return {
      prompt: this.promptText,
      text: decodeGenerated(this.model.tokenizer, this.generated),
      tokenIds: this.ids.slice(),
      newTokenCount: this.generated.length,
      stopped: this.stopped,
      elapsedMs,
      computeMs,
      computeTokensPerSecond:
        computeMs > 0 ? (this.generated.length * 1000) / computeMs : 0,
    };
  }
}

/** Creates an incremental decoder for one prompt. */
export function createSolid2Decoder(
  model: Solid2WasmModel,
  prompt: string,
  options: Solid2DecodeOptions = {},
): Solid2Decoder {
  return new Solid2Decoder(model, prompt, options);
}

/**
 * Builds the integer-only WASM model from an in-memory checkpoint.
 *
 * Browser training already has the packed float32 weights, so it can skip the
 * checkpoint/base64 round trip and construct the quantized model directly.
 */
export async function createSolid2WasmFromData(
  checkpoint: Solid2Checkpoint,
  tokenizer: BpeTokenizer,
  weights: Float32Array,
): Promise<Solid2WasmModel> {
  const c = checkpoint.config;
  // Kept in sync with `solid2ParameterCount` in solid2Config.ts; this module
  // avoids importing it so the standalone regression test can transpile it.
  const expected =
    c.vocabSize * c.dim +
    c.context * c.dim +
    c.dim +
    c.layers *
      (2 * c.dim + 4 * c.dim * c.dim + 2 * c.dim * c.mlp);
  if (weights.length !== expected) {
    throw new Error(
      `weights length ${weights.length} != expected ${expected} (config mismatch or corrupt weights)`,
    );
  }
  const { module, backend } = await loadOnlyGatesModuleAuto();
  const model = createTinyGpt(module, { config: c, weights });
  return {
    config: c,
    checkpoint,
    tokenizer,
    module,
    model,
    parameterCount: checkpoint.parameterCount ?? expected,
    backend,
  };
}

/** Load the Solid 2 ~1M BPE model for integer-only WASM inference. */
export async function loadSolid2Wasm(baseUrl: string): Promise<Solid2WasmModel> {
  const base = baseUrl.replace(/\/?$/, "/");
  const [ckRes, tok, binRes] = await Promise.all([
    fetch(base + "checkpoint.json"),
    loadBpeTokenizer(base + "tokenizer.json"),
    fetch(base + "weights.bin"),
  ]);
  if (!ckRes.ok) throw new Error(`checkpoint: ${ckRes.status}`);
  if (!binRes.ok) throw new Error(`weights: ${binRes.status}`);
  const checkpoint = (await ckRes.json()) as Solid2Checkpoint;
  const bin = await binRes.arrayBuffer();
  const weights = new Float32Array(bin);
  return createSolid2WasmFromData(checkpoint, tok, weights);
}

export function generateSolid2Wasm(
  m: Solid2WasmModel,
  prompt: string,
  opts: Solid2DecodeOptions = {},
): Solid2Generation {
  return createSolid2Decoder(m, prompt, opts).run();
}
