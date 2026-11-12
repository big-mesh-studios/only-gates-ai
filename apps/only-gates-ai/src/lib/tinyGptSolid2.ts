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
  elapsedMs: number;
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
  private readonly rand: () => number;
  private readonly startedAt = performance.now();
  private stopped = false;
  private finished = false;
  private steps = 0;

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
    this.rand = mulberry32(this.opts.seed ^ 0x9e3779b9);
    model.model.resetCache();
    if (this.ids.length > 0) {
      model.model.forwardRange(this.ids, 0, this.ids.length);
    } else {
      this.finished = true;
    }
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

    const row = this.model.model.logitsRow(this.position);
    const next = sampleWithRepetitionPenalty(this.model, this.ids, row, {
      temperature: this.opts.temperature,
      topK: this.opts.topK,
      repetitionPenalty: this.opts.repetitionPenalty,
      repWindow: this.opts.repWindow,
      rand: this.rand,
    });

    if (isStopToken(this.model.tokenizer, next) || next === PAD) {
      this.stopped = true;
      this.finished = true;
      return { token: null, done: true };
    }

    this.ids.push(next);
    this.generated.push(next);
    this.steps += 1;
    if (this.ids.length - 1 < context) {
      this.model.model.forwardRange(this.ids, this.ids.length - 1, 1);
    }
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
    return {
      prompt: this.promptText,
      text: decodeGenerated(this.model.tokenizer, this.generated),
      tokenIds: this.ids.slice(),
      newTokenCount: this.generated.length,
      stopped: this.stopped,
      elapsedMs: performance.now() - this.startedAt,
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
  // quick sanity: expected param count
  const c = checkpoint.config;
  const expected = c.vocabSize * c.dim + c.context * c.dim + c.dim +
    c.layers * (c.dim + c.dim * c.dim * 3 + c.dim * c.dim + c.dim + c.dim * c.mlp + c.mlp * c.dim);
  if (weights.length !== expected) {
    throw new Error(`weights length ${weights.length} != expected ${expected} (config mismatch or corrupt weights.bin)`);
  }
  const { module, backend } = await loadOnlyGatesModuleAuto();
  const model = createTinyGpt(module, { config: c, weights });
  return { config: c, checkpoint, tokenizer: tok, module, model, parameterCount: checkpoint.parameterCount ?? expected, backend };
}

function mulberry32(seed: number): () => number {
  let t = seed >>> 0;
  return () => {
    t += 0x6d2b79f5;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

function sampleWithRepetitionPenalty(
  m: Solid2WasmModel,
  ids: readonly number[],
  row: Int32Array,
  opts: { temperature: number; topK: number; repetitionPenalty: number; repWindow: number; rand: () => number },
): number {
  const V = m.config.vocabSize;
  const temp = Math.max(opts.temperature, 1e-4);
  const scores = new Float64Array(V);
  for (let j = 0; j < V; j++) {
    scores[j] = (row[j] ?? 0) / 65536 / temp;
  }
  if (opts.repetitionPenalty > 1) {
    const recent = ids.slice(-opts.repWindow);
    const seen = new Set<number>();
    for (const t of recent) {
      if (t < 0 || t >= V || seen.has(t)) continue;
      seen.add(t);
      const s = scores[t] ?? 0;
      scores[t] = s > 0 ? s / opts.repetitionPenalty : s * opts.repetitionPenalty;
    }
  }
  if (opts.topK > 0 && opts.topK < V) {
    // partial selection via copy+sort of indices is fine for V=4096
    const idx = Array.from({ length: V }, (_, i) => i);
    idx.sort((a, b) => (scores[b] ?? 0) - (scores[a] ?? 0));
    const kth = Math.min(opts.topK - 1, V - 1);
    const thresh = scores[idx[kth] ?? 0] ?? -1e9;
    for (let j = 0; j < V; j++) if ((scores[j] ?? -1e9) < thresh) scores[j] = -1e9;
  }
  let max = -Infinity;
  for (let j = 0; j < V; j++) { const v = scores[j] ?? -1e9; if (v > max) max = v; }
  let sum = 0;
  for (let j = 0; j < V; j++) { const e = Math.exp((scores[j] ?? -1e9) - max); scores[j] = e; sum += e; }
  let r = opts.rand() * sum;
  for (let j = 0; j < V; j++) { r -= scores[j] ?? 0; if (r <= 0) return j; }
  return V - 1;
}

export function generateSolid2Wasm(
  m: Solid2WasmModel,
  prompt: string,
  opts: Solid2DecodeOptions = {},
): Solid2Generation {
  return createSolid2Decoder(m, prompt, opts).run();
}
