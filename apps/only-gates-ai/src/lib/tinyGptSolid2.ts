import {
  createTinyGpt,
  loadOnlyGatesModule,
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
}

export interface Solid2Generation {
  prompt: string;
  text: string;
  tokenIds: number[];
  newTokenCount: number;
  stopped: boolean;
  elapsedMs: number;
}

const BOS = 1; // <|user|> is a reasonable start sentinel if we don't have <bos>; but our format starts with <|user|>
const PAD = 0;

function windowTokens(ids: readonly number[], context: number, padId: number): number[] {
  const w = ids.slice(-context);
  while (w.length < context) w.unshift(padId);
  return w;
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
  const module = await loadOnlyGatesModule();
  const model = createTinyGpt(module, { config: c, weights });
  return { config: c, checkpoint, tokenizer: tok, module, model, parameterCount: checkpoint.parameterCount ?? expected };
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
  opts: { temperature: number; topK: number; repetitionPenalty: number; repWindow: number; rand: () => number },
): number {
  const V = m.config.vocabSize;
  const ctx = m.config.context;
  const all = m.model.logits();
  const base = (ctx - 1) * V;
  const temp = Math.max(opts.temperature, 1e-4);
  const scores = new Float64Array(V);
  for (let j = 0; j < V; j++) {
    scores[j] = (all[base + j] ?? 0) / 65536 / temp;
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
  opts: { maxTokens?: number; temperature?: number; topK?: number; seed?: number; repetitionPenalty?: number; repWindow?: number } = {},
): Solid2Generation {
  const started = performance.now();
  const padId = m.tokenizer.idFor("<|pad|>") ?? PAD;
  let ids = encodeChatPrompt(m.tokenizer, prompt);
  // If prompt encoding is empty (shouldn't happen), seed with user token.
  if (ids.length === 0) ids = [m.tokenizer.idFor("<|user|>") ?? 1, m.tokenizer.idFor("<|assistant|>") ?? 2];
  const maxTokens = opts.maxTokens ?? 160;
  const temperature = opts.temperature ?? 0.7;
  const topK = opts.topK ?? 40;
  const repetitionPenalty = opts.repetitionPenalty ?? 1.15;
  const repWindow = opts.repWindow ?? 64;
  const seed = opts.seed ?? (Date.now() >>> 0);
  m.model.setSeed(seed);
  const rand = mulberry32(seed ^ 0x9e3779b9);
  const gen: number[] = [];
  let stopped = false;
  for (let i = 0; i < maxTokens; i++) {
    m.model.forward(windowTokens(ids, m.config.context, padId));
    const next = repetitionPenalty > 1
      ? sampleWithRepetitionPenalty(m, ids, { temperature, topK, repetitionPenalty, repWindow, rand })
      : m.model.sample(temperature, topK);
    if (isStopToken(m.tokenizer, next)) { stopped = true; break; }
    gen.push(next);
    ids.push(next);
    // avoid degenerate runaway on pad
    if (next === padId) { stopped = true; break; }
  }
  return {
    prompt,
    text: decodeGenerated(m.tokenizer, gen),
    tokenIds: ids,
    newTokenCount: gen.length,
    stopped,
    elapsedMs: performance.now() - started,
  };
}
