import {
  createTinyGpt,
  loadOnlyGatesModuleAuto,
  type OnlyGatesBackend,
  type OnlyGatesWasmModule,
  type TinyGpt as WasmTinyGpt,
  type TinyGptConfig,
} from "@only-gates-ai/core";

import {
  BPE_SPECIAL,
  encodeChatPrompt,
  loadBpeTokenizer,
  type BpeTokenizer,
} from "./tinyGptBpe";

/*
 * Masked-diffusion Luau model.
 *
 * The checkpoint is a bidirectional Tiny GPT trained with an absorbing mask
 * token. Inference is still integer-only: `maskedStep` runs the C denoiser,
 * which performs one full bidirectional forward pass and fills the next
 * left-to-right block of masked answer positions. JavaScript only supplies the
 * prompt, schedules UI frames, and decodes the partially filled token window
 * for display.
 */

export interface MaskedDiffusionInfo {
  readonly maskTokenId?: number;
  readonly answerBudget?: number;
  readonly steps?: number;
  readonly schedule?: string;
}

export interface MaskedDiffusionCheckpoint {
  readonly format: string;
  readonly version: number;
  readonly config: TinyGptConfig;
  readonly parameterCount: number;
  readonly tokenizer: string;
  readonly diffusion?: MaskedDiffusionInfo;
  readonly meta?: {
    readonly name?: string;
    readonly source?: string;
    readonly license?: string;
    readonly training?: Record<string, unknown>;
  };
  readonly samples?: ReadonlyArray<{ prompt: string; text: string }>;
  readonly prompts?: readonly string[];
}

export interface MaskedDiffusionModel {
  readonly config: TinyGptConfig;
  readonly checkpoint: MaskedDiffusionCheckpoint;
  readonly tokenizer: BpeTokenizer;
  readonly module: OnlyGatesWasmModule;
  readonly model: WasmTinyGpt;
  readonly parameterCount: number;
  readonly backend: OnlyGatesBackend;
  readonly maskId: number;
}

export interface MaskedDiffusionDecodeOptions {
  readonly steps?: number;
  readonly temperature?: number;
  readonly maxAnswerTokens?: number;
  readonly seed?: number;
  readonly mode?: "write" | "ask";
}

export interface MaskedDiffusionSnapshot {
  readonly tokens: Int32Array;
  readonly length: number;
  readonly promptLength: number;
  readonly answerLength: number;
  readonly remaining: number;
  readonly initialMasks: number;
  readonly step: number;
  readonly totalSteps: number;
  readonly answer: string;
  /** Partially denoised answer; runs of masks are rendered as `▁`. */
  readonly preview: string;
  readonly done: boolean;
  readonly elapsedMs: number;
  readonly computeMs: number;
}

/** Mirrors the packed-weight count used by `tiny_gpt_create`. */
export function maskedDiffusionParameterCount(config: TinyGptConfig): number {
  const { vocabSize, context, dim, layers, mlp } = config;
  return (
    vocabSize * dim +
    context * dim +
    dim +
    layers * (2 * dim + 4 * dim * dim + 2 * dim * mlp)
  );
}

export async function createMaskedDiffusionWasmFromData(
  checkpoint: MaskedDiffusionCheckpoint,
  tokenizer: BpeTokenizer,
  weights: Float32Array,
): Promise<MaskedDiffusionModel> {
  const expected = maskedDiffusionParameterCount(checkpoint.config);
  if (weights.length !== expected) {
    throw new Error(
      `weights length ${weights.length} != expected ${expected} (config mismatch or corrupt weights)`,
    );
  }
  const maskId =
    checkpoint.diffusion?.maskTokenId ??
    tokenizer.idFor(BPE_SPECIAL.mask) ??
    checkpoint.config.vocabSize - 1;
  if (maskId < 0 || maskId >= checkpoint.config.vocabSize) {
    throw new Error(`invalid mask token id ${maskId}`);
  }

  const { module, backend } = await loadOnlyGatesModuleAuto();
  const model = createTinyGpt(module, {
    config: checkpoint.config,
    weights,
  });
  model.setBidirectional(true);
  return {
    config: checkpoint.config,
    checkpoint,
    tokenizer,
    module,
    model,
    parameterCount: checkpoint.parameterCount ?? expected,
    backend,
    maskId,
  };
}

export async function loadMaskedDiffusionWasm(
  baseUrl: string,
): Promise<MaskedDiffusionModel> {
  const base = baseUrl.replace(/\/?$/, "/");
  const [checkpointRes, tokenizer, weightsRes] = await Promise.all([
    fetch(base + "checkpoint.json"),
    loadBpeTokenizer(base + "tokenizer.json"),
    fetch(base + "weights.bin"),
  ]);
  if (!checkpointRes.ok) {
    throw new Error(`checkpoint: ${checkpointRes.status} ${checkpointRes.statusText}`);
  }
  if (!weightsRes.ok) {
    throw new Error(`weights: ${weightsRes.status} ${weightsRes.statusText}`);
  }
  const checkpoint = (await checkpointRes.json()) as MaskedDiffusionCheckpoint;
  const weights = new Float32Array(await weightsRes.arrayBuffer());
  return createMaskedDiffusionWasmFromData(checkpoint, tokenizer, weights);
}

/**
 * One prompt whose answer region is filled by repeated C denoising steps.
 *
 * `step()` is deliberately exposed instead of a single `run()` so the UI can
 * paint between passes. Each pass is a full bidirectional forward over the
 * currently unmasked prefix, not a cached decode step.
 */
export class MaskedDiffusionDecoder {
  readonly promptIds: readonly number[];
  readonly promptText: string;
  readonly totalSteps: number;
  readonly temperature: number;
  private readonly startedAt = performance.now();
  private tokens: Int32Array;
  private stepIndex = 0;
  private remaining: number;
  private computeMs = 0;
  private finished = false;
  private readonly endId: number | undefined;
  private readonly padId: number | undefined;
  private readonly userId: number | undefined;
  private readonly assistantId: number | undefined;
  private readonly factsId: number | undefined;

  constructor(
    private readonly model: MaskedDiffusionModel,
    prompt: string,
    options: MaskedDiffusionDecodeOptions = {},
  ) {
    this.promptText = prompt;
    this.totalSteps = Math.max(1, Math.round(options.steps ?? model.checkpoint.diffusion?.steps ?? 8));
    this.temperature = options.temperature ?? 0.4;
    this.endId = model.tokenizer.idFor(BPE_SPECIAL.end);
    this.padId = model.tokenizer.idFor(BPE_SPECIAL.pad);
    this.userId = model.tokenizer.idFor(BPE_SPECIAL.user);
    this.assistantId = model.tokenizer.idFor(BPE_SPECIAL.assistant);
    this.factsId = model.tokenizer.idFor(BPE_SPECIAL.facts);

    const prefix = options.mode === "ask" ? "Question: " : "Task: ";
    this.promptIds = encodeChatPrompt(model.tokenizer, prefix + prompt);
    const context = model.config.context;
    const available = context - this.promptIds.length;
    // Training masks the entire remaining context, and the C denoiser expects
    // the same shape at inference. `answerBudget` can still cap the region for
    // experiments, but 0 or omitted means "fill the context".
    const maxAnswer =
      options.maxAnswerTokens ?? model.checkpoint.diffusion?.answerBudget ?? 0;
    const answerLength =
      maxAnswer > 0 ? Math.min(maxAnswer, available) : available;
    if (answerLength < 8) {
      throw new Error(
        `prompt uses ${this.promptIds.length} of ${context} tokens; there is no room to denoise an answer`,
      );
    }

    this.length = this.promptIds.length + answerLength;
    this.promptLength = this.promptIds.length;
    this.answerLength = answerLength;
    this.initialMasks = answerLength;
    this.remaining = answerLength;
    this.tokens = new Int32Array(this.length);
    this.tokens.set(this.promptIds, 0);
    for (let i = this.promptLength; i < this.length; i++) {
      this.tokens[i] = model.maskId;
    }
    model.model.setSeed(options.seed ?? (Date.now() >>> 0));
  }

  readonly length: number;
  readonly promptLength: number;
  readonly answerLength: number;
  readonly initialMasks: number;

  get stepCount(): number {
    return this.stepIndex;
  }

  get done(): boolean {
    return this.finished;
  }

  /** Runs one denoising pass and returns the new partial snapshot. */
  step(): MaskedDiffusionSnapshot {
    if (this.finished) {
      return this.snapshot();
    }
    const started = performance.now();
    const result = this.model.model.maskedStep(
      this.tokens,
      this.length,
      this.promptLength,
      this.model.maskId,
      this.temperature,
      this.stepIndex,
      this.totalSteps,
    );
    this.computeMs += performance.now() - started;
    this.tokens = result.tokens;
    this.remaining = result.remaining;
    this.stepIndex += 1;
    // Never stop just because `<|end|>` appeared early. In diffusion the end
    // position is often confident before the tokens before it are filled; the
    // decoder must finish the schedule so those positions can resolve. `end`
    // is still used to trim the preview and final answer.
    this.finished =
      this.remaining === 0 || this.stepIndex >= this.totalSteps;
    return this.snapshot();
  }

  /** Runs to completion. Use `step()` for a time-sliced animation. */
  run(): MaskedDiffusionSnapshot {
    while (!this.finished) {
      this.step();
    }
    return this.snapshot();
  }

  snapshot(): MaskedDiffusionSnapshot {
    return {
      tokens: this.tokens.slice(),
      length: this.length,
      promptLength: this.promptLength,
      answerLength: this.answerLength,
      remaining: this.remaining,
      initialMasks: this.initialMasks,
      step: this.stepIndex,
      totalSteps: this.totalSteps,
      answer: this.decodeAnswer(),
      preview: this.decodePreview(),
      done: this.finished,
      elapsedMs: performance.now() - this.startedAt,
      computeMs: this.computeMs,
    };
  }

  private isSpecial(id: number): boolean {
    return (
      id === this.userId ||
      id === this.assistantId ||
      id === this.factsId ||
      id === this.padId
    );
  }

  private stopPosition(): number {
    if (this.endId === undefined) return -1;
    for (let i = this.promptLength; i < this.length; i++) {
      if (this.tokens[i] === this.endId) return i;
    }
    return -1;
  }

  private decodeAnswer(): string {
    const ids: number[] = [];
    for (let i = this.promptLength; i < this.length; i++) {
      const id = this.tokens[i]!;
      if (id === this.endId) break;
      if (id === this.padId || id === this.model.maskId) continue;
      if (this.isSpecial(id)) continue;
      ids.push(id);
    }
    return this.model.tokenizer.decode(ids).replace(/\s+$/, "");
  }

  private decodePreview(): string {
    const stop = this.stopPosition();
    const limit = stop >= 0 ? stop : this.length;
    let output = "";
    let chunk: number[] = [];
    let masks = 0;
    const flush = () => {
      if (chunk.length > 0) {
        output += this.model.tokenizer.decode(chunk);
        chunk = [];
      }
      if (masks > 0) {
        output += masks === 1 ? "▁" : `▁${masks}▁`;
        masks = 0;
      }
    };
    for (let i = this.promptLength; i < limit; i++) {
      const id = this.tokens[i]!;
      if (id === this.model.maskId) {
        masks += 1;
        continue;
      }
      if (id === this.padId || id === this.endId) {
        flush();
        break;
      }
      if (this.isSpecial(id)) continue;
      chunk.push(id);
    }
    flush();
    return output.length > 900 ? output.slice(0, 900) + "…" : output;
  }
}

export function createMaskedDiffusionDecoder(
  model: MaskedDiffusionModel,
  prompt: string,
  options: MaskedDiffusionDecodeOptions = {},
): MaskedDiffusionDecoder {
  return new MaskedDiffusionDecoder(model, prompt, options);
}
