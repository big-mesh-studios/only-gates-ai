import type { OnlyGatesWasmModule } from "./wasm-types.js";
import { allocate, readInts, withMemory } from "./memory.js";
import { OnlyGatesError, OnlyGatesErrorCode } from "./errors.js";

/*
 * Integer-only Tiny GPT wrapper.
 *
 * The C module keeps the whole inference path in int8/int32/int64: int8
 * weights with per-column Q16 scales, a Q16 int32 activation stream, integer
 * RMSNorm/GELU, and DI-Exp softmax over Q12 log-scores. This file only moves
 * tokens and buffers across the WebAssembly boundary; it performs no
 * arithmetic on the model besides converting a temperature to Q16.
 */

/** Model dimensions, matching the packed checkpoint. */
export interface TinyGptConfig {
  readonly vocabSize: number;
  readonly context: number;
  readonly dim: number;
  readonly layers: number;
  readonly heads: number;
  readonly mlp: number;
}

/** Internal activation buffers exposed for tests and diagnostics. */
export enum TinyGptBuffer {
  X = 0,
  X1 = 1,
  H = 2,
  QKV = 3,
  Attention = 4,
  Y = 5,
  Z = 6,
  Gelu = 7,
  Mlp = 8,
  Logits = 9,
}

export interface CreateTinyGptOptions {
  readonly config: TinyGptConfig;
  /**
   * Packed float32 checkpoint in the order expected by `tiny_gpt_create`:
   * tokenEmb, posEmb, gFinal, then per layer g1, wqkv, wo, g2, w1, w2.
   * The module quantizes it once; the buffer is not retained.
   */
  readonly weights: Float32Array;
}

const Q16_ONE = 65536;

function temperatureToQ16(temperature: number): number {
  const safe = Number.isFinite(temperature) ? temperature : 1;
  const clamped = Math.max(0.01, safe);
  return Math.round(clamped * Q16_ONE);
}

/** Integer-only Tiny GPT model backed by the WebAssembly module. */
export class TinyGpt {
  private tokensPointer: number;
  private destroyed = false;

  private constructor(
    private readonly module: OnlyGatesWasmModule,
    private readonly modelPointer: number,
    readonly config: TinyGptConfig,
  ) {
    this.tokensPointer = allocate(module, config.context * 4);
  }

  /** Quantizes a packed float32 checkpoint and creates the model. */
  static create(
    module: OnlyGatesWasmModule,
    options: CreateTinyGptOptions,
  ): TinyGpt {
    const { config, weights } = options;
    const pointer = withMemory(module, weights.byteLength, (weightsPointer) => {
      module.HEAPF32.set(weights, weightsPointer >>> 2);
      return module._tiny_gpt_create(
        config.vocabSize,
        config.context,
        config.dim,
        config.layers,
        config.heads,
        config.mlp,
        weightsPointer,
      );
    });
    if (pointer === 0) {
      throw new OnlyGatesError(
        OnlyGatesErrorCode.InvalidParams,
        `tiny_gpt_create failed with error ${module._tiny_gpt_last_error()}`,
      );
    }
    return new TinyGpt(module, pointer, { ...config });
  }

  /** Runs the integer forward pass. `tokens` must hold `config.context` ids. */
  forward(tokens: ArrayLike<number>): void {
    this.assertAlive();
    if (tokens.length !== this.config.context) {
      throw new OnlyGatesError(
        OnlyGatesErrorCode.InvalidParams,
        `expected ${this.config.context} tokens, got ${tokens.length}`,
      );
    }
    this.module.HEAP32.set(tokens, this.tokensPointer >>> 2);
    this.module._tiny_gpt_forward(this.modelPointer, this.tokensPointer);
  }

  /**
   * Computes only the absolute positions `[start, start + count)`, reusing the
   * keys and values cached for earlier positions.
   *
   * `tokens` is indexed by absolute position. Prefill once with
   * `forwardRange(tokens, 0, promptLength)`, then call
   * `forwardRange(tokens, position, 1)` per generated token. The result is
   * bit-identical to `forward` over the same window.
   */
  forwardRange(tokens: ArrayLike<number>, start: number, count: number): void {
    this.assertAlive();
    if (
      !Number.isInteger(start) ||
      !Number.isInteger(count) ||
      start < 0 ||
      count < 0 ||
      start + count > this.config.context ||
      tokens.length < start + count
    ) {
      throw new OnlyGatesError(
        OnlyGatesErrorCode.InvalidParams,
        `forwardRange(${start}, ${count}) is outside the ${this.config.context}-position context`,
      );
    }
    this.module.HEAP32.set(tokens, this.tokensPointer >>> 2);
    this.module._tiny_gpt_forward_range(
      this.modelPointer,
      this.tokensPointer,
      start,
      count,
    );
  }

  /** Clears cached keys/values. Not required before a full `forward`. */
  resetCache(): void {
    this.assertAlive();
    this.module._tiny_gpt_reset_cache(this.modelPointer);
  }

  /** Q16 logits for every context position, row-major context x vocab. */
  logits(): Int32Array {
    this.assertAlive();
    const pointer = this.module._tiny_gpt_logits(this.modelPointer);
    return readInts(
      this.module,
      pointer,
      this.config.context * this.config.vocabSize,
    );
  }

  /**
   * Q16 logits for one absolute position. This is the decode hot path: one
   * vocab-sized row instead of the whole context x vocab buffer.
   */
  logitsRow(position: number): Int32Array {
    this.assertAlive();
    const pointer = this.module._tiny_gpt_logits_row(
      this.modelPointer,
      position,
    );
    if (pointer === 0) {
      throw new OnlyGatesError(
        OnlyGatesErrorCode.InvalidParams,
        `logits position ${position} is outside the context`,
      );
    }
    return readInts(this.module, pointer, this.config.vocabSize);
  }

  /** Normalized probabilities for the last context position. */
  probabilities(temperature = 1): Float32Array {
    this.assertAlive();
    const output = withMemory(
      this.module,
      this.config.vocabSize * 4,
      (pointer) => {
        this.module._tiny_gpt_probabilities(
          this.modelPointer,
          temperatureToQ16(temperature),
          pointer,
        );
        return readInts(this.module, pointer, this.config.vocabSize);
      },
    );
    const probabilities = new Float32Array(output.length);
    for (let i = 0; i < output.length; i++) {
      probabilities[i] = output[i]! / Q16_ONE;
    }
    return probabilities;
  }

  /** Samples one token from the top-k integer distribution. */
  sample(temperature = 1, topK = 40): number {
    this.assertAlive();
    return this.module._tiny_gpt_sample(
      this.modelPointer,
      temperatureToQ16(temperature),
      topK,
    );
  }

  setSeed(seed: number): void {
    this.assertAlive();
    this.module._tiny_gpt_set_seed(this.modelPointer, seed >>> 0);
  }

  /** Reads one internal Q16 buffer. Length depends on the buffer kind. */
  readBuffer(which: TinyGptBuffer, length: number): Int32Array {
    this.assertAlive();
    const pointer = this.module._tiny_gpt_buffer(this.modelPointer, which);
    return readInts(this.module, pointer, length);
  }

  get lastError(): number {
    return this.module._tiny_gpt_last_error();
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.module._free(this.tokensPointer);
    this.module._tiny_gpt_destroy(this.modelPointer);
  }

  private assertAlive(): void {
    if (this.destroyed) {
      throw new OnlyGatesError(
        OnlyGatesErrorCode.InvalidParams,
        "TinyGpt model has been destroyed",
      );
    }
  }
}

/** Convenience factory mirroring the other package entry points. */
export function createTinyGpt(
  module: OnlyGatesWasmModule,
  options: CreateTinyGptOptions,
): TinyGpt {
  return TinyGpt.create(module, options);
}
