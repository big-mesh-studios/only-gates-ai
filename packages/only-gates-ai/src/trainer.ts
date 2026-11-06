import type { OnlyGatesWasmModule } from "./wasm.js";
import { checkError } from "./wasm.js";

/** A dataset stored as one input/output array per sample. */
export interface Dataset {
  readonly inputs: readonly (readonly boolean[])[];
  readonly outputs: readonly (readonly number[])[];
}

/**
 * Steps through a dataset one sample at a time. Created with
 * `Circuit.createTrainer()`.
 */
export class Trainer {
  private readonly module: OnlyGatesWasmModule;
  private readonly pointer: number;
  private readonly inputsPointer: number;
  private readonly outputsPointer: number;
  private learningRateValue: number;
  private indexValue = 0;
  private destroyed = false;

  /** @internal Use `Circuit.createTrainer()` instead. */
  constructor(
    module: OnlyGatesWasmModule,
    pointer: number,
    readonly sampleCount: number,
    inputsPointer: number,
    outputsPointer: number,
    learningRate: number,
  ) {
    this.module = module;
    this.pointer = pointer;
    this.inputsPointer = inputsPointer;
    this.outputsPointer = outputsPointer;
    this.learningRateValue = learningRate;
  }

  /** Index of the sample the next `step()` will use. */
  get index(): number {
    return this.indexValue;
  }

  get learningRate(): number {
    return this.learningRateValue;
  }

  set learningRate(value: number) {
    this.assertAlive();
    this.module._trainer_set_learning_rate(this.pointer, value);
    checkError(this.module);
    this.learningRateValue = value;
  }

  /**
   * Trains on one sample and returns its loss:
   * `0.5 * sum((output - target)^2)`.
   */
  step(): number {
    this.assertAlive();
    const loss = this.module._trainer_step(this.pointer);
    checkError(this.module);
    this.indexValue = (this.indexValue + 1) % this.sampleCount;
    return loss;
  }

  /** Runs `epochs` passes over the dataset and returns the mean loss. */
  trainEpoch(epochs = 1): number {
    this.assertAlive();
    let total = 0;
    let count = 0;
    for (let epoch = 0; epoch < epochs; epoch++) {
      for (let sample = 0; sample < this.sampleCount; sample++) {
        total += this.step();
        count++;
      }
    }
    return total / count;
  }

  /** Restarts at the first sample. */
  reset(): this {
    this.assertAlive();
    this.module._trainer_reset(this.pointer);
    checkError(this.module);
    this.indexValue = 0;
    return this;
  }

  /** Releases the trainer and its copied dataset. */
  destroy(): void {
    if (this.destroyed) {
      return;
    }
    this.destroyed = true;
    this.module._destroy_trainer(this.pointer);
    this.module._free(this.inputsPointer);
    this.module._free(this.outputsPointer);
  }

  private assertAlive(): void {
    if (this.destroyed) {
      throw new Error("This trainer has already been destroyed.");
    }
  }
}
