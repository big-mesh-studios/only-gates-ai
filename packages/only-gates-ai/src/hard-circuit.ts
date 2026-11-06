import type { OnlyGatesWasmModule } from "./wasm.js";
import { checkError } from "./wasm.js";
import { readFloats, withMemory, writeFloats } from "./memory.js";

/** Deterministic circuit produced by `Circuit.createHardCircuit()`. */
export class HardCircuit {
  private readonly module: OnlyGatesWasmModule;
  private readonly pointer: number;
  private destroyed = false;

  /** @internal Use `Circuit.createHardCircuit()` instead. */
  constructor(
    module: OnlyGatesWasmModule,
    pointer: number,
    readonly shape: readonly number[],
  ) {
    this.module = module;
    this.pointer = pointer;
  }

  get inputsLength(): number {
    return this.shape[0]!;
  }

  get outputsLength(): number {
    return this.shape[this.shape.length - 1]!;
  }

  /** Selected previous-layer index for each pin, two per gate. */
  get connections(): Int32Array {
    this.assertAlive();
    const length = this.module._hard_circuit_get_connections_len(this.pointer);
    const pointer = this.module._hard_circuit_get_connections(this.pointer);
    checkError(this.module);
    if (length === 0) {
      return new Int32Array(0);
    }
    const start = pointer >>> 2;
    return this.module.HEAP32.slice(start, start + length);
  }

  /** Selected gate index for every cell, including the output layer. */
  get gates(): Int32Array {
    this.assertAlive();
    const length = this.module._hard_circuit_get_gates_len(this.pointer);
    const pointer = this.module._hard_circuit_get_gates(this.pointer);
    checkError(this.module);
    if (length === 0) {
      return new Int32Array(0);
    }
    const start = pointer >>> 2;
    return this.module.HEAP32.slice(start, start + length);
  }

  setInput(input: ArrayLike<boolean>): this {
    this.assertAlive();
    if (input.length !== this.inputsLength) {
      throw new RangeError(
        `Expected ${this.inputsLength} inputs, received ${input.length}.`,
      );
    }
    withMemory(this.module, this.inputsLength, (pointer) => {
      const heap = this.module.HEAPU8;
      for (let i = 0; i < this.inputsLength; i++) {
        heap[pointer + i] = input[i] ? 1 : 0;
      }
      this.module._hard_circuit_set_input(this.pointer, pointer);
      checkError(this.module);
    });
    return this;
  }

  feedForward(): this {
    this.assertAlive();
    this.module._hard_circuit_feed_forward(this.pointer);
    checkError(this.module);
    return this;
  }

  getOutput(): number[] {
    this.assertAlive();
    const output = withMemory(
      this.module,
      this.outputsLength * Float32Array.BYTES_PER_ELEMENT,
      (pointer) => {
        this.module._hard_circuit_read_output(this.pointer, pointer);
        checkError(this.module);
        return Array.from(readFloats(this.module, pointer, this.outputsLength));
      },
    );
    return output;
  }

  getActivations(): Float32Array {
    this.assertAlive();
    const length = this.module._hard_circuit_get_activations_len(this.pointer);
    const pointer = this.module._hard_circuit_get_activations(this.pointer);
    checkError(this.module);
    return readFloats(this.module, pointer, length);
  }

  destroy(): void {
    if (this.destroyed) {
      return;
    }
    this.destroyed = true;
    this.module._destroy_hard_circuit(this.pointer);
  }

  private assertAlive(): void {
    if (this.destroyed) {
      throw new Error("This hard circuit has already been destroyed.");
    }
  }
}
