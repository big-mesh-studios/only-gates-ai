import { OnlyGatesError, OnlyGatesErrorCode } from "./errors.js";
import { HardCircuit } from "./hard-circuit.js";
import {
  allocate,
  readFloats,
  withMemory,
  writeBooleans,
  writeFloats,
} from "./memory.js";
import { fillRandomDistribution, mulberry32 } from "./random.js";
import { Trainer, type Dataset } from "./trainer.js";
import { checkError, type OnlyGatesWasmModule } from "./wasm.js";

/** A differentiable logic circuit. Create one with `OnlyGatesAI.createCircuit`. */
export class Circuit {
  private readonly module: OnlyGatesWasmModule;
  private readonly pointerValue: number;
  private destroyed = false;

  /** @internal Use `OnlyGatesAI.createCircuit()` instead. */
  constructor(
    module: OnlyGatesWasmModule,
    pointer: number,
    readonly shape: readonly number[],
  ) {
    this.module = module;
    this.pointerValue = pointer;
  }

  /** Raw pointer, for advanced interop with the C API. */
  get pointer(): number {
    return this.pointerValue;
  }

  get inputsLength(): number {
    return this.shape[0]!;
  }

  get outputsLength(): number {
    return this.shape[this.shape.length - 1]!;
  }

  get cellCount(): number {
    return this.module._circuit_get_cell_len(this.pointerValue);
  }

  get connectionProbabilitiesLength(): number {
    return this.module._circuit_get_connection_probabilities_len(
      this.pointerValue,
    );
  }

  /**
   * Replaces all wire and gate probabilities with random normalized
   * distributions. Pass `seed` for reproducible initialisation.
   */
  randomize(seed = Math.floor(Math.random() * 0x100000000) >>> 0): this {
    this.assertAlive();
    const random = mulberry32(seed);
    const heap = this.module.HEAPF32;
    const connectionLength = this.connectionProbabilitiesLength;
    if (connectionLength > 0) {
      const connectionPointer =
        this.module._circuit_get_connection_probabilities(this.pointerValue);
      checkError(this.module);
      const base = connectionPointer >>> 2;
      let offset = 0;
      for (let layer = 1; layer < this.shape.length; layer++) {
        const previousLength = this.shape[layer - 1]!;
        const currentLength = this.shape[layer]!;
        for (let pin = 0; pin < currentLength * 2; pin++) {
          fillRandomDistribution(
            heap,
            base + offset,
            previousLength,
            random,
          );
          offset += previousLength;
        }
      }
    }

    for (let cell = 0; cell < this.cellCount; cell++) {
      const gatePointer = this.module._circuit_get_cell_gate_probabilities(
        this.pointerValue,
        cell,
      );
      checkError(this.module);
      fillRandomDistribution(heap, gatePointer >>> 2, 16, random);
    }
    return this;
  }

  setInput(input: ArrayLike<boolean>): this {
    this.assertAlive();
    if (input.length !== this.inputsLength) {
      throw new RangeError(
        `Expected ${this.inputsLength} inputs, received ${input.length}.`,
      );
    }
    withMemory(this.module, this.inputsLength, (pointer) => {
      writeBooleans(this.module, pointer, input);
      this.module._circuit_set_input(this.pointerValue, pointer);
      checkError(this.module);
    });
    return this;
  }

  feedForward(): this {
    this.assertAlive();
    this.module._circuit_feed_forward(this.pointerValue);
    checkError(this.module);
    return this;
  }

  getOutput(): number[] {
    this.assertAlive();
    return withMemory(
      this.module,
      this.outputsLength * Float32Array.BYTES_PER_ELEMENT,
      (pointer) => {
        this.module._circuit_read_output(this.pointerValue, pointer);
        checkError(this.module);
        return Array.from(
          readFloats(this.module, pointer, this.outputsLength),
        );
      },
    );
  }

  /** All wire probabilities, copied out of WebAssembly memory. */
  getConnections(): Float32Array {
    this.assertAlive();
    const length = this.connectionProbabilitiesLength;
    if (length === 0) {
      return new Float32Array(0);
    }
    const pointer = this.module._circuit_get_connection_probabilities(
      this.pointerValue,
    );
    checkError(this.module);
    return readFloats(this.module, pointer, length);
  }

  /** Replaces all wire probabilities (length must match exactly). */
  setConnections(values: ArrayLike<number>): this {
    this.assertAlive();
    const length = this.connectionProbabilitiesLength;
    if (values.length !== length) {
      throw new RangeError(
        `Expected ${length} connection probabilities, received ${values.length}.`,
      );
    }
    if (length === 0) {
      return this;
    }
    const pointer = this.module._circuit_get_connection_probabilities(
      this.pointerValue,
    );
    checkError(this.module);
    writeFloats(this.module, pointer, values);
    return this;
  }

  /** The 16 gate probabilities for one cell. */
  getGates(cell: number): Float32Array {
    this.assertAlive();
    this.assertCell(cell);
    const pointer = this.module._circuit_get_cell_gate_probabilities(
      this.pointerValue,
      cell,
    );
    checkError(this.module);
    return readFloats(this.module, pointer, 16);
  }

  /** Replaces all 16 gate probabilities for one cell. */
  setGates(cell: number, values: ArrayLike<number>): this {
    this.assertAlive();
    this.assertCell(cell);
    if (values.length !== 16) {
      throw new RangeError(
        `Expected 16 gate probabilities, received ${values.length}.`,
      );
    }
    const pointer = this.module._circuit_get_cell_gate_probabilities(
      this.pointerValue,
      cell,
    );
    checkError(this.module);
    writeFloats(this.module, pointer, values);
    return this;
  }

  /** Activations of every layer, copied out of WebAssembly memory. */
  getActivations(): Float32Array {
    this.assertAlive();
    const length = this.module._circuit_get_activations_len(
      this.pointerValue,
    );
    const pointer = this.module._circuit_get_activations(this.pointerValue);
    checkError(this.module);
    return readFloats(this.module, pointer, length);
  }

  /** Runs one back-propagation step. `error` is `output - target`. */
  backpropagate(error: ArrayLike<number>, learningRate: number): this {
    this.assertAlive();
    if (error.length !== this.outputsLength) {
      throw new RangeError(
        `Expected ${this.outputsLength} error values, received ${error.length}.`,
      );
    }
    withMemory(
      this.module,
      this.outputsLength * Float32Array.BYTES_PER_ELEMENT,
      (pointer) => {
        writeFloats(this.module, pointer, error);
        this.module._circuit_back_propergate(
          this.pointerValue,
          pointer,
          learningRate,
        );
        checkError(this.module);
      },
    );
    return this;
  }

  /** Creates a deterministic circuit using the highest-probability choices. */
  createHardCircuit(): HardCircuit {
    this.assertAlive();
    const pointer = this.module._create_hard_circuit(this.pointerValue);
    if (pointer === 0) {
      checkError(this.module);
      throw new OnlyGatesError(OnlyGatesErrorCode.InvalidParams);
    }
    checkError(this.module);
    return new HardCircuit(this.module, pointer, [...this.shape]);
  }

  /**
   * Copies a dataset into WebAssembly memory and returns a step-able trainer.
   * Each input sample has `inputsLength` booleans; each output sample has
   * `outputsLength` numbers.
   */
  createTrainer(dataset: Dataset, learningRate = 0.1): Trainer {
    this.assertAlive();
    if (!Number.isFinite(learningRate)) {
      throw new RangeError("learningRate must be a finite number.");
    }
    if (
      dataset === null ||
      typeof dataset !== "object" ||
      !Array.isArray(dataset.inputs) ||
      !Array.isArray(dataset.outputs)
    ) {
      throw new TypeError(
        "The dataset must have `inputs` and `outputs` arrays.",
      );
    }

    const sampleCount = dataset.inputs.length;
    if (sampleCount === 0) {
      throw new RangeError("The dataset must contain at least one sample.");
    }
    if (dataset.outputs.length !== sampleCount) {
      throw new RangeError(
        `Expected ${sampleCount} output samples, received ${dataset.outputs.length}.`,
      );
    }

    const inputLength = this.inputsLength;
    const outputLength = this.outputsLength;
    const flatInputs = new Uint8Array(sampleCount * inputLength);
    const flatOutputs = new Float32Array(sampleCount * outputLength);

    for (let sample = 0; sample < sampleCount; sample++) {
      const sampleInputs = dataset.inputs[sample]!;
      const sampleOutputs = dataset.outputs[sample]!;
      if (sampleInputs.length !== inputLength) {
        throw new RangeError(
          `Sample ${sample} has ${sampleInputs.length} inputs, expected ${inputLength}.`,
        );
      }
      if (sampleOutputs.length !== outputLength) {
        throw new RangeError(
          `Sample ${sample} has ${sampleOutputs.length} outputs, expected ${outputLength}.`,
        );
      }
      for (let i = 0; i < inputLength; i++) {
        flatInputs[sample * inputLength + i] = sampleInputs[i] ? 1 : 0;
      }
      flatOutputs.set(sampleOutputs, sample * outputLength);
    }

    const inputsPointer = allocate(this.module, flatInputs.length);
    let outputsPointer: number;
    try {
      outputsPointer = allocate(this.module, flatOutputs.byteLength);
    } catch (error) {
      this.module._free(inputsPointer);
      throw error;
    }

    try {
      this.module.HEAPU8.set(flatInputs, inputsPointer);
      this.module.HEAPF32.set(flatOutputs, outputsPointer >>> 2);
      const pointer = this.module._create_trainer(
        this.pointerValue,
        inputsPointer,
        outputsPointer,
        sampleCount,
        learningRate,
      );
      if (pointer === 0) {
        checkError(this.module);
        throw new OnlyGatesError(OnlyGatesErrorCode.InvalidParams);
      }
      checkError(this.module);
      return new Trainer(
        this.module,
        pointer,
        sampleCount,
        inputsPointer,
        outputsPointer,
        learningRate,
      );
    } catch (error) {
      this.module._free(inputsPointer);
      this.module._free(outputsPointer);
      throw error;
    }
  }

  destroy(): void {
    if (this.destroyed) {
      return;
    }
    this.destroyed = true;
    this.module._destroy_circuit(this.pointerValue);
  }

  private assertAlive(): void {
    if (this.destroyed) {
      throw new Error("This circuit has already been destroyed.");
    }
  }

  private assertCell(cell: number): void {
    if (!Number.isInteger(cell) || cell < 0 || cell >= this.cellCount) {
      throw new RangeError(
        `Cell index ${cell} is out of range (0..${this.cellCount - 1}).`,
      );
    }
  }
}
