import { Circuit } from "./circuit.js";
import { OnlyGatesError, OnlyGatesErrorCode } from "./errors.js";
import { withMemory } from "./memory.js";
import {
  checkError,
  loadOnlyGatesModule,
  type EmscriptenModuleOptions,
  type OnlyGatesWasmModule,
} from "./wasm.js";

/**
 * Loaded only-gates-ai WebAssembly module. Use `createOnlyGatesAI()` to
 * initialise it, then create circuits from it.
 */
export class OnlyGatesAI {
  private constructor(private readonly module: OnlyGatesWasmModule) {}

  static async create(
    options: EmscriptenModuleOptions = {},
  ): Promise<OnlyGatesAI> {
    return new OnlyGatesAI(await loadOnlyGatesModule(options));
  }

  /** Underlying Emscripten module, for advanced interop. */
  get raw(): OnlyGatesWasmModule {
    return this.module;
  }

  /** Creates a circuit with the given layer sizes, e.g. `[2, 4, 1]`. */
  createCircuit(shape: readonly number[]): Circuit {
    if (!Array.isArray(shape) || shape.length === 0) {
      throw new OnlyGatesError(
        OnlyGatesErrorCode.InvalidShape,
        "A circuit needs at least one layer.",
      );
    }
    for (const size of shape) {
      if (!Number.isInteger(size) || size <= 0) {
        throw new OnlyGatesError(
          OnlyGatesErrorCode.InvalidShape,
          "Layer sizes must be positive integers.",
        );
      }
    }

    const shapeArray = Int32Array.from(shape);
    const pointer = withMemory(
      this.module,
      shapeArray.byteLength,
      (shapePointer) => {
        this.module.HEAP32.set(shapeArray, shapePointer >>> 2);
        return this.module._create_circuit(shapePointer, shapeArray.length);
      },
    );
    if (pointer === 0) {
      checkError(this.module);
      throw new OnlyGatesError(OnlyGatesErrorCode.InvalidShape);
    }
    checkError(this.module);
    return new Circuit(this.module, pointer, [...shape]);
  }
}

/** Loads the WebAssembly module and returns the high-level API. */
export async function createOnlyGatesAI(
  options: EmscriptenModuleOptions = {},
): Promise<OnlyGatesAI> {
  return OnlyGatesAI.create(options);
}
