import type { OnlyGatesWasmModule } from "./wasm.js";
import { OnlyGatesError, OnlyGatesErrorCode } from "./errors.js";

export function allocate(
  module: OnlyGatesWasmModule,
  byteLength: number,
): number {
  const pointer = module._malloc(byteLength);
  if (pointer === 0) {
    throw new OnlyGatesError(OnlyGatesErrorCode.OutOfMemory);
  }
  return pointer;
}

export function withMemory<T>(
  module: OnlyGatesWasmModule,
  byteLength: number,
  callback: (pointer: number) => T,
): T {
  const pointer = allocate(module, byteLength);
  try {
    return callback(pointer);
  } finally {
    module._free(pointer);
  }
}

export function writeBooleans(
  module: OnlyGatesWasmModule,
  pointer: number,
  values: ArrayLike<boolean>,
): void {
  const heap = module.HEAPU8;
  for (let i = 0; i < values.length; i++) {
    heap[pointer + i] = values[i] ? 1 : 0;
  }
}

export function writeFloats(
  module: OnlyGatesWasmModule,
  pointer: number,
  values: ArrayLike<number>,
): void {
  module.HEAPF32.set(values, pointer >>> 2);
}

export function readFloats(
  module: OnlyGatesWasmModule,
  pointer: number,
  length: number,
): Float32Array {
  const start = pointer >>> 2;
  return module.HEAPF32.slice(start, start + length);
}

export function readInts(
  module: OnlyGatesWasmModule,
  pointer: number,
  length: number,
): Int32Array {
  const start = pointer >>> 2;
  return module.HEAP32.slice(start, start + length);
}
