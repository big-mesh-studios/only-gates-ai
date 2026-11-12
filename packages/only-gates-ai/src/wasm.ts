import createModule from "../wasm/only-gates-ai.js";
import createSimdModule from "../wasm/only-gates-ai-simd.js";
import type {
  EmscriptenModuleOptions,
  OnlyGatesWasmModule,
} from "./wasm-types.js";
import { OnlyGatesError, OnlyGatesErrorCode } from "./errors.js";

export type { EmscriptenModuleOptions, OnlyGatesWasmModule } from "./wasm-types.js";

/** Inference lane resolved at load time. */
export type OnlyGatesBackend = "wasm" | "wasm-simd";

export interface LoadedOnlyGatesModule {
  readonly module: OnlyGatesWasmModule;
  readonly backend: OnlyGatesBackend;
}

export async function loadOnlyGatesModule(
  options: EmscriptenModuleOptions = {},
): Promise<OnlyGatesWasmModule> {
  return (await createModule(options)) as OnlyGatesWasmModule;
}

/**
 * Loads the wasm SIMD lane. It exposes the same C API as the scalar module and
 * produces identical integer results; only the dot-product instructions differ.
 */
export async function loadOnlyGatesSimdModule(
  options: EmscriptenModuleOptions = {},
): Promise<OnlyGatesWasmModule> {
  return (await createSimdModule(options)) as OnlyGatesWasmModule;
}

/**
 * Loads the fastest available integer lane. Browsers that do not implement
 * wasm SIMD reject the module at instantiation, so the scalar lane is the
 * fallback without any separate feature test.
 */
export async function loadOnlyGatesModuleAuto(
  options: EmscriptenModuleOptions = {},
): Promise<LoadedOnlyGatesModule> {
  try {
    return {
      module: await loadOnlyGatesSimdModule(options),
      backend: "wasm-simd",
    };
  } catch {
    return {
      module: await loadOnlyGatesModule(options),
      backend: "wasm",
    };
  }
}

/** Throws if the last C call left an error code set. */
export function checkError(module: OnlyGatesWasmModule): void {
  const code = module._get_error() as OnlyGatesErrorCode;
  if (code !== OnlyGatesErrorCode.None) {
    throw new OnlyGatesError(code);
  }
}
