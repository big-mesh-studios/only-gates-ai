import createModule from "../wasm/only-gates-ai.js";
import type {
  EmscriptenModuleOptions,
  OnlyGatesWasmModule,
} from "./wasm-types.js";
import { OnlyGatesError, OnlyGatesErrorCode } from "./errors.js";

export type { EmscriptenModuleOptions, OnlyGatesWasmModule } from "./wasm-types.js";

export async function loadOnlyGatesModule(
  options: EmscriptenModuleOptions = {},
): Promise<OnlyGatesWasmModule> {
  return (await createModule(options)) as OnlyGatesWasmModule;
}

/** Throws if the last C call left an error code set. */
export function checkError(module: OnlyGatesWasmModule): void {
  const code = module._get_error() as OnlyGatesErrorCode;
  if (code !== OnlyGatesErrorCode.None) {
    throw new OnlyGatesError(code);
  }
}
