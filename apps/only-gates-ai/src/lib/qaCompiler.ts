import runtimeSource from "./qaRuntime.ts?raw";
import { BUNDLED_QA_LEXICON } from "./qaAssets.ts";
import {
  compileQuestionAnswererCode,
  type CompiledQuestionAnswerer,
} from "./qaCodegen.ts";
import type { QaDocument } from "./qaRuntime.ts";

export interface CompileQuestionAnswererOptions {
  readonly documents: readonly QaDocument[];
  readonly suggestions?: readonly string[];
  readonly name: string;
  readonly createdAt?: string;
  readonly updatedAt?: string;
}

/** Compiles the in-app QA model into a standalone TypeScript API module. */
export function compileQuestionAnswerer(
  options: CompileQuestionAnswererOptions,
): CompiledQuestionAnswerer {
  return compileQuestionAnswererCode({
    ...options,
    lexicon: BUNDLED_QA_LEXICON,
    runtimeSource,
  });
}

export type { CompiledQuestionAnswerer };
