/**
 * The part of a trained hard circuit the compiler needs.
 *
 * `HardCircuit` from `@only-gates-ai/core` matches this structure, so callers
 * can pass hard circuits directly without importing the compiler's types.
 */
export interface HardCircuitSource {
  /** Layer sizes, from the input layer to the output layer. */
  readonly shape: readonly number[];
  /** Two previous-layer indices per cell, in layer order. */
  readonly connections: ArrayLike<number>;
  /** One gate index (0..15) per cell, in layer order. */
  readonly gates: ArrayLike<number>;
}

/** One label's ensemble of hard circuits. */
export interface ForestLabelSource {
  readonly id: string;
  readonly name: string;
  readonly color: string;
  readonly circuits: readonly HardCircuitSource[];
}

/**
 * The parallel axis used by the generated module.
 *
 * - `scalar` evaluates one sample per call and returns booleans as 0/1.
 * - `packed` is reserved for a future sample-parallel target.
 */
export type CompileTarget = "scalar" | "packed";

export interface ForestCompileOptions {
  /** Defaults to `scalar`. */
  readonly target?: CompileTarget;
  /** Name of the exported prediction function. Defaults to `predict`. */
  readonly functionName?: string;
}

export interface CompiledForest {
  readonly code: string;
  readonly target: CompileTarget;
  readonly labelCount: number;
  readonly circuitCount: number;
  /** Number of gates in the trained hard circuits. */
  readonly gateCount: number;
  /** Number of gate assignments left after AST simplification. */
  readonly emittedGateCount: number;
  readonly inputLength: number;
  readonly outputLength: number;
  readonly lineCount: number;
  readonly byteLength: number;
}
