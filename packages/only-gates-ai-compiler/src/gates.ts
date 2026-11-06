import {
  type AstNode,
  binaryNode,
  CONST_ONE,
  CONST_ZERO,
  notNode,
} from "./ast.js";

/**
 * Gate order used by the C core and `HardCircuit.gates`.
 *
 * Keep this list in sync with `GateProbabilities` in `src/main.c`.
 */
export const GATE_NAMES = [
  "zero",
  "nor",
  "lnt",
  "not_a",
  "rnt",
  "not_b",
  "xor",
  "nand",
  "and",
  "xnor",
  "proj_b",
  "imp",
  "proj_a",
  "cimp",
  "or",
  "one",
] as const;

export type GateName = (typeof GATE_NAMES)[number];

type ExpressionBuilder = (a: string, b: string) => string;

/**
 * Scalar expressions for every gate. `a` and `b` are already 0 or 1, and each
 * result is masked back to 0 or 1 so the generated code stays pure logic.
 */
const SCALAR_EXPRESSIONS: readonly ExpressionBuilder[] = [
  () => "0", // zero
  (a, b) => `(~(${a} | ${b})) & 1`, // nor
  (a, b) => `(${a} & ~${b}) & 1`, // lnt: a && !b
  (a) => `(~${a}) & 1`, // not_a
  (a, b) => `(~${a} & ${b}) & 1`, // rnt: !a && b
  (_a, b) => `(~${b}) & 1`, // not_b
  (a, b) => `(${a} ^ ${b}) & 1`, // xor
  (a, b) => `(~(${a} & ${b})) & 1`, // nand
  (a, b) => `(${a} & ${b}) & 1`, // and
  (a, b) => `(~(${a} ^ ${b})) & 1`, // xnor
  (_a, b) => `(${b}) & 1`, // proj_b
  (a, b) => `((~${a}) | ${b}) & 1`, // imp: !a || b
  (a) => `(${a}) & 1`, // proj_a
  (a, b) => `(${a} | ~${b}) & 1`, // cimp: a || !b
  (a, b) => `(${a} | ${b}) & 1`, // or
  () => "1", // one
];

/** Truth tables for the same gate order, used by the simplifier. */
const GATE_TRUTH: readonly ((a: boolean, b: boolean) => boolean)[] = [
  () => false, // zero
  (a, b) => !(a || b), // nor
  (a, b) => a && !b, // lnt
  (a) => !a, // not_a
  (a, b) => !a && b, // rnt
  (_a, b) => !b, // not_b
  (a, b) => a !== b, // xor
  (a, b) => !(a && b), // nand
  (a, b) => a && b, // and
  (a, b) => a === b, // xnor
  (_a, b) => b, // proj_b
  (a, b) => !a || b, // imp
  (a) => a, // proj_a
  (a, b) => a || !b, // cimp
  (a, b) => a || b, // or
  () => true, // one
];

/** Builds the scalar JavaScript expression for one gate. */
export function gateExpression(gate: number, a: string, b: string): string {
  const expression = SCALAR_EXPRESSIONS[gate];
  if (expression === undefined) {
    throw new RangeError(`Unknown gate index ${gate}.`);
  }
  return expression(a, b);
}

/** Evaluates one gate with booleans. */
export function evaluateGate(gate: number, a: boolean, b: boolean): boolean {
  const truth = GATE_TRUTH[gate];
  if (truth === undefined) {
    throw new RangeError(`Unknown gate index ${gate}.`);
  }
  return truth(a, b);
}

/**
 * Builds the raw AST for one gate.
 *
 * Every gate is expressed with `and`, `or`, `xor` and `not` so the simplifier
 * only needs rules for those four operations.
 */
export function gateAst(gate: number, a: AstNode, b: AstNode): AstNode {
  switch (gate) {
    case 0: // zero
      return CONST_ZERO;
    case 1: // nor
      return notNode(binaryNode("or", a, b));
    case 2: // lnt: a && !b
      return binaryNode("and", a, notNode(b));
    case 3: // not_a
      return notNode(a);
    case 4: // rnt: !a && b
      return binaryNode("and", notNode(a), b);
    case 5: // not_b
      return notNode(b);
    case 6: // xor
      return binaryNode("xor", a, b);
    case 7: // nand
      return notNode(binaryNode("and", a, b));
    case 8: // and
      return binaryNode("and", a, b);
    case 9: // xnor
      return notNode(binaryNode("xor", a, b));
    case 10: // proj_b
      return b;
    case 11: // imp: !a || b
      return binaryNode("or", notNode(a), b);
    case 12: // proj_a
      return a;
    case 13: // cimp: a || !b
      return binaryNode("or", a, notNode(b));
    case 14: // or
      return binaryNode("or", a, b);
    case 15: // one
      return CONST_ONE;
    default:
      throw new RangeError(`Unknown gate index ${gate}.`);
  }
}
