import { type AstNode, inputNode, Simplifier } from "./ast.js";
import { emitCircuitBody } from "./codegen.js";
import { gateAst } from "./gates.js";
import type {
  CompiledForest,
  ForestCompileOptions,
  ForestLabelSource,
  HardCircuitSource,
} from "./types.js";

interface CircuitPlan {
  readonly shape: readonly number[];
  readonly gateCount: number;
  readonly root: AstNode;
}

interface GeneratedCode {
  readonly code: string;
  readonly emittedGateCount: number;
}

const JS_KEYWORDS = new Set([
  "await",
  "break",
  "case",
  "catch",
  "class",
  "const",
  "continue",
  "debugger",
  "default",
  "delete",
  "do",
  "else",
  "enum",
  "export",
  "extends",
  "false",
  "finally",
  "for",
  "function",
  "if",
  "implements",
  "import",
  "in",
  "instanceof",
  "interface",
  "let",
  "new",
  "null",
  "package",
  "private",
  "protected",
  "public",
  "return",
  "static",
  "super",
  "switch",
  "this",
  "throw",
  "true",
  "try",
  "typeof",
  "var",
  "void",
  "while",
  "with",
  "yield",
]);

const RESERVED_NAMES = new Set([
  "INPUT_LENGTH",
  "LABELS",
  "CIRCUIT_LABELS",
  "CIRCUITS",
]);

function isPositiveInteger(value: number): boolean {
  return Number.isSafeInteger(value) && value > 0;
}

function validateFunctionName(name: string): void {
  if (!/^[A-Za-z_$][A-Za-z0-9_$]*$/.test(name)) {
    throw new Error(`"${name}" is not a valid JavaScript function name.`);
  }
  if (JS_KEYWORDS.has(name)) {
    throw new Error(`"${name}" is a reserved JavaScript keyword.`);
  }
  if (RESERVED_NAMES.has(name) || /^circuit\d+$/.test(name)) {
    throw new Error(`"${name}" is reserved by the generated module.`);
  }
}

/**
 * Compiles a whole hard-mode forest into a standalone ES module.
 *
 * The generated module exports `INPUT_LENGTH`, `LABELS` and a `predict(input)`
 * function. `predict` returns the same sorted vote/score objects as the app's
 * hard-mode inference.
 */
export function compileForest(
  labels: readonly ForestLabelSource[],
  options: ForestCompileOptions = {},
): CompiledForest {
  const target = options.target ?? "scalar";
  if (target !== "scalar") {
    throw new Error(
      'The "packed" target is not implemented yet. Use the "scalar" target.',
    );
  }

  const functionName = options.functionName ?? "predict";
  validateFunctionName(functionName);

  if (labels.length === 0) {
    throw new Error("The forest needs at least one label.");
  }

  const plans: CircuitPlan[][] = [];
  let circuitCount = 0;
  let gateCount = 0;
  let forestShape: readonly number[] | null = null;

  labels.forEach((label, labelIndex) => {
    if (typeof label.id !== "string" || label.id.length === 0) {
      throw new Error(`Label ${labelIndex + 1} needs a non-empty id.`);
    }
    if (typeof label.name !== "string" || label.name.length === 0) {
      throw new Error(`Label ${labelIndex + 1} needs a non-empty name.`);
    }
    if (typeof label.color !== "string" || label.color.length === 0) {
      throw new Error(`Label ${labelIndex + 1} needs a non-empty color.`);
    }
    if (label.circuits.length === 0) {
      throw new Error(`Label "${label.name}" has no circuits.`);
    }

    const labelPlans: CircuitPlan[] = [];
    for (let circuitIndex = 0; circuitIndex < label.circuits.length; circuitIndex++) {
      const circuit = label.circuits[circuitIndex]!;
      const plan = planCircuit(circuit, label.name, circuitIndex);

      if (forestShape === null) {
        forestShape = plan.shape;
      } else if (!sameShape(forestShape, plan.shape)) {
        throw new Error(
          `Label "${label.name}", circuit ${circuitIndex + 1}: all circuits in a forest must share the same shape (${forestShape.join(
            ", ",
          )} vs ${plan.shape.join(", ")}).`,
        );
      }

      gateCount += plan.gateCount;
      circuitCount++;
      labelPlans.push(plan);
    }
    plans.push(labelPlans);
  });

  const generated = generateCode(labels, plans, functionName);
  const shape = forestShape ?? [];

  return {
    code: generated.code,
    target,
    labelCount: labels.length,
    circuitCount,
    gateCount,
    emittedGateCount: generated.emittedGateCount,
    inputLength: shape[0] ?? 0,
    outputLength: shape[shape.length - 1] ?? 0,
    lineCount: generated.code.trimEnd().split("\n").length,
    byteLength: new TextEncoder().encode(generated.code).length,
  };
}

function sameShape(a: readonly number[], b: readonly number[]): boolean {
  if (a.length !== b.length) {
    return false;
  }
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) {
      return false;
    }
  }
  return true;
}

/**
 * Validates one hard circuit and turns it into a simplified logic AST.
 *
 * Gates are built into a raw AST first, then `Simplifier` maps over the whole
 * DAG. The simplifier applies constant folding, identity, idempotence,
 * complement, absorption, De Morgan and double-negation rules.
 */
function planCircuit(
  circuit: HardCircuitSource,
  labelName: string,
  circuitIndex: number,
): CircuitPlan {
  const prefix = `Label "${labelName}", circuit ${circuitIndex + 1}:`;
  const shape = Array.from(circuit.shape);

  if (shape.length < 2) {
    throw new Error(
      `${prefix} a circuit needs an input layer and an output layer.`,
    );
  }
  for (let layer = 0; layer < shape.length; layer++) {
    if (!isPositiveInteger(shape[layer]!)) {
      throw new Error(
        `${prefix} layer ${layer + 1} must be a positive whole number.`,
      );
    }
  }
  if (shape[shape.length - 1] !== 1) {
    throw new Error(
      `${prefix} the output layer must be 1, received ${shape[shape.length - 1]}.`,
    );
  }

  const connections = Array.from(circuit.connections);
  const gates = Array.from(circuit.gates);

  let expectedConnections = 0;
  let expectedGates = 0;
  for (let layer = 1; layer < shape.length; layer++) {
    expectedConnections += shape[layer]! * 2;
    expectedGates += shape[layer]!;
  }
  if (connections.length !== expectedConnections) {
    throw new Error(
      `${prefix} expected ${expectedConnections} connection${
        expectedConnections === 1 ? "" : "s"
      }, received ${connections.length}.`,
    );
  }
  if (gates.length !== expectedGates) {
    throw new Error(
      `${prefix} expected ${expectedGates} gate${
        expectedGates === 1 ? "" : "s"
      }, received ${gates.length}.`,
    );
  }

  const totalCells = expectedGates;
  const rawRoots: AstNode[] = new Array(totalCells);
  let cellOffset = 0;
  let connectionOffset = 0;
  let gateOffset = 0;

  for (let layer = 1; layer < shape.length; layer++) {
    const size = shape[layer]!;
    const previousSize = shape[layer - 1]!;
    const previousCellOffset = cellOffset - previousSize;

    for (let cell = 0; cell < size; cell++) {
      const connectionIndex = connectionOffset + cell * 2;
      const aIndex = connections[connectionIndex]!;
      const bIndex = connections[connectionIndex + 1]!;

      if (!Number.isInteger(aIndex) || aIndex < 0 || aIndex >= previousSize) {
        throw new Error(
          `${prefix} connection ${connectionIndex} is ${aIndex}, but layer ${layer} expects an index from 0 to ${
            previousSize - 1
          }.`,
        );
      }
      if (!Number.isInteger(bIndex) || bIndex < 0 || bIndex >= previousSize) {
        throw new Error(
          `${prefix} connection ${connectionIndex + 1} is ${bIndex}, but layer ${layer} expects an index from 0 to ${
            previousSize - 1
          }.`,
        );
      }

      const gate = gates[gateOffset + cell]!;
      if (!Number.isInteger(gate) || gate < 0 || gate > 15) {
        throw new Error(
          `${prefix} gate ${gateOffset + cell} is ${gate}, but gates must be 0..15.`,
        );
      }

      const aNode =
        layer === 1
          ? inputNode(aIndex)
          : rawRoots[previousCellOffset + aIndex]!;
      const bNode =
        layer === 1
          ? inputNode(bIndex)
          : rawRoots[previousCellOffset + bIndex]!;

      rawRoots[cellOffset + cell] = gateAst(gate, aNode, bNode);
    }

    cellOffset += size;
    connectionOffset += size * 2;
    gateOffset += size;
  }

  const outputRoot = rawRoots[totalCells - 1]!;
  const simplifier = new Simplifier();
  const root = simplifier.simplify(outputRoot);

  return { shape, gateCount: expectedGates, root };
}

function generateCode(
  labels: readonly ForestLabelSource[],
  plans: readonly (readonly CircuitPlan[])[],
  functionName: string,
): GeneratedCode {
  const bodies: string[][] = [];
  let emittedGateCount = 0;
  let circuitIndex = 0;

  for (const labelPlans of plans) {
    for (const plan of labelPlans) {
      const emitted = emitCircuitBody(plan.root);
      emittedGateCount += emitted.emittedGates;
      bodies.push([
        `function circuit${circuitIndex}(input) {`,
        ...emitted.lines,
        "}",
        "",
      ]);
      circuitIndex++;
    }
  }

  const shape = plans[0]?.[0]?.shape ?? [];
  const circuitCount = bodies.length;
  const lines: string[] = [];

  lines.push("// Generated by @only-gates-ai/compiler.");
  lines.push(
    `// Hard-mode gate forest: ${labels.length} label${
      labels.length === 1 ? "" : "s"
    }, ${circuitCount} circuit${circuitCount === 1 ? "" : "s"}, shape ${shape.join(
      " -> ",
    )}.`,
  );
  lines.push(
    `// ${sumPlans(plans)} trained gates simplified to ${emittedGateCount} emitted gate${
      emittedGateCount === 1 ? "" : "s"
    }.`,
  );
  lines.push(
    "// Each circuit is a pure boolean function. Gates use bitwise operators and",
  );
  lines.push("// every intermediate value is masked to 0 or 1.");
  lines.push("");
  lines.push(`export const INPUT_LENGTH = ${shape[0] ?? 0};`);
  lines.push("");
  lines.push("export const LABELS = [");
  for (const label of labels) {
    lines.push(
      `  { id: ${JSON.stringify(label.id)}, name: ${JSON.stringify(
        label.name,
      )}, color: ${JSON.stringify(label.color)} },`,
    );
  }
  lines.push("];");
  lines.push("");
  lines.push("const CIRCUIT_LABELS = [");

  const circuitLabels: number[] = [];
  plans.forEach((labelPlans, labelIndex) => {
    for (let i = 0; i < labelPlans.length; i++) {
      circuitLabels.push(labelIndex);
    }
  });
  for (let i = 0; i < circuitLabels.length; i += 16) {
    lines.push(`  ${circuitLabels.slice(i, i + 16).join(", ")},`);
  }
  lines.push("];");
  lines.push("");

  for (const body of bodies) {
    lines.push(...body);
  }

  lines.push("const CIRCUITS = [");
  const functionNames = Array.from(
    { length: circuitCount },
    (_, index) => `circuit${index}`,
  );
  for (let i = 0; i < functionNames.length; i += 8) {
    lines.push(`  ${functionNames.slice(i, i + 8).join(", ")},`);
  }
  lines.push("];");
  lines.push("");
  lines.push(`export function ${functionName}(input) {`);
  lines.push('  if (input == null || typeof input.length !== "number") {');
  lines.push(
    `    throw new TypeError("${functionName}(input) expects an array of booleans.");`,
  );
  lines.push("  }");
  lines.push("  if (input.length !== INPUT_LENGTH) {");
  lines.push(
    "    throw new RangeError(`Expected ${INPUT_LENGTH} inputs, received ${input.length}.`);",
  );
  lines.push("  }");
  lines.push("");
  lines.push("  const bits = new Uint8Array(INPUT_LENGTH);");
  lines.push("  for (let i = 0; i < INPUT_LENGTH; i++) {");
  lines.push("    bits[i] = input[i] ? 1 : 0;");
  lines.push("  }");
  lines.push("");
  lines.push("  const votes = new Array(LABELS.length).fill(0);");
  lines.push("  const totals = new Array(LABELS.length).fill(0);");
  lines.push("  for (let i = 0; i < CIRCUITS.length; i++) {");
  lines.push("    const output = CIRCUITS[i](bits);");
  lines.push("    const label = CIRCUIT_LABELS[i];");
  lines.push("    votes[label] += output;");
  lines.push("    totals[label] += 1;");
  lines.push("  }");
  lines.push("");
  lines.push("  const results = [];");
  lines.push("  for (let i = 0; i < LABELS.length; i++) {");
  lines.push("    const total = totals[i];");
  lines.push("    if (total === 0) {");
  lines.push("      continue;");
  lines.push("    }");
  lines.push("    results.push({");
  lines.push("      labelId: LABELS[i].id,");
  lines.push("      labelName: LABELS[i].name,");
  lines.push("      color: LABELS[i].color,");
  lines.push("      votes: votes[i],");
  lines.push("      totalCircuits: total,");
  lines.push("      score: votes[i] / total,");
  lines.push("    });");
  lines.push("  }");
  lines.push("");
  lines.push("  results.sort(");
  lines.push("    (a, b) =>");
  lines.push("      b.votes - a.votes ||");
  lines.push("      b.score - a.score ||");
  lines.push("      a.labelName.localeCompare(b.labelName),");
  lines.push("  );");
  lines.push("  return results;");
  lines.push("}");
  lines.push("");

  return { code: lines.join("\n"), emittedGateCount };
}

function sumPlans(plans: readonly (readonly CircuitPlan[])[]): number {
  let total = 0;
  for (const labelPlans of plans) {
    for (const plan of labelPlans) {
      total += plan.gateCount;
    }
  }
  return total;
}
