import type { AstNode } from "./ast.js";

export interface EmittedCircuit {
  readonly lines: readonly string[];
  readonly emittedGates: number;
}

/**
 * Emits the body of one circuit function from its simplified AST.
 *
 * A post-order walk assigns one variable to every non-leaf node. Shared nodes
 * are emitted once, and nodes that do not reach the output are never visited,
 * so this also performs common-subexpression elimination and dead-code
 * elimination.
 */
export function emitCircuitBody(root: AstNode): EmittedCircuit {
  const order: AstNode[] = [];
  const seen = new Set<AstNode>();

  const visit = (node: AstNode): void => {
    if (seen.has(node)) {
      return;
    }
    seen.add(node);
    if (node.kind === "not") {
      visit(node.value);
    } else if (node.kind === "and" || node.kind === "or" || node.kind === "xor") {
      visit(node.left);
      visit(node.right);
    }
    order.push(node);
  };

  visit(root);

  const names = new Map<AstNode, string>();
  const lines: string[] = [];
  let emittedGates = 0;

  for (const node of order) {
    if (node.kind === "const" || node.kind === "input") {
      continue;
    }

    const name = `v${emittedGates}`;
    emittedGates++;
    names.set(node, name);

    if (node.kind === "not") {
      lines.push(`  const ${name} = (~${operand(node.value, names)}) & 1;`);
    } else {
      const operator =
        node.kind === "and" ? "&" : node.kind === "or" ? "|" : "^";
      lines.push(
        `  const ${name} = (${operand(node.left, names)} ${operator} ${operand(
          node.right,
          names,
        )}) & 1;`,
      );
    }
  }

  lines.push(`  return ${operand(root, names)};`);
  return { lines, emittedGates };
}

function operand(
  node: AstNode,
  names: ReadonlyMap<AstNode, string>,
): string {
  if (node.kind === "const") {
    return node.value === 1 ? "1" : "0";
  }
  if (node.kind === "input") {
    return `input[${node.index}]`;
  }
  const name = names.get(node);
  if (name === undefined) {
    throw new Error("Internal error: AST node was not emitted.");
  }
  return name;
}
