/**
 * A small boolean expression AST.
 *
 * Circuits are converted into this form before code generation. The AST is a
 * DAG: a cell's node is reused by every later cell that reads it, so mapping a
 * simplifier over the tree also shares identical subexpressions.
 */
export type AstNode = ConstNode | InputNode | NotNode | BinaryNode;

export interface ConstNode {
  readonly kind: "const";
  readonly value: 0 | 1;
}

export interface InputNode {
  readonly kind: "input";
  readonly index: number;
}

export interface NotNode {
  readonly kind: "not";
  readonly value: AstNode;
}

export interface BinaryNode {
  readonly kind: "and" | "or" | "xor";
  readonly left: AstNode;
  readonly right: AstNode;
}

export const CONST_ZERO: ConstNode = { kind: "const", value: 0 };
export const CONST_ONE: ConstNode = { kind: "const", value: 1 };

export function inputNode(index: number): InputNode {
  return { kind: "input", index };
}

export function notNode(value: AstNode): NotNode {
  return { kind: "not", value };
}

export function binaryNode(
  kind: "and" | "or" | "xor",
  left: AstNode,
  right: AstNode,
): BinaryNode {
  return { kind, left, right };
}

/**
 * Maps a simplifier over an AST and returns a canonical tree.
 *
 * Nodes are interned by a structural key, so `and(a, b)` and `and(b, a)` become
 * the same node and repeated subexpressions are shared. The `memo` table makes
 * the traversal linear in the number of raw AST nodes.
 */
export class Simplifier {
  private readonly interned = new Map<string, AstNode>();
  private readonly ids = new WeakMap<AstNode, number>();
  private readonly memo = new Map<AstNode, AstNode>();
  private nextId = 0;

  constructor() {
    this.intern(CONST_ZERO);
    this.intern(CONST_ONE);
  }

  simplify(node: AstNode): AstNode {
    const cached = this.memo.get(node);
    if (cached !== undefined) {
      return cached;
    }

    let result: AstNode;
    switch (node.kind) {
      case "const":
      case "input":
        result = this.intern(node);
        break;
      case "not":
        result = this.simplifyNot(this.simplify(node.value));
        break;
      case "and":
        result = this.simplifyAnd(
          this.simplify(node.left),
          this.simplify(node.right),
        );
        break;
      case "or":
        result = this.simplifyOr(
          this.simplify(node.left),
          this.simplify(node.right),
        );
        break;
      case "xor":
        result = this.simplifyXor(
          this.simplify(node.left),
          this.simplify(node.right),
        );
        break;
    }

    this.memo.set(node, result);
    return result;
  }

  private simplifyNot(value: AstNode): AstNode {
    if (value.kind === "const") {
      return value.value === 0 ? CONST_ONE : CONST_ZERO;
    }
    if (value.kind === "not") {
      return value.value;
    }
    // De Morgan is applied from simplifyAnd/simplifyOr, not here. Applying it
    // in both directions would loop forever: !(a & b) -> !a | !b -> !(a & b).
    return this.intern(notNode(value));
  }

  private simplifyAnd(a: AstNode, b: AstNode): AstNode {
    if (a.kind === "const") {
      return a.value === 0 ? CONST_ZERO : b;
    }
    if (b.kind === "const") {
      return b.value === 0 ? CONST_ZERO : a;
    }
    if (a === b) {
      return a;
    }
    if (isNotOf(a, b)) {
      return CONST_ZERO;
    }
    // Absorption: a & (a | b) = a.
    if (absorbsOr(b, a)) {
      return a;
    }
    if (absorbsOr(a, b)) {
      return b;
    }
    // De Morgan: !a & !b = !(a | b).
    if (a.kind === "not" && b.kind === "not") {
      return this.simplifyNot(this.simplifyOr(a.value, b.value));
    }
    return this.intern(binaryNode("and", a, b));
  }

  private simplifyOr(a: AstNode, b: AstNode): AstNode {
    if (a.kind === "const") {
      return a.value === 1 ? CONST_ONE : b;
    }
    if (b.kind === "const") {
      return b.value === 1 ? CONST_ONE : a;
    }
    if (a === b) {
      return a;
    }
    if (isNotOf(a, b)) {
      return CONST_ONE;
    }
    // Absorption: a | (a & b) = a.
    if (absorbsAnd(b, a)) {
      return a;
    }
    if (absorbsAnd(a, b)) {
      return b;
    }
    // De Morgan: !a | !b = !(a & b).
    if (a.kind === "not" && b.kind === "not") {
      return this.simplifyNot(this.simplifyAnd(a.value, b.value));
    }
    return this.intern(binaryNode("or", a, b));
  }

  private simplifyXor(a: AstNode, b: AstNode): AstNode {
    if (a.kind === "const") {
      return a.value === 0 ? b : this.simplifyNot(b);
    }
    if (b.kind === "const") {
      return b.value === 0 ? a : this.simplifyNot(a);
    }
    if (a === b) {
      return CONST_ZERO;
    }
    if (isNotOf(a, b)) {
      return CONST_ONE;
    }
    // !a ^ !b = a ^ b.
    if (a.kind === "not" && b.kind === "not") {
      return this.simplifyXor(a.value, b.value);
    }
    // !a ^ b = !(a ^ b) and a ^ !b = !(a ^ b).
    if (a.kind === "not") {
      return this.simplifyNot(this.simplifyXor(a.value, b));
    }
    if (b.kind === "not") {
      return this.simplifyNot(this.simplifyXor(a, b.value));
    }
    return this.intern(binaryNode("xor", a, b));
  }

  private intern(node: AstNode): AstNode {
    const key = this.keyOf(node);
    const existing = this.interned.get(key);
    if (existing !== undefined) {
      return existing;
    }
    this.interned.set(key, node);
    this.ids.set(node, this.nextId++);
    return node;
  }

  private keyOf(node: AstNode): string {
    switch (node.kind) {
      case "const":
        return `c${node.value}`;
      case "input":
        return `i${node.index}`;
      case "not":
        return `n${this.idOf(node.value)}`;
      case "and":
      case "or":
      case "xor": {
        const left = this.idOf(node.left);
        const right = this.idOf(node.right);
        // Canonicalise commutative operands so and(a, b) and and(b, a) share
        // the same interned node.
        const low = left < right ? left : right;
        const high = left < right ? right : left;
        return `${node.kind}${low},${high}`;
      }
    }
  }

  private idOf(node: AstNode): number {
    const id = this.ids.get(node);
    if (id === undefined) {
      throw new Error("Internal error: AST child was not interned.");
    }
    return id;
  }
}

function isNotOf(a: AstNode, b: AstNode): boolean {
  return (
    (a.kind === "not" && a.value === b) ||
    (b.kind === "not" && b.value === a)
  );
}

function absorbsOr(outer: AstNode, inner: AstNode): boolean {
  return (
    outer.kind === "or" && (outer.left === inner || outer.right === inner)
  );
}

function absorbsAnd(outer: AstNode, inner: AstNode): boolean {
  return (
    outer.kind === "and" && (outer.left === inner || outer.right === inner)
  );
}
