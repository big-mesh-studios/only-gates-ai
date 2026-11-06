# @only-gates-ai/compiler

Compiles a trained hard-mode gate forest into a standalone ES module of pure
bitwise logic. No WebAssembly, no weights, no dependencies.

```ts
import { compileForest } from "@only-gates-ai/compiler";

const compiled = compileForest(
  [
    {
      id: "cat",
      name: "Cat",
      color: "#7ee787",
      circuits: [
        { shape: [4, 1], connections: [0, 1], gates: [8] },
      ],
    },
  ],
  { functionName: "predict" },
);

console.log(compiled.code);
```

`circuits` accepts `HardCircuit` objects directly. The generated module exports:

- `INPUT_LENGTH` — the number of input bits.
- `LABELS` — `{ id, name, color }` for each label.
- `predict(input)` — returns the same sorted vote/score objects as the app's
  hard-mode inference.

## Pipeline

1. Validate the forest and build a boolean AST for every circuit. The AST is a
   DAG, so shared cells are shared nodes.
2. Map a `Simplifier` over the AST. It applies constant folding, identity,
   idempotence, complement, absorption, double negation, and one-directional
   De Morgan. Nodes are interned, which also gives common-subexpression
   elimination and commutative canonicalisation.
3. Emit code from the simplified AST. Only nodes reachable from the output are
   emitted, so dead gates disappear. Every intermediate value is masked with
   `& 1`, keeping the output exactly `0` or `1`.

Simplification is best-effort, not a SAT solver: it applies local rewrite rules
and does not prove arbitrary equivalent expressions.

## Targets

The API accepts `target: "scalar" | "packed"`. `scalar` evaluates one sample
per call. `packed` is reserved for a future sample-parallel backend and throws
`not implemented yet`.
