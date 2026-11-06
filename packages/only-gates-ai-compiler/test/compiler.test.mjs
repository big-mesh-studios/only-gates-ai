import assert from "node:assert/strict";
import test from "node:test";

import { compileForest, evaluateGate } from "../dist/index.js";

const GATE_TABLES = [
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

function label(name, circuits, id = name.toLowerCase()) {
  return { id, name, color: "#7ee787", circuits };
}

function circuit(shape, connections, gates) {
  return { shape, connections, gates };
}

/** Evaluates the generated ES module without writing a file. */
function evaluateModule(code) {
  const source = code.replace(/^export /gm, "");
  return new Function(
    `${source}\nreturn { INPUT_LENGTH, LABELS, predict };`,
  )();
}

function referencePredict(labels, input) {
  const results = [];

  for (const group of labels) {
    let votes = 0;
    let score = 0;
    let evaluated = 0;

    for (const model of group.circuits) {
      const { shape, connections, gates } = model;
      if (shape[0] !== input.length) {
        continue;
      }

      let previous = input.map(Boolean);
      let connectionOffset = 0;
      let gateOffset = 0;

      for (let layer = 1; layer < shape.length; layer++) {
        const size = shape[layer];
        const current = new Array(size);
        for (let cell = 0; cell < size; cell++) {
          const a = previous[connections[connectionOffset + cell * 2]];
          const b = previous[connections[connectionOffset + cell * 2 + 1]];
          current[cell] = GATE_TABLES[gates[gateOffset + cell]](a, b);
        }
        previous = current;
        connectionOffset += size * 2;
        gateOffset += size;
      }

      const output = previous[0] ? 1 : 0;
      score += output;
      if (output >= 0.5) {
        votes++;
      }
      evaluated++;
    }

    if (evaluated > 0) {
      results.push({
        labelId: group.id,
        labelName: group.name,
        color: group.color,
        votes,
        totalCircuits: evaluated,
        score: score / evaluated,
      });
    }
  }

  results.sort(
    (a, b) =>
      b.votes - a.votes ||
      b.score - a.score ||
      a.labelName.localeCompare(b.labelName),
  );
  return results;
}

function makeRandom(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

function randomCircuit(random, shape) {
  const connections = [];
  const gates = [];

  for (let layer = 1; layer < shape.length; layer++) {
    const previous = shape[layer - 1];
    const size = shape[layer];
    for (let cell = 0; cell < size; cell++) {
      connections.push(Math.floor(random() * previous));
      connections.push(Math.floor(random() * previous));
      gates.push(Math.floor(random() * 16));
    }
  }

  return circuit(shape, connections, gates);
}

test("evaluates all sixteen gates", () => {
  for (let gate = 0; gate < 16; gate++) {
    const code = compileForest([
      label("Gate", [circuit([2, 1], [0, 1], [gate])]),
    ]).code;
    const { predict } = evaluateModule(code);

    for (let mask = 0; mask < 4; mask++) {
      const a = Boolean(mask & 2);
      const b = Boolean(mask & 1);
      const expected = GATE_TABLES[gate](a, b);
      const result = predict([a, b]);

      assert.equal(result.length, 1);
      assert.equal(result[0].votes, expected ? 1 : 0);
      assert.equal(result[0].score, expected ? 1 : 0);
      assert.equal(expected, evaluateGate(gate, a, b));
    }
  }
});

test("matches the hard-mode reference for a random forest", () => {
  const random = makeRandom(123456);
  const labels = [
    label(
      "Alpha",
      Array.from({ length: 3 }, () => randomCircuit(random, [4, 3, 1])),
      "alpha",
    ),
    label(
      "Beta",
      Array.from({ length: 3 }, () => randomCircuit(random, [4, 3, 1])),
      "beta",
    ),
    label(
      "Gamma",
      Array.from({ length: 2 }, () => randomCircuit(random, [4, 3, 1])),
      "gamma",
    ),
  ];

  const { predict } = evaluateModule(compileForest(labels).code);

  for (let mask = 0; mask < 16; mask++) {
    const input = [0, 1, 2, 3].map((bit) => Boolean(mask & (1 << bit)));
    assert.deepEqual(predict(input), referencePredict(labels, input));
  }
});

test("cancels two not gates and removes the emitted gate", () => {
  const compiled = compileForest([
    label("Not", [circuit([1, 1, 1], [0, 0, 0, 0], [3, 3])]),
  ]);
  const { predict } = evaluateModule(compiled.code);

  assert.equal(compiled.emittedGateCount, 0);
  assert.match(compiled.code, /return input\[0\];/);
  assert.equal(predict([true])[0].votes, 1);
  assert.equal(predict([false])[0].votes, 0);
});

test("cancels a not around a nand gate", () => {
  const compiled = compileForest([
    label("Nand", [circuit([2, 1, 1], [0, 1, 0, 0], [7, 3])]),
  ]);
  const { predict } = evaluateModule(compiled.code);

  assert.equal(compiled.emittedGateCount, 1);
  assert.match(compiled.code, /input\[0\] & input\[1\]/);
  for (let mask = 0; mask < 4; mask++) {
    const a = Boolean(mask & 2);
    const b = Boolean(mask & 1);
    assert.equal(predict([a, b])[0].votes, a && b ? 1 : 0);
  }
});

test("folds constants", () => {
  const compiled = compileForest([
    label("Constant", [circuit([1, 1, 1], [0, 0, 0, 0], [0, 8])]),
  ]);
  const { predict } = evaluateModule(compiled.code);

  assert.equal(compiled.emittedGateCount, 0);
  assert.match(compiled.code, /return 0;/);
  assert.equal(predict([true])[0].votes, 0);
});

test("uses idempotence and complement rules", () => {
  // xor(x, x) = 0.
  const same = compileForest([
    label("Same", [circuit([1, 1, 1], [0, 0, 0, 0], [3, 6])]),
  ]);
  assert.equal(same.emittedGateCount, 0);
  assert.equal(evaluateModule(same.code).predict([true])[0].votes, 0);

  // x & !x = 0.
  const complement = compileForest([
    label("Complement", [
      circuit([1, 2, 1], [0, 0, 0, 0, 0, 1], [12, 3, 8]),
    ]),
  ]);
  assert.equal(complement.emittedGateCount, 0);
  assert.equal(evaluateModule(complement.code).predict([true])[0].votes, 0);
  assert.equal(evaluateModule(complement.code).predict([false])[0].votes, 0);
});

test("applies absorption and dead-code elimination", () => {
  // layer 1: x, x & y, x | (x & y) = x
  // layer 2: x
  const compiled = compileForest([
    label("Absorption", [
      circuit(
        [3, 3, 1],
        [0, 1, 0, 1, 0, 1, 0, 0],
        [12, 8, 14, 12],
      ),
    ]),
  ]);
  const { predict } = evaluateModule(compiled.code);

  assert.equal(compiled.emittedGateCount, 0);
  assert.match(compiled.code, /return input\[0\];/);
  assert.equal(predict([true, true, false])[0].votes, 1);
  assert.equal(predict([false, true, false])[0].votes, 0);
});

test("shares common subexpressions before simplifying", () => {
  // layer 1: x & y, x & y
  // layer 2: (x & y) ^ (x & y) = 0, then 0
  // layer 3: 0
  const compiled = compileForest([
    label("Common", [
      circuit(
        [2, 2, 2, 1],
        [0, 1, 0, 1, 0, 1, 0, 0, 0, 0],
        [8, 8, 6, 12, 12],
      ),
    ]),
  ]);
  const { predict } = evaluateModule(compiled.code);

  assert.equal(compiled.emittedGateCount, 0);
  assert.match(compiled.code, /return 0;/);
  assert.equal(predict([true, true])[0].votes, 0);
  assert.equal(predict([true, false])[0].votes, 0);
});

test("reports compilation statistics", () => {
  const compiled = compileForest([
    label("A", [
      circuit([2, 2, 1], [0, 1, 0, 1, 0, 1], [8, 14, 14]),
      circuit([2, 2, 1], [0, 1, 1, 0, 0, 1], [6, 9, 8]),
    ]),
    label("B", [
      circuit([2, 2, 1], [0, 1, 0, 1, 0, 1], [1, 8, 6]),
    ]),
  ]);

  assert.equal(compiled.target, "scalar");
  assert.equal(compiled.labelCount, 2);
  assert.equal(compiled.circuitCount, 3);
  assert.equal(compiled.gateCount, 9);
  assert.ok(compiled.emittedGateCount <= compiled.gateCount);
  assert.equal(compiled.inputLength, 2);
  assert.equal(compiled.outputLength, 1);
  assert.ok(compiled.lineCount > 0);
  assert.ok(compiled.byteLength > 0);
  assert.match(compiled.code, /export function predict/);
  assert.match(compiled.code, /& 1/);
});

test("escapes label text in the generated module", () => {
  const name = 'He said "hi"\nand left \\o/';
  const compiled = compileForest([
    label(name, [circuit([1, 1], [0, 0], [12])], "quoted"),
  ]);
  const { LABELS, predict } = evaluateModule(compiled.code);

  assert.equal(LABELS[0].name, name);
  assert.equal(predict([false])[0].labelName, name);
});

test("rejects invalid forests", () => {
  assert.throws(() => compileForest([]), /at least one label/);

  assert.throws(
    () => compileForest([label("Empty", [])]),
    /has no circuits/,
  );

  assert.throws(
    () =>
      compileForest([
        label("NoOutput", [circuit([1], [], [])]),
      ]),
    /input layer and an output layer/,
  );

  assert.throws(
    () =>
      compileForest([
        label("MultiOutput", [circuit([2, 2], [0, 1, 0, 1], [8, 8])]),
      ]),
    /output layer must be 1/,
  );

  assert.throws(
    () =>
      compileForest([
        label("ShortConnections", [circuit([2, 1], [0], [8])]),
      ]),
    /expected 2 connections/,
  );

  assert.throws(
    () =>
      compileForest([
        label("ShortGates", [circuit([2, 1], [0, 1], [])]),
      ]),
    /expected 1 gate/,
  );

  assert.throws(
    () =>
      compileForest([
        label("BadWire", [circuit([2, 1], [0, 2], [8])]),
      ]),
    /index from 0 to 1/,
  );

  assert.throws(
    () =>
      compileForest([
        label("BadGate", [circuit([2, 1], [0, 1], [16])]),
      ]),
    /gates must be 0\.\.15/,
  );

  assert.throws(
    () =>
      compileForest([
        label("Mismatch", [
          circuit([2, 1], [0, 1], [8]),
          circuit([3, 1], [0, 1], [8]),
        ]),
      ]),
    /share the same shape/,
  );

  assert.throws(
    () =>
      compileForest(
        [label("Name", [circuit([2, 1], [0, 1], [8])])],
        { functionName: "class" },
      ),
    /reserved JavaScript keyword/,
  );

  assert.throws(
    () =>
      compileForest(
        [label("Name", [circuit([2, 1], [0, 1], [8])])],
        { functionName: "1bad" },
      ),
    /not a valid JavaScript function name/,
  );

  assert.throws(
    () =>
      compileForest(
        [label("Name", [circuit([2, 1], [0, 1], [8])])],
        { functionName: "CIRCUITS" },
      ),
    /reserved by the generated module/,
  );

  assert.throws(
    () =>
      compileForest(
        [label("Name", [circuit([2, 1], [0, 1], [8])])],
        { target: "packed" },
      ),
    /not implemented yet/,
  );
});
