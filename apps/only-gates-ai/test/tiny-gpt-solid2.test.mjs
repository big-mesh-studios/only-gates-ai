import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import ts from "typescript";

/*
 * Regression tests for Solid 2 generation.
 *
 * Training right-pads fixed-length examples, but inference now pre-fills only
 * the real prompt tokens and reads logits at the last one. Causal attention
 * makes trailing pads irrelevant, so the KV-cache decode path must never pad,
 * and must never sample from a padded position.
 */

const here = new URL(".", import.meta.url);
const source = await readFile(
  new URL("../src/lib/tinyGptSolid2.ts", here),
  "utf8",
);

function evaluateTypeScript(source, requireFn) {
  const output = ts.transpileModule(source, {
    compilerOptions: {
      module: ts.ModuleKind.CommonJS,
      target: ts.ScriptTarget.ES2022,
    },
  }).outputText;
  const module = { exports: {} };
  new Function("exports", "require", "module", output)(
    module.exports,
    requireFn,
    module,
  );
  return module.exports;
}

const isStopToken = () => false;

function fakeTokenizer() {
  const ids = new Map([
    ["<|pad|>", 0],
    ["<|user|>", 1],
    ["<|assistant|>", 2],
    ["<|end|>", 3],
  ]);
  return {
    idFor: (token) => ids.get(token),
    encodeChatPrompt: () => [1, 5, 2],
    isStopToken,
    decodeGenerated: (_tokenizer, tokens) =>
      tokens.map((id) => String.fromCharCode(65 + id)).join(""),
  };
}

function fakeModel(overrides = {}) {
  const config = {
    vocabSize: 16,
    context: 8,
    dim: 8,
    layers: 1,
    heads: 1,
    mlp: 16,
    ...overrides,
  };
  const calls = [];
  const rows = new Map();
  const blank = new Int32Array(config.vocabSize);

  return {
    config,
    tokenizer: fakeTokenizer(),
    calls,
    model: {
      setSeed() {},
      resetCache() {
        calls.length = 0;
      },
      forwardRange(tokens, start, count) {
        calls.push({ start, count, ids: Array.from(tokens).slice(0, start + count) });
      },
      logitsRow(position) {
        return rows.get(position) ?? blank;
      },
      sample() {
        throw new Error("generateSolid2Wasm must sample at the prompt position");
      },
    },
    setRow(position, token, value = 1_000_000) {
      let row = rows.get(position);
      if (!row) {
        row = new Int32Array(config.vocabSize);
        rows.set(position, row);
      }
      row[token] = value;
    },
  };
}

const solid2 = evaluateTypeScript(source, (specifier) => {
  if (specifier === "@only-gates-ai/core") {
    return {
      createTinyGpt: () => {
        throw new Error("not used by this test");
      },
      loadOnlyGatesModule: () => {
        throw new Error("not used by this test");
      },
    };
  }
  if (specifier === "./tinyGptBpe") {
    return { ...fakeTokenizer() };
  }
  throw new Error(`unexpected import: ${specifier}`);
});

test("Solid 2 generation prefills only the real prompt tokens", () => {
  const m = fakeModel();
  solid2.generateSolid2Wasm(m, "hi", { maxTokens: 1, temperature: 1, topK: 0, repetitionPenalty: 1 });

  assert.deepEqual(m.calls.at(0), { start: 0, count: 3, ids: [1, 5, 2] });
});

test("Solid 2 generation samples from the last real prompt token", () => {
  const m = fakeModel();
  // Position 2 is the last prompt token. A padded window used to sample from a
  // pad position; the decode path must not.
  m.setRow(2, 7);
  m.setRow(7, 9);

  const result = solid2.generateSolid2Wasm(m, "hi", {
    maxTokens: 1,
    temperature: 1,
    topK: 0,
    repetitionPenalty: 1,
  });

  assert.equal(result.tokenIds[3], 7);
  assert.equal(result.text, "H");
});

test("Solid 2 generation computes one row per new token", () => {
  const m = fakeModel();
  m.setRow(2, 4);
  m.setRow(3, 5);

  const result = solid2.generateSolid2Wasm(m, "hi", {
    maxTokens: 2,
    temperature: 1,
    topK: 0,
    repetitionPenalty: 1,
  });

  assert.deepEqual(result.tokenIds, [1, 5, 2, 4, 5]);
  assert.deepEqual(
    m.calls.map(({ start, count }) => [start, count]),
    [
      [0, 3],
      [3, 1],
      [4, 1],
    ],
  );
});

test("Solid 2 generation stops at the context limit instead of sliding", () => {
  const m = fakeModel({ context: 5 });
  m.setRow(2, 4);
  m.setRow(3, 4);
  m.setRow(4, 4);

  const result = solid2.generateSolid2Wasm(m, "hi", {
    maxTokens: 100,
    temperature: 1,
    topK: 0,
    repetitionPenalty: 1,
  });

  assert.equal(result.tokenIds.length, 5);
  assert.equal(result.newTokenCount, 2);
  assert.equal(result.stopped, false);
});

test("Solid 2 generation emits one token from a full window, then stops", () => {
  const m = fakeModel({ context: 3 });
  m.setRow(2, 7);

  const result = solid2.generateSolid2Wasm(m, "hi", {
    maxTokens: 100,
    temperature: 1,
    topK: 0,
    repetitionPenalty: 1,
  });

  assert.equal(result.newTokenCount, 1);
  assert.equal(result.tokenIds[3], 7);
});

test("Stepwise decoding matches a one-shot run", () => {
  const options = { maxTokens: 3, temperature: 1, topK: 0, repetitionPenalty: 1 };
  const a = fakeModel();
  a.setRow(2, 6);
  a.setRow(3, 7);
  a.setRow(4, 8);
  const oneShot = solid2.generateSolid2Wasm(a, "hi", options);

  const b = fakeModel();
  b.setRow(2, 6);
  b.setRow(3, 7);
  b.setRow(4, 8);
  const decoder = solid2.createSolid2Decoder(b, "hi", options);
  const steps = [];
  for (;;) {
    const step = decoder.step();
    if (step.done) break;
    steps.push(step.token);
  }

  assert.deepEqual(steps, oneShot.tokenIds.slice(3));
  assert.deepEqual(decoder.result().tokenIds, oneShot.tokenIds);
});
