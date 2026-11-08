import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import ts from "typescript";

/*
 * Integration test for the integer-only WebAssembly path.
 *
 * The app sources are TypeScript with bundler-style imports, so this test
 * transpiles the two modules it needs and supplies their imports directly.
 * The float32 TS path is the reference distribution.
 */

const here = new URL(".", import.meta.url);
const tinySource = await readFile(new URL("../src/lib/tinyGpt.ts", here), "utf8");
const wasmSource = await readFile(
  new URL("../src/lib/tinyGptWasm.ts", here),
  "utf8",
);
const checkpointSource = await readFile(
  new URL("../src/assets/tiny-gpt-weights.ts", here),
  "utf8",
);
const core = await import(
  new URL("../../../packages/only-gates-ai/dist/index.js", here).href
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

const tiny = evaluateTypeScript(tinySource, () => ({}));
const checkpoint = evaluateTypeScript(
  checkpointSource,
  (id) => (id.includes("tinyGpt") ? tiny : {}),
).TINY_GPT_CHECKPOINT;
const wasm = evaluateTypeScript(wasmSource, (id) => {
  if (id === "@only-gates-ai/core") return core;
  if (id === "./tinyGpt" || id === "./tinyGpt.ts") return tiny;
  throw new Error(`unexpected import ${id}`);
});

const reference = tiny.loadTinyGpt(checkpoint);
const integer = await wasm.loadTinyGptWasm(checkpoint);

const prompts = [
  "Alice was",
  "the rabbit",
  "she was",
  "down the rabbit hole",
  "the door",
  "white rabbit",
  "alice said",
  "in the garden",
  "the queen",
  "i think",
  "the cat",
  "oh dear",
  "what is",
  "the pool of tears",
  "she had",
  "the mouse",
  "the duchess",
  "in a minute",
  "the key",
  "poor alice",
];

test("WASM integer logits are Q16 and finite", () => {
  const window = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
  integer.model.forward(window);
  const logits = integer.model.logits();
  assert.equal(logits.length, integer.config.context * integer.config.vocabSize);
  for (const value of logits) {
    assert.ok(Number.isInteger(value));
    assert.ok(value > -1_000_000 && value < 1_000_000);
  }
  assert.ok(logits.some((value) => value !== 0));
});

test("WASM probabilities match the float32 baseline across prompts", () => {
  let topOneMatches = 0;
  let meanDeltaTotal = 0;
  let maxDelta = 0;

  for (const prompt of prompts) {
    const comparison = wasm.compareTinyGptWasm(integer, reference, prompt);
    if (comparison.topOneMatches) topOneMatches++;
    meanDeltaTotal += comparison.meanProbabilityDelta;
    maxDelta = Math.max(maxDelta, comparison.maxProbabilityDelta);
  }

  const meanDelta = meanDeltaTotal / prompts.length;
  assert.ok(
    topOneMatches >= 18,
    `top-1 agreement ${topOneMatches}/${prompts.length} below 18`,
  );
  assert.ok(meanDelta <= 1e-3, `mean |dp| ${meanDelta} above 1e-3`);
  assert.ok(maxDelta <= 0.2, `max |dp| ${maxDelta} above 0.2`);
  assert.ok(meanDelta < 5e-3);
});

test("WASM generation is deterministic and in-vocabulary", () => {
  const first = wasm.generateTinyGptWasm(integer, "Alice was", {
    maxTokens: 12,
    temperature: 0.8,
    topK: 40,
    seed: 12345,
  });
  const second = wasm.generateTinyGptWasm(integer, "Alice was", {
    maxTokens: 12,
    temperature: 0.8,
    topK: 40,
    seed: 12345,
  });

  assert.deepEqual(first.tokenIds, second.tokenIds);
  assert.equal(first.steps, first.newTokens.length);
  assert.ok(first.steps > 0);
  for (const id of first.tokenIds) {
    assert.ok(Number.isInteger(id));
    assert.ok(id >= 0 && id < integer.config.vocabSize);
  }
  assert.ok(first.text.length > 0);
});

test("WASM sampler respects the seed and temperature", () => {
  const window = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
  integer.model.forward(window);

  integer.model.setSeed(7);
  const a = integer.model.sample(0.8, 40);
  integer.model.setSeed(7);
  const b = integer.model.sample(0.8, 40);
  assert.equal(a, b);
  assert.ok(a >= 0 && a < integer.config.vocabSize);

  const greedyLow = integer.model.sample(0.01, 1);
  assert.ok(greedyLow >= 0 && greedyLow < integer.config.vocabSize);
});

test.after(() => {
  integer.model.destroy();
});
