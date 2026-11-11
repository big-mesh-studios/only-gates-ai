import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import ts from "typescript";

/*
 * Regression test for Solid 2 generation.
 *
 * Training right-pads fixed-length examples, so inference must right-pad too
 * and read logits at the last real prompt token. The textbook left-pad
 * convention shifts the prompt to the end of the window and samples from a
 * context position the model never saw during training.
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

function fakeModel() {
  const config = {
    vocabSize: 16,
    context: 8,
    dim: 8,
    layers: 1,
    heads: 1,
    mlp: 16,
  };
  let window = [];
  let logits = new Int32Array(config.context * config.vocabSize);

  return {
    config,
    tokenizer: fakeTokenizer(),
    get window() {
      return window;
    },
    get logitsBuffer() {
      return logits;
    },
    model: {
      setSeed() {},
      forward(tokens) {
        window = Array.from(tokens);
      },
      logits() {
        return logits;
      },
      sample() {
        throw new Error("generateSolid2Wasm must sample at the prompt position");
      },
    },
    setLogit(position, token) {
      logits[position * config.vocabSize + token] = 1_000_000;
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

test("Solid 2 generation right-pads the context window", () => {
  const m = fakeModel();
  solid2.generateSolid2Wasm(m, "hi", { maxTokens: 1, temperature: 1, topK: 0, repetitionPenalty: 1 });

  assert.deepEqual(m.window, [1, 5, 2, 0, 0, 0, 0, 0]);
});

test("Solid 2 generation samples from the last real prompt token", () => {
  const m = fakeModel();
  // The correct position (2) points at token 7; the final padded position
  // points at token 9, which is what the old left-pad code would have used.
  m.setLogit(2, 7);
  m.setLogit(7, 9);

  const result = solid2.generateSolid2Wasm(m, "hi", {
    maxTokens: 1,
    temperature: 1,
    topK: 0,
    repetitionPenalty: 1,
  });

  assert.equal(result.tokenIds[3], 7);
  assert.equal(result.text, "H");
});
