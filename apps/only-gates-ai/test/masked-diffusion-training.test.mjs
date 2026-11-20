import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import test from "node:test";

import { createServer } from "vite";

/*
 * Masked-diffusion browser trainer smoke test.
 *
 * This runs the real TensorFlow.js trainer on the bundled Luau task subset and
 * FAQ, using a deliberately tiny bidirectional config so the CPU backend can
 * finish in a couple of steps. It covers:
 *
 *   - JSONL/FAQ loading and packing with the deployed tokenizer;
 *   - the absorbing-mask objective (bidirectional forward + masked loss);
 *   - Adam updates, validation, and checkpoint state;
 *   - the checkpoint/tokenizer/weights zip round-trip used by export/import.
 *
 * The browser-only pieces (IndexedDB, WebGL/WebGPU selection, UI) are covered
 * by the app build and manual checks.
 */

const appRoot = fileURLToPath(new URL("..", import.meta.url));
const publicRoot = fileURLToPath(new URL("../public/", import.meta.url));
const modelRoot = fileURLToPath(
  new URL("../public/models/masked-diffusion-luau/", import.meta.url),
);
const dataRoot = fileURLToPath(
  new URL("../public/data/luau-tasks/", import.meta.url),
);

async function withStubbedFetch(run) {
  const tokenizerJson = await readFile(`${modelRoot}tokenizer.json`, "utf8");
  const trainJsonl = await readFile(`${dataRoot}train-small.jsonl`, "utf8");
  const valJsonl = await readFile(`${dataRoot}val-small.jsonl`, "utf8");
  const previous = globalThis.fetch;

  globalThis.fetch = async (input) => {
    const url = typeof input === "string" ? input : input.url;
    if (url.endsWith("tokenizer.json")) {
      return new Response(tokenizerJson, {
        status: 200,
        headers: { "content-type": "application/json" },
      });
    }
    if (url.endsWith("train-small.jsonl")) {
      return new Response(trainJsonl, { status: 200 });
    }
    if (url.endsWith("val-small.jsonl")) {
      return new Response(valJsonl, { status: 200 });
    }
    throw new Error(`unexpected fetch in test: ${url}`);
  };

  try {
    return await run({ tokenizerJson });
  } finally {
    globalThis.fetch = previous;
  }
}

test("masked-diffusion trainer trains on bundled Luau data and round-trips a bundle", async () => {
  const server = await createServer({
    root: appRoot,
    appType: "custom",
    logLevel: "silent",
    server: { middlewareMode: true },
    optimizeDeps: { noDiscovery: true },
  });

  try {
    const trainer = await server.ssrLoadModule(
      "/src/lib/trainMaskedDiffusion.ts",
    );
    const bpe = await server.ssrLoadModule("/src/lib/tinyGptBpe.ts");
    const masked = await server.ssrLoadModule("/src/lib/maskedDiffusion.ts");

    await withStubbedFetch(async ({ tokenizerJson }) => {
      const tokenizer = bpe.parseBpeTokenizer(JSON.parse(tokenizerJson));
      assert.equal(tokenizer.vocab.length, 1722);

      const context = 64;
      const data = await trainer.loadMaskedDiffusionTrainingData({
        fullData: false,
        context,
        tokenizer,
      });
      assert.equal(data.info.fullData, false);
      assert.ok(data.info.taskRows > 0, "task rows were parsed");
      assert.ok(data.info.valRows > 0, "validation rows were parsed");
      assert.ok(data.info.faqPairs > 0, "FAQ pairs were parsed");
      assert.ok(data.info.trainExamples > 0, "training examples were packed");
      assert.ok(data.info.valExamples > 0, "validation examples were packed");
      assert.match(data.corpusHash, /^[0-9a-f]{8}$/);

      // A tiny bidirectional config keeps the CPU test fast while exercising
      // the same code path as the deployed 9.56M-parameter model.
      const config = {
        vocabSize: tokenizer.vocab.length,
        context,
        dim: 16,
        layers: 1,
        heads: 2,
        mlp: 32,
      };
      const parameterCount = masked.maskedDiffusionParameterCount(config);
      const initialState = {
        format: "only-gates-masked-diffusion-train",
        version: 1,
        config,
        step: 0,
        steps: 2,
        batch: 1,
        lr: 4e-4,
        dropout: 0,
        seed: 1,
        bestStep: 0,
        bestLoss: Number.POSITIVE_INFINITY,
        history: [],
        weights: new Float32Array(parameterCount),
        optimizer: null,
        bestWeights: null,
        corpusHash: data.corpusHash,
        elapsedMs: 0,
        savedAt: Date.now(),
        fullData: false,
        warmStart: false,
      };

      const progress = [];
      const checkpoints = [];
      const result = await trainer.trainMaskedDiffusion({
        initialState,
        steps: 2,
        batch: 1,
        dropout: 0,
        validationEvery: 1,
        maxValidationExamples: 1,
        yieldEvery: 1,
        saveEvery: 1,
        backend: "cpu",
        onProgress: (point) => progress.push(point),
        onCheckpoint: (state) => checkpoints.push(state),
      });

      assert.equal(result.backend, "cpu");
      assert.equal(result.state.step, 2);
      assert.equal(result.history.length, 2);
      assert.equal(progress.length, 2);
      assert.ok(checkpoints.length >= 1, "at least one checkpoint was emitted");
      assert.equal(result.state.weights.length, parameterCount);
      assert.equal(result.weights.length, parameterCount);
      assert.deepEqual(result.checkpoint.config, config);
      assert.equal(result.checkpoint.parameterCount, parameterCount);
      assert.equal(
        result.checkpoint.diffusion.maskTokenId,
        tokenizer.idFor("<|mask|>"),
      );
      assert.ok(
        result.history.some((point) => point.lr > 0),
        "the cosine schedule warms up above zero",
      );
      for (const point of result.history) {
        assert.ok(Number.isFinite(point.loss), "loss is finite");
        assert.ok(Number.isFinite(point.smoothedLoss), "smoothed loss is finite");
        assert.ok(point.lr >= 0, "learning rate is non-negative");
      }
      assert.ok(
        Number.isFinite(result.state.bestLoss),
        "validation produced a best loss",
      );

      // Export writes checkpoint.json, tokenizer.json, weights.bin, and a
      // training summary; import validates and restores the model payload.
      const bundle = await trainer.exportMaskedDiffusionBundle(result);
      assert.ok(bundle.zip.byteLength > 0);
      assert.match(bundle.filename, /^only-gates-masked-diffusion-.*\.zip$/);

      const imported = await trainer.importMaskedDiffusionBundle(bundle.zip);
      assert.deepEqual(imported.checkpoint.config, config);
      assert.equal(imported.tokenizer.vocab.length, tokenizer.vocab.length);
      assert.equal(imported.weights.length, parameterCount);
      assert.deepEqual(imported.weights, result.weights);

      // The public directory exists so the app can serve the same data.
      const served = await readFile(`${publicRoot}data/luau-tasks/train-small.jsonl`);
      assert.ok(served.byteLength > 0);
    });
  } finally {
    await server.close();
  }
});
