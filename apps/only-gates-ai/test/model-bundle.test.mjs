import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import test from "node:test";

import { createServer } from "vite";

/*
 * Model bundle round-trip tests.
 *
 * The Alice test uses the real browser trainer on the CPU backend. The BPE
 * test builds a tiny Solid 2-shaped checkpoint directly, so it covers the
 * manifest/tokenizer/optimizer path without needing a GPU or the deployed
 * model files.
 */

const appRoot = fileURLToPath(new URL("..", import.meta.url));

const corpus = [
  "alice was beginning to get very tired of sitting by her sister.",
  "alice was beginning to get very tired of sitting by her sister on the bank.",
  "the white rabbit ran close by her and she followed it.",
  "the white rabbit ran close by her and she followed it down the hole.",
  "she said the rabbit was late and the rabbit said she was late.",
  "she said the rabbit was late and the rabbit said she was late again.",
].join("\n\n");

async function createTestServer() {
  return createServer({
    root: appRoot,
    appType: "custom",
    logLevel: "silent",
    server: { middlewareMode: true },
    optimizeDeps: { noDiscovery: true },
  });
}

test("Alice model bundle exports, imports, and resumes training", async () => {
  const server = await createTestServer();
  try {
    const trainer = await server.ssrLoadModule("/src/lib/trainTinyGpt.ts");
    const tiny = await server.ssrLoadModule("/src/lib/tinyGpt.ts");
    const bundle = await server.ssrLoadModule("/src/lib/modelBundle.ts");

    const config = {
      context: 4,
      dim: 8,
      layers: 1,
      heads: 2,
      mlp: 16,
    };
    const result = await trainer.trainTinyGpt({
      corpus,
      config,
      vocabLimit: 32,
      steps: 3,
      batch: 2,
      lr: 0.003,
      dropout: 0,
      seed: 7,
      backend: "cpu",
      yieldEvery: 1,
      saveEvery: 1,
    });

    const source = bundle.aliceBundleSource(result);
    const prepared = await bundle.createModelBundle(source, {
      includeResumeState: true,
    });

    assert.match(prepared.filename, /^only-gates-alice-.*\.zip$/);
    assert.equal(prepared.manifest.kind, "alice");
    assert.equal(prepared.manifest.target.example, "alice");
    assert.equal(prepared.manifest.model.tokenizer.type, "word");
    assert.equal(prepared.manifest.training.resumable, true);
    assert.ok(prepared.zip.byteLength > 0);
    assert.ok(prepared.manifest.integrity === undefined || /^[a-f0-9]{64}$/.test(prepared.manifest.integrity.weights));

    const imported = await bundle.importModelBundle(prepared.zip);
    assert.equal(imported.kind, "alice");
    assert.equal(imported.manifest.format, "only-gates-model-bundle");
    assert.equal(imported.checkpoint.config.vocabSize, result.checkpoint.config.vocabSize);
    assert.deepEqual(imported.vocab, result.checkpoint.vocab);
    assert.equal(
      imported.weights.length,
      imported.manifest.model.parameterCount,
    );
    assert.ok(imported.trainingState);
    assert.equal(imported.trainingState.step, 3);

    // The imported checkpoint must satisfy the existing loader.
    const model = tiny.loadTinyGpt(imported.checkpoint);
    assert.equal(model.config.vocabSize, result.checkpoint.config.vocabSize);

    const resumed = await trainer.trainTinyGpt({
      corpus,
      steps: 5,
      batch: 2,
      backend: "cpu",
      initialState: imported.trainingState,
      yieldEvery: 1,
      saveEvery: 1,
    });
    assert.equal(resumed.state.step, 5);
    assert.equal(resumed.history.length, 5);

    const inferenceOnly = await bundle.createModelBundle(source, {
      includeResumeState: false,
    });
    assert.equal(inferenceOnly.manifest.training.resumable, false);
    const importedInference = await bundle.importModelBundle(inferenceOnly.zip);
    assert.equal(importedInference.trainingState, null);
  } finally {
    await server.close();
  }
});

test("BPE model bundle round-trips tokenizer and optimizer state", async () => {
  const server = await createTestServer();
  try {
    const core = await server.ssrLoadModule("@only-gates-ai/core");
    const wasm = await server.ssrLoadModule("/src/lib/tinyGptWasm.ts");
    const bundle = await server.ssrLoadModule("/src/lib/modelBundle.ts");

    const specialTokens = [
      "<|pad|>",
      "<|user|>",
      "<|assistant|>",
      "<|end|>",
      "<|facts|>",
    ];
    const vocab = [...specialTokens];
    for (let index = 0; vocab.length < 24; index++) {
      vocab.push(`token${index}`);
    }
    const tokenizer = core.parseBpeModel({
      version: 1,
      specialTokens,
      vocab,
      merges: [],
    });

    const config = {
      vocabSize: vocab.length,
      context: 4,
      dim: 8,
      layers: 1,
      heads: 2,
      mlp: 16,
    };
    const weights = new Float32Array(wasm.tinyGptParameterCount(config));
    for (let index = 0; index < weights.length; index++) {
      weights[index] = Math.sin(index * 0.01) * 0.05;
    }
    const serialized = wasm.unpackTinyGptWeights(config, weights);
    const checkpoint = {
      format: "only-gates-tiny-gpt-solid2",
      version: 1,
      config,
      parameterCount: weights.length,
      tokenizer: "tokenizer.json",
      meta: { name: "bundle-test", source: "unit test", license: "test" },
      samples: [{ prompt: "hello", text: "world" }],
      prompts: ["hello"],
    };
    const state = {
      format: "only-gates-browser-train",
      version: 1,
      kind: "solid2",
      config,
      vocab,
      step: 7,
      steps: 20,
      batch: 2,
      lr: 0.001,
      dropout: 0.1,
      seed: 42,
      bestStep: 5,
      bestLoss: 1.25,
      history: [
        {
          step: 1,
          loss: 2,
          smoothedLoss: 2,
          lr: 0.001,
          elapsedMs: 10,
          valLoss: 1.5,
        },
      ],
      weights: serialized,
      optimizer: { m: serialized, v: serialized },
      bestWeights: serialized,
      corpusHash: "test-corpus",
      elapsedMs: 1000,
      savedAt: Date.now(),
    };

    const prepared = await bundle.createModelBundle(
      {
        kind: "solid2",
        checkpoint,
        tokenizer,
        wordVocab: null,
        weights,
        state,
        prompts: ["hello"],
        backend: "cpu",
      },
      { includeResumeState: true },
    );
    assert.equal(prepared.manifest.kind, "solid2");
    assert.equal(prepared.manifest.model.tokenizer.type, "bpe");
    assert.equal(prepared.manifest.training.assistant, undefined);

    const imported = await bundle.importModelBundle(prepared.zip);
    assert.equal(imported.kind, "solid2");
    assert.equal(imported.checkpoint.meta.name, "bundle-test");
    assert.equal(imported.tokenizer.size, vocab.length);
    assert.deepEqual(Array.from(imported.weights), Array.from(weights));
    assert.ok(imported.trainingState);
    assert.equal(imported.trainingState.step, 7);
    assert.ok(imported.trainingState.bestWeights);
    assert.deepEqual(
      Array.from(imported.trainingState.optimizer.m.tokenEmb),
      Array.from(serialized.tokenEmb),
    );
  } finally {
    await server.close();
  }
});

test("model bundle import rejects a bad manifest and a corrupt checksum", async () => {
  const server = await createTestServer();
  try {
    const trainer = await server.ssrLoadModule("/src/lib/trainTinyGpt.ts");
    const bundle = await server.ssrLoadModule("/src/lib/modelBundle.ts");
    const zip = await server.ssrLoadModule("/src/lib/zipArchive.ts");

    const result = await trainer.trainTinyGpt({
      corpus,
      config: { context: 4, dim: 8, layers: 1, heads: 2, mlp: 16 },
      vocabLimit: 32,
      steps: 2,
      batch: 2,
      backend: "cpu",
      yieldEvery: 1,
      saveEvery: 1,
    });
    const prepared = await bundle.createModelBundle(
      bundle.aliceBundleSource(result),
      { includeResumeState: false },
    );

    const files = await zip.readZip(prepared.zip);
    const manifest = JSON.parse(
      new TextDecoder().decode(files.get("manifest.json")),
    );
    manifest.format = "not-an-only-gates-bundle";
    files.set(
      "manifest.json",
      new TextEncoder().encode(JSON.stringify(manifest)),
    );
    const badManifest = await zip.createZip(
      Array.from(files, ([name, data]) => ({ name, data })),
    );
    await assert.rejects(
      () => bundle.importModelBundle(badManifest),
      /not an Only Gates model bundle/,
    );

    if (prepared.manifest.integrity !== undefined) {
      const corruptFiles = await zip.readZip(prepared.zip);
      const weights = corruptFiles.get("weights.bin");
      weights[0] ^= 0xff;
      const corrupt = await zip.createZip(
        Array.from(corruptFiles, ([name, data]) => ({ name, data })),
      );
      await assert.rejects(
        () => bundle.importModelBundle(corrupt),
        /checksum/,
      );
    }
  } finally {
    await server.close();
  }
});
