import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import test from "node:test";

import { createServer } from "vite";

/*
 * Browser training smoke test.
 *
 * TensorFlow.js runs on the CPU backend in Node, which is enough to exercise
 * the real forward pass, backward pass, Adam update, validation, checkpoint
 * export, and resume path. The browser-only pieces (IndexedDB, WebGL/WebGPU
 * selection, UI) are covered by the app build and manual checks.
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

test("browser training tokenizes, trains, exports, and resumes", async () => {
  const server = await createServer({
    root: appRoot,
    appType: "custom",
    logLevel: "silent",
    server: { middlewareMode: true },
    optimizeDeps: { noDiscovery: true },
  });

  try {
    const trainer = await server.ssrLoadModule("/src/lib/trainTinyGpt.ts");
    const tiny = await server.ssrLoadModule("/src/lib/tinyGpt.ts");
    const base64 = await server.ssrLoadModule("/src/lib/base64Float32.ts");

    assert.deepEqual(trainer.tokenizeWords("Alice's ADVENTURES!"), [
      "alice's",
      "adventures",
    ]);

    const corpusData = trainer.buildWordCorpus(corpus, 32);
    assert.equal(corpusData.vocab[0], "<unk>");
    assert.equal(corpusData.vocab[1], "<bos>");
    assert.equal(corpusData.vocab[2], "<eos>");
    assert.ok(corpusData.vocab.length <= 32);
    assert.ok(corpusData.trainingTokens.length > 8);
    assert.ok(corpusData.validationTokens.length > 0);

    const progress = [];
    const checkpoints = [];
    const config = {
      context: 4,
      dim: 8,
      layers: 1,
      heads: 2,
      mlp: 16,
    };

    const first = await trainer.trainTinyGpt({
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
      onProgress: (point) => progress.push(point),
      onCheckpoint: (state) => checkpoints.push(state),
    });

    assert.equal(first.backend, "cpu");
    assert.equal(first.stopped, false);
    assert.equal(first.state.step, 3);
    assert.equal(first.history.length, 3);
    assert.equal(progress.length, 3);
    assert.ok(checkpoints.length >= 1);

    for (const point of progress) {
      assert.ok(Number.isFinite(point.loss));
      assert.ok(point.loss > 0);
      assert.ok(Number.isFinite(point.smoothedLoss));
      assert.ok(point.stepsPerSecond > 0);
      assert.ok(Number.isFinite(point.lr));
      assert.ok(point.lr > 0);
    }
    // The cosine schedule is applied and reported, so it must decay.
    assert.ok(progress[0].lr > progress[progress.length - 1].lr);

    assert.equal(
      first.checkpoint.config.vocabSize,
      first.checkpoint.vocab.length,
    );
    assert.equal(first.checkpoint.config.context, 4);
    assert.equal(first.checkpoint.weights.layers.length, 1);
    assert.equal(first.state.optimizer.m.layers.length, 1);
    assert.equal(first.state.optimizer.v.layers.length, 1);
    assert.ok(first.state.bestWeights !== null);

    const decoded = base64.decodeFloat32(
      first.checkpoint.weights.tokenEmb,
    );
    assert.equal(
      decoded.length,
      first.checkpoint.config.vocabSize * first.checkpoint.config.dim,
    );

    const model = tiny.loadTinyGpt(first.checkpoint);
    const generated = tiny.generateTinyGpt(model, "alice was", "float", {
      maxTokens: 4,
      temperature: 0.8,
      topK: 10,
      seed: 1,
    });
    assert.equal(typeof generated.text, "string");
    assert.ok(generated.steps >= 1);

    const resumed = await trainer.trainTinyGpt({
      corpus,
      config,
      vocabLimit: 32,
      steps: 5,
      batch: 2,
      lr: 0.003,
      dropout: 0,
      seed: 7,
      backend: "cpu",
      yieldEvery: 1,
      saveEvery: 5,
      initialState: first.state,
    });

    assert.equal(resumed.state.step, 5);
    assert.equal(resumed.history.length, 5);
    assert.deepEqual(
      resumed.state.history.slice(0, 3).map((point) => point.step),
      [1, 2, 3],
    );
    assert.deepEqual(
      resumed.history.slice(3).map((point) => point.step),
      [4, 5],
    );
    assert.equal(resumed.checkpoint.config.vocabSize, first.checkpoint.vocab.length);
    assert.equal(
      resumed.state.corpusHash,
      first.state.corpusHash,
    );
  } finally {
    await server.close();
  }
});
