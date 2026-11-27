#!/usr/bin/env node
/*
 * Gate-wise convolutional feature detectors on MNIST.
 *
 * This is an isolated experiment, not part of the app bundle. It asks one
 * question: can a shared Boolean gate circuit act as a convolutional feature
 * detector and still learn a classifier?
 *
 * Every "neuron" below is the same kind of cell as `src/main.c`:
 *   - two input pins, each a softmax over a local patch (or a dense layer),
 *   - one softmax over the 16 Boolean gates,
 *   - soft output = sum_k p(gate=k) * soft_gate_k(a, b).
 *
 * A convolutional cell shares its two wire distributions and gate
 * distribution across every spatial position. A two-layer detector is two
 * such cells stacked before pooling.
 *
 * Usage:
 *   node experiments/gate-cnn-mnist/run.mjs --train 10000 --test 2000 --epochs 8
 *
 * Data is downloaded from the standard MNIST mirror on first run and cached
 * under ./data. Results are written to ./out/results.json.
 */

import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";
import { gunzipSync } from "node:zlib";

import * as tf from "@tensorflow/tfjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const DEFAULT_DATA = resolve(HERE, "data");
const DEFAULT_OUT = resolve(HERE, "out");

const MNIST_BASE = "https://storage.googleapis.com/cvdf-datasets/mnist";
const MNIST_FILES = {
  trainImages: "train-images-idx3-ubyte.gz",
  trainLabels: "train-labels-idx1-ubyte.gz",
  testImages: "t10k-images-idx3-ubyte.gz",
  testLabels: "t10k-labels-idx1-ubyte.gz",
};

const { values: args } = parseArgs({
  options: {
    train: { type: "string", default: "10000" },
    test: { type: "string", default: "2000" },
    epochs: { type: "string", default: "8" },
    batch: { type: "string", default: "64" },
    lr: { type: "string", default: "0.01" },
    "circuits-per-label": { type: "string", default: "1" },
    "dense-hidden": { type: "string", default: "128,64" },
    "score-scale": { type: "string", default: "1" },
    "mask-constants": { type: "boolean", default: false },
    simplify: { type: "boolean", default: false },
    "t-start": { type: "string", default: "1" },
    "t-min": { type: "string", default: "0.25" },
    seed: { type: "string", default: "1337" },
    size: { type: "string", default: "14" },
    threshold: { type: "string", default: "0.25" },
    models: { type: "string", default: "dense,conv-shallow,conv-deep" },
    data: { type: "string", default: DEFAULT_DATA },
    out: { type: "string", default: DEFAULT_OUT },
  },
});

const TRAIN_COUNT = Number(args.train);
const TEST_COUNT = Number(args.test);
const EPOCHS = Number(args.epochs);
const BATCH = Number(args.batch);
const LR = Number(args.lr);
const T_START = Number(args["t-start"]);
const T_MIN = Number(args["t-min"]);
const CIRCUITS_PER_LABEL = Number(args["circuits-per-label"]);
const DENSE_HIDDEN = String(args["dense-hidden"])
  .split(",")
  .map(Number);
const MASK_CONSTANTS = args["mask-constants"];
const SIMPLIFY = args.simplify;
const OUTPUTS = 10 * CIRCUITS_PER_LABEL;
const SCORE_SCALE = Number(args["score-scale"]);
const SEED = Number(args.seed);
const IMAGE_SIZE = Number(args.size);
const THRESHOLD = Number(args.threshold);
const MODEL_NAMES = args.models.split(",").map((name) => name.trim()).filter(Boolean);
const DATA_DIR = resolve(args.data);
const OUT_DIR = resolve(args.out);

/* ------------------------------------------------------------------ */
/* Deterministic helpers                                              */
/* ------------------------------------------------------------------ */

function mulberry32(seed) {
  let state = seed >>> 0;
  return function random() {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function shuffledIndices(count, random) {
  const indices = new Uint32Array(count);
  for (let i = 0; i < count; i++) indices[i] = i;
  for (let i = count - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    const swap = indices[i];
    indices[i] = indices[j];
    indices[j] = swap;
  }
  return indices;
}

function argmax(values, offset, count) {
  let best = 0;
  for (let i = 1; i < count; i++) {
    if (values[offset + i] > values[offset + best]) best = i;
  }
  return best;
}

/* ------------------------------------------------------------------ */
/* MNIST                                                              */
/* ------------------------------------------------------------------ */

async function downloadIfMissing(path, url) {
  if (existsSync(path)) return path;
  mkdirSync(dirname(path), { recursive: true });
  process.stdout.write(`downloading ${url}\n`);
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  const compressed = Buffer.from(await response.arrayBuffer());
  const raw = gunzipSync(compressed);
  writeFileSync(path, raw);
  return path;
}

function parseImages(path) {
  const buffer = readFileSync(path);
  const magic = buffer.readUInt32BE(0);
  if (magic !== 0x00000803) throw new Error(`${path}: not an IDX image file`);
  const count = buffer.readUInt32BE(4);
  const rows = buffer.readUInt32BE(8);
  const cols = buffer.readUInt32BE(12);
  return { count, rows, cols, pixels: buffer.subarray(16) };
}

function parseLabels(path) {
  const buffer = readFileSync(path);
  const magic = buffer.readUInt32BE(0);
  if (magic !== 0x00000801) throw new Error(`${path}: not an IDX label file`);
  const count = buffer.readUInt32BE(4);
  return { count, labels: buffer.subarray(8) };
}

/**
 * 28x28 -> size x size by block averaging, then a hard threshold. Gate inputs
 * are Boolean; averaging first keeps thin strokes from disappearing as badly
 * as nearest-neighbour sampling.
 */
function downsampleBinarize(images, outSize, threshold) {
  const factor = images.rows / outSize;
  if (!Number.isInteger(factor)) {
    throw new Error(`MNIST is ${images.rows}x${images.cols}; size must divide 28`);
  }
  const inputPerImage = images.rows * images.cols;
  const output = new Float32Array(images.count * outSize * outSize);
  const area = factor * factor * 255;

  for (let image = 0; image < images.count; image++) {
    const source = image * inputPerImage;
    const target = image * outSize * outSize;
    for (let y = 0; y < outSize; y++) {
      for (let x = 0; x < outSize; x++) {
        let sum = 0;
        for (let dy = 0; dy < factor; dy++) {
          const row = source + (y * factor + dy) * images.cols + x * factor;
          for (let dx = 0; dx < factor; dx++) sum += images.pixels[row + dx];
        }
        output[target + y * outSize + x] = sum / area >= threshold ? 1 : 0;
      }
    }
  }
  return output;
}

async function loadMnist() {
  const paths = {};
  for (const [key, filename] of Object.entries(MNIST_FILES)) {
    paths[key] = await downloadIfMissing(
      resolve(DATA_DIR, filename.replace(/\.gz$/, "")),
      `${MNIST_BASE}/${filename}`,
    );
  }
  const trainImages = parseImages(paths.trainImages);
  const trainLabels = parseLabels(paths.trainLabels);
  const testImages = parseImages(paths.testImages);
  const testLabels = parseLabels(paths.testLabels);

  const trainPixels = downsampleBinarize(trainImages, IMAGE_SIZE, THRESHOLD);
  const testPixels = downsampleBinarize(testImages, IMAGE_SIZE, THRESHOLD);
  return {
    trainPixels,
    trainLabels: trainLabels.labels,
    testPixels,
    testLabels: testLabels.labels,
  };
}

function takeSamples(pixels, labels, count, random, outSize) {
  const order = shuffledIndices(labels.length, random);
  const selected = order.subarray(0, count);
  const imageLength = outSize * outSize;
  const x = new Float32Array(count * imageLength);
  const y = new Uint8Array(count);
  for (let i = 0; i < count; i++) {
    const source = selected[i] * imageLength;
    x.set(pixels.subarray(source, source + imageLength), i * imageLength);
    y[i] = labels[selected[i]];
  }
  return { x, y };
}

/* ------------------------------------------------------------------ */
/* Soft gates, matching src/main.c                                    */
/* ------------------------------------------------------------------ */

/**
 * Evaluates all 16 gates on activations a and b, then mixes them with the
 * per-cell gate probabilities. a and b are convex combinations of previous
 * Boolean activations, so every gate stays in [0, 1].
 */
function gateMix(a, b, gateProbabilities) {
  const one = tf.onesLike(a);
  const ab = tf.mul(a, b);
  const xor = tf.sub(tf.add(a, b), tf.mul(ab, 2));
  const values = [
    tf.zerosLike(a), // 0 zero
    tf.sub(one, tf.sub(tf.add(a, b), ab)), // 1 nor
    tf.sub(a, ab), // 2 lnt: a && !b
    tf.sub(one, a), // 3 not_a
    tf.sub(b, ab), // 4 rnt: !a && b
    tf.sub(one, b), // 5 not_b
    xor, // 6 xor
    tf.sub(one, ab), // 7 nand
    ab, // 8 and
    tf.sub(one, xor), // 9 xnor
    b, // 10 proj_b
    tf.add(tf.sub(one, a), ab), // 11 imp: !a || b
    a, // 12 proj_a
    tf.add(tf.sub(one, b), ab), // 13 cimp: a || !b
    tf.sub(tf.add(a, b), ab), // 14 or
    one, // 15 one
  ];
  const stacked = tf.stack(values, -1);
  const rank = a.rank;
  const channels = a.shape[rank - 1];
  const shape = new Array(rank).fill(1);
  shape[rank - 2] = channels;
  shape[rank - 1] = 16;
  const weights = tf.reshape(gateProbabilities, shape);
  return tf.clipByValue(tf.sum(tf.mul(stacked, weights), -1), 0, 1);
}

/**
 * Matches the C core's `fillRandomDistribution`: draw exponential weights,
 * normalize, then store the log probabilities as the trainable logits.
 *
 * This matters. A near-uniform gate distribution makes the 16-gate mixture a
 * constant 0.5 (the 16 truth tables average to 8), so gradients vanish until
 * noise slowly breaks the symmetry. The C core starts from a Dirichlet sample
 * for exactly this reason.
 */
function randomDistributionLogits(shape, random) {
  const count = shape.reduce((product, value) => product * value, 1);
  const values = new Float32Array(count);
  let sum = 0;
  for (let i = 0; i < count; i++) {
    const value = -Math.log(Math.max(random(), Number.MIN_VALUE));
    values[i] = value;
    sum += value;
  }
  for (let i = 0; i < count; i++) values[i] = Math.log(values[i] / sum);
  return tf.tensor(values, shape);
}

function makeVariable(shape, random, name, variables) {
  const variable = tf.variable(randomDistributionLogits(shape, random), true, name);
  variables.push(variable);
  return variable;
}

/* ------------------------------------------------------------------ */
/* Model definitions                                                  */
/* ------------------------------------------------------------------ */

function buildDenseModel(inputSize, hidden, outputSize, random) {
  const variables = [];
  const layers = [];
  let previous = inputSize;
  let index = 0;
  for (const size of [...hidden, outputSize]) {
    const wireA = makeVariable([size, previous], random, `dense${index}_a`, variables);
    const wireB = makeVariable([size, previous], random, `dense${index}_b`, variables);
    const gates = makeVariable([size, 16], random, `dense${index}_g`, variables);
    layers.push({ type: "dense", inputSize: previous, outputSize: size, wireA, wireB, gates });
    previous = size;
    index++;
  }
  return { name: "dense", kind: "dense", inputSize, outputSize, layers, variables };
}

function buildConvModel(name, inputSize, blocks, head, outputSize, random) {
  const variables = [];
  let inChannels = 1;
  let height = inputSize;
  let width = inputSize;
  let index = 0;

  const builtBlocks = [];
  for (const block of blocks) {
    const layers = [];
    for (const spec of block.layers) {
      const outHeight = Math.ceil(height / spec.stride);
      const outWidth = Math.ceil(width / spec.stride);
      const kernelSize = spec.kernel ?? 3;
      const wireA = makeVariable(
        [spec.filters, kernelSize * kernelSize * inChannels],
        random,
        `conv${index}_a`,
        variables,
      );
      const wireB = makeVariable(
        [spec.filters, kernelSize * kernelSize * inChannels],
        random,
        `conv${index}_b`,
        variables,
      );
      const gates = makeVariable([spec.filters, 16], random, `conv${index}_g`, variables);
      layers.push({
        type: "conv",
        kernelSize,
        stride: spec.stride,
        inChannels,
        filters: spec.filters,
        inHeight: height,
        inWidth: width,
        outHeight,
        outWidth,
        wireA,
        wireB,
        gates,
      });
      inChannels = spec.filters;
      height = outHeight;
      width = outWidth;
      index++;
    }
    if (block.pool === "max2") {
      height = Math.ceil(height / 2);
      width = Math.ceil(width / 2);
    } else if (block.pool === "global") {
      height = 1;
      width = 1;
    }
    builtBlocks.push({ ...block, layers });
  }

  const headInput = height * width * inChannels;
  let previous = headInput;
  const headLayers = [];
  let headIndex = 0;
  for (const size of [...head, outputSize]) {
    const wireA = makeVariable([size, previous], random, `head${headIndex}_a`, variables);
    const wireB = makeVariable([size, previous], random, `head${headIndex}_b`, variables);
    const gates = makeVariable([size, 16], random, `head${headIndex}_g`, variables);
    headLayers.push({ type: "dense", inputSize: previous, outputSize: size, wireA, wireB, gates });
    previous = size;
    headIndex++;
  }

  return {
    name,
    kind: "conv",
    inputSize,
    outputSize,
    blocks: builtBlocks,
    headInput,
    headLayers,
    variables,
  };
}

function gateWeights(logits, temperature) {
  const probabilities = tf.softmax(tf.div(logits, temperature), 1);
  if (!MASK_CONSTANTS) return probabilities;
  const mask = tf.tensor1d([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]);
  const masked = tf.mul(probabilities, mask);
  return tf.div(masked, tf.add(tf.sum(masked, 1, true), 1e-8));
}

function convLayer(x, layer, temperature) {
  const kernelSize = layer.kernelSize;
  const kernelShape = [kernelSize, kernelSize, layer.inChannels, layer.filters];
  const kernelA = tf.reshape(tf.softmax(tf.div(layer.wireA, temperature), 1), kernelShape);
  const kernelB = tf.reshape(tf.softmax(tf.div(layer.wireB, temperature), 1), kernelShape);
  const a = tf.conv2d(x, kernelA, [layer.stride, layer.stride], "same");
  const b = tf.conv2d(x, kernelB, [layer.stride, layer.stride], "same");
  return gateMix(a, b, gateWeights(layer.gates, temperature));
}

function denseLayer(x, layer, temperature) {
  const wireA = tf.transpose(tf.softmax(tf.div(layer.wireA, temperature), 1));
  const wireB = tf.transpose(tf.softmax(tf.div(layer.wireB, temperature), 1));
  const a = tf.matMul(x, wireA);
  const b = tf.matMul(x, wireB);
  return gateMix(a, b, gateWeights(layer.gates, temperature));
}

function forward(model, input) {
  const temperature = model.temperature ?? 1;
  if (model.kind === "dense") {
    let value = tf.reshape(input, [input.shape[0], -1]);
    for (const layer of model.layers) value = denseLayer(value, layer, temperature);
    return value;
  }

  let value = input;
  for (const block of model.blocks) {
    for (const layer of block.layers) value = convLayer(value, layer, temperature);
    if (block.pool === "max2") {
      value = tf.maxPool(value, 2, 2, "same");
    } else if (block.pool === "global") {
      value = tf.max(value, [1, 2]);
    }
  }
  value = tf.reshape(value, [input.shape[0], -1]);
  for (const layer of model.headLayers) value = denseLayer(value, layer, temperature);
  return value;
}

/* ------------------------------------------------------------------ */
/* Training / evaluation                                              */
/* ------------------------------------------------------------------ */

function forestCrossEntropy(output, yBatch) {
  const batch = output.shape[0];
  const scores = tf.sum(
    tf.reshape(output, [batch, 10, CIRCUITS_PER_LABEL]),
    2,
  );
  const scaled = tf.mul(scores, SCORE_SCALE);
  const labels = tf.floorDiv(tf.argMax(yBatch, 1), CIRCUITS_PER_LABEL);
  return tf.losses.softmaxCrossEntropy(tf.oneHot(labels, 10), scaled);
}

function parameterCount(model) {
  return model.variables.reduce((sum, variable) => sum + variable.size, 0);
}

function gateCellCount(model) {
  let count = 0;
  if (model.kind === "dense") {
    for (const layer of model.layers) count += layer.outputSize;
  } else {
    for (const block of model.blocks) {
      for (const layer of block.layers) {
        count += layer.outHeight * layer.outWidth * layer.filters;
      }
    }
    for (const layer of model.headLayers) count += layer.outputSize;
  }
  return count;
}

function evaluateSoft(model, xTensor, labels, limit) {
  let correct = 0;
  for (let start = 0; start < limit; start += 256) {
    const end = Math.min(limit, start + 256);
    const batch = tf.slice(xTensor, [start, 0, 0, 0], [end - start, IMAGE_SIZE, IMAGE_SIZE, 1]);
    const predictions = tf.tidy(() => {
      const output = forward(model, batch);
      const votes = tf.sum(tf.reshape(output, [end - start, 10, CIRCUITS_PER_LABEL]), 2);
      return tf.argMax(votes, 1).dataSync();
    });
    for (let i = 0; i < predictions.length; i++) {
      if (predictions[i] === labels[start + i]) correct++;
    }
    batch.dispose();
  }
  return correct / limit;
}

async function trainModel(model, dataset, options) {
  const optimizer = tf.train.adam(options.lr);
  const random = mulberry32(options.seed);
  const { xTensor, yTensor, labels } = dataset;
  const total = labels.length;
  const history = [];

  for (let epoch = 0; epoch < options.epochs; epoch++) {
    model.temperature =
      options.epochs <= 1
        ? options.tMin
        : options.tStart *
          Math.pow(options.tMin / options.tStart, epoch / (options.epochs - 1));
    const order = shuffledIndices(total, random);
    let lossSum = 0;
    let batches = 0;

    for (let start = 0; start < total; start += options.batch) {
      const count = Math.min(options.batch, total - start);
      const indices = Array.from(order.subarray(start, start + count));
      const lossValue = tf.tidy(() => {
        const indexTensor = tf.tensor1d(indices, "int32");
        const xBatch = tf.gather(xTensor, indexTensor);
        const yBatch = tf.gather(yTensor, indexTensor);
        const cost = optimizer.minimize(
          () => forestCrossEntropy(forward(model, xBatch), yBatch),
          true,
          model.variables,
        );
        return cost;
      });
      lossSum += lossValue.dataSync()[0];
      lossValue.dispose();
      batches++;
    }

    const accuracy = evaluateSoft(model, xTensor, labels, total);
    const entry = {
      epoch: epoch + 1,
      temperature: model.temperature,
      loss: lossSum / batches,
      accuracy,
    };
    history.push(entry);
    process.stdout.write(
      `  epoch ${epoch + 1}/${options.epochs}  T ${entry.temperature.toFixed(2)}  loss ${entry.loss.toFixed(4)}  soft acc ${(accuracy * 100).toFixed(2)}%\n`,
    );
  }

  optimizer.dispose();
  return history;
}

/* ------------------------------------------------------------------ */
/* Hardening                                                          */
/* ------------------------------------------------------------------ */

function argmaxRows(values, rows, columns) {
  const output = new Int32Array(rows);
  for (let row = 0; row < rows; row++) {
    output[row] = argmax(values, row * columns, columns);
  }
  return output;
}

function argmaxGateRows(values, rows) {
  const output = new Int32Array(rows);
  const start = MASK_CONSTANTS ? 1 : 0;
  for (let row = 0; row < rows; row++) {
    let best = start;
    for (let gate = start; gate < 16; gate++) {
      if (MASK_CONSTANTS && gate === 15) continue;
      if (values[row * 16 + gate] > values[row * 16 + best]) best = gate;
    }
    output[row] = best;
  }
  return output;
}

function exportHardModel(model) {
  const layers = [];
  if (model.kind === "dense") {
    for (const layer of model.layers) {
      layers.push({
        type: "dense",
        inputSize: layer.inputSize,
        outputSize: layer.outputSize,
        wireA: argmaxRows(layer.wireA.dataSync(), layer.outputSize, layer.inputSize),
        wireB: argmaxRows(layer.wireB.dataSync(), layer.outputSize, layer.inputSize),
        gates: argmaxGateRows(layer.gates.dataSync(), layer.outputSize),
      });
    }
    return { layers };
  }

  for (const block of model.blocks) {
    for (const layer of block.layers) {
      const patchSize = layer.kernelSize * layer.kernelSize * layer.inChannels;
      layers.push({
        type: "conv",
        kernelSize: layer.kernelSize,
        stride: layer.stride,
        inChannels: layer.inChannels,
        filters: layer.filters,
        inHeight: layer.inHeight,
        inWidth: layer.inWidth,
        outHeight: layer.outHeight,
        outWidth: layer.outWidth,
        wireA: argmaxRows(layer.wireA.dataSync(), layer.filters, patchSize),
        wireB: argmaxRows(layer.wireB.dataSync(), layer.filters, patchSize),
        gates: argmaxGateRows(layer.gates.dataSync(), layer.filters),
      });
    }
    layers.push({ type: block.pool });
  }
  for (const layer of model.headLayers) {
    layers.push({
      type: "dense",
      inputSize: layer.inputSize,
      outputSize: layer.outputSize,
      wireA: argmaxRows(layer.wireA.dataSync(), layer.outputSize, layer.inputSize),
      wireB: argmaxRows(layer.wireB.dataSync(), layer.outputSize, layer.inputSize),
      gates: argmaxGateRows(layer.gates.dataSync(), layer.outputSize),
    });
  }
  return { layers };
}

function hardGate(gate, a, b) {
  switch (gate) {
    case 0: return 0;
    case 1: return a || b ? 0 : 1;
    case 2: return a && !b ? 1 : 0;
    case 3: return a ? 0 : 1;
    case 4: return !a && b ? 1 : 0;
    case 5: return b ? 0 : 1;
    case 6: return a !== b ? 1 : 0;
    case 7: return a && b ? 0 : 1;
    case 8: return a && b ? 1 : 0;
    case 9: return a === b ? 1 : 0;
    case 10: return b ? 1 : 0;
    case 11: return !a || b ? 1 : 0;
    case 12: return a ? 1 : 0;
    case 13: return a || !b ? 1 : 0;
    case 14: return a || b ? 1 : 0;
    case 15: return 1;
    default: return 0;
  }
}

function convHard(input, layer) {
  const { kernelSize, stride, inChannels, filters, inHeight, inWidth } = layer;
  const outHeight = layer.outHeight;
  const outWidth = layer.outWidth;
  const output = new Float32Array(outHeight * outWidth * filters);
  const padHeight = Math.max((outHeight - 1) * stride + kernelSize - inHeight, 0);
  const padWidth = Math.max((outWidth - 1) * stride + kernelSize - inWidth, 0);
  const padTop = Math.floor(padHeight / 2);
  const padLeft = Math.floor(padWidth / 2);

  const decode = (index) => {
    const dy = Math.floor(index / (kernelSize * inChannels));
    const remainder = index % (kernelSize * inChannels);
    const dx = Math.floor(remainder / inChannels);
    const channel = remainder % inChannels;
    return { dy, dx, channel };
  };

  for (let filter = 0; filter < filters; filter++) {
    const offsetA = decode(layer.wireA[filter]);
    const offsetB = decode(layer.wireB[filter]);
    const gate = layer.gates[filter];
    for (let y = 0; y < outHeight; y++) {
      for (let x = 0; x < outWidth; x++) {
        const ay = y * stride - padTop + offsetA.dy;
        const ax = x * stride - padLeft + offsetA.dx;
        const by = y * stride - padTop + offsetB.dy;
        const bx = x * stride - padLeft + offsetB.dx;
        const a =
          ay >= 0 && ay < inHeight && ax >= 0 && ax < inWidth
            ? input[(ay * inWidth + ax) * inChannels + offsetA.channel] !== 0
            : false;
        const b =
          by >= 0 && by < inHeight && bx >= 0 && bx < inWidth
            ? input[(by * inWidth + bx) * inChannels + offsetB.channel] !== 0
            : false;
        output[(y * outWidth + x) * filters + filter] = hardGate(gate, a, b);
      }
    }
  }
  return { data: output, height: outHeight, width: outWidth, channels: filters };
}

function poolHard(input, mode) {
  if (mode === "global") {
    const output = new Float32Array(input.channels);
    for (let y = 0; y < input.height; y++) {
      for (let x = 0; x < input.width; x++) {
        for (let channel = 0; channel < input.channels; channel++) {
          if (input.data[(y * input.width + x) * input.channels + channel] !== 0) {
            output[channel] = 1;
          }
        }
      }
    }
    return { data: output, height: 1, width: 1, channels: input.channels };
  }

  const height = Math.ceil(input.height / 2);
  const width = Math.ceil(input.width / 2);
  const output = new Float32Array(height * width * input.channels);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      for (let channel = 0; channel < input.channels; channel++) {
        let value = 0;
        for (let dy = 0; dy < 2; dy++) {
          for (let dx = 0; dx < 2; dx++) {
            const sy = y * 2 + dy;
            const sx = x * 2 + dx;
            if (sy < input.height && sx < input.width) {
              value |= input.data[(sy * input.width + sx) * input.channels + channel] !== 0 ? 1 : 0;
            }
          }
        }
        output[(y * width + x) * input.channels + channel] = value;
      }
    }
  }
  return { data: output, height, width, channels: input.channels };
}

function denseHard(input, layer) {
  const output = new Float32Array(layer.outputSize);
  for (let cell = 0; cell < layer.outputSize; cell++) {
    const a = input[layer.wireA[cell]] !== 0;
    const b = input[layer.wireB[cell]] !== 0;
    output[cell] = hardGate(layer.gates[cell], a, b);
  }
  return output;
}

/* ------------------------------------------------------------------ */
/* Gate simplification                                                */
/* ------------------------------------------------------------------ */

/*
 * Hardened cells fall into four kinds once the argmax wires and gate are
 * fixed:
 *
 *   constant  gate 0 / 15      -> a literal, needs no inputs at all
 *   identity  gate 12 / 10     -> a copy of wire a / b, needs no cell
 *   inverter  gate 3 / 5       -> NOT a / NOT b, needs one input not two
 *   real      everything else  -> a genuine two-input Boolean function
 *
 * Constants and identities fold away completely, inverters halve their fan-in,
 * and any cell nobody references after folding is dead. This is exact: the
 * folded circuit computes the same function, it just has fewer cells.
 */
const IDENTITY_GATES = new Map([
  [12, "wireA"],
  [10, "wireB"],
]);
const INVERTER_GATES = new Map([
  [3, "wireA"],
  [5, "wireB"],
]);

function foldHardDense(layers) {
  const last = layers.length - 1;
  const needed = layers.map(() => new Set());
  for (let cell = 0; cell < layers[last].outputSize; cell++) {
    needed[last].add(cell);
  }

  const constantCount = new Map();
  for (let index = last; index >= 0; index--) {
    const layer = layers[index];
    const upstream = index > 0 ? needed[index - 1] : null;
    for (const cell of needed[index]) {
      if (!upstream) continue;
      const gate = layer.gates[cell];
      if (gate === 0 || gate === 15) {
        constantCount.set(index, (constantCount.get(index) ?? 0) + 1);
        continue;
      }
      if (IDENTITY_GATES.has(gate)) {
        upstream.add(layer[IDENTITY_GATES.get(gate)][cell]);
        continue;
      }
      if (INVERTER_GATES.has(gate)) {
        upstream.add(layer[INVERTER_GATES.get(gate)][cell]);
        continue;
      }
      upstream.add(layer.wireA[cell]);
      upstream.add(layer.wireB[cell]);
    }
  }

  const folded = [];
  const maps = [];
  for (let index = 0; index < layers.length; index++) {
    const layer = layers[index];
    const map = new Int32Array(layer.outputSize).fill(-1);
    const previous = index > 0 ? maps[index - 1] : null;
    const cells = [];
    for (const cell of [...needed[index]].sort((a, b) => a - b)) {
      const gate = layer.gates[cell];
      const remap = (origin) => (previous ? previous[origin] : origin);
      let node;
      if (gate === 0) node = { kind: "const", value: 0 };
      else if (gate === 15) node = { kind: "const", value: 1 };
      else if (IDENTITY_GATES.has(gate)) {
        node = { kind: "wire", src: remap(layer[IDENTITY_GATES.get(gate)][cell]) };
      } else if (INVERTER_GATES.has(gate)) {
        node = { kind: "not", src: remap(layer[INVERTER_GATES.get(gate)][cell]) };
      } else {
        node = {
          kind: "gate",
          gate,
          srcA: remap(layer.wireA[cell]),
          srcB: remap(layer.wireB[cell]),
        };
      }
      map[cell] = cells.length;
      cells.push(node);
    }
    maps.push(map);
    folded.push({ cells });
  }
  return folded;
}

function runFoldedDense(folded, input) {
  let value = input;
  for (const layer of folded) {
    const next = new Float32Array(layer.cells.length);
    for (let index = 0; index < layer.cells.length; index++) {
      const node = layer.cells[index];
      if (node.kind === "const") next[index] = node.value;
      else if (node.kind === "wire") next[index] = value[node.src] !== 0 ? 1 : 0;
      else if (node.kind === "not") next[index] = value[node.src] !== 0 ? 0 : 1;
      else {
        next[index] = hardGate(
          node.gate,
          value[node.srcA] !== 0,
          value[node.srcB] !== 0,
        );
      }
    }
    value = next;
  }
  return value;
}

function foldStats(folded) {
  let cells = 0;
  let constant = 0;
  let wire = 0;
  let not = 0;
  let gate = 0;
  for (const layer of folded) {
    for (const node of layer.cells) {
      cells++;
      if (node.kind === "const") constant++;
      else if (node.kind === "wire") wire++;
      else if (node.kind === "not") not++;
      else if (node.kind === "gate") gate++;
    }
  }
  return { cells, constant, wire, not, gate };
}

function evaluateFolded(model, x, labels) {
  const hard = exportHardModel(model);
  if (model.kind !== "dense") return null;
  const folded = foldHardDense(
    hard.layers.filter((layer) => layer.type === "dense"),
  );
  const circuits = CIRCUITS_PER_LABEL;
  const imageLength = IMAGE_SIZE * IMAGE_SIZE;
  let correct = 0;
  let identical = 0;
  for (let sample = 0; sample < labels.length; sample++) {
    const input = x.subarray(sample * imageLength, (sample + 1) * imageLength);
    const output = runFoldedDense(folded, input);
    if (output.length !== hard.layers.at(-1).outputSize) {
      throw new Error("folded circuit lost output cells");
    }
    const votes = new Float32Array(10);
    for (let i = 0; i < output.length; i++) {
      votes[Math.floor(i / circuits)] += output[i];
    }
    let best = 0;
    for (let label = 1; label < 10; label++) {
      if (votes[label] > votes[best]) best = label;
    }
    if (best === labels[sample]) correct++;
    if (sample % 8 === 0) {
      // Cross-check the fold against the original wiring on a sample.
      let value = input;
      for (const layer of hard.layers) value = denseHard(value, layer);
      let same = value.length === output.length;
      for (let i = 0; same && i < value.length; i++) {
        if (value[i] !== output[i]) same = false;
      }
      if (same) identical++;
    }
  }
  return {
    accuracy: correct / labels.length,
    stats: foldStats(folded),
    verified: identical,
  };
}

function runHardModel(hard, input) {
  let value = { data: input, height: IMAGE_SIZE, width: IMAGE_SIZE, channels: 1 };
  for (const layer of hard.layers) {
    if (layer.type === "conv") {
      value = convHard(value.data, layer);
    } else if (layer.type === "max2" || layer.type === "global") {
      value = poolHard(value, layer.type);
    } else if (layer.type === "dense") {
      value = { data: denseHard(value.data, layer), height: 1, width: 1, channels: layer.outputSize };
    }
  }
  return value.data;
}

function evaluateHard(model, x, labels) {
  const hard = exportHardModel(model);
  const circuits = CIRCUITS_PER_LABEL;
  let correct = 0;
  let allZero = 0;
  const imageLength = IMAGE_SIZE * IMAGE_SIZE;
  for (let sample = 0; sample < labels.length; sample++) {
    const input = x.subarray(sample * imageLength, (sample + 1) * imageLength);
    const output = runHardModel(hard, input);
    let any = false;
    const votes = new Float32Array(10);
    for (let i = 0; i < output.length; i++) {
      if (output[i] !== 0) any = true;
      votes[Math.floor(i / circuits)] += output[i];
    }
    let best = 0;
    for (let label = 1; label < 10; label++) {
      if (votes[label] > votes[best]) best = label;
    }
    if (!any) allZero++;
    if (best === labels[sample]) correct++;
  }
  return { accuracy: correct / labels.length, allZeroRate: allZero / labels.length };
}

/* ------------------------------------------------------------------ */
/* Architectures                                                      */
/* ------------------------------------------------------------------ */

function architecture(name, inputSize, random) {
  switch (name) {
    case "dense":
      return buildDenseModel(inputSize * inputSize, DENSE_HIDDEN, OUTPUTS, random);
    case "dense-wide":
      return buildDenseModel(inputSize * inputSize, [256, 128], OUTPUTS, random);
    case "dense-small":
      return buildDenseModel(inputSize * inputSize, [48, 48], OUTPUTS, random);
    case "conv-shallow":
      return buildConvModel(
        "conv-shallow",
        inputSize,
        [
          { layers: [{ filters: 8, kernel: 3, stride: 2 }], pool: "max2" },
          { layers: [{ filters: 16, kernel: 3, stride: 2 }], pool: "global" },
        ],
        [64, 64],
        OUTPUTS,
        random,
      );
    case "conv-deep":
      return buildConvModel(
        "conv-deep",
        inputSize,
        [
          {
            layers: [
              { filters: 8, kernel: 3, stride: 1 },
              { filters: 8, kernel: 3, stride: 1 },
            ],
            pool: "max2",
          },
          {
            layers: [
              { filters: 16, kernel: 3, stride: 1 },
              { filters: 16, kernel: 3, stride: 1 },
            ],
            pool: "global",
          },
        ],
        [64, 64],
        OUTPUTS,
        random,
      );
    case "conv-grid":
      return buildConvModel(
        "conv-grid",
        inputSize,
        [
          { layers: [{ filters: 16, kernel: 3, stride: 1 }], pool: "max2" },
          { layers: [{ filters: 32, kernel: 3, stride: 1 }], pool: "max2" },
        ],
        [64, 64],
        OUTPUTS,
        random,
      );
    case "conv-grid-head":
      return buildConvModel(
        "conv-grid-head",
        inputSize,
        [
          { layers: [{ filters: 16, kernel: 3, stride: 1 }], pool: "max2" },
          { layers: [{ filters: 32, kernel: 3, stride: 1 }], pool: "max2" },
          { layers: [{ filters: 64, kernel: 1, stride: 1 }], pool: "global" },
        ],
        [64],
        OUTPUTS,
        random,
      );
    case "conv-deep-grid":
      return buildConvModel(
        "conv-deep-grid",
        inputSize,
        [
          {
            layers: [
              { filters: 12, kernel: 3, stride: 1 },
              { filters: 12, kernel: 3, stride: 1 },
            ],
            pool: "max2",
          },
          {
            layers: [
              { filters: 24, kernel: 3, stride: 1 },
              { filters: 24, kernel: 3, stride: 1 },
            ],
            pool: "max2",
          },
        ],
        [64, 64],
        OUTPUTS,
        random,
      );
    default:
      throw new Error(`Unknown model ${name}`);
  }
}

/* ------------------------------------------------------------------ */
/* Main                                                               */
/* ------------------------------------------------------------------ */

async function main() {
  await tf.setBackend("cpu");
  await tf.ready();
  mkdirSync(OUT_DIR, { recursive: true });

  process.stdout.write(`tfjs ${tf.version.tfjs} (${tf.getBackend()})\n`);
  process.stdout.write(`loading MNIST, ${IMAGE_SIZE}x${IMAGE_SIZE} binary...\n`);
  const mnist = await loadMnist();
  const random = mulberry32(SEED);
  const train = takeSamples(mnist.trainPixels, mnist.trainLabels, TRAIN_COUNT, random, IMAGE_SIZE);
  const test = takeSamples(mnist.testPixels, mnist.testLabels, TEST_COUNT, random, IMAGE_SIZE);

  const trainX = tf.tensor4d(train.x, [TRAIN_COUNT, IMAGE_SIZE, IMAGE_SIZE, 1]);
  // Label-major targets: unit y*K + c is the c-th circuit for label y, which
  // is exactly how the readout reshapes the output to [batch, 10, K].
  const oneHotTrain = tf.oneHot(tf.tensor1d(train.y, "int32"), 10);
  const trainY = tf
    .reshape(
      tf.tile(tf.expandDims(oneHotTrain, 2), [1, 1, CIRCUITS_PER_LABEL]),
      [TRAIN_COUNT, 10 * CIRCUITS_PER_LABEL],
    )
    .toFloat();
  oneHotTrain.dispose();
  const testX = tf.tensor4d(test.x, [TEST_COUNT, IMAGE_SIZE, IMAGE_SIZE, 1]);

  const results = [];
  for (const name of MODEL_NAMES) {
    process.stdout.write(`\n=== ${name} ===\n`);
    const model = architecture(name, IMAGE_SIZE, mulberry32(SEED));
    process.stdout.write(
      `params ${parameterCount(model).toLocaleString()}  gate cells ${gateCellCount(model).toLocaleString()}\n`,
    );
    const started = Date.now();
    const history = await trainModel(model, { xTensor: trainX, yTensor: trainY, labels: train.y }, {
      epochs: EPOCHS,
      batch: BATCH,
      lr: LR,
      tStart: T_START,
      tMin: T_MIN,
      seed: SEED,
    });
    const soft = evaluateSoft(model, testX, test.y, TEST_COUNT);
    const hard = evaluateHard(model, test.x, test.y);
    const simplified = SIMPLIFY ? evaluateFolded(model, test.x, test.y) : null;
    const result = {
      model: name,
      params: parameterCount(model),
      gateCells: gateCellCount(model),
      softAccuracy: soft,
      hardAccuracy: hard.accuracy,
      hardAllZeroRate: hard.allZeroRate,
      simplified: simplified && {
        accuracy: simplified.accuracy,
        foldedCells: simplified.stats.cells,
        constantCells: simplified.stats.constant,
        wireCells: simplified.stats.wire,
        notCells: simplified.stats.not,
        gateCells: simplified.stats.gate,
        verifiedSamples: simplified.verified,
      },
      seconds: (Date.now() - started) / 1000,
      history,
    };
    results.push(result);
    process.stdout.write(
      `  test soft ${(soft * 100).toFixed(2)}%  hard ${(hard.accuracy * 100).toFixed(2)}%  hard-zero ${(hard.allZeroRate * 100).toFixed(1)}%  ${result.seconds.toFixed(1)}s\n`,
    );
    if (simplified) {
      const stats = simplified.stats;
      process.stdout.write(
        `  folded ${result.gateCells} -> ${stats.cells} cells ` +
          `(${stats.gate} gates, ${stats.not} not, ${stats.wire} wires, ` +
          `${stats.constant} const)  ` +
          `accuracy ${(simplified.accuracy * 100).toFixed(2)}%  ` +
          `verified on ${simplified.verified} samples\n`,
      );
    }
    for (const variable of model.variables) variable.dispose();
  }

  writeFileSync(
    resolve(OUT_DIR, "results.json"),
    JSON.stringify(
      {
        config: {
          train: TRAIN_COUNT,
          test: TEST_COUNT,
          epochs: EPOCHS,
          batch: BATCH,
          lr: LR,
          seed: SEED,
          imageSize: IMAGE_SIZE,
          threshold: THRESHOLD,
        },
        results,
      },
      null,
      2,
    ),
  );

  process.stdout.write(`\n${"model".padEnd(14)} ${"params".padStart(9)} ${"gates".padStart(8)} ${"soft".padStart(8)} ${"hard".padStart(8)}\n`);
  for (const result of results) {
    process.stdout.write(
      `${result.model.padEnd(14)} ${String(result.params).padStart(9)} ${String(result.gateCells).padStart(8)} ${(result.softAccuracy * 100).toFixed(2).padStart(8)} ${(result.hardAccuracy * 100).toFixed(2).padStart(8)}\n`,
    );
  }

  trainX.dispose();
  trainY.dispose();
  testX.dispose();
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
