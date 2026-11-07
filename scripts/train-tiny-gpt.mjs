#!/usr/bin/env node
/*
 * Train a tiny decoder-only transformer with float32 and export a checkpoint
 * for the browser's integer inference demo.
 *
 * This is deliberately dependency-free. It is not a fast trainer: the point is
 * to produce a small, inspectable model and a checkpoint format that the
 * TypeScript inference engine can load directly. For better weights, use the
 * PyTorch/Colab script with the same export format.
 *
 * Usage:
 *   node scripts/train-tiny-gpt.mjs
 *   node scripts/train-tiny-gpt.mjs --steps 3000 --dim 64
 *   node scripts/train-tiny-gpt.mjs --check
 */

import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const CORPUS_PATH = resolve(
  ROOT,
  "apps/only-gates-ai/src/assets/tiny-gpt-corpus.txt",
);
const OUTPUT_PATH = resolve(
  ROOT,
  "apps/only-gates-ai/src/assets/tiny-gpt-weights.ts",
);

const args = process.argv.slice(2);
function arg(name, fallback) {
  const index = args.indexOf(name);
  return index === -1 ? fallback : args[index + 1];
}

const CHECK = args.includes("--check");
const STEPS = Number(arg("--steps", 1800));
const BATCH = Number(arg("--batch", 8));
const CONTEXT = Number(arg("--context", 16));
const DIM = Number(arg("--dim", 48));
const LAYERS = Number(arg("--layers", 2));
const HEADS = Number(arg("--heads", 2));
const MLP = Number(arg("--mlp", 96));
const VOCAB_LIMIT = Number(arg("--vocab", 512));
const LR = Number(arg("--lr", 0.003));
const SEED = Number(arg("--seed", 1337));
const SAMPLE_EVERY = Number(arg("--sample-every", 300));

const SPECIALS = ["<unk>", "<bos>", "<eos>"];
const UNK = 0;
const BOS = 1;
const EOS = 2;
const EPS = 1e-5;
// Float32 for the browser checkpoint and training; Float64 makes the
// finite-difference gradient check meaningful.
let DTYPE = Float32Array;

/* ------------------------------------------------------------------ */
/* Random and small helpers                                           */
/* ------------------------------------------------------------------ */

function mulberry32(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function makeNormal(random) {
  let spare = null;
  return () => {
    if (spare !== null) {
      const value = spare;
      spare = null;
      return value;
    }
    let u = 0;
    let v = 0;
    while (u === 0) u = random();
    while (v === 0) v = random();
    const mag = Math.sqrt(-2 * Math.log(u));
    spare = mag * Math.sin(2 * Math.PI * v);
    return mag * Math.cos(2 * Math.PI * v);
  };
}

function tokenizeWords(text) {
  return text.toLowerCase().match(/[a-z0-9']+/g) ?? [];
}

function buildVocabulary(tokens, limit) {
  const counts = new Map();
  for (const token of tokens) {
    counts.set(token, (counts.get(token) ?? 0) + 1);
  }
  const words = [...counts.entries()]
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .slice(0, Math.max(0, limit - SPECIALS.length))
    .map(([word]) => word);
  return [...SPECIALS, ...words];
}

function encodeTokens(tokens, index) {
  return tokens.map((token) => index.get(token) ?? UNK);
}

function matmul(x, rows, cols, w, outCols, out) {
  // x: rows x cols, w: cols x outCols, out: rows x outCols
  for (let row = 0; row < rows; row++) {
    const xOffset = row * cols;
    const outOffset = row * outCols;
    for (let j = 0; j < outCols; j++) {
      let sum = 0;
      for (let k = 0; k < cols; k++) {
        sum += x[xOffset + k] * w[k * outCols + j];
      }
      out[outOffset + j] = sum;
    }
  }
}

function rmsnorm(x, rows, cols, gain, out) {
  for (let row = 0; row < rows; row++) {
    const offset = row * cols;
    let sumSq = 0;
    for (let i = 0; i < cols; i++) {
      const value = x[offset + i];
      sumSq += value * value;
    }
    const inv = 1 / Math.sqrt(sumSq / cols + EPS);
    for (let i = 0; i < cols; i++) {
      out[offset + i] = x[offset + i] * inv * gain[i];
    }
  }
}

function rmsnormBackward(x, dy, gain, dx, dgain, rows, cols) {
  for (let row = 0; row < rows; row++) {
    const offset = row * cols;
    let sumSq = 0;
    let dot = 0;
    for (let i = 0; i < cols; i++) {
      sumSq += x[offset + i] * x[offset + i];
      dot += dy[offset + i] * gain[i] * x[offset + i];
    }
    const inv = 1 / Math.sqrt(sumSq / cols + EPS);
    const inv3 = inv * inv * inv;
    for (let i = 0; i < cols; i++) {
      const value = x[offset + i];
      dx[offset + i] =
        dy[offset + i] * gain[i] * inv - (dot * value * inv3) / cols;
      dgain[i] += dy[offset + i] * value * inv;
    }
  }
}

function gelu(x) {
  const inner = 0.7978845608028654 * (x + 0.044715 * x * x * x);
  return 0.5 * x * (1 + Math.tanh(inner));
}

function geluGrad(x) {
  const inner = 0.7978845608028654 * (x + 0.044715 * x * x * x);
  const t = Math.tanh(inner);
  const derivative =
    0.7978845608028654 * (1 + 3 * 0.044715 * x * x);
  return 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * derivative;
}

function softmaxRow(values, offset, length, out, outOffset) {
  let max = -Infinity;
  for (let i = 0; i < length; i++) {
    const value = values[offset + i];
    if (value > max) max = value;
  }
  let sum = 0;
  for (let i = 0; i < length; i++) {
    const value = Math.exp(values[offset + i] - max);
    out[outOffset + i] = value;
    sum += value;
  }
  const inv = sum > 0 ? 1 / sum : 0;
  for (let i = 0; i < length; i++) {
    out[outOffset + i] *= inv;
  }
}

/* ------------------------------------------------------------------ */
/* Model                                                              */
/* ------------------------------------------------------------------ */

function makeConfig(overrides = {}) {
  const vocabSize = overrides.vocabSize ?? 16;
  const context = overrides.context ?? 8;
  const dim = overrides.dim ?? 16;
  const layers = overrides.layers ?? 1;
  const heads = overrides.heads ?? 1;
  const mlp = overrides.mlp ?? dim * 2;
  if (dim % heads !== 0) {
    throw new Error("dim must be divisible by heads");
  }
  return {
    vocabSize,
    context,
    dim,
    layers,
    heads,
    mlp,
    headDim: dim / heads,
    eps: EPS,
  };
}

function makeModel(config, random) {
  const normal = makeNormal(random);
  const { vocabSize, context, dim, layers, heads, mlp } = config;
  const make = (size, std) => {
    const array = new DTYPE(size);
    for (let i = 0; i < size; i++) array[i] = normal() * std;
    return array;
  };
  const ones = (size) => {
    const array = new DTYPE(size);
    array.fill(1);
    return array;
  };

  const model = {
    tokenEmb: make(vocabSize * dim, 0.02),
    posEmb: make(context * dim, 0.01),
    layers: [],
    gFinal: ones(dim),
  };

  const linearStd = 0.02;
  const residualStd = 0.02 / Math.sqrt(2 * Math.max(1, layers));
  for (let layer = 0; layer < layers; layer++) {
    model.layers.push({
      g1: ones(dim),
      wqkv: make(dim * dim * 3, linearStd),
      wo: make(dim * dim, residualStd),
      g2: ones(dim),
      w1: make(dim * mlp, linearStd),
      w2: make(mlp * dim, residualStd),
    });
  }
  return model;
}

function zeroGrads(model, config) {
  const { vocabSize, context, dim, layers, mlp } = config;
  const grads = {
    tokenEmb: new DTYPE(vocabSize * dim),
    posEmb: new DTYPE(context * dim),
    layers: [],
    gFinal: new DTYPE(dim),
  };
  for (let layer = 0; layer < layers; layer++) {
    grads.layers.push({
      g1: new DTYPE(dim),
      wqkv: new DTYPE(dim * dim * 3),
      wo: new DTYPE(dim * dim),
      g2: new DTYPE(dim),
      w1: new DTYPE(dim * mlp),
      w2: new DTYPE(mlp * dim),
    });
  }
  return grads;
}

function forward(model, tokens, config, collect = true) {
  const { vocabSize, context, dim, layers, heads, mlp, headDim } = config;
  const B = tokens.length / context;
  const T = context;
  if (!Number.isInteger(B)) {
    throw new Error("token batch is not a multiple of the context length");
  }
  const N = B * T;

  const x0 = new DTYPE(N * dim);
  for (let b = 0; b < B; b++) {
    for (let t = 0; t < T; t++) {
      const token = tokens[b * T + t];
      const dst = (b * T + t) * dim;
      const tok = token * dim;
      const pos = t * dim;
      for (let d = 0; d < dim; d++) {
        x0[dst + d] = model.tokenEmb[tok + d] + model.posEmb[pos + d];
      }
    }
  }

  const cacheLayers = [];
  let x = x0;

  for (let layer = 0; layer < layers; layer++) {
    const params = model.layers[layer];
    const h1 = new DTYPE(N * dim);
    rmsnorm(x, N, dim, params.g1, h1);

    const qkv = new DTYPE(N * dim * 3);
    matmul(h1, N, dim, params.wqkv, dim * 3, qkv);

    const attn = new DTYPE(B * heads * T * T);
    const attnOut = new DTYPE(N * dim);
    const scale = 1 / Math.sqrt(headDim);

    for (let b = 0; b < B; b++) {
      for (let h = 0; h < heads; h++) {
        const headOffset = h * headDim;
        for (let t = 0; t < T; t++) {
          const row = (b * heads + h) * T * T + t * T;
          // Scores for j <= t.
          for (let j = 0; j <= t; j++) {
            let dot = 0;
            const qBase = (b * T + t) * dim * 3 + headOffset;
            const kBase = (b * T + j) * dim * 3 + dim + headOffset;
            for (let d = 0; d < headDim; d++) {
              dot += qkv[qBase + d] * qkv[kBase + d];
            }
            attn[row + j] = dot * scale;
          }
          // Causal mask.
          for (let j = t + 1; j < T; j++) {
            attn[row + j] = -Infinity;
          }
          softmaxRow(attn, row, T, attn, row);
          // Weighted sum of values.
          const outBase = (b * T + t) * dim + headOffset;
          for (let d = 0; d < headDim; d++) {
            let sum = 0;
            for (let j = 0; j <= t; j++) {
              const vBase = (b * T + j) * dim * 3 + dim * 2 + headOffset;
              sum += attn[row + j] * qkv[vBase + d];
            }
            attnOut[outBase + d] = sum;
          }
        }
      }
    }

    const y = new DTYPE(N * dim);
    matmul(attnOut, N, dim, params.wo, dim, y);
    const x1 = new DTYPE(N * dim);
    for (let i = 0; i < N * dim; i++) x1[i] = x[i] + y[i];

    const h2 = new DTYPE(N * dim);
    rmsnorm(x1, N, dim, params.g2, h2);

    const z = new DTYPE(N * mlp);
    matmul(h2, N, dim, params.w1, mlp, z);
    const gz = new DTYPE(N * mlp);
    for (let i = 0; i < N * mlp; i++) gz[i] = gelu(z[i]);

    const m = new DTYPE(N * dim);
    matmul(gz, N, mlp, params.w2, dim, m);
    const x2 = new DTYPE(N * dim);
    for (let i = 0; i < N * dim; i++) x2[i] = x1[i] + m[i];

    if (collect) {
      cacheLayers.push({ input: x, h1, qkv, attn, attnOut, y, x1, h2, z, gz, m, x2 });
    }
    x = x2;
  }

  const hf = new DTYPE(N * dim);
  rmsnorm(x, N, dim, model.gFinal, hf);

  const logits = new DTYPE(N * vocabSize);
  // Tied output head: logits = hf @ tokenEmb^T.
  for (let n = 0; n < N; n++) {
    const hBase = n * dim;
    const outBase = n * vocabSize;
    for (let v = 0; v < vocabSize; v++) {
      let sum = 0;
      const eBase = v * dim;
      for (let d = 0; d < dim; d++) {
        sum += hf[hBase + d] * model.tokenEmb[eBase + d];
      }
      logits[outBase + v] = sum;
    }
  }

  return { x0, layers: cacheLayers, hf, x, logits };
}

function lossAndGrad(model, grads, tokens, config) {
  const { vocabSize, context, dim, layers, heads, mlp, headDim } = config;
  const B = tokens.length / context;
  const T = context;
  const N = B * T;
  const cache = forward(model, tokens, config, true);

  let loss = 0;
  let count = 0;
  const dlogits = new DTYPE(N * vocabSize);
  const probs = new DTYPE(vocabSize);

  for (let b = 0; b < B; b++) {
    for (let t = 0; t < T - 1; t++) {
      const n = b * T + t;
      const target = tokens[n + 1];
      softmaxRow(cache.logits, n * vocabSize, vocabSize, probs, 0);
      loss += -Math.log(Math.max(probs[target], 1e-12));
      count++;
      const outBase = n * vocabSize;
      for (let v = 0; v < vocabSize; v++) {
        dlogits[outBase + v] = probs[v] - (v === target ? 1 : 0);
      }
    }
  }
  const invCount = count > 0 ? 1 / count : 0;
  for (let i = 0; i < dlogits.length; i++) dlogits[i] *= invCount;
  loss *= invCount;

  // Output head gradient and dx.
  const dxFinal = new DTYPE(N * dim);
  for (let n = 0; n < N; n++) {
    const dBase = n * vocabSize;
    const hBase = n * dim;
    for (let v = 0; v < vocabSize; v++) {
      const d = dlogits[dBase + v];
      if (d === 0) continue;
      const eBase = v * dim;
      for (let k = 0; k < dim; k++) {
        grads.tokenEmb[eBase + k] += d * cache.hf[hBase + k];
        dxFinal[hBase + k] += d * model.tokenEmb[eBase + k];
      }
    }
  }

  let dx = new DTYPE(N * dim);
  rmsnormBackward(
    cache.x,
    dxFinal,
    model.gFinal,
    dx,
    grads.gFinal,
    N,
    dim,
  );

  for (let layer = layers - 1; layer >= 0; layer--) {
    const params = model.layers[layer];
    const gp = grads.layers[layer];
    const c = cache.layers[layer];

    // MLP branch.
    for (let n = 0; n < N; n++) {
      const gzBase = n * mlp;
      const dBase = n * dim;
      for (let f = 0; f < mlp; f++) {
        const g = c.gz[gzBase + f];
        for (let d = 0; d < dim; d++) {
          gp.w2[f * dim + d] += g * dx[dBase + d];
        }
      }
    }

    const dgz = new DTYPE(N * mlp);
    for (let n = 0; n < N; n++) {
      const gzBase = n * mlp;
      const dBase = n * dim;
      for (let f = 0; f < mlp; f++) {
        let sum = 0;
        for (let d = 0; d < dim; d++) {
          sum += dx[dBase + d] * params.w2[f * dim + d];
        }
        dgz[gzBase + f] = sum;
      }
    }

    const dz = new DTYPE(N * mlp);
    for (let i = 0; i < N * mlp; i++) {
      dz[i] = dgz[i] * geluGrad(c.z[i]);
    }

    for (let n = 0; n < N; n++) {
      const h2Base = n * dim;
      const dzBase = n * mlp;
      for (let d = 0; d < dim; d++) {
        const h = c.h2[h2Base + d];
        for (let f = 0; f < mlp; f++) {
          gp.w1[d * mlp + f] += h * dz[dzBase + f];
        }
      }
    }

    const dh2 = new DTYPE(N * dim);
    for (let n = 0; n < N; n++) {
      const h2Base = n * dim;
      const dzBase = n * mlp;
      for (let d = 0; d < dim; d++) {
        let sum = 0;
        for (let f = 0; f < mlp; f++) {
          sum += dz[dzBase + f] * params.w1[d * mlp + f];
        }
        dh2[h2Base + d] = sum;
      }
    }

    const dx1Add = new DTYPE(N * dim);
    rmsnormBackward(
      c.x1,
      dh2,
      params.g2,
      dx1Add,
      gp.g2,
      N,
      dim,
    );

    const dx1 = new DTYPE(N * dim);
    for (let i = 0; i < N * dim; i++) dx1[i] = dx[i] + dx1Add[i];

    // Attention branch.
    for (let n = 0; n < N; n++) {
      const aBase = n * dim;
      for (let d = 0; d < dim; d++) {
        const a = c.attnOut[aBase + d];
        for (let out = 0; out < dim; out++) {
          gp.wo[d * dim + out] += a * dx1[aBase + out];
        }
      }
    }

    const dAttnOut = new DTYPE(N * dim);
    for (let n = 0; n < N; n++) {
      const aBase = n * dim;
      for (let d = 0; d < dim; d++) {
        let sum = 0;
        for (let out = 0; out < dim; out++) {
          sum += dx1[aBase + out] * params.wo[d * dim + out];
        }
        dAttnOut[aBase + d] = sum;
      }
    }

    const dqkv = new DTYPE(N * dim * 3);
    const scale = 1 / Math.sqrt(headDim);
    for (let b = 0; b < B; b++) {
      for (let h = 0; h < heads; h++) {
        const headOffset = h * headDim;
        for (let t = 0; t < T; t++) {
          const row = (b * heads + h) * T * T + t * T;
          const dAttn = new DTYPE(T);
          for (let j = 0; j <= t; j++) {
            let sum = 0;
            const outBase = (b * T + t) * dim + headOffset;
            const vBase = (b * T + j) * dim * 3 + dim * 2 + headOffset;
            for (let d = 0; d < headDim; d++) {
              sum += dAttnOut[outBase + d] * cache.layers[layer].qkv[vBase + d];
            }
            dAttn[j] = sum;
          }
          let dot = 0;
          for (let j = 0; j <= t; j++) {
            dot += cache.layers[layer].attn[row + j] * dAttn[j];
          }
          const dOutBase = (b * T + t) * dim + headOffset;
          for (let j = 0; j <= t; j++) {
            const ds = cache.layers[layer].attn[row + j] * (dAttn[j] - dot);
            const qBase = (b * T + t) * dim * 3 + headOffset;
            const kBase = (b * T + j) * dim * 3 + dim + headOffset;
            const vBase = (b * T + j) * dim * 3 + dim * 2 + headOffset;
            const dqBase = (b * T + t) * dim * 3 + headOffset;
            const dkBase = (b * T + j) * dim * 3 + dim + headOffset;
            const dvBase = (b * T + j) * dim * 3 + dim * 2 + headOffset;
            for (let d = 0; d < headDim; d++) {
              dqkv[dqBase + d] += ds * cache.layers[layer].qkv[kBase + d] * scale;
              dqkv[dkBase + d] += ds * cache.layers[layer].qkv[qBase + d] * scale;
              dqkv[dvBase + d] += cache.layers[layer].attn[row + j] * dAttnOut[dOutBase + d];
            }
          }
        }
      }
    }

    for (let n = 0; n < N; n++) {
      const hBase = n * dim;
      const qBase = n * dim * 3;
      for (let d = 0; d < dim; d++) {
        const h = c.h1[hBase + d];
        for (let j = 0; j < dim * 3; j++) {
          gp.wqkv[d * dim * 3 + j] += h * dqkv[qBase + j];
        }
      }
    }

    const dh1 = new DTYPE(N * dim);
    for (let n = 0; n < N; n++) {
      const hBase = n * dim;
      const qBase = n * dim * 3;
      for (let d = 0; d < dim; d++) {
        let sum = 0;
        for (let j = 0; j < dim * 3; j++) {
          sum += dqkv[qBase + j] * params.wqkv[d * dim * 3 + j];
        }
        dh1[hBase + d] = sum;
      }
    }

    const dxAdd = new DTYPE(N * dim);
    rmsnormBackward(
      c.input,
      dh1,
      params.g1,
      dxAdd,
      gp.g1,
      N,
      dim,
    );

    const nextDx = new DTYPE(N * dim);
    for (let i = 0; i < N * dim; i++) nextDx[i] = dx1[i] + dxAdd[i];
    dx = nextDx;
  }

  // Embedding gradients.
  for (let b = 0; b < B; b++) {
    for (let t = 0; t < T; t++) {
      const token = tokens[b * T + t];
      const src = (b * T + t) * dim;
      const tok = token * dim;
      const pos = t * dim;
      for (let d = 0; d < dim; d++) {
        grads.tokenEmb[tok + d] += dx[src + d];
        grads.posEmb[pos + d] += dx[src + d];
      }
    }
  }

  return { loss, cache };
}

function computeLoss(model, tokens, config) {
  const { vocabSize, context } = config;
  const B = tokens.length / context;
  const T = context;
  const cache = forward(model, tokens, config, false);
  let loss = 0;
  let count = 0;
  const probs = new DTYPE(vocabSize);
  for (let b = 0; b < B; b++) {
    for (let t = 0; t < T - 1; t++) {
      const n = b * T + t;
      const target = tokens[n + 1];
      softmaxRow(cache.logits, n * vocabSize, vocabSize, probs, 0);
      loss += -Math.log(Math.max(probs[target], 1e-12));
      count++;
    }
  }
  return count > 0 ? loss / count : 0;
}

/* ------------------------------------------------------------------ */
/* Optimizer                                                          */
/* ------------------------------------------------------------------ */

function eachParam(model, fn) {
  fn(model.tokenEmb, "tokenEmb");
  fn(model.posEmb, "posEmb");
  for (const layer of model.layers) {
    fn(layer.g1, "g1");
    fn(layer.wqkv, "wqkv");
    fn(layer.wo, "wo");
    fn(layer.g2, "g2");
    fn(layer.w1, "w1");
    fn(layer.w2, "w2");
  }
  fn(model.gFinal, "gFinal");
}

function makeAdam(model) {
  const state = [];
  eachParam(model, (param) => {
    state.push({ m: new DTYPE(param.length), v: new DTYPE(param.length) });
  });
  return state;
}

function applyAdam(model, grads, state, step, lr) {
  const beta1 = 0.9;
  const beta2 = 0.999;
  const eps = 1e-8;
  const bias1 = 1 - Math.pow(beta1, step);
  const bias2 = 1 - Math.pow(beta2, step);
  let index = 0;
  const update = (param, grad) => {
    const slot = state[index++];
    for (let i = 0; i < param.length; i++) {
      const g = grad[i];
      slot.m[i] = beta1 * slot.m[i] + (1 - beta1) * g;
      slot.v[i] = beta2 * slot.v[i] + (1 - beta2) * g * g;
      const m = slot.m[i] / bias1;
      const v = slot.v[i] / bias2;
      param[i] -= (lr * m) / (Math.sqrt(v) + eps);
    }
  };
  update(model.tokenEmb, grads.tokenEmb);
  update(model.posEmb, grads.posEmb);
  for (let layer = 0; layer < model.layers.length; layer++) {
    const p = model.layers[layer];
    const g = grads.layers[layer];
    update(p.g1, g.g1);
    update(p.wqkv, g.wqkv);
    update(p.wo, g.wo);
    update(p.g2, g.g2);
    update(p.w1, g.w1);
    update(p.w2, g.w2);
  }
  update(model.gFinal, grads.gFinal);
}

function clipGradients(grads, maxNorm) {
  let sumSq = 0;
  const each = (grad) => {
    for (let i = 0; i < grad.length; i++) sumSq += grad[i] * grad[i];
  };
  each(grads.tokenEmb);
  each(grads.posEmb);
  for (const layer of grads.layers) {
    each(layer.g1);
    each(layer.wqkv);
    each(layer.wo);
    each(layer.g2);
    each(layer.w1);
    each(layer.w2);
  }
  each(grads.gFinal);
  const norm = Math.sqrt(sumSq);
  if (norm <= maxNorm) return norm;
  const scale = maxNorm / (norm + 1e-12);
  const mul = (grad) => {
    for (let i = 0; i < grad.length; i++) grad[i] *= scale;
  };
  mul(grads.tokenEmb);
  mul(grads.posEmb);
  for (const layer of grads.layers) {
    mul(layer.g1);
    mul(layer.wqkv);
    mul(layer.wo);
    mul(layer.g2);
    mul(layer.w1);
    mul(layer.w2);
  }
  mul(grads.gFinal);
  return norm;
}

/* ------------------------------------------------------------------ */
/* Sampling and generation                                            */
/* ------------------------------------------------------------------ */

function sampleLogits(logits, offset, vocabSize, options, random) {
  const temperature = Math.max(0.01, options.temperature ?? 1);
  const topK = Math.max(1, Math.min(vocabSize, options.topK ?? 40));
  const candidates = [];
  for (let i = 0; i < vocabSize; i++) {
    candidates.push({ index: i, value: logits[offset + i] / temperature });
  }
  candidates.sort((a, b) => b.value - a.value);
  const top = candidates.slice(0, topK);
  let max = top[0].value;
  let sum = 0;
  for (const item of top) {
    item.weight = Math.exp(item.value - max);
    sum += item.weight;
  }
  let target = random() * sum;
  for (const item of top) {
    target -= item.weight;
    if (target <= 0) return item.index;
  }
  return top[top.length - 1].index;
}

function generate(model, vocab, prompt, config, options = {}) {
  const random = mulberry32(options.seed ?? 12345);
  const maxTokens = options.maxTokens ?? 24;
  const index = new Map(vocab.map((word, i) => [word, i]));
  const context = config.context;
  const tokens = [BOS];
  for (const word of tokenizeWords(prompt)) {
    tokens.push(index.get(word) ?? UNK);
  }
  if (tokens.length > context) tokens.splice(0, tokens.length - context);

  const out = [];
  for (let step = 0; step < maxTokens; step++) {
    const window = tokens.slice(-context);
    while (window.length < context) window.unshift(BOS);
    const cache = forward(model, window, config, false);
    const last = (context - 1) * config.vocabSize;
    const next = sampleLogits(cache.logits, last, config.vocabSize, options, random);
    if (next === EOS) break;
    out.push(next);
    tokens.push(next);
  }
  return out;
}

/* ------------------------------------------------------------------ */
/* Gradient check                                                     */
/* ------------------------------------------------------------------ */

function gradientCheck() {
  console.log("Running gradient check...");
  DTYPE = Float64Array;
  const config = makeConfig({
    vocabSize: 12,
    context: 4,
    dim: 8,
    layers: args.includes("--check-emb") ? 0 : 1,
    heads: 1,
    mlp: 16,
  });
  const random = mulberry32(99);
  const model = makeModel(config, random);
  if (args.includes("--check-no-mlp")) {
    for (const layer of model.layers) {
      layer.w1.fill(0);
      layer.w2.fill(0);
    }
  }
  if (args.includes("--check-no-attn")) {
    for (const layer of model.layers) {
      layer.wqkv.fill(0);
      layer.wo.fill(0);
    }
  }
  if (args.includes("--check-no-wo")) {
    for (const layer of model.layers) {
      layer.wo.fill(0);
    }
  }
  const tokens = [1, 3, 4, 5, 2, 6, 7, 8];
  const grads = zeroGrads(model, config);
  const { loss } = lossAndGrad(model, grads, tokens, config);
  console.log("loss", loss);

  const directionRandom = mulberry32(123);
  const checkGroup = (name, array, grad) => {
    const direction = new DTYPE(array.length);
    let norm = 0;
    for (let i = 0; i < direction.length; i++) {
      const value = directionRandom() - 0.5;
      direction[i] = value;
      norm += value * value;
    }
    norm = Math.sqrt(norm) || 1;
    for (let i = 0; i < direction.length; i++) direction[i] /= norm;

    const original = new DTYPE(array);
    const eps = 1e-5;
    for (let i = 0; i < array.length; i++) {
      array[i] = original[i] + eps * direction[i];
    }
    const plus = computeLoss(model, tokens, config);
    for (let i = 0; i < array.length; i++) {
      array[i] = original[i] - eps * direction[i];
    }
    const minus = computeLoss(model, tokens, config);
    for (let i = 0; i < array.length; i++) array[i] = original[i];

    let analytic = 0;
    for (let i = 0; i < direction.length; i++) analytic += grad[i] * direction[i];
    const numeric = (plus - minus) / (2 * eps);
    const denom = Math.max(1e-5, Math.abs(numeric), Math.abs(analytic));
    const error = Math.abs(numeric - analytic) / denom;
    console.log(
      `${name.padEnd(9)} numeric=${numeric.toFixed(6)} analytic=${analytic.toFixed(6)} rel=${error.toFixed(4)}`,
    );
    return error;
  };

  let worst = 0;
  worst = Math.max(worst, checkGroup("tokenEmb", model.tokenEmb, grads.tokenEmb));
  worst = Math.max(worst, checkGroup("posEmb", model.posEmb, grads.posEmb));
  if (config.layers > 0) {
    worst = Math.max(worst, checkGroup("g1", model.layers[0].g1, grads.layers[0].g1));
    worst = Math.max(worst, checkGroup("wqkv", model.layers[0].wqkv, grads.layers[0].wqkv));
    worst = Math.max(worst, checkGroup("wo", model.layers[0].wo, grads.layers[0].wo));
    if (!args.includes("--check-no-mlp")) {
      worst = Math.max(worst, checkGroup("g2", model.layers[0].g2, grads.layers[0].g2));
      worst = Math.max(worst, checkGroup("w1", model.layers[0].w1, grads.layers[0].w1));
      worst = Math.max(worst, checkGroup("w2", model.layers[0].w2, grads.layers[0].w2));
    }
  }
  worst = Math.max(worst, checkGroup("gFinal", model.gFinal, grads.gFinal));

  console.log(`worst relative error: ${worst.toFixed(4)}`);
  if (args.includes("--check-pos")) {
    for (let t = 0; t < config.context; t++) {
      let maxError = 0;
      let at = -1;
      for (let d = 0; d < config.dim; d++) {
        const index = t * config.dim + d;
        const original = model.posEmb[index];
        const eps = 1e-5;
        model.posEmb[index] = original + eps;
        const plus = computeLoss(model, tokens, config);
        model.posEmb[index] = original - eps;
        const minus = computeLoss(model, tokens, config);
        model.posEmb[index] = original;
        const numeric = (plus - minus) / (2 * eps);
        const analytic = grads.posEmb[index];
        const error = Math.abs(numeric - analytic);
        if (error > maxError) {
          maxError = error;
          at = d;
        }
      }
      console.log(`pos ${t} worst dim ${at} abs=${maxError.toExponential(3)}`);
    }
  }
  if (worst > 0.02) {
    throw new Error("Gradient check failed");
  }
}

/* ------------------------------------------------------------------ */
/* Export                                                             */
/* ------------------------------------------------------------------ */

function toBase64Float32(array) {
  return Buffer.from(array.buffer, array.byteOffset, array.byteLength).toString(
    "base64",
  );
}

function exportCheckpoint(model, config, vocab, loss, samples) {
  const weights = {
    tokenEmb: toBase64Float32(model.tokenEmb),
    posEmb: toBase64Float32(model.posEmb),
    layers: model.layers.map((layer) => ({
      g1: toBase64Float32(layer.g1),
      wqkv: toBase64Float32(layer.wqkv),
      wo: toBase64Float32(layer.wo),
      g2: toBase64Float32(layer.g2),
      w1: toBase64Float32(layer.w1),
      w2: toBase64Float32(layer.w2),
    })),
    gFinal: toBase64Float32(model.gFinal),
  };
  const meta = {
    name: "tiny-gpt-alice",
    source:
      "Lewis Carroll, Alice's Adventures in Wonderland (1865), Project Gutenberg #11. Public domain.",
    license: "Public domain text; model weights generated by this repository.",
    training: {
      corpusWords: 4338,
      steps: STEPS,
      batch: BATCH,
      context: config.context,
      dim: config.dim,
      layers: config.layers,
      heads: config.heads,
      mlp: config.mlp,
      loss,
    },
  };
  const content = `/*
 * Generated by scripts/train-tiny-gpt.mjs. Do not edit by hand.
 *
 * A tiny decoder-only transformer trained on a public-domain excerpt from
 * Alice's Adventures in Wonderland. The browser demo loads the float32
 * weights and quantizes them to int8 at startup.
 */

import type { TinyGptCheckpoint } from "../lib/tinyGpt";

export const TINY_GPT_CHECKPOINT: TinyGptCheckpoint = {
  config: ${JSON.stringify(
    {
      vocabSize: config.vocabSize,
      context: config.context,
      dim: config.dim,
      layers: config.layers,
      heads: config.heads,
      mlp: config.mlp,
    },
    null,
    2,
  )},
  meta: ${JSON.stringify(meta, null, 2)},
  vocab: ${JSON.stringify(vocab)},
  weights: ${JSON.stringify(weights, null, 2)},
  samples: ${JSON.stringify(samples, null, 2)},
};
`;
  writeFileSync(OUTPUT_PATH, content);
  console.log(`wrote ${OUTPUT_PATH}`);
}

/* ------------------------------------------------------------------ */
/* Main                                                               */
/* ------------------------------------------------------------------ */

function main() {
  if (CHECK) {
    gradientCheck();
    return;
  }

  const corpus = readFileSync(CORPUS_PATH, "utf8");
  const words = tokenizeWords(corpus);
  const vocab = buildVocabulary(words, VOCAB_LIMIT);
  const index = new Map(vocab.map((word, i) => [word, i]));
  const encoded = encodeTokens(words, index);
  const stream = [];
  for (let i = 0; i < encoded.length; i++) {
    stream.push(encoded[i]);
    // Paragraph boundary approximated by the corpus newlines.
  }
  // Put an <eos> between paragraphs.
  const paragraphs = corpus.split(/\n\s*\n/);
  const withEos = [];
  for (const paragraph of paragraphs) {
    for (const word of tokenizeWords(paragraph)) {
      withEos.push(index.get(word) ?? UNK);
    }
    withEos.push(EOS);
  }

  const config = makeConfig({
    vocabSize: vocab.length,
    context: CONTEXT,
    dim: DIM,
    layers: LAYERS,
    heads: HEADS,
    mlp: MLP,
  });
  console.log(
    `vocab=${config.vocabSize} params=${countParams(config)} context=${config.context} dim=${config.dim} layers=${config.layers} heads=${config.heads}`,
  );
  console.log(`corpus=${words.length} words, ${withEos.length} tokens`);

  const random = mulberry32(SEED);
  const model = makeModel(config, random);
  const grads = zeroGrads(model, config);
  const adam = makeAdam(model);
  const sampleRandom = mulberry32(SEED + 1);

  // Hold out the final 15% of the token stream. The last paragraphs are used
  // only to choose a checkpoint, never to update the weights.
  const validationStart = Math.floor(withEos.length * 0.85);
  const validationTokens = withEos
    .slice(validationStart)
    .slice(
      0,
      Math.floor((withEos.length - validationStart) / config.context) *
        config.context,
    );

  let bestLoss = Infinity;
  let bestModel = null;
  let bestStep = 0;
  let smooth = 0;
  const trainingRandom = mulberry32(SEED + 2);
  const VALIDATE_EVERY = 100;

  for (let step = 1; step <= STEPS; step++) {
    // Reset gradients.
    eachGrad(grads, (grad) => grad.fill(0));

    const tokens = new Uint16Array(BATCH * config.context);
    for (let b = 0; b < BATCH; b++) {
      const start =
        Math.floor(trainingRandom() * (withEos.length - config.context - 1)) + 0;
      for (let t = 0; t < config.context; t++) {
        tokens[b * config.context + t] = withEos[start + t];
      }
    }

    const { loss } = lossAndGrad(model, grads, tokens, config);
    clipGradients(grads, 1.0);
    const progress = step / STEPS;
    const lr = LR * (0.1 + 0.9 * 0.5 * (1 + Math.cos(Math.PI * progress)));
    applyAdam(model, grads, adam, step, lr);

    smooth = smooth === 0 ? loss : smooth * 0.95 + loss * 0.05;
    if (step % 25 === 0 || step === 1) {
      console.log(
        `step ${step}/${STEPS} loss=${loss.toFixed(4)} smooth=${smooth.toFixed(4)} lr=${lr.toFixed(5)}`,
      );
    }
    if (step % VALIDATE_EVERY === 0 || step === STEPS) {
      const validationLoss = computeLoss(model, validationTokens, config);
      if (validationLoss < bestLoss) {
        bestLoss = validationLoss;
        bestModel = cloneModel(model, config);
        bestStep = step;
      }
      if (step % 500 === 0 || step === STEPS) {
        console.log(
          `validation step ${step} loss=${validationLoss.toFixed(4)} best=${bestLoss.toFixed(4)} @${bestStep}`,
        );
      }
    }

    if (step % SAMPLE_EVERY === 0) {
      const sample = generate(model, vocab, "Alice was", config, {
        maxTokens: 24,
        temperature: 0.8,
        topK: 40,
        seed: SEED + step,
      });
      console.log(
        "  sample: " +
          sample.map((id) => vocab[id] ?? "<unk>").join(" "),
      );
    }
  }

  const finalModel = bestModel ?? model;
  console.log(`best validation loss ${bestLoss.toFixed(4)} at step ${bestStep}`);
  const samples = [];
  for (const prompt of ["Alice was", "the rabbit", "she said", "the door"]) {
    const ids = generate(finalModel, vocab, prompt, config, {
      maxTokens: 28,
      temperature: 0.75,
      topK: 40,
      seed: sampleRandom() * 1e9,
    });
    const text = ids.map((id) => vocab[id] ?? "<unk>").join(" ");
    samples.push({ prompt, text });
    console.log(`sample ${prompt}: ${text}`);
  }

  exportCheckpoint(finalModel, config, vocab, bestLoss, samples);
}

function eachGrad(grads, fn) {
  fn(grads.tokenEmb);
  fn(grads.posEmb);
  for (const layer of grads.layers) {
    fn(layer.g1);
    fn(layer.wqkv);
    fn(layer.wo);
    fn(layer.g2);
    fn(layer.w1);
    fn(layer.w2);
  }
  fn(grads.gFinal);
}

function countParams(config) {
  const { vocabSize, context, dim, layers, mlp } = config;
  return (
    vocabSize * dim +
    context * dim +
    layers * (dim + dim * dim * 3 + dim * dim + dim + dim * mlp + mlp * dim) +
    dim
  );
}

function cloneModel(model, config) {
  const clone = {
    tokenEmb: new DTYPE(model.tokenEmb),
    posEmb: new DTYPE(model.posEmb),
    layers: model.layers.map((layer) => ({
      g1: new DTYPE(layer.g1),
      wqkv: new DTYPE(layer.wqkv),
      wo: new DTYPE(layer.wo),
      g2: new DTYPE(layer.g2),
      w1: new DTYPE(layer.w1),
      w2: new DTYPE(layer.w2),
    })),
    gFinal: new DTYPE(model.gFinal),
  };
  return clone;
}

main();
