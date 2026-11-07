import {
  WORD_VECTOR_BITS,
  WORD_VECTOR_DATA,
  WORD_VECTOR_WORDS,
} from "../assets/word-vectors";

/*
 * A tiny word-level language model that runs entirely in the browser.
 *
 * Training is done with ordinary floats: a real-valued query projection is
 * quantized to ternary weights with a straight-through estimator, then Adam
 * updates the shadow weights. Inference is integer-only:
 *
 *   context bits -> ternary query projection (popcount) -> query bits
 *   score        = popcount(xnor(query, key)) / sqrt(dim)
 *   weights      = Q15 fixed-point softmax
 *
 * The key memory is the bundled Word2Bits subset. Unknown words get a stable
 * hash vector so a user can still train on their own vocabulary.
 */

export const VECTOR_BITS = WORD_VECTOR_BITS;
export const VECTOR_WORDS = VECTOR_BITS / 32;
export const CONTEXT_WINDOW = 3;
export const MAX_VOCABULARY = 128;
export const MAX_SAMPLES = 1200;
export const DEFAULT_EPOCHS = 12;
export const LEARNING_RATE = 0.03;

const TWO_POW_NEG_Q15 = [
  32768, 32066, 31379, 30706, 30048, 29405, 28774, 28158, 27554, 26964, 26386,
  25821, 25268, 24726, 24196, 23678, 23170, 22674, 22188, 21713, 21247, 20792,
  20347, 19911, 19484, 19066, 18658, 18258, 17867, 17484, 17109, 16743, 16384,
] as const;

/* round(256 / sqrt(64)). Kept as an integer so the inference path never needs
 * a floating-point square root. */
const INV_SQRT_Q8 = 32;

let bundledLexicon: Map<string, Uint32Array> | null = null;
let bundledRanks: Map<string, number> | null = null;

function decodeVector(encoded: string): Uint32Array {
  const binary = globalThis.atob(encoded);
  const vector = new Uint32Array(VECTOR_WORDS);
  for (let i = 0; i < binary.length; i++) {
    const index = i >> 2;
    vector[index] = vector[index]! | (binary.charCodeAt(i) << ((i & 3) * 8));
  }
  return vector;
}

function getBundledLexicon(): Map<string, Uint32Array> {
  if (bundledLexicon !== null) {
    return bundledLexicon;
  }
  const lexicon = new Map<string, Uint32Array>();
  const ranks = new Map<string, number>();
  for (let i = 0; i < WORD_VECTOR_WORDS.length; i++) {
    const word = WORD_VECTOR_WORDS[i]!;
    lexicon.set(word, decodeVector(WORD_VECTOR_DATA[i]!));
    ranks.set(word, i);
  }
  bundledLexicon = lexicon;
  bundledRanks = ranks;
  return lexicon;
}

export function bundledWordCount(): number {
  return WORD_VECTOR_WORDS.length;
}

export function hasBundledVector(word: string): boolean {
  return getBundledLexicon().has(word.toLowerCase());
}

export function bundledRank(word: string): number {
  getBundledLexicon();
  return bundledRanks?.get(word.toLowerCase()) ?? Number.MAX_SAFE_INTEGER;
}

export function getWordVector(word: string): Uint32Array {
  const lower = word.toLowerCase();
  const bundled = getBundledLexicon().get(lower);
  return bundled ? new Uint32Array(bundled) : hashVector(lower);
}

function hashVector(word: string): Uint32Array {
  const vector = new Uint32Array(VECTOR_WORDS);
  let hash = 0x811c9dc5;
  for (let wordIndex = 0; wordIndex < VECTOR_WORDS; wordIndex++) {
    for (let i = 0; i < word.length; i++) {
      hash ^= word.charCodeAt(i);
      hash = Math.imul(hash, 0x01000193);
    }
    hash ^= 0x9e3779b9 + wordIndex;
    hash = Math.imul(hash, 0x85ebca6b);
    vector[wordIndex] = hash >>> 0;
  }
  if (vector.every((value) => value === 0)) {
    vector[0] = 1;
  }
  return vector;
}

function popcount32(value: number): number {
  let x = value >>> 0;
  x -= (x >>> 1) & 0x55555555;
  x = (x & 0x33333333) + ((x >>> 2) & 0x33333333);
  x = (x + (x >>> 4)) & 0x0f0f0f0f;
  return Math.imul(x, 0x01010101) >>> 24;
}

function expNegQ15(xQ8: number): number {
  if (xQ8 <= 0) {
    return 32768;
  }
  if (xQ8 > 2048) {
    return 0;
  }
  const yQ8 = (xQ8 * 369) >> 8;
  const integerPart = yQ8 >> 8;
  const fraction = yQ8 & 255;
  const index = fraction >> 3;
  const remainder = fraction & 7;
  const a = TWO_POW_NEG_Q15[index]!;
  const b = TWO_POW_NEG_Q15[index + 1]!;
  const value = a - (((a - b) * remainder) >> 3);
  return value >> integerPart;
}

function majorityVector(vectors: readonly Uint32Array[]): Uint32Array {
  const out = new Uint32Array(VECTOR_WORDS);
  if (vectors.length === 0) {
    return out;
  }
  if (vectors.length === 1) {
    out.set(vectors[0]!);
    return out;
  }

  const threshold = vectors.length / 2;
  const latest = vectors[vectors.length - 1]!;
  for (let word = 0; word < VECTOR_WORDS; word++) {
    let result = 0;
    for (let bit = 0; bit < 32; bit++) {
      let ones = 0;
      for (const vector of vectors) {
        if (((vector[word]! >>> bit) & 1) === 1) {
          ones++;
        }
      }
      const latestBit = (latest[word]! >>> bit) & 1;
      if (ones > threshold || (ones === threshold && latestBit === 1)) {
        result |= 1 << bit;
      }
    }
    out[word] = result >>> 0;
  }
  return out;
}

function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function shuffle<T>(values: T[], random: () => number): void {
  for (let i = values.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    const value = values[i]!;
    values[i] = values[j]!;
    values[j] = value;
  }
}

export interface Token {
  readonly text: string;
  readonly start: number;
  readonly end: number;
}

const TOKEN_PATTERN = /[A-Za-z0-9']+/g;

export function tokenize(text: string): Token[] {
  const tokens: Token[] = [];
  TOKEN_PATTERN.lastIndex = 0;
  let match: RegExpExecArray | null;
  while ((match = TOKEN_PATTERN.exec(text)) !== null) {
    tokens.push({
      text: match[0],
      start: match.index,
      end: match.index + match[0].length,
    });
  }
  return tokens;
}

export interface LanguageTrainingStats {
  readonly vocabulary: number;
  readonly samples: number;
  readonly epochs: number;
  readonly parameters: number;
  readonly averageLoss: number;
  accuracy: number;
  readonly coverage: number;
  readonly contextWindow: number;
  readonly vectorBits: number;
  readonly vocabularyWords: readonly string[];
}

export interface TrainingProgress {
  readonly epoch: number;
  readonly epochs: number;
  readonly loss: number;
  readonly fraction: number;
}

export interface TrainOptions {
  readonly epochs?: number;
  readonly onProgress?: (progress: TrainingProgress) => void;
  readonly shouldCancel?: () => boolean;
}

export class TrainingCancelledError extends Error {
  constructor() {
    super("Training cancelled.");
    this.name = "TrainingCancelledError";
  }
}

export interface Completion {
  readonly word: string;
  readonly probability: number | null;
  readonly count: number;
  readonly source: "model" | "lexicon";
}

export interface CompletionResult {
  readonly mode: "next" | "prefix" | "empty";
  readonly prefix: string;
  readonly context: string;
  readonly items: readonly Completion[];
}

export interface CompletionOptions {
  /**
   * Words to skip while there are other candidates. The panel passes the
   * recent tokens here so accepting suggestions cannot bounce between the
   * same two words forever.
   */
  readonly avoid?: readonly string[];
}

interface Sample {
  readonly query: Uint32Array;
  readonly target: number;
}

function quantizeWeights(
  weights: Float32Array,
  ternary: Int8Array,
  pos: Uint32Array,
  neg: Uint32Array,
  delta: Int32Array,
): void {
  const dimension = VECTOR_BITS;
  for (let row = 0; row < dimension; row++) {
    let positive = 0;
    let negative = 0;
    const rowOffset = row * dimension;
    for (let word = 0; word < VECTOR_WORDS; word++) {
      let posWord = 0;
      let negWord = 0;
      const start = word * 32;
      const end = Math.min(start + 32, dimension);
      for (let bit = start; bit < end; bit++) {
        const value = weights[rowOffset + bit]!;
        const quantized = value > 0.5 ? 1 : value < -0.5 ? -1 : 0;
        ternary[rowOffset + bit] = quantized;
        if (quantized > 0) {
          posWord |= 1 << (bit - start);
        } else if (quantized < 0) {
          negWord |= 1 << (bit - start);
        }
      }
      pos[row * VECTOR_WORDS + word] = posWord >>> 0;
      neg[row * VECTOR_WORDS + word] = negWord >>> 0;
      positive += popcount32(posWord);
      negative += popcount32(negWord);
    }
    delta[row] = positive - negative;
  }
}

function projectQuery(
  pos: Uint32Array,
  neg: Uint32Array,
  delta: Int32Array,
  query: Uint32Array,
  out: Uint32Array,
): void {
  for (let row = 0; row < VECTOR_BITS; row++) {
    let positive = 0;
    let negative = 0;
    for (let word = 0; word < VECTOR_WORDS; word++) {
      const index = row * VECTOR_WORDS + word;
      positive += popcount32(pos[index]! & query[word]!);
      negative += popcount32(neg[index]! & query[word]!);
    }
    const dot = 2 * (positive - negative) - delta[row]!;
    const word = row >> 5;
    const bit = row & 31;
    if (dot >= 0) {
      out[word] = out[word]! | (1 << bit);
    } else {
      out[word] = out[word]! & ~(1 << bit);
    }
  }
}

function scoreKey(query: Uint32Array, key: Uint32Array): number {
  let matches = 0;
  for (let word = 0; word < VECTOR_WORDS; word++) {
    matches += popcount32(~(query[word]! ^ key[word]!));
  }
  return 2 * matches - VECTOR_BITS;
}

export class MiniLanguageModel {
  readonly stats: LanguageTrainingStats;

  private readonly vocabulary: readonly string[];
  private readonly vocabularyIndex: Map<string, number>;
  private readonly counts: readonly number[];
  private readonly keyWords: Uint32Array;
  private readonly ternary: Int8Array;
  private readonly pos: Uint32Array;
  private readonly neg: Uint32Array;
  private readonly delta: Int32Array;
  private readonly projected: Uint32Array;

  constructor(
    vocabulary: readonly string[],
    counts: readonly number[],
    keyWords: Uint32Array,
    weights: Float32Array,
    stats: LanguageTrainingStats,
  ) {
    this.vocabulary = vocabulary;
    this.counts = counts;
    this.keyWords = keyWords;
    this.stats = stats;
    this.vocabularyIndex = new Map(
      vocabulary.map((word, index) => [word, index]),
    );
    this.ternary = new Int8Array(VECTOR_BITS * VECTOR_BITS);
    this.pos = new Uint32Array(VECTOR_BITS * VECTOR_WORDS);
    this.neg = new Uint32Array(VECTOR_BITS * VECTOR_WORDS);
    this.delta = new Int32Array(VECTOR_BITS);
    this.projected = new Uint32Array(VECTOR_WORDS);
    this.quantize(weights);
  }

  private quantize(weights: Float32Array): void {
    quantizeWeights(weights, this.ternary, this.pos, this.neg, this.delta);
  }

  private contextVector(tokens: readonly Token[]): Uint32Array {
    const recent = tokens.slice(-CONTEXT_WINDOW).map((token) =>
      getWordVector(token.text),
    );
    return majorityVector(recent);
  }

  private integerLogits(context: Uint32Array): Int32Array {
    this.projected.fill(0);
    projectQuery(this.pos, this.neg, this.delta, context, this.projected);
    const logits = new Int32Array(this.vocabulary.length);
    for (let word = 0; word < this.vocabulary.length; word++) {
      const dot = scoreKey(
        this.projected,
        this.keyWords.subarray(
          word * VECTOR_WORDS,
          (word + 1) * VECTOR_WORDS,
        ),
      );
      logits[word] = dot * INV_SQRT_Q8;
    }
    return logits;
  }

  complete(
    text: string,
    limit = 6,
    options: CompletionOptions = {},
  ): CompletionResult {
    const allTokens = tokenize(text);
    const trailing = /([A-Za-z0-9']+)\s*$/.exec(text);
    const endsWithWord =
      trailing !== null && trailing.index + trailing[1]!.length === text.length;
    const prefix = endsWithWord ? trailing[1]!.toLowerCase() : "";
    const contextTokens = endsWithWord
      ? tokenize(text.slice(0, trailing!.index))
      : allTokens;
    const contextWord =
      contextTokens.length > 0
        ? contextTokens[contextTokens.length - 1]!.text
        : "";

    if (prefix.length > 0) {
      return {
        mode: "prefix",
        prefix,
        context: contextWord,
        items: this.prefixCompletions(prefix, limit),
      };
    }

    if (contextTokens.length === 0) {
      return {
        mode: "empty",
        prefix: "",
        context: "",
        items: this.frequentWords(limit),
      };
    }

    const logits = this.integerLogits(this.contextVector(contextTokens));
    let best = logits[0]!;
    for (let i = 1; i < logits.length; i++) {
      if (logits[i]! > best) {
        best = logits[i]!;
      }
    }

    const weights = new Float64Array(logits.length);
    let sum = 0;
    for (let i = 0; i < logits.length; i++) {
      const value = expNegQ15(best - logits[i]!);
      weights[i] = value;
      sum += value;
    }
    if (sum <= 0) {
      sum = 1;
    }

    const ordered = Array.from({ length: logits.length }, (_, index) => index)
      .sort((a, b) => logits[b]! - logits[a]!);
    const avoided = new Set(
      (options.avoid ?? []).map((word) => word.toLowerCase()),
    );
    const allowed =
      avoided.size === 0
        ? ordered
        : ordered.filter((index) => !avoided.has(this.vocabulary[index]!));
    const ranked = (allowed.length > 0 ? allowed : ordered).slice(0, limit);

    return {
      mode: "next",
      prefix: "",
      context: contextWord,
      items: ranked.map((index) => ({
        word: this.vocabulary[index]!,
        probability: weights[index]! / sum,
        count: this.counts[index]!,
        source: "model" as const,
      })),
    };
  }

  private prefixCompletions(prefix: string, limit: number): Completion[] {
    const matches: Completion[] = [];
    for (let index = 0; index < this.vocabulary.length; index++) {
      const word = this.vocabulary[index]!;
      if (word.startsWith(prefix)) {
        matches.push({
          word,
          probability: null,
          count: this.counts[index]!,
          source: "model",
        });
      }
    }
    matches.sort((a, b) => b.count - a.count || a.word.localeCompare(b.word));

    if (matches.length < limit) {
      const seen = new Set(matches.map((match) => match.word));
      const extra: Completion[] = [];
      for (const word of WORD_VECTOR_WORDS) {
        if (
          word.startsWith(prefix) &&
          !seen.has(word) &&
          extra.length < limit - matches.length
        ) {
          extra.push({
            word,
            probability: null,
            count: 0,
            source: "lexicon",
          });
        }
      }
      matches.push(...extra);
    }

    return matches.slice(0, limit);
  }

  private frequentWords(limit: number): Completion[] {
    return this.vocabulary
      .map((word, index) => ({
        word,
        probability: null,
        count: this.counts[index]!,
        source: "model" as const,
      }))
      .sort((a, b) => b.count - a.count)
      .slice(0, limit);
  }

  get vocabularyWords(): readonly string[] {
    return this.vocabulary;
  }

  evaluateQueries(
    queries: readonly Uint32Array[],
    targets: readonly number[],
  ): number {
    if (queries.length === 0) {
      return 0;
    }
    let correct = 0;
    for (let i = 0; i < queries.length; i++) {
      const logits = this.integerLogits(queries[i]!);
      let best = 0;
      for (let word = 1; word < logits.length; word++) {
        if (logits[word]! > logits[best]!) {
          best = word;
        }
      }
      if (best === targets[i]) {
        correct++;
      }
    }
    return correct / queries.length;
  }
}

interface PreparedTraining {
  readonly vocabulary: string[];
  readonly counts: number[];
  readonly keyWords: Uint32Array;
  readonly samples: Sample[];
  readonly keySigns: Int8Array;
  readonly coverage: number;
  readonly totalTokens: number;
}

function prepareTraining(text: string): PreparedTraining {
  const tokens = tokenize(text);
  if (tokens.length < 8) {
    throw new Error("Add at least eight words so the model has something to learn.");
  }

  const frequencies = new Map<string, number>();
  let bundledTokens = 0;
  for (const token of tokens) {
    const word = token.text.toLowerCase();
    frequencies.set(word, (frequencies.get(word) ?? 0) + 1);
    if (hasBundledVector(word)) {
      bundledTokens++;
    }
  }

  const vocabulary = [...frequencies.entries()]
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .slice(0, MAX_VOCABULARY)
    .map(([word]) => word);

  if (vocabulary.length < 2) {
    throw new Error("Use at least two distinct words to train the model.");
  }

  const counts = vocabulary.map((word) => frequencies.get(word) ?? 0);
  const index = new Map(vocabulary.map((word, position) => [word, position]));
  const keyWords = new Uint32Array(vocabulary.length * VECTOR_WORDS);
  const keySigns = new Int8Array(vocabulary.length * VECTOR_BITS);
  for (let i = 0; i < vocabulary.length; i++) {
    const vector = getWordVector(vocabulary[i]!);
    keyWords.set(vector, i * VECTOR_WORDS);
    for (let bit = 0; bit < VECTOR_BITS; bit++) {
      keySigns[i * VECTOR_BITS + bit] =
        ((vector[bit >> 5]! >>> (bit & 31)) & 1) === 1 ? 1 : -1;
    }
  }

  const positions: number[] = [];
  for (let position = 1; position < tokens.length; position++) {
    if (index.has(tokens[position]!.text.toLowerCase())) {
      positions.push(position);
    }
  }
  if (positions.length === 0) {
    throw new Error("The text does not contain a usable pair of words.");
  }

  const random = mulberry32(0x51f15e);
  shuffle(positions, random);
  const selected = positions.slice(0, MAX_SAMPLES);
  selected.sort((a, b) => a - b);

  const samples: Sample[] = selected.map((position) => {
    const contextTokens = tokens.slice(
      Math.max(0, position - CONTEXT_WINDOW),
      position,
    );
    const query = majorityVector(
      contextTokens.map((token) => getWordVector(token.text)),
    );
    return {
      query,
      target: index.get(tokens[position]!.text.toLowerCase())!,
    };
  });

  return {
    vocabulary,
    counts,
    keyWords,
    keySigns,
    samples,
    coverage: bundledTokens / tokens.length,
    totalTokens: tokens.length,
  };
}

export async function trainMiniLanguageModel(
  text: string,
  options: TrainOptions = {},
): Promise<MiniLanguageModel> {
  const epochs = Math.max(1, Math.min(20, options.epochs ?? DEFAULT_EPOCHS));
  const prepared = prepareTraining(text);
  const samples = prepared.samples;
  const vocabulary = prepared.vocabulary;
  const keyWords = prepared.keyWords;
  const keySigns = prepared.keySigns;

  const dimension = VECTOR_BITS;
  const weights = new Float32Array(dimension * dimension);
  for (let i = 0; i < dimension; i++) {
    weights[i * dimension + i] = 1;
  }

  const firstMoment = new Float32Array(weights.length);
  const secondMoment = new Float32Array(weights.length);
  const gradients = new Float32Array(weights.length);
  const ternary = new Int8Array(weights.length);
  const pos = new Uint32Array(dimension * VECTOR_WORDS);
  const neg = new Uint32Array(dimension * VECTOR_WORDS);
  const delta = new Int32Array(dimension);
  const projected = new Uint32Array(VECTOR_WORDS);
  const scores = new Float64Array(vocabulary.length);
  const alpha = new Float64Array(vocabulary.length);
  const dscore = new Float64Array(vocabulary.length);
  const dq = new Float64Array(dimension);
  const inputSigns = new Int8Array(dimension);
  const inverseSqrt = 1 / Math.sqrt(dimension);

  const beta1 = 0.9;
  const beta2 = 0.999;
  const epsilon = 1e-8;
  let beta1Power = 1;
  let beta2Power = 1;
  let totalLoss = 0;
  const random = mulberry32(0x1234abcd);

  const randomOrder = () => {
    const order = Array.from({ length: samples.length }, (_, index) => index);
    shuffle(order, random);
    return order;
  };

  for (let epoch = 0; epoch < epochs; epoch++) {
    const order = randomOrder();
    let epochLoss = 0;

    for (let position = 0; position < order.length; position++) {
      if (options.shouldCancel?.()) {
        throw new TrainingCancelledError();
      }

      const sample = samples[order[position]!]!;
      quantizeWeights(weights, ternary, pos, neg, delta);
      projected.fill(0);
      projectQuery(pos, neg, delta, sample.query, projected);

      for (let i = 0; i < dimension; i++) {
        inputSigns[i] =
          ((sample.query[i >> 5]! >>> (i & 31)) & 1) === 1 ? 1 : -1;
      }

      let maxScore = Number.NEGATIVE_INFINITY;
      for (let word = 0; word < vocabulary.length; word++) {
        let matches = 0;
        const offset = word * VECTOR_WORDS;
        for (let part = 0; part < VECTOR_WORDS; part++) {
          matches += popcount32(
            ~(projected[part]! ^ keyWords[offset + part]!),
          );
        }
        const score = (2 * matches - dimension) * inverseSqrt;
        scores[word] = score;
        if (score > maxScore) {
          maxScore = score;
        }
      }

      let sumExp = 0;
      for (let word = 0; word < vocabulary.length; word++) {
        const value = Math.exp(scores[word]! - maxScore);
        alpha[word] = value;
        sumExp += value;
      }
      for (let word = 0; word < vocabulary.length; word++) {
        alpha[word] = alpha[word]! / sumExp;
      }
      epochLoss += -Math.log(alpha[sample.target]! + 1e-9);

      for (let word = 0; word < vocabulary.length; word++) {
        dscore[word] = alpha[word]! - (word === sample.target ? 1 : 0);
      }

      dq.fill(0);
      for (let word = 0; word < vocabulary.length; word++) {
        const weight = dscore[word]!;
        if (weight === 0) {
          continue;
        }
        const offset = word * dimension;
        for (let i = 0; i < dimension; i++) {
          dq[i] = dq[i]! + weight * keySigns[offset + i]!;
        }
      }
      for (let i = 0; i < dimension; i++) {
        dq[i] = dq[i]! * inverseSqrt;
      }

      for (let i = 0; i < dimension; i++) {
        const row = i * dimension;
        const scaled = dq[i]!;
        for (let j = 0; j < dimension; j++) {
          gradients[row + j] = scaled * inputSigns[j]!;
        }
      }

      beta1Power *= beta1;
      beta2Power *= beta2;
      const bias1 = 1 - beta1Power;
      const bias2 = 1 - beta2Power;

      for (let index = 0; index < weights.length; index++) {
        let gradient = gradients[index]!;
        if (gradient > 2) {
          gradient = 2;
        } else if (gradient < -2) {
          gradient = -2;
        }
        firstMoment[index] =
          beta1 * firstMoment[index]! + (1 - beta1) * gradient;
        secondMoment[index] =
          beta2 * secondMoment[index]! + (1 - beta2) * gradient * gradient;
        const m = firstMoment[index]! / bias1;
        const v = secondMoment[index]! / bias2;
        weights[index] =
          weights[index]! - (LEARNING_RATE * m) / (Math.sqrt(v) + epsilon);
      }

      if (position % 64 === 63) {
        options.onProgress?.({
          epoch: epoch + 1,
          epochs,
          loss: epochLoss / (position + 1),
          fraction: (epoch * samples.length + position + 1) /
            (epochs * samples.length),
        });
        await new Promise<void>((resolve) => setTimeout(resolve, 0));
      }
    }

    totalLoss += epochLoss / samples.length;
    options.onProgress?.({
      epoch: epoch + 1,
      epochs,
      loss: epochLoss / samples.length,
      fraction: (epoch + 1) / epochs,
    });
    await new Promise<void>((resolve) => setTimeout(resolve, 0));
  }

  const model = new MiniLanguageModel(
    vocabulary,
    prepared.counts,
    keyWords,
    weights,
    {
      vocabulary: vocabulary.length,
      samples: samples.length,
      epochs,
      parameters: dimension * dimension,
      averageLoss: totalLoss / epochs,
      accuracy: 0,
      coverage: prepared.coverage,
      contextWindow: CONTEXT_WINDOW,
      vectorBits: VECTOR_BITS,
      vocabularyWords: vocabulary,
    },
  );

  model.stats.accuracy = model.evaluateQueries(
    samples.map((sample) => sample.query),
    samples.map((sample) => sample.target),
  );

  return model;
}

export function trainingParameters(): number {
  return VECTOR_BITS * VECTOR_BITS;
}

/*
 * Question answering.
 *
 * A 64-bit next-word model cannot generate a reliable free-form answer, so
 * this mode is retrieval: each training question is stored as a binary
 * vector plus a set of content words. Asking a question scores every stored
 * question with a bit similarity and a weighted word overlap, then returns
 * the answer attached to the best match.
 */

const QA_STOP_WORDS = new Set([
  "a",
  "an",
  "and",
  "are",
  "as",
  "at",
  "be",
  "but",
  "by",
  "can",
  "could",
  "did",
  "do",
  "does",
  "for",
  "from",
  "had",
  "has",
  "have",
  "he",
  "her",
  "his",
  "how",
  "i",
  "if",
  "in",
  "into",
  "is",
  "it",
  "its",
  "me",
  "my",
  "of",
  "on",
  "or",
  "our",
  "she",
  "should",
  "so",
  "that",
  "the",
  "their",
  "them",
  "then",
  "there",
  "these",
  "they",
  "this",
  "those",
  "to",
  "was",
  "we",
  "were",
  "what",
  "when",
  "where",
  "which",
  "who",
  "why",
  "will",
  "with",
  "would",
  "you",
  "your",
  // Solid 2 questions often include these, but they do not help distinguish
  // one concept from another.
  "about",
  "after",
  "available",
  "before",
  "between",
  "change",
  "changed",
  "changes",
  "difference",
  "different",
  "example",
  "examples",
  "explain",
  "happen",
  "happened",
  "happens",
  "mean",
  "meaning",
  "means",
  "need",
  "needs",
  "new",
  "old",
  "replace",
  "replaced",
  "replacement",
  "show",
  "solid",
  "still",
  "tell",
  "use",
  "used",
  "uses",
  "using",
  "version",
  "want",
  "wants",
  "way",
  "ways",
  "work",
  "working",
  "works",
  "2",
  "0",
  "rc",
]);

/*
 * A few 1.x -> 2.0 renames are useful as query expansions. Adding the
 * counterpart to both the query and the indexed question lets "onMount"
 * find "onSettled" without making the whole index fuzzy.
 */
const QA_ALIASES: ReadonlyMap<string, readonly string[]> = new Map([
  ["onmount", ["onsettled"]],
  ["onsettled", ["onmount"]],
  ["createcomputed", ["createeffect", "creatememo"]],
  ["creatememo", ["createcomputed"]],
  ["batch", ["flush"]],
  ["flush", ["batch"]],
  ["unwrap", ["snapshot"]],
  ["snapshot", ["unwrap"]],
  ["mergeprops", ["merge"]],
  ["merge", ["mergeprops"]],
  ["splitprops", ["omit"]],
  ["omit", ["splitprops"]],
  ["suspense", ["loading"]],
  ["loading", ["suspense"]],
  ["errorboundary", ["errored"]],
  ["errored", ["errorboundary"]],
  ["suspenselist", ["reveal"]],
  ["reveal", ["suspenselist"]],
  ["initialvalue", ["prev"]],
  ["prev", ["initialvalue"]],
  ["once", ["untrack"]],
  ["untrack", ["once"]],
  ["createresource", ["creatememo", "fetch"]],
  ["createmutable", ["createstore"]],
  ["produce", ["draft"]],
]);

export interface QaPair {
  readonly question: string;
  readonly answer: string;
}

export interface QaDocument {
  readonly id: string;
  readonly questions: readonly string[];
  readonly answer: string;
  readonly source?: string;
  readonly tags?: readonly string[];
  readonly keywords?: readonly string[];
}

export interface QaParseResult {
  readonly pairs: readonly QaPair[];
  readonly errors: readonly string[];
}

/**
 * Parse Q&A training text. Supported shapes:
 *
 *   question => answer            (one line)
 *   question                      (blank line between pairs)
 *   answer
 *   Q: question                   (Q/A labels also work)
 *   A: answer
 */
export function parseQaPairs(text: string): QaParseResult {
  const pairs: QaPair[] = [];
  const errors: string[] = [];
  const blocks = text
    .replace(/\r\n?/g, "\n")
    .split(/\n\s*\n/)
    .map((block) => block.trim())
    .filter((block) => block.length > 0);

  blocks.forEach((block, index) => {
    const lines = block
      .split("\n")
      .map((line) => line.trim())
      .filter((line) => line.length > 0);
    if (lines.length === 0) {
      return;
    }

    const arrowPattern = /^(.+?)\s*(?:=>|->)\s*(.+)$/;
    const arrows = lines.map((line) => arrowPattern.exec(line));
    if (arrows.every((match) => match !== null)) {
      for (const arrow of arrows) {
        const question = arrow![1]!.trim();
        const answer = arrow![2]!.trim();
        if (question.length === 0 || answer.length === 0) {
          errors.push(`Pair ${index + 1} needs both a question and an answer.`);
          return;
        }
        pairs.push({ question, answer });
      }
      return;
    }

    if (arrows[0] !== null) {
      const question = arrows[0]![1]!.trim();
      const answer = [arrows[0]![2]!.trim(), ...lines.slice(1)].join(" ").trim();
      if (question.length === 0 || answer.length === 0) {
        errors.push(`Pair ${index + 1} needs both a question and an answer.`);
        return;
      }
      pairs.push({ question, answer });
      return;
    }

    const answerIndex = lines.findIndex((line) =>
      /^(?:a|answer)\s*[:.)-]\s*/i.test(line),
    );
    if (answerIndex > 0) {
      const question = stripQaMarker("q", lines.slice(0, answerIndex).join(" "));
      const answer = lines
        .slice(answerIndex)
        .map((line, position) =>
          position === 0 ? stripQaMarker("a", line) : line,
        )
        .join(" ")
        .trim();
      if (question.length === 0 || answer.length === 0) {
        errors.push(`Pair ${index + 1} needs both a question and an answer.`);
        return;
      }
      pairs.push({ question, answer });
      return;
    }

    if (lines.length >= 2) {
      const question = stripQaMarker("q", lines[0]!);
      const answer = lines.slice(1).join(" ").trim();
      if (question.length === 0 || answer.length === 0) {
        errors.push(`Pair ${index + 1} needs both a question and an answer.`);
        return;
      }
      pairs.push({ question, answer });
      return;
    }

    errors.push(
      `Pair ${index + 1}: put the answer on the next line, use "question => answer", or label it with "A:".`,
    );
  });

  return { pairs, errors };
}

function stripQaMarker(kind: "q" | "a", line: string): string {
  const pattern =
    kind === "q"
      ? /^(?:q|question)\s*[:.)-]\s*/i
      : /^(?:a|answer)\s*[:.)-]\s*/i;
  return line.replace(pattern, "").trim();
}

function singularize(token: string): string {
  if (token.length <= 3) {
    return token;
  }
  if (token.endsWith("ies") && token.length > 4) {
    return `${token.slice(0, -3)}y`;
  }
  if (/(?:ses|xes|zes|ches|shes)$/.test(token)) {
    return token.slice(0, -2);
  }
  if (token.endsWith("s") && !token.endsWith("ss")) {
    return token.slice(0, -1);
  }
  return token;
}

function contentTokens(text: string): string[] {
  const rawTokens = tokenize(text).map((token) => token.text);
  const expanded: string[] = [];

  for (const raw of rawTokens) {
    const lower = raw.toLowerCase().replace(/^'+|'+$/g, "");
    if (lower.length === 0) {
      continue;
    }

    const candidates = new Set<string>([lower, singularize(lower)]);
    if (/[a-z][A-Z]/.test(raw)) {
      for (const part of raw.split(/(?=[A-Z])/)) {
        const normalized = part.toLowerCase();
        if (normalized.length > 0) {
          candidates.add(normalized);
        }
      }
    }
    if (raw.includes("_")) {
      for (const part of raw.split("_")) {
        const normalized = part.toLowerCase();
        if (normalized.length > 0) {
          candidates.add(normalized);
        }
      }
    }

    for (const candidate of candidates) {
      if (QA_STOP_WORDS.has(candidate)) {
        continue;
      }
      expanded.push(candidate);
      for (const alias of QA_ALIASES.get(candidate) ?? []) {
        expanded.push(alias);
      }
    }
  }

  if (expanded.length > 0) {
    return expanded;
  }
  return rawTokens.map((token) => token.toLowerCase()).filter(Boolean);
}

function questionVector(tokens: readonly string[]): Uint32Array {
  return majorityVector(tokens.map((token) => getWordVector(token)));
}

interface QaEntry {
  readonly question: string;
  readonly answer: string;
  readonly tokens: ReadonlySet<string>;
  readonly vector: Uint32Array;
  readonly documentId: string;
  readonly source?: string;
  readonly tags: readonly string[];
}

export interface QaStats {
  readonly pairs: number;
  readonly concepts: number;
  readonly variants: number;
  readonly questionWords: number;
  readonly vectorBits: number;
  readonly coverage: number;
}

export interface QaMatch {
  readonly question: string;
  readonly confidence: number;
}

export interface QaAnswer extends QaMatch {
  readonly answer: string;
  readonly matched: boolean;
  readonly candidates: readonly QaMatch[];
  readonly documentId: string;
  readonly source?: string;
  readonly tags: readonly string[];
}

function weightedOverlap(
  query: ReadonlySet<string>,
  entry: ReadonlySet<string>,
  idf: ReadonlyMap<string, number>,
): number {
  let shared = 0;
  let union = 0;
  for (const token of query) {
    const weight = idf.get(token) ?? 1;
    union += weight;
    if (entry.has(token)) {
      shared += weight;
    }
  }
  for (const token of entry) {
    if (!query.has(token)) {
      union += idf.get(token) ?? 1;
    }
  }
  return union === 0 ? 0 : shared / union;
}

export class MiniQuestionAnswerer {
  readonly stats: QaStats;
  readonly questions: readonly string[];
  readonly suggestions: readonly string[];

  private readonly entries: readonly QaEntry[];
  private readonly idf: ReadonlyMap<string, number>;

  constructor(
    entries: readonly QaEntry[],
    idf: ReadonlyMap<string, number>,
    stats: QaStats,
    suggestions: readonly string[],
  ) {
    this.entries = entries;
    this.idf = idf;
    this.stats = stats;
    this.questions = entries.map((entry) => entry.question);
    this.suggestions = suggestions;
  }

  ask(text: string): QaAnswer | null {
    const tokens = contentTokens(text);
    if (tokens.length === 0 || this.entries.length === 0) {
      return null;
    }

    const unique = new Set(tokens);
    const queryVector = questionVector([...unique]);
    const scored = this.entries
      .map((entry, index) => {
        const overlap = weightedOverlap(unique, entry.tokens, this.idf);
        const matches = (scoreKey(queryVector, entry.vector) + VECTOR_BITS) / 2;
        const similarity = matches / VECTOR_BITS;
        return {
          index,
          question: entry.question,
          confidence: 0.8 * overlap + 0.2 * similarity,
        };
      })
      .sort((a, b) => b.confidence - a.confidence);

    const best = scored[0]!;
    const entry = this.entries[best.index]!;
    let shared = 0;
    for (const token of unique) {
      if (entry.tokens.has(token)) {
        shared++;
      }
    }
    const matched =
      best.confidence >= 0.34 && (shared > 0 || best.confidence >= 0.72);

    return {
      answer: entry.answer,
      question: entry.question,
      confidence: best.confidence,
      matched,
      candidates: scored.slice(0, 3).map((candidate) => ({
        question: candidate.question,
        confidence: candidate.confidence,
      })),
      documentId: entry.documentId,
      source: entry.source,
      tags: entry.tags,
    };
  }
}

export function trainQuestionAnswererFromDocuments(
  documents: readonly QaDocument[],
): MiniQuestionAnswerer {
  if (documents.length === 0) {
    throw new Error("Add at least one question and answer pair.");
  }

  const prepared = documents.map((document) => {
    if (document.questions.length === 0) {
      throw new Error(`"${document.id}" needs at least one question.`);
    }
    const variantTokens = document.questions.map((question) =>
      contentTokens(question),
    );
    const keywordTokens = (document.keywords ?? []).flatMap((keyword) =>
      contentTokens(keyword),
    );
    const union = new Set<string>();
    for (const tokens of variantTokens) {
      for (const token of tokens) {
        union.add(token);
      }
    }
    for (const token of keywordTokens) {
      union.add(token);
    }
    return { document, variantTokens, union };
  });

  const documentFrequency = new Map<string, number>();
  for (const { union } of prepared) {
    for (const token of union) {
      documentFrequency.set(token, (documentFrequency.get(token) ?? 0) + 1);
    }
  }

  const total = prepared.length;
  const idf = new Map<string, number>();
  for (const [token, count] of documentFrequency) {
    idf.set(token, Math.log((total + 1) / (count + 1)) + 1);
  }

  let questionWords = 0;
  let bundledWords = 0;
  const entries: QaEntry[] = [];

  for (const { document, variantTokens } of prepared) {
    document.questions.forEach((question, index) => {
      const tokens = variantTokens[index] ?? [];
      if (tokens.length === 0) {
        return;
      }
      questionWords += tokens.length;
      for (const token of tokens) {
        if (hasBundledVector(token)) {
          bundledWords++;
        }
      }
      entries.push({
        question,
        answer: document.answer,
        tokens: new Set(tokens),
        vector: questionVector([...new Set(tokens)]),
        documentId: document.id,
        source: document.source,
        tags: document.tags ?? [],
      });
    });
  }

  if (entries.length === 0) {
    throw new Error("Add at least one question and answer pair.");
  }

  const suggestions = documents
    .map((document) => document.questions[0])
    .filter(
      (question): question is string =>
        question !== undefined && question.length > 0,
    );

  return new MiniQuestionAnswerer(
    entries,
    idf,
    {
      pairs: entries.length,
      concepts: documents.length,
      variants: entries.length,
      questionWords,
      vectorBits: VECTOR_BITS,
      coverage: questionWords === 0 ? 0 : bundledWords / questionWords,
    },
    suggestions,
  );
}

export function trainQuestionAnswerer(text: string): MiniQuestionAnswerer {
  const parsed = parseQaPairs(text);
  if (parsed.errors.length > 0) {
    throw new Error(parsed.errors[0]);
  }
  if (parsed.pairs.length === 0) {
    throw new Error('Add at least one "question => answer" pair.');
  }

  return trainQuestionAnswererFromDocuments(
    parsed.pairs.map((pair, index) => ({
      id: `pair-${index + 1}`,
      questions: [pair.question],
      answer: pair.answer,
    })),
  );
}
