import { createTokenStreamCache } from "./tokenStreamCache";
import { loadBpeTokenizer, type BpeTokenizer } from "./tinyGptBpe";
import {
  createSolid2WasmFromData,
  type Solid2Checkpoint,
  type Solid2WasmModel,
} from "./tinyGptSolid2";
import { packTinyGptWeights } from "./tinyGptWasm";
import {
  hashString,
  trainTokenGpt,
  yieldToBrowser,
  type BrowserTrainingBackend,
  type BrowserTrainingProgress,
  type BrowserTrainingResult,
  type BrowserTrainingState,
  type TokenTrainingData,
} from "./trainTinyGpt";
import {
  SOLID2_DATA_BASE,
  SOLID2_MODEL_BASE,
  SOLID2_TRAIN_CONFIG,
  solid2ParameterCount,
} from "./solid2Config";

/*
 * Browser-side training for the Solid 2 docs + Q&A model.
 *
 * The raw corpus is fetched from the repository instead of being bundled. It is
 * tokenized with the same BPE tokenizer the deployed demo uses, then the token
 * stream is cached in IndexedDB. Training itself is the shared float32
 * `trainTokenGpt` core; the integer path starts when the result is quantized
 * into the WASM model.
 */

const USER_ID = 1;
const ASSISTANT_ID = 2;
const END_ID = 3;
const TOKEN_SCHEME_VERSION = 1;
// The Colab job oversamples chat-formatted QA 8×. A short browser run needs the
// inference format to dominate, but 3× keeps the docs half of the stream.
const QA_REPEATS = 3;

const tokenCache = createTokenStreamCache(
  "gate-forest-solid2-training-data",
);

export interface Solid2QaRow {
  readonly id?: string;
  readonly question: string;
  readonly answer: string;
}

export interface Solid2TrainingDataInfo {
  readonly docsChars: number;
  readonly qaTrainRows: number;
  readonly qaValRows: number;
  readonly trainingTokens: number;
  readonly validationTokens: number;
  readonly tokenizerVocab: number;
  readonly cached: boolean;
}

export interface Solid2TrainingData {
  readonly tokenizer: BpeTokenizer;
  readonly trainingData: TokenTrainingData;
  readonly prompts: readonly string[];
  readonly info: Solid2TrainingDataInfo;
}

export interface Solid2DataOptions {
  readonly tokenizerUrl?: string;
  readonly dataBaseUrl?: string;
  /** Base URL for the deployed checkpoint/tokenizer. Defaults to `SOLID2_MODEL_BASE`. */
  readonly modelBaseUrl?: string;
  readonly onStatus?: (status: string) => void;
  readonly shouldCancel?: () => boolean;
}

export interface Solid2TrainingOptions extends Solid2DataOptions {
  readonly steps?: number;
  readonly batch?: number;
  readonly lr?: number;
  readonly seed?: number;
  readonly dropout?: number;
  readonly maxMinutes?: number;
  readonly backend?: "auto" | BrowserTrainingBackend;
  readonly initialState?: BrowserTrainingState | null;
  /**
   * Initializes the model from the bundled Solid 2 checkpoint instead of random
   * weights. Ignored when resuming a saved run (`initialState` wins).
   */
  readonly warmStart?: boolean;
  readonly onProgress?: (progress: BrowserTrainingProgress) => void;
  readonly onCheckpoint?: (state: BrowserTrainingState) => void;
  readonly yieldEvery?: number;
  readonly saveEvery?: number;
}

export interface Solid2TrainingResult {
  readonly result: BrowserTrainingResult;
  readonly checkpoint: Solid2Checkpoint;
  readonly tokenizer: BpeTokenizer;
  /** Packed in `tiny_gpt_create` order, ready for the WASM quantizer. */
  readonly weights: Float32Array;
  readonly data: Solid2TrainingDataInfo;
  readonly prompts: readonly string[];
}

/* ------------------------------------------------------------------ */
/* Fetching and parsing                                               */
/* ------------------------------------------------------------------ */

async function fetchText(url: string): Promise<string> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`fetch failed: ${response.status} ${url}`);
  }
  return response.text();
}

function parseQaRows(text: string, label: string): Solid2QaRow[] {
  const rows: Solid2QaRow[] = [];
  const lines = text.split(/\r?\n/);
  for (let index = 0; index < lines.length; index++) {
    const line = lines[index]!.trim();
    if (line.length === 0) continue;
    let value: unknown;
    try {
      value = JSON.parse(line);
    } catch {
      throw new Error(`${label} line ${index + 1} is not valid JSON.`);
    }
    if (typeof value !== "object" || value === null) {
      throw new Error(`${label} line ${index + 1} is not an object.`);
    }
    const record = value as Record<string, unknown>;
    if (typeof record.question !== "string" || typeof record.answer !== "string") {
      throw new Error(`${label} line ${index + 1} is missing question/answer.`);
    }
    rows.push({
      id: typeof record.id === "string" ? record.id : undefined,
      question: record.question,
      answer: record.answer,
    });
  }
  if (rows.length === 0) {
    throw new Error(`${label} is empty.`);
  }
  return rows;
}

export interface BundledSolid2Model {
  readonly checkpoint: Solid2Checkpoint;
  readonly weights: Float32Array;
}

/** Reads the deployed checkpoint and packed weights for a warm start. */
export async function loadBundledSolid2Model(
  base: string,
): Promise<BundledSolid2Model> {
  const [checkpointResponse, weightsResponse] = await Promise.all([
    fetch(`${base}checkpoint.json`),
    fetch(`${base}weights.bin`),
  ]);
  if (!checkpointResponse.ok) {
    throw new Error(
      `fetch failed: ${checkpointResponse.status} ${base}checkpoint.json`,
    );
  }
  if (!weightsResponse.ok) {
    throw new Error(
      `fetch failed: ${weightsResponse.status} ${base}weights.bin`,
    );
  }
  return {
    checkpoint: (await checkpointResponse.json()) as Solid2Checkpoint,
    weights: new Float32Array(await weightsResponse.arrayBuffer()),
  };
}

/* ------------------------------------------------------------------ */
/* Tokenization                                                       */
/* ------------------------------------------------------------------ */

function appendIds(target: number[], ids: ArrayLike<number>): void {
  for (let index = 0; index < ids.length; index++) {
    target.push(ids[index]!);
  }
}

function encodeQaRows(
  tokenizer: BpeTokenizer,
  rows: readonly Solid2QaRow[],
): number[] {
  const ids: number[] = [];
  for (const row of rows) {
    ids.push(USER_ID);
    appendIds(ids, tokenizer.encodeOrdinary(row.question));
    ids.push(ASSISTANT_ID);
    appendIds(ids, tokenizer.encodeOrdinary(row.answer));
    ids.push(END_ID);
  }
  return ids;
}

async function tokenizeDocs(
  tokenizer: BpeTokenizer,
  docs: string,
  onProgress: (fraction: number) => void,
  shouldCancel?: () => boolean,
): Promise<number[]> {
  const chunks = docs.split(/\n\s*\n/);
  const ids: number[] = [];
  let lastPercent = -1;
  for (let index = 0; index < chunks.length; index++) {
    const chunk = chunks[index]!.trim();
    if (chunk.length > 0) {
      appendIds(ids, tokenizer.encodeOrdinary(chunk));
      ids.push(END_ID);
    }
    if (index % 8 === 0) {
      const fraction = chunks.length > 0 ? index / chunks.length : 1;
      const percent = Math.floor(fraction * 100);
      if (percent !== lastPercent) {
        lastPercent = percent;
        onProgress(fraction);
      }
      await yieldToBrowser();
      if (shouldCancel?.()) {
        throw new Error("Training was cancelled.");
      }
    }
  }
  onProgress(1);
  return ids;
}

/* ------------------------------------------------------------------ */
/* Data loading                                                       */
/* ------------------------------------------------------------------ */

function modelBase(options: Solid2DataOptions): string {
  return (options.modelBaseUrl ?? SOLID2_MODEL_BASE).replace(/\/?$/, "/");
}

export async function loadSolid2TrainingData(
  options: Solid2DataOptions = {},
): Promise<Solid2TrainingData> {
  const tokenizerUrl =
    options.tokenizerUrl ?? `${modelBase(options)}tokenizer.json`;
  const dataBase = (options.dataBaseUrl ?? SOLID2_DATA_BASE).replace(
    /\/?$/,
    "/",
  );

  options.onStatus?.("Loading the deployed Solid 2 BPE tokenizer...");
  const tokenizer = await loadBpeTokenizer(tokenizerUrl);

  options.onStatus?.("Downloading Solid 2 docs and Q&A from the repository...");
  const [docsText, qaTrainText, qaValText] = await Promise.all([
    fetchText(`${dataBase}docs.txt`),
    fetchText(`${dataBase}qa-train.jsonl`),
    fetchText(`${dataBase}qa-val.jsonl`),
  ]);
  const qaTrain = parseQaRows(qaTrainText, "qa-train.jsonl");
  const qaVal = parseQaRows(qaValText, "qa-val.jsonl");

  const dataHash = hashString(
    `${docsText.length}\u0000${docsText}\u0000${qaTrainText}\u0000${qaValText}`,
  );
  const tokenizerHash = hashString(JSON.stringify(tokenizer.toJSON()));
  const cacheKey = `v${TOKEN_SCHEME_VERSION}:${tokenizerHash}:${dataHash}`;

  let trainingTokens: Uint16Array;
  let validationTokens: Uint16Array;
  let cached = false;

  const hit = await tokenCache.read(cacheKey);
  if (hit) {
    trainingTokens = hit.trainingTokens;
    validationTokens = hit.validationTokens;
    cached = true;
    options.onStatus?.(
      `Loaded the cached token stream (${trainingTokens.length.toLocaleString()} training tokens).`,
    );
  } else {
    options.onStatus?.("Tokenizing Solid 2 docs with the deployed BPE tokenizer...");
    const docsIds = await tokenizeDocs(
      tokenizer,
      docsText,
      (fraction) =>
        options.onStatus?.(
          `Tokenizing Solid 2 docs (${Math.round(fraction * 100)}%)...`,
        ),
      options.shouldCancel,
    );

    options.onStatus?.("Tokenizing the Solid 2 Q&A examples...");
    const qaIds = encodeQaRows(tokenizer, qaTrain);
    const trainingIds: number[] = [];
    appendIds(trainingIds, docsIds);
    for (let repeat = 0; repeat < QA_REPEATS; repeat++) {
      appendIds(trainingIds, qaIds);
    }
    trainingTokens = Uint16Array.from(trainingIds);
    validationTokens = Uint16Array.from(encodeQaRows(tokenizer, qaVal));

    options.onStatus?.("Caching the token stream in IndexedDB...");
    await tokenCache.write({
      key: cacheKey,
      trainingTokens,
      validationTokens,
      savedAt: Date.now(),
    });
  }

  const corpusHash = hashString(
    `${TOKEN_SCHEME_VERSION}:${tokenizerHash}:${dataHash}`,
  );
  const prompts = [...new Set(qaTrain.map((row) => row.question))].slice(0, 5);

  const info: Solid2TrainingDataInfo = {
    docsChars: docsText.length,
    qaTrainRows: qaTrain.length,
    qaValRows: qaVal.length,
    trainingTokens: trainingTokens.length,
    validationTokens: validationTokens.length,
    tokenizerVocab: tokenizer.vocab.length,
    cached,
  };

  options.onStatus?.(
    `Ready: ${trainingTokens.length.toLocaleString()} training tokens, ` +
      `${validationTokens.length.toLocaleString()} validation tokens` +
      `${cached ? " (cached)" : ""}.`,
  );

  return {
    tokenizer,
    prompts,
    info,
    trainingData: {
      vocab: tokenizer.vocab,
      trainingTokens,
      validationTokens,
      corpusHash,
      corpusSize: trainingTokens.length,
      corpusLabel: "corpusTokens",
    },
  };
}

/* ------------------------------------------------------------------ */
/* Training and export                                                */
/* ------------------------------------------------------------------ */

export async function trainSolid2Gpt(
  options: Solid2TrainingOptions = {},
): Promise<Solid2TrainingResult> {
  const data = await loadSolid2TrainingData(options);

  let config = SOLID2_TRAIN_CONFIG;
  let initialWeights: Float32Array | undefined;
  if (options.warmStart && !options.initialState) {
    options.onStatus?.(
      "Loading the bundled Solid 2 checkpoint for a warm start...",
    );
    const bundled = await loadBundledSolid2Model(modelBase(options));
    const expected = solid2ParameterCount(bundled.checkpoint.config);
    if (bundled.weights.length !== expected) {
      throw new Error(
        `Bundled weights.bin has ${bundled.weights.length.toLocaleString()} values but ` +
          `the checkpoint config needs ${expected.toLocaleString()}.`,
      );
    }
    if (bundled.checkpoint.config.vocabSize !== data.trainingData.vocab.length) {
      throw new Error(
        "The bundled Solid 2 vocabulary does not match the BPE tokenizer.",
      );
    }
    config = bundled.checkpoint.config;
    initialWeights = bundled.weights;
    options.onStatus?.(
      `Warm-starting from the bundled model (${expected.toLocaleString()} params).`,
    );
  } else {
    options.onStatus?.(
      "Selecting the TensorFlow.js backend and starting training...",
    );
  }

  const result = await trainTokenGpt({
    data: data.trainingData,
    config,
    defaultConfig: SOLID2_TRAIN_CONFIG,
    steps: options.steps ?? 400,
    batch: options.batch ?? 4,
    lr: options.lr ?? 4e-4,
    seed: options.seed ?? 1337,
    dropout: options.dropout ?? 0.1,
    maxMinutes: options.maxMinutes ?? 30,
    backend: options.backend,
    initialState: options.initialState,
    initialWeights,
    onProgress: options.onProgress,
    onCheckpoint: options.onCheckpoint,
    shouldCancel: options.shouldCancel,
    yieldEvery: options.yieldEvery ?? 2,
    saveEvery: options.saveEvery ?? 100,
    kind: "solid2",
    modelMeta: {
      name: "tiny-gpt-solid2-browser",
      source:
        "Solid 2 docs and curated Q&A from the only-gates-ai repository, " +
        "tokenized with the deployed BPE tokenizer and trained in the browser.",
      license:
        "Docs and Q&A from the only-gates-ai repository; model weights " +
        "trained in the browser by this repository.",
    },
  });

  const checkpoint: Solid2Checkpoint = {
    format: "only-gates-tiny-gpt-solid2",
    version: 1,
    config: result.checkpoint.config,
    parameterCount: solid2ParameterCount(result.checkpoint.config),
    tokenizer: "tokenizer.json",
    meta: {
      name: result.checkpoint.meta.name,
      source: result.checkpoint.meta.source,
      license: result.checkpoint.meta.license,
      training: result.checkpoint.meta.training,
    },
    samples: [...result.checkpoint.samples],
    prompts: [...data.prompts],
  };

  return {
    result,
    checkpoint,
    tokenizer: data.tokenizer,
    weights: packTinyGptWeights(result.checkpoint),
    data: data.info,
    prompts: data.prompts,
  };
}

/** Builds the integer-only WASM model from a completed browser training run. */
export function buildSolid2WasmFromTraining(
  result: Solid2TrainingResult,
): Promise<Solid2WasmModel> {
  return createSolid2WasmFromData(
    result.checkpoint,
    result.tokenizer,
    result.weights,
  );
}
