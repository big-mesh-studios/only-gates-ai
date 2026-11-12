import { createTokenStreamCache } from "./tokenStreamCache";
import { SOLID2_MODEL_BASE, SOLID2_TRAIN_CONFIG, solid2ParameterCount } from "./solid2Config";
import {
  BPE_SPECIAL,
  loadBpeTokenizer,
  type BpeTokenizer,
} from "./tinyGptBpe";
import {
  createSolid2WasmFromData,
  type Solid2Checkpoint,
  type Solid2WasmModel,
} from "./tinyGptSolid2";
import { packTinyGptWeights } from "./tinyGptWasm";
import { loadBundledSolid2Model } from "./trainSolid2Gpt";
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

/*
 * Browser-side training on a user-supplied list of Markdown documents.
 *
 * The model uses the deployed Solid 2 architecture and BPE tokenizer, so the
 * integer-only WASM engine can load the result unchanged. Only the corpus is
 * different: every URL is fetched, split into paragraphs, and tokenized with
 * the same 4096-token vocabulary.
 */

const tokenCache = createTokenStreamCache("gate-forest-custom-training-data");
const TOKEN_SCHEME_VERSION = 1;
const VALIDATION_FRACTION = 0.1;
const MAX_PROMPTS = 5;

export interface CustomDoc {
  readonly url: string;
  readonly text: string;
}

export interface CustomTrainingDataInfo {
  readonly documentCount: number;
  readonly documentChars: number;
  readonly paragraphs: number;
  readonly trainingTokens: number;
  readonly validationTokens: number;
  readonly tokenizerVocab: number;
  readonly cached: boolean;
  readonly warmStart: boolean;
}

export interface CustomTrainingData {
  readonly tokenizer: BpeTokenizer;
  readonly trainingData: TokenTrainingData;
  readonly prompts: readonly string[];
  readonly info: CustomTrainingDataInfo;
}

export interface CustomDataOptions {
  readonly urls: readonly string[];
  readonly tokenizerUrl?: string;
  /** Base URL for the deployed tokenizer. Defaults to `SOLID2_MODEL_BASE`. */
  readonly modelBaseUrl?: string;
  readonly onStatus?: (status: string) => void;
  readonly shouldCancel?: () => boolean;
}

export interface CustomTrainingOptions extends CustomDataOptions {
  readonly steps?: number;
  readonly batch?: number;
  readonly lr?: number;
  readonly seed?: number;
  readonly dropout?: number;
  readonly maxMinutes?: number;
  readonly backend?: "auto" | BrowserTrainingBackend;
  readonly initialState?: BrowserTrainingState | null;
  /**
   * Initializes from the bundled Solid 2 checkpoint instead of random weights.
   * The architecture and tokenizer are identical; only the corpus changes.
   * Ignored when resuming a saved run (`initialState` wins).
   */
  readonly warmStart?: boolean;
  readonly onProgress?: (progress: BrowserTrainingProgress) => void;
  readonly onCheckpoint?: (state: BrowserTrainingState) => void;
  readonly yieldEvery?: number;
  readonly saveEvery?: number;
}

export interface CustomTrainingResult {
  readonly result: BrowserTrainingResult;
  readonly checkpoint: Solid2Checkpoint;
  readonly tokenizer: BpeTokenizer;
  /** Packed in `tiny_gpt_create` order, ready for the WASM quantizer. */
  readonly weights: Float32Array;
  readonly data: CustomTrainingDataInfo;
  readonly prompts: readonly string[];
}

/* ------------------------------------------------------------------ */
/* URL handling                                                       */
/* ------------------------------------------------------------------ */

/**
 * Normalizes GitHub "blob" links to their raw form. Raw hosts already serve
 * plain text with CORS enabled, which the browser fetch needs.
 */
function normalizeDocUrl(value: string): string {
  let url: URL;
  try {
    url = new URL(value);
  } catch {
    throw new Error(`Not a valid URL: ${value}`);
  }
  if (url.protocol !== "http:" && url.protocol !== "https:") {
    throw new Error(`Only http(s) URLs are supported: ${value}`);
  }
  if (url.hostname === "github.com" || url.hostname === "www.github.com") {
    const parts = url.pathname.split("/").filter(Boolean);
    if (parts.length >= 5 && parts[2] === "blob") {
      const [owner, repo, , ...rest] = parts;
      return `https://raw.githubusercontent.com/${owner}/${repo}/${rest.join("/")}`;
    }
  }
  return url.toString();
}

/** Parses a newline-separated list of document URLs. */
export function parseDocUrls(input: string | readonly string[]): string[] {
  const seen = new Set<string>();
  const urls: string[] = [];
  const lines = typeof input === "string" ? input.split(/\r?\n/) : input;
  for (const line of lines) {
    const value = line.trim();
    if (value.length === 0 || value.startsWith("#")) continue;
    const normalized = normalizeDocUrl(value);
    if (seen.has(normalized)) continue;
    seen.add(normalized);
    urls.push(normalized);
  }
  return urls;
}

async function fetchCustomDocs(
  urls: readonly string[],
  options: CustomDataOptions,
): Promise<CustomDoc[]> {
  const docs: CustomDoc[] = [];
  for (let index = 0; index < urls.length; index++) {
    if (options.shouldCancel?.()) {
      throw new Error("Training was cancelled.");
    }
    const url = urls[index]!;
    options.onStatus?.(
      `Downloading document ${index + 1} of ${urls.length} (${url})...`,
    );
    let response: Response;
    try {
      response = await fetch(url);
    } catch (reason) {
      throw new Error(
        `Could not fetch ${url}. Check that the URL is public and allows ` +
          `cross-origin reads (CORS). ${(reason as Error).message}`,
      );
    }
    if (!response.ok) {
      throw new Error(`fetch failed: ${response.status} ${url}`);
    }
    docs.push({ url, text: await response.text() });
    await yieldToBrowser();
  }
  return docs;
}

/* ------------------------------------------------------------------ */
/* Tokenization                                                       */
/* ------------------------------------------------------------------ */

function appendIds(target: number[], ids: ArrayLike<number>): void {
  for (let index = 0; index < ids.length; index++) {
    target.push(ids[index]!);
  }
}

function countParagraphs(docs: readonly CustomDoc[]): number {
  let count = 0;
  for (const doc of docs) {
    for (const paragraph of doc.text.split(/\n\s*\n/)) {
      if (paragraph.trim().length > 0) count++;
    }
  }
  return count;
}

function stripMarkdown(value: string): string {
  return value
    .replace(/!\[([^\]]*)\]\([^)]*\)/g, "$1")
    .replace(/\[([^\]]+)\]\([^)]*\)/g, "$1")
    .replace(/[*_`~]/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

/** Derives short prompts from Markdown headings, falling back to first lines. */
export function extractCustomPrompts(
  docs: readonly CustomDoc[],
  limit = MAX_PROMPTS,
): string[] {
  const prompts: string[] = [];
  const seen = new Set<string>();
  const add = (value: string): void => {
    const prompt = stripMarkdown(value).slice(0, 120).trim();
    if (prompt.length < 4 || seen.has(prompt)) return;
    seen.add(prompt);
    prompts.push(prompt);
  };

  for (const doc of docs) {
    const headings = doc.text.matchAll(/^#{1,6}[ \t]+(.+?)[ \t]*#*[ \t]*$/gm);
    for (const heading of headings) {
      add(heading[1] ?? "");
      if (prompts.length >= limit) return prompts;
    }
  }
  if (prompts.length === 0) {
    for (const doc of docs) {
      const firstLine = doc.text
        .split(/\r?\n/)
        .map((line) => line.trim())
        .find((line) => line.length > 0);
      if (firstLine) add(firstLine);
      if (prompts.length >= limit) break;
    }
  }
  return prompts;
}

interface CustomTokenStream {
  readonly trainingTokens: Uint16Array;
  readonly validationTokens: Uint16Array;
}

async function buildCustomTokenStream(
  tokenizer: BpeTokenizer,
  docs: readonly CustomDoc[],
  context: number,
  onProgress: (fraction: number) => void,
  shouldCancel?: () => boolean,
): Promise<CustomTokenStream> {
  const endId = tokenizer.idFor(BPE_SPECIAL.end);
  if (endId === undefined) {
    throw new Error("The tokenizer is missing its end-of-document token.");
  }

  const chunks: string[] = [];
  for (const doc of docs) {
    for (const paragraph of doc.text.split(/\n\s*\n/)) {
      const text = paragraph.trim();
      if (text.length > 0) chunks.push(text);
    }
  }
  if (chunks.length === 0) {
    throw new Error("The documents have no text to train on.");
  }

  const tokens: number[] = [];
  let lastPercent = -1;
  for (let index = 0; index < chunks.length; index++) {
    appendIds(tokens, tokenizer.encodeOrdinary(chunks[index]!));
    tokens.push(endId);
    if (index % 8 === 0) {
      const fraction = index / chunks.length;
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

  const minTraining = context + 2;
  if (tokens.length < minTraining) {
    throw new Error(
      `The documents are too short: ${tokens.length.toLocaleString()} tokens, ` +
        `but this model needs at least ${minTraining.toLocaleString()}. ` +
        `Add more Markdown files or content.`,
    );
  }

  // Hold out the newest ~10% of the stream, but keep a full training window.
  // A short corpus may end up with no validation tokens; trainTokenGpt treats
  // that as "no validation result" instead of failing.
  let split = Math.floor(tokens.length * (1 - VALIDATION_FRACTION));
  if (tokens.length - split < minTraining) {
    split = Math.max(minTraining, tokens.length - minTraining);
  }
  if (split > tokens.length) split = tokens.length;

  return {
    trainingTokens: Uint16Array.from(tokens.slice(0, split)),
    validationTokens: Uint16Array.from(tokens.slice(split)),
  };
}

/* ------------------------------------------------------------------ */
/* Data loading                                                       */
/* ------------------------------------------------------------------ */

function modelBase(options: CustomDataOptions): string {
  return (options.modelBaseUrl ?? SOLID2_MODEL_BASE).replace(/\/?$/, "/");
}

export async function loadCustomTrainingData(
  options: CustomDataOptions,
): Promise<CustomTrainingData> {
  const urls = parseDocUrls(options.urls);
  if (urls.length === 0) {
    throw new Error("Add at least one document URL.");
  }

  const tokenizerUrl =
    options.tokenizerUrl ?? `${modelBase(options)}tokenizer.json`;
  options.onStatus?.("Loading the deployed Solid 2 BPE tokenizer...");
  const tokenizer = await loadBpeTokenizer(tokenizerUrl);

  const docs = await fetchCustomDocs(urls, options);
  const documentChars = docs.reduce((total, doc) => total + doc.text.length, 0);
  if (documentChars === 0) {
    throw new Error("The downloaded documents are empty.");
  }

  const dataHash = hashString(
    docs.map((doc) => `${doc.url}\u0000${doc.text}`).join("\u0001"),
  );
  const tokenizerHash = hashString(JSON.stringify(tokenizer.toJSON()));
  const cacheKey = `v${TOKEN_SCHEME_VERSION}:${tokenizerHash}:${dataHash}`;

  let stream: CustomTokenStream;
  let cached = false;
  const hit = await tokenCache.read(cacheKey);
  if (hit) {
    stream = {
      trainingTokens: hit.trainingTokens,
      validationTokens: hit.validationTokens,
    };
    cached = true;
    options.onStatus?.(
      `Loaded the cached token stream (${hit.trainingTokens.length.toLocaleString()} training tokens).`,
    );
  } else {
    options.onStatus?.(
      `Tokenizing ${urls.length} document${urls.length === 1 ? "" : "s"} with the BPE tokenizer...`,
    );
    stream = await buildCustomTokenStream(
      tokenizer,
      docs,
      SOLID2_TRAIN_CONFIG.context,
      (fraction) =>
        options.onStatus?.(`Tokenizing documents (${Math.round(fraction * 100)}%)...`),
      options.shouldCancel,
    );
    options.onStatus?.("Caching the token stream in IndexedDB...");
    await tokenCache.write({
      key: cacheKey,
      trainingTokens: stream.trainingTokens,
      validationTokens: stream.validationTokens,
      savedAt: Date.now(),
    });
  }

  const prompts = extractCustomPrompts(docs);
  const corpusHash = hashString(
    `${TOKEN_SCHEME_VERSION}:${tokenizerHash}:${dataHash}`,
  );
  const info: CustomTrainingDataInfo = {
    documentCount: docs.length,
    documentChars,
    paragraphs: countParagraphs(docs),
    trainingTokens: stream.trainingTokens.length,
    validationTokens: stream.validationTokens.length,
    tokenizerVocab: tokenizer.vocab.length,
    cached,
    warmStart: false,
  };

  options.onStatus?.(
    `Ready: ${stream.trainingTokens.length.toLocaleString()} training tokens from ` +
      `${docs.length} document${docs.length === 1 ? "" : "s"}` +
      `${cached ? " (cached)" : ""}.`,
  );

  return {
    tokenizer,
    prompts,
    info,
    trainingData: {
      vocab: tokenizer.vocab,
      trainingTokens: stream.trainingTokens,
      validationTokens: stream.validationTokens,
      corpusHash,
      corpusSize: stream.trainingTokens.length,
      corpusLabel: "corpusTokens",
    },
  };
}

/* ------------------------------------------------------------------ */
/* Training and export                                                */
/* ------------------------------------------------------------------ */

export async function trainCustomGpt(
  options: CustomTrainingOptions = { urls: [] },
): Promise<CustomTrainingResult> {
  const data = await loadCustomTrainingData(options);
  const urls = parseDocUrls(options.urls);

  let initialWeights: Float32Array | undefined;
  let warmStarted = false;
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
    initialWeights = bundled.weights;
    warmStarted = true;
    options.onStatus?.(
      `Warm-starting from the bundled model (${expected.toLocaleString()} params).`,
    );
  } else {
    options.onStatus?.(
      "Selecting the TensorFlow.js backend and starting training...",
    );
  }

  const attachSources = (state: BrowserTrainingState): BrowserTrainingState => ({
    ...state,
    sourceUrls: urls,
  });

  const result = await trainTokenGpt({
    data: data.trainingData,
    config: SOLID2_TRAIN_CONFIG,
    defaultConfig: SOLID2_TRAIN_CONFIG,
    steps: options.steps ?? 600,
    batch: options.batch ?? 4,
    lr: options.lr ?? 4e-4,
    seed: options.seed ?? 1337,
    dropout: options.dropout ?? 0.1,
    maxMinutes: options.maxMinutes ?? 30,
    backend: options.backend,
    initialState: options.initialState,
    initialWeights,
    onProgress: options.onProgress,
    onCheckpoint: options.onCheckpoint
      ? (state) => options.onCheckpoint!(attachSources(state))
      : undefined,
    shouldCancel: options.shouldCancel,
    yieldEvery: options.yieldEvery ?? 2,
    saveEvery: options.saveEvery ?? 100,
    kind: "custom",
    modelMeta: {
      name: "tiny-gpt-custom-browser",
      source:
        "Custom Markdown documents supplied by the user: " +
        `${urls.slice(0, 8).join(", ")}` +
        (urls.length > 8 ? ` and ${urls.length - 8} more` : "") +
        ".",
      license:
        "Upstream documents retain their own licenses; model weights trained " +
        "in the browser by this repository.",
    },
  });

  const state = attachSources(result.state);
  const checkpoint: Solid2Checkpoint = {
    format: "only-gates-tiny-gpt-custom",
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
    result: { ...result, state },
    checkpoint,
    tokenizer: data.tokenizer,
    weights: packTinyGptWeights(result.checkpoint),
    data: { ...data.info, warmStart: warmStarted },
    prompts: data.prompts,
  };
}

/** Builds the integer-only WASM model from a completed custom training run. */
export function buildCustomWasmFromTraining(
  result: CustomTrainingResult,
): Promise<Solid2WasmModel> {
  return createSolid2WasmFromData(
    result.checkpoint,
    result.tokenizer,
    result.weights,
  );
}
