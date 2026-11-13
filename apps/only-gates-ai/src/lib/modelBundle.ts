import {
  flattenSerializedWeights,
  type BrowserTrainingKind,
  type BrowserTrainingResult,
  type BrowserTrainingState,
  type TrainingPoint,
} from "./trainTinyGpt";
import { parseBrowserTrainingState } from "./browserTrainingStorage";
import type { CustomTrainingResult } from "./trainCustomGpt";
import type { Solid2TrainingResult } from "./trainSolid2Gpt";
import type { TinyGptCheckpoint, TinyGptConfig } from "./tinyGpt";
import type { Solid2Checkpoint } from "./tinyGptSolid2";
import { parseBpeModel, type BpeModel } from "@only-gates-ai/core";
import {
  serializeBpeTokenizer,
  type BpeTokenizer,
} from "./tinyGptBpe";
import {
  packTinyGptWeights,
  tinyGptParameterCount,
  tinyGptParameterLengths,
  unpackTinyGptWeights,
} from "./tinyGptWasm";
import {
  createZip,
  readZip,
  type ZipFileInput,
  type ZipSource,
} from "./zipArchive";

/*
 * Portable model bundles for browser-trained Integer GPT models.
 *
 * A bundle is a standard ZIP containing the float32 weights, the tokenizer,
 * a manifest, a human-readable training summary, and optionally the full
 * browser training state (weights + Adam moments + best-validation weights)
 * so another person can continue the same run.
 */

export type ModelBundleKind = BrowserTrainingKind;

export interface ModelBundleSource {
  readonly kind: ModelBundleKind;
  readonly checkpoint: TinyGptCheckpoint | Solid2Checkpoint;
  readonly tokenizer: BpeTokenizer | null;
  readonly wordVocab: readonly string[] | null;
  readonly weights: Float32Array;
  readonly state: BrowserTrainingState;
  readonly prompts: readonly string[];
  readonly backend?: string;
}

export interface ModelBundleMeta {
  readonly name: string;
  readonly source: string;
  readonly license: string;
  readonly training?: Record<string, unknown>;
}

export interface ModelBundleTrainingInfo {
  readonly steps: number;
  readonly batch: number;
  readonly learningRate: number;
  readonly dropout: number;
  readonly seed: number;
  readonly bestStep: number;
  readonly bestLoss: number | null;
  readonly elapsedMs: number;
  readonly corpusHash: string;
  readonly resumable: boolean;
  readonly stateFile?: string;
  readonly sourceUrls?: readonly string[];
  readonly assistant?: boolean;
  readonly backend?: string;
}

export interface ModelBundleManifest {
  readonly format: "only-gates-model-bundle";
  readonly version: 1;
  readonly kind: ModelBundleKind;
  readonly createdAt: string;
  readonly generator: {
    readonly app: string;
    readonly feature: string;
  };
  readonly target: {
    readonly panel: "integer-gpt";
    readonly example: ModelBundleKind;
    readonly runtime: "integer-wasm";
    readonly promptFormat: "chat" | "text";
  };
  readonly model: {
    readonly name: string;
    readonly architecture: "tiny-gpt";
    readonly config: TinyGptConfig;
    readonly parameterCount: number;
    readonly tokenizer: {
      readonly file: string;
      readonly type: "bpe" | "word";
    };
    readonly weights: {
      readonly file: string;
      readonly dtype: "float32";
      readonly endianness: "little";
      readonly layout: "tiny_gpt_create";
    };
    readonly prompts: readonly string[];
    readonly samples: readonly { readonly prompt: string; readonly text: string }[];
    readonly meta: ModelBundleMeta;
  };
  readonly training: ModelBundleTrainingInfo;
  readonly integrity?: {
    readonly algorithm: "sha256";
    readonly weights: string;
  };
}

export interface PreparedModelBundle {
  readonly manifest: ModelBundleManifest;
  readonly zip: Uint8Array;
  readonly filename: string;
}

export interface ImportedAliceBundle {
  readonly kind: "alice";
  readonly manifest: ModelBundleManifest;
  readonly checkpoint: TinyGptCheckpoint;
  readonly tokenizer: null;
  readonly vocab: readonly string[];
  readonly weights: Float32Array;
  readonly prompts: readonly string[];
  readonly trainingState: BrowserTrainingState | null;
}

export interface ImportedBpeBundle {
  readonly kind: "solid2" | "custom";
  readonly manifest: ModelBundleManifest;
  readonly checkpoint: Solid2Checkpoint;
  readonly tokenizer: BpeTokenizer;
  readonly vocab: null;
  readonly weights: Float32Array;
  readonly prompts: readonly string[];
  readonly trainingState: BrowserTrainingState | null;
}

export type ImportedModelBundle = ImportedAliceBundle | ImportedBpeBundle;

export interface CreateModelBundleOptions {
  /** Include the full BrowserTrainingState so another person can resume. */
  readonly includeResumeState?: boolean;
}

const MAX_MANIFEST_BYTES = 2 * 1024 * 1024;
const MAX_TRAINING_STATE_BYTES = 256 * 1024 * 1024;
const MAX_PROMPTS = 64;
const MAX_SAMPLES = 64;
const MAX_TEXT = 200_000;

/* ------------------------------------------------------------------ */
/* Source builders                                                    */
/* ------------------------------------------------------------------ */

export function aliceBundleSource(
  result: BrowserTrainingResult,
): ModelBundleSource {
  return {
    kind: "alice",
    checkpoint: result.checkpoint,
    tokenizer: null,
    wordVocab: result.checkpoint.vocab,
    weights: packTinyGptWeights(result.checkpoint),
    state: result.state,
    prompts: [],
    backend: result.backend,
  };
}

export function solid2BundleSource(
  result: Solid2TrainingResult,
): ModelBundleSource {
  return {
    kind: "solid2",
    checkpoint: result.checkpoint,
    tokenizer: result.tokenizer,
    wordVocab: null,
    weights: result.weights,
    state: result.result.state,
    prompts: result.prompts,
    backend: result.result.backend,
  };
}

export function customBundleSource(
  result: CustomTrainingResult,
): ModelBundleSource {
  return {
    kind: "custom",
    checkpoint: result.checkpoint,
    tokenizer: result.tokenizer,
    wordVocab: null,
    weights: result.weights,
    state: result.result.state,
    prompts: result.prompts,
    backend: result.result.backend,
  };
}

/* ------------------------------------------------------------------ */
/* Export                                                             */
/* ------------------------------------------------------------------ */

function checkpointMeta(source: ModelBundleSource): ModelBundleMeta {
  const meta = source.checkpoint.meta;
  return {
    name: meta?.name ?? `${source.kind}-browser-model`,
    source: meta?.source ?? "Trained in the Only Gates AI browser trainer.",
    license:
      meta?.license ??
      "Model weights generated in the browser by the Only Gates AI app.",
    training: meta?.training as Record<string, unknown> | undefined,
  };
}

function checkpointSamples(
  source: ModelBundleSource,
): readonly { prompt: string; text: string }[] {
  return source.checkpoint.samples ?? [];
}

function checkpointPrompts(source: ModelBundleSource): readonly string[] {
  if (source.prompts.length > 0) return source.prompts;
  const prompts = "prompts" in source.checkpoint
    ? (source.checkpoint as Solid2Checkpoint).prompts
    : undefined;
  return prompts ?? [];
}

function promptFormatFor(source: ModelBundleSource): "chat" | "text" {
  if (source.kind === "custom") {
    const training = source.checkpoint.meta?.training as
      | Record<string, unknown>
      | undefined;
    return training?.assistant === true ? "chat" : "text";
  }
  return source.kind === "solid2" ? "chat" : "text";
}

function trainingInfo(
  source: ModelBundleSource,
  includeResumeState: boolean,
): ModelBundleTrainingInfo {
  const state = source.state;
  return {
    steps: state.steps,
    batch: state.batch,
    learningRate: state.lr,
    dropout: state.dropout,
    seed: state.seed,
    bestStep: state.bestStep,
    bestLoss: Number.isFinite(state.bestLoss) ? state.bestLoss : null,
    elapsedMs: state.elapsedMs,
    corpusHash: state.corpusHash,
    resumable: includeResumeState,
    stateFile: includeResumeState ? "training-state.json" : undefined,
    sourceUrls: state.sourceUrls,
    assistant: state.assistant,
    backend: source.backend,
  };
}

function buildManifest(
  source: ModelBundleSource,
  includeResumeState: boolean,
  integrity: ModelBundleManifest["integrity"],
): ModelBundleManifest {
  const config = source.checkpoint.config;
  const meta = checkpointMeta(source);
  const prompts = checkpointPrompts(source);
  const samples = checkpointSamples(source);
  return {
    format: "only-gates-model-bundle",
    version: 1,
    kind: source.kind,
    createdAt: new Date().toISOString(),
    generator: {
      app: "only-gates-ai",
      feature: "browser-training",
    },
    target: {
      panel: "integer-gpt",
      example: source.kind,
      runtime: "integer-wasm",
      promptFormat: promptFormatFor(source),
    },
    model: {
      name: meta.name,
      architecture: "tiny-gpt",
      config,
      parameterCount: tinyGptParameterCount(config),
      tokenizer: {
        file: "tokenizer.json",
        type: source.tokenizer === null ? "word" : "bpe",
      },
      weights: {
        file: "weights.bin",
        dtype: "float32",
        endianness: "little",
        layout: "tiny_gpt_create",
      },
      prompts: prompts.map((prompt) => prompt.slice(0, 4000)),
      samples: samples.map((sample) => ({
        prompt: sample.prompt.slice(0, 4000),
        text: sample.text.slice(0, MAX_TEXT),
      })),
      meta,
    },
    training: trainingInfo(source, includeResumeState),
    integrity,
  };
}

function weightBytes(weights: Float32Array): Uint8Array {
  return new Uint8Array(
    weights.buffer,
    weights.byteOffset,
    weights.byteLength,
  ).slice();
}

function checkpointJson(
  source: ModelBundleSource,
  manifest: ModelBundleManifest,
): string | null {
  if (source.kind === "alice") return null;
  const checkpoint = source.checkpoint as Solid2Checkpoint;
  return JSON.stringify(
    {
      ...checkpoint,
      tokenizer: "tokenizer.json",
      config: manifest.model.config,
      parameterCount: manifest.model.parameterCount,
      meta: manifest.model.meta,
      samples: manifest.model.samples,
      prompts: manifest.model.prompts,
    },
    null,
    2,
  );
}

function tokenizerJson(source: ModelBundleSource): string {
  if (source.tokenizer !== null) {
    return JSON.stringify(serializeBpeTokenizer(source.tokenizer), null, 2);
  }
  if (source.wordVocab === null) {
    throw new Error("A word-level model bundle needs a vocabulary.");
  }
  return JSON.stringify(
    { version: 1, kind: "word", vocab: source.wordVocab },
    null,
  );
}

function trainingSummary(source: ModelBundleSource): string {
  const state = source.state;
  return JSON.stringify(
    {
      format: "only-gates-training-summary",
      version: 1,
      kind: source.kind,
      training: {
        ...trainingInfo(source, false),
        resumable: false,
      },
      history: state.history.map((point: TrainingPoint) => ({
        step: point.step,
        loss: point.loss,
        smoothedLoss: point.smoothedLoss,
        valLoss: point.valLoss,
        lr: point.lr,
        elapsedMs: point.elapsedMs,
      })),
    },
    null,
    2,
  );
}

function readmeText(
  manifest: ModelBundleManifest,
  includeResumeState: boolean,
): string {
  const c = manifest.model.config;
  const lines: (string | null)[] = [
    "Only Gates AI — Integer GPT model bundle",
    "",
    `Kind: ${manifest.kind}`,
    `Model: ${manifest.model.name}`,
    `Created: ${manifest.createdAt}`,
    `Architecture: ${manifest.model.architecture}`,
    `Config: ${c.vocabSize} vocab / ${c.context} context / ${c.dim} dim / ${c.layers} layers / ${c.heads} heads / ${c.mlp} mlp`,
    `Parameters: ${manifest.model.parameterCount.toLocaleString()}`,
    "",
    "Import this file from the browser training panel in the Integer GPT demo.",
    'Choose "Import model (.zip)" and the app will load the matching tab.',
    "",
  ];
  if (includeResumeState) {
    lines.push(
      "Resume state is included. After importing, click \"Resume saved run\"",
      "in the training panel to continue from the saved step and optimizer state.",
      "",
    );
  } else {
    lines.push(
      "This bundle is inference-only and cannot be resumed for further training.",
      "",
    );
  }
  lines.push(
    "Files:",
    "  manifest.json        describes the model and where it loads",
    "  checkpoint.json      model metadata (BPE models)",
    "  tokenizer.json       tokenizer definition",
    "  weights.bin          float32 little-endian weights",
    "  training.json        training summary and loss history",
    includeResumeState ? "  training-state.json  full resume state" : null,
  );
  return lines.filter((line): line is string => line !== null).join("\n");
}

function bundleFilename(kind: ModelBundleKind): string {
  const stamp = new Date()
    .toISOString()
    .replace(/\.\d{3}Z$/, "Z")
    .replace(/:/g, "-");
  return `only-gates-${kind}-${stamp}.zip`;
}

async function sha256Hex(bytes: Uint8Array): Promise<string | null> {
  const subtle = globalThis.crypto?.subtle;
  if (!subtle) return null;
  const digest = await subtle.digest("SHA-256", new Uint8Array(bytes));
  return Array.from(new Uint8Array(digest), (byte) =>
    byte.toString(16).padStart(2, "0"),
  ).join("");
}

export async function createModelBundle(
  source: ModelBundleSource,
  options: CreateModelBundleOptions = {},
): Promise<PreparedModelBundle> {
  const includeResumeState = options.includeResumeState ?? true;
  if (source.state.kind !== source.kind) {
    throw new Error("Training state belongs to a different model kind.");
  }
  if (!configsEqual(source.state.config, source.checkpoint.config)) {
    throw new Error("Training state config does not match the model.");
  }
  const expectedWeights = tinyGptParameterCount(source.checkpoint.config);
  if (source.weights.length !== expectedWeights) {
    throw new Error(
      `This model has ${source.weights.length.toLocaleString()} weights but its config needs ` +
        `${expectedWeights.toLocaleString()}.`,
    );
  }
  const weights = weightBytes(source.weights);
  const digest = await sha256Hex(weights);
  const manifest = buildManifest(
    source,
    includeResumeState,
    digest === null
      ? undefined
      : { algorithm: "sha256", weights: digest },
  );

  const files: ZipFileInput[] = [
    { name: "manifest.json", data: JSON.stringify(manifest, null, 2) },
    { name: "tokenizer.json", data: tokenizerJson(source) },
    { name: "weights.bin", data: weights },
    { name: "training.json", data: trainingSummary(source) },
    { name: "README.txt", data: readmeText(manifest, includeResumeState) },
  ];
  const checkpoint = checkpointJson(source, manifest);
  if (checkpoint !== null) {
    files.push({ name: "checkpoint.json", data: checkpoint });
  }
  if (includeResumeState) {
    files.push({
      name: "training-state.json",
      data: JSON.stringify(source.state),
    });
  }

  const zip = await createZip(files);
  return { manifest, zip, filename: bundleFilename(source.kind) };
}

export function downloadModelBundle(bundle: PreparedModelBundle): void {
  const blob = new Blob([new Uint8Array(bundle.zip)], {
    type: "application/zip",
  });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = bundle.filename;
  anchor.rel = "noopener";
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

export function formatByteSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/* ------------------------------------------------------------------ */
/* Import                                                             */
/* ------------------------------------------------------------------ */

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function expectRecord(value: unknown, label: string): Record<string, unknown> {
  if (!isRecord(value)) throw new Error(`${label} must be an object.`);
  return value;
}

function expectString(value: unknown, label: string): string {
  if (typeof value !== "string") throw new Error(`${label} must be a string.`);
  return value;
}

function expectNumber(value: unknown, label: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`${label} must be a finite number.`);
  }
  return value;
}

function expectInteger(value: unknown, label: string): number {
  const number = expectNumber(value, label);
  if (!Number.isSafeInteger(number)) {
    throw new Error(`${label} must be an integer.`);
  }
  return number;
}

function expectBoolean(value: unknown, label: string): boolean {
  if (typeof value !== "boolean") throw new Error(`${label} must be a boolean.`);
  return value;
}

function expectStringArray(
  value: unknown,
  label: string,
  limit: number,
): string[] {
  if (!Array.isArray(value)) throw new Error(`${label} must be an array.`);
  if (value.length > limit) throw new Error(`${label} has too many entries.`);
  return value.map((entry, index) =>
    expectString(entry, `${label}[${index}]`).slice(0, MAX_TEXT),
  );
}

function parseConfig(value: unknown): TinyGptConfig {
  const config = expectRecord(value, "model.config");
  const parsed = {
    vocabSize: expectInteger(config.vocabSize, "model.config.vocabSize"),
    context: expectInteger(config.context, "model.config.context"),
    dim: expectInteger(config.dim, "model.config.dim"),
    layers: expectInteger(config.layers, "model.config.layers"),
    heads: expectInteger(config.heads, "model.config.heads"),
    mlp: expectInteger(config.mlp, "model.config.mlp"),
  };
  if (parsed.vocabSize < 1 || parsed.vocabSize > 65536) {
    throw new Error("model.config.vocabSize is out of range.");
  }
  if (parsed.context < 1 || parsed.context > 4096) {
    throw new Error("model.config.context is out of range.");
  }
  if (parsed.dim < 1 || parsed.dim > 4096) {
    throw new Error("model.config.dim is out of range.");
  }
  if (parsed.layers < 0 || parsed.layers > 64) {
    throw new Error("model.config.layers is out of range.");
  }
  if (parsed.heads < 1 || parsed.heads > 256) {
    throw new Error("model.config.heads is out of range.");
  }
  if (parsed.mlp < 1 || parsed.mlp > 32768) {
    throw new Error("model.config.mlp is out of range.");
  }
  if (parsed.dim % parsed.heads !== 0) {
    throw new Error("model.config.dim must be divisible by heads.");
  }
  if (tinyGptParameterCount(parsed) > 64_000_000) {
    throw new Error("Model bundle is larger than this app supports.");
  }
  return parsed;
}

function parseMeta(value: unknown, label: string): ModelBundleMeta {
  const meta = expectRecord(value, label);
  const training = meta.training;
  return {
    name: expectString(meta.name, `${label}.name`).slice(0, 400),
    source: expectString(meta.source, `${label}.source`).slice(0, 4000),
    license: expectString(meta.license, `${label}.license`).slice(0, 4000),
    training: isRecord(training) ? training : undefined,
  };
}

function parseSamples(value: unknown): { prompt: string; text: string }[] {
  if (!Array.isArray(value)) {
    throw new Error("model.samples must be an array.");
  }
  if (value.length > MAX_SAMPLES) {
    throw new Error("model.samples has too many entries.");
  }
  return value.map((entry, index) => {
    const sample = expectRecord(entry, `model.samples[${index}]`);
    return {
      prompt: expectString(sample.prompt, `model.samples[${index}].prompt`).slice(
        0,
        4000,
      ),
      text: expectString(sample.text, `model.samples[${index}].text`).slice(
        0,
        MAX_TEXT,
      ),
    };
  });
}

function parseTrainingInfo(value: unknown): ModelBundleTrainingInfo {
  const training = expectRecord(value, "training");
  const bestLoss = training.bestLoss;
  const resumable = expectBoolean(training.resumable, "training.resumable");
  const stateFile = resumable
    ? expectString(training.stateFile, "training.stateFile")
    : undefined;
  return {
    steps: expectInteger(training.steps, "training.steps"),
    batch: expectInteger(training.batch, "training.batch"),
    learningRate: expectNumber(training.learningRate, "training.learningRate"),
    dropout: expectNumber(training.dropout, "training.dropout"),
    seed: expectInteger(training.seed, "training.seed"),
    bestStep: expectInteger(training.bestStep, "training.bestStep"),
    bestLoss:
      bestLoss === null ? null : expectNumber(bestLoss, "training.bestLoss"),
    elapsedMs: expectNumber(training.elapsedMs, "training.elapsedMs"),
    corpusHash: expectString(training.corpusHash, "training.corpusHash"),
    resumable,
    stateFile,
    sourceUrls:
      training.sourceUrls === undefined
        ? undefined
        : expectStringArray(training.sourceUrls, "training.sourceUrls", 256),
    assistant:
      training.assistant === undefined
        ? undefined
        : expectBoolean(training.assistant, "training.assistant"),
    backend:
      training.backend === undefined
        ? undefined
        : expectString(training.backend, "training.backend").slice(0, 100),
  };
}

function parseIntegrity(
  value: unknown,
): ModelBundleManifest["integrity"] {
  if (value === undefined) return undefined;
  const integrity = expectRecord(value, "integrity");
  if (integrity.algorithm !== "sha256") {
    throw new Error("Only sha256 bundle integrity is supported.");
  }
  const weights = expectString(integrity.weights, "integrity.weights");
  if (!/^[a-f0-9]{64}$/i.test(weights)) {
    throw new Error("integrity.weights is not a sha256 digest.");
  }
  return { algorithm: "sha256", weights };
}

function parseManifest(value: unknown): ModelBundleManifest {
  const root = expectRecord(value, "manifest.json");
  if (root.format !== "only-gates-model-bundle") {
    throw new Error("This ZIP is not an Only Gates model bundle.");
  }
  if (root.version !== 1) {
    throw new Error(`Unsupported model bundle version ${String(root.version)}.`);
  }
  const kind = root.kind;
  if (kind !== "alice" && kind !== "solid2" && kind !== "custom") {
    throw new Error("manifest.json has an unknown model kind.");
  }
  const generator = expectRecord(root.generator, "generator");
  const target = expectRecord(root.target, "target");
  if (target.panel !== "integer-gpt") {
    throw new Error("manifest.json targets an unknown panel.");
  }
  if (target.example !== kind) {
    throw new Error("manifest.json target does not match the model kind.");
  }
  if (target.runtime !== "integer-wasm") {
    throw new Error("manifest.json targets an unknown runtime.");
  }
  if (target.promptFormat !== "chat" && target.promptFormat !== "text") {
    throw new Error("manifest.json has an unknown prompt format.");
  }

  const model = expectRecord(root.model, "model");
  if (model.architecture !== "tiny-gpt") {
    throw new Error("Model bundle architecture must be tiny-gpt.");
  }
  const config = parseConfig(model.config);
  const parameterCount = expectInteger(model.parameterCount, "model.parameterCount");
  const expectedParameters = tinyGptParameterCount(config);
  if (parameterCount !== expectedParameters) {
    throw new Error(
      `Model bundle declares ${parameterCount.toLocaleString()} parameters but its config needs ` +
        `${expectedParameters.toLocaleString()}.`,
    );
  }

  const tokenizer = expectRecord(model.tokenizer, "model.tokenizer");
  const tokenizerType = tokenizer.type;
  if (tokenizerType !== "bpe" && tokenizerType !== "word") {
    throw new Error("model.tokenizer.type must be bpe or word.");
  }
  const weights = expectRecord(model.weights, "model.weights");
  if (weights.dtype !== undefined && weights.dtype !== "float32") {
    throw new Error("Only float32 model weights are supported.");
  }
  if (weights.endianness !== undefined && weights.endianness !== "little") {
    throw new Error("Only little-endian model weights are supported.");
  }
  if (weights.layout !== undefined && weights.layout !== "tiny_gpt_create") {
    throw new Error("Unsupported model weight layout.");
  }

  return {
    format: "only-gates-model-bundle",
    version: 1,
    kind,
    createdAt: expectString(root.createdAt, "createdAt"),
    generator: {
      app: expectString(generator.app, "generator.app"),
      feature: expectString(generator.feature, "generator.feature"),
    },
    target: {
      panel: "integer-gpt",
      example: kind,
      runtime: "integer-wasm",
      promptFormat: target.promptFormat,
    },
    model: {
      name: expectString(model.name, "model.name").slice(0, 400),
      architecture: "tiny-gpt",
      config,
      parameterCount,
      tokenizer: {
        file: expectString(tokenizer.file, "model.tokenizer.file"),
        type: tokenizerType,
      },
      weights: {
        file: expectString(weights.file, "model.weights.file"),
        dtype: "float32",
        endianness: "little",
        layout: "tiny_gpt_create",
      },
      prompts: expectStringArray(model.prompts, "model.prompts", MAX_PROMPTS),
      samples: parseSamples(model.samples),
      meta: parseMeta(model.meta, "model.meta"),
    },
    training: parseTrainingInfo(root.training),
    integrity: parseIntegrity(root.integrity),
  };
}

function decodeUtf8(bytes: Uint8Array, label: string): string {
  try {
    return new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  } catch {
    throw new Error(`${label} is not valid UTF-8.`);
  }
}

function parseJson(bytes: Uint8Array, label: string): unknown {
  try {
    return JSON.parse(decodeUtf8(bytes, label)) as unknown;
  } catch (error) {
    if (error instanceof SyntaxError) {
      throw new Error(`${label} is not valid JSON.`);
    }
    throw error;
  }
}

function requireFile(
  files: ReadonlyMap<string, Uint8Array>,
  name: string,
): Uint8Array {
  const file = files.get(name);
  if (!file) throw new Error(`Model bundle is missing ${name}.`);
  return file;
}

function configsEqual(a: TinyGptConfig, b: TinyGptConfig): boolean {
  return (
    a.vocabSize === b.vocabSize &&
    a.context === b.context &&
    a.dim === b.dim &&
    a.layers === b.layers &&
    a.heads === b.heads &&
    a.mlp === b.mlp
  );
}

function vocabEqual(a: readonly string[], b: readonly string[]): boolean {
  if (a.length !== b.length) return false;
  for (let index = 0; index < a.length; index++) {
    if (a[index] !== b[index]) return false;
  }
  return true;
}

function base64FloatCount(value: string): number {
  if (
    value.length === 0 ||
    value.length % 4 !== 0 ||
    !/^[A-Za-z0-9+/]*={0,2}$/.test(value)
  ) {
    throw new Error("Resume state contains malformed base64 weights.");
  }
  const padding = value.endsWith("==") ? 2 : value.endsWith("=") ? 1 : 0;
  const bytes = (value.length / 4) * 3 - padding;
  if (bytes % 4 !== 0) {
    throw new Error("Resume state contains a partial float32 value.");
  }
  return bytes / 4;
}

function validateSerializedWeights(
  config: TinyGptConfig,
  weights: BrowserTrainingState["weights"],
  label: string,
): void {
  const flat = flattenSerializedWeights(config, weights);
  const expected = tinyGptParameterLengths(config);
  if (flat.length !== expected.length) {
    throw new Error(`${label} has the wrong number of tensors.`);
  }
  flat.forEach((value, index) => {
    const length = base64FloatCount(value);
    if (length !== expected[index]) {
      throw new Error(
        `${label} tensor ${index} has ${length.toLocaleString()} values but the config needs ` +
          `${expected[index]!.toLocaleString()}.`,
      );
    }
  });
}

function validateTrainingState(
  state: BrowserTrainingState,
  manifest: ModelBundleManifest,
): void {
  if (state.kind !== manifest.kind) {
    throw new Error("Resume state belongs to a different model kind.");
  }
  if (!configsEqual(state.config, manifest.model.config)) {
    throw new Error("Resume state config does not match the model bundle.");
  }
  if (state.vocab.length !== manifest.model.config.vocabSize) {
    throw new Error("Resume state vocabulary does not match the model config.");
  }
  validateSerializedWeights(state.config, state.weights, "Resume state weights");
  validateSerializedWeights(
    state.config,
    state.optimizer.m,
    "Resume state optimizer m",
  );
  validateSerializedWeights(
    state.config,
    state.optimizer.v,
    "Resume state optimizer v",
  );
  if (state.bestWeights !== null) {
    validateSerializedWeights(
      state.config,
      state.bestWeights,
      "Resume state best weights",
    );
  }
}

function parseWordVocab(value: unknown, config: TinyGptConfig): string[] {
  const root = expectRecord(value, "tokenizer.json");
  if (root.version !== 1) {
    throw new Error("Unsupported word tokenizer version.");
  }
  if (root.kind !== "word") {
    throw new Error("Expected a word tokenizer for this model bundle.");
  }
  if (!Array.isArray(root.vocab)) {
    throw new Error("Word tokenizer is missing its vocabulary.");
  }
  if (root.vocab.length !== config.vocabSize) {
    throw new Error(
      `Tokenizer has ${root.vocab.length.toLocaleString()} tokens but the config needs ` +
        `${config.vocabSize.toLocaleString()}.`,
    );
  }
  return root.vocab.map((entry, index) =>
    expectString(entry, `tokenizer.json vocab[${index}]`),
  );
}

function parseSolid2Checkpoint(
  value: unknown,
  manifest: ModelBundleManifest,
): Solid2Checkpoint {
  const root = expectRecord(value, "checkpoint.json");
  const config = parseConfig(root.config);
  if (!configsEqual(config, manifest.model.config)) {
    throw new Error("checkpoint.json config does not match manifest.json.");
  }
  const format = expectString(root.format, "checkpoint.json format");
  if (!format.startsWith("only-gates-tiny-gpt")) {
    throw new Error("checkpoint.json is not an Only Gates GPT checkpoint.");
  }
  if (root.version !== 1) {
    throw new Error("Unsupported checkpoint.json version.");
  }
  return {
    format,
    version: 1,
    config,
    parameterCount: manifest.model.parameterCount,
    tokenizer: "tokenizer.json",
    meta: parseMeta(root.meta, "checkpoint.json meta"),
    samples:
      root.samples === undefined
        ? [...manifest.model.samples]
        : parseSamples(root.samples),
    prompts:
      root.prompts === undefined
        ? [...manifest.model.prompts]
        : expectStringArray(root.prompts, "checkpoint.json prompts", MAX_PROMPTS),
  };
}

export async function importModelBundle(
  source: ZipSource,
): Promise<ImportedModelBundle> {
  const files = await readZip(source);
  const manifestBytes = requireFile(files, "manifest.json");
  if (manifestBytes.byteLength > MAX_MANIFEST_BYTES) {
    throw new Error("manifest.json is too large.");
  }
  const manifest = parseManifest(parseJson(manifestBytes, "manifest.json"));

  const weightsBytes = requireFile(files, manifest.model.weights.file);
  const expectedWeightBytes = manifest.model.parameterCount * 4;
  if (weightsBytes.byteLength !== expectedWeightBytes) {
    throw new Error(
      `weights.bin has ${weightsBytes.byteLength.toLocaleString()} bytes but the config needs ` +
        `${expectedWeightBytes.toLocaleString()}.`,
    );
  }
  if (manifest.integrity !== undefined) {
    const digest = await sha256Hex(weightsBytes);
    if (digest !== null && digest !== manifest.integrity.weights.toLowerCase()) {
      throw new Error("weights.bin does not match the manifest checksum.");
    }
  }
  const copy = weightsBytes.slice();
  const weights = new Float32Array(
    copy.buffer,
    0,
    copy.byteLength / Float32Array.BYTES_PER_ELEMENT,
  );

  const tokenizerBytes = requireFile(files, manifest.model.tokenizer.file);
  const tokenizerValue = parseJson(tokenizerBytes, manifest.model.tokenizer.file);

  let tokenizer: BpeTokenizer | null = null;
  let vocab: readonly string[] | null = null;
  if (manifest.model.tokenizer.type === "bpe") {
    let parsed: BpeTokenizer;
    try {
      parsed = parseBpeModel(tokenizerValue as BpeModel);
    } catch (error) {
      throw new Error(
        `tokenizer.json is not a valid BPE tokenizer: ${(error as Error).message}`,
      );
    }
    if (parsed.size !== manifest.model.config.vocabSize) {
      throw new Error(
        `Tokenizer has ${parsed.size.toLocaleString()} tokens but the config needs ` +
          `${manifest.model.config.vocabSize.toLocaleString()}.`,
      );
    }
    tokenizer = parsed;
  } else {
    vocab = parseWordVocab(tokenizerValue, manifest.model.config);
  }

  let trainingState: BrowserTrainingState | null = null;
  if (manifest.training.resumable) {
    const stateFile = manifest.training.stateFile ?? "training-state.json";
    const stateBytes = requireFile(files, stateFile);
    if (stateBytes.byteLength > MAX_TRAINING_STATE_BYTES) {
      throw new Error("training-state.json is too large.");
    }
    try {
      trainingState = parseBrowserTrainingState(
        parseJson(stateBytes, stateFile),
      );
      validateTrainingState(trainingState, manifest);
      const expectedVocab = tokenizer?.vocab ?? vocab;
      if (
        expectedVocab !== null &&
        !vocabEqual(trainingState.vocab, expectedVocab)
      ) {
        throw new Error("vocabulary does not match the tokenizer");
      }
    } catch (error) {
      throw new Error(
        `Resume state is invalid: ${(error as Error).message}`,
      );
    }
  }

  if (manifest.kind === "alice") {
    if (vocab === null) {
      throw new Error("An Alice model bundle needs a word tokenizer.");
    }
    const checkpoint: TinyGptCheckpoint = {
      config: manifest.model.config,
      meta: {
        name: manifest.model.meta.name,
        source: manifest.model.meta.source,
        license: manifest.model.meta.license,
        training: (manifest.model.meta.training ?? {}) as Readonly<
          Record<string, number>
        >,
      },
      vocab,
      weights: unpackTinyGptWeights(manifest.model.config, weights),
      samples: manifest.model.samples,
    };
    return {
      kind: "alice",
      manifest,
      checkpoint,
      tokenizer: null,
      vocab,
      weights,
      prompts: manifest.model.prompts,
      trainingState,
    };
  }

  if (tokenizer === null) {
    throw new Error("A BPE model bundle needs a tokenizer.");
  }
  const checkpointBytes = files.get("checkpoint.json");
  const checkpoint =
    checkpointBytes === undefined
      ? {
          format: `only-gates-tiny-gpt-${manifest.kind}`,
          version: 1,
          config: manifest.model.config,
          parameterCount: manifest.model.parameterCount,
          tokenizer: "tokenizer.json",
          meta: manifest.model.meta,
          samples: [...manifest.model.samples],
          prompts: [...manifest.model.prompts],
        }
      : parseSolid2Checkpoint(
          parseJson(checkpointBytes, "checkpoint.json"),
          manifest,
        );
  return {
    kind: manifest.kind === "custom" ? "custom" : "solid2",
    manifest,
    checkpoint,
    tokenizer,
    vocab: null,
    weights,
    prompts: checkpoint.prompts ?? manifest.model.prompts,
    trainingState,
  };
}
