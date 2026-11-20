import type { TrainingPoint } from "./trainTinyGpt";
import type { TinyGptConfig } from "./tinyGpt";

/*
 * IndexedDB persistence for the masked-diffusion browser trainer.
 *
 * The masked model is much larger than the Alice trainer (9.5M parameters in
 * the deployed Luau checkpoint), so the resume state stores raw Float32Arrays
 * instead of base64 strings. IndexedDB structured-clones typed arrays
 * efficiently and avoids the ~33% base64 overhead.
 */

const DB_NAME = "gate-forest-masked-diffusion-training";
const DB_VERSION = 1;
const STORE_NAME = "runs";
const RUN_KEY = "current";

export interface MaskedDiffusionOptimizerState {
  readonly m: Float32Array;
  readonly v: Float32Array;
}

export interface MaskedDiffusionTrainingState {
  readonly format: "only-gates-masked-diffusion-train";
  readonly version: 1;
  readonly config: TinyGptConfig;
  readonly step: number;
  readonly steps: number;
  readonly batch: number;
  readonly lr: number;
  readonly dropout: number;
  readonly seed: number;
  readonly bestStep: number;
  readonly bestLoss: number;
  readonly history: readonly TrainingPoint[];
  /** Packed in `tiny_gpt_create` order. */
  readonly weights: Float32Array;
  readonly optimizer: MaskedDiffusionOptimizerState | null;
  readonly bestWeights: Float32Array | null;
  readonly corpusHash: string;
  readonly elapsedMs: number;
  readonly savedAt: number;
  readonly fullData: boolean;
  readonly warmStart: boolean;
}

function openDatabase(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    if (typeof indexedDB === "undefined") {
      reject(new Error("This browser does not support IndexedDB."));
      return;
    }
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = () => {
      const database = request.result;
      if (!database.objectStoreNames.contains(STORE_NAME)) {
        database.createObjectStore(STORE_NAME);
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () =>
      reject(request.error ?? new Error("Could not open IndexedDB."));
    request.onblocked = () =>
      reject(new Error("IndexedDB is blocked by another tab."));
  });
}

function withStore<T>(
  mode: IDBTransactionMode,
  operation: (store: IDBObjectStore) => IDBRequest<T>,
): Promise<T> {
  return openDatabase().then(
    (database) =>
      new Promise<T>((resolve, reject) => {
        let settled = false;
        const close = () => {
          if (!settled) {
            settled = true;
            database.close();
          }
        };
        const fail = (error: unknown) => {
          close();
          reject(error);
        };

        let transaction: IDBTransaction;
        try {
          transaction = database.transaction(STORE_NAME, mode);
        } catch (error) {
          fail(error);
          return;
        }

        let result: T | undefined;
        let request: IDBRequest<T>;
        try {
          request = operation(transaction.objectStore(STORE_NAME));
        } catch (error) {
          fail(error);
          return;
        }
        request.onsuccess = () => {
          result = request.result;
        };
        request.onerror = () =>
          fail(request.error ?? new Error("The IndexedDB request failed."));
        transaction.oncomplete = () => {
          close();
          resolve(result as T);
        };
        transaction.onerror = () =>
          fail(transaction.error ?? new Error("The IndexedDB request failed."));
        transaction.onabort = () =>
          fail(
            transaction.error ?? new Error("The IndexedDB request was aborted."),
          );
      }),
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isFloat32Array(value: unknown): value is Float32Array {
  return value instanceof Float32Array;
}

function parsePoint(value: unknown): TrainingPoint {
  if (!isRecord(value)) {
    throw new Error("Saved masked-diffusion history is malformed.");
  }
  const { step, loss, smoothedLoss, lr, elapsedMs } = value;
  if (
    typeof step !== "number" ||
    typeof loss !== "number" ||
    typeof smoothedLoss !== "number" ||
    typeof lr !== "number" ||
    typeof elapsedMs !== "number"
  ) {
    throw new Error("Saved masked-diffusion history is malformed.");
  }
  return {
    step,
    loss,
    smoothedLoss,
    lr,
    elapsedMs,
    valLoss: typeof value.valLoss === "number" ? value.valLoss : undefined,
  };
}

function parseState(value: unknown): MaskedDiffusionTrainingState {
  if (!isRecord(value) || value.format !== "only-gates-masked-diffusion-train") {
    throw new Error("Saved data is not a masked-diffusion training run.");
  }
  if (value.version !== 1) {
    throw new Error(
      `Unsupported masked-diffusion state version ${String(value.version)}.`,
    );
  }
  if (
    !isRecord(value.config) ||
    !isFloat32Array(value.weights) ||
    !Array.isArray(value.history)
  ) {
    throw new Error("Saved masked-diffusion state is incomplete.");
  }

  const optimizerValue = value.optimizer;
  let optimizer: MaskedDiffusionOptimizerState | null = null;
  if (optimizerValue !== null && optimizerValue !== undefined) {
    if (
      !isRecord(optimizerValue) ||
      !isFloat32Array(optimizerValue.m) ||
      !isFloat32Array(optimizerValue.v)
    ) {
      throw new Error("Saved masked-diffusion optimizer state is malformed.");
    }
    optimizer = { m: optimizerValue.m, v: optimizerValue.v };
  }

  const bestWeightsValue = value.bestWeights;
  if (
    bestWeightsValue !== null &&
    bestWeightsValue !== undefined &&
    !isFloat32Array(bestWeightsValue)
  ) {
    throw new Error("Saved masked-diffusion best weights are malformed.");
  }

  return {
    format: "only-gates-masked-diffusion-train",
    version: 1,
    config: value.config as unknown as TinyGptConfig,
    step: typeof value.step === "number" ? value.step : 0,
    steps: typeof value.steps === "number" ? value.steps : 400,
    batch: typeof value.batch === "number" ? value.batch : 4,
    lr: typeof value.lr === "number" ? value.lr : 4e-4,
    dropout: typeof value.dropout === "number" ? value.dropout : 0.1,
    seed: typeof value.seed === "number" ? value.seed : 1337,
    bestStep: typeof value.bestStep === "number" ? value.bestStep : 0,
    bestLoss:
      typeof value.bestLoss === "number"
        ? value.bestLoss
        : Number.POSITIVE_INFINITY,
    history: value.history.map(parsePoint),
    weights: value.weights,
    optimizer,
    bestWeights:
      bestWeightsValue === null || bestWeightsValue === undefined
        ? null
        : bestWeightsValue,
    corpusHash: typeof value.corpusHash === "string" ? value.corpusHash : "",
    elapsedMs: typeof value.elapsedMs === "number" ? value.elapsedMs : 0,
    savedAt: typeof value.savedAt === "number" ? value.savedAt : Date.now(),
    fullData: value.fullData === true,
    warmStart: value.warmStart !== false,
  };
}

export async function saveMaskedDiffusionTrainingState(
  state: MaskedDiffusionTrainingState,
): Promise<void> {
  await withStore("readwrite", (store) => store.put(state, RUN_KEY));
}

export async function loadMaskedDiffusionTrainingState(): Promise<MaskedDiffusionTrainingState | null> {
  const value = await withStore<unknown>("readonly", (store) =>
    store.get(RUN_KEY),
  );
  if (value === undefined) {
    return null;
  }
  return parseState(value);
}

export async function deleteMaskedDiffusionTrainingState(): Promise<void> {
  await withStore("readwrite", (store) => store.delete(RUN_KEY));
}
