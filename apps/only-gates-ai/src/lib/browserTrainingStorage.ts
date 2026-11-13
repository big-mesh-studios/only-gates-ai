import type {
  BrowserTrainingKind,
  BrowserTrainingState,
  TrainingPoint,
} from "./trainTinyGpt";

/*
 * IndexedDB persistence for browser training.
 *
 * The payload includes the model, Adam moments, history, and best-validation
 * weights. It is deliberately separate from the symbol/QA model stores so a
 * training run can be cleared without touching the rest of the app.
 */

const DB_NAME = "gate-forest-browser-training";
const DB_VERSION = 1;
const STORE_NAME = "runs";
const DEFAULT_RUN_KEY = "current";

export type BrowserTrainingRunKey = "current" | "solid2" | "custom";

/** Maps a model kind to the IndexedDB run slot used by its training panel. */
export function browserTrainingRunKey(
  kind: BrowserTrainingKind,
): BrowserTrainingRunKey {
  return kind === "alice" ? "current" : kind;
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

function parsePoint(value: unknown): TrainingPoint {
  if (!isRecord(value)) {
    throw new Error("Saved training history is malformed.");
  }
  const step = value.step;
  const loss = value.loss;
  const smoothedLoss = value.smoothedLoss;
  const lr = value.lr;
  const elapsedMs = value.elapsedMs;
  if (
    typeof step !== "number" ||
    typeof loss !== "number" ||
    typeof smoothedLoss !== "number" ||
    typeof lr !== "number" ||
    typeof elapsedMs !== "number"
  ) {
    throw new Error("Saved training history is malformed.");
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

export function parseBrowserTrainingState(value: unknown): BrowserTrainingState {
  if (!isRecord(value) || value.format !== "only-gates-browser-train") {
    throw new Error("Saved training data is not a browser training run.");
  }
  if (value.version !== 1) {
    throw new Error(`Unsupported training state version ${String(value.version)}.`);
  }
  if (
    !isRecord(value.config) ||
    !Array.isArray(value.vocab) ||
    !Array.isArray(value.history) ||
    !isRecord(value.weights) ||
    !isRecord(value.optimizer)
  ) {
    throw new Error("Saved training data is incomplete.");
  }
  return {
    format: "only-gates-browser-train",
    version: 1,
    kind:
      value.kind === "solid2"
        ? "solid2"
        : value.kind === "custom"
          ? "custom"
          : "alice",
    config: value.config as unknown as BrowserTrainingState["config"],
    vocab: value.vocab.filter((entry): entry is string => typeof entry === "string"),
    step: typeof value.step === "number" ? value.step : 0,
    steps: typeof value.steps === "number" ? value.steps : 1800,
    batch: typeof value.batch === "number" ? value.batch : 8,
    lr: typeof value.lr === "number" ? value.lr : 0.003,
    dropout: typeof value.dropout === "number" ? value.dropout : 0,
    seed: typeof value.seed === "number" ? value.seed : 1337,
    bestStep: typeof value.bestStep === "number" ? value.bestStep : 0,
    bestLoss: typeof value.bestLoss === "number" ? value.bestLoss : Number.POSITIVE_INFINITY,
    history: value.history.map(parsePoint),
    weights: value.weights as BrowserTrainingState["weights"],
    optimizer: value.optimizer as BrowserTrainingState["optimizer"],
    bestWeights:
      value.bestWeights === null
        ? null
        : (value.bestWeights as BrowserTrainingState["bestWeights"]),
    corpusHash: typeof value.corpusHash === "string" ? value.corpusHash : "",
    elapsedMs: typeof value.elapsedMs === "number" ? value.elapsedMs : 0,
    savedAt: typeof value.savedAt === "number" ? value.savedAt : Date.now(),
    sourceUrls: Array.isArray(value.sourceUrls)
      ? value.sourceUrls.filter(
          (entry): entry is string => typeof entry === "string",
        )
      : undefined,
    assistant:
      typeof value.assistant === "boolean" ? value.assistant : undefined,
  };
}

export async function saveBrowserTrainingState(
  state: BrowserTrainingState,
  key: BrowserTrainingRunKey = DEFAULT_RUN_KEY,
): Promise<void> {
  await withStore("readwrite", (store) => store.put(state, key));
}

export async function loadBrowserTrainingState(
  key: BrowserTrainingRunKey = DEFAULT_RUN_KEY,
): Promise<BrowserTrainingState | null> {
  const value = await withStore("readonly", (store) => store.get(key));
  if (value === undefined) {
    return null;
  }
  return parseBrowserTrainingState(value);
}

export async function deleteBrowserTrainingState(
  key: BrowserTrainingRunKey = DEFAULT_RUN_KEY,
): Promise<void> {
  await withStore("readwrite", (store) => store.delete(key));
}
