import { parseQaModelFile } from "./qaModel.ts";
import type { QaModelFile } from "./qaRuntime.ts";

const DB_NAME = "gate-forest-qa";
const DB_VERSION = 1;
const STORE_NAME = "models";
const MODEL_KEY = "current";

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

export async function saveQaModelToStorage(
  model: QaModelFile,
): Promise<void> {
  await withStore("readwrite", (store) => store.put(model, MODEL_KEY));
}

export async function loadQaModelFromStorage(): Promise<QaModelFile | null> {
  const value = await withStore("readonly", (store) => store.get(MODEL_KEY));
  if (value === undefined) {
    return null;
  }
  return parseQaModelFile(value);
}

export async function deleteQaModelFromStorage(): Promise<void> {
  await withStore("readwrite", (store) => store.delete(MODEL_KEY));
}
