/*
 * IndexedDB cache for tokenized training corpora.
 *
 * Training data is expensive to tokenize, especially with the 4096-token BPE
 * tokenizer, so each trainer keeps its token streams in its own database. The
 * cache is best-effort: a browser without IndexedDB still trains, it just pays
 * the tokenization cost on every run.
 */

const CACHE_STORE = "token-streams";

export interface CachedTokenStream {
  readonly key: string;
  readonly trainingTokens: Uint16Array;
  readonly validationTokens: Uint16Array;
  readonly savedAt: number;
}

export interface TokenStreamCache {
  read(key: string): Promise<CachedTokenStream | null>;
  write(entry: CachedTokenStream): Promise<void>;
}

function openDatabase(name: string): Promise<IDBDatabase | null> {
  return new Promise((resolve) => {
    if (typeof indexedDB === "undefined") {
      resolve(null);
      return;
    }
    const request = indexedDB.open(name, 1);
    request.onupgradeneeded = () => {
      const database = request.result;
      if (!database.objectStoreNames.contains(CACHE_STORE)) {
        database.createObjectStore(CACHE_STORE);
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => resolve(null);
    request.onblocked = () => resolve(null);
  });
}

function withTokenStore<T>(
  databaseName: string,
  mode: IDBTransactionMode,
  operation: (store: IDBObjectStore) => IDBRequest<T>,
): Promise<T | null> {
  return openDatabase(databaseName).then((database) => {
    if (database === null) return null;
    return new Promise<T | null>((resolve) => {
      let settled = false;
      const finish = (value: T | null): void => {
        if (settled) return;
        settled = true;
        database.close();
        resolve(value);
      };
      try {
        const transaction = database.transaction(CACHE_STORE, mode);
        const request = operation(transaction.objectStore(CACHE_STORE));
        let result: T | undefined;
        request.onsuccess = () => {
          result = request.result;
        };
        request.onerror = () => finish(null);
        transaction.oncomplete = () => finish(result ?? null);
        transaction.onerror = () => finish(null);
        transaction.onabort = () => finish(null);
      } catch {
        finish(null);
      }
    });
  });
}

function isCachedTokenStream(value: unknown): value is CachedTokenStream {
  if (typeof value !== "object" || value === null) return false;
  const record = value as Record<string, unknown>;
  return (
    record.trainingTokens instanceof Uint16Array &&
    record.validationTokens instanceof Uint16Array
  );
}

/** Creates a best-effort token-stream cache backed by one IndexedDB database. */
export function createTokenStreamCache(databaseName: string): TokenStreamCache {
  return {
    async read(key: string): Promise<CachedTokenStream | null> {
      const value = await withTokenStore<unknown>(
        databaseName,
        "readonly",
        (store) => store.get(key),
      );
      return isCachedTokenStream(value) ? value : null;
    },
    async write(entry: CachedTokenStream): Promise<void> {
      await withTokenStore(databaseName, "readwrite", (store) =>
        store.put(entry, entry.key),
      );
    },
  };
}
