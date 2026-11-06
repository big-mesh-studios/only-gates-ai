import type {
  Circuit,
  HardCircuit,
  OnlyGatesAI,
} from "@only-gates-ai/core";

import { parseShape as parseCircuitShape } from "./model";
import type { LabelEnsemble, ModelInfo } from "./types";

const MODEL_FORMAT = "gate-forest-model" as const;
const MODEL_VERSION = 2 as const;

const DB_NAME = "gate-forest";
const DB_VERSION = 1;
const STORE_NAME = "models";
const MODEL_KEY = "current";

const MAX_STORED_ENSEMBLES = 24;
const MAX_STORED_CIRCUITS = 512;

export interface SerializedCircuit {
  readonly connections: string;
  readonly gates: string;
}

export interface SerializedEnsemble {
  readonly labelId: string;
  readonly labelName: string;
  readonly color: string;
  readonly averageLoss: number;
  readonly positives: number;
  readonly negatives: number;
  readonly trainedAt: number;
  readonly circuits: readonly SerializedCircuit[];
}

/** The complete, JSON-safe representation of a trained forest. */
export interface SerializedModel {
  readonly format: typeof MODEL_FORMAT;
  readonly version: typeof MODEL_VERSION;
  readonly savedAt: number;
  readonly shape: readonly number[];
  readonly gridSize: number;
  readonly labelCount: number;
  readonly totalCircuits: number;
  readonly averageLoss: number;
  readonly trainedAt: number;
  readonly ensembles: readonly SerializedEnsemble[];
}

export interface RestoredModel {
  readonly ensembles: LabelEnsemble[];
  readonly info: ModelInfo;
}

function encodeFloat32(values: Float32Array): string {
  const bytes = new Uint8Array(
    values.buffer,
    values.byteOffset,
    values.byteLength,
  );
  let binary = "";
  const chunkSize = 0x8000;
  for (let offset = 0; offset < bytes.length; offset += chunkSize) {
    binary += String.fromCharCode(
      ...bytes.subarray(offset, offset + chunkSize),
    );
  }
  return btoa(binary);
}

function decodeFloat32(
  value: string,
  expectedCount: number,
  context: string,
): Float32Array {
  let binary: string;
  try {
    binary = atob(value);
  } catch {
    throw new Error(`${context} is not valid model data.`);
  }
  if (binary.length !== expectedCount * Float32Array.BYTES_PER_ELEMENT) {
    throw new Error(
      `${context} has ${Math.floor(
        binary.length / Float32Array.BYTES_PER_ELEMENT,
      )} values, but the circuit needs ${expectedCount}.`,
    );
  }
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index++) {
    bytes[index] = binary.charCodeAt(index);
  }
  return new Float32Array(bytes.buffer);
}

/**
 * Copies every wire and gate distribution out of the live circuits. The soft
 * circuit is enough: `createHardCircuit()` deterministically rebuilds the
 * argmax circuit during restore.
 */
export function serializeModel(
  info: ModelInfo,
  ensembles: readonly LabelEnsemble[],
): SerializedModel {
  return {
    format: MODEL_FORMAT,
    version: MODEL_VERSION,
    savedAt: Date.now(),
    shape: [...info.shape],
    gridSize: info.gridSize,
    labelCount: info.labelCount,
    totalCircuits: info.totalCircuits,
    averageLoss: info.averageLoss,
    trainedAt: info.trainedAt,
    ensembles: ensembles.map((ensemble) => ({
      labelId: ensemble.labelId,
      labelName: ensemble.labelName,
      color: ensemble.color,
      averageLoss: ensemble.averageLoss,
      positives: ensemble.positives,
      negatives: ensemble.negatives,
      trainedAt: ensemble.trainedAt,
      circuits: ensemble.circuits.map((circuit) => {
        const gates = new Float32Array(circuit.cellCount * 16);
        for (let cell = 0; cell < circuit.cellCount; cell++) {
          gates.set(circuit.getGates(cell), cell * 16);
        }
        return {
          connections: encodeFloat32(circuit.getConnections()),
          gates: encodeFloat32(gates),
        };
      }),
    })),
  };
}

function asRecord(
  value: unknown,
  context: string,
): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error(`${context} must be an object.`);
  }
  return value as Record<string, unknown>;
}

function asArray(value: unknown, context: string): readonly unknown[] {
  if (!Array.isArray(value)) {
    throw new Error(`${context} must be an array.`);
  }
  return value;
}

function asString(value: unknown, context: string): string {
  if (typeof value !== "string") {
    throw new Error(`${context} must be text.`);
  }
  return value;
}

function asFiniteNumber(value: unknown, context: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`${context} must be a number.`);
  }
  return value;
}

function asPositiveInteger(value: unknown, context: string): number {
  if (typeof value !== "number" || !Number.isInteger(value) || value <= 0) {
    throw new Error(`${context} must be a positive whole number.`);
  }
  return value;
}

function asNonNegativeInteger(value: unknown, context: string): number {
  if (typeof value !== "number" || !Number.isInteger(value) || value < 0) {
    throw new Error(`${context} must be a non-negative whole number.`);
  }
  return value;
}

function parseModelShape(value: unknown, gridSize: number): number[] {
  const layers = asArray(value, "The model shape").map((entry, index) =>
    asPositiveInteger(entry, `Layer ${index + 1}`),
  );
  const parsed = parseCircuitShape(layers.join(", "), gridSize);
  if (parsed.error !== null) {
    throw new Error(parsed.error);
  }
  return parsed.shape;
}

function parseCircuit(value: unknown, context: string): SerializedCircuit {
  const record = asRecord(value, context);
  return {
    connections: asString(record.connections, `${context} wire data`),
    gates: asString(record.gates, `${context} gate data`),
  };
}

function parseEnsemble(value: unknown, index: number): SerializedEnsemble {
  const context = `Label ${index + 1}`;
  const record = asRecord(value, context);
  const circuits = asArray(record.circuits, `${context} circuits`);
  if (circuits.length === 0) {
    throw new Error(`${context} has no circuits.`);
  }
  if (circuits.length > MAX_STORED_CIRCUITS) {
    throw new Error(
      `${context} has ${circuits.length} circuits; the limit is ${MAX_STORED_CIRCUITS}.`,
    );
  }
  return {
    labelId: asString(record.labelId, `${context} id`),
    labelName: asString(record.labelName, `${context} name`),
    color: asString(record.color, `${context} color`),
    averageLoss: asFiniteNumber(record.averageLoss, `${context} average loss`),
    positives: asNonNegativeInteger(record.positives, `${context} positive count`),
    negatives: asNonNegativeInteger(record.negatives, `${context} negative count`),
    trainedAt: asFiniteNumber(record.trainedAt, `${context} trained time`),
    circuits: circuits.map((circuit, circuitIndex) =>
      parseCircuit(circuit, `${context} circuit ${circuitIndex + 1}`),
    ),
  };
}

/** Validates untrusted data from IndexedDB or a user-selected file. */
export function parseSerializedModel(value: unknown): SerializedModel {
  const record = asRecord(value, "The model");
  if (record.format !== MODEL_FORMAT) {
    throw new Error("This is not a Gate Forest model.");
  }
  if (record.version !== MODEL_VERSION) {
    throw new Error(
      `Unsupported model version: ${String(record.version)}. Retrain the model to save it in the current format.`,
    );
  }

  const gridSize = asPositiveInteger(record.gridSize, "The grid size");
  const shape = parseModelShape(record.shape, gridSize);
  const ensembles = asArray(record.ensembles, "The model ensembles");
  if (ensembles.length === 0) {
    throw new Error("The model has no trained labels.");
  }
  if (ensembles.length > MAX_STORED_ENSEMBLES) {
    throw new Error(
      `The model has ${ensembles.length} labels; the limit is ${MAX_STORED_ENSEMBLES}.`,
    );
  }

  return {
    format: MODEL_FORMAT,
    version: MODEL_VERSION,
    savedAt: asFiniteNumber(record.savedAt, "The saved time"),
    shape,
    gridSize,
    labelCount: asNonNegativeInteger(record.labelCount, "The label count"),
    totalCircuits: asNonNegativeInteger(
      record.totalCircuits,
      "The circuit count",
    ),
    averageLoss: asFiniteNumber(record.averageLoss, "The average loss"),
    trainedAt: asFiniteNumber(record.trainedAt, "The trained time"),
    ensembles: ensembles.map((ensemble, index) => parseEnsemble(ensemble, index)),
  };
}

function destroyCircuitLists(
  circuits: readonly Circuit[],
  hardCircuits: readonly HardCircuit[],
): void {
  for (const circuit of circuits) {
    circuit.destroy();
  }
  for (const circuit of hardCircuits) {
    circuit.destroy();
  }
}

/** Rebuilds live circuits from a validated serialized model. */
export function restoreModel(
  og: OnlyGatesAI,
  stored: SerializedModel,
): RestoredModel {
  const shape = [...stored.shape];
  const ensembles: LabelEnsemble[] = [];

  try {
    for (const serialized of stored.ensembles) {
      const circuits: Circuit[] = [];
      const hardCircuits: HardCircuit[] = [];

      try {
        for (const storedCircuit of serialized.circuits) {
          const circuit = og.createCircuit(shape);
          try {
            circuit.setConnections(
              decodeFloat32(
                storedCircuit.connections,
                circuit.connectionProbabilitiesLength,
                `${serialized.labelName} wire probabilities`,
              ),
            );

            const gates = decodeFloat32(
              storedCircuit.gates,
              circuit.cellCount * 16,
              `${serialized.labelName} gate probabilities`,
            );
            for (let cell = 0; cell < circuit.cellCount; cell++) {
              circuit.setGates(
                cell,
                gates.subarray(cell * 16, cell * 16 + 16),
              );
            }

            const hardCircuit = circuit.createHardCircuit();
            circuits.push(circuit);
            hardCircuits.push(hardCircuit);
          } catch (error) {
            circuit.destroy();
            throw error;
          }
        }
      } catch (error) {
        destroyCircuitLists(circuits, hardCircuits);
        throw error;
      }

      ensembles.push({
        labelId: serialized.labelId,
        labelName: serialized.labelName,
        color: serialized.color,
        circuits,
        hardCircuits,
        averageLoss: serialized.averageLoss,
        positives: serialized.positives,
        negatives: serialized.negatives,
        trainedAt: serialized.trainedAt,
      });
    }
  } catch (error) {
    for (const ensemble of ensembles) {
      destroyCircuitLists(ensemble.circuits, ensemble.hardCircuits);
    }
    throw error;
  }

  const totalCircuits = ensembles.reduce(
    (sum, ensemble) => sum + ensemble.circuits.length,
    0,
  );
  const averageLoss =
    totalCircuits > 0
      ? ensembles.reduce(
          (sum, ensemble) =>
            sum + ensemble.averageLoss * ensemble.circuits.length,
          0,
        ) / totalCircuits
      : 0;

  return {
    ensembles,
    info: {
      shape,
      gridSize: stored.gridSize,
      labelCount: ensembles.length,
      totalCircuits,
      averageLoss,
      trainedAt: stored.trainedAt,
    },
  };
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

export async function saveModelToStorage(
  model: SerializedModel,
): Promise<void> {
  await withStore("readwrite", (store) => store.put(model, MODEL_KEY));
}

export async function loadModelFromStorage(): Promise<SerializedModel | null> {
  const value = await withStore("readonly", (store) => store.get(MODEL_KEY));
  if (value === undefined) {
    return null;
  }
  return parseSerializedModel(value);
}

export async function deleteModelFromStorage(): Promise<void> {
  await withStore("readwrite", (store) => store.delete(MODEL_KEY));
}
