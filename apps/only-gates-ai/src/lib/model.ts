import type { Circuit, HardCircuit, OnlyGatesAI } from "@only-gates-ai/core";

import type {
  DrawingSample,
  InferenceMode,
  LabelEnsemble,
  PredictionResult,
  SymbolLabel,
} from "./types";

export const CIRCUITS_PER_LABEL = 8;
export const DEFAULT_HIDDEN_LAYERS = [32] as const;
export const TRAINING_EPOCHS = 40;
export const LEARNING_RATE = 0.1;
export const MAX_SAMPLES_PER_CLASS = 12;
export const MAX_PARAMETERS = 2_000_000;

const MAX_LAYER_SIZE = 2_048;

export type ParsedShape =
  | { shape: number[]; error: null; parameterCount: number }
  | { shape: null; error: string; parameterCount: 0 };

export function defaultShape(gridSize: number): number[] {
  return [gridSize * gridSize, ...DEFAULT_HIDDEN_LAYERS, 1];
}

export function formatShape(shape: readonly number[]): string {
  return shape.join(", ");
}

export function shapeParameterCount(shape: readonly number[]): number {
  let count = 0;
  for (let layer = 1; layer < shape.length; layer++) {
    const previous = shape[layer - 1]!;
    const current = shape[layer]!;
    // Every cell has two input pins, each with a distribution over the
    // previous layer.
    count += previous * current * 2;
    // Hidden cells also have a 16-way gate distribution. The output layer has
    // no gate.
    if (layer < shape.length - 1) {
      count += current * 16;
    }
  }
  return count;
}

/**
 * Parses a full circuit shape such as `576, 32, 1`. The first layer must match
 * the current grid, and the output layer is always a single boolean.
 */
export function parseShape(text: string, gridSize: number): ParsedShape {
  const inputSize = gridSize * gridSize;
  const tokens = text
    .split(",")
    .map((token) => token.trim())
    .filter((token) => token.length > 0);

  if (tokens.length < 2) {
    return {
      shape: null,
      error: "Enter at least an input layer and an output layer, e.g. 576, 32, 1.",
      parameterCount: 0,
    };
  }

  const layers: number[] = [];
  for (const token of tokens) {
    if (!/^\d+$/.test(token)) {
      return {
        shape: null,
        error: `"${token}" is not a positive whole number.`,
        parameterCount: 0,
      };
    }
    const value = Number(token);
    if (!Number.isSafeInteger(value) || value <= 0) {
      return {
        shape: null,
        error: "Layer sizes must be positive whole numbers.",
        parameterCount: 0,
      };
    }
    if (value > MAX_LAYER_SIZE) {
      return {
        shape: null,
        error: `Layers are limited to ${MAX_LAYER_SIZE.toLocaleString()} neurons.`,
        parameterCount: 0,
      };
    }
    layers.push(value);
  }

  if (layers[0] !== inputSize) {
    return {
      shape: null,
      error: `The first layer must be ${inputSize} for a ${gridSize} × ${gridSize} grid.`,
      parameterCount: 0,
    };
  }

  if (layers[layers.length - 1] !== 1) {
    return {
      shape: null,
      error: "The last layer must be 1, because each circuit answers one yes/no question.",
      parameterCount: 0,
    };
  }

  if (layers.length === 2) {
    return {
      shape: null,
      error: "Add at least one hidden layer between the input and the output.",
      parameterCount: 0,
    };
  }

  const parameterCount = shapeParameterCount(layers);
  if (parameterCount > MAX_PARAMETERS) {
    return {
      shape: null,
      error: `This shape needs ${parameterCount.toLocaleString()} parameters. The limit is ${MAX_PARAMETERS.toLocaleString()} per circuit.`,
      parameterCount: 0,
    };
  }

  return { shape: layers, error: null, parameterCount };
}

export class TrainingCancelledError extends Error {
  constructor() {
    super("Training was cancelled.");
    this.name = "TrainingCancelledError";
  }
}

export interface TrainingProgress {
  readonly completed: number;
  readonly total: number;
  readonly labelName: string;
}

export interface SkippedLabel {
  readonly label: SymbolLabel;
  readonly reason: string;
}

export interface TrainEnsemblesOptions {
  readonly og: OnlyGatesAI;
  readonly labels: readonly SymbolLabel[];
  readonly samples: readonly DrawingSample[];
  readonly shape: readonly number[];
  readonly circuitsPerLabel?: number;
  readonly epochs?: number;
  readonly learningRate?: number;
  readonly maxSamplesPerClass?: number;
  readonly isCancelled?: () => boolean;
  readonly onProgress?: (progress: TrainingProgress) => void;
}

export interface TrainEnsemblesResult {
  readonly ensembles: LabelEnsemble[];
  readonly skipped: SkippedLabel[];
  readonly totalCircuits: number;
  readonly averageLoss: number;
}

function pickRandom<T>(items: readonly T[]): T {
  return items[Math.floor(Math.random() * items.length)]!;
}

function shuffle<T>(items: T[]): void {
  for (let index = items.length - 1; index > 0; index--) {
    const swapIndex = Math.floor(Math.random() * (index + 1));
    const value = items[index]!;
    items[index] = items[swapIndex]!;
    items[swapIndex] = value;
  }
}

interface BootstrapDataset {
  readonly inputs: (readonly boolean[])[];
  readonly outputs: (readonly number[])[];
}

/**
 * One balanced bootstrap sample: positives and negatives are drawn with
 * replacement, which gives each circuit in the forest a slightly different
 * view of the data.
 */
function bootstrapDataset(
  positives: readonly DrawingSample[],
  negatives: readonly DrawingSample[],
  perClass: number,
): BootstrapDataset {
  const inputs: (readonly boolean[])[] = [];
  const outputs: (readonly number[])[] = [];

  for (let index = 0; index < perClass; index++) {
    inputs.push(pickRandom(positives).input);
    outputs.push([1]);
  }
  for (let index = 0; index < perClass; index++) {
    inputs.push(pickRandom(negatives).input);
    outputs.push([0]);
  }

  for (let index = inputs.length - 1; index > 0; index--) {
    const swapIndex = Math.floor(Math.random() * (index + 1));
    const input = inputs[index]!;
    inputs[index] = inputs[swapIndex]!;
    inputs[swapIndex] = input;
    const output = outputs[index]!;
    outputs[index] = outputs[swapIndex]!;
    outputs[swapIndex] = output;
  }

  return { inputs, outputs };
}

function yieldToBrowser(): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, 0);
  });
}

function checkCancelled(isCancelled: (() => boolean) | undefined): void {
  if (isCancelled?.()) {
    throw new TrainingCancelledError();
  }
}

export async function trainEnsembles(
  options: TrainEnsemblesOptions,
): Promise<TrainEnsemblesResult> {
  const {
    og,
    labels,
    samples,
    shape,
    circuitsPerLabel = CIRCUITS_PER_LABEL,
    epochs = TRAINING_EPOCHS,
    learningRate = LEARNING_RATE,
    maxSamplesPerClass = MAX_SAMPLES_PER_CLASS,
    isCancelled,
    onProgress,
  } = options;

  const expectedInputs = shape[0]!;
  if (!Number.isInteger(expectedInputs) || expectedInputs <= 0) {
    throw new Error("The circuit shape needs a valid input layer.");
  }
  for (const sample of samples) {
    if (sample.input.length !== expectedInputs) {
      throw new Error(
        `A drawing has ${sample.input.length} inputs, but the circuit expects ${expectedInputs}.`,
      );
    }
  }

  const samplesByLabel = new Map<string, DrawingSample[]>();
  for (const sample of samples) {
    const group = samplesByLabel.get(sample.labelId);
    if (group) {
      group.push(sample);
    } else {
      samplesByLabel.set(sample.labelId, [sample]);
    }
  }

  const skipped: SkippedLabel[] = [];
  const trainable: Array<{
    label: SymbolLabel;
    positives: DrawingSample[];
    negatives: DrawingSample[];
  }> = [];

  for (const label of labels) {
    const positives = samplesByLabel.get(label.id) ?? [];
    const negatives = samples.filter((sample) => sample.labelId !== label.id);
    if (positives.length === 0) {
      skipped.push({ label, reason: "no drawings yet" });
      continue;
    }
    if (negatives.length === 0) {
      skipped.push({ label, reason: "needs drawings from another label" });
      continue;
    }
    trainable.push({ label, positives, negatives });
  }

  const total = trainable.length * circuitsPerLabel;
  if (total === 0) {
    return { ensembles: [], skipped, totalCircuits: 0, averageLoss: 0 };
  }

  const ensembles: LabelEnsemble[] = [];
  const createdCircuits: Circuit[] = [];
  const createdHardCircuits: HardCircuit[] = [];
  let completed = 0;
  let lossAcrossLabels = 0;

  try {
    for (const { label, positives, negatives } of trainable) {
      const perClass = Math.max(
        1,
        Math.min(maxSamplesPerClass, positives.length, negatives.length),
      );
      const circuits: Circuit[] = [];
      const hardCircuits: HardCircuit[] = [];
      let lossAcrossCircuits = 0;

      for (let circuitIndex = 0; circuitIndex < circuitsPerLabel; circuitIndex++) {
        checkCancelled(isCancelled);

        const dataset = bootstrapDataset(positives, negatives, perClass);
        const circuit = og.createCircuit(shape);
        let hardCircuit: HardCircuit | null = null;

        try {
          circuit.randomize();
          const trainer = circuit.createTrainer(dataset, learningRate);
          let lossTotal = 0;
          try {
            let remaining = epochs;
            while (remaining > 0) {
              checkCancelled(isCancelled);
              const chunk = Math.min(10, remaining);
              lossTotal += trainer.trainEpoch(chunk) * chunk;
              remaining -= chunk;
              if (remaining > 0) {
                await yieldToBrowser();
              }
            }
          } finally {
            trainer.destroy();
          }

          checkCancelled(isCancelled);
          hardCircuit = circuit.createHardCircuit();
          lossAcrossCircuits += lossTotal / epochs;
        } catch (error) {
          circuit.destroy();
          throw error;
        }

        circuits.push(circuit);
        hardCircuits.push(hardCircuit);
        createdCircuits.push(circuit);
        createdHardCircuits.push(hardCircuit);

        completed++;
        onProgress?.({
          completed,
          total,
          labelName: label.name,
        });
        await yieldToBrowser();
      }

      const averageLoss = lossAcrossCircuits / circuitsPerLabel;
      lossAcrossLabels += averageLoss;
      ensembles.push({
        labelId: label.id,
        labelName: label.name,
        color: label.color,
        circuits,
        hardCircuits,
        averageLoss,
        positives: positives.length,
        negatives: negatives.length,
        trainedAt: Date.now(),
      });
    }
  } catch (error) {
    for (const circuit of createdCircuits) {
      circuit.destroy();
    }
    for (const hardCircuit of createdHardCircuits) {
      hardCircuit.destroy();
    }
    throw error;
  }

  return {
    ensembles,
    skipped,
    totalCircuits: completed,
    averageLoss: lossAcrossLabels / ensembles.length,
  };
}

export function destroyEnsembles(ensembles: readonly LabelEnsemble[]): void {
  for (const ensemble of ensembles) {
    for (const circuit of ensemble.circuits) {
      circuit.destroy();
    }
    for (const hardCircuit of ensemble.hardCircuits) {
      hardCircuit.destroy();
    }
  }
}

/**
 * Runs every circuit in the forest and turns its answer into one vote. In soft
 * mode the value is the probability-weighted output. In hard mode the circuit
 * locks every wire and gate to its highest-probability choice first.
 */
export function predictEnsembles(
  ensembles: readonly LabelEnsemble[],
  input: readonly boolean[],
  mode: InferenceMode,
): PredictionResult[] {
  const results: PredictionResult[] = [];

  for (const ensemble of ensembles) {
    let votes = 0;
    let score = 0;
    let evaluated = 0;

    if (mode === "hard") {
      for (const circuit of ensemble.hardCircuits) {
        if (circuit.inputsLength !== input.length) {
          continue;
        }
        circuit.setInput(input);
        circuit.feedForward();
        const output = circuit.getOutput()[0] ?? 0;
        score += output;
        if (output >= 0.5) {
          votes++;
        }
        evaluated++;
      }
    } else {
      for (const circuit of ensemble.circuits) {
        if (circuit.inputsLength !== input.length) {
          continue;
        }
        circuit.setInput(input);
        circuit.feedForward();
        const output = circuit.getOutput()[0] ?? 0;
        score += output;
        if (output >= 0.5) {
          votes++;
        }
        evaluated++;
      }
    }

    if (evaluated === 0) {
      continue;
    }

    results.push({
      labelId: ensemble.labelId,
      labelName: ensemble.labelName,
      color: ensemble.color,
      votes,
      totalCircuits: evaluated,
      score: score / evaluated,
    });
  }

  results.sort(
    (a, b) =>
      b.votes - a.votes ||
      b.score - a.score ||
      a.labelName.localeCompare(b.labelName),
  );
  return results;
}
