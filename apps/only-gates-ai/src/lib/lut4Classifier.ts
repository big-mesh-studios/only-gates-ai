import { rasterizeStrokes } from "./rasterizer";
import type { Strokes } from "./types";

/**
 * Hardened three- or four-input LUT circuit exported from
 * `experiments/gate-cnn-mnist` (`--export-hard`).
 */
export interface LutLayer {
  /** Four wire indices per cell, flattened. */
  readonly wires: readonly number[];
  /** One 16-bit truth table per cell (only the low `2^pins` bits are used). */
  readonly tables: readonly number[];
}

export interface LutModel {
  readonly kind: "lut3" | "lut4";
  readonly shape: readonly number[];
  readonly inputSize: number;
  readonly circuitsPerLabel: number;
  readonly layers: readonly LutLayer[];
}

export interface DigitPrediction {
  readonly label: number;
  readonly scores: readonly number[];
  readonly totalCircuits: number;
}

export const DIGIT_GRID = 14;
export const DIGIT_LABELS = 10;

/** Load the bundled hard circuit. */
export async function loadLutModel(url: string): Promise<LutModel> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Could not load digit circuit (${response.status}).`);
  }
  return (await response.json()) as LutModel;
}

/**
 * Runs the deterministic LUT network. Cells read four (or three) bits from the
 * previous layer and look their truth table up by the packed input index.
 */
export function predictDigit(
  model: LutModel,
  input: readonly boolean[],
): DigitPrediction {
  if (input.length !== model.inputSize) {
    throw new Error(
      `Digit input has ${input.length} cells, circuit expects ${model.inputSize}.`,
    );
  }

  let values: Uint8Array = Uint8Array.from(input, (value) => (value ? 1 : 0));
  for (const layer of model.layers) {
    const count = layer.tables.length;
    const next = new Uint8Array(count);
    const pins = layer.wires.length / count;
    for (let cell = 0; cell < count; cell++) {
      const base = cell * pins;
      let index = 0;
      for (let pin = 0; pin < pins; pin++) {
        index = (index << 1) | values[layer.wires[base + pin]!]!;
      }
      next[cell] = (layer.tables[cell]! >>> index) & 1;
    }
    values = next;
  }

  const circuits = model.circuitsPerLabel;
  const scores = new Array<number>(DIGIT_LABELS).fill(0);
  for (let index = 0; index < values.length; index++) {
    const label = Math.floor(index / circuits);
    if (label < DIGIT_LABELS) {
      scores[label] = scores[label]! + values[index]!;
    }
  }

  let label = 0;
  for (let candidate = 1; candidate < DIGIT_LABELS; candidate++) {
    if (scores[candidate]! > scores[label]!) {
      label = candidate;
    }
  }

  return { label, scores, totalCircuits: circuits };
}

/**
 * Turns strokes into the same 14×14 binary grid the circuit was trained on:
 * rasterise at 24×24, centre it in a 28×28 frame, then 2×2 any-pool. The
 * training pipeline thresholded 2×2 average pooling at 0.25, which is exactly
 * "at least one of the four cells is set".
 */
export function digitInputFromStrokes(strokes: Strokes): boolean[] {
  const grid = rasterizeStrokes(strokes, 24);
  const frame = new Uint8Array(28 * 28);
  for (let y = 0; y < 24; y++) {
    for (let x = 0; x < 24; x++) {
      if (grid[y * 24 + x]) {
        frame[(y + 2) * 28 + (x + 2)] = 1;
      }
    }
  }

  const output: boolean[] = new Array<boolean>(DIGIT_GRID * DIGIT_GRID);
  for (let y = 0; y < DIGIT_GRID; y++) {
    for (let x = 0; x < DIGIT_GRID; x++) {
      const sy = y * 2;
      const sx = x * 2;
      output[y * DIGIT_GRID + x] = Boolean(
        frame[sy * 28 + sx] ||
          frame[sy * 28 + sx + 1] ||
          frame[(sy + 1) * 28 + sx] ||
          frame[(sy + 1) * 28 + sx + 1],
      );
    }
  }
  return output;
}
