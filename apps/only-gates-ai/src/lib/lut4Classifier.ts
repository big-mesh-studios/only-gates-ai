import { flattenStrokes } from "./rasterizer";
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

const FRAME = 28;
const FRAME_BOX = 20;
const SUPERSAMPLE = 4;
const BRUSH = 1.15;
const COVERAGE = 0.25;

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
 * Turns strokes into the 14×14 binary grid the circuit was trained on,
 * following MNIST's preprocessing: scale the bounding box to 20×20, centre by
 * centre of mass in a 28×28 frame, then 2×2 average-pool and threshold.
 *
 * The previous version centred by bounding box and used a thinner brush, which
 * shifted digits like 7 and 9 relative to the training distribution.
 */
export function digitInputFromStrokes(strokes: Strokes): boolean[] {
  const output: boolean[] = new Array<boolean>(DIGIT_GRID * DIGIT_GRID).fill(
    false,
  );
  const points = flattenStrokes(strokes);
  if (points.length === 0) {
    return output;
  }

  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  let sumX = 0;
  let sumY = 0;
  for (const point of points) {
    if (point.x < minX) minX = point.x;
    if (point.x > maxX) maxX = point.x;
    if (point.y < minY) minY = point.y;
    if (point.y > maxY) maxY = point.y;
    sumX += point.x;
    sumY += point.y;
  }

  const scale = FRAME_BOX / Math.max(1, maxX - minX, maxY - minY);
  const centreX = sumX / points.length;
  const centreY = sumY / points.length;
  const half = FRAME / 2;

  // Map drawing coordinates into the 28×28 frame, centred on the ink centroid.
  const map = (x: number, y: number): [number, number] => [
    (x - centreX) * scale + half,
    (y - centreY) * scale + half,
  ];

  const size = FRAME * SUPERSAMPLE;
  const bitmap = new Uint8Array(size * size);
  const radius = BRUSH * SUPERSAMPLE;
  const radiusSquared = radius * radius;

  const stamp = (frameX: number, frameY: number): void => {
    const centreXpx = frameX * SUPERSAMPLE;
    const centreYpx = frameY * SUPERSAMPLE;
    const startX = Math.max(0, Math.floor(centreXpx - radius));
    const endX = Math.min(size - 1, Math.ceil(centreXpx + radius));
    const startY = Math.max(0, Math.floor(centreYpx - radius));
    const endY = Math.min(size - 1, Math.ceil(centreYpx + radius));
    for (let y = startY; y <= endY; y++) {
      const dy = y - centreYpx;
      for (let x = startX; x <= endX; x++) {
        const dx = x - centreXpx;
        if (dx * dx + dy * dy <= radiusSquared) {
          bitmap[y * size + x] = 1;
        }
      }
    }
  };

  for (const stroke of strokes) {
    for (let index = 0; index < stroke.length; index++) {
      const point = stroke[index]!;
      const [fx, fy] = map(point.x, point.y);
      const next = stroke[index + 1];
      if (!next) {
        stamp(fx, fy);
        continue;
      }
      const [nx, ny] = map(next.x, next.y);
      const steps = Math.max(1, Math.ceil(Math.hypot(nx - fx, ny - fy)));
      for (let step = 0; step < steps; step++) {
        const t = step / steps;
        stamp(fx + (nx - fx) * t, fy + (ny - fy) * t);
      }
    }
  }

  // 28×28 coverage (0..1) from the supersampled bitmap, thresholded like the
  // training pipeline's average-pool-of-4 at 0.25, then 2×2 any-pooled.
  const frame = new Uint8Array(FRAME * FRAME);
  const samples = SUPERSAMPLE * SUPERSAMPLE;
  for (let y = 0; y < FRAME; y++) {
    for (let x = 0; x < FRAME; x++) {
      let covered = 0;
      for (let dy = 0; dy < SUPERSAMPLE; dy++) {
        for (let dx = 0; dx < SUPERSAMPLE; dx++) {
          covered +=
            bitmap[(y * SUPERSAMPLE + dy) * size + (x * SUPERSAMPLE + dx)]!;
        }
      }
      frame[y * FRAME + x] = covered / samples >= COVERAGE ? 1 : 0;
    }
  }

  for (let y = 0; y < DIGIT_GRID; y++) {
    for (let x = 0; x < DIGIT_GRID; x++) {
      const sy = y * 2;
      const sx = x * 2;
      output[y * DIGIT_GRID + x] = Boolean(
        frame[sy * FRAME + sx] ||
          frame[sy * FRAME + sx + 1] ||
          frame[(sy + 1) * FRAME + sx] ||
          frame[(sy + 1) * FRAME + sx + 1],
      );
    }
  }
  return output;
}
