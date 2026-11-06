import type { Point, Strokes } from "./types";

/** The default input resolution: a 24 × 24 boolean grid (576 inputs). */
export const DEFAULT_GRID_SIZE = 24;
export const MIN_GRID_SIZE = 8;
export const MAX_GRID_SIZE = 48;

/** Stroke width in grid cells, applied after scaling and centring. */
export const STROKE_WIDTH_CELLS = 1.35;

const COVERAGE_THRESHOLD = 0.16;
const MARGIN_CELLS = 1.5;

function supersampleFor(gridSize: number): number {
  if (gridSize <= 16) return 8;
  if (gridSize <= 24) return 6;
  if (gridSize <= 32) return 5;
  if (gridSize <= 40) return 4;
  return 3;
}

/** Returns every point in the drawing, in drawing-canvas coordinates. */
export function flattenStrokes(strokes: Strokes): Point[] {
  const points: Point[] = [];
  for (const stroke of strokes) {
    for (const point of stroke) {
      points.push(point);
    }
  }
  return points;
}

export function hasInk(strokes: Strokes): boolean {
  for (const stroke of strokes) {
    if (stroke.length > 0) {
      return true;
    }
  }
  return false;
}

export function emptyGrid(gridSize: number): boolean[] {
  return new Array<boolean>(gridSize * gridSize).fill(false);
}

/**
 * Scales the drawing so its longest side fits inside the grid (preserving
 * aspect ratio) and centres it. The stroke coordinates are in the same unit
 * as the target grid, so a constant raster width becomes a constant thickness
 * in cells regardless of how large the user drew.
 */
export function normalizeStrokes(strokes: Strokes, gridSize: number): Point[][] {
  const points = flattenStrokes(strokes);
  if (points.length === 0) {
    return [];
  }

  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;

  for (const point of points) {
    if (point.x < minX) minX = point.x;
    if (point.x > maxX) maxX = point.x;
    if (point.y < minY) minY = point.y;
    if (point.y > maxY) maxY = point.y;
  }

  const width = Math.max(1, maxX - minX);
  const height = Math.max(1, maxY - minY);
  const scale = (gridSize - MARGIN_CELLS * 2) / Math.max(width, height);
  const centerX = (minX + maxX) / 2;
  const centerY = (minY + maxY) / 2;
  const half = gridSize / 2;

  return strokes.map((stroke) =>
    stroke.map((point) => ({
      x: (point.x - centerX) * scale + half,
      y: (point.y - centerY) * scale + half,
    })),
  );
}

/**
 * Replays the stored strokes onto a boolean grid. This is the canonical input
 * generator for both training and prediction: strokes are normalised into the
 * grid, then stamped with a fixed-width round brush at a supersampled
 * resolution. A cell becomes true when enough of it is covered.
 */
export function rasterizeStrokes(
  strokes: Strokes,
  gridSize = DEFAULT_GRID_SIZE,
): boolean[] {
  const normalized = normalizeStrokes(strokes, gridSize);
  if (normalized.length === 0) {
    return emptyGrid(gridSize);
  }

  const supersample = supersampleFor(gridSize);
  const bitmapSize = gridSize * supersample;
  const bitmap = new Uint8Array(bitmapSize * bitmapSize);
  const radius = (STROKE_WIDTH_CELLS * supersample) / 2;
  const radiusSquared = radius * radius;

  const stamp = (gridX: number, gridY: number): void => {
    const centerX = gridX * supersample;
    const centerY = gridY * supersample;
    const startX = Math.max(0, Math.floor(centerX - radius));
    const endX = Math.min(bitmapSize - 1, Math.ceil(centerX + radius));
    const startY = Math.max(0, Math.floor(centerY - radius));
    const endY = Math.min(bitmapSize - 1, Math.ceil(centerY + radius));

    for (let y = startY; y <= endY; y++) {
      const dy = y - centerY;
      for (let x = startX; x <= endX; x++) {
        const dx = x - centerX;
        if (dx * dx + dy * dy <= radiusSquared) {
          bitmap[y * bitmapSize + x] = 1;
        }
      }
    }
  };

  for (const stroke of normalized) {
    if (stroke.length === 0) {
      continue;
    }
    const first = stroke[0]!;
    if (stroke.length === 1) {
      stamp(first.x, first.y);
      continue;
    }
    for (let index = 1; index < stroke.length; index++) {
      const previous = stroke[index - 1]!;
      const current = stroke[index]!;
      const dx = current.x - previous.x;
      const dy = current.y - previous.y;
      const length = Math.hypot(dx, dy);
      const steps = Math.max(
        1,
        Math.ceil(length / Math.max(0.1, STROKE_WIDTH_CELLS * 0.35)),
      );
      for (let step = 0; step <= steps; step++) {
        const t = step / steps;
        stamp(previous.x + dx * t, previous.y + dy * t);
      }
    }
  }

  const cellsPerSupersample = supersample * supersample;
  const input = new Array<boolean>(gridSize * gridSize).fill(false);
  for (let cellY = 0; cellY < gridSize; cellY++) {
    for (let cellX = 0; cellX < gridSize; cellX++) {
      let covered = 0;
      for (let y = 0; y < supersample; y++) {
        const row = (cellY * supersample + y) * bitmapSize;
        for (let x = 0; x < supersample; x++) {
          covered += bitmap[row + cellX * supersample + x]!;
        }
      }
      input[cellY * gridSize + cellX] =
        covered / cellsPerSupersample >= COVERAGE_THRESHOLD;
    }
  }
  return input;
}

export function inputHasInk(input: readonly boolean[]): boolean {
  for (const value of input) {
    if (value) {
      return true;
    }
  }
  return false;
}

export function cloneStrokes(strokes: Strokes): Strokes {
  return strokes.map((stroke) => stroke.map((point) => ({ ...point })));
}
