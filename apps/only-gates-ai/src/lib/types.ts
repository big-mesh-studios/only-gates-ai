import type { Circuit, HardCircuit } from "@only-gates-ai/core";

export interface Point {
  readonly x: number;
  readonly y: number;
}

export type Stroke = readonly Point[];
export type Strokes = readonly Stroke[];

export interface SymbolLabel {
  readonly id: string;
  readonly name: string;
  readonly color: string;
}

export interface DrawingSample {
  readonly id: string;
  readonly labelId: string;
  readonly strokes: Strokes;
  readonly input: readonly boolean[];
  readonly gridSize: number;
  readonly createdAt: number;
}

export interface LabelEnsemble {
  readonly labelId: string;
  readonly labelName: string;
  readonly color: string;
  readonly circuits: readonly Circuit[];
  readonly hardCircuits: readonly HardCircuit[];
  readonly averageLoss: number;
  readonly positives: number;
  readonly negatives: number;
  readonly trainedAt: number;
}

export interface PredictionResult {
  readonly labelId: string;
  readonly labelName: string;
  readonly color: string;
  readonly votes: number;
  readonly totalCircuits: number;
  readonly score: number;
}

export type InferenceMode = "soft" | "hard";

export interface ModelInfo {
  readonly shape: readonly number[];
  readonly gridSize: number;
  readonly labelCount: number;
  readonly totalCircuits: number;
  readonly averageLoss: number;
  readonly trainedAt: number;
}
