import {
  createEffect,
  createMemo,
  createSignal,
  onSettled,
  Show,
  untrack,
} from "solid-js";
import { createOnlyGatesAI, type OnlyGatesAI } from "@only-gates-ai/core";
import type { ForestLabelSource } from "@only-gates-ai/compiler";

import CompileDialog from "./components/CompileDialog";
import DrawingPad from "./components/DrawingPad";
import IntegerGptPanel from "./components/IntegerGptPanel";
import LanguageModelPanel from "./components/LanguageModelPanel";
import LivePrediction from "./components/LivePrediction";
import LuauPanel from "./components/LuauPanel";
import ModelStoragePanel from "./components/ModelStoragePanel";
import SettingsPanel from "./components/SettingsPanel";
import TestingPanel from "./components/TestingPanel";
import TrainingPanel from "./components/TrainingPanel";
import {
  CIRCUITS_PER_LABEL,
  defaultShape,
  destroyEnsembles,
  formatShape,
  parseShape,
  predictEnsembles,
  trainEnsembles,
  TrainingCancelledError,
  type TrainingProgress,
} from "./lib/model";
import {
  deleteModelFromStorage,
  loadModelFromStorage,
  parseSerializedModel,
  restoreModel,
  saveModelToStorage,
  serializeModel,
  type SerializedModel,
} from "./lib/modelStorage";
import {
  cloneStrokes,
  DEFAULT_GRID_SIZE,
  inputHasInk,
  MAX_GRID_SIZE,
  MIN_GRID_SIZE,
  rasterizeStrokes,
} from "./lib/rasterizer";
import type {
  DrawingSample,
  InferenceMode,
  LabelEnsemble,
  ModelInfo,
  PredictionResult,
  Point,
  Strokes,
  SymbolLabel,
} from "./lib/types";

const MAX_LABELS = 24;
const PREDICTION_DEBOUNCE_MS = 140;
const LABEL_COLORS = [
  "#7ee787",
  "#79c0ff",
  "#ffa657",
  "#d2a8ff",
  "#ff7b72",
  "#56d4dd",
  "#f2cc60",
  "#ff9bce",
  "#a5d6ff",
  "#c3e88d",
  "#ffd580",
  "#b392f0",
];

interface StatusMessage {
  readonly kind: "info" | "success" | "error";
  readonly text: string;
}

function createId(): string {
  const uuid = globalThis.crypto?.randomUUID?.();
  if (uuid) {
    return uuid;
  }
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

function errorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }
  return String(error);
}

export default function App() {
  const [runtime, setRuntime] = createSignal<"loading" | "ready" | "error">(
    "loading",
  );
  const [runtimeError, setRuntimeError] = createSignal("");
  const [instance, setInstance] = createSignal<OnlyGatesAI | null>(null);

  const [activeTab, setActiveTab] = createSignal<"symbols" | "language" | "integer" | "luau">("symbols");
  const [mode, setMode] = createSignal<"train" | "test">("train");
  const [inferenceMode, setInferenceMode] = createSignal<InferenceMode>("hard");

  const [labels, setLabels] = createSignal<SymbolLabel[]>([]);
  const [samples, setSamples] = createSignal<DrawingSample[]>([]);
  const [strokes, setStrokes] = createSignal<Strokes>([]);
  const [selectedLabelId, setSelectedLabelId] = createSignal<string | null>(
    null,
  );

  const [gridSize, setGridSize] = createSignal(DEFAULT_GRID_SIZE);
  const [gridDraft, setGridDraft] = createSignal(String(DEFAULT_GRID_SIZE));
  const [shapeText, setShapeText] = createSignal(
    formatShape(defaultShape(DEFAULT_GRID_SIZE)),
  );

  const [ensembles, setEnsembles] = createSignal<LabelEnsemble[]>([]);
  const [modelInfo, setModelInfo] = createSignal<ModelInfo | null>(null);
  const [modelStale, setModelStale] = createSignal(false);
  const [predictions, setPredictions] = createSignal<PredictionResult[]>([]);

  const [busy, setBusy] = createSignal(false);
  const [progress, setProgress] = createSignal<TrainingProgress | null>(null);
  const [status, setStatus] = createSignal<StatusMessage | null>(null);

  const [storageBusy, setStorageBusy] = createSignal(false);
  const [storageAvailable, setStorageAvailable] = createSignal(true);
  const [storageError, setStorageError] = createSignal<string | null>(null);
  const [storedModelAt, setStoredModelAt] = createSignal<number | null>(null);

  const [compileOpen, setCompileOpen] = createSignal(false);

  let disposed = false;
  let loadingRuntime = false;
  let restoreStarted = false;
  let statusTimer: ReturnType<typeof setTimeout> | undefined;
  let trainingToken: { cancelled: boolean } | null = null;

  const inputSize = createMemo(() => gridSize() * gridSize());
  const currentInput = createMemo(() =>
    rasterizeStrokes(strokes(), gridSize()),
  );
  const hasDrawing = createMemo(() => inputHasInk(currentInput()));
  const parsedShape = createMemo(() => parseShape(shapeText(), gridSize()));
  const selectedLabel = createMemo(
    () => labels().find((label) => label.id === selectedLabelId()) ?? null,
  );
  const compileLabels = createMemo<ForestLabelSource[]>(() =>
    ensembles().map((ensemble) => ({
      id: ensemble.labelId,
      name: ensemble.labelName,
      color: ensemble.color,
      circuits: ensemble.hardCircuits,
    })),
  );

  const trainableLabels = createMemo(() => {
    const allSamples = samples();
    return labels().filter((label) => {
      let hasPositive = false;
      let hasNegative = false;
      for (const sample of allSamples) {
        if (sample.labelId === label.id) {
          hasPositive = true;
        } else {
          hasNegative = true;
        }
        if (hasPositive && hasNegative) {
          return true;
        }
      }
      return false;
    });
  });

  const canTrain = createMemo(
    () =>
      runtime() === "ready" &&
      !busy() &&
      !storageBusy() &&
      parsedShape().error === null &&
      trainableLabels().length > 0,
  );

  const trainHint = createMemo(() => {
    if (runtime() !== "ready") {
      return "Waiting for the WebAssembly runtime.";
    }
    if (parsedShape().error) {
      return "Fix the circuit shape before training.";
    }
    if (labels().length < 2) {
      return "Add at least two labels, each with drawings, so every circuit has positive and negative examples.";
    }
    if (trainableLabels().length === 0) {
      return "Each trainable label needs at least one drawing, plus drawings from other labels to use as negatives.";
    }
    return `Ready to train ${trainableLabels().length} × ${CIRCUITS_PER_LABEL} circuits with balanced bootstrap samples.`;
  });

  const predictHint = createMemo(() => {
    if (runtime() !== "ready") {
      return "Waiting for the WebAssembly runtime.";
    }
    if (!modelInfo()) {
      return "Train the forest first, then come back to test it.";
    }
    if (!hasDrawing()) {
      return "Draw a symbol in the canvas — predictions appear automatically.";
    }
    if (modelStale()) {
      return "Using the last trained model. Retrain to include your latest drawings.";
    }
    return "Predictions update automatically as you draw.";
  });

  // In testing mode, recompute the forest's vote shortly after the drawing
  // settles. Debouncing keeps continuous strokes from re-running every circuit
  // on every pointer move.
  createEffect(
    () => ({
      mode: mode(),
      runtime: runtime(),
      busy: busy(),
      ensembles: ensembles(),
      inferenceMode: inferenceMode(),
      input: currentInput(),
    }),
    (state) => {
      if (state.mode !== "test" || state.runtime !== "ready" || state.busy) {
        return;
      }
      if (state.ensembles.length === 0 || !inputHasInk(state.input)) {
        clearPredictions();
        return;
      }

      const timer = setTimeout(() => {
        try {
          setPredictions(
            predictEnsembles(
              state.ensembles,
              state.input,
              state.inferenceMode,
            ),
          );
        } catch (error) {
          clearPredictions();
          showStatus("error", `Prediction failed: ${errorMessage(error)}`);
        }
      }, PREDICTION_DEBOUNCE_MS);

      return () => clearTimeout(timer);
    },
  );

  function showStatus(kind: StatusMessage["kind"], text: string): void {
    setStatus({ kind, text });
    if (statusTimer) {
      clearTimeout(statusTimer);
    }
    statusTimer = setTimeout(
      () => setStatus(null),
      kind === "error" ? 9000 : 5000,
    );
  }

  async function loadRuntime(): Promise<void> {
    if (loadingRuntime) {
      return;
    }
    loadingRuntime = true;
    setRuntime("loading");
    setRuntimeError("");
    try {
      const loaded = await createOnlyGatesAI();
      if (disposed) {
        return;
      }
      setInstance(loaded);
      setRuntime("ready");
      void restoreSavedModel(loaded);
    } catch (error) {
      if (disposed) {
        return;
      }
      setRuntimeError(errorMessage(error));
      setRuntime("error");
    } finally {
      loadingRuntime = false;
    }
  }

  onSettled(() => {
    // Defer the first signal write out of the component's owned scope.
    const startTimer = setTimeout(() => {
      void loadRuntime();
    }, 0);

    return () => {
      disposed = true;
      clearTimeout(startTimer);
      if (statusTimer) {
        clearTimeout(statusTimer);
      }
      if (trainingToken) {
        trainingToken.cancelled = true;
      }
      untrack(() => destroyEnsembles(ensembles()));
    };
  });

  function clearPredictions(): void {
    if (predictions().length > 0) {
      setPredictions([]);
    }
  }

  function markModelStale(): void {
    if (ensembles().length > 0) {
      setModelStale(true);
    }
    clearPredictions();
  }

  function installModel(
    restored: { ensembles: LabelEnsemble[]; info: ModelInfo },
    loadedGridSize: number,
    loadedShape: readonly number[],
  ): void {
    if (loadedGridSize !== gridSize()) {
      setSamples((previous) =>
        previous.map((sample) => ({
          ...sample,
          gridSize: loadedGridSize,
          input: rasterizeStrokes(sample.strokes, loadedGridSize),
        })),
      );
      setGridSize(loadedGridSize);
      setGridDraft(String(loadedGridSize));
    }

    setShapeText(formatShape(loadedShape));
    destroyEnsembles(ensembles());
    setEnsembles(restored.ensembles);
    setModelInfo(restored.info);
    setModelStale(false);
    setPredictions([]);
  }

  async function restoreSavedModel(og: OnlyGatesAI): Promise<void> {
    if (restoreStarted) {
      return;
    }
    restoreStarted = true;
    setStorageBusy(true);

    try {
      const stored = await loadModelFromStorage();
      if (disposed) {
        return;
      }
      setStorageAvailable(true);
      setStorageError(null);
      if (!stored) {
        return;
      }

      const restored = restoreModel(og, stored);
      if (disposed) {
        destroyEnsembles(restored.ensembles);
        return;
      }

      installModel(restored, stored.gridSize, stored.shape);
      setStoredModelAt(stored.savedAt);
      setLabels(
        restored.ensembles.map((e) => ({
          id: e.labelId,
          name: e.labelName,
          color: e.color,
        })),
      );
      showStatus(
        "info",
        `Restored the model saved in this browser (${restored.ensembles.length} label${
          restored.ensembles.length === 1 ? "" : "s"
        }).`,
      );
    } catch (error) {
      if (disposed) {
        return;
      }
      const message = errorMessage(error);
      setStorageAvailable(false);
      setStorageError(message);
    } finally {
      if (!disposed) {
        setStorageBusy(false);
      }
    }
  }

  async function persistModel(
    info: ModelInfo,
    currentEnsembles: readonly LabelEnsemble[],
  ): Promise<string | null> {
    setStorageBusy(true);
    try {
      const stored = serializeModel(info, currentEnsembles);
      await saveModelToStorage(stored);
      setStorageAvailable(true);
      setStorageError(null);
      setStoredModelAt(stored.savedAt);
      return null;
    } catch (error) {
      const message = errorMessage(error);
      setStorageAvailable(false);
      setStorageError(message);
      return message;
    } finally {
      if (!disposed) {
        setStorageBusy(false);
      }
    }
  }

  function handleSaveModelFile(): void {
    const info = modelInfo();
    if (!info || ensembles().length === 0) {
      return;
    }

    try {
      const stored = serializeModel(info, ensembles());
      const blob = new Blob([JSON.stringify(stored)], {
        type: "application/json",
      });
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      const stamp = new Date().toISOString().replace(/[:.]/g, "-");
      anchor.href = url;
      anchor.download = `gate-forest-model-${stamp}.json`;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      setTimeout(() => URL.revokeObjectURL(url), 0);
      showStatus("success", "Model saved to a file.");
    } catch (error) {
      showStatus(
        "error",
        `Could not save the model file: ${errorMessage(error)}`,
      );
    }
  }

  async function handleLoadModelFile(file: File): Promise<void> {
    const og = instance();
    if (!og || runtime() !== "ready") {
      showStatus("error", "Wait for the runtime before loading a model.");
      return;
    }
    if (busy() || storageBusy()) {
      return;
    }

    setStorageBusy(true);
    try {
      let parsed: unknown;
      try {
        parsed = JSON.parse(await file.text());
      } catch {
        throw new Error("The selected file is not valid JSON.");
      }

      const stored = parseSerializedModel(parsed);
      const restored = restoreModel(og, stored);
      if (disposed) {
        destroyEnsembles(restored.ensembles);
        return;
      }

      installModel(restored, stored.gridSize, stored.shape);
      setLabels(
        restored.ensembles.map((e) => ({
          id: e.labelId,
          name: e.labelName,
          color: e.color,
        })),
      );
      setStoredModelAt(stored.savedAt);
      let saveError: string | null = null;
      try {
        const persisted: SerializedModel = { ...stored, savedAt: Date.now() };
        await saveModelToStorage(persisted);
        setStorageAvailable(true);
        setStorageError(null);
        setStoredModelAt(persisted.savedAt);
      } catch (error) {
        saveError = errorMessage(error);
        setStorageAvailable(false);
        setStorageError(saveError);
      }

      const summary = `Loaded ${restored.ensembles.length} label${
        restored.ensembles.length === 1 ? "" : "s"
      } from the file.`;
      showStatus(
        saveError ? "error" : "success",
        saveError
          ? `${summary} Could not save it in this browser: ${saveError}`
          : `${summary} Saved it in this browser.`,
      );
    } catch (error) {
      showStatus(
        "error",
        `Could not load the model: ${errorMessage(error)}`,
      );
    } finally {
      if (!disposed) {
        setStorageBusy(false);
      }
    }
  }

  async function handleClearModel(): Promise<void> {
    if (busy() || storageBusy()) {
      return;
    }
    if (
      !window.confirm(
        "Clear the trained model from memory and this browser? Your labels and drawings will be kept.",
      )
    ) {
      return;
    }

    setStorageBusy(true);
    let storageErrorText: string | null = null;
    try {
      await deleteModelFromStorage();
      setStorageAvailable(true);
      setStorageError(null);
      setStoredModelAt(null);
    } catch (error) {
      storageErrorText = errorMessage(error);
      setStorageAvailable(false);
      setStorageError(storageErrorText);
    } finally {
      if (!disposed) {
        setStorageBusy(false);
      }
    }

    destroyEnsembles(ensembles());
    setEnsembles([]);
    setModelInfo(null);
    setModelStale(false);
    setPredictions([]);

    if (storageErrorText) {
      showStatus(
        "error",
        `Model cleared from memory, but browser storage could not be cleared: ${storageErrorText}`,
      );
    } else {
      showStatus("success", "Model cleared from memory and browser storage.");
    }
  }

  function handleStrokeStart(point: Point): void {
    setStrokes((previous) => [...previous, [point]]);
  }

  function handleStrokeMove(point: Point): void {
    setStrokes((previous) => {
      if (previous.length === 0) {
        return previous;
      }
      const next = previous.slice();
      const last = next[next.length - 1];
      if (!last || last.length === 0) {
        return previous;
      }
      const lastPoint = last[last.length - 1]!;
      const dx = point.x - lastPoint.x;
      const dy = point.y - lastPoint.y;
      if (dx * dx + dy * dy < 2.25) {
        return previous;
      }
      next[next.length - 1] = [...last, point];
      return next;
    });
  }

  function handleStrokeEnd(): void {
    // The drawing is already committed point-by-point.
  }

  function handleUndo(): void {
    if (busy()) {
      return;
    }
    setStrokes((previous) => previous.slice(0, -1));
    clearPredictions();
  }

  function handleClear(): void {
    if (busy()) {
      return;
    }
    setStrokes([]);
    clearPredictions();
  }

  function handleAddLabel(name: string): boolean {
    const trimmed = name.trim();
    if (trimmed.length === 0) {
      return false;
    }
    if (busy()) {
      return false;
    }
    if (
      labels().some(
        (label) => label.name.toLowerCase() === trimmed.toLowerCase(),
      )
    ) {
      showStatus("error", `A label named “${trimmed}” already exists.`);
      return false;
    }
    if (labels().length >= MAX_LABELS) {
      showStatus("error", `You can have up to ${MAX_LABELS} labels.`);
      return false;
    }

    const label: SymbolLabel = {
      id: createId(),
      name: trimmed,
      color: LABEL_COLORS[labels().length % LABEL_COLORS.length]!,
    };
    setLabels((previous) => [...previous, label]);
    setSelectedLabelId(label.id);
    markModelStale();
    showStatus("info", `Added label “${label.name}”.`);
    return true;
  }

  function handleRemoveLabel(id: string): void {
    if (busy()) {
      return;
    }
    const label = labels().find((candidate) => candidate.id === id);
    if (!label) {
      return;
    }
    const labelSamples = samples().filter((sample) => sample.labelId === id);
    if (
      labelSamples.length > 0 &&
      !window.confirm(
        `Delete “${label.name}” and its ${labelSamples.length} drawing${
          labelSamples.length === 1 ? "" : "s"
        }?`,
      )
    ) {
      return;
    }

    const removedEnsembles = ensembles().filter(
      (ensemble) => ensemble.labelId === id,
    );
    destroyEnsembles(removedEnsembles);
    const remainingEnsembles = ensembles().filter(
      (ensemble) => ensemble.labelId !== id,
    );
    const remainingLabels = labels().filter((candidate) => candidate.id !== id);

    setEnsembles(remainingEnsembles);
    setLabels(remainingLabels);
    setSamples((previous) =>
      previous.filter((sample) => sample.labelId !== id),
    );
    if (selectedLabelId() === id) {
      setSelectedLabelId(remainingLabels[0]?.id ?? null);
    }
    setPredictions([]);

    if (remainingEnsembles.length === 0) {
      setModelInfo(null);
      setModelStale(false);
    } else {
      setModelStale(true);
    }

    showStatus("info", `Removed “${label.name}”.`);
  }

  function handleSaveSample(): void {
    if (busy()) {
      return;
    }
    const label = selectedLabel();
    if (!label) {
      showStatus("error", "Choose a label before saving a drawing.");
      return;
    }
    const input = currentInput();
    if (!inputHasInk(input)) {
      showStatus("error", "Draw something in the canvas first.");
      return;
    }

    const sample: DrawingSample = {
      id: createId(),
      labelId: label.id,
      strokes: cloneStrokes(strokes()),
      input,
      gridSize: gridSize(),
      createdAt: Date.now(),
    };

    setSamples((previous) => [...previous, sample]);
    setStrokes([]);
    setPredictions([]);
    markModelStale();
    showStatus("success", `Added a drawing to “${label.name}”.`);
  }

  function handleLoadSample(sample: DrawingSample): void {
    if (busy()) {
      return;
    }
    setStrokes(cloneStrokes(sample.strokes));
    setSelectedLabelId(sample.labelId);
    setPredictions([]);
    showStatus("info", "Replayed the saved drawing.");
  }

  function handleDeleteSample(id: string): void {
    if (busy()) {
      return;
    }
    setSamples((previous) => previous.filter((sample) => sample.id !== id));
    setPredictions([]);
    markModelStale();
    showStatus("info", "Deleted the drawing.");
  }

  function applyGridSize(nextGrid: number): void {
    const previousInputSize = inputSize();
    const parsed = parsedShape();
    const nextShapeText =
      parsed.shape && parsed.shape[0] === previousInputSize
        ? [nextGrid * nextGrid, ...parsed.shape.slice(1)].join(", ")
        : formatShape(defaultShape(nextGrid));

    destroyEnsembles(ensembles());
    setEnsembles([]);
    setModelInfo(null);
    setModelStale(false);
    setPredictions([]);

    setGridSize(nextGrid);
    setGridDraft(String(nextGrid));
    setShapeText(nextShapeText);
    setSamples((previous) =>
      previous.map((sample) => ({
        ...sample,
        gridSize: nextGrid,
        input: rasterizeStrokes(sample.strokes, nextGrid),
      })),
    );

    showStatus(
      "info",
      `Grid changed to ${nextGrid} × ${nextGrid}. Retrain the model to use it.`,
    );
  }

  function handleGridCommit(): void {
    if (busy()) {
      setGridDraft(String(gridSize()));
      return;
    }
    const value = Number(gridDraft().trim());
    if (
      !Number.isInteger(value) ||
      value < MIN_GRID_SIZE ||
      value > MAX_GRID_SIZE
    ) {
      setGridDraft(String(gridSize()));
      showStatus(
        "error",
        `Grid size must be a whole number from ${MIN_GRID_SIZE} to ${MAX_GRID_SIZE}.`,
      );
      return;
    }
    if (value === gridSize()) {
      setGridDraft(String(value));
      return;
    }
    applyGridSize(value);
  }

  function handleShapeTextChange(value: string): void {
    setShapeText(value);
    if (ensembles().length > 0) {
      setModelStale(true);
      setPredictions([]);
    }
  }

  function handleModeChange(next: "train" | "test"): void {
    if (mode() === next) {
      return;
    }
    setMode(next);
    setPredictions([]);
  }

  function handleInferenceModeChange(next: InferenceMode): void {
    if (busy() || next === inferenceMode()) {
      return;
    }
    setInferenceMode(next);
  }

  function handleCancelTrain(): void {
    if (!trainingToken) {
      return;
    }
    trainingToken.cancelled = true;
    showStatus("info", "Cancelling training…");
  }

  async function handleTrain(): Promise<void> {
    const runtimeInstance = instance();
    const parsed = parsedShape();
    if (
      !runtimeInstance ||
      runtime() !== "ready" ||
      busy() ||
      storageBusy() ||
      parsed.error !== null
    ) {
      return;
    }
    if (trainableLabels().length === 0) {
      showStatus("error", trainHint());
      return;
    }

    const token = { cancelled: false };
    trainingToken = token;
    setBusy(true);
    setStatus(null);
    setProgress({
      completed: 0,
      total: trainableLabels().length * CIRCUITS_PER_LABEL,
      labelName: "",
    });

    try {
      const result = await trainEnsembles({
        og: runtimeInstance,
        labels: labels(),
        samples: samples(),
        shape: parsed.shape,
        isCancelled: () => token.cancelled,
        onProgress: (nextProgress) => setProgress(nextProgress),
      });

      if (token.cancelled) {
        destroyEnsembles(result.ensembles);
        showStatus("info", "Training cancelled. The previous model was kept.");
        return;
      }

      const previousEnsembles = ensembles();
      const nextInfo: ModelInfo = {
        shape: [...parsed.shape],
        gridSize: gridSize(),
        labelCount: result.ensembles.length,
        totalCircuits: result.totalCircuits,
        averageLoss: result.averageLoss,
        trainedAt: Date.now(),
      };
      setEnsembles(result.ensembles);
      destroyEnsembles(previousEnsembles);
      setModelInfo(nextInfo);
      setModelStale(false);
      setPredictions([]);

      const saveError = await persistModel(nextInfo, result.ensembles);
      const skipped =
        result.skipped.length > 0
          ? ` Skipped ${result.skipped.map((entry) => entry.label.name).join(", ")}.`
          : "";
      const summary = `Trained ${result.totalCircuits} circuits across ${result.ensembles.length} label${
        result.ensembles.length === 1 ? "" : "s"
      }.${skipped}`;
      showStatus(
        saveError ? "error" : "success",
        saveError
          ? `${summary} Could not save it in this browser: ${saveError}`
          : `${summary} Saved in this browser.`,
      );
    } catch (error) {
      if (error instanceof TrainingCancelledError || token.cancelled) {
        showStatus("info", "Training cancelled. The previous model was kept.");
      } else {
        showStatus("error", `Training failed: ${errorMessage(error)}`);
      }
    } finally {
      if (trainingToken === token) {
        trainingToken = null;
      }
      setBusy(false);
      setProgress(null);
    }
  }

  return (
    <div class="app-shell">
      <header class="app-header">
        <div class="brand">
          <div class="brand-mark" aria-hidden="true">
            GF
          </div>
          <div>
            <span class="eyebrow">Differentiable logic circuits</span>
            <h1>Gate Forest</h1>
            <p>
              Draw a symbol and grow a forest of boolean circuits, or train a
              tiny binary language model on your own text.
            </p>
          </div>
        </div>
        <div class="header-actions">
          <a
            class="button secondary"
            href={import.meta.env.DEV ? "http://localhost:5175/" : "learn/"}
          >
            How it works
          </a>
          <Show when={activeTab() === "symbols"}>
            <button
              type="button"
              class="button secondary"
              disabled={busy() || modelInfo() === null}
              title={
                modelInfo() === null
                  ? "Train a forest before compiling it."
                  : "Compile the hard-mode forest into standalone JavaScript."
              }
              onClick={() => setCompileOpen(true)}
            >
              Compile
            </button>
          </Show>
          <Show when={activeTab() === "symbols"}>
            <div class={["runtime-pill", runtime()]}>
              <span class="runtime-dot" />
              {runtime() === "ready"
                ? "Runtime ready"
                : runtime() === "loading"
                  ? "Loading runtime"
                  : "Runtime error"}
            </div>
          </Show>
        </div>
      </header>

      <nav class="app-tabs" role="tablist" aria-label="Workspace">
        <button
          type="button"
          role="tab"
          aria-selected={activeTab() === "symbols" ? "true" : "false"}
          class={{ active: activeTab() === "symbols" }}
          onClick={() => setActiveTab("symbols")}
        >
          Symbol classifier
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={activeTab() === "language" ? "true" : "false"}
          class={{ active: activeTab() === "language" }}
          onClick={() => setActiveTab("language")}
        >
          Mini language model
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={activeTab() === "integer" ? "true" : "false"}
          class={{ active: activeTab() === "integer" }}
          onClick={() => setActiveTab("integer")}
        >
          Integer GPT
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={activeTab() === "luau" ? "true" : "false"}
          class={{ active: activeTab() === "luau" }}
          onClick={() => setActiveTab("luau")}
        >
          Luau
        </button>
      </nav>

      <div class="tab-panel" hidden={activeTab() !== "symbols"}>
        <Show when={status()}>
          {(message) => (
            <div class={["banner", message().kind]}>
              <span>{message().text}</span>
              <button
                type="button"
                class="banner-close"
                aria-label="Dismiss message"
                onClick={() => setStatus(null)}
              >
                ×
              </button>
            </div>
          )}
        </Show>

        <Show when={runtime() === "error"}>
          <div class="banner error runtime-error">
            <div>
              <strong>Could not load the WebAssembly runtime.</strong>
              <p>{runtimeError()}</p>
            </div>
            <button
              type="button"
              class="button secondary"
              onClick={() => void loadRuntime()}
            >
              Retry
            </button>
          </div>
        </Show>

        <main class="workspace">
          <section class="card draw-card">
            <div class="card-heading">
              <div>
                <span class="eyebrow">Canvas</span>
                <h2>
                  {mode() === "train"
                    ? "Draw a training example"
                    : "Draw a symbol to identify"}
                </h2>
              </div>
              <div class="mode-switch" role="group" aria-label="Mode">
                <button
                  type="button"
                  class={{ active: mode() === "train" }}
                  onClick={() => handleModeChange("train")}
                >
                  Training
                </button>
                <button
                  type="button"
                  class={{ active: mode() === "test" }}
                  onClick={() => handleModeChange("test")}
                >
                  Testing
                </button>
              </div>
            </div>

            <DrawingPad
              strokes={strokes()}
              input={currentInput()}
              gridSize={gridSize()}
              color={selectedLabel()?.color}
              disabled={runtime() !== "ready" || busy()}
              onStrokeStart={handleStrokeStart}
              onStrokeMove={handleStrokeMove}
              onStrokeEnd={handleStrokeEnd}
              onUndo={handleUndo}
              onClear={handleClear}
            />

            <Show when={mode() === "test"}>
              <LivePrediction
                predictions={predictions()}
                hasDrawing={hasDrawing()}
                hasModel={modelInfo() !== null}
                modelStale={modelStale()}
                inferenceMode={inferenceMode()}
              />
            </Show>

            <p class="hint">
              Saved drawings remember their strokes. The input is rebuilt by
              replaying those strokes onto a {gridSize()} × {gridSize()} grid after
              scaling and centring, with a constant stroke thickness.
            </p>
          </section>

          <div class="side-stack">
            <section class="card mode-card">
              <Show
                when={mode() === "train"}
                fallback={
                  <TestingPanel
                    predictions={predictions()}
                    modelInfo={modelInfo()}
                    modelStale={modelStale()}
                    inferenceMode={inferenceMode()}
                    busy={busy()}
                    hasDrawing={hasDrawing()}
                    predictHint={predictHint()}
                    onInferenceModeChange={handleInferenceModeChange}
                  />
                }
              >
                <TrainingPanel
                  labels={labels()}
                  samples={samples()}
                  selectedLabelId={selectedLabelId()}
                  busy={busy()}
                  progress={progress()}
                  modelInfo={modelInfo()}
                  modelStale={modelStale()}
                  hasDrawing={hasDrawing()}
                  canTrain={canTrain()}
                  trainHint={trainHint()}
                  onSelectLabel={setSelectedLabelId}
                  onAddLabel={handleAddLabel}
                  onRemoveLabel={handleRemoveLabel}
                  onSaveSample={handleSaveSample}
                  onLoadSample={handleLoadSample}
                  onDeleteSample={handleDeleteSample}
                  onTrain={() => void handleTrain()}
                  onCancel={handleCancelTrain}
                />
              </Show>
            </section>

            <SettingsPanel
              gridSize={gridSize()}
              gridDraft={gridDraft()}
              shapeText={shapeText()}
              shapeError={parsedShape().error}
              parameterCount={parsedShape().parameterCount}
              inputSize={inputSize()}
              disabled={busy() || storageBusy()}
              onGridDraftChange={setGridDraft}
              onGridCommit={handleGridCommit}
              onShapeTextChange={handleShapeTextChange}
            />

            <ModelStoragePanel
              hasModel={modelInfo() !== null}
              busy={busy() || storageBusy()}
              storageAvailable={storageAvailable()}
              storageError={storageError()}
              storedAt={storedModelAt()}
              onSave={handleSaveModelFile}
              onLoad={(file) => void handleLoadModelFile(file)}
              onClear={() => void handleClearModel()}
            />
          </div>
        </main>

        <CompileDialog
          open={compileOpen()}
          labels={compileLabels()}
          onClose={() => setCompileOpen(false)}
        />
      </div>

      <div class="tab-panel" hidden={activeTab() !== "language"}>
        <LanguageModelPanel />
      </div>

      <div class="tab-panel" hidden={activeTab() !== "integer"}>
        <IntegerGptPanel />
      </div>

      <div class="tab-panel" hidden={activeTab() !== "luau"}>
        <LuauPanel active={activeTab() === "luau"} />
      </div>

      <footer class="app-footer">
        <Show
          when={activeTab() === "luau"}
          fallback={
            <Show
              when={activeTab() === "symbols"}
              fallback={
                <>
                  <span>
                    Word vectors: a Word2Bits subset (Apache-2.0). Autocomplete
                    trains with floats and predicts with popcount and Q15
                    fixed-point softmax.
                  </span>
                  <span>
                    Q&A mode retrieves the trained answer closest to your
                    question. Nothing is uploaded; everything stays in this
                    browser tab.
                  </span>
                </>
              }
            >
              <span>
                {CIRCUITS_PER_LABEL} circuits per label · saved strokes are
                replayed, never re-rasterised from the display canvas.
              </span>
              <span>
                Soft circuits weigh every gate; hard circuits lock each wire and
                gate to its most likely choice.
              </span>
            </Show>
          }
        >
          <span>
            Luau answers come from a task-tuned tiny GPT. Write prompts use a
            `Task:` prefix, question prompts use `Question:`; both run through
            the same integer-only WASM inference path.
          </span>
          <span>
            Generated code is shown for reading and copying. It is not executed
            in the browser; run it in the Luau CLI or Roblox Studio.
          </span>
        </Show>
      </footer>
    </div>
  );
}
