import { createMemo, createSignal, onSettled, Show, untrack } from "solid-js";
import { createOnlyGatesAI, type OnlyGatesAI } from "@only-gates-ai/core";

import DrawingPad from "./components/DrawingPad";
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

  const [mode, setMode] = createSignal<"train" | "test">("train");
  const [inferenceMode, setInferenceMode] = createSignal<InferenceMode>("soft");

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

  let disposed = false;
  let loadingRuntime = false;
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
      parsedShape().error === null &&
      trainableLabels().length > 0,
  );

  const canPredict = createMemo(
    () =>
      runtime() === "ready" &&
      !busy() &&
      hasDrawing() &&
      ensembles().length > 0,
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
      return "Draw a symbol in the canvas to get a prediction.";
    }
    if (modelStale()) {
      return "This uses the last trained model. Retrain to include your latest drawings.";
    }
    return "Ready to predict with the current forest.";
  });

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

  function handleStrokeStart(point: Point): void {
    setStrokes((previous) => [...previous, [point]]);
    clearPredictions();
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
    clearPredictions();
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
    if (
      predictions().length > 0 &&
      hasDrawing() &&
      ensembles().length > 0
    ) {
      setPredictions(predictEnsembles(ensembles(), currentInput(), next));
    }
  }

  function handlePredict(): void {
    if (runtime() !== "ready" || busy()) {
      return;
    }
    if (ensembles().length === 0) {
      showStatus("error", "Train the forest before predicting.");
      return;
    }
    const input = currentInput();
    if (!inputHasInk(input)) {
      showStatus("error", "Draw a symbol in the canvas first.");
      return;
    }
    try {
      setPredictions(predictEnsembles(ensembles(), input, inferenceMode()));
    } catch (error) {
      showStatus("error", `Prediction failed: ${errorMessage(error)}`);
    }
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
      setEnsembles(result.ensembles);
      destroyEnsembles(previousEnsembles);
      setModelInfo({
        shape: [...parsed.shape],
        gridSize: gridSize(),
        labelCount: result.ensembles.length,
        totalCircuits: result.totalCircuits,
        averageLoss: result.averageLoss,
        trainedAt: Date.now(),
      });
      setModelStale(false);
      setPredictions([]);

      const skipped =
        result.skipped.length > 0
          ? ` Skipped ${result.skipped.map((entry) => entry.label.name).join(", ")}.`
          : "";
      showStatus(
        "success",
        `Trained ${result.totalCircuits} circuits across ${result.ensembles.length} label${
          result.ensembles.length === 1 ? "" : "s"
        }.${skipped}`,
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
              Draw a symbol. Grow a forest of boolean circuits. Let them vote.
            </p>
          </div>
        </div>
        <div class={["runtime-pill", runtime()]}>
          <span class="runtime-dot" />
          {runtime() === "ready"
            ? "Runtime ready"
            : runtime() === "loading"
              ? "Loading runtime"
              : "Runtime error"}
        </div>
      </header>

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
                  canPredict={canPredict()}
                  predictHint={predictHint()}
                  onPredict={handlePredict}
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
            disabled={busy()}
            onGridDraftChange={setGridDraft}
            onGridCommit={handleGridCommit}
            onShapeTextChange={handleShapeTextChange}
          />
        </div>
      </main>

      <footer class="app-footer">
        <span>
          {CIRCUITS_PER_LABEL} circuits per label · saved strokes are replayed,
          never re-rasterised from the display canvas.
        </span>
        <span>
          Soft circuits weigh every gate; hard circuits lock each wire and gate
          to its most likely choice.
        </span>
      </footer>
    </div>
  );
}
