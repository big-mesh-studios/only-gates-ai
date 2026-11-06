import { createMemo, For, Show } from "solid-js";

import type {
  InferenceMode,
  ModelInfo,
  PredictionResult,
} from "../lib/types";

interface TestingPanelProps {
  readonly predictions: readonly PredictionResult[];
  readonly modelInfo: ModelInfo | null;
  readonly modelStale: boolean;
  readonly inferenceMode: InferenceMode;
  readonly busy: boolean;
  readonly hasDrawing: boolean;
  readonly predictHint: string;
  readonly onInferenceModeChange: (mode: InferenceMode) => void;
}

export default function TestingPanel(props: TestingPanelProps) {
  const topPrediction = createMemo(() => props.predictions[0] ?? null);
  const otherPredictions = createMemo(() => props.predictions.slice(1));

  return (
    <div class="panel-stack">
      <div>
        <span class="eyebrow">Testing mode</span>
        <h2>Ask the forest</h2>
        <p class="hint">
          Draw a symbol and every circuit votes yes or no for its label. The
          label with the most yes votes wins; the average output breaks ties.
        </p>
      </div>

      <div class="inference-block">
        <div class="section-heading">
          <span>Circuit mode</span>
          <span class="muted">
            {props.inferenceMode === "soft" ? "probability-weighted" : "argmax choices"}
          </span>
        </div>
        <div class="inference-toggle" role="group" aria-label="Circuit mode">
          <button
            type="button"
            class={{ active: props.inferenceMode === "soft" }}
            disabled={props.busy}
            onClick={() => props.onInferenceModeChange("soft")}
          >
            Soft circuit
          </button>
          <button
            type="button"
            class={{ active: props.inferenceMode === "hard" }}
            disabled={props.busy}
            onClick={() => props.onInferenceModeChange("hard")}
          >
            Hard circuit
          </button>
        </div>
        <p class="hint">
          {props.inferenceMode === "soft"
            ? "Soft mode lets probability-weighted wires and gates answer together."
            : "Hard mode locks every wire and gate to its highest-probability choice before feeding the drawing through."}
        </p>
      </div>

      <p class="hint">{props.predictHint}</p>

      <Show when={topPrediction()}>
        {(result) => (
          <div
            class="prediction-card"
            style={{ "border-color": result().color }}
          >
            <div class="prediction-head">
              <span
                class="label-dot large"
                style={{ background: result().color }}
              />
              <div>
                <span class="eyebrow">
                  {result().votes > 0 ? "Top match" : "Best guess"}
                </span>
                <h3>{result().labelName}</h3>
              </div>
              <span class="vote-pill">
                {result().votes} / {result().totalCircuits}
              </span>
            </div>

            <div class="score-track">
              <div
                class="score-fill"
                style={{
                  width: `${Math.round(result().score * 100)}%`,
                  background: result().color,
                }}
              />
            </div>
            <div class="prediction-meta">
              <span>{Math.round(result().score * 100)}% average output</span>
              <span>
                {result().votes === 0
                  ? "No circuit voted yes"
                  : `${result().votes} of ${result().totalCircuits} circuits say yes`}
              </span>
            </div>
          </div>
        )}
      </Show>

      <Show
        when={otherPredictions().length > 0}
        fallback={
          <Show when={topPrediction() === null}>
            <div class="empty-state">
              No prediction yet. Draw a symbol in the canvas — results update
              automatically.
            </div>
          </Show>
        }
      >
        <div class="results-list">
          <div class="section-heading">
            <span>All labels</span>
            <span class="muted">votes, then score</span>
          </div>
          <For each={otherPredictions()}>
            {(result) => (
              <div class="result-row">
                <span
                  class="label-dot"
                  style={{ background: result.color }}
                />
                <span class="result-name">{result.labelName}</span>
                <span class="vote-pill small">
                  {result.votes} / {result.totalCircuits}
                </span>
                <span class="mini-track">
                  <span
                    class="mini-fill"
                    style={{
                      width: `${Math.round(result.score * 100)}%`,
                      background: result.color,
                    }}
                  />
                </span>
                <span class="result-score">
                  {Math.round(result.score * 100)}%
                </span>
              </div>
            )}
          </For>
        </div>
      </Show>

      <Show when={props.modelInfo}>
        {(info) => (
          <div class={["model-status", { stale: props.modelStale }]}>
            <div class="model-status-title">
              <span class="status-dot" />
              {props.modelStale ? "Using an older model" : "Model ready"}
            </div>
            <p>
              {info().totalCircuits} circuits · shape {info().shape.join(" → ")}
            </p>
            <Show when={props.modelStale}>
              <p>Retrain in training mode to include your latest drawings.</p>
            </Show>
          </div>
        )}
      </Show>

      <Show when={!props.modelInfo}>
        <div class="empty-state">
          No forest yet. Switch to training mode, add drawings for at least two
          labels, and train the model.
        </div>
      </Show>

      <Show when={!props.hasDrawing && props.modelInfo}>
        <p class="hint">The canvas is empty, so there is nothing to predict yet.</p>
      </Show>
    </div>
  );
}
