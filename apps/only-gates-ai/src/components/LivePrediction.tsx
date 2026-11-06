import { For, Show } from "solid-js";

import type { InferenceMode, PredictionResult } from "../lib/types";

interface LivePredictionProps {
  readonly predictions: readonly PredictionResult[];
  readonly hasDrawing: boolean;
  readonly hasModel: boolean;
  readonly modelStale: boolean;
  readonly inferenceMode: InferenceMode;
}

export default function LivePrediction(props: LivePredictionProps) {
  const topPrediction = () => props.predictions[0] ?? null;
  const runnersUp = () => props.predictions.slice(1, 4);

  return (
    <div class="live-prediction" aria-live="polite">
      <div class="section-heading">
        <span>Live prediction</span>
        <span class="muted">
          {props.inferenceMode === "soft" ? "soft circuit" : "hard circuit"}
          {props.modelStale ? " · older model" : ""}
        </span>
      </div>

      <Show
        when={props.hasModel}
        fallback={
          <p class="hint">
            Train the forest in training mode to see predictions here as you
            draw.
          </p>
        }
      >
        <Show
          when={props.hasDrawing ? topPrediction() : null}
          fallback={
            <p class="hint">
              {props.hasDrawing
                ? "Reading the forest…"
                : "Draw a symbol to see a prediction."}
            </p>
          }
        >
          {(result) => (
            <>
              <div class="live-verdict">
                <span
                  class="label-dot large"
                  style={{ background: result().color }}
                />
                <div class="live-verdict-text">
                  <span class="eyebrow">
                    {result().votes > 0 ? "Top match" : "Best guess"}
                  </span>
                  <strong class="live-verdict-name">
                    {result().labelName}
                  </strong>
                </div>
                <span class="live-score">
                  {Math.round(result().score * 100)}%
                </span>
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

              <Show when={runnersUp().length > 0}>
                <div class="live-runners">
                  <For each={runnersUp()}>
                    {(runner) => (
                      <span class="live-runner">
                        <span
                          class="label-dot"
                          style={{ background: runner.color }}
                        />
                        {runner.labelName}
                        <strong>{Math.round(runner.score * 100)}%</strong>
                      </span>
                    )}
                  </For>
                </div>
              </Show>
            </>
          )}
        </Show>
      </Show>
    </div>
  );
}
