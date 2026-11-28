import {
  For,
  Show,
  createMemo,
  createSignal,
  onSettled,
} from "solid-js";

import {
  DIGIT_GRID,
  digitInputFromStrokes,
  loadLutModel,
  predictDigit,
  type LutModel,
} from "../lib/lut4Classifier";
import { hasInk } from "../lib/rasterizer";
import type { Point, Strokes } from "../lib/types";
import DrawingPad from "./DrawingPad";
// Imported through Vite so the file gets a content hash in its name. A cached
// copy of an older build then keeps requesting its own model instead of
// pairing old JS with a new circuit of the wrong input size.
import modelUrl from "../assets/lut4-mnist-784.json?url";

const DIGIT_COLOR = "#7cc4ff";

export default function DigitsPanel() {
  const [strokes, setStrokes] = createSignal<Strokes>([]);
  const [model, setModel] = createSignal<LutModel | null>(null);
  const [loading, setLoading] = createSignal(true);
  const [loadError, setLoadError] = createSignal<string | null>(null);

  onSettled(() => {
    void (async () => {
      try {
        setModel(await loadLutModel(modelUrl));
      } catch (error) {
        setLoadError(error instanceof Error ? error.message : String(error));
      } finally {
        setLoading(false);
      }
    })();
  });

  const input = createMemo(() => digitInputFromStrokes(strokes()));
  const hasDrawing = createMemo(() => hasInk(strokes()));
  const prediction = createMemo(() => {
    const loaded = model();
    if (!loaded || !hasDrawing()) {
      return null;
    }
    return predictDigit(loaded, input());
  });

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

  function handleUndo(): void {
    setStrokes((previous) => previous.slice(0, -1));
  }

  function handleClear(): void {
    setStrokes([]);
  }

  return (
    <main class="workspace">
      <section class="card draw-card">
        <div class="card-heading">
          <div>
            <span class="eyebrow">Canvas</span>
            <h2>Draw a digit</h2>
          </div>
          <span class="muted">{DIGIT_GRID} × {DIGIT_GRID} binary</span>
        </div>

        <DrawingPad
          strokes={strokes()}
          input={input()}
          gridSize={DIGIT_GRID}
          color={DIGIT_COLOR}
          onStrokeStart={handleStrokeStart}
          onStrokeMove={handleStrokeMove}
          onStrokeEnd={() => undefined}
          onUndo={handleUndo}
          onClear={handleClear}
        />
      </section>

      <section class="card">
        <div class="live-prediction" aria-live="polite">
          <div class="section-heading">
            <span>Prediction</span>
            <span class="muted">
              hard 4-input LUT forest · 92.25% MNIST
            </span>
          </div>

          <Show
            when={!loading()}
            fallback={<p class="hint">Loading the digit circuit…</p>}
          >
            <Show
              when={!loadError()}
              fallback={
                <p class="hint">
                  Could not load the digit circuit. Run the export in
                  `experiments/gate-cnn-mnist` and rebuild.
                </p>
              }
            >
              <Show
                when={prediction()}
                fallback={
                  <p class="hint">
                    {hasDrawing()
                      ? "Reading the forest…"
                      : "Draw a digit from 0 to 9."}
                  </p>
                }
              >
                {(result) => (
                  <>
                    <div class="live-verdict">
                      <span
                        class="label-dot large"
                        style={{ background: DIGIT_COLOR }}
                      />
                      <div class="live-verdict-text">
                        <span class="eyebrow">Digit</span>
                        <strong class="live-verdict-name">
                          {result().label}
                        </strong>
                      </div>
                      <span class="vote-pill">
                        {result().scores[result().label]} /{" "}
                        {result().totalCircuits} votes
                      </span>
                    </div>

                    <div class="digit-scores">
                      <For each={result().scores}>
                        {(score, index) => (
                          <div
                            class={{
                              "digit-score": true,
                              best: index() === result().label,
                            }}
                          >
                            <span class="digit-score-label">{index()}</span>
                            <div class="score-track">
                              <div
                                class="score-fill"
                                style={{
                                  width: `${(score / result().totalCircuits) * 100}%`,
                                  background:
                                    index() === result().label
                                      ? DIGIT_COLOR
                                      : undefined,
                                }}
                              />
                            </div>
                            <span class="digit-score-value">{score}</span>
                          </div>
                        )}
                      </For>
                    </div>
                  </>
                )}
              </Show>
            </Show>
          </Show>
        </div>
      </section>
    </main>
  );
}
