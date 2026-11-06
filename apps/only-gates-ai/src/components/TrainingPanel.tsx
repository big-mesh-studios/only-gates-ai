import { createMemo, createSignal, For, Show } from "solid-js";

import { CIRCUITS_PER_LABEL, type TrainingProgress } from "../lib/model";
import type { DrawingSample, ModelInfo, SymbolLabel } from "../lib/types";
import GridPreview from "./GridPreview";

interface TrainingPanelProps {
  readonly labels: readonly SymbolLabel[];
  readonly samples: readonly DrawingSample[];
  readonly selectedLabelId: string | null;
  readonly busy: boolean;
  readonly progress: TrainingProgress | null;
  readonly modelInfo: ModelInfo | null;
  readonly modelStale: boolean;
  readonly hasDrawing: boolean;
  readonly canTrain: boolean;
  readonly trainHint: string;
  readonly onSelectLabel: (id: string) => void;
  readonly onAddLabel: (name: string) => boolean;
  readonly onRemoveLabel: (id: string) => void;
  readonly onSaveSample: () => void;
  readonly onLoadSample: (sample: DrawingSample) => void;
  readonly onDeleteSample: (id: string) => void;
  readonly onTrain: () => void;
  readonly onCancel: () => void;
}

export default function TrainingPanel(props: TrainingPanelProps) {
  const [draftName, setDraftName] = createSignal("");

  const counts = createMemo(() => {
    const map = new Map<string, number>();
    for (const sample of props.samples) {
      map.set(sample.labelId, (map.get(sample.labelId) ?? 0) + 1);
    }
    return map;
  });

  const selectedLabel = createMemo(
    () => props.labels.find((label) => label.id === props.selectedLabelId) ?? null,
  );

  const selectedSamples = createMemo(() =>
    props.samples
      .filter((sample) => sample.labelId === props.selectedLabelId)
      .sort((a, b) => b.createdAt - a.createdAt)
      .slice(0, 12),
  );

  function handleAddLabel(event: SubmitEvent): void {
    event.preventDefault();
    const name = draftName().trim();
    if (name.length === 0) {
      return;
    }
    if (props.onAddLabel(name)) {
      setDraftName("");
    }
  }

  return (
    <div class="panel-stack">
      <div>
        <span class="eyebrow">Training mode</span>
        <h2>Teach by drawing</h2>
        <p class="hint">
          Pick a label, draw one example, and add it to that label. Each circuit
          learns to answer “is this the label?” and the forest votes.
        </p>
      </div>

      <div class="label-section">
        <div class="section-heading">
          <span>Labels</span>
          <span class="muted">{props.labels.length} / 24</span>
        </div>

        <div class="label-list">
          <For each={props.labels}>
            {(label) => (
              <div
                class={[
                  "label-chip",
                  { selected: label.id === props.selectedLabelId },
                ]}
              >
                <button
                  type="button"
                  class="label-select"
                  disabled={props.busy}
                  onClick={() => props.onSelectLabel(label.id)}
                >
                  <span class="label-dot" style={{ background: label.color }} />
                  <span class="label-name">{label.name}</span>
                  <span class="label-count">{counts().get(label.id) ?? 0}</span>
                </button>
                <button
                  type="button"
                  class="label-remove"
                  title={`Delete ${label.name}`}
                  aria-label={`Delete label ${label.name}`}
                  disabled={props.busy}
                  onClick={() => props.onRemoveLabel(label.id)}
                >
                  ×
                </button>
              </div>
            )}
          </For>
        </div>

        <form class="add-label" onSubmit={handleAddLabel}>
          <input
            type="text"
            maxlength="24"
            placeholder="Add a label, e.g. star"
            value={draftName()}
            disabled={props.busy}
            onInput={(event) => setDraftName(event.currentTarget.value)}
          />
          <button
            type="submit"
            class="button secondary"
            disabled={props.busy || draftName().trim().length === 0}
          >
            Add label
          </button>
        </form>
      </div>

      <Show
        when={selectedLabel()}
        fallback={
          <p class="empty-state">
            Add your first label above, then draw an example for it.
          </p>
        }
      >
        {(label) => (
          <div class="save-block">
            <div class="section-heading">
              <span>
                <span
                  class="label-dot inline"
                  style={{ background: label().color }}
                />
                Saving as {label().name}
              </span>
              <span class="muted">
                {counts().get(label().id) ?? 0} drawing
                {(counts().get(label().id) ?? 0) === 1 ? "" : "s"}
              </span>
            </div>
            <button
              type="button"
              class="button primary wide-button"
              disabled={!props.hasDrawing || props.busy}
              onClick={props.onSaveSample}
            >
              Add drawing to “{label().name}”
            </button>
            <p class="hint">
              The canvas is cleared after saving. Click a saved drawing to replay
              its strokes.
            </p>
          </div>
        )}
      </Show>

      <Show when={selectedSamples().length > 0}>
        <div class="sample-section">
          <div class="section-heading">
            <span>Recent drawings</span>
            <span class="muted">click to replay</span>
          </div>
          <div class="sample-strip">
            <For each={selectedSamples()}>
              {(sample) => (
                <div class="sample-card">
                  <button
                    type="button"
                    class="sample-load"
                    title="Replay this drawing"
                    disabled={props.busy}
                    onClick={() => props.onLoadSample(sample)}
                  >
                    <GridPreview
                      input={sample.input}
                      gridSize={sample.gridSize}
                      size={62}
                      color={selectedLabel()?.color}
                    />
                  </button>
                  <button
                    type="button"
                    class="sample-remove"
                    title="Delete drawing"
                    aria-label="Delete drawing"
                    disabled={props.busy}
                    onClick={() => props.onDeleteSample(sample.id)}
                  >
                    ×
                  </button>
                </div>
              )}
            </For>
          </div>
        </div>
      </Show>

      <div class="divider" />

      <div class="train-block">
        <span class="eyebrow">Ensemble</span>
        <h2>Train the forest</h2>
        <p class="hint">{props.trainHint}</p>

        <div class="button-row">
          <button
            type="button"
            class="button primary"
            disabled={!props.canTrain || props.busy}
            onClick={props.onTrain}
          >
            Train {CIRCUITS_PER_LABEL} circuits per label
          </button>
          <Show when={props.busy}>
            <button type="button" class="button danger" onClick={props.onCancel}>
              Cancel
            </button>
          </Show>
        </div>

        <Show when={props.progress}>
          {(progress) => (
            <div class="progress-block">
              <div class="progress-label">
                <span>
                  Training {progress().labelName || "the forest"}
                </span>
                <span>
                  {progress().completed} / {progress().total} circuits
                </span>
              </div>
              <div class="progress-track">
                <div
                  class="progress-fill"
                  style={{
                    width: `${
                      progress().total > 0
                        ? (progress().completed / progress().total) * 100
                        : 0
                    }%`,
                  }}
                />
              </div>
            </div>
          )}
        </Show>

        <Show when={props.modelInfo}>
          {(info) => (
            <div
              class={["model-status", { stale: props.modelStale }]}
            >
              <div class="model-status-title">
                <span class="status-dot" />
                {props.modelStale ? "Last model is out of date" : "Model ready"}
              </div>
              <p>
                {info().totalCircuits} circuits · {info().labelCount} labels ·
                shape {info().shape.join(" → ")} · avg loss{" "}
                {info().averageLoss.toFixed(3)}
              </p>
              <Show when={props.modelStale}>
                <p>Retrain to include your latest drawings.</p>
              </Show>
            </div>
          )}
        </Show>
      </div>
    </div>
  );
}
