import { MAX_GRID_SIZE, MIN_GRID_SIZE } from "../lib/rasterizer";
import { CIRCUITS_PER_LABEL } from "../lib/model";

interface SettingsPanelProps {
  readonly gridSize: number;
  readonly gridDraft: string;
  readonly shapeText: string;
  readonly shapeError: string | null;
  readonly parameterCount: number;
  readonly inputSize: number;
  readonly disabled?: boolean;
  readonly onGridDraftChange: (value: string) => void;
  readonly onGridCommit: () => void;
  readonly onShapeTextChange: (value: string) => void;
}

export default function SettingsPanel(props: SettingsPanelProps) {
  return (
    <section class="card settings-card">
      <div class="card-heading compact">
        <div>
          <span class="eyebrow">Configuration</span>
          <h2>Model settings</h2>
        </div>
      </div>

      <div class="settings-grid">
        <label class="field">
          <span>Grid size</span>
          <input
            type="number"
            min={MIN_GRID_SIZE}
            max={MAX_GRID_SIZE}
            step="1"
            value={props.gridDraft}
            disabled={props.disabled}
            onInput={(event) => props.onGridDraftChange(event.currentTarget.value)}
            onChange={props.onGridCommit}
            onKeyDown={(event) => {
              if (event.key === "Enter") {
                event.currentTarget.blur();
              }
            }}
          />
          <small>
            {props.gridSize} × {props.gridSize} = {props.inputSize} boolean inputs
          </small>
        </label>

        <label class="field wide">
          <span>Circuit shape</span>
          <input
            type="text"
            value={props.shapeText}
            disabled={props.disabled}
            spellcheck={false}
            onInput={(event) => props.onShapeTextChange(event.currentTarget.value)}
          />
          <small>
            Comma-separated layer sizes. The first layer is the grid input; the
            last is always one yes/no output.
          </small>
        </label>
      </div>

      {props.shapeError ? (
        <p class="error-text">{props.shapeError}</p>
      ) : (
        <div class="stat-row">
          <span class="stat">
            <strong>{props.parameterCount.toLocaleString()}</strong>
            parameters / circuit
          </span>
          <span class="stat">
            <strong>{CIRCUITS_PER_LABEL}</strong>
            circuits / label
          </span>
        </div>
      )}

      <p class="hint">
        Strokes are remembered and replayed onto the grid. The drawing is scaled
        and centred first, then drawn with a constant stroke thickness. Changing
        the grid size or shape requires retraining.
      </p>
    </section>
  );
}
