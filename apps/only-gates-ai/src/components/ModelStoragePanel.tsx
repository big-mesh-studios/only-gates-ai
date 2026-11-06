import { createMemo } from "solid-js";

interface ModelStoragePanelProps {
  readonly hasModel: boolean;
  readonly busy: boolean;
  readonly storageAvailable: boolean;
  readonly storageError: string | null;
  readonly storedAt: number | null;
  readonly onSave: () => void;
  readonly onLoad: (file: File) => void;
  readonly onClear: () => void;
}

export default function ModelStoragePanel(props: ModelStoragePanelProps) {
  let fileInput: HTMLInputElement | undefined;

  const savedAtText = createMemo(() => {
    const timestamp = props.storedAt;
    return timestamp === null ? null : new Date(timestamp).toLocaleString();
  });

  const statusTitle = createMemo(() => {
    if (!props.storageAvailable) {
      return "Browser storage unavailable";
    }
    if (props.hasModel && savedAtText() !== null) {
      return "Model saved in this browser";
    }
    if (props.hasModel) {
      return "Model is in memory only";
    }
    return "No model saved yet";
  });

  const statusText = createMemo(() => {
    if (props.storageError) {
      return props.storageError;
    }
    if (!props.storageAvailable) {
      return "IndexedDB could not be opened, so the forest will not persist. Download it to a file instead.";
    }
    if (props.hasModel && savedAtText() !== null) {
      return `Saved automatically at ${savedAtText()}. Loading a file replaces the current forest, but leaves your labels and drawings alone.`;
    }
    if (props.hasModel) {
      return "This model will disappear when the tab closes unless you save it to a file.";
    }
    return "Train the forest and it will be saved here automatically. You can also load a model from a file.";
  });

  return (
    <section class="card storage-card">
      <div class="card-heading compact">
        <div>
          <span class="eyebrow">Persistence</span>
          <h2>Model storage</h2>
        </div>
      </div>

      <div
        class={[
          "storage-status",
          {
            saved: props.hasModel && savedAtText() !== null,
            error: !props.storageAvailable,
          },
        ]}
      >
        <span class="status-dot" />
        <div>
          <strong>{statusTitle()}</strong>
          <p>{statusText()}</p>
        </div>
      </div>

      <div class="button-row storage-actions">
        <button
          type="button"
          class="button secondary"
          disabled={!props.hasModel || props.busy}
          onClick={props.onSave}
        >
          Save model
        </button>
        <button
          type="button"
          class="button secondary"
          disabled={props.busy}
          onClick={() => fileInput?.click()}
        >
          Load model
        </button>
        <button
          type="button"
          class="button danger"
          disabled={props.busy}
          onClick={props.onClear}
        >
          Clear model
        </button>
      </div>

      <input
        ref={(element) => {
          fileInput = element;
        }}
        type="file"
        accept=".json,application/json"
        hidden
        onChange={(event) => {
          const file = event.currentTarget.files?.[0];
          event.currentTarget.value = "";
          if (file) {
            props.onLoad(file);
          }
        }}
      />

      <p class="hint">
        The trained circuits are stored in IndexedDB. Clearing the model erases
        both the live forest and its saved copy, but keeps your labels and
        drawings.
      </p>
    </section>
  );
}
