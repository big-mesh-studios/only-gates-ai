import {
  createEffect,
  createMemo,
  createSignal,
  onSettled,
  Show,
} from "solid-js";

import {
  compileForest,
  type CompiledForest,
  type ForestLabelSource,
} from "@only-gates-ai/compiler";

interface CompileDialogProps {
  readonly open: boolean;
  readonly labels: readonly ForestLabelSource[];
  readonly onClose: () => void;
}

interface CompilationState {
  readonly result: CompiledForest | null;
  readonly error: string | null;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) {
    return `${bytes} B`;
  }
  return `${(bytes / 1024).toFixed(1)} KB`;
}

function fallbackCopy(text: string): void {
  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.setAttribute("readonly", "");
  textarea.style.position = "fixed";
  textarea.style.opacity = "0";
  document.body.append(textarea);
  textarea.select();
  document.execCommand("copy");
  textarea.remove();
}

export default function CompileDialog(props: CompileDialogProps) {
  let dialogRef: HTMLDialogElement | undefined;
  let copyTimer: ReturnType<typeof setTimeout> | undefined;

  const [copyLabel, setCopyLabel] = createSignal("Copy");

  const compilation = createMemo<CompilationState>(() => {
    if (!props.open) {
      return { result: null, error: null };
    }
    try {
      return {
        result: compileForest(props.labels, { target: "scalar" }),
        error: null,
      };
    } catch (error) {
      return { result: null, error: errorMessage(error) };
    }
  });

  const savedPercent = createMemo(() => {
    const result = compilation().result;
    if (!result || result.gateCount === 0) {
      return 0;
    }
    return Math.round(
      ((result.gateCount - result.emittedGateCount) / result.gateCount) * 100,
    );
  });

  createEffect(
    () => props.open,
    (open) => {
      const dialog = dialogRef;
      if (!dialog) {
        return;
      }
      if (open && !dialog.open) {
        dialog.showModal();
      } else if (!open && dialog.open) {
        dialog.close();
      }
    },
  );

  onSettled(() => {
    return () => {
      if (copyTimer) {
        clearTimeout(copyTimer);
      }
    };
  });

  function resetCopyLabel(): void {
    if (copyTimer) {
      clearTimeout(copyTimer);
    }
    copyTimer = setTimeout(() => setCopyLabel("Copy"), 1800);
  }

  async function handleCopy(): Promise<void> {
    const code = compilation().result?.code;
    if (!code) {
      return;
    }
    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(code);
      } else {
        fallbackCopy(code);
      }
      setCopyLabel("Copied");
    } catch {
      try {
        fallbackCopy(code);
        setCopyLabel("Copied");
      } catch {
        setCopyLabel("Copy failed");
      }
    }
    resetCopyLabel();
  }

  function handleDownload(): void {
    const code = compilation().result?.code;
    if (!code) {
      return;
    }
    const url = URL.createObjectURL(
      new Blob([code], { type: "text/javascript;charset=utf-8" }),
    );
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "gate-forest.js";
    document.body.append(anchor);
    anchor.click();
    anchor.remove();
    URL.revokeObjectURL(url);
  }

  return (
    <dialog
      ref={(element) => {
        dialogRef = element;
      }}
      class="compile-dialog"
      aria-labelledby="compile-title"
      onCancel={(event) => {
        event.preventDefault();
        props.onClose();
      }}
      onClose={() => {
        if (props.open) {
          props.onClose();
        }
      }}
    >
      <div class="compile-shell">
        <header class="compile-header">
          <div>
            <span class="eyebrow">Hard-mode export</span>
            <h2 id="compile-title">Compiled forest</h2>
            <p class="hint">
              The whole forest as one standalone ES module. Every circuit is a
              pure boolean function using bitwise gates, so outputs are exactly
              0 or 1.
            </p>
          </div>
          <button
            type="button"
            class="banner-close"
            aria-label="Close compiler"
            onClick={() => props.onClose()}
          >
            ×
          </button>
        </header>

        <Show when={compilation().error}>
          {(error) => (
            <div class="compile-error" role="alert">
              <strong>Could not compile the forest.</strong>
              <p>{error()}</p>
            </div>
          )}
        </Show>

        <Show when={compilation().result}>
          {(result) => (
            <>
              <div class="compile-stats">
                <div>
                  <span>Labels</span>
                  <strong>{result().labelCount}</strong>
                </div>
                <div>
                  <span>Circuits</span>
                  <strong>{result().circuitCount}</strong>
                </div>
                <div>
                  <span>Inputs</span>
                  <strong>{result().inputLength}</strong>
                </div>
                <div>
                  <span>Trained gates</span>
                  <strong>{result().gateCount}</strong>
                </div>
                <div>
                  <span>Emitted gates</span>
                  <strong>{result().emittedGateCount}</strong>
                </div>
                <div>
                  <span>Simplified</span>
                  <strong>{savedPercent()}%</strong>
                </div>
                <div>
                  <span>Module size</span>
                  <strong>{formatBytes(result().byteLength)}</strong>
                </div>
                <div>
                  <span>Lines</span>
                  <strong>{result().lineCount}</strong>
                </div>
              </div>

              <div class="compile-toolbar">
                <button
                  type="button"
                  class="button primary"
                  onClick={() => void handleCopy()}
                >
                  {copyLabel()}
                </button>
                <button
                  type="button"
                  class="button secondary"
                  onClick={handleDownload}
                >
                  Download .js
                </button>
                <span class="compile-filename">gate-forest.js</span>
                <button
                  type="button"
                  class="button ghost compile-close"
                  onClick={() => props.onClose()}
                >
                  Close
                </button>
              </div>

              <pre class="compile-code" tabindex="0">
                <code>{result().code}</code>
              </pre>
            </>
          )}
        </Show>
      </div>
    </dialog>
  );
}
