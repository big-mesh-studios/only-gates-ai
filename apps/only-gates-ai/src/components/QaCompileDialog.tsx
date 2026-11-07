import {
  createEffect,
  createMemo,
  createSignal,
  onSettled,
  Show,
} from "solid-js";

import {
  compileQuestionAnswerer,
  type CompiledQuestionAnswerer,
} from "../lib/qaCompiler.ts";
import { BUNDLED_QA_LEXICON } from "../lib/qaAssets.ts";
import type { QaDocument } from "../lib/qaRuntime.ts";

const PREVIEW_LIMIT = 60_000;

interface QaCompileDialogProps {
  readonly open: boolean;
  readonly documents: readonly QaDocument[];
  readonly suggestions?: readonly string[];
  readonly name: string;
  readonly createdAt?: string;
  readonly updatedAt?: string;
  readonly onClose: () => void;
}

interface CompilationState {
  readonly result: CompiledQuestionAnswerer | null;
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

export default function QaCompileDialog(props: QaCompileDialogProps) {
  let dialogRef: HTMLDialogElement | undefined;
  let copyTimer: ReturnType<typeof setTimeout> | undefined;

  const [copyLabel, setCopyLabel] = createSignal("Copy");

  const compilation = createMemo<CompilationState>(() => {
    if (!props.open) {
      return { result: null, error: null };
    }
    try {
      return {
        result: compileQuestionAnswerer({
          documents: props.documents,
          suggestions: props.suggestions,
          name: props.name,
          createdAt: props.createdAt,
          updatedAt: props.updatedAt,
        }),
        error: null,
      };
    } catch (error) {
      return { result: null, error: errorMessage(error) };
    }
  });

  const preview = createMemo(() => {
    const code = compilation().result?.code;
    if (!code) {
      return "";
    }
    if (code.length <= PREVIEW_LIMIT) {
      return code;
    }
    return `${code.slice(
      0,
      PREVIEW_LIMIT,
    )}\n\n// … preview truncated. Copy or download for the full module.`;
  });

  const truncated = createMemo(
    () => (compilation().result?.code.length ?? 0) > PREVIEW_LIMIT,
  );

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
    const result = compilation().result;
    if (!result) {
      return;
    }
    const url = URL.createObjectURL(
      new Blob([result.code], { type: "text/typescript;charset=utf-8" }),
    );
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = result.filename;
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
      aria-labelledby="qa-compile-title"
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
            <span class="eyebrow">Portable TypeScript export</span>
            <h2 id="qa-compile-title">QA model API</h2>
            <p class="hint">
              A standalone TypeScript module with the retrieval engine, your
              concepts, and the full{" "}
              {BUNDLED_QA_LEXICON.words.length.toLocaleString()}-word 64-bit
              lexicon. No dependencies, no DOM, no UI. Import <code>ask()</code>{" "}
              or <code>createAnswerer()</code> in any app or site.
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
              <strong>Could not compile the QA model.</strong>
              <p>{error()}</p>
            </div>
          )}
        </Show>

        <Show when={compilation().result}>
          {(result) => (
            <>
              <div class="compile-stats">
                <div>
                  <span>Concepts</span>
                  <strong>{result().conceptCount}</strong>
                </div>
                <div>
                  <span>Variants</span>
                  <strong>{result().variantCount}</strong>
                </div>
                <div>
                  <span>Question words</span>
                  <strong>{result().questionWordCount}</strong>
                </div>
                <div>
                  <span>Unique tokens</span>
                  <strong>{result().uniqueTokenCount}</strong>
                </div>
                <div>
                  <span>Lexicon</span>
                  <strong>{result().lexiconWordCount.toLocaleString()}</strong>
                </div>
                <div>
                  <span>Module size</span>
                  <strong>{formatBytes(result().byteLength)}</strong>
                </div>
                <div>
                  <span>Lines</span>
                  <strong>{result().lineCount.toLocaleString()}</strong>
                </div>
                <div>
                  <span>Engine</span>
                  <strong>v{result().engineVersion}</strong>
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
                  Download .ts
                </button>
                <span class="compile-filename">{result().filename}</span>
                <span class="compile-filename">
                  {result().fingerprint}
                </span>
                <button
                  type="button"
                  class="button ghost compile-close"
                  onClick={() => props.onClose()}
                >
                  Close
                </button>
              </div>

              <Show when={truncated()}>
                <p class="hint">
                  The preview is truncated to keep the dialog responsive. Copy
                  or download always includes the complete module.
                </p>
              </Show>

              <pre class="compile-code" tabindex="0">
                <code>{preview()}</code>
              </pre>
            </>
          )}
        </Show>
      </div>
    </dialog>
  );
}
