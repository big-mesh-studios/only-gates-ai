import {
  createMemo,
  createSignal,
  For,
  onCleanup,
  Show,
} from "solid-js";

import {
  bundledWordCount,
  DEFAULT_EPOCHS,
  hasBundledVector,
  tokenize,
  trainMiniLanguageModel,
  TrainingCancelledError,
  type CompletionResult,
  type MiniLanguageModel,
  type TrainingProgress,
} from "../lib/miniLanguageModel";

const SAMPLE_TEXT = `The gate forest learns from drawings. Draw a symbol and the forest will vote. Each circuit answers a simple question, and the forest combines the answers into one prediction. The small language model reads the text you give it. It learns which word tends to follow another word. Train it on a paragraph, then type a few words to see what it suggests. Binary word vectors give every word a short code of bits. The model compares those codes with popcount and integer arithmetic. No floating point is used while it predicts. Try training on your own notes, a story, or a list of sentences. The more text you provide, the more patterns it can learn. The forest watches the drawing, and the language model watches the words. Both use small binary circuits to make a prediction.`;

const MAX_FILE_BYTES = 2 * 1024 * 1024;

interface LocalStatus {
  readonly kind: "info" | "success" | "error";
  readonly text: string;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export default function LanguageModelPanel() {
  const [trainingText, setTrainingText] = createSignal(SAMPLE_TEXT);
  const [fileName, setFileName] = createSignal<string | null>(null);
  const [model, setModel] = createSignal<MiniLanguageModel | null>(null);
  const [modelVersion, setModelVersion] = createSignal(0);
  const [training, setTraining] = createSignal(false);
  const [progress, setProgress] = createSignal<TrainingProgress | null>(null);
  const [draft, setDraft] = createSignal("the ");
  const [status, setStatus] = createSignal<LocalStatus | null>(null);

  let disposed = false;
  let trainingToken: { cancelled: boolean } | null = null;
  let statusTimer: ReturnType<typeof setTimeout> | undefined;

  onCleanup(() => {
    disposed = true;
    if (trainingToken !== null) {
      trainingToken.cancelled = true;
    }
    if (statusTimer !== undefined) {
      clearTimeout(statusTimer);
    }
  });

  const textStats = createMemo(() => {
    const text = trainingText();
    const tokens = tokenize(text);
    let bundled = 0;
    for (const token of tokens) {
      if (hasBundledVector(token.text)) {
        bundled++;
      }
    }
    return {
      characters: text.length,
      tokens: tokens.length,
      coverage: tokens.length === 0 ? 0 : bundled / tokens.length,
    };
  });

  const completion = createMemo<CompletionResult | null>(() => {
    modelVersion();
    const active = model();
    if (active === null) {
      return null;
    }
    try {
      return active.complete(draft(), 8);
    } catch {
      return null;
    }
  });

  function showStatus(kind: LocalStatus["kind"], text: string): void {
    if (statusTimer !== undefined) {
      clearTimeout(statusTimer);
    }
    setStatus({ kind, text });
    statusTimer = setTimeout(() => setStatus(null), 8000);
  }

  async function handleFile(input: HTMLInputElement): Promise<void> {
    const file = input.files?.[0];
    input.value = "";
    if (!file) {
      return;
    }
    if (file.size > MAX_FILE_BYTES) {
      showStatus("error", "Choose a text file smaller than 2 MB.");
      return;
    }

    try {
      const text = await file.text();
      if (disposed) {
        return;
      }
      if (text.trim().length === 0) {
        showStatus("error", "That file is empty.");
        return;
      }
      setTrainingText(text);
      setFileName(file.name);
      showStatus(
        "info",
        `Loaded ${file.name}. Train the model to use it.`,
      );
    } catch (error) {
      showStatus("error", `Could not read the file: ${errorMessage(error)}`);
    }
  }

  async function handleTrain(): Promise<void> {
    if (training()) {
      return;
    }
    const text = trainingText().trim();
    if (tokenize(text).length < 8) {
      showStatus("error", "Add at least eight words before training.");
      return;
    }

    const token = { cancelled: false };
    trainingToken = token;
    setTraining(true);
    setProgress(null);
    showStatus("info", "Training on your text. This runs locally in the browser.");

    try {
      const trained = await trainMiniLanguageModel(text, {
        epochs: DEFAULT_EPOCHS,
        shouldCancel: () => token.cancelled,
        onProgress: (next) => {
          if (!disposed) {
            setProgress(next);
          }
        },
      });
      if (disposed || token.cancelled) {
        return;
      }
      setModel(trained);
      setModelVersion((version) => version + 1);
      setProgress(null);
      setDraft("the ");
      showStatus(
        "success",
        `Trained on ${trained.stats.samples} word pairs with ${trained.stats.parameters.toLocaleString()} ternary weights.`,
      );
    } catch (error) {
      if (error instanceof TrainingCancelledError || token.cancelled) {
        showStatus("info", "Training cancelled.");
      } else {
        showStatus("error", `Training failed: ${errorMessage(error)}`);
      }
    } finally {
      if (trainingToken === token) {
        trainingToken = null;
      }
      if (!disposed) {
        setTraining(false);
        setProgress(null);
      }
    }
  }

  function cancelTraining(): void {
    if (trainingToken !== null) {
      trainingToken.cancelled = true;
      showStatus("info", "Stopping after the current batch…");
    }
  }

  function acceptSuggestion(word: string): void {
    const text = draft();
    const trailing = /([A-Za-z0-9']+)\s*$/.exec(text);
    if (
      trailing !== null &&
      trailing.index + trailing[1]!.length === text.length
    ) {
      setDraft(`${text.slice(0, trailing.index)}${word} `);
      return;
    }
    const trimmed = text.replace(/\s+$/, "");
    setDraft(`${trimmed.length > 0 ? `${trimmed} ` : ""}${word} `);
  }

  function handleDraftKeyDown(event: KeyboardEvent): void {
    if (event.key !== "Tab") {
      return;
    }
    const first = completion()?.items[0];
    if (first) {
      event.preventDefault();
      acceptSuggestion(first.word);
    }
  }

  return (
    <main class="workspace language-workspace">
      <section class="card language-card">
        <div class="card-heading compact">
          <div>
            <span class="eyebrow">Training data</span>
            <h2>Read a text</h2>
          </div>
          <div class="runtime-pill ready">
            <span class="runtime-dot" />
            Local only
          </div>
        </div>

        <p class="hint">
          Paste text or choose a file. The model learns which word tends to
          follow the previous few words. Everything is trained and inferred in
          this browser tab.
        </p>

        <div class="file-row">
          <label class="button secondary file-button">
            Choose text file
            <input
              type="file"
              accept=".txt,.md,.markdown,text/plain,text/markdown"
              disabled={training()}
              onChange={(event) => void handleFile(event.currentTarget)}
            />
          </label>
          <Show when={fileName()}>
            {(name) => <span class="file-name">{name()}</span>}
          </Show>
        </div>

        <textarea
          class="text-area training-text"
          rows="11"
          spellcheck={false}
          disabled={training()}
          value={trainingText()}
          onInput={(event) => {
            setTrainingText(event.currentTarget.value);
            setFileName(null);
          }}
        />

        <div class="text-meta">
          <span>
            {textStats().characters.toLocaleString()} chars ·{" "}
            {textStats().tokens.toLocaleString()} tokens ·{" "}
            {Math.round(textStats().coverage * 100)}% bundled vectors
          </span>
          <span>{bundledWordCount().toLocaleString()} word vectors</span>
        </div>

        <div class="button-row">
          <button
            type="button"
            class="button primary"
            disabled={training() || textStats().tokens < 8}
            onClick={() => void handleTrain()}
          >
            {training()
              ? "Training…"
              : `Train ${DEFAULT_EPOCHS} epochs`}
          </button>
          <Show when={training()}>
            <button type="button" class="button danger" onClick={cancelTraining}>
              Cancel
            </button>
          </Show>
          <button
            type="button"
            class="button ghost"
            disabled={training()}
            onClick={() => {
              setTrainingText(SAMPLE_TEXT);
              setFileName(null);
              setStatus(null);
            }}
          >
            Use sample
          </button>
        </div>

        <Show when={progress()}>
          {(current) => (
            <div class="progress-block">
              <div class="progress-label">
                <span>
                  Epoch {current().epoch} / {current().epochs} · loss{" "}
                  {current().loss.toFixed(3)}
                </span>
                <span>{Math.round(current().fraction * 100)}%</span>
              </div>
              <div class="progress-track">
                <div
                  class="progress-fill"
                  style={{ width: `${current().fraction * 100}%` }}
                />
              </div>
            </div>
          )}
        </Show>

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
      </section>

      <section class="card language-card">
        <div class="card-heading compact">
          <div>
            <span class="eyebrow">Autocomplete</span>
            <h2>Continue a sentence</h2>
          </div>
          <div class="runtime-pill ready">
            <span class="runtime-dot" />
            Integer inference
          </div>
        </div>

        <p class="hint">
          Type a few words. Leave a space to predict the next word, or type part
          of a word to complete it. Click a suggestion, or press Tab for the
          first one.
        </p>

        <textarea
          class="text-area draft-text"
          rows="9"
          spellcheck={false}
          placeholder="Start typing, for example: the forest "
          value={draft()}
          onInput={(event) => setDraft(event.currentTarget.value)}
          onKeyDown={handleDraftKeyDown}
        />

        <div class="button-row">
          <button
            type="button"
            class="button ghost"
            onClick={() => setDraft("")}
          >
            Clear
          </button>
          <button
            type="button"
            class="button secondary"
            disabled={draft().trim().length === 0}
            onClick={() => setDraft(`${draft().replace(/\s+$/, "")} `)}
          >
            Add a space
          </button>
        </div>

        <Show
          when={model() !== null}
          fallback={
            <p class="empty-state">
              Train the mini model first. It only reads the text on the left, so
              nothing leaves your browser.
            </p>
          }
        >
          <Show
            when={completion()}
            fallback={<p class="empty-state">Type some text to get suggestions.</p>}
          >
            {(result) => (
              <div class="completion-block">
                <div class="section-heading">
                  <span>{completionTitle(result())}</span>
                  <span class="muted">click to use · Tab accepts the first</span>
                </div>
                <Show
                  when={result().items.length > 0}
                  fallback={
                    <p class="empty-state">
                      No vocabulary word matches that prefix. Try another word.
                    </p>
                  }
                >
                  <div class="completion-list">
                    <For each={result().items}>
                      {(item) => (
                        <button
                          type="button"
                          class="completion-chip"
                          onClick={() => acceptSuggestion(item.word)}
                        >
                          <span class="completion-word">{item.word}</span>
                          <span class="completion-score">
                            {item.probability !== null
                              ? `${(item.probability * 100).toFixed(1)}%`
                              : item.source === "lexicon"
                                ? "vector"
                                : `${item.count}×`}
                          </span>
                        </button>
                      )}
                    </For>
                  </div>
                </Show>
              </div>
            )}
          </Show>
        </Show>
      </section>

      <section class="card language-stats">
        <div class="card-heading compact">
          <div>
            <span class="eyebrow">Model</span>
            <h2>What it learned</h2>
          </div>
          <Show when={model()}>
            <div class="runtime-pill ready">
              <span class="runtime-dot" />
              Ternary weights ready
            </div>
          </Show>
        </div>

        <Show
          when={model()}
          fallback={
            <p class="empty-state">
              Train a model to see its vocabulary, training pairs, and accuracy
              here.
            </p>
          }
        >
          {(active) => (
            <>
              <div class="stat-row">
                <div class="stat">
                  <span>Vocabulary</span>
                  <strong>{active().stats.vocabulary}</strong>
                </div>
                <div class="stat">
                  <span>Training pairs</span>
                  <strong>{active().stats.samples.toLocaleString()}</strong>
                </div>
                <div class="stat">
                  <span>Ternary weights</span>
                  <strong>{active().stats.parameters.toLocaleString()}</strong>
                </div>
                <div class="stat">
                  <span>Train accuracy</span>
                  <strong>{(active().stats.accuracy * 100).toFixed(1)}%</strong>
                </div>
                <div class="stat">
                  <span>Average loss</span>
                  <strong>{active().stats.averageLoss.toFixed(3)}</strong>
                </div>
                <div class="stat">
                  <span>Vector coverage</span>
                  <strong>{Math.round(active().stats.coverage * 100)}%</strong>
                </div>
              </div>

              <div class="vocab-block">
                <div class="section-heading">
                  <span>Most frequent words</span>
                  <span class="muted">
                    {active().stats.vectorBits}-bit keys · context{" "}
                    {active().stats.contextWindow}
                  </span>
                </div>
                <div class="vocab-list">
                  <For each={active().stats.vocabularyWords.slice(0, 28)}>
                    {(word) => <span class="vocab-chip">{word}</span>}
                  </For>
                  <Show when={active().stats.vocabularyWords.length > 28}>
                    <span class="vocab-chip">
                      +{active().stats.vocabularyWords.length - 28} more
                    </span>
                  </Show>
                </div>
              </div>
            </>
          )}
        </Show>

        <p class="hint language-attribution">
          Word vectors come from{" "}
          <a
            href="https://github.com/agnusmaximus/Word2Bits"
            target="_blank"
            rel="noreferrer"
          >
            Word2Bits
          </a>{" "}
          (Apache-2.0): a subset of {bundledWordCount().toLocaleString()} words,
          using 64 of the 800 bit-vector dimensions. Unknown words get a deterministic hash vector. Training
          uses floats with a straight-through estimator; prediction uses
          popcount, integer weights, and Q15 fixed-point softmax only.
        </p>
      </section>
    </main>
  );
}

function completionTitle(result: CompletionResult): string {
  if (result.mode === "prefix") {
    return `Words starting with “${result.prefix}”`;
  }
  if (result.mode === "next") {
    return result.context.length > 0
      ? `Likely after “${result.context}”`
      : "Likely next words";
  }
  return "Frequent words";
}
