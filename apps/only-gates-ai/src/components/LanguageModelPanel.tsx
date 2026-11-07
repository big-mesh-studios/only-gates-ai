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
  parseQaPairs,
  tokenize,
  trainMiniLanguageModel,
  trainQuestionAnswerer,
  TrainingCancelledError,
  type CompletionResult,
  type MiniLanguageModel,
  type MiniQuestionAnswerer,
  type QaAnswer,
  type TrainingProgress,
} from "../lib/miniLanguageModel";

const SAMPLE_TEXT = `The gate forest learns from drawings. Draw a symbol and the forest will vote. Each circuit answers a simple question, and the forest combines the answers into one prediction. The small language model reads the text you give it. It learns which word tends to follow another word. Train it on a paragraph, then type a few words to see what it suggests. Binary word vectors give every word a short code of bits. The model compares those codes with popcount and integer arithmetic. No floating point is used while it predicts. Try training on your own notes, a story, or a list of sentences. The more text you provide, the more patterns it can learn. The forest watches the drawing, and the language model watches the words. Both use small binary circuits to make a prediction.`;

const SAMPLE_QA = `What is the gate forest?
A: A random forest of differentiable boolean circuits. Each circuit votes, and the forest combines the votes into one prediction.

What does the mini language model use?
A: Binary word vectors, ternary weights, popcount, and Q15 fixed-point softmax.

How is the mini language model trained?
A: Training uses floats and a straight-through estimator. Prediction is integer-only.

What can I train it on?
A: Paste any text, or upload a .txt or .md file. Everything stays in your browser.

What is a word vector?
A: A short code of bits that stands for a word. Similar words tend to share bits.

How does autocomplete work?
A: It looks at the last few words and scores every word in the vocabulary.`;

const MAX_FILE_BYTES = 2 * 1024 * 1024;

interface LocalStatus {
  readonly kind: "info" | "success" | "error";
  readonly text: string;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export default function LanguageModelPanel() {
  const [subMode, setSubMode] = createSignal<"complete" | "qa">("complete");
  const [trainingText, setTrainingText] = createSignal(SAMPLE_TEXT);
  const [qaText, setQaText] = createSignal(SAMPLE_QA);
  const [fileName, setFileName] = createSignal<string | null>(null);
  const [model, setModel] = createSignal<MiniLanguageModel | null>(null);
  const [modelVersion, setModelVersion] = createSignal(0);
  const [qaModel, setQaModel] = createSignal<MiniQuestionAnswerer | null>(null);
  const [training, setTraining] = createSignal(false);
  const [progress, setProgress] = createSignal<TrainingProgress | null>(null);
  const [draft, setDraft] = createSignal("the ");
  const [question, setQuestion] = createSignal("");
  const [qaAnswer, setQaAnswer] = createSignal<QaAnswer | null>(null);
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

  const qaStats = createMemo(() => {
    const parsed = parseQaPairs(qaText());
    let questionWords = 0;
    let bundled = 0;
    for (const pair of parsed.pairs) {
      const tokens = tokenize(pair.question);
      questionWords += tokens.length;
      for (const token of tokens) {
        if (hasBundledVector(token.text)) {
          bundled++;
        }
      }
    }
    return {
      pairs: parsed.pairs.length,
      errors: parsed.errors,
      questionWords,
      coverage: questionWords === 0 ? 0 : bundled / questionWords,
    };
  });

  const completion = createMemo<CompletionResult | null>(() => {
    modelVersion();
    const active = model();
    if (active === null) {
      return null;
    }
    try {
      const used = [
        ...new Set(tokenize(draft()).map((token) => token.text.toLowerCase())),
      ];
      return active.complete(draft(), 8, { avoid: used });
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
      if (subMode() === "qa") {
        setQaText(text);
        setQaModel(null);
        setQaAnswer(null);
      } else {
        setTrainingText(text);
      }
      setFileName(file.name);
      showStatus(
        "info",
        `Loaded ${file.name}. ${
          subMode() === "qa"
            ? "Learn the answers to use it."
            : "Train the model to use it."
        }`,
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

  function handleLearnAnswers(): void {
    const parsed = parseQaPairs(qaText());
    if (parsed.errors.length > 0) {
      showStatus("error", parsed.errors[0]!);
      return;
    }
    if (parsed.pairs.length === 0) {
      showStatus("error", "Add at least one question and answer pair.");
      return;
    }

    try {
      const trained = trainQuestionAnswerer(qaText());
      setQaModel(trained);
      const first = trained.questions[0] ?? "";
      setQuestion(first);
      setQaAnswer(first.length > 0 ? trained.ask(first) : null);
      showStatus(
        "success",
        `Indexed ${trained.stats.pairs} question${
          trained.stats.pairs === 1 ? "" : "s"
        }. Ask one on the right.`,
      );
    } catch (error) {
      showStatus("error", `Could not learn answers: ${errorMessage(error)}`);
    }
  }

  function askQuestion(preset?: string): void {
    const active = qaModel();
    if (active === null) {
      return;
    }
    const next = (preset ?? question()).trim();
    if (next.length === 0) {
      showStatus("error", "Type a question first.");
      return;
    }
    if (preset !== undefined) {
      setQuestion(preset);
    }
    const result = active.ask(next);
    setQaAnswer(result);
    if (result === null) {
      showStatus("error", "Add some answer pairs first.");
    }
  }

  function handleQuestionKeyDown(event: KeyboardEvent): void {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      askQuestion();
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
      <div class="language-toolbar">
        <div class="mode-switch" role="group" aria-label="Language model mode">
          <button
            type="button"
            class={{ active: subMode() === "complete" }}
            onClick={() => setSubMode("complete")}
          >
            Autocomplete
          </button>
          <button
            type="button"
            class={{ active: subMode() === "qa" }}
            onClick={() => setSubMode("qa")}
          >
            Question & answer
          </button>
        </div>
        <span class="muted">
          {subMode() === "qa"
            ? "Retrieval over the pairs you train"
            : "Next-word prediction"}
        </span>
      </div>

      <section class="card language-card">
        <div class="card-heading compact">
          <div>
            <span class="eyebrow">
              {subMode() === "qa" ? "Answer data" : "Training data"}
            </span>
            <h2>{subMode() === "qa" ? "Write Q&A pairs" : "Read a text"}</h2>
          </div>
          <div class="runtime-pill ready">
            <span class="runtime-dot" />
            Local only
          </div>
        </div>

        <p class="hint">
          {subMode() === "qa"
            ? "Write question and answer pairs. Separate pairs with a blank line, or use question => answer on one line. Q: and A: labels also work."
            : "Paste text or choose a file. The model learns which word tends to follow the previous few words. Everything is trained and inferred in this browser tab."}
        </p>

        <div class="file-row">
          <label class="button secondary file-button">
            Choose text file
            <input
              type="file"
              accept=".txt,.md,.markdown,text/plain,text/markdown"
              disabled={subMode() === "complete" && training()}
              onChange={(event) => void handleFile(event.currentTarget)}
            />
          </label>
          <Show when={fileName()}>
            {(name) => <span class="file-name">{name()}</span>}
          </Show>
        </div>

        <textarea
          class={`text-area ${
            subMode() === "qa" ? "qa-text" : "training-text"
          }`}
          rows={subMode() === "qa" ? 13 : 11}
          spellcheck={false}
          disabled={subMode() === "complete" && training()}
          value={subMode() === "qa" ? qaText() : trainingText()}
          onInput={(event) => {
            if (subMode() === "qa") {
              setQaText(event.currentTarget.value);
              setQaModel(null);
              setQaAnswer(null);
            } else {
              setTrainingText(event.currentTarget.value);
            }
            setFileName(null);
          }}
        />

        <Show
          when={subMode() === "qa"}
          fallback={
            <div class="text-meta">
              <span>
                {textStats().characters.toLocaleString()} chars ·{" "}
                {textStats().tokens.toLocaleString()} tokens ·{" "}
                {Math.round(textStats().coverage * 100)}% bundled vectors
              </span>
              <span>{bundledWordCount().toLocaleString()} word vectors</span>
            </div>
          }
        >
          <div class="text-meta">
            <span>
              {qaStats().pairs} pair{qaStats().pairs === 1 ? "" : "s"} ·{" "}
              {qaStats().questionWords} question words ·{" "}
              {Math.round(qaStats().coverage * 100)}% bundled vectors
            </span>
            <span>Blank line between pairs</span>
          </div>
        </Show>

        <Show when={subMode() === "qa" && qaStats().errors[0]}>
          {(error) => <p class="qa-error">{error()}</p>}
        </Show>

        <div class="button-row">
          <Show
            when={subMode() === "complete"}
            fallback={
              <>
                <button
                  type="button"
                  class="button primary"
                  disabled={
                    qaStats().pairs === 0 || qaStats().errors.length > 0
                  }
                  onClick={handleLearnAnswers}
                >
                  {qaStats().pairs === 1
                    ? "Learn 1 answer"
                    : `Learn ${qaStats().pairs} answers`}
                </button>
                <button
                  type="button"
                  class="button ghost"
                  onClick={() => {
                    setQaText(SAMPLE_QA);
                    setQaModel(null);
                    setQaAnswer(null);
                    setFileName(null);
                    setStatus(null);
                  }}
                >
                  Use sample
                </button>
              </>
            }
          >
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
              <button
                type="button"
                class="button danger"
                onClick={cancelTraining}
              >
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
          </Show>
        </div>

        <Show when={subMode() === "complete" && progress()}>
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
            <span class="eyebrow">
              {subMode() === "qa" ? "Ask" : "Autocomplete"}
            </span>
            <h2>
              {subMode() === "qa" ? "Ask a question" : "Continue a sentence"}
            </h2>
          </div>
          <div class="runtime-pill ready">
            <span class="runtime-dot" />
            {subMode() === "qa" ? "Answer index" : "Integer inference"}
          </div>
        </div>

        <Show
          when={subMode() === "complete"}
          fallback={
            <>
              <p class="hint">
                Type a question using words from the pairs. The model finds the
                closest trained question and returns its answer. This is
                retrieval, not free-form generation.
              </p>

              <textarea
                class="text-area question-text"
                rows="3"
                spellcheck={false}
                placeholder="For example: How does autocomplete work?"
                value={question()}
                onInput={(event) => {
                  setQuestion(event.currentTarget.value);
                  setQaAnswer(null);
                }}
                onKeyDown={handleQuestionKeyDown}
              />

              <div class="button-row">
                <button
                  type="button"
                  class="button primary"
                  disabled={
                    qaModel() === null || question().trim().length === 0
                  }
                  onClick={() => askQuestion()}
                >
                  Ask
                </button>
                <button
                  type="button"
                  class="button ghost"
                  onClick={() => {
                    setQuestion("");
                    setQaAnswer(null);
                  }}
                >
                  Clear
                </button>
              </div>

              <Show
                when={qaModel()}
                fallback={
                  <p class="empty-state">
                    Learn the answers on the left first. Each pair becomes a
                    binary question vector plus its answer.
                  </p>
                }
              >
                {(active) => (
                  <>
                    <div class="sample-questions">
                      <div class="section-heading">
                        <span>Try a trained question</span>
                        <span class="muted">
                          {active().stats.pairs} indexed
                        </span>
                      </div>
                      <div class="completion-list">
                        <For each={active().questions.slice(0, 4)}>
                          {(trained) => (
                            <button
                              type="button"
                              class="completion-chip"
                              onClick={() => askQuestion(trained)}
                            >
                              <span class="completion-word">{trained}</span>
                            </button>
                          )}
                        </For>
                      </div>
                    </div>

                    <Show
                      when={qaAnswer()}
                      fallback={
                        <p class="empty-state">
                          Ask a question to see the closest trained answer.
                        </p>
                      }
                    >
                      {(result) => (
                        <Show
                          when={result().matched}
                          fallback={
                            <div class="answer-block low">
                              <div class="answer-label">No close match</div>
                              <p class="answer-text">
                                I could not find a trained question close enough
                                to that. Closest: “{result().question}”.
                              </p>
                              <div class="answer-meta">
                                <span>
                                  {Math.round(result().confidence * 100)}% match
                                </span>
                                <span>Try words from a trained question</span>
                              </div>
                            </div>
                          }
                        >
                          <div class="answer-block">
                            <div class="answer-label">Answer</div>
                            <p class="answer-text">{result().answer}</p>
                            <div class="answer-meta">
                              <span>Matched “{result().question}”</span>
                              <span>
                                {Math.round(result().confidence * 100)}% match
                              </span>
                            </div>
                          </div>
                        </Show>
                      )}
                    </Show>
                  </>
                )}
              </Show>
            </>
          }
        >
          <p class="hint">
            Type a few words. Leave a space to predict the next word, or type
            part of a word to complete it. Click a suggestion, or press Tab for
            the first one. Words already in the draft are skipped so Tab
            cannot get stuck repeating itself.
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
                Train the mini model first. It only reads the text on the left,
                so nothing leaves your browser.
              </p>
            }
          >
            <Show
              when={completion()}
              fallback={
                <p class="empty-state">Type some text to get suggestions.</p>
              }
            >
              {(result) => (
                <div class="completion-block">
                  <div class="section-heading">
                    <span>{completionTitle(result())}</span>
                    <span class="muted">
                      click to use · Tab accepts the first
                    </span>
                  </div>
                  <Show
                    when={result().items.length > 0}
                    fallback={
                      <p class="empty-state">
                        No vocabulary word matches that prefix. Try another
                        word.
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
        </Show>
      </section>

      <section class="card language-stats">
        <div class="card-heading compact">
          <div>
            <span class="eyebrow">Model</span>
            <h2>What it learned</h2>
          </div>
          <Show when={subMode() === "complete" ? model() : qaModel()}>
            <div class="runtime-pill ready">
              <span class="runtime-dot" />
              {subMode() === "qa"
                ? "Answer index ready"
                : "Ternary weights ready"}
            </div>
          </Show>
        </div>

        <Show when={subMode() === "complete"}>
          <Show
            when={model()}
            fallback={
              <p class="empty-state">
                Train a model to see its vocabulary, training pairs, and
                accuracy here.
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
        </Show>

        <Show when={subMode() === "qa"}>
          <Show
            when={qaModel()}
            fallback={
              <p class="empty-state">
                Learn the answer pairs to see how many questions are indexed and
                which words have bundled vectors.
              </p>
            }
          >
            {(active) => (
              <>
                <div class="stat-row">
                  <div class="stat">
                    <span>Indexed pairs</span>
                    <strong>{active().stats.pairs}</strong>
                  </div>
                  <div class="stat">
                    <span>Question words</span>
                    <strong>{active().stats.questionWords}</strong>
                  </div>
                  <div class="stat">
                    <span>Answer vector</span>
                    <strong>{active().stats.vectorBits} bits</strong>
                  </div>
                  <div class="stat">
                    <span>Vector coverage</span>
                    <strong>{Math.round(active().stats.coverage * 100)}%</strong>
                  </div>
                </div>

                <div class="vocab-block">
                  <div class="section-heading">
                    <span>Trained questions</span>
                    <span class="muted">
                      {active().stats.vectorBits}-bit vectors · shared words
                    </span>
                  </div>
                  <div class="vocab-list">
                    <For each={active().questions}>
                      {(trained) => (
                        <span class="vocab-chip">{trained}</span>
                      )}
                    </For>
                  </div>
                </div>
              </>
            )}
          </Show>
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
          using 64 of the 800 bit-vector dimensions. Unknown words get a
          deterministic hash vector.{" "}
          <Show
            when={subMode() === "qa"}
            fallback={
              <>
                Training uses floats with a straight-through estimator;
                prediction uses popcount, integer weights, and Q15 fixed-point
                softmax only.
              </>
            }
          >
            Answers are retrieved from the pairs you wrote by comparing shared
            words and binary question vectors; no answer text is generated.
          </Show>
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
