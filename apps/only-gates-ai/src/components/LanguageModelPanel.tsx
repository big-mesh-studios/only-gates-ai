import {
  createEffect,
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
  trainQuestionAnswererFromDocuments,
  TrainingCancelledError,
  type CompletionResult,
  type MiniLanguageModel,
  type MiniQuestionAnswerer,
  type TrainingProgress,
} from "../lib/miniLanguageModel";
import {
  SOLID2_QA_DOCUMENTS,
  SOLID2_QA_SUGGESTIONS,
} from "../assets/solid2-qa";

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

interface ChatMessage {
  readonly id: number;
  readonly role: "user" | "assistant";
  readonly text: string;
  readonly matched?: boolean;
  readonly match?: string;
  readonly confidence?: number;
  readonly source?: string;
  readonly tags?: readonly string[];
}

type QaPreset = "custom" | "solid2";

interface AnswerSegment {
  readonly kind: "text" | "code";
  readonly text: string;
  readonly language?: string;
}

/*
 * The answers are written with small fenced code blocks. Rendering them as
 * plain text would show the backticks, so split the answer into prose and code
 * segments. This is deliberately not a full Markdown parser.
 */
function splitAnswer(text: string): AnswerSegment[] {
  const segments: AnswerSegment[] = [];
  const fence = /```([^\n`]*)\n([\s\S]*?)```/g;
  let cursor = 0;
  let match: RegExpExecArray | null;

  while ((match = fence.exec(text)) !== null) {
    const prose = text.slice(cursor, match.index).trim();
    if (prose.length > 0) {
      segments.push({ kind: "text", text: prose });
    }
    const language = (match[1] ?? "").trim();
    segments.push({
      kind: "code",
      text: (match[2] ?? "").replace(/\n$/, ""),
      language,
    });
    cursor = match.index + match[0].length;
  }

  const tail = text.slice(cursor).trim();
  if (tail.length > 0) {
    segments.push({ kind: "text", text: tail });
  }

  return segments.length > 0 ? segments : [{ kind: "text", text }];
}

function splitInlineCode(text: string): AnswerSegment[] {
  const parts: AnswerSegment[] = [];
  const code = /`([^`\n]+)`/g;
  let cursor = 0;
  let match: RegExpExecArray | null;

  while ((match = code.exec(text)) !== null) {
    if (match.index > cursor) {
      parts.push({ kind: "text", text: text.slice(cursor, match.index) });
    }
    parts.push({ kind: "code", text: match[1] ?? "" });
    cursor = match.index + match[0].length;
  }
  if (cursor < text.length) {
    parts.push({ kind: "text", text: text.slice(cursor) });
  }

  return parts.length > 0 ? parts : [{ kind: "text", text }];
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export default function LanguageModelPanel() {
  const [subMode, setSubMode] = createSignal<"complete" | "qa">("complete");
  const [trainingText, setTrainingText] = createSignal(SAMPLE_TEXT);
  const [qaText, setQaText] = createSignal(SAMPLE_QA);
  const [qaPreset, setQaPreset] = createSignal<QaPreset>("solid2");
  const [fileName, setFileName] = createSignal<string | null>(null);
  const [model, setModel] = createSignal<MiniLanguageModel | null>(null);
  const [modelVersion, setModelVersion] = createSignal(0);
  const [qaModel, setQaModel] = createSignal<MiniQuestionAnswerer | null>(
    trainQuestionAnswererFromDocuments(SOLID2_QA_DOCUMENTS),
  );
  const [training, setTraining] = createSignal(false);
  const [progress, setProgress] = createSignal<TrainingProgress | null>(null);
  const [draft, setDraft] = createSignal("the ");
  const [question, setQuestion] = createSignal("");
  const [chat, setChat] = createSignal<readonly ChatMessage[]>([]);
  const [status, setStatus] = createSignal<LocalStatus | null>(null);

  let disposed = false;
  let trainingToken: { cancelled: boolean } | null = null;
  let statusTimer: ReturnType<typeof setTimeout> | undefined;
  let nextMessageId = 1;
  let chatLog: HTMLDivElement | undefined;
  let questionInput: HTMLTextAreaElement | undefined;

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

  const solid2Stats = createMemo(() => {
    let questionWords = 0;
    let bundled = 0;
    let variants = 0;
    for (const document of SOLID2_QA_DOCUMENTS) {
      for (const question of document.questions) {
        variants++;
        const tokens = tokenize(question);
        questionWords += tokens.length;
        for (const token of tokens) {
          if (hasBundledVector(token.text)) {
            bundled++;
          }
        }
      }
    }
    return {
      concepts: SOLID2_QA_DOCUMENTS.length,
      variants,
      questionWords,
      coverage: questionWords === 0 ? 0 : bundled / questionWords,
    };
  });

  const qaStats = createMemo(() => {
    if (qaPreset() === "solid2") {
      const stats = solid2Stats();
      return {
        pairs: stats.variants,
        concepts: stats.concepts,
        variants: stats.variants,
        errors: [] as string[],
        questionWords: stats.questionWords,
        coverage: stats.coverage,
      };
    }

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
      concepts: parsed.pairs.length,
      variants: parsed.pairs.length,
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

  createEffect(
    () => chat(),
    () => {
      if (chatLog !== undefined) {
        chatLog.scrollTop = chatLog.scrollHeight;
      }
    },
  );

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
        setQaPreset("custom");
        setQaText(text);
        setQaModel(null);
        setChat([]);
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

  function rememberAnswerer(trained: MiniQuestionAnswerer): void {
    setQaModel(trained);
    setChat([]);
    setQuestion("");
  }

  function handleLearnAnswers(): void {
    if (qaPreset() === "solid2") {
      try {
        const trained = trainQuestionAnswererFromDocuments(
          SOLID2_QA_DOCUMENTS,
        );
        rememberAnswerer(trained);
        showStatus(
          "success",
          `Indexed ${trained.stats.concepts} concepts and ${trained.stats.variants} question variants.`,
        );
        queueMicrotask(() => questionInput?.focus());
      } catch (error) {
        showStatus("error", `Could not index the docs: ${errorMessage(error)}`);
      }
      return;
    }

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
      rememberAnswerer(trained);
      showStatus(
        "success",
        `Indexed ${trained.stats.pairs} question${
          trained.stats.pairs === 1 ? "" : "s"
        }. Ask one on the right.`,
      );
      queueMicrotask(() => questionInput?.focus());
    } catch (error) {
      showStatus("error", `Could not learn answers: ${errorMessage(error)}`);
    }
  }

  function loadSolidQa(): void {
    setQaPreset("solid2");
    setQaModel(null);
    setChat([]);
    setQuestion("");
    setFileName(null);
    setStatus(null);
    try {
      const trained = trainQuestionAnswererFromDocuments(SOLID2_QA_DOCUMENTS);
      rememberAnswerer(trained);
      showStatus(
        "success",
        `Loaded ${trained.stats.concepts} Solid 2.0 concepts (${trained.stats.variants} question variants).`,
      );
      queueMicrotask(() => questionInput?.focus());
    } catch (error) {
      showStatus(
        "error",
        `Could not load Solid 2.0 Q&A: ${errorMessage(error)}`,
      );
    }
  }

  function useCustomPairs(): void {
    setQaPreset("custom");
    setQaModel(null);
    setChat([]);
    setQuestion("");
    setFileName(null);
    setStatus(null);
  }

  function askQuestion(preset?: string): void {
    const active = qaModel();
    if (active === null) {
      return;
    }
    const next = (preset ?? question()).trim();
    if (next.length === 0) {
      return;
    }

    const result = active.ask(next);
    const reply: ChatMessage =
      result === null
        ? {
            id: nextMessageId++,
            role: "assistant",
            text: "I need some words to match. Try asking with a few words from a trained question.",
            matched: false,
          }
        : {
            id: nextMessageId++,
            role: "assistant",
            text: result.matched
              ? result.answer
              : `I couldn't find a trained question close enough to that. Closest: “${result.question}”.`,
            matched: result.matched,
            match: result.question,
            confidence: result.confidence,
            // A rejected match has no trustworthy source, so do not offer one.
            source: result.matched ? result.source : undefined,
            tags: result.matched ? result.tags : undefined,
          };

    setChat((current) => [
      ...current,
      { id: nextMessageId++, role: "user", text: next },
      reply,
    ]);
    setQuestion("");
    questionInput?.focus();
  }

  function handleQuestionKeyDown(event: KeyboardEvent): void {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
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
            ? qaPreset() === "solid2"
              ? "Retrieval over a curated Solid 2.0 corpus"
              : "Retrieval over the pairs you train"
            : "Next-word prediction"}
        </span>
      </div>

      <section class="card language-card">
        <div class="card-heading compact">
          <div>
            <span class="eyebrow">
              {subMode() === "qa"
                ? qaPreset() === "solid2"
                  ? "Bundled dataset"
                  : "Answer data"
                : "Training data"}
            </span>
            <h2>
              {subMode() === "qa"
                ? qaPreset() === "solid2"
                  ? "Solid 2.0 Q&A"
                  : "Write Q&A pairs"
                : "Read a text"}
            </h2>
          </div>
          <div class="runtime-pill ready">
            <span class="runtime-dot" />
            Local only
          </div>
        </div>

        <p class="hint">
          {subMode() === "qa"
            ? qaPreset() === "solid2"
              ? "Curated from the official Solid 2.0 migration guide and reactivity RFCs. Each answer links back to the source section. Ask a question on the right, or switch to custom pairs to train on your own text."
              : "Write question and answer pairs. Separate pairs with a blank line, or use question => answer on one line. Q: and A: labels also work."
            : "Paste text or choose a file. The model learns which word tends to follow the previous few words. Everything is trained and inferred in this browser tab."}
        </p>

        <Show when={subMode() === "complete" || qaPreset() === "custom"}>
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
        </Show>

        <Show
          when={subMode() === "complete" || qaPreset() === "custom"}
          fallback={
            <div class="qa-preset">
              <div class="qa-preset-copy">
                <strong>
                  {solid2Stats().concepts} concepts ·{" "}
                  {solid2Stats().variants} question variants
                </strong>
                <span>
                  Answers are retrieval-only and written for this demo. Each
                  one links to the official Solid 2.0 documentation.
                </span>
              </div>
              <a
                href="https://github.com/solidjs/solid/tree/next/documentation/solid-2.0"
                target="_blank"
                rel="noreferrer"
              >
                Browse the source docs
              </a>
            </div>
          }
        >
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
                setQaPreset("custom");
                setQaText(event.currentTarget.value);
                setQaModel(null);
                setChat([]);
              } else {
                setTrainingText(event.currentTarget.value);
              }
              setFileName(null);
            }}
          />
        </Show>

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
              {qaPreset() === "solid2" ? (
                <>
                  {qaStats().concepts} concepts · {qaStats().variants} question
                  variants
                </>
              ) : (
                <>
                  {qaStats().pairs} pair{qaStats().pairs === 1 ? "" : "s"}
                </>
              )}{" "}
              · {qaStats().questionWords} question words ·{" "}
              {Math.round(qaStats().coverage * 100)}% bundled vectors
            </span>
            <span>
              {qaPreset() === "solid2"
                ? "Official Solid 2.0 docs"
                : "Blank line between pairs"}
            </span>
          </div>
        </Show>

        <Show when={subMode() === "qa" && qaStats().errors[0]}>
          {(error) => <p class="qa-error">{error()}</p>}
        </Show>

        <div class="button-row">
          <Show
            when={subMode() === "complete"}
            fallback={
              <Show
                when={qaPreset() === "solid2"}
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
                      onClick={loadSolidQa}
                    >
                      Use Solid 2.0 Q&A
                    </button>
                    <button
                      type="button"
                      class="button ghost"
                      onClick={() => {
                        setQaPreset("custom");
                        setQaText(SAMPLE_QA);
                        setQaModel(null);
                        setChat([]);
                        setQuestion("");
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
                  onClick={handleLearnAnswers}
                >
                  Reindex {qaStats().concepts} concepts
                </button>
                <button
                  type="button"
                  class="button ghost"
                  onClick={useCustomPairs}
                >
                  Use custom pairs
                </button>
              </Show>
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
              {subMode() === "qa" ? "Chat" : "Autocomplete"}
            </span>
            <h2>
              {subMode() === "qa" ? "Ask a question" : "Continue a sentence"}
            </h2>
          </div>
          <div
            class={`runtime-pill${
              subMode() !== "qa" || qaModel() !== null ? " ready" : ""
            }`}
          >
            <span class="runtime-dot" />
            {subMode() === "qa"
              ? qaModel() !== null
                ? "Answer index ready"
                : "Not learned yet"
              : "Integer inference"}
          </div>
        </div>

        <Show
          when={subMode() === "complete"}
          fallback={
            <>
              <p class="hint">
                {qaPreset() === "solid2"
                  ? "Ask about Solid 2.0. Answers are retrieved from a curated set of concepts and link back to the official docs."
                  : "Ask with words from the pairs. Your message appears in the chat and the closest trained answer appears underneath it."}
              </p>

              <div
                class="chat-log"
                role="log"
                aria-live="polite"
                aria-label="Question and answer chat"
                ref={(element) => {
                  chatLog = element;
                  element.scrollTop = element.scrollHeight;
                }}
              >
                <Show
                  when={chat().length > 0}
                  fallback={
                    <Show
                      when={qaModel()}
                      fallback={
                        <div class="chat-empty">
                          <strong>No answers learned yet</strong>
                          <span>
                            Write question and answer pairs on the left, then
                            click “Learn answers”.
                          </span>
                        </div>
                      }
                    >
                      {(active) => (
                        <div class="chat-empty">
                          <strong>
                            {qaPreset() === "solid2"
                              ? `${active().stats.concepts} Solid 2.0 concepts · ${active().stats.variants} question variants`
                              : `${active().stats.pairs} question${
                                  active().stats.pairs === 1 ? "" : "s"
                                } indexed`}
                          </strong>
                          <span>
                            Ask below, or start with a trained question:
                          </span>
                          <div class="completion-list">
                            <For
                              each={
                                qaPreset() === "solid2"
                                  ? SOLID2_QA_SUGGESTIONS.slice(0, 6)
                                  : active().suggestions.slice(0, 4)
                              }
                            >
                              {(trained) => (
                                <button
                                  type="button"
                                  class="completion-chip"
                                  onClick={() => askQuestion(trained)}
                                >
                                  <span class="completion-word">
                                    {trained}
                                  </span>
                                </button>
                              )}
                            </For>
                          </div>
                        </div>
                      )}
                    </Show>
                  }
                >
                  <For each={chat()}>
                    {(message) => (
                      <div class={`chat-message ${message.role}`}>
                        <div
                          class={`chat-bubble${
                            message.matched === false ? " low" : ""
                          }`}
                        >
                          <span class="chat-role">
                            {message.role === "user" ? "You" : "Model"}
                          </span>
                          <For each={splitAnswer(message.text)}>
                            {(segment) =>
                              segment.kind === "code" ? (
                                <pre class="chat-code">
                                  <code>{segment.text}</code>
                                </pre>
                              ) : (
                                <p>
                                  <For each={splitInlineCode(segment.text)}>
                                    {(part) =>
                                      part.kind === "code" ? (
                                        <code class="chat-inline-code">
                                          {part.text}
                                        </code>
                                      ) : (
                                        part.text
                                      )
                                    }
                                  </For>
                                </p>
                              )
                            }
                          </For>
                          <Show
                            when={
                              message.role === "assistant" &&
                              message.confidence !== undefined
                            }
                          >
                            <div class="chat-meta">
                              <span>
                                {message.matched ? "Matched" : "Closest"} “
                                {message.match}”
                              </span>
                              <span>
                                {Math.round((message.confidence ?? 0) * 100)}%
                                match
                              </span>
                            </div>
                          </Show>
                          <Show
                            when={message.role === "assistant" && message.source}
                          >
                            <a
                              class="chat-source"
                              href={message.source}
                              target="_blank"
                              rel="noreferrer"
                            >
                              Read the source docs
                            </a>
                          </Show>
                        </div>
                      </div>
                    )}
                  </For>
                </Show>
              </div>

              <div class="chat-composer">
                <textarea
                  ref={(element) => {
                    questionInput = element;
                  }}
                  class="text-area chat-input"
                  aria-label="Ask a question"
                  rows="2"
                  spellcheck={false}
                  disabled={qaModel() === null}
                  placeholder={
                    qaModel() === null
                      ? "Learn answers first…"
                      : "Ask a question…"
                  }
                  value={question()}
                  onInput={(event) => setQuestion(event.currentTarget.value)}
                  onKeyDown={handleQuestionKeyDown}
                />
                <button
                  type="button"
                  class="button primary chat-send"
                  disabled={
                    qaModel() === null || question().trim().length === 0
                  }
                  onClick={() => askQuestion()}
                >
                  Send
                </button>
              </div>
              <p class="chat-hint">
                Enter to send · Shift+Enter for a new line
              </p>
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
                    <span>Concepts</span>
                    <strong>{active().stats.concepts}</strong>
                  </div>
                  <div class="stat">
                    <span>Question variants</span>
                    <strong>{active().stats.variants}</strong>
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
                    <For each={active().questions.slice(0, 28)}>
                      {(trained) => (
                        <span class="vocab-chip">{trained}</span>
                      )}
                    </For>
                    <Show when={active().questions.length > 28}>
                      <span class="vocab-chip">
                        +{active().questions.length - 28} more
                      </span>
                    </Show>
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
            {qaPreset() === "solid2"
              ? "Answers are curated from the official Solid 2.0 migration guide and RFCs, then retrieved by shared words and binary question vectors. No answer text is generated."
              : "Answers are retrieved from the pairs you wrote by comparing shared words and binary question vectors; no answer text is generated."}
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
