import { For, Show, createEffect, createMemo, createSignal } from "solid-js";

import {
  LUAU_QA_DOCUMENTS,
  LUAU_QA_SUGGESTIONS,
} from "../assets/luau-qa";
import { splitAnswer, splitInlineCode } from "../lib/chatFormat";
import {
  trainQuestionAnswererFromDocuments,
  type MiniQuestionAnswerer,
} from "../lib/miniLanguageModel";

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

function topicLabel(tag: string): string {
  return tag
    .replace(/-/g, " ")
    .replace(/^\w/, (letter) => letter.toUpperCase());
}

export default function LuauQaPanel() {
  const answerer: MiniQuestionAnswerer =
    trainQuestionAnswererFromDocuments(LUAU_QA_DOCUMENTS);
  const [question, setQuestion] = createSignal("");
  const [chat, setChat] = createSignal<readonly ChatMessage[]>([]);

  let nextMessageId = 1;
  let chatLog: HTMLDivElement | undefined;
  let questionInput: HTMLTextAreaElement | undefined;

  const stats = answerer.stats;
  const topics = createMemo(() => {
    const seen = new Set<string>();
    const labels: string[] = [];
    for (const document of LUAU_QA_DOCUMENTS) {
      for (const tag of document.tags ?? []) {
        if (!seen.has(tag)) {
          seen.add(tag);
          labels.push(topicLabel(tag));
        }
      }
    }
    return labels.slice(0, 28);
  });

  createEffect(
    () => chat(),
    () => {
      if (chatLog !== undefined) {
        chatLog.scrollTop = chatLog.scrollHeight;
      }
    },
  );

  function askQuestion(preset?: string): void {
    const next = (preset ?? question()).trim();
    if (next.length === 0) {
      return;
    }

    const result = answerer.ask(next);
    const reply: ChatMessage =
      result === null
        ? {
            id: nextMessageId++,
            role: "assistant",
            text: "I need a few words to match. Try asking with words from a Luau question, such as “strict mode” or “table.freeze”.",
            matched: false,
          }
        : {
            id: nextMessageId++,
            role: "assistant",
            text: result.matched
              ? result.answer
              : `I couldn't find an indexed question close enough to that. Closest: “${result.question}”.`,
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

  return (
    <main class="workspace language-workspace">
      <div class="language-toolbar">
        <span class="eyebrow">Luau guide</span>
        <span class="muted">
          Retrieval over the official Luau documentation · no answer text is
          generated
        </span>
      </div>

      <section class="card language-card">
        <div class="card-heading compact">
          <div>
            <span class="eyebrow">Bundled dataset</span>
            <h2>Luau Q&amp;A</h2>
          </div>
          <div class="runtime-pill ready">
            <span class="runtime-dot" />
            Local only
          </div>
        </div>

        <p class="hint">
          Curated from the official Luau documentation at luau.org. Ask about
          syntax, the gradual type system, the standard library, performance,
          lints, sandboxing, and Lua compatibility. The closest indexed question
          supplies the answer, and every answer links back to its source page.
        </p>

        <div class="qa-preset">
          <div class="qa-preset-copy">
            <strong>
              {stats.concepts} concepts · {stats.variants} question variants
            </strong>
            <span>
              {stats.questionWords.toLocaleString()} indexed question words ·{" "}
              {Math.round(stats.coverage * 100)}% are covered by the bundled word
              vectors. Unknown words fall back to a deterministic hash vector.
            </span>
          </div>
          <a href="https://luau.org/" target="_blank" rel="noreferrer">
            Browse the official docs
          </a>
        </div>

        <div class="vocab-block">
          <div class="section-heading">
            <span>Topics covered</span>
            <span class="muted">syntax · types · library · performance</span>
          </div>
          <div class="vocab-list">
            <For each={topics()}>
              {(topic) => <span class="vocab-chip">{topic}</span>}
            </For>
          </div>
        </div>

        <div class="vocab-block">
          <div class="section-heading">
            <span>Try a question</span>
            <span class="muted">click to ask</span>
          </div>
          <div class="completion-list">
            <For each={LUAU_QA_SUGGESTIONS}>
              {(suggestion) => (
                <button
                  type="button"
                  class="completion-chip"
                  onClick={() => askQuestion(suggestion)}
                >
                  <span class="completion-word">{suggestion}</span>
                </button>
              )}
            </For>
          </div>
        </div>
      </section>

      <section class="card language-card">
        <div class="card-heading compact">
          <div>
            <span class="eyebrow">Retrieval</span>
            <h2>Ask about Luau</h2>
          </div>
          <div class="runtime-pill ready">
            <span class="runtime-dot" />
            Answer index ready
          </div>
        </div>

        <p class="hint">
          Use words from the documentation. The reply shows the matched
          question, a confidence score, and a link to the relevant page. Weak
          matches are labelled as the closest match instead of being presented
          as an answer.
        </p>

        <div
          class="chat-log"
          role="log"
          aria-live="polite"
          aria-label="Luau question and answer chat"
          ref={(element) => {
            chatLog = element;
            element.scrollTop = element.scrollHeight;
          }}
        >
          <Show
            when={chat().length > 0}
            fallback={
              <div class="chat-empty">
                <strong>
                  {stats.concepts} Luau concepts · {stats.variants} question
                  variants
                </strong>
                <span>
                  Ask below, or start with one of the bundled questions:
                </span>
                <div class="completion-list">
                  <For each={LUAU_QA_SUGGESTIONS.slice(0, 6)}>
                    {(suggestion) => (
                      <button
                        type="button"
                        class="completion-chip"
                        onClick={() => askQuestion(suggestion)}
                      >
                        <span class="completion-word">{suggestion}</span>
                      </button>
                    )}
                  </For>
                </div>
              </div>
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
                      {message.role === "user" ? "You" : "Luau docs"}
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
                          {Math.round((message.confidence ?? 0) * 100)}% match
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
                        Read the source page
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
            aria-label="Ask about Luau"
            rows="2"
            spellcheck={false}
            placeholder="Ask about Luau…"
            value={question()}
            onInput={(event) => setQuestion(event.currentTarget.value)}
            onKeyDown={handleQuestionKeyDown}
          />
          <button
            type="button"
            class="button primary chat-send"
            disabled={question().trim().length === 0}
            onClick={() => askQuestion()}
          >
            Send
          </button>
        </div>
        <p class="chat-hint">
          Enter to send · Shift+Enter for a new line ·{" "}
          <button
            type="button"
            class="link-button"
            onClick={() => setChat([])}
          >
            clear chat
          </button>
        </p>
      </section>

      <section class="card language-stats">
        <div class="card-heading compact">
          <div>
            <span class="eyebrow">Model</span>
            <h2>What it indexes</h2>
          </div>
          <div class="runtime-pill ready">
            <span class="runtime-dot" />
            Answer index ready
          </div>
        </div>

        <div class="stat-row">
          <div class="stat">
            <span>Concepts</span>
            <strong>{stats.concepts}</strong>
          </div>
          <div class="stat">
            <span>Question variants</span>
            <strong>{stats.variants}</strong>
          </div>
          <div class="stat">
            <span>Question words</span>
            <strong>{stats.questionWords.toLocaleString()}</strong>
          </div>
          <div class="stat">
            <span>Answer vector</span>
            <strong>{stats.vectorBits} bits</strong>
          </div>
          <div class="stat">
            <span>Vector coverage</span>
            <strong>{Math.round(stats.coverage * 100)}%</strong>
          </div>
        </div>

        <div class="vocab-block">
          <div class="section-heading">
            <span>Sample indexed questions</span>
            <span class="muted">
              {stats.vectorBits}-bit vectors · shared words
            </span>
          </div>
          <div class="vocab-list">
            <For each={answerer.questions.slice(0, 28)}>
              {(trained) => <span class="vocab-chip">{trained}</span>}
            </For>
            <Show when={answerer.questions.length > 28}>
              <span class="vocab-chip">
                +{answerer.questions.length - 28} more
              </span>
            </Show>
          </div>
        </div>

        <p class="hint language-attribution">
          Word vectors come from{" "}
          <a
            href="https://github.com/agnusmaximus/Word2Bits"
            target="_blank"
            rel="noreferrer"
          >
            Word2Bits
          </a>{" "}
          (Apache-2.0): a subset of the bundled vocabulary, using 64 of the 800
          bit-vector dimensions. Answers are curated from the official Luau
          documentation and retrieved by shared words and binary question
          vectors. No answer text is generated.
        </p>
      </section>
    </main>
  );
}