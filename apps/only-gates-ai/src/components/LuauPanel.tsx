import { For, Show, createEffect, createMemo, createSignal, onCleanup } from "solid-js";
import {
  createSolid2Decoder,
  loadSolid2Wasm,
  type Solid2Decoder,
  type Solid2Generation,
  type Solid2WasmModel,
} from "../lib/tinyGptSolid2";
import {
  LUAU_MODEL_BASE,
  LUAU_PRESETS,
  LUAU_TRAIN_CONFIG,
  luauParameterCount,
  type LuauMode,
} from "../lib/luauConfig";

interface LuauPanelProps {
  /** The tab is selected. The model loads lazily on first open. */
  readonly active: boolean;
}

function generationMeta(
  result: Solid2Generation,
  parameterCount: number,
): string {
  const tokens = result.newTokenCount;
  const wallRate =
    result.elapsedMs > 0 ? (tokens / result.elapsedMs) * 1000 : 0;
  return [
    `${parameterCount.toLocaleString()} params`,
    `${tokens} tokens`,
    `${wallRate.toFixed(0)} tok/s wall`,
    `${result.computeTokensPerSecond.toFixed(0)} tok/s WASM`,
    result.stopped ? "stopped at <|end|>" : "context limit",
  ].join(" · ");
}

function copyText(value: string): Promise<void> {
  if (navigator.clipboard?.writeText) {
    return navigator.clipboard.writeText(value);
  }
  const area = document.createElement("textarea");
  area.value = value;
  area.style.position = "fixed";
  area.style.opacity = "0";
  document.body.appendChild(area);
  area.select();
  try {
    document.execCommand("copy");
    return Promise.resolve();
  } finally {
    area.remove();
  }
}

export default function LuauPanel(props: LuauPanelProps) {
  const [mode, setMode] = createSignal<LuauMode>("write");
  const [prompt, setPrompt] = createSignal(LUAU_PRESETS[0]!.prompt);
  const [model, setModel] = createSignal<Solid2WasmModel | null>(null);
  const [loading, setLoading] = createSignal(false);
  const [error, setError] = createSignal<string | null>(null);
  const [output, setOutput] = createSignal("");
  const [meta, setMeta] = createSignal("");
  const [generating, setGenerating] = createSignal(false);
  const [copied, setCopied] = createSignal(false);
  const [reload, setReload] = createSignal(0);

  let decoder: Solid2Decoder | null = null;
  let timer: ReturnType<typeof setTimeout> | null = null;
  let loadInFlight = false;

  const visiblePresets = createMemo(() =>
    LUAU_PRESETS.filter((preset) => preset.mode === mode()),
  );

  const activeConfig = createMemo(() => model()?.config ?? LUAU_TRAIN_CONFIG);
  const activeParameters = createMemo(() =>
    model() ? model()!.parameterCount : luauParameterCount(LUAU_TRAIN_CONFIG),
  );

  const stop = (): void => {
    if (timer !== null) {
      clearTimeout(timer);
      timer = null;
    }
    decoder = null;
    setGenerating(false);
  };

  onCleanup(stop);

  createEffect(
    () => ({ active: props.active, attempt: reload() }),
    ({ active, attempt }) => {
      if (!active || attempt < 0 || model() || loadInFlight) return;
      loadInFlight = true;
      setLoading(true);
      setError(null);
      loadSolid2Wasm(LUAU_MODEL_BASE)
        .then((loaded) => {
          setModel(loaded);
          setError(null);
        })
        .catch((reason: unknown) => {
          setError(
            `Could not load the Luau model from ${LUAU_MODEL_BASE}: ${
              (reason as Error).message
            }. Train it with scripts/train_tiny_gpt_luau_colab.py, then copy ` +
              "checkpoint.json, tokenizer.json, and weights.bin into that folder and rebuild.",
          );
        })
        .finally(() => {
          loadInFlight = false;
          setLoading(false);
        });
    },
  );

  const handleGenerate = (): void => {
    if (generating()) return;
    const currentModel = model();
    const input = prompt().trim();
    if (!currentModel || input.length === 0) return;

    stop();
    setOutput("");
    setMeta("");
    setCopied(false);
    setGenerating(true);

    const prefix = mode() === "write" ? "Task: " : "Question: ";
    let current: Solid2Decoder;
    try {
      current = createSolid2Decoder(currentModel, prefix + input, {
        maxTokens: mode() === "write" ? 160 : 200,
        // Code benefits from low temperature and a mild repetition penalty.
        temperature: mode() === "write" ? 0.15 : 0.3,
        topK: 40,
        repetitionPenalty: 1.05,
        promptFormat: "chat",
      });
    } catch (reason) {
      setGenerating(false);
      setError(`Generation error: ${(reason as Error).message}`);
      return;
    }

    decoder = current;
    const tick = (): void => {
      timer = null;
      if (decoder !== current) return;
      const deadline = performance.now() + 40;
      let done = false;
      while (performance.now() < deadline) {
        if (current.step().done) {
          done = true;
          break;
        }
      }
      const result = current.result();
      setOutput(result.text);
      setMeta(generationMeta(result, currentModel.parameterCount));
      if (done) {
        decoder = null;
        setGenerating(false);
        return;
      }
      timer = setTimeout(tick, 0);
    };
    timer = setTimeout(tick, 0);
  };

  const handleCopy = async (): Promise<void> => {
    if (!output()) return;
    try {
      await copyText(output());
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch (reason) {
      setError(`Could not copy: ${(reason as Error).message}`);
    }
  };

  const handleKeyDown = (event: KeyboardEvent): void => {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
      event.preventDefault();
      handleGenerate();
    }
  };

  const choosePreset = (preset: (typeof LUAU_PRESETS)[number]): void => {
    setMode(preset.mode);
    setPrompt(preset.prompt);
    setError(null);
  };

  return (
    <section class="card luau-panel">
      <div class="card-heading compact">
        <div>
          <span class="eyebrow">Luau</span>
          <h2>Ask about Luau or write a snippet</h2>
        </div>
        <div
          class={
            model()
              ? "runtime-pill ready"
              : error()
                ? "runtime-pill warning"
                : "runtime-pill"
          }
        >
          <span class="runtime-dot" />
          {model()
            ? "Integer WASM ready"
            : loading()
              ? "Loading model"
              : error()
                ? "Model missing"
                : "Idle"}
        </div>
      </div>

      <p class="hint">
        A dedicated Luau model trained on beginner tasks and the repository
        quick-answer FAQ. Inference is the same integer-only WASM path as
        Integer GPT: int8 weights, Q16 activations. Code is generated for you
        to read and copy; it is not executed in the browser.
      </p>

      <div class="luau-mode" role="tablist" aria-label="Luau response mode">
        <button
          type="button"
          role="tab"
          aria-selected={mode() === "write" ? "true" : "false"}
          class={{ active: mode() === "write" }}
          onClick={() => {
            setMode("write");
            setPrompt(LUAU_PRESETS[0]!.prompt);
          }}
        >
          Write code
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={mode() === "ask" ? "true" : "false"}
          class={{ active: mode() === "ask" }}
          onClick={() => {
            setMode("ask");
            setPrompt(LUAU_PRESETS.find((preset) => preset.mode === "ask")!.prompt);
          }}
        >
          Ask a question
        </button>
      </div>

      <div class="luau-presets">
        <span class="muted">Examples:</span>
        <For each={visiblePresets()}>
          {(preset) => (
            <button
              type="button"
              class="button secondary"
              disabled={generating()}
              onClick={() => choosePreset(preset)}
            >
              {preset.label}
            </button>
          )}
        </For>
      </div>

      <label class="luau-field">
        {mode() === "write" ? "Task" : "Question"}
        <textarea
          rows={4}
          value={prompt()}
          disabled={generating()}
          onInput={(event) => setPrompt(event.currentTarget.value)}
          onKeyDown={handleKeyDown}
          placeholder={
            mode() === "write"
              ? "Write a for loop that sums the first 50 elements of `values`..."
              : "How do I write a for loop in Luau?"
          }
        />
      </label>

      <div class="button-row">
        <button
          type="button"
          class="button primary"
          disabled={generating() || !model() || prompt().trim().length === 0}
          onClick={handleGenerate}
        >
          {generating()
            ? "Generating..."
            : mode() === "write"
              ? "Write Luau"
              : "Answer"}
        </button>
        <Show when={generating()}>
          <button type="button" class="button ghost" onClick={stop}>
            Stop
          </button>
        </Show>
        <span class="chat-hint">
          {mode() === "write"
            ? "Ctrl/Cmd + Enter to generate"
            : "Answers are short explanations with examples"}
        </span>
      </div>

      <Show when={error()}>
        <div class="banner error">
          <span>{error()}</span>
          <button
            type="button"
            class="button ghost"
            onClick={() => {
              setError(null);
              setReload((value) => value + 1);
            }}
          >
            Retry
          </button>
        </div>
      </Show>

      <Show when={loading()}>
        <div class="banner info">
          <span>
            Loading the Luau model ({activeConfig().context}-token context,
            BPE tokenizer)...
          </span>
        </div>
      </Show>

      <Show when={output() || generating()}>
        <div class="luau-output">
          <div class="luau-output-head">
            <strong>{mode() === "write" ? "Generated Luau" : "Answer"}</strong>
            <button
              type="button"
              class="button ghost"
              disabled={!output()}
              onClick={() => void handleCopy()}
            >
              {copied() ? "Copied" : "Copy"}
            </button>
          </div>
          <pre class="chat-code" tabindex="0">
            <code>{output() || "..."}</code>
          </pre>
          <div class="chat-meta">{meta()}</div>
        </div>
      </Show>

      <Show when={model()}>
        <p class="hint luau-model-meta">
          {activeParameters().toLocaleString()} parameters ·{" "}
          {activeConfig().vocabSize.toLocaleString()} BPE tokens ·{" "}
          {activeConfig().context}-token context ·{" "}
          {model()!.backend === "wasm-simd" ? "WASM SIMD" : "WASM scalar"} ·{" "}
          {model()!.checkpoint.meta?.name ?? "tiny-gpt-luau"}
        </p>
      </Show>
    </section>
  );
}
