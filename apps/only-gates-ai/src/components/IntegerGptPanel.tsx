import {
  createEffect,
  createMemo,
  createSignal,
  For,
  Show,
} from "solid-js";
import {
  TinyGptModel,
  TinyGptMode,
  TinyGptComparison,
  TinyGptGeneration,
  loadTinyGpt,
  compareTinyGpt,
  generateTinyGpt,
} from "../lib/tinyGpt";
import {
  TinyGptWasmModel,
  TinyGptWasmComparison,
  TinyGptWasmGeneration,
  loadTinyGptWasm,
  generateTinyGptWasm,
  compareTinyGptWasm,
} from "../lib/tinyGptWasm";
import { TINY_GPT_CHECKPOINT } from "../assets/tiny-gpt-weights";

type PanelMode = TinyGptMode | "wasm";

function parameterCount(config: {
  vocabSize: number;
  context: number;
  dim: number;
  layers: number;
  mlp: number;
}): number {
  const { vocabSize, context, dim, layers, mlp } = config;
  return (
    vocabSize * dim +
    context * dim +
    dim +
    layers * (2 * dim + 4 * dim * dim + 2 * dim * mlp)
  );
}

export default function IntegerGptPanel() {
  const [model, setModel] = createSignal<TinyGptModel | null>(null);
  const [mode, setMode] = createSignal<PanelMode>("int8");
  const [prompt, setPrompt] = createSignal("Alice was");
  const [loading, setLoading] = createSignal(false);
  const [status, setStatus] = createSignal<string | null>(null);
  const [comparison, setComparison] = createSignal<TinyGptComparison | null>(null);
  const [generating, setGenerating] = createSignal(false);
  const [generation, setGeneration] = createSignal<TinyGptGeneration | null>(null);

  const [wasm, setWasm] = createSignal<TinyGptWasmModel | null>(null);
  const [wasmLoading, setWasmLoading] = createSignal(false);
  const [wasmError, setWasmError] = createSignal<string | null>(null);
  const [wasmGeneration, setWasmGeneration] =
    createSignal<TinyGptWasmGeneration | null>(null);
  const [wasmComparison, setWasmComparison] =
    createSignal<TinyGptWasmComparison | null>(null);

  // Auto-load the float/int8 model on mount.
  createEffect(
    model,
    (current) => {
      if (current) return;
      setLoading(true);
      setStatus("Loading transformer weights...");
      try {
        const loaded = loadTinyGpt(TINY_GPT_CHECKPOINT);
        setModel(loaded);
        setStatus(null);
      } catch (e) {
        setStatus(`Error loading model: ${(e as Error).message}`);
      } finally {
        setLoading(false);
      }
    },
  );

  // Lazily build the integer-only WASM model the first time it is selected.
  createEffect(
    mode,
    (current) => {
      if (current !== "wasm" || wasm() || wasmLoading()) return;
      setWasmLoading(true);
      setWasmError(null);
      loadTinyGptWasm(TINY_GPT_CHECKPOINT)
        .then((loaded) => setWasm(loaded))
        .catch((e) => setWasmError(`WASM error: ${(e as Error).message}`))
        .finally(() => setWasmLoading(false));
    },
  );

  const displayGeneration = createMemo(() => {
    if (mode() === "wasm") {
      const current = wasmGeneration();
      return current ? { ...current, mode: "wasm" as PanelMode } : null;
    }
    return generation();
  });

  const displayComparison = createMemo(() => {
    if (mode() === "wasm") {
      const current = wasmComparison();
      if (!current) return null;
      return {
        leftLabel: "Float32 Top 5",
        rightLabel: "WASM Integer Top 5",
        leftTop: current.floatTop,
        rightTop: current.wasmTop,
        topOneMatches: current.topOneMatches,
        overlapAtFive: current.overlapAtFive,
        meanProbabilityDelta: current.meanProbabilityDelta,
      };
    }
    const current = comparison();
    if (!current) return null;
    return {
      leftLabel: "Float32 Top 5",
      rightLabel: "Int8 Top 5",
      leftTop: current.floatTop,
      rightTop: current.int8Top,
      topOneMatches: current.topOneMatches,
      overlapAtFive: current.overlapAtFive,
      meanProbabilityDelta: current.meanProbabilityDelta,
    };
  });

  const handleGenerate = () => {
    if (generating()) return;

    if (mode() === "wasm") {
      const current = wasm();
      if (!current) return;
      setGenerating(true);
      setWasmGeneration(null);
      setStatus(null);
      setTimeout(() => {
        try {
          setWasmGeneration(
            generateTinyGptWasm(current, prompt(), {
              maxTokens: 24,
              temperature: 0.8,
              topK: 40,
            }),
          );
        } catch (e) {
          setStatus(`Generation error: ${(e as Error).message}`);
        } finally {
          setGenerating(false);
        }
      }, 10);
      return;
    }

    const current = model();
    if (!current) return;
    setGenerating(true);
    setGeneration(null);
    setTimeout(() => {
      try {
        const result = generateTinyGpt(current, prompt(), mode() as TinyGptMode, {
          maxTokens: 24,
          temperature: 0.8,
          topK: 40,
        });
        setGeneration(result);
      } catch (e) {
        setStatus(`Generation error: ${(e as Error).message}`);
      } finally {
        setGenerating(false);
      }
    }, 10);
  };

  const handleCompare = () => {
    if (mode() === "wasm") {
      const wasmModel = wasm();
      const tsModel = model();
      if (!wasmModel || !tsModel) return;
      try {
        setWasmComparison(compareTinyGptWasm(wasmModel, tsModel, prompt()));
      } catch (e) {
        setStatus(`Comparison error: ${(e as Error).message}`);
      }
      return;
    }

    const current = model();
    if (!current) return;
    try {
      setComparison(compareTinyGpt(current, prompt()));
    } catch (e) {
      setStatus(`Comparison error: ${(e as Error).message}`);
    }
  };

  const wasmReady = () => wasm() !== null || mode() !== "wasm";
  const generateDisabled = () =>
    generating() || (mode() === "wasm" ? !wasm() : !model());

  return (
    <main class="workspace language-workspace">
      <div class="language-toolbar">
        <div class="mode-switch" role="group" aria-label="Inference mode">
          <button
            type="button"
            class={{ active: mode() === "int8" }}
            onClick={() => setMode("int8")}
          >
            Int8 Quantized
          </button>
          <button
            type="button"
            class={{ active: mode() === "float" }}
            onClick={() => setMode("float")}
          >
            Float32 Baseline
          </button>
          <button
            type="button"
            class={{ active: mode() === "wasm" }}
            onClick={() => setMode("wasm")}
          >
            Integer-only (WASM)
          </button>
        </div>
        <span class="muted">
          {mode() === "int8"
            ? "Integer matmuls, int8 weights/activations"
            : mode() === "float"
              ? "Original float32 weights and arithmetic"
              : "Q16 activations, int8 weights, integer RMSNorm/GELU/softmax in WebAssembly"}
        </span>
      </div>

      <section class="card language-card">
        <div class="card-heading compact">
          <div>
            <span class="eyebrow">Integer GPT</span>
            <h2>Alice in Wonderland</h2>
          </div>
          <Show when={mode() === "wasm" ? wasm() : model()}>
            {(current) => (
              <div class="runtime-pill ready">
                <span class="runtime-dot" />
                {parameterCount(current().config).toLocaleString()} params
              </div>
            )}
          </Show>
        </div>

        <p class="hint">
          This tiny transformer was trained on Chapters I &amp; II of Alice's
          Adventures in Wonderland.
          <Show when={mode() === "int8"}>
            <span>
              {" "}The <b>Int8</b> mode uses per-channel weight quantization and
              per-token activation quantization; RMSNorm, GELU and softmax stay
              in float32.
            </span>
          </Show>
          <Show when={mode() === "float"}>
            <span> Original float32 weights and arithmetic throughout.</span>
          </Show>
          <Show when={mode() === "wasm"}>
            <span>
              {" "}The <b>Integer-only (WASM)</b> mode runs the entire forward
              pass in WebAssembly with no floating point: Q16 int32 activations,
              int8 weights, integer RMSNorm, a Q16 GELU lookup table and DI-Exp
              softmax. Float32 is used only once, to quantize the checkpoint at
              load time.
            </span>
          </Show>
        </p>

        <Show when={status()}>
          <div class="banner info">
            <span>{status()}</span>
          </div>
        </Show>
        <Show when={mode() === "wasm" && wasmLoading()}>
          <div class="banner info">
            <span>Loading WebAssembly integer kernels...</span>
          </div>
        </Show>
        <Show when={mode() === "wasm" && wasmError()}>
          <div class="banner warning">
            <span>{wasmError()}</span>
          </div>
        </Show>

        <div class="prompt-row" style="display: flex; gap: 8px; margin-top: 16px;">
          <input
            type="text"
            class="chat-input"
            style="flex: 1; background: var(--bg); border: 1px solid var(--border); border-radius: 8px; padding: 8px 12px; color: var(--text);"
            placeholder="Type a prompt..."
            value={prompt()}
            onInput={(e) => setPrompt(e.currentTarget.value)}
            onKeyDown={(e) => e.key === "Enter" && handleGenerate()}
            disabled={generating()}
          />
          <button
            type="button"
            class="button primary"
            onClick={handleGenerate}
            disabled={generateDisabled()}
          >
            {generating() ? "..." : "Generate"}
          </button>
          <button
            type="button"
            class="button secondary"
            onClick={handleCompare}
            disabled={generating() || !wasmReady() || !model()}
          >
            Compare
          </button>
        </div>

        <Show when={displayGeneration()}>
          {(current) => (
            <div class="chat-log" style="margin-top: 24px;">
              <div class="chat-message user">
                <div class="chat-bubble">
                  <p>
                    <strong>Prompt:</strong> {current().promptTokens.join(" ")}
                  </p>
                </div>
              </div>
              <div class="chat-message assistant">
                <div class="chat-bubble">
                  <p>{current().text}</p>
                  <div class="chat-meta">
                    {current().steps} tokens in{" "}
                    {current().elapsedMs.toFixed(1)}ms ({current().mode})
                  </div>
                </div>
              </div>
            </div>
          )}
        </Show>

        <Show when={displayComparison()}>
          {(current) => (
            <div
              class="comparison-results"
              style="margin-top: 24px; padding-top: 16px; border-top: 1px solid var(--border);"
            >
              <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 24px;">
                <div>
                  <h4 style="margin: 0 0 8px; font-size: 0.9rem; color: var(--muted);">
                    {current().leftLabel}
                  </h4>
                  <ul style="list-style: none; padding: 0; margin: 0; font-size: 0.85rem;">
                    <For each={current().leftTop}>
                      {(candidate) => (
                        <li style="display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid rgba(255,255,255,0.05);">
                          <span>{candidate.token}</span>
                          <span class="muted">
                            {(candidate.probability * 100).toFixed(1)}%
                          </span>
                        </li>
                      )}
                    </For>
                  </ul>
                </div>
                <div>
                  <h4 style="margin: 0 0 8px; font-size: 0.9rem; color: var(--muted);">
                    {current().rightLabel}
                  </h4>
                  <ul style="list-style: none; padding: 0; margin: 0; font-size: 0.85rem;">
                    <For each={current().rightTop}>
                      {(candidate) => (
                        <li style="display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid rgba(255,255,255,0.05);">
                          <span>{candidate.token}</span>
                          <span class="muted">
                            {(candidate.probability * 100).toFixed(1)}%
                          </span>
                        </li>
                      )}
                    </For>
                  </ul>
                </div>
              </div>
              <div class="chat-meta" style="margin-top: 12px; text-align: center;">
                Top-1 match: {current().topOneMatches ? "✅ Yes" : "❌ No"} |
                Overlap@5: {current().overlapAtFive}/5 | Mean Prob Delta:{" "}
                {current().meanProbabilityDelta.toExponential(2)}
              </div>
            </div>
          )}
        </Show>

        <Show when={mode() !== "wasm" && model()}>
          {(current) => (
            <div class="text-meta" style="margin-top: 24px;">
              <span>{current().stats.compression.toFixed(1)}x compression</span>
              <span>
                Weight RMS Error:{" "}
                {current().stats.weightRmsError.toExponential(2)}
              </span>
            </div>
          )}
        </Show>
      </section>
    </main>
  );
}
