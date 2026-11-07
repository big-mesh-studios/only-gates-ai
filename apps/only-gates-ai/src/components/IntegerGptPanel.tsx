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
import { TINY_GPT_CHECKPOINT } from "../assets/tiny-gpt-weights";

export default function IntegerGptPanel() {
  const [model, setModel] = createSignal<TinyGptModel | null>(null);
  const [mode, setMode] = createSignal<TinyGptMode>("int8");
  const [prompt, setPrompt] = createSignal("Alice was");
  const [loading, setLoading] = createSignal(false);
  const [status, setStatus] = createSignal<string | null>(null);
  const [comparison, setComparison] = createSignal<TinyGptComparison | null>(null);
  const [generating, setGenerating] = createSignal(false);
  const [generation, setGeneration] = createSignal<TinyGptGeneration | null>(null);

  // Auto-load the model on mount
  createEffect(
    model,
    (model) => {
      if (model) return;
      setLoading(true);
      setStatus("Loading transformer weights...");
      try {
        const m = loadTinyGpt(TINY_GPT_CHECKPOINT);
        setModel(m);
        setStatus(null);
      } catch (e: any) {
        setStatus(`Error loading model: ${e.message}`);
      } finally {
        setLoading(false);
      }
    }
  );

  const handleGenerate = () => {
    const m = model();
    if (!m || generating()) return;
    
    setGenerating(true);
    setGeneration(null);
    
    // Use a small timeout to let the UI update if needed, 
    // though this is synchronous in our current implementation.
    setTimeout(() => {
      try {
        const g = generateTinyGpt(m, prompt(), mode(), {
          maxTokens: 24,
          temperature: 0.8,
          topK: 40,
        });
        setGeneration(g);
      } catch (e: any) {
        setStatus(`Generation error: ${e.message}`);
      } finally {
        setGenerating(false);
      }
    }, 10);
  };

  const handleCompare = () => {
    const m = model();
    if (!m) return;
    try {
      const cmp = compareTinyGpt(m, prompt());
      setComparison(cmp);
    } catch (e: any) {
      setStatus(`Comparison error: ${e.message}`);
    }
  };

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
        </div>
        <span class="muted">
          {mode() === "int8" 
            ? "Integer matmuls, int8 weights/activations" 
            : "Original float32 weights and arithmetic"}
        </span>
      </div>

      <section class="card language-card">
        <div class="card-heading compact">
          <div>
            <span class="eyebrow">Integer GPT</span>
            <h2>Alice in Wonderland</h2>
          </div>
          <Show when={model()}>
            {(m) => (
              <div class="runtime-pill ready">
                <span class="runtime-dot" />
                {m().stats.parameters.toLocaleString()} params
              </div>
            )}
          </Show>
        </div>

        <p class="hint">
          This tiny transformer was trained on Chapters I & II of Alice's Adventures in Wonderland. 
          The <b>Int8</b> mode uses per-channel weight quantization and per-token activation quantization.
          Softmax and RMSNorm remain in float32.
        </p>

        <Show when={status()}>
          <div class="banner info">
            <span>{status()}</span>
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
            disabled={generating() || !model()}
          >
            {generating() ? "..." : "Generate"}
          </button>
          <button
            type="button"
            class="button secondary"
            onClick={handleCompare}
            disabled={generating() || !model()}
          >
            Compare
          </button>
        </div>

        <Show when={generation()}>
          {(g) => (
            <div class="chat-log" style="margin-top: 24px;">
              <div class="chat-message user">
                <div class="chat-bubble">
                  <p><strong>Prompt:</strong> {g().promptTokens.join(" ")}</p>
                </div>
              </div>
              <div class="chat-message assistant">
                <div class="chat-bubble">
                  <p>{g().text}</p>
                  <div class="chat-meta">
                    {g().steps} tokens in {g().elapsedMs.toFixed(1)}ms ({g().mode})
                  </div>
                </div>
              </div>
            </div>
          )}
        </Show>

        <Show when={comparison()}>
          {(cmp) => (
            <div class="comparison-results" style="margin-top: 24px; padding-top: 16px; border-top: 1px solid var(--border);">
              <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 24px;">
                <div>
                  <h4 style="margin: 0 0 8px; font-size: 0.9rem; color: var(--muted);">Float32 Top 5</h4>
                  <ul style="list-style: none; padding: 0; margin: 0; font-size: 0.85rem;">
                    <For each={cmp().floatTop.slice(0, 5)}>
                      {(c) => (
                        <li style="display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid rgba(255,255,255,0.05);">
                          <span>{c.token}</span>
                          <span class="muted">{(c.probability * 100).toFixed(1)}%</span>
                        </li>
                      )}
                    </For>
                  </ul>
                </div>
                <div>
                  <h4 style="margin: 0 0 8px; font-size: 0.9rem; color: var(--muted);">Int8 Top 5</h4>
                  <ul style="list-style: none; padding: 0; margin: 0; font-size: 0.85rem;">
                    <For each={cmp().int8Top.slice(0, 5)}>
                      {(c) => (
                        <li style="display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid rgba(255,255,255,0.05);">
                          <span>{c.token}</span>
                          <span class="muted">{(c.probability * 100).toFixed(1)}%</span>
                        </li>
                      )}
                    </For>
                  </ul>
                </div>
              </div>
              <div class="chat-meta" style="margin-top: 12px; text-align: center;">
                Top-1 match: {cmp().topOneMatches ? "✅ Yes" : "❌ No"} | 
                Overlap@5: {cmp().overlapAtFive}/5 | 
                Mean Prob Delta: {cmp().meanProbabilityDelta.toExponential(2)}
              </div>
            </div>
          )}
        </Show>
        
        <Show when={model()}>
          {(m) => (
            <div class="text-meta" style="margin-top: 24px;">
              <span>{m().stats.compression.toFixed(1)}x compression</span>
              <span>Weight RMS Error: {m().stats.weightRmsError.toExponential(2)}</span>
            </div>
          )}
        </Show>
      </section>
    </main>
  );
}
