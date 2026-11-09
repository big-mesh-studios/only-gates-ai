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
import {
  loadSolid2Wasm,
  generateSolid2Wasm,
  type Solid2WasmModel,
  type Solid2Generation,
} from "../lib/tinyGptSolid2";

type ExampleId = "alice" | "solid2";
type AliceMode = TinyGptMode | "wasm";
type PanelMode = AliceMode | "solid2-wasm";

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

// Base-relative URL works under Vite dev and the repo's relative `base: "./"` build.
const SOLID2_MODEL_BASE = `${import.meta.env.BASE_URL}models/tiny-gpt-solid2/`;

export default function IntegerGptPanel() {
  const [example, setExample] = createSignal<ExampleId>("alice");

  // --- Alice (word-level, small) ---
  const [model, setModel] = createSignal<TinyGptModel | null>(null);
  const [aliceMode, setAliceMode] = createSignal<AliceMode>("int8");
  const [loading, setLoading] = createSignal(false);
  const [status, setStatus] = createSignal<string | null>(null);
  const [comparison, setComparison] = createSignal<TinyGptComparison | null>(null);
  const [generating, setGenerating] = createSignal(false);
  const [generation, setGeneration] = createSignal<TinyGptGeneration | null>(null);
  const [wasm, setWasm] = createSignal<TinyGptWasmModel | null>(null);
  const [wasmLoading, setWasmLoading] = createSignal(false);
  const [wasmError, setWasmError] = createSignal<string | null>(null);
  const [wasmGeneration, setWasmGeneration] = createSignal<TinyGptWasmGeneration | null>(null);
  const [wasmComparison, setWasmComparison] = createSignal<TinyGptWasmComparison | null>(null);

  // --- Solid 2 (BPE, ~1M params, WASM-primary) ---
  const [solid2, setSolid2] = createSignal<Solid2WasmModel | null>(null);
  const [solid2Loading, setSolid2Loading] = createSignal(false);
  const [solid2Error, setSolid2Error] = createSignal<string | null>(null);
  const [solid2Gen, setSolid2Gen] = createSignal<Solid2Generation | null>(null);

  const [prompt, setPrompt] = createSignal("Alice was");

  // Auto-load Alice.
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

  // Lazily build Alice WASM model.
  createEffect(
    () => ({ ex: example(), mode: aliceMode() }),
    (s) => {
      if (s.ex !== "alice" || s.mode !== "wasm" || wasm() || wasmLoading()) return;
      setWasmLoading(true);
      setWasmError(null);
      loadTinyGptWasm(TINY_GPT_CHECKPOINT)
        .then((loaded) => setWasm(loaded))
        .catch((e) => setWasmError(`WASM error: ${(e as Error).message}`))
        .finally(() => setWasmLoading(false));
    },
  );

  // Lazily load Solid 2 WASM model.
  createEffect(
    example,
    (ex) => {
      if (ex !== "solid2" || solid2() || solid2Loading()) return;
      // switch prompt to a Solid 2 question when first entering
      setPrompt((p) => (p === "Alice was" ? "What are split effects?" : p));
      setSolid2Loading(true);
      setSolid2Error(null);
      loadSolid2Wasm(SOLID2_MODEL_BASE)
        .then((m) => setSolid2(m))
        .catch((e) =>
          setSolid2Error(
            `Could not load Solid 2 model from ${SOLID2_MODEL_BASE}: ${(e as Error).message}. ` +
              `Train it with: colab run --gpu T4 scripts/train_tiny_gpt_solid2_colab.py -- --out /content/out ` +
              `then copy weights.bin, tokenizer.json, checkpoint.json to apps/only-gates-ai/public/models/tiny-gpt-solid2/ and rebuild.`,
          ),
        )
        .finally(() => setSolid2Loading(false));
    },
  );

  const mode = (): PanelMode => (example() === "solid2" ? "solid2-wasm" : aliceMode());

  const activeConfig = createMemo(() => {
    if (example() === "solid2") return solid2()?.config ?? null;
    return (aliceMode() === "wasm" ? wasm()?.config : model()?.config) ?? model()?.config ?? null;
  });

  const handleGenerate = () => {
    if (generating()) return;
    if (example() === "solid2") {
      const m = solid2();
      if (!m) return;
      setGenerating(true);
      setSolid2Gen(null);
      setStatus(null);
      setTimeout(() => {
        try {
          setSolid2Gen(generateSolid2Wasm(m, prompt(), { maxTokens: 160, temperature: 0.7, topK: 40 }));
        } catch (e) {
          setStatus(`Generation error: ${(e as Error).message}`);
        } finally {
          setGenerating(false);
        }
      }, 10);
      return;
    }
    if (aliceMode() === "wasm") {
      const current = wasm();
      if (!current) return;
      setGenerating(true);
      setWasmGeneration(null);
      setStatus(null);
      setTimeout(() => {
        try {
          setWasmGeneration(generateTinyGptWasm(current, prompt(), { maxTokens: 24, temperature: 0.8, topK: 40 }));
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
        setGeneration(generateTinyGpt(current, prompt(), aliceMode() as TinyGptMode, { maxTokens: 24, temperature: 0.8, topK: 40 }));
      } catch (e) {
        setStatus(`Generation error: ${(e as Error).message}`);
      } finally {
        setGenerating(false);
      }
    }, 10);
  };

  const handleCompare = () => {
    if (example() === "solid2") {
      setStatus("Comparison against a float baseline is not enabled for the 4k-vocab BPE model (JS float/int8 inference is too slow in-browser). Use the Alice example for float-vs-integer comparison.");
      return;
    }
    if (aliceMode() === "wasm") {
      const wasmModel = wasm();
      const tsModel = model();
      if (!wasmModel || !tsModel) return;
      try { setWasmComparison(compareTinyGptWasm(wasmModel, tsModel, prompt())); }
      catch (e) { setStatus(`Comparison error: ${(e as Error).message}`); }
      return;
    }
    const current = model();
    if (!current) return;
    try { setComparison(compareTinyGpt(current, prompt())); }
    catch (e) { setStatus(`Comparison error: ${(e as Error).message}`); }
  };

  const generateDisabled = () => generating() || (example() === "solid2" ? !solid2() : (aliceMode() === "wasm" ? !wasm() : !model()));

  const aliceDisplayGen = createMemo(() => {
    if (example() !== "alice") return null;
    if (aliceMode() === "wasm") { const c = wasmGeneration(); return c ? { prompt: c.promptTokens.join(" "), text: c.text, steps: c.steps, elapsedMs: c.elapsedMs, mode: "wasm" } : null; }
    const c = generation(); return c ? { prompt: c.promptTokens.join(" "), text: c.text, steps: c.steps, elapsedMs: c.elapsedMs, mode: c.mode } : null;
  });

  const aliceComparison = createMemo(() => {
    if (example() !== "alice") return null;
    if (aliceMode() === "wasm") {
      const c = wasmComparison(); if (!c) return null;
      return { leftLabel: "Float32 Top 5", rightLabel: "WASM Integer Top 5", leftTop: c.floatTop, rightTop: c.wasmTop, topOneMatches: c.topOneMatches, overlapAtFive: c.overlapAtFive, meanProbabilityDelta: c.meanProbabilityDelta };
    }
    const c = comparison(); if (!c) return null;
    return { leftLabel: "Float32 Top 5", rightLabel: "Int8 Top 5", leftTop: c.floatTop, rightTop: c.int8Top, topOneMatches: c.topOneMatches, overlapAtFive: c.overlapAtFive, meanProbabilityDelta: c.meanProbabilityDelta };
  });

  return (
    <main class="workspace language-workspace">
      <div class="language-toolbar" style="flex-wrap: wrap; gap: 8px;">
        <div class="mode-switch" role="group" aria-label="Example">
          <button type="button" class={{ active: example() === "alice" }} onClick={() => { setExample("alice"); setPrompt("Alice was"); setStatus(null); }}>Alice (word)</button>
          <button type="button" class={{ active: example() === "solid2" }} onClick={() => { setExample("solid2"); setStatus(null); }}>Solid 2 Docs (BPE)</button>
        </div>
        <Show when={example() === "alice"}>
          <div class="mode-switch" role="group" aria-label="Inference mode">
            <button type="button" class={{ active: aliceMode() === "int8" }} onClick={() => setAliceMode("int8")}>Int8</button>
            <button type="button" class={{ active: aliceMode() === "float" }} onClick={() => setAliceMode("float")}>Float32</button>
            <button type="button" class={{ active: aliceMode() === "wasm" }} onClick={() => setAliceMode("wasm")}>Integer-only (WASM)</button>
          </div>
        </Show>
        <Show when={example() === "solid2"}>
          <span class="muted">Integer-only (WASM), Q16/int8, BPE 4096, ctx 256. TS float/int8 disabled (too slow at 4k vocab in JS).</span>
        </Show>
      </div>

      <section class="card language-card">
        <div class="card-heading compact">
          <div>
            <span class="eyebrow">Integer GPT</span>
            <h2>{example() === "alice" ? "Alice in Wonderland" : "Solid 2 Docs Q&A"}</h2>
          </div>
          <Show when={activeConfig()}>
            {(c) => (
              <div class="runtime-pill ready">
                <span class="runtime-dot" />
                {parameterCount(c()).toLocaleString()} params
              </div>
            )}
          </Show>
        </div>

        <p class="hint">
          <a href={import.meta.env.DEV ? "http://localhost:5175/" : "learn/"}>
            How Integer GPT works under the hood →
          </a>
        </p>

        <p class="hint">
          <Show when={example() === "alice"}>
            Tiny transformer trained on Chapters I &amp; II of Alice's Adventures in Wonderland (word-level). Float32 is used only to quantize at load; inference modes show float/int8/fully-integer WASM behavior.
          </Show>
          <Show when={example() === "solid2"}>
            Decoder-only transformer (byte-level BPE, 4096 vocab, 256 ctx; config sized to the small docs+Q&A corpus) trained from scratch on <b>pinned Solid 2 docs + curated Q&A</b> via <code>scripts/train_tiny_gpt_solid2_colab.py</code> (runnable with the Termux <code>colab</code> CLI). The browser runs the <b>entire forward pass as integer-only WebAssembly</b> (Q16 activations, int8 weights, integer RMSNorm/GELU/DI-Exp softmax). Float32 is used only once to quantize the checkpoint at load.
            <Show when={activeConfig()}>{(c) => <span> Loaded config: {c().vocabSize}v/{c().context}ctx, {c().dim}d/{c().layers}L/{c().heads}H/{c().mlp}mlp ({parameterCount(c()).toLocaleString()} params).</span>}</Show>
          </Show>
        </p>

        <Show when={status()}><div class="banner info"><span>{status()}</span></div></Show>
        <Show when={example() === "alice" && aliceMode() === "wasm" && wasmLoading()}><div class="banner info"><span>Loading WebAssembly integer kernels...</span></div></Show>
        <Show when={example() === "alice" && aliceMode() === "wasm" && wasmError()}><div class="banner warning"><span>{wasmError()}</span></div></Show>
        <Show when={example() === "solid2" && solid2Loading()}><div class="banner info"><span>Loading Solid 2 model (weights.bin + tokenizer + WASM quantization)...</span></div></Show>
        <Show when={example() === "solid2" && solid2Error()}><div class="banner warning"><span style="white-space: pre-wrap;">{solid2Error()}</span></div></Show>

        <div class="prompt-row" style="display: flex; gap: 8px; margin-top: 16px; flex-wrap: wrap;">
          <input
            type="text"
            class="chat-input"
            style="flex: 1; min-width: 220px; background: var(--bg); border: 1px solid var(--border); border-radius: 8px; padding: 8px 12px; color: var(--text);"
            placeholder={example() === "solid2" ? "Ask a Solid 2 question..." : "Type a prompt..."}
            value={prompt()}
            onInput={(e) => setPrompt(e.currentTarget.value)}
            onKeyDown={(e) => e.key === "Enter" && handleGenerate()}
            disabled={generating()}
          />
          <button type="button" class="button primary" onClick={handleGenerate} disabled={generateDisabled()}>
            {generating() ? "..." : "Generate"}
          </button>
          <Show when={example() === "alice"}>
            <button type="button" class="button secondary" onClick={handleCompare} disabled={generating() || !model()}>Compare</button>
          </Show>
        </div>

        <Show when={example() === "solid2" && solid2()?.checkpoint.prompts?.length}>
          <div style="margin-top: 10px; display: flex; gap: 6px; flex-wrap: wrap;">
            <For each={solid2()!.checkpoint.prompts!.slice(0, 5)}>
              {(p) => <button type="button" class="button secondary" style="font-size: 0.8rem; padding: 4px 8px;" onClick={() => setPrompt(p)}>{p}</button>}
            </For>
          </div>
        </Show>

        <Show when={example() === "alice" && aliceDisplayGen()}>
          {(c) => (
            <div class="chat-log" style="margin-top: 24px;">
              <div class="chat-message user"><div class="chat-bubble"><p><strong>Prompt:</strong> {c().prompt}</p></div></div>
              <div class="chat-message assistant"><div class="chat-bubble"><p>{c().text}</p><div class="chat-meta">{c().steps} tokens in {c().elapsedMs.toFixed(1)}ms ({c().mode})</div></div></div>
            </div>
          )}
        </Show>

        <Show when={example() === "solid2" && solid2Gen()}>
          {(c) => (
            <div class="chat-log" style="margin-top: 24px;">
              <div class="chat-message user"><div class="chat-bubble"><p><strong>You:</strong> {c().prompt}</p></div></div>
              <div class="chat-message assistant"><div class="chat-bubble"><p style="white-space: pre-wrap;">{c().text || <span class="muted">(empty generation)</span>}</p><div class="chat-meta">{c().newTokenCount} tokens in {c().elapsedMs.toFixed(1)}ms (integer-only WASM, BPE){c().stopped ? "" : " · max tokens"}</div></div></div>
            </div>
          )}
        </Show>

        <Show when={example() === "alice" && aliceComparison()}>
          {(c) => (
            <div class="comparison-results" style="margin-top: 24px; padding-top: 16px; border-top: 1px solid var(--border);">
              <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 24px;">
                <div><h4 style="margin: 0 0 8px; font-size: 0.9rem; color: var(--muted);">{c().leftLabel}</h4>
                  <ul style="list-style: none; padding: 0; margin: 0; font-size: 0.85rem;">
                    <For each={c().leftTop}>{(x) => <li style="display: flex; justify-content: space-between; padding: 4px 0;"><span>{x.token}</span><span class="muted">{(x.probability * 100).toFixed(1)}%</span></li>}</For>
                  </ul></div>
                <div><h4 style="margin: 0 0 8px; font-size: 0.9rem; color: var(--muted);">{c().rightLabel}</h4>
                  <ul style="list-style: none; padding: 0; margin: 0; font-size: 0.85rem;">
                    <For each={c().rightTop}>{(x) => <li style="display: flex; justify-content: space-between; padding: 4px 0;"><span>{x.token}</span><span class="muted">{(x.probability * 100).toFixed(1)}%</span></li>}</For>
                  </ul></div>
              </div>
              <div class="chat-meta" style="margin-top: 12px; text-align: center;">Top-1 match: {c().topOneMatches ? "✅ Yes" : "❌ No"} | Overlap@5: {c().overlapAtFive}/5 | Mean Prob Delta: {c().meanProbabilityDelta.toExponential(2)}</div>
            </div>
          )}
        </Show>

        <Show when={example() === "alice" && aliceMode() !== "wasm" && model()}>
          {(current) => (
            <div class="text-meta" style="margin-top: 24px;">
              <span>{current().stats.compression.toFixed(1)}x compression</span>
              <span>Weight RMS Error: {current().stats.weightRmsError.toExponential(2)}</span>
            </div>
          )}
        </Show>
      </section>
    </main>
  );
}
