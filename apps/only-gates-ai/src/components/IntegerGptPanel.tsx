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
  type TinyGptCheckpoint,
  loadTinyGpt,
  compareTinyGpt,
  generateTinyGpt,
} from "../lib/tinyGpt";
import {
  TinyGptWasmModel,
  TinyGptWasmComparison,
  loadTinyGptWasm,
  generateTinyGptWasm,
  compareTinyGptWasm,
} from "../lib/tinyGptWasm";
import { TINY_GPT_CHECKPOINT } from "../assets/tiny-gpt-weights";
import { splitAnswer, splitInlineCode } from "../lib/chatFormat";
import BrowserTrainingPanel from "./BrowserTrainingPanel";
import {
  loadSolid2Wasm,
  createSolid2Decoder,
  type Solid2WasmModel,
  type Solid2Generation,
  type Solid2Decoder,
} from "../lib/tinyGptSolid2";
import { SOLID2_MODEL_BASE } from "../lib/solid2Config";

type ExampleId = "alice" | "solid2" | "custom";
type AliceMode = TinyGptMode | "wasm";
type PanelMode = AliceMode | "solid2-wasm" | "custom-wasm";

interface ChatMessage {
  readonly id: number;
  readonly role: "user" | "assistant";
  readonly text: string;
  readonly meta?: string;
}

const ALICE_SUGGESTIONS = [
  "Alice was",
  "the white rabbit",
  "she said",
  "the Queen",
];

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
  const [example, setExample] = createSignal<ExampleId>("alice");

  // --- Alice (word-level, small) ---
  const [checkpoint, setCheckpoint] =
    createSignal<TinyGptCheckpoint>(TINY_GPT_CHECKPOINT);
  const [model, setModel] = createSignal<TinyGptModel | null>(null);
  const [aliceMode, setAliceMode] = createSignal<AliceMode>("int8");
  const [loading, setLoading] = createSignal(false);
  const [status, setStatus] = createSignal<string | null>(null);
  const [comparison, setComparison] = createSignal<TinyGptComparison | null>(null);
  const [generating, setGenerating] = createSignal(false);
  const [wasm, setWasm] = createSignal<TinyGptWasmModel | null>(null);
  const [wasmLoading, setWasmLoading] = createSignal(false);
  const [wasmError, setWasmError] = createSignal<string | null>(null);
  const [wasmComparison, setWasmComparison] = createSignal<TinyGptWasmComparison | null>(null);

  // --- Solid 2 (BPE, ~1M params, WASM-primary) ---
  const [solid2, setSolid2] = createSignal<Solid2WasmModel | null>(null);
  const [customSolid2, setCustomSolid2] = createSignal<Solid2WasmModel | null>(
    null,
  );
  // --- Custom docs (same BPE/WASM engine, user-supplied corpus) ---
  const [custom, setCustom] = createSignal<Solid2WasmModel | null>(null);
  const [customAssistant, setCustomAssistant] = createSignal(false);
  const [solid2Loading, setSolid2Loading] = createSignal(false);
  const [solid2Error, setSolid2Error] = createSignal<string | null>(null);
  const [solid2Gen, setSolid2Gen] = createSignal<Solid2Generation | null>(null);
  const [solid2Streaming, setSolid2Streaming] = createSignal(false);
  let bundledSolid2: Solid2WasmModel | null = null;
  let bundledSolid2Loading = false;
  let solid2Decoder: Solid2Decoder | null = null;
  let solid2Timer: ReturnType<typeof setTimeout> | null = null;

  const [chat, setChat] = createSignal<readonly ChatMessage[]>([]);
  let nextMessageId = 1;
  let chatLog: HTMLDivElement | undefined;

  const pushMessage = (
    role: ChatMessage["role"],
    text: string,
    meta?: string,
  ): void => {
    setChat((previous) => [
      ...previous,
      { id: nextMessageId++, role, text, meta },
    ]);
  };

  const clearChat = (): void => {
    setChat([]);
  };

  const bpeMeta = (generation: Solid2Generation): string =>
    `${generation.newTokenCount} tokens · ${(
      (generation.newTokenCount / Math.max(generation.elapsedMs, 1)) *
      1000
    ).toFixed(0)} tok/s wall · ${generation.computeTokensPerSecond.toFixed(
      0,
    )} tok/s WASM${generation.stopped ? " · stopped" : ""}`;

  const stopSolid2 = (): void => {
    if (solid2Timer !== null) {
      clearTimeout(solid2Timer);
      solid2Timer = null;
    }
    solid2Decoder = null;
    setSolid2Streaming(false);
    setGenerating(false);
  };

  /** Stops the decoder and appends the partial answer to the chat. */
  const finishBpe = (finalize: boolean): void => {
    const decoder = solid2Decoder;
    stopSolid2();
    if (finalize && decoder) {
      const result = decoder.result();
      pushMessage(
        "assistant",
        result.text || "(empty generation)",
        bpeMeta(result),
      );
    }
    setSolid2Gen(null);
  };

  createEffect(
    () => ({ count: chat().length, live: solid2Gen()?.text }),
    () => {
      if (chatLog !== undefined) chatLog.scrollTop = chatLog.scrollHeight;
    },
  );

  const [prompt, setPrompt] = createSignal("Alice was");

  // Load (or reload) Alice whenever the active checkpoint changes.
  createEffect(
    checkpoint,
    (current) => {
      setLoading(true);
      setStatus("Loading transformer weights...");
      try {
        const loaded = loadTinyGpt(current);
        setModel(loaded);
        setStatus(null);
      } catch (e) {
        setModel(null);
        setStatus(`Error loading model: ${(e as Error).message}`);
      } finally {
        setLoading(false);
      }
    },
  );

  // Lazily build Alice WASM model.
  createEffect(
    () => ({ ex: example(), mode: aliceMode(), cp: checkpoint() }),
    (s) => {
      if (s.ex !== "alice" || s.mode !== "wasm" || wasm() || wasmLoading()) return;
      setWasmLoading(true);
      setWasmError(null);
      loadTinyGptWasm(s.cp)
        .then((loaded) => setWasm(loaded))
        .catch((e) => setWasmError(`WASM error: ${(e as Error).message}`))
        .finally(() => setWasmLoading(false));
    },
  );

  // Lazily load the bundled Solid 2 WASM model, or use a browser-trained one.
  createEffect(
    () => ({ example: example(), custom: customSolid2() }),
    (source) => {
      if (source.example !== "solid2") {
        finishBpe(false);
        return;
      }
      if (source.custom) {
        setSolid2(source.custom);
        setSolid2Error(null);
        setSolid2Loading(false);
        return;
      }
      if (bundledSolid2) {
        setSolid2(bundledSolid2);
        return;
      }
      if (bundledSolid2Loading) return;

      // Switch the prompt to a Solid 2 question when first entering.
      setPrompt((p) => (p === "Alice was" ? "What are split effects?" : p));
      bundledSolid2Loading = true;
      setSolid2Loading(true);
      setSolid2Error(null);
      loadSolid2Wasm(SOLID2_MODEL_BASE)
        .then((m) => {
          bundledSolid2 = m;
          if (!customSolid2()) setSolid2(m);
        })
        .catch((e) =>
          setSolid2Error(
            `Could not load Solid 2 model from ${SOLID2_MODEL_BASE}: ${(e as Error).message}. ` +
              `Train it with: colab run --gpu T4 scripts/train_tiny_gpt_solid2_colab.py -- --out /content/out ` +
              `then copy weights.bin, tokenizer.json, checkpoint.json to apps/only-gates-ai/public/models/tiny-gpt-solid2/ and rebuild.`,
          ),
        )
        .finally(() => {
          bundledSolid2Loading = false;
          setSolid2Loading(false);
        });
    },
  );

  const mode = (): PanelMode =>
    example() === "solid2"
      ? "solid2-wasm"
      : example() === "custom"
        ? "custom-wasm"
        : aliceMode();

  const activePrompts = (): readonly string[] => {
    if (example() === "custom") return custom()?.checkpoint.prompts ?? [];
    if (example() === "solid2") return solid2()?.checkpoint.prompts ?? [];
    return [];
  };

  const isCustomCheckpoint = createMemo(
    () => checkpoint() !== TINY_GPT_CHECKPOINT,
  );

  const handleUseCheckpoint = (next: TinyGptCheckpoint): void => {
    finishBpe(false);
    clearChat();
    setExample("alice");
    if (next === checkpoint()) {
      // The trained checkpoint is already active. Avoid clearing the loaded
      // model: setting the same signal value would not re-run the loader.
      return;
    }
    setCheckpoint(next);
    setModel(null);
    setWasm(null);
    setWasmError(null);
    setComparison(null);
    setWasmComparison(null);
  };

  const handleResetCheckpoint = (): void => {
    finishBpe(false);
    clearChat();
    setCheckpoint(TINY_GPT_CHECKPOINT);
    setModel(null);
    setWasm(null);
    setWasmError(null);
    setComparison(null);
    setWasmComparison(null);
  };

  const handleUseSolid2Model = (model: Solid2WasmModel): void => {
    finishBpe(false);
    clearChat();
    setCustomSolid2(model);
    setSolid2(model);
    setSolid2Error(null);
    setSolid2Gen(null);
    setPrompt(model.checkpoint.prompts?.[0] ?? "What are split effects?");
    setExample("solid2");
    setStatus(null);
  };

  const handleResetSolid2 = (): void => {
    finishBpe(false);
    clearChat();
    setCustomSolid2(null);
    setSolid2(null);
    setSolid2Gen(null);
    setSolid2Error(null);
    setPrompt("What are split effects?");
  };

  const handleUseCustomModel = (model: Solid2WasmModel): void => {
    finishBpe(false);
    clearChat();
    const assistant = model.checkpoint.meta?.training?.assistant === true;
    setCustomAssistant(assistant);
    setCustom(model);
    setSolid2Gen(null);
    setPrompt(
      assistant
        ? ""
        : (model.checkpoint.prompts?.[0] ?? "Write a short introduction."),
    );
    setExample("custom");
    setStatus(null);
  };

  const handleResetCustom = (): void => {
    finishBpe(false);
    clearChat();
    setCustomAssistant(false);
    setCustom(null);
    setSolid2Gen(null);
    setPrompt("Write a short introduction.");
    setStatus(null);
  };

  const activeConfig = createMemo(() => {
    if (example() === "solid2") return solid2()?.config ?? null;
    if (example() === "custom") return custom()?.config ?? null;
    return (aliceMode() === "wasm" ? wasm()?.config : model()?.config) ?? model()?.config ?? null;
  });

  const hasActiveModel = (): boolean =>
    example() === "solid2"
      ? solid2() !== null
      : example() === "custom"
        ? custom() !== null
        : aliceMode() === "wasm"
          ? wasm() !== null
          : model() !== null;

  const handleGenerate = (inputOverride?: string): void => {
    if (generating()) return;
    const input = (inputOverride ?? prompt()).trim();
    if (input.length === 0 || !hasActiveModel()) return;

    setStatus(null);
    setPrompt("");
    pushMessage("user", input);

    if (example() !== "alice") {
      const isCustomModel = example() === "custom";
      const m = isCustomModel ? custom() : solid2();
      if (!m) return;
      setSolid2Gen(null);
      setGenerating(true);
      setSolid2Streaming(true);
      let decoder: Solid2Decoder;
      try {
        decoder = createSolid2Decoder(m, input, {
          maxTokens: isCustomModel && customAssistant() ? 200 : 160,
          // Low temperature helps the small Q&A model stay on the learned
          // answer instead of drifting into a nearby documentation section.
          temperature: isCustomModel && customAssistant() ? 0.2 : 0.7,
          topK: 40,
          promptFormat:
            isCustomModel && !customAssistant() ? "text" : "chat",
        });
      } catch (e) {
        setGenerating(false);
        setSolid2Streaming(false);
        const message = `Generation error: ${(e as Error).message}`;
        setStatus(message);
        pushMessage("assistant", message);
        return;
      }
      solid2Decoder = decoder;
      // Run a tight integer-decode burst, then yield to the event loop. This is
      // the same schedule the reference worker uses: 40 ms of compute per turn
      // instead of one animation frame, so paint time no longer caps tok/s.
      const tick = () => {
        solid2Timer = null;
        if (solid2Decoder !== decoder) return;
        const deadline = performance.now() + 40;
        let done = false;
        while (performance.now() < deadline) {
          if (decoder.step().done) {
            done = true;
            break;
          }
        }
        setSolid2Gen(decoder.result());
        if (done) {
          finishBpe(true);
          return;
        }
        solid2Timer = setTimeout(tick, 0);
      };
      solid2Timer = setTimeout(tick, 0);
      return;
    }

    const requestExample = example();
    setGenerating(true);
    setTimeout(() => {
      try {
        if (example() !== requestExample) return;
        if (aliceMode() === "wasm") {
          const current = wasm();
          if (!current) throw new Error("The WASM model is still loading.");
          const result = generateTinyGptWasm(current, input, {
            maxTokens: 24,
            temperature: 0.8,
            topK: 40,
          });
          pushMessage(
            "assistant",
            result.text || "(empty generation)",
            `${result.steps} tokens in ${result.elapsedMs.toFixed(1)}ms · WASM`,
          );
        } else {
          const current = model();
          if (!current) throw new Error("The model is still loading.");
          const result = generateTinyGpt(
            current,
            input,
            aliceMode() as TinyGptMode,
            { maxTokens: 24, temperature: 0.8, topK: 40 },
          );
          pushMessage(
            "assistant",
            result.text || "(empty generation)",
            `${result.steps} tokens in ${result.elapsedMs.toFixed(1)}ms · ${result.mode}`,
          );
        }
      } catch (e) {
        const message = `Generation error: ${(e as Error).message}`;
        setStatus(message);
        pushMessage("assistant", message);
      } finally {
        setGenerating(false);
      }
    }, 10);
  };

  const lastUserPrompt = (): string => {
    const messages = chat();
    for (let index = messages.length - 1; index >= 0; index--) {
      const message = messages[index]!;
      if (message.role === "user") return message.text;
    }
    return prompt();
  };

  const handleCompare = () => {
    if (example() !== "alice") {
      setStatus("Comparison against a float baseline is not enabled for the 4k-vocab BPE model (JS float/int8 inference is too slow in-browser). Use the Alice example for float-vs-integer comparison.");
      return;
    }
    const query = prompt().trim().length > 0 ? prompt() : lastUserPrompt();
    if (query.trim().length === 0) return;
    if (aliceMode() === "wasm") {
      const wasmModel = wasm();
      const tsModel = model();
      if (!wasmModel || !tsModel) return;
      try { setWasmComparison(compareTinyGptWasm(wasmModel, tsModel, query)); }
      catch (e) { setStatus(`Comparison error: ${(e as Error).message}`); }
      return;
    }
    const current = model();
    if (!current) return;
    try { setComparison(compareTinyGpt(current, query)); }
    catch (e) { setStatus(`Comparison error: ${(e as Error).message}`); }
  };

  const generateDisabled = (): boolean => generating() || !hasActiveModel();
  const sendDisabled = (): boolean =>
    generateDisabled() || prompt().trim().length === 0;

  const chatSuggestions = (): readonly string[] =>
    (example() === "alice" ? ALICE_SUGGESTIONS : activePrompts()).slice(0, 5);

  const activeModelLabel = (): string =>
    example() === "alice"
      ? "Alice"
      : example() === "solid2"
        ? "Solid 2"
        : customAssistant()
          ? "custom assistant"
          : "custom model";

  const liveAnswerText = (): string => {
    if (example() === "alice") return "";
    return solid2Gen()?.text ?? "";
  };

  const liveAnswerMeta = (): string | undefined => {
    if (example() === "alice") return undefined;
    const generation = solid2Gen();
    return generation ? bpeMeta(generation) : undefined;
  };

  const handleComposerKeyDown = (event: KeyboardEvent): void => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      handleGenerate();
    }
  };

  const aliceComparison = createMemo(() => {
    if (example() !== "alice") return null;
    if (aliceMode() === "wasm") {
      const c = wasmComparison(); if (!c) return null;
      return { leftLabel: "Float32 Top 5", rightLabel: "WASM Integer Top 5", leftTop: c.floatTop, rightTop: c.wasmTop, topOneMatches: c.topOneMatches, overlapAtFive: c.overlapAtFive, meanProbabilityDelta: c.meanProbabilityDelta };
    }
    const c = comparison(); if (!c) return null;
    return { leftLabel: "Float32 Top 5", rightLabel: "Int8 Top 5", leftTop: c.floatTop, rightTop: c.int8Top, topOneMatches: c.topOneMatches, overlapAtFive: c.overlapAtFive, meanProbabilityDelta: c.meanProbabilityDelta };
  });

  const renderAnswer = (text: string) => (
    <For each={splitAnswer(text)}>
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
                  <code class="chat-inline-code">{part.text}</code>
                ) : (
                  part.text
                )
              }
            </For>
          </p>
        )
      }
    </For>
  );

  return (
    <main class="workspace language-workspace">
      <div class="language-toolbar" style="flex-wrap: wrap; gap: 8px;">
        <div class="mode-switch" role="group" aria-label="Example">
          <button
            type="button"
            class={{ active: example() === "alice" }}
            onClick={() => {
              finishBpe(false);
              clearChat();
              setExample("alice");
              setPrompt("Alice was");
              setStatus(null);
            }}
          >
            Alice (word)
          </button>
          <button
            type="button"
            class={{ active: example() === "solid2" }}
            onClick={() => {
              finishBpe(false);
              clearChat();
              setExample("solid2");
              setPrompt("What are split effects?");
              setStatus(null);
            }}
          >
            Solid 2 Docs (BPE)
          </button>
          <button
            type="button"
            class={{ active: example() === "custom" }}
            onClick={() => {
              finishBpe(false);
              clearChat();
              setExample("custom");
              setPrompt(
                customAssistant()
                  ? ""
                  : (custom()?.checkpoint.prompts?.[0] ??
                      "Write a short introduction."),
              );
              setStatus(null);
            }}
          >
            Custom Docs (BPE)
          </button>
        </div>
        <Show when={example() === "alice"}>
          <div class="mode-switch" role="group" aria-label="Inference mode">
            <button type="button" class={{ active: aliceMode() === "int8" }} onClick={() => setAliceMode("int8")}>Int8</button>
            <button type="button" class={{ active: aliceMode() === "float" }} onClick={() => setAliceMode("float")}>Float32</button>
            <button type="button" class={{ active: aliceMode() === "wasm" }} onClick={() => setAliceMode("wasm")}>Integer-only (WASM)</button>
          </div>
        </Show>
        <Show when={example() === "solid2"}>
          <span class="muted">
            Integer-only {solid2()?.backend === "wasm-simd" ? "WebAssembly SIMD" : "WebAssembly"}, Q16/int8, BPE 4096, ctx 256.
            {" "}TS float/int8 disabled (too slow at 4k vocab in JS).
          </span>
        </Show>
        <Show when={example() === "custom"}>
          <span class="muted">
            {custom()
              ? `Integer-only ${
                  custom()!.backend === "wasm-simd"
                    ? "WebAssembly SIMD"
                    : "WebAssembly"
                }, BPE 4096, ctx 256 · ${
                  customAssistant() ? "assistant (Q&A)" : "text continuation"
                }.`
              : "Train a model on your own Markdown docs below, then use it here."}
          </span>
        </Show>
      </div>

      <section class="card language-card">
        <div class="card-heading compact">
          <div>
            <span class="eyebrow">Integer GPT</span>
            <h2>
              {example() === "alice"
                ? "Alice in Wonderland"
                : example() === "custom"
                  ? "Custom Docs"
                  : "Solid 2 Docs Q&A"}
            </h2>
          </div>
          <Show when={activeConfig()}>
            {(c) => (
              <div class="runtime-pill ready">
                <span class="runtime-dot" />
                {parameterCount(c()).toLocaleString()} params
                <Show when={example() === "solid2" && solid2()}>
                  {solid2()!.backend === "wasm-simd" ? " · SIMD" : " · scalar"}
                </Show>
                <Show when={example() === "custom" && custom()}>
                  {custom()!.backend === "wasm-simd" ? " · SIMD" : " · scalar"}
                </Show>
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
          <a
            href={
              import.meta.env.DEV
                ? "http://localhost:5175/deep-dive/"
                : "learn/deep-dive/"
            }
          >
            Build the integer engine from scratch →
          </a>
        </p>

        <p class="hint">
          <a
            href={
              import.meta.env.DEV
                ? "http://localhost:5175/deep-dive/#beyond"
                : "learn/deep-dive/#beyond"
            }
          >
            How this lowers to FPGA gates →
          </a>
        </p>

        <p class="hint">
          <Show when={example() === "alice"}>
            Tiny transformer trained on Chapters I &amp; II of Alice's Adventures in Wonderland (word-level). Float32 is used only to quantize at load; inference modes show float/int8/fully-integer WASM behavior.
          </Show>
          <Show when={example() === "solid2"}>
            Decoder-only transformer (byte-level BPE, 4096 vocab, 256 ctx; config sized to the small docs+Q&A corpus) trained from scratch on <b>pinned Solid 2 docs + curated Q&A</b> via <code>scripts/train_tiny_gpt_solid2_colab.py</code> (runnable with the Termux <code>colab</code> CLI). The browser runs the <b>entire forward pass as integer-only WebAssembly</b> — {solid2()?.backend === "wasm-simd" ? "the SIMD lane" : "the scalar lane"} (Q16 activations, int8 weights, integer RMSNorm/GELU/DI-Exp softmax). The prompt is prefilled once, then each token costs one context row against an integer KV cache, which is the schedule an FPGA datapath would use. Float32 is used only once to quantize the checkpoint at load.
            <Show when={activeConfig()}>{(c) => <span> Loaded config: {c().vocabSize}v/{c().context}ctx, {c().dim}d/{c().layers}L/{c().heads}H/{c().mlp}mlp ({parameterCount(c()).toLocaleString()} params).</span>}</Show>
          </Show>
          <Show when={example() === "custom"}>
            Decoder-only transformer with the same 4096-token BPE vocabulary and
            256-context architecture as the Solid 2 demo, trained in your browser
            on the Markdown, text, or HTML pages you list below. Warm-starting
            from the bundled checkpoint is optional; the result is quantized and runs
            through the same <b>integer-only WebAssembly</b> engine (Q16
            activations, int8 weights).
            <Show when={activeConfig()}>{(c) => <span> Loaded config: {c().vocabSize}v/{c().context}ctx, {c().dim}d/{c().layers}L/{c().heads}H/{c().mlp}mlp ({parameterCount(c()).toLocaleString()} params).</span>}</Show>
          </Show>
        </p>

        <Show when={status()}><div class="banner info"><span>{status()}</span></div></Show>
        <Show when={example() === "alice" && isCustomCheckpoint()}>
          <div class="banner success">
            <span>
              Using a checkpoint trained in this browser. The demo quantized it
              at load; inference is still int8 or integer-only WASM.
            </span>
          </div>
        </Show>
        <Show when={example() === "solid2" && customSolid2() !== null}>
          <div class="banner success">
            <span>
              Using a Solid 2 checkpoint trained in this browser. It was
              quantized to int8/Q16 for the integer-only WASM path.
            </span>
          </div>
        </Show>
        <Show when={example() === "custom" && custom() !== null}>
          <div class="banner success">
            <span>
              Using a custom checkpoint trained in this browser. It was
              quantized to int8/Q16 for the integer-only WASM path.
            </span>
          </div>
        </Show>
        <Show when={example() === "custom" && custom() === null}>
          <div class="banner info">
            <span>
              No custom model is loaded yet. Add document URLs in the training
              panel below, train, then click <b>Use in Integer GPT</b>.
            </span>
          </div>
        </Show>
        <Show when={example() === "alice" && aliceMode() === "wasm" && wasmLoading()}><div class="banner info"><span>Loading WebAssembly integer kernels...</span></div></Show>
        <Show when={example() === "alice" && aliceMode() === "wasm" && wasmError()}><div class="banner warning"><span>{wasmError()}</span></div></Show>
        <Show when={example() === "solid2" && solid2Loading()}><div class="banner info"><span>Loading Solid 2 model (weights.bin + tokenizer + WASM quantization)...</span></div></Show>
        <Show when={example() === "solid2" && solid2Error()}><div class="banner warning"><span style="white-space: pre-wrap;">{solid2Error()}</span></div></Show>

        <div
          class="chat-log"
          role="log"
          aria-live="polite"
          aria-label="Integer GPT chat"
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
                  {hasActiveModel()
                    ? `Chat with the ${activeModelLabel()}`
                    : `No ${activeModelLabel()} model loaded`}
                </strong>
                <span>
                  {hasActiveModel()
                    ? example() === "alice"
                      ? "Ask the word-level model to continue a phrase, or start with a suggestion."
                      : customAssistant()
                        ? "Ask a question about your documents, or start with a suggestion."
                        : "Type a prompt to continue from your documents, or start with a suggestion."
                    : "Load or train a model first; the training panel is below."}
                </span>
                <Show when={hasActiveModel() && chatSuggestions().length > 0}>
                  <div class="completion-list">
                    <For each={chatSuggestions()}>
                      {(suggestion) => (
                        <button
                          type="button"
                          class="completion-chip"
                          onClick={() => handleGenerate(suggestion)}
                        >
                          <span class="completion-word">{suggestion}</span>
                        </button>
                      )}
                    </For>
                  </div>
                </Show>
              </div>
            }
          >
            <For each={chat()}>
              {(message) => (
                <div class={`chat-message ${message.role}`}>
                  <div class="chat-bubble">
                    <span class="chat-role">
                      {message.role === "user" ? "You" : "Model"}
                    </span>
                    {renderAnswer(message.text)}
                    <Show when={message.meta}>
                      <div class="chat-meta">
                        <span>{message.meta}</span>
                      </div>
                    </Show>
                  </div>
                </div>
              )}
            </For>
          </Show>

          <Show when={generating()}>
            <div class="chat-message assistant">
              <div class="chat-bubble">
                <span class="chat-role">Model</span>
                <Show
                  when={liveAnswerText()}
                  fallback={<p class="muted">Thinking…</p>}
                >
                  {renderAnswer(liveAnswerText())}
                </Show>
                <Show when={liveAnswerMeta()}>
                  {(meta) => (
                    <div class="chat-meta">
                      <span>{meta()}</span>
                      <Show when={solid2Streaming()}>
                        <span>streaming</span>
                      </Show>
                    </div>
                  )}
                </Show>
              </div>
            </div>
          </Show>
        </div>

        <div class="chat-composer">
          <textarea
            class="text-area chat-input"
            aria-label="Chat message"
            rows="2"
            spellcheck={false}
            disabled={generating() || !hasActiveModel()}
            placeholder={
              !hasActiveModel()
                ? "Load a model first…"
                : example() === "solid2"
                  ? "Ask a Solid 2 question…"
                  : example() === "custom"
                    ? customAssistant()
                      ? "Ask about your documents…"
                      : "Continue from your documents…"
                    : "Ask the Alice model…"
            }
            value={prompt()}
            onInput={(event) => setPrompt(event.currentTarget.value)}
            onKeyDown={handleComposerKeyDown}
          />
          <button
            type="button"
            class="button primary chat-send"
            disabled={sendDisabled()}
            onClick={() => handleGenerate()}
          >
            Send
          </button>
          <Show when={example() !== "alice" && solid2Streaming()}>
            <button
              type="button"
              class="button secondary chat-send"
              onClick={() => finishBpe(true)}
            >
              Stop
            </button>
          </Show>
        </div>
        <p class="chat-hint">Enter to send · Shift+Enter for a new line</p>

        <Show when={example() === "alice"}>
          <div class="button-row">
            <button
              type="button"
              class="button ghost"
              onClick={handleCompare}
              disabled={generating() || !model()}
            >
              Compare float vs int8
            </button>
          </div>
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

      <Show when={example() === "alice"}>
        <BrowserTrainingPanel
          variant="alice"
          isCustomCheckpoint={isCustomCheckpoint()}
          onUseCheckpoint={handleUseCheckpoint}
          onResetCheckpoint={handleResetCheckpoint}
          onBundleStatus={setStatus}
        />
      </Show>
      <Show when={example() === "solid2"}>
        <BrowserTrainingPanel
          variant="solid2"
          isCustomCheckpoint={customSolid2() !== null}
          onUseSolid2Model={handleUseSolid2Model}
          onResetCheckpoint={handleResetSolid2}
          onBundleStatus={setStatus}
        />
      </Show>
      <Show when={example() === "custom"}>
        <BrowserTrainingPanel
          variant="custom"
          isCustomCheckpoint={custom() !== null}
          onUseCustomModel={handleUseCustomModel}
          onResetCheckpoint={handleResetCustom}
          onBundleStatus={setStatus}
        />
      </Show>
    </main>
  );
}
