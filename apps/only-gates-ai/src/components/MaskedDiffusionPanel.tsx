import {
  For,
  Show,
  createEffect,
  createMemo,
  createSignal,
  onCleanup,
} from "solid-js";

import {
  createMaskedDiffusionDecoder,
  loadMaskedDiffusionWasm,
  type MaskedDiffusionDecoder,
  type MaskedDiffusionModel,
  type MaskedDiffusionSnapshot,
} from "../lib/maskedDiffusion";
import { LUAU_PRESETS, type LuauMode } from "../lib/luauConfig";
import {
  SOLID2_MASKED_PRESETS as SOLID2_PRESETS,
  SOLID2_MASKED_MODEL_BASE as SOLID2_MODEL_BASE,
  solid2MaskedParameterCount as solid2ParameterCount,
} from "../lib/solid2MaskedConfig";

import MaskedDiffusionTrainingPanel from "./MaskedDiffusionTrainingPanel";
interface MaskedDiffusionPanelProps {
  /** The tab is selected. The model loads lazily on first open. */
  readonly active: boolean;
}

const MASKED_MODEL_BASE = `${import.meta.env.BASE_URL}models/masked-diffusion-luau/`;
const SOLID2_MASKED_MODEL_BASE = SOLID2_MODEL_BASE;

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

export default function MaskedDiffusionPanel(props: MaskedDiffusionPanelProps) {
  const [subTab, setSubTab] = createSignal<"luau" | "solid2">("luau");
  const [mode, setMode] = createSignal<LuauMode>("write");
  // Start on a prompt the denoiser handles cleanly; the first preset ("Sum
  // first 50") is a harder case for left-to-right block filling.
  const [prompt, setPrompt] = createSignal(LUAU_PRESETS[1]!.prompt);
  const [model, setModel] = createSignal<MaskedDiffusionModel | null>(null);
  const [loading, setLoading] = createSignal(false);
  const [error, setError] = createSignal<string | null>(null);
  const [steps, setSteps] = createSignal(8);
  // Code denoising is much more coherent when sampling stays near the argmax:
  // the dataset randomizes variable names, and hot sampling mixes them.
  const [temperature, setTemperature] = createSignal(0.01);
  const [generating, setGenerating] = createSignal(false);
  const [snapshot, setSnapshot] = createSignal<MaskedDiffusionSnapshot | null>(
    null,
  );
  const [preview, setPreview] = createSignal("");
  const [output, setOutput] = createSignal("");
  const [copied, setCopied] = createSignal(false);
  const [reload, setReload] = createSignal(0);
  const [notice, setNotice] = createSignal<string | null>(null);

  let decoder: MaskedDiffusionDecoder | null = null;
  let timer: ReturnType<typeof setTimeout> | null = null;
  let loadInFlight = false;

  const visiblePresets = createMemo(() => {
    if (subTab() === "solid2") {
      return SOLID2_PRESETS;
    }
    return LUAU_PRESETS.filter((preset) => preset.mode === mode());
  });

  const activeConfig = createMemo(() => model()?.config);
  const activeParameters = createMemo(() => {
    const m = model();
    if (!m) return 0;
    if (subTab() === "solid2") {
      return solid2ParameterCount(m.config);
    }
    return m.parameterCount;
  });
  const maskId = createMemo(() => model()?.maskId ?? -1);

  /*
   * Solid 2 RC has no `Index`, and `For` keys by value. Mask ids repeat for
   * every unfilled slot, so wrap each cell in a unique object instead of
   * rendering the raw ids.
   */
  const answerCells = createMemo<Array<{ slot: number; id: number }>>(() => {
    const current = snapshot();
    if (!current) return [];
    const ids = current.tokens.subarray(
      current.promptLength,
      current.length,
    );
    return Array.from(ids, (id, slot) => ({ slot, id }));
  });

  const progress = createMemo(() => {
    const current = snapshot();
    if (!current || current.initialMasks === 0) return 0;
    // The decoder stops as soon as `<|end|>` appears, which can leave layout
    // masks after the answer. Treat a completed answer as fully progressed.
    if (current.done) return 1;
    return Math.min(
      1,
      (current.initialMasks - current.remaining) / current.initialMasks,
    );
  });

  const statusLabel = createMemo(() => {
    if (generating()) {
      const current = snapshot();
      return current
        ? `Denoising ${current.step}/${current.totalSteps} · ${current.remaining} masks`
        : "Starting denoiser";
    }
    if (output()) return "Answer complete";
    if (model()) {
      if (subTab() === "solid2") {
        return `Solid 2 model — ${activeParameters().toLocaleString()} parameters`;
      }
      return "Integer WASM ready";
    }
    if (loading()) return "Loading model";
    if (error()) return "Model missing";
    return "Idle";
  });

  const stop = (): void => {
    if (timer !== null) {
      clearTimeout(timer);
      timer = null;
    }
    decoder = null;
    setGenerating(false);
  };

  onCleanup(stop);

  /**
   * Called by the training panel when the user clicks "Use this model".
   * Swap the active denoiser, reset any in-flight generation so it cannot
   * keep drawing from the old model, and pick up the new checkpoint's
   * recommended schedule.
   */
  const handleUseModel = (next: MaskedDiffusionModel): void => {
    stop();
    setSnapshot(null);
    setPreview("");
    setOutput("");
    setCopied(false);
    setError(null);
    setModel(next);
    setSteps(Math.max(1, Math.min(32, next.checkpoint.diffusion?.steps ?? 8)));
    setNotice(
      `Using ${next.checkpoint.meta?.name ?? "the trained model"} in this demo.`,
    );
  };

  const modelBase = createMemo(() =>
    subTab() === "solid2" ? SOLID2_MASKED_MODEL_BASE : MASKED_MODEL_BASE,
  );

  createEffect(
    () => ({ active: props.active, attempt: reload(), subTab: subTab() }),
    ({ active, attempt, subTab }) => {
      if (!active || attempt < 0 || loadInFlight) return;
      loadInFlight = true;
      setLoading(true);
      setError(null);
      setModel(null);
      loadMaskedDiffusionWasm(modelBase())
        .then((loaded) => {
          setModel(loaded);
          setSteps(
            Math.max(
              1,
              Math.min(32, loaded.checkpoint.diffusion?.steps ?? 8),
            ),
          );
          setError(null);
        })
        .catch((reason: unknown) => {
          setError(
            `Could not load the masked-diffusion model from ${modelBase()}: ${
              (reason as Error).message
            }. Train it with the appropriate training script, then copy ` +
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
    setPreview("");
    setSnapshot(null);
    setCopied(false);
    setNotice(null);
    setGenerating(true);

    let current: MaskedDiffusionDecoder;
    try {
      current = createMaskedDiffusionDecoder(currentModel, input, {
        steps: steps(),
        temperature: temperature(),
        mode: subTab() === "solid2" ? "ask" : mode(),
      });
    } catch (reason) {
      setGenerating(false);
      setError(`Generation error: ${(reason as Error).message}`);
      return;
    }

    decoder = current;
    setSnapshot(current.snapshot());
    setPreview(current.snapshot().preview);

    const tick = (): void => {
      timer = null;
      if (decoder !== current) return;
      try {
        // One full bidirectional pass per frame. The UI paints between
        // passes, so the mask strip and text preview animate live.
        const next = current.step();
        setSnapshot(next);
        setPreview(next.preview);
        if (next.done) {
          decoder = null;
          setOutput(next.answer);
          setGenerating(false);
          return;
        }
        timer = setTimeout(tick, 16);
      } catch (reason) {
        decoder = null;
        setGenerating(false);
        setError(`Generation error: ${(reason as Error).message}`);
      }
    };
    timer = setTimeout(tick, 16);
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

  const choosePreset = (preset: any): void => {
    setPrompt(preset.prompt);
    setError(null);
  };

  return (
    <section class="card masked-panel">
      <div class="card-heading compact">
        <div>
          <span class="eyebrow">Masked diffusion</span>
          <h2>
            {subTab() === "solid2"
              ? "Denoise a Solid 2 answer in parallel"
              : "Denoise a Luau answer in parallel"}
          </h2>
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
          {statusLabel()}
        </div>
      </div>

      <p class="hint">
        A bidirectional transformer fills a whole answer region of{" "}
        <code>&lt;|mask|&gt;</code> tokens. Every denoising step is one
        integer-only WASM pass: the C sampler fills the next block of masked
        positions left to right. The text and token strip below update after
        each step.
      </p>

      <div class="luau-mode" role="tablist" aria-label="Masked diffusion subtab">
        <button
          type="button"
          role="tab"
          aria-selected={subTab() === "luau" ? "true" : "false"}
          class={{ active: subTab() === "luau" }}
          onClick={() => {
            setSubTab("luau");
            setMode("write");
            setPrompt(LUAU_PRESETS[1]!.prompt);
            setReload((v) => v + 1);
          }}
        >
          Luau
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={subTab() === "solid2" ? "true" : "false"}
          class={{ active: subTab() === "solid2" }}
          onClick={() => {
            setSubTab("solid2");
            setPrompt(SOLID2_PRESETS[1]!.prompt);
            setReload((v) => v + 1);
          }}
        >
          Solid 2
        </button>
      </div>
      <Show when={subTab() === "luau"}>
        <div class="luau-mode" role="tablist" aria-label="Masked answer mode">
          <button
            type="button"
            role="tab"
            aria-selected={mode() === "write" ? "true" : "false"}
            class={{ active: mode() === "write" }}
            onClick={() => {
              setMode("write");
              setPrompt(LUAU_PRESETS[1]!.prompt);
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
      </Show>

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
        {subTab() === "solid2" ? "Question" : mode() === "write" ? "Task" : "Question"}
        <textarea
          rows={4}
          value={prompt()}
          disabled={generating()}
          onInput={(event) => setPrompt(event.currentTarget.value)}
          onKeyDown={handleKeyDown}
          placeholder={
            subTab() === "solid2"
              ? "Ask a question about Solid 2..."
              : mode() === "write"
                ? "Write a for loop that sums the first 50 elements of `values`..."
                : "How do I write a for loop in Luau?"
          }
        />
      </label>

      <div class="masked-controls">
        <label class="masked-control">
          <span>Denoising steps</span>
          <select
            value={String(steps())}
            disabled={generating()}
            onChange={(event) => setSteps(Number(event.currentTarget.value))}
          >
            <option value="4">4 · fast</option>
            <option value="6">6</option>
            <option value="8">8 · balanced</option>
            <option value="12">12</option>
            <option value="16">16 · quality</option>
          </select>
        </label>
        <label class="masked-control masked-temperature">
          <span>
            Temperature <strong>{temperature().toFixed(2)}</strong>
          </span>
          <input
            type="range"
            min="0.01"
            max="1.2"
            step="0.05"
            value={temperature()}
            disabled={generating()}
            onInput={(event) =>
              setTemperature(Number(event.currentTarget.value))
            }
          />
        </label>
      </div>

      <div class="button-row">
        <button
          type="button"
          class="button primary"
          disabled={generating() || !model() || prompt().trim().length === 0}
          onClick={handleGenerate}
        >
          {generating()
            ? "Denoising..."
            : subTab() === "solid2"
              ? "Denoise Solid 2"
              : mode() === "write"
                ? "Denoise Luau"
                : "Denoise answer"}
        </button>
        <Show when={generating()}>
          <button type="button" class="button ghost" onClick={stop}>
            Stop
          </button>
        </Show>
        <span class="chat-hint">
          {mode() === "write"
            ? "Ctrl/Cmd + Enter to denoise"
            : "The model sees the prompt and fills the answer region"}
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
            Loading the masked-diffusion model. The float checkpoint is
            quantized to int8 once on first open.
          </span>
        </div>
      </Show>

      <Show when={notice()}>
        <div class="banner info">
          <span>{notice()}</span>
        </div>
      </Show>

      <Show when={snapshot()}>
        {(current) => (
          <div class="masked-animation">
            <div class="masked-animation-head">
              <strong>Live denoising</strong>
              <span>
                {current().step}/{current().totalSteps} steps ·{" "}
                {current().remaining} mask
                {current().remaining === 1 ? "" : "s"} left
              </span>
            </div>
            <div
              class="masked-progress"
              role="progressbar"
              aria-valuemin={0}
              aria-valuemax={current().initialMasks}
              aria-valuenow={current().initialMasks - current().remaining}
            >
              <div
                class="masked-progress-fill"
                style={{ width: `${(progress() * 100).toFixed(1)}%` }}
              />
            </div>
            <div class="masked-token-strip" aria-hidden="true">
              <For each={answerCells()}>
                {(cell) => (
                  <span
                    class={{
                      "masked-token": true,
                      filled: cell.id !== maskId(),
                      masked: cell.id === maskId(),
                    }}
                  />
                )}
              </For>
            </div>
            <pre class="masked-preview" tabindex="0">
              <code>{preview() || "▁"}</code>
            </pre>
            <div class="chat-meta">
              {current().elapsedMs.toFixed(0)} ms wall ·{" "}
              {current().computeMs.toFixed(0)} ms WASM ·{" "}
              {current().answerLength} answer slots
            </div>
          </div>
        )}
      </Show>

      <Show when={output()}>
        <div class="luau-output">
          <div class="luau-output-head">
            <strong>
              {subTab() === "solid2"
                ? "Denoised Solid 2"
                : mode() === "write"
                  ? "Denoised Luau"
                  : "Denoised answer"}
            </strong>
            <button
              type="button"
              class="button ghost"
              onClick={() => void handleCopy()}
            >
              {copied() ? "Copied" : "Copy"}
            </button>
          </div>
          <pre class="chat-code" tabindex="0">
            <code>{output()}</code>
          </pre>
        </div>
      </Show>
      <Show when={subTab()} keyed>
        {(tab) => (
          <MaskedDiffusionTrainingPanel
            active={props.active}
            onUseModel={handleUseModel}
            variant={tab === "solid2" ? "solid2" : "luau"}
          />
        )}
      </Show>

      <Show when={model()}>
        <p class="hint luau-model-meta">
          {activeParameters().toLocaleString()} parameters ·{" "}
          {activeConfig()!.vocabSize.toLocaleString()} BPE tokens (including
          `&lt;|mask|&gt;`) · {activeConfig()!.context}-token context ·{" "}
          {model()!.backend === "wasm-simd" ? "WASM SIMD" : "WASM scalar"} ·{" "}
          {model()!.checkpoint.meta?.name ?? (subTab() === "solid2"
            ? "Solid 2 masked diffusion"
            : "masked-diffusion-luau")}
        </p>
      </Show>
    </section>
  );
}
