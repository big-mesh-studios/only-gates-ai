import { createSignal, onCleanup, onSettled, Show } from "solid-js";

import corpus from "../assets/tiny-gpt-corpus.txt?raw";
import {
  deleteBrowserTrainingState,
  loadBrowserTrainingState,
  saveBrowserTrainingState,
  type BrowserTrainingRunKey,
} from "../lib/browserTrainingStorage";
import {
  createSolid2Decoder,
  type Solid2WasmModel,
} from "../lib/tinyGptSolid2";
import {
  generateTinyGpt,
  loadTinyGpt,
  type TinyGptCheckpoint,
} from "../lib/tinyGpt";
import {
  SOLID2_TRAIN_CONFIG,
  solid2ParameterCount,
} from "../lib/solid2Config";
import {
  buildSolid2WasmFromTraining,
  trainSolid2Gpt,
  type Solid2TrainingResult,
} from "../lib/trainSolid2Gpt";
import {
  buildCustomWasmFromTraining,
  parseDocUrls,
  trainCustomGpt,
  type CustomTrainingResult,
} from "../lib/trainCustomGpt";
import {
  DEFAULT_BROWSER_TRAIN_CONFIG,
  trainTinyGpt,
  type BrowserTrainingProgress,
  type BrowserTrainingResult,
  type BrowserTrainingState,
  type TrainingPoint,
} from "../lib/trainTinyGpt";
import LossChart from "./LossChart";

type BrowserTrainingVariant = "alice" | "solid2" | "custom";

interface BrowserTrainingPanelProps {
  readonly variant?: BrowserTrainingVariant;
  readonly isCustomCheckpoint: boolean;
  readonly onUseCheckpoint?: (checkpoint: TinyGptCheckpoint) => void;
  readonly onUseSolid2Model?: (model: Solid2WasmModel) => void;
  readonly onUseCustomModel?: (model: Solid2WasmModel) => void;
  readonly onResetCheckpoint: () => void;
}

function backendLabel(backend: string | null): string {
  if (backend === "webgpu") return "WebGPU";
  if (backend === "webgl") return "WebGL (WebGPU unavailable)";
  if (backend === "cpu") return "CPU (GPU unavailable)";
  return "Detected on start";
}

export default function BrowserTrainingPanel(
  props: BrowserTrainingPanelProps,
) {
  const isSolid2 = (): boolean => props.variant === "solid2";
  const isCustom = (): boolean => props.variant === "custom";
  const isLarge = (): boolean => isSolid2() || isCustom();
  const runKey: BrowserTrainingRunKey = isSolid2()
    ? "solid2"
    : isCustom()
      ? "custom"
      : "current";
  const defaultSteps = isSolid2() ? 400 : isCustom() ? 600 : 1800;
  const defaultBatch = isLarge() ? 4 : 8;
  const defaultLearningRate = isLarge() ? 0.0004 : 0.003;
  const defaultDropout = isLarge() ? 0.1 : 0;
  const defaultMaxMinutes = isLarge() ? 30 : 0;

  const [history, setHistory] = createSignal<readonly TrainingPoint[]>([]);
  const [progress, setProgress] = createSignal<BrowserTrainingProgress | null>(
    null,
  );
  const [trained, setTrained] = createSignal<BrowserTrainingResult | null>(null);
  const [solid2Trained, setSolid2Trained] =
    createSignal<Solid2TrainingResult | null>(null);
  const [solid2Model, setSolid2Model] = createSignal<Solid2WasmModel | null>(
    null,
  );
  const [customTrained, setCustomTrained] =
    createSignal<CustomTrainingResult | null>(null);
  const [customModel, setCustomModel] = createSignal<Solid2WasmModel | null>(
    null,
  );
  const [saved, setSaved] = createSignal<BrowserTrainingState | null>(null);
  const [busy, setBusy] = createSignal(false);
  const [error, setError] = createSignal<string | null>(null);
  const [status, setStatus] = createSignal<string | null>(null);
  const [backend, setBackend] = createSignal<string | null>(null);
  const [sample, setSample] = createSignal<{
    readonly prompt: string;
    readonly text: string;
  } | null>(null);

  const [steps, setSteps] = createSignal(defaultSteps);
  const [batch, setBatch] = createSignal(defaultBatch);
  const [learningRate, setLearningRate] = createSignal(defaultLearningRate);
  const [dropout, setDropout] = createSignal(defaultDropout);
  const [seed, setSeed] = createSignal(1337);
  const [maxMinutes, setMaxMinutes] = createSignal(defaultMaxMinutes);
  const [warmStart, setWarmStart] = createSignal(isLarge());
  const [urls, setUrls] = createSignal("");

  let cancelRequested = false;
  let saveChain: Promise<void> = Promise.resolve();

  onCleanup(() => {
    // Switching examples unmounts this panel. Stop the loop so it does not keep
    // the GPU busy in the background; the training loop's normal shutdown path
    // saves the last checkpoint.
    cancelRequested = true;
  });

  const persist = (state: BrowserTrainingState): void => {
    setSaved(state);
    saveChain = saveChain
      .then(() => saveBrowserTrainingState(state, runKey))
      .catch((reason: unknown) => {
        setError(
          `Training continues, but the run could not be saved: ${
            (reason as Error).message
          }`,
        );
      });
  };

  onSettled(() => {
    void loadBrowserTrainingState(runKey)
      .then((state) => {
        if (!state) return;
        setSaved(state);
        setHistory(state.history);
        setSteps(
          Math.min(
            20000,
            Math.max(state.steps, state.step + (isLarge() ? 100 : 200)),
          ),
        );
        setBatch(state.batch);
        setLearningRate(state.lr);
        setDropout(state.dropout);
        setSeed(state.seed);
        if (isCustom() && state.sourceUrls && state.sourceUrls.length > 0) {
          setUrls(state.sourceUrls.join("\n"));
        }
        setStatus(
          `Saved run found: ${state.step} steps, ${
            Number.isFinite(state.bestLoss)
              ? `best validation loss ${state.bestLoss.toFixed(4)}`
              : "no validation result yet"
          }.`,
        );
      })
      .catch((reason: unknown) => {
        setError(`Could not read the saved run: ${(reason as Error).message}`);
      });
  });

  const startTraining = async (resume: boolean): Promise<void> => {
    if (busy()) return;
    const initial = resume ? saved() : null;
    if (resume && !initial) {
      setError("There is no saved run to resume.");
      return;
    }

    let customUrls: string[] = [];
    if (isCustom()) {
      try {
        customUrls = parseDocUrls(urls());
      } catch (reason) {
        setError((reason as Error).message);
        return;
      }
      if (customUrls.length === 0) {
        setError("Add at least one document URL before training.");
        return;
      }
    }

    cancelRequested = false;
    setBusy(true);
    setError(null);
    setTrained(null);
    setSolid2Trained(null);
    setSolid2Model(null);
    setCustomTrained(null);
    setCustomModel(null);
    setSample(null);
    setProgress(null);
    setStatus(resume ? "Resuming the saved run..." : "Preparing TensorFlow.js...");
    setHistory(initial ? initial.history : []);

    try {
      if (isSolid2()) {
        const result = await trainSolid2Gpt({
          steps: steps(),
          batch: batch(),
          lr: learningRate(),
          seed: seed(),
          dropout: dropout(),
          maxMinutes: maxMinutes(),
          initialState: initial,
          warmStart: !resume && warmStart(),
          onStatus: (value) => setStatus(value),
          onProgress: (value) => {
            setBackend(value.backend);
            setProgress(value);
            setHistory((previous) => [...previous, value]);
          },
          onCheckpoint: persist,
          shouldCancel: () => cancelRequested,
        });

        setSolid2Trained(result);
        setHistory(result.result.history);
        setSaved(result.result.state);
        setBackend(result.result.backend);
        const finished = result.result.stopped
          ? `Training stopped after ${result.result.state.step} steps. The partial model is saved.`
          : `Training complete after ${result.result.state.step} steps.`;
        setStatus(`${finished} Quantizing for integer-only inference...`);

        try {
          const model = await buildSolid2WasmFromTraining(result);
          setSolid2Model(model);
          const prompt = result.prompts[0] ?? "What are split effects?";
          const generated = createSolid2Decoder(model, prompt, {
            maxTokens: 24,
            temperature: 0.7,
            topK: 40,
          }).run();
          setSample({
            prompt,
            text: generated.text || "(empty generation)",
          });
          setStatus(`${finished} The model is quantized and ready to use.`);
        } catch (reason) {
          setError(
            `Training finished, but the integer model could not be built: ${
              (reason as Error).message
            }`,
          );
        }
        return;
      }

      if (isCustom()) {
        const result = await trainCustomGpt({
          urls: customUrls,
          steps: steps(),
          batch: batch(),
          lr: learningRate(),
          seed: seed(),
          dropout: dropout(),
          maxMinutes: maxMinutes(),
          initialState: initial,
          warmStart: !resume && warmStart(),
          onStatus: (value) => setStatus(value),
          onProgress: (value) => {
            setBackend(value.backend);
            setProgress(value);
            setHistory((previous) => [...previous, value]);
          },
          onCheckpoint: persist,
          shouldCancel: () => cancelRequested,
        });

        setCustomTrained(result);
        setHistory(result.result.history);
        setSaved(result.result.state);
        setBackend(result.result.backend);
        const finished = result.result.stopped
          ? `Training stopped after ${result.result.state.step} steps. The partial model is saved.`
          : `Training complete after ${result.result.state.step} steps.`;
        setStatus(`${finished} Quantizing for integer-only inference...`);

        try {
          const model = await buildCustomWasmFromTraining(result);
          setCustomModel(model);
          const prompt = result.prompts[0] ?? "Write a short introduction.";
          const generated = createSolid2Decoder(model, prompt, {
            maxTokens: 24,
            temperature: 0.7,
            topK: 40,
            promptFormat: "text",
          }).run();
          setSample({
            prompt,
            text: generated.text || "(empty generation)",
          });
          setStatus(`${finished} The model is quantized and ready to use.`);
        } catch (reason) {
          setError(
            `Training finished, but the integer model could not be built: ${
              (reason as Error).message
            }`,
          );
        }
        return;
      }

      const result = await trainTinyGpt({
        corpus,
        steps: steps(),
        batch: batch(),
        lr: learningRate(),
        seed: seed(),
        dropout: dropout(),
        maxMinutes: maxMinutes(),
        initialState: initial,
        onProgress: (value) => {
          setBackend(value.backend);
          setProgress(value);
          setHistory((previous) => [...previous, value]);
        },
        onCheckpoint: persist,
        shouldCancel: () => cancelRequested,
      });

      setTrained(result);
      setHistory(result.history);
      setSaved(result.state);
      setBackend(result.backend);
      setStatus(
        result.stopped
          ? `Training stopped after ${result.state.step} steps. The partial model is saved.`
          : `Training complete after ${result.state.step} steps.`,
      );

      try {
        const model = loadTinyGpt(result.checkpoint);
        const generated = generateTinyGpt(model, "Alice was", "float", {
          maxTokens: 24,
          temperature: 0.8,
          topK: 40,
        });
        setSample({ prompt: "Alice was", text: generated.text });
      } catch (reason) {
        setError(
          `Training finished, but the sample could not be generated: ${
            (reason as Error).message
          }`,
        );
      }
    } catch (reason) {
      if (cancelRequested) {
        setStatus("Training stopped.");
      } else {
        setError(`Training failed: ${(reason as Error).message}`);
        setStatus(null);
      }
    } finally {
      setBusy(false);
    }
  };

  const stopTraining = (): void => {
    cancelRequested = true;
    setStatus("Stopping after the current step...");
  };

  const useTrainedModel = async (): Promise<void> => {
    if (isSolid2()) {
      const result = solid2Trained();
      if (!result) return;
      let model = solid2Model();
      if (!model) {
        setBusy(true);
        setStatus("Building the integer-only WASM model...");
        try {
          model = await buildSolid2WasmFromTraining(result);
          setSolid2Model(model);
        } catch (reason) {
          setError(
            `Could not build the WASM model: ${(reason as Error).message}`,
          );
          return;
        } finally {
          setBusy(false);
        }
      }
      props.onUseSolid2Model?.(model);
      setStatus("The trained model is now active in the Solid 2 Integer GPT demo.");
      return;
    }
    if (isCustom()) {
      const result = customTrained();
      if (!result) return;
      let model = customModel();
      if (!model) {
        setBusy(true);
        setStatus("Building the integer-only WASM model...");
        try {
          model = await buildCustomWasmFromTraining(result);
          setCustomModel(model);
        } catch (reason) {
          setError(
            `Could not build the WASM model: ${(reason as Error).message}`,
          );
          return;
        } finally {
          setBusy(false);
        }
      }
      props.onUseCustomModel?.(model);
      setStatus("The trained custom model is now active in the Integer GPT demo.");
      return;
    }
    const result = trained();
    if (!result) return;
    props.onUseCheckpoint?.(result.checkpoint);
    setStatus("The trained model is now active in the Integer GPT demo.");
  };

  const clearSavedRun = async (): Promise<void> => {
    try {
      await saveChain.catch(() => undefined);
      await deleteBrowserTrainingState(runKey);
      setSaved(null);
      setHistory([]);
      setProgress(null);
      setTrained(null);
      setSolid2Trained(null);
      setSolid2Model(null);
      setCustomTrained(null);
      setCustomModel(null);
      setSample(null);
      setStatus("Saved browser training run deleted.");
      setError(null);
    } catch (reason) {
      setError(`Could not delete the saved run: ${(reason as Error).message}`);
    }
  };

  const progressFraction = (): number => {
    const current = progress();
    if (!current || current.totalSteps <= 0) return 0;
    return Math.max(0, Math.min(1, current.step / current.totalSteps));
  };

  return (
    <section class="card training-card">
      <div class="card-heading compact">
        <div>
          <span class="eyebrow">Browser training</span>
          <h2>Train this GPT in your browser</h2>
        </div>
        <div
          class={
            backend() === null
              ? "runtime-pill"
              : backend() === "cpu"
                ? "runtime-pill warning"
                : "runtime-pill ready"
          }
        >
          <span class="runtime-dot" />
          {backendLabel(backend())}
        </div>
      </div>

      <Show when={!isLarge()}>
        <p class="hint">
          Trains the same {DEFAULT_BROWSER_TRAIN_CONFIG.vocabSize}-token,
          {" "}{DEFAULT_BROWSER_TRAIN_CONFIG.context}-context, 62k-parameter
          Alice model with float32 gradients on the bundled public-domain
          corpus. The integer path starts when you click <b>Use in Integer GPT</b>:
          the app quantizes the result and runs the existing int8/WASM
          inference.
        </p>
        <p class="hint">
          Backend order is WebGPU → WebGL → CPU. WebGL is the normal fallback
          on browsers without WebGPU; it is not an error. Training is a
          small educational run, not a replacement for the Solid 2 Colab job.
        </p>
      </Show>
      <Show when={isSolid2()}>
        <p class="hint">
          Trains a{" "}
          {solid2ParameterCount(SOLID2_TRAIN_CONFIG).toLocaleString()}-parameter,
          4096-token BPE, 256-context Solid 2 docs + Q&amp;A model with float32
          gradients. The training corpus is downloaded from the repository
          instead of being bundled; it is tokenized with the deployed BPE
          tokenizer and the token stream is cached in IndexedDB after the first
          load. By default it warm-starts from the bundled checkpoint's weights;
          uncheck the option to train from scratch. The integer path starts when
          you click <b>Use in Integer GPT</b>.
        </p>
        <p class="hint">
          Solid 2 is roughly 20× the Alice model and uses a much larger context.
          WebGPU or WebGL is strongly recommended. A short run on CPU may take
          hours, so keep the time budget small.
        </p>
      </Show>
      <Show when={isCustom()}>
        <p class="hint">
          Trains a{" "}
          {solid2ParameterCount(SOLID2_TRAIN_CONFIG).toLocaleString()}-parameter,
          4096-token BPE, 256-context model on Markdown documents you choose.
          The documents are fetched in the browser, tokenized with the deployed
          Solid 2 tokenizer, and cached in IndexedDB after the first load. The
          integer path starts when you click <b>Use in Integer GPT</b>.
        </p>
        <p class="hint">
          Mirrors the Solid 2 model exactly so the same integer-only WASM engine
          can run it. Warm-starting from the bundled checkpoint is recommended
          for short runs; uncheck it to train from random weights. URLs must be
          public and allow cross-origin reads (CORS), such as raw GitHub links.
        </p>
      </Show>

      <Show when={error()}>
        <div class="banner error"><span>{error()}</span></div>
      </Show>
      <Show when={status()}>
        <div class="banner info"><span>{status()}</span></div>
      </Show>
      <Show when={isLarge()}>
        <div class="banner warning">
          <span>
            {isCustom() ? "Custom" : "Solid 2"} training is heavy: 1.34M
            parameters at 256 context. Use a GPU backend and expect a short run
            to take a while even then.
          </span>
        </div>
      </Show>
      <Show when={backend() === "cpu"}>
        <div class="banner warning">
          <span>
            No GPU backend is available. Training will be much slower; use a
            short run here or continue on Colab.
          </span>
        </div>
      </Show>

      <Show when={isCustom()}>
        <label class="training-field training-field-wide">
          Markdown document URLs (one per line)
          <textarea
            rows="5"
            placeholder={"https://raw.githubusercontent.com/owner/repo/main/docs/guide.md\nhttps://raw.githubusercontent.com/owner/repo/main/docs/api.md"}
            value={urls()}
            disabled={busy()}
            onInput={(event) => setUrls(event.currentTarget.value)}
          />
        </label>
      </Show>

      <div class="training-grid">
        <label class="training-field">
          Steps
          <input
            type="number"
            min="10"
            max="20000"
            step="10"
            value={steps()}
            disabled={busy()}
            onInput={(event) =>
              setSteps(Math.max(10, Number(event.currentTarget.value) || 10))
            }
          />
        </label>
        <label class="training-field">
          Batch
          <input
            type="number"
            min="1"
            max="32"
            step="1"
            value={batch()}
            disabled={busy()}
            onInput={(event) =>
              setBatch(Math.max(1, Number(event.currentTarget.value) || 1))
            }
          />
        </label>
        <label class="training-field">
          Learning rate
          <input
            type="number"
            min="0.00001"
            max="1"
            step="0.0001"
            value={learningRate()}
            disabled={busy()}
            onInput={(event) =>
              setLearningRate(
                Math.max(0.00001, Number(event.currentTarget.value) || 0.00001),
              )
            }
          />
        </label>
        <label class="training-field">
          Dropout
          <input
            type="number"
            min="0"
            max="0.9"
            step="0.01"
            value={dropout()}
            disabled={busy()}
            onInput={(event) =>
              setDropout(
                Math.max(0, Math.min(0.9, Number(event.currentTarget.value) || 0)),
              )
            }
          />
        </label>
        <label class="training-field">
          Seed
          <input
            type="number"
            min="0"
            step="1"
            value={seed()}
            disabled={busy()}
            onInput={(event) =>
              setSeed(Math.max(0, Math.floor(Number(event.currentTarget.value) || 0)))
            }
          />
        </label>
        <label class="training-field">
          Time budget (minutes, 0 = none)
          <input
            type="number"
            min="0"
            step="1"
            value={maxMinutes()}
            disabled={busy()}
            onInput={(event) =>
              setMaxMinutes(Math.max(0, Number(event.currentTarget.value) || 0))
            }
          />
        </label>
      </div>

      <Show when={isLarge()}>
        <label class="training-toggle">
          <input
            type="checkbox"
            checked={warmStart()}
            disabled={busy()}
            onChange={(event) => setWarmStart(event.currentTarget.checked)}
          />
          <span>
            <b>Warm start from the bundled Solid 2 model</b>
            <small>
              Starts from the deployed checkpoint's weights and a fresh
              optimizer instead of random initialization.
              <Show when={isCustom()}>
                {" "}Uncheck to train the custom model from scratch.
              </Show>{" "}
              Resuming a saved run ignores this.
            </small>
          </span>
        </label>
      </Show>

      <div class="button-row">
        <button
          type="button"
          class="button primary"
          disabled={busy()}
          onClick={() => void startTraining(false)}
        >
          {busy() ? "Training..." : "Start training"}
        </button>
        <Show when={busy()}>
          <button type="button" class="button danger" onClick={stopTraining}>
            Stop
          </button>
        </Show>
        <button
          type="button"
          class="button secondary"
          disabled={busy()}
          onClick={() => {
            const quick = isLarge() ? 100 : 200;
            setSteps(quick);
            setStatus(`Quick run selected: ${quick} steps.`);
          }}
        >
          {isLarge() ? "Quick 100 steps" : "Quick 200 steps"}
        </button>
        <Show when={saved() && !busy()}>
          <button
            type="button"
            class="button secondary"
            onClick={() => void startTraining(true)}
          >
            Resume saved run
          </button>
        </Show>
        <Show when={saved() && !busy()}>
          <button
            type="button"
            class="button ghost"
            onClick={() => void clearSavedRun()}
          >
            Delete saved run
          </button>
        </Show>
      </div>

      <Show when={progress()}>
        {(current) => (
          <div class="progress-block">
            <div class="progress-label">
              <span>
                Step {current().step} / {current().totalSteps}
              </span>
              <span>{current().stepsPerSecond.toFixed(1)} steps/s</span>
            </div>
            <div class="progress-track">
              <div
                class="progress-fill"
                style={{ width: `${progressFraction() * 100}%` }}
              />
            </div>
            <div class="training-stats">
              <span>
                loss <b>{current().loss.toFixed(4)}</b>
              </span>
              <span>
                smoothed <b>{current().smoothedLoss.toFixed(4)}</b>
              </span>
              <Show when={current().valLoss !== undefined}>
                <span>
                  validation <b>{current().valLoss!.toFixed(4)}</b>
                </span>
              </Show>
              <Show when={current().bestLoss !== null}>
                <span>
                  best validation <b>{current().bestLoss!.toFixed(4)}</b> at step{" "}
                  {current().bestStep}
                </span>
              </Show>
              <span>lr {current().lr.toExponential(2)}</span>
              <span>{(current().elapsedMs / 1000).toFixed(1)}s</span>
            </div>
          </div>
        )}
      </Show>

      <LossChart points={history()} bestStep={progress()?.bestStep ?? saved()?.bestStep} />

      <Show when={sample()}>
        {(result) => (
          <div class="training-sample">
            <div class="section-heading">
              <span>Sample from the trained model</span>
              <span class="muted">prompt: {result().prompt}</span>
            </div>
            <p>{result().text}</p>
          </div>
        )}
      </Show>

      <div class="button-row">
        <button
          type="button"
          class="button primary"
          disabled={
            busy() ||
            (isSolid2()
              ? !solid2Trained()
              : isCustom()
                ? !customTrained()
                : !trained())
          }
          onClick={() => void useTrainedModel()}
        >
          Use in Integer GPT
        </button>
        <Show when={props.isCustomCheckpoint}>
          <button
            type="button"
            class="button secondary"
            onClick={props.onResetCheckpoint}
          >
            {isSolid2()
              ? "Reset demo to bundled Solid 2"
              : isCustom()
                ? "Unload custom model"
                : "Reset demo to bundled Alice"}
          </button>
        </Show>
      </div>

      <p class="hint">
        Saved runs stay in this browser's IndexedDB. The checkpoint includes the
        model, Adam state, history, and best-validation weights, so a later
        session can continue from the same step.
        <Show when={isLarge()}>
          {" "}The tokenized training corpus is cached separately, so the second
          load skips BPE encoding.
        </Show>
      </p>
    </section>
  );
}
