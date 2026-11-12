import { createSignal, onCleanup, onSettled, Show } from "solid-js";

import corpus from "../assets/tiny-gpt-corpus.txt?raw";
import {
  deleteBrowserTrainingState,
  loadBrowserTrainingState,
  saveBrowserTrainingState,
} from "../lib/browserTrainingStorage";
import {
  generateTinyGpt,
  loadTinyGpt,
  type TinyGptCheckpoint,
} from "../lib/tinyGpt";
import {
  DEFAULT_BROWSER_TRAIN_CONFIG,
  trainTinyGpt,
  type BrowserTrainingProgress,
  type BrowserTrainingResult,
  type BrowserTrainingState,
  type TrainingPoint,
} from "../lib/trainTinyGpt";
import LossChart from "./LossChart";

interface BrowserTrainingPanelProps {
  readonly isCustomCheckpoint: boolean;
  readonly onUseCheckpoint: (checkpoint: TinyGptCheckpoint) => void;
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
  const [history, setHistory] = createSignal<readonly TrainingPoint[]>([]);
  const [progress, setProgress] = createSignal<BrowserTrainingProgress | null>(
    null,
  );
  const [trained, setTrained] = createSignal<BrowserTrainingResult | null>(null);
  const [saved, setSaved] = createSignal<BrowserTrainingState | null>(null);
  const [busy, setBusy] = createSignal(false);
  const [error, setError] = createSignal<string | null>(null);
  const [status, setStatus] = createSignal<string | null>(null);
  const [backend, setBackend] = createSignal<string | null>(null);
  const [sample, setSample] = createSignal<{
    readonly prompt: string;
    readonly text: string;
  } | null>(null);

  const [steps, setSteps] = createSignal(1800);
  const [batch, setBatch] = createSignal(8);
  const [learningRate, setLearningRate] = createSignal(0.003);
  const [dropout, setDropout] = createSignal(0);
  const [seed, setSeed] = createSignal(1337);
  const [maxMinutes, setMaxMinutes] = createSignal(0);

  let cancelRequested = false;
  let saveChain: Promise<void> = Promise.resolve();

  onCleanup(() => {
    // Switching away from Alice unmounts this panel. Stop the loop so it does
    // not keep the GPU busy in the background; the last checkpoint is saved by
    // the training loop's normal shutdown path.
    cancelRequested = true;
  });

  const persist = (state: BrowserTrainingState): void => {
    setSaved(state);
    saveChain = saveChain
      .then(() => saveBrowserTrainingState(state))
      .catch((reason: unknown) => {
        setError(
          `Training continues, but the run could not be saved: ${
            (reason as Error).message
          }`,
        );
      });
  };

  onSettled(() => {
    void loadBrowserTrainingState()
      .then((state) => {
        if (!state) return;
        setSaved(state);
        setHistory(state.history);
        setSteps(Math.min(20000, Math.max(state.steps, state.step + 200)));
        setBatch(state.batch);
        setLearningRate(state.lr);
        setDropout(state.dropout);
        setSeed(state.seed);
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

    cancelRequested = false;
    setBusy(true);
    setError(null);
    setTrained(null);
    setSample(null);
    setProgress(null);
    setStatus(resume ? "Resuming the saved run..." : "Preparing TensorFlow.js...");
    setHistory(initial ? initial.history : []);

    try {
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
      setError(`Training failed: ${(reason as Error).message}`);
      setStatus(null);
    } finally {
      setBusy(false);
    }
  };

  const stopTraining = (): void => {
    cancelRequested = true;
    setStatus("Stopping after the current step...");
  };

  const useTrainedModel = (): void => {
    const result = trained();
    if (!result) return;
    props.onUseCheckpoint(result.checkpoint);
    setStatus("The trained model is now active in the Integer GPT demo.");
  };

  const clearSavedRun = async (): Promise<void> => {
    try {
      await saveChain.catch(() => undefined);
      await deleteBrowserTrainingState();
      setSaved(null);
      setHistory([]);
      setProgress(null);
      setTrained(null);
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

      <p class="hint">
        Trains the same {DEFAULT_BROWSER_TRAIN_CONFIG.vocabSize}-token,
        {" "}{DEFAULT_BROWSER_TRAIN_CONFIG.context}-context, 62k-parameter Alice
        model with float32 gradients on the bundled public-domain corpus. The
        integer path starts when you click <b>Use in Integer GPT</b>: the app
        quantizes the result and runs the existing int8/WASM inference.
      </p>
      <p class="hint">
        Backend order is WebGPU → WebGL → CPU. WebGL is the normal fallback on
        browsers without WebGPU; it is not an error. Training is a small
        educational run, not a replacement for the Solid 2 Colab job.
      </p>

      <Show when={error()}>
        <div class="banner error"><span>{error()}</span></div>
      </Show>
      <Show when={status()}>
        <div class="banner info"><span>{status()}</span></div>
      </Show>
      <Show when={backend() === "cpu"}>
        <div class="banner warning">
          <span>
            No GPU backend is available. Training will be much slower; use a
            short run here or continue on Colab.
          </span>
        </div>
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
            setSteps(200);
            setStatus("Quick run selected: 200 steps.");
          }}
        >
          Quick 200 steps
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
          disabled={!trained() || busy()}
          onClick={useTrainedModel}
        >
          Use in Integer GPT
        </button>
        <Show when={props.isCustomCheckpoint}>
          <button
            type="button"
            class="button secondary"
            onClick={props.onResetCheckpoint}
          >
            Reset demo to bundled Alice
          </button>
        </Show>
      </div>

      <p class="hint">
        Saved runs stay in this browser's IndexedDB. The checkpoint includes the
        model, Adam state, history, and best-validation weights, so a later
        session can continue from the same step.
      </p>
    </section>
  );
}
