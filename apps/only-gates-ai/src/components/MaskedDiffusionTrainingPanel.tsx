import {
  createEffect,
  createSignal,
  onCleanup,
  onSettled,
  Show,
} from "solid-js";

import {
  createMaskedDiffusionDecoder,
  createMaskedDiffusionWasmFromData,
  type MaskedDiffusionModel,
} from "../lib/maskedDiffusion";
import {
  deleteMaskedDiffusionTrainingState,
  loadMaskedDiffusionTrainingState,
  saveMaskedDiffusionTrainingState,
  type MaskedDiffusionTrainingState,
} from "../lib/maskedDiffusionStorage";
import {
  buildMaskedDiffusionWasmFromTraining,
  downloadMaskedDiffusionBundle,
  exportMaskedDiffusionBundle,
  importMaskedDiffusionBundle,
  trainMaskedDiffusion,
  type MaskedDiffusionTrainingResult,
} from "../lib/trainMaskedDiffusion";
import type {
  BrowserTrainingProgress,
  TrainingPoint,
} from "../lib/trainTinyGpt";
import LossChart from "./LossChart";

export default function MaskedDiffusionTrainingPanel(props: {
  /** Activates the trained masked-diffusion model in the demo panel above. */
  onUseModel?: (model: MaskedDiffusionModel) => void;
  /** Restores the saved run only once the masked tab is visible. */
  active?: boolean;
  /** Which masked-diffusion model to train: "luau" or "solid2". */
  variant?: "luau" | "solid2";
}) {
  const variant = (): "luau" | "solid2" => props.variant ?? "luau";
  const isSolid2 = (): boolean => variant() === "solid2";

  // ==== Controls ==== //
  const [steps, setSteps] = createSignal(400);
  const [batch, setBatch] = createSignal(4);
  const [learningRate, setLearningRate] = createSignal(0.0004);
  const [dropout, setDropout] = createSignal(0.1);
  const [seed, setSeed] = createSignal(1337);
  const [maxMinutes, setMaxMinutes] = createSignal(30);
  const [validationEvery, setValidationEvery] = createSignal(20);
  const [useFull, setUseFull] = createSignal(false);
  const [warmStart, setWarmStart] = createSignal(true);

  // ==== Runtime state ==== //
  const [busy, setBusy] = createSignal(false);
  const [restoring, setRestoring] = createSignal(false);
  const [error, setError] = createSignal<string | null>(null);
  const [status, setStatus] = createSignal<string | null>(null);
  const [backend, setBackend] = createSignal<string | null>(null);
  const [progress, setProgress] = createSignal<BrowserTrainingProgress | null>(
    null,
  );
  const [history, setHistory] = createSignal<readonly TrainingPoint[]>([]);
  const [saved, setSaved] = createSignal<MaskedDiffusionTrainingState | null>(
    null,
  );
  const [trained, setTrained] =
    createSignal<MaskedDiffusionTrainingResult | null>(null);
  const [trainedModel, setTrainedModel] =
    createSignal<MaskedDiffusionModel | null>(null);
  const [sample, setSample] = createSignal<{
    readonly prompt: string;
    readonly text: string;
  } | null>(null);

  let cancelRequested = false;
  // A deployed-size masked-diffusion state contains about 150 MB of typed
  // arrays (weights, Adam moments, and best weights). Keep only the latest
  // pending checkpoint instead of queuing every intermediate state.
  let pendingState: MaskedDiffusionTrainingState | null = null;
  let saveInFlight: Promise<void> | null = null;

  onCleanup(() => {
    cancelRequested = true;
  });

  const flushSaves = (): Promise<void> => {
    if (saveInFlight === null) {
      saveInFlight = (async () => {
        while (pendingState !== null) {
          const state = pendingState;
          pendingState = null;
          try {
            await saveMaskedDiffusionTrainingState(state, variant());
          } catch (reason) {
            setError(
              `Training continues, but the run could not be saved: ${
                (reason as Error).message
              }`,
            );
          }
        }
        saveInFlight = null;
      })();
    }
    return saveInFlight;
  };

  const persist = (state: MaskedDiffusionTrainingState): void => {
    setSaved(state);
    pendingState = state;
    void flushSaves();
  };

  let restoredVariant: string | null = null;
  const restoreSavedRun = (): void => {
    if (restoredVariant === variant()) return;
    restoredVariant = variant();
    setRestoring(true);
    void loadMaskedDiffusionTrainingState(variant())
      .then((state) => {
        if (!state) return;
        setSaved(state);
        setHistory(state.history);
        setSteps(Math.max(state.steps, state.step + 50));
        setBatch(state.batch);
        setLearningRate(state.lr);
        setDropout(state.dropout);
        setSeed(state.seed);
        setUseFull(state.fullData);
        setWarmStart(state.warmStart);
        setStatus(
          `Saved run found: ${state.step} steps${
            Number.isFinite(state.bestLoss)
              ? `, best validation loss ${state.bestLoss.toFixed(4)}`
              : ""
          }.`,
        );
      })
      .catch((reason: unknown) => {
        setError(
          `Could not read the saved masked-diffusion run: ${
            (reason as Error).message
          }`,
        );
      })
      .finally(() => setRestoring(false));
  };

  // The tab panels stay mounted, so avoid deserializing a ~150 MB saved state
  // until the masked tab has actually been opened.
  let settled = false;
  onSettled(() => {
    settled = true;
    if (props.active !== false) restoreSavedRun();
  });
  createEffect(
    () => props.active,
    (active) => {
      if (settled && active) restoreSavedRun();
    },
  );

  let lastVariant = variant();
  createEffect(
    () => variant(),
    (v) => {
      if (v === lastVariant) {
        lastVariant = v;
        return;
      }
      lastVariant = v;
      setSaved(null);
      setHistory([]);
      setProgress(null);
      setTrained(null);
      setTrainedModel(null);
      setSample(null);
      setError(null);
      setStatus(null);
      restoredVariant = null;
      if (settled && props.active !== false) restoreSavedRun();
    },
  );

  const startTraining = async (resume: boolean): Promise<void> => {
    if (busy()) return;
    const initial = resume ? saved() : null;
    if (resume && !initial) {
      setError("There is no saved masked-diffusion run to resume.");
      return;
    }

    cancelRequested = false;
    setBusy(true);
    setError(null);
    setProgress(null);
    setTrained(null);
    setTrainedModel(null);
    setSample(null);
    setHistory(initial ? initial.history : []);
    setStatus(
      resume
        ? "Resuming the saved masked-diffusion run..."
        : "Preparing TensorFlow.js...",
    );

    try {
      const result = await trainMaskedDiffusion({
        // Resuming must keep the dataset the saved run was trained on; the
        // dataset radios only choose the data for a fresh run.
        variant: variant(),
        fullData: initial?.fullData ?? useFull(),
        steps: steps(),
        batch: batch(),
        lr: learningRate(),
        seed: seed(),
        dropout: dropout(),
        maxMinutes: maxMinutes(),
        validationEvery: validationEvery(),
        initialState: initial,
        warmStart: warmStart(),
        onStatus: (value) => setStatus(value),
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
      const finished = result.stopped
        ? `Training stopped after ${result.state.step} steps. The partial model is saved.`
        : `Training complete after ${result.state.step} steps.`;

      setStatus(`${finished} Quantizing for integer-only inference...`);
      try {
        const model = await buildMaskedDiffusionWasmFromTraining(result);
        setTrainedModel(model);
        const prompt =
          result.prompts[1] ??
          result.prompts[0] ??
          (isSolid2()
            ? "What are split effects in Solid 2?"
            : "Write a for loop in Luau that runs from 1 to 20 and prints every number.");
        const generated = createMaskedDiffusionDecoder(model, prompt, {
          steps: 8,
          temperature: 0.01,
          mode: isSolid2() ? "ask" : "write",
        }).run();
        setSample({
          prompt,
          text: generated.answer || "(empty answer)",
        });
        setStatus(`${finished} The quantized model is ready to use.`);
      } catch (reason) {
        setError(
          `Training finished, but the integer model could not be built: ${
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

  const useTrainedModel = (): void => {
    const model = trainedModel();
    if (!model) {
      setError("Train or import a masked-diffusion model first.");
      return;
    }
    props.onUseModel?.(model);
    setError(null);
    setStatus(
      "The trained model is now active in the Masked diffusion demo.",
    );
  };

  const clearRun = async (): Promise<void> => {
    try {
      await flushSaves();
      await deleteMaskedDiffusionTrainingState(variant());
      setSaved(null);
      setHistory([]);
      setProgress(null);
      setTrained(null);
      setTrainedModel(null);
      setSample(null);
      setStatus("Saved masked-diffusion run deleted.");
      setError(null);
    } catch (reason) {
      setError(`Could not clear the run: ${(reason as Error).message}`);
    }
  };

  const exportBundle = async (): Promise<void> => {
    const result = trained();
    if (!result) {
      setError("Train a model before exporting it.");
      return;
    }
    setBusy(true);
    setError(null);
    setStatus("Packaging the masked-diffusion model...");
    try {
      const bundle = await exportMaskedDiffusionBundle(result);
      downloadMaskedDiffusionBundle(bundle);
      setStatus(
        `Exported ${bundle.filename}. Copy the three model files into ` +
          (isSolid2()
            ? "public/models/solid2-masked-diffusion/"
            : "public/models/masked-diffusion-luau/") + " to publish it.",
      );
    } catch (reason) {
      setError(`Export failed: ${(reason as Error).message}`);
    } finally {
      setBusy(false);
    }
  };

  const importBundle = async (file: File): Promise<void> => {
    if (busy()) return;
    setBusy(true);
    setError(null);
    setStatus(`Reading ${file.name}...`);
    try {
      const bundle = await importMaskedDiffusionBundle(file);
      setStatus("Building the integer-only WASM model...");
      const model = await createMaskedDiffusionWasmFromData(
        bundle.checkpoint,
        bundle.tokenizer,
        bundle.weights,
      );
      setTrained(null);
      setTrainedModel(model);
      setHistory([]);
      setProgress(null);
      setSample(null);
      setStatus(
        `Imported ${file.name} (${bundle.checkpoint.parameterCount.toLocaleString()} parameters). ` +
          "Click “Use this model” to activate it.",
      );
    } catch (reason) {
      setError(`Import failed: ${(reason as Error).message}`);
    } finally {
      setBusy(false);
    }
  };

  const progressFraction = (): number => {
    const current = progress();
    if (!current || current.totalSteps <= 0) return 0;
    return Math.max(0, Math.min(1, current.step / current.totalSteps));
  };

  const backendLabel = (): string => {
    const current = backend();
    if (current === "webgpu") return "WebGPU";
    if (current === "webgl") return "WebGL";
    if (current === "cpu") return "CPU (slow)";
    return busy() ? "Starting…" : "TensorFlow.js";
  };

  let fileInput: HTMLInputElement | undefined;

  return (
    <section class="card training-card">
      <div class="card-heading compact">
        <div>
          <span class="eyebrow">Masked-Diffusion Training</span>
          <h2>Fine-tune in the browser</h2>
        </div>
        <div class={`runtime-pill ${busy() ? "warning" : "ready"}`}>
          <span class="runtime-dot" />
          {backendLabel()}
        </div>
      </div>

      <p class="hint">
        This is a real TensorFlow.js fine-tune of the bidirectional Luau
        denoiser, not a simulated loss curve. It trains on the repository's
        generated Luau tasks plus the Luau FAQ, using the same absorbing{" "}
        <code>&lt;|mask|&gt;</code> objective as the Colab trainer: a random
        fraction of the answer region is masked, and only those masked, non-pad
        positions contribute to the loss. Backend order is WebGPU → WebGL →
        CPU; the deployed 9.56M-parameter checkpoint is warm-started by default,
        so a short run is a fine-tune rather than a from-scratch train.
      </p>

      <Show when={error()}>
        <div class="banner error">
          <span>{error()}</span>
        </div>
      </Show>
      <Show when={status()}>
        <div class="banner info">
          <span>{status()}</span>
        </div>
      </Show>
      <Show when={backend() === "cpu"}>
        <div class="banner warning">
          <span>
            No GPU backend is available. The 9.56M-parameter model will be much
            slower on CPU; use a short run, or continue on Colab.
          </span>
        </div>
      </Show>

      <Show when={!isSolid2()}>
        <div class="training-data">
          <label style={{ "margin-right": "1rem" }}>
            <input
              type="radio"
              name="masked-dataset"
              checked={!useFull()}
              disabled={busy()}
              onChange={() => setUseFull(false)}
            />
            Small data (2,000 bundled tasks)
          </label>
          <label>
            <input
              type="radio"
              name="masked-dataset"
              checked={useFull()}
              disabled={busy()}
              onChange={() => setUseFull(true)}
            />
            Full data (30,000 tasks, 14 MB download)
          </label>
        </div>
      </Show>

      <div class="training-grid">
        <label class="training-field">
          Steps
          <input
            type="number"
            min="1"
            max="20000"
            step="10"
            value={steps()}
            disabled={busy()}
            onInput={(event) =>
              setSteps(Math.max(1, Number(event.currentTarget.value) || 1))
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
            min="0.000001"
            max="0.1"
            step="0.0001"
            value={learningRate()}
            disabled={busy()}
            onInput={(event) =>
              setLearningRate(
                Math.max(0.000001, Number(event.currentTarget.value) || 0.0004),
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
                Math.max(
                  0,
                  Math.min(0.9, Number(event.currentTarget.value) || 0),
                ),
              )
            }
          />
        </label>
        <label class="training-field">
          Validation every (steps)
          <input
            type="number"
            min="1"
            max="1000"
            step="1"
            value={validationEvery()}
            disabled={busy()}
            onInput={(event) =>
              setValidationEvery(
                Math.max(1, Number(event.currentTarget.value) || 20),
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
              setSeed(
                Math.max(0, Math.floor(Number(event.currentTarget.value) || 0)),
              )
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
              setMaxMinutes(
                Math.max(0, Number(event.currentTarget.value) || 0),
              )
            }
          />
        </label>
      </div>

      <label class="training-toggle">
        <input
          type="checkbox"
          checked={warmStart()}
          disabled={busy()}
          onChange={(event) => setWarmStart(event.currentTarget.checked)}
        />
        <span>
          <b>Warm start from the deployed masked-diffusion checkpoint</b>
          <small>
            Starts from the bundled 9.56M-parameter Luau denoiser and a fresh
            optimizer. Uncheck to train a smaller model from random weights.
            Resuming a saved run ignores this.
          </small>
        </span>
      </label>

      <div class="button-row">
        <button
          type="button"
          class="button primary"
          disabled={busy() || restoring()}
          onClick={() => void startTraining(false)}
        >
          {busy() ? "Training…" : restoring() ? "Loading saved run…" : "Start Training"}
        </button>
        <Show when={busy()}>
          <button type="button" class="button danger" onClick={stopTraining}>
            Stop
          </button>
        </Show>
        <button
          type="button"
          class="button secondary"
          disabled={busy() || saved() === null}
          onClick={() => void startTraining(true)}
        >
          Resume Training
        </button>
        <button
          type="button"
          class="button secondary"
          disabled={busy() || trainedModel() === null}
          onClick={useTrainedModel}
        >
          Use this model
        </button>
        <button
          type="button"
          class="button ghost"
          disabled={busy()}
          onClick={() => fileInput?.click()}
        >
          Import Bundle
        </button>
        <input
          type="file"
          accept=".zip,application/zip"
          ref={(element) => {
            fileInput = element;
          }}
          style={{ display: "none" }}
          onChange={(event) => {
            const file = event.currentTarget.files?.[0];
            event.currentTarget.value = "";
            if (file) void importBundle(file);
          }}
        />
        <button
          type="button"
          class="button secondary"
          disabled={busy() || trained() === null}
          onClick={() => void exportBundle()}
        >
          Export Bundle
        </button>
        <Show when={saved() !== null && !busy()}>
          <button
            type="button"
            class="button ghost"
            onClick={() => void clearRun()}
          >
            Clear saved run
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
              <span>{current().stepsPerSecond.toFixed(2)} steps/s</span>
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

      <LossChart
        points={history()}
        bestStep={progress()?.bestStep ?? saved()?.bestStep}
      />

      <Show when={sample()}>
        {(result) => (
          <div class="training-sample">
            <div class="section-heading">
              <span>Sample from the trained denoiser</span>
              <span class="muted">prompt: {result().prompt}</span>
            </div>
            <pre class="chat-code" tabindex="0">
              <code>{result().text}</code>
            </pre>
          </div>
        )}
      </Show>

      <p class="hint">
        Saved runs stay in this browser's IndexedDB and include the weights,
        Adam moments, best-validation weights, and loss history. Export writes
        the same <code>checkpoint.json</code>, <code>tokenizer.json</code>, and{" "}
        <code>weights.bin</code> that the Masked diffusion tab loads, so you can
        copy them into <code>public/models/masked-diffusion-luau/</code> to
        publish the run.
      </p>
    </section>
  );
}
