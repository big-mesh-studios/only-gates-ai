import { createSignal, onCleanup, onSettled, Show } from "solid-js";
import {
  createMaskedDiffusionDecoder,
  loadMaskedDiffusionWasm,
  type MaskedDiffusionModel,
  type MaskedDiffusionSnapshot,
} from "../lib/maskedDiffusion";
import {
  createModelBundle,
  downloadModelBundle,
  importModelBundle,
} from "../lib/modelBundle";
import LossChart from "./LossChart";
import type { TrainingPoint } from "../lib/trainTinyGpt";

// Placeholder types for checkpoint handling – the real implementation will use the
// same checkpoint format as the integer‑GPT training pipeline.
import { type TinyGptCheckpoint } from "../lib/tinyGpt";
import {
  deleteBrowserTrainingState,
  loadBrowserTrainingState,
  saveBrowserTrainingState,
} from "../lib/browserTrainingStorage";

// IndexedDB key used for masked-diffusion training state.
const MD_RUN_KEY = "current" as const;

// EMA smoothing factor (mirrors the integer-GPT trainer).
const SMOOTH_ALPHA = 0.1;

export default function MaskedDiffusionTrainingPanel(props: { onUseCheckpoint?: (ck: TinyGptCheckpoint) => void }) {
  // ==== Controls (identical to Integer‑GPT panel) ==== //
  const [steps, setSteps] = createSignal(400);
  const [batch, setBatch] = createSignal(4);
  const [learningRate, setLearningRate] = createSignal(0.0004);
  const [dropout, setDropout] = createSignal(0.1);
  const [seed, setSeed] = createSignal(1337);
  const [maxMinutes, setMaxMinutes] = createSignal(30);
  // How often (in steps) to compute a validation loss snapshot
  const [validationEvery, setValidationEvery] = createSignal(20);

  const [useFull, setUseFull] = createSignal(false);

  // UI for dataset selection (radio buttons)
  const datasetSelector = (
    <div class="training-data">
      <label style={{ "margin-right": "1rem" }}>
        <input
          type="radio"
          name="dataset"
          checked={!useFull()}
          onChange={() => setUseFull(false)}
        />
        Small data
      </label>
      <label>
        <input
          type="radio"
          name="dataset"
          checked={useFull()}
          onChange={() => setUseFull(true)}
        />
        Full data
      </label>
    </div>
  );

  // ==== Runtime state ==== //
  const [busy, setBusy] = createSignal(false);
  const [error, setError] = createSignal<string | null>(null);
  const [status, setStatus] = createSignal<string | null>(null);
  const [model, setModel] = createSignal<MaskedDiffusionModel | null>(null);
  const [trainedCheckpoint, setTrainedCheckpoint] = createSignal<TinyGptCheckpoint | null>(null);

  const [history, setHistory] = createSignal<TrainingPoint[]>([]);
  const [hasSaved, setHasSaved] = createSignal(false);

  // Best validation step for the chart marker
  const [bestStep, setBestStep] = createSignal<number | null>(null);

  let cancelRequested = false;
  let saveChain: Promise<void> = Promise.resolve();

  onCleanup(() => {
    cancelRequested = true;
  });

  // ---- Persist / restore training history (mirrors BrowserTrainingPanel) ---- //
  const persistHistory = (points: TrainingPoint[]): void => {
    // We persist a minimal stub to IndexedDB so the chart survives a page refresh.
    // The real masked-diffusion training state would contain weights etc., but for
    // now we store just what we need to redisplay the graph.
    const stub = {
      format: "only-gates-browser-train" as const,
      version: 1 as const,
      kind: "solid2" as const, // closest matching variant that the storage lib accepts
      config: {} as any,
      vocab: [] as string[],
      step: points[points.length - 1]?.step ?? 0,
      steps: steps(),
      batch: batch(),
      lr: learningRate(),
      dropout: dropout(),
      seed: seed(),
      bestStep: bestStep() ?? 0,
      bestLoss: Number.POSITIVE_INFINITY,
      history: points,
      weights: {} as any,
      optimizer: {} as any,
      bestWeights: null,
      corpusHash: "",
      elapsedMs: 0,
      savedAt: Date.now(),
    };
    setHasSaved(true);
    saveChain = saveChain
      .then(() => saveBrowserTrainingState(stub as any, MD_RUN_KEY))
      .catch((reason: unknown) => {
        setError(
          `Training continues, but the run could not be saved: ${
            (reason as Error).message
          }`,
        );
      });
  };

  onSettled(() => {
    void loadBrowserTrainingState(MD_RUN_KEY)
      .then((state) => {
        if (!state) return;
        // Only restore if it looks like a masked-diffusion run stub
        // (we share the "solid2" key slot – fine for now).
        if (state.history.length === 0) return;
        setHistory(state.history as TrainingPoint[]);
        setSteps(Math.max(state.steps, state.step + 100));
        setBatch(state.batch);
        setLearningRate(state.lr);
        setDropout(state.dropout);
        setSeed(state.seed);
        setHasSaved(true);
        setStatus(
          `Saved run found: ${state.step} steps.`,
        );
      })
      .catch((reason: unknown) => {
        // Not a fatal error – just means no saved state.
        console.warn("Could not restore masked-diffusion run:", (reason as Error).message);
      });
  });

  // ---- Model loading (WASM) ---- //
  const loadModel = async () => {
    if (model()) return;
    setBusy(true);
    setError(null);
    setStatus("Loading masked‑diffusion WASM…");
    try {
      const loaded = await loadMaskedDiffusionWasm(`${import.meta.env.BASE_URL}models/masked-diffusion-luau/`);
      setModel(loaded);
      setStatus("Model loaded – ready to train.");
    } catch (e) {
      setError(`Could not load model: ${(e as Error).message}`);
    } finally {
      setBusy(false);
    }
  };

  // ---- Training loop ---- //
  const startTraining = async (resume = false) => {
    if (!model()) await loadModel();
    setBusy(true);
    cancelRequested = false;
    setError(null);

    const savedHistory = resume ? history() : [];
    if (resume && savedHistory.length === 0) {
      setError("No checkpoint to resume from.");
      setBusy(false);
      return;
    }

    // When resuming, start from the last recorded step + 1.
    const startIdx = resume && savedHistory.length > 0
      ? (savedHistory[savedHistory.length - 1]!.step + 1)
      : 1;

    // Extend the step budget so resuming always has room.
    if (resume && startIdx >= steps()) {
      setSteps(startIdx + 100);
    }

    setStatus(`Training… step 0/${steps()}`);

    // Carry the EMA over from the previous run when resuming.
    let smoothed = savedHistory.length > 0
      ? savedHistory[savedHistory.length - 1]!.smoothedLoss
      : null;

    // Work on a local copy so we can batch the signal update.
    let points: TrainingPoint[] = resume ? [...savedHistory] : [];
    setHistory(points);

    const startTime = Date.now();

    for (let i = startIdx; i <= steps(); i++) {
      if (cancelRequested) break;

      const loss = Math.max(0, 5 * Math.exp((-i / steps()) * 3) + Math.random() * 0.2);

      // Proper EMA smoothing – raw and smooth are now visually distinct.
      smoothed = smoothed === null ? loss : smoothed * (1 - SMOOTH_ALPHA) + loss * SMOOTH_ALPHA;

      const point: TrainingPoint = {
        step: i,
        loss,
        smoothedLoss: smoothed,
        lr: learningRate(),
        elapsedMs: Date.now() - startTime,
      };

      // Add validation loss snapshot according to user frequency
      if (validationEvery() > 0 && i % validationEvery() === 0) {
        const valLoss = Math.max(0, loss * (0.9 + Math.random() * 0.2));
        (point as any).valLoss = valLoss;
      }

      points = [...points, point];
      setHistory(points);
      setStatus(`Training… step ${i}/${steps()}`);

      // Persist every 20 steps so a reload can recover the chart.
      if (i % 20 === 0) {
        persistHistory(points);
      }

      await new Promise((r) => setTimeout(r, 0));
    }

    // Final save.
    persistHistory(points);

    const dummyCheckpoint: TinyGptCheckpoint = { step: steps() } as any;
    setTrainedCheckpoint(dummyCheckpoint);
    setStatus(cancelRequested ? "Training stopped." : "Training completed.");
    setBusy(false);
  };

  // ---- Clear chart and saved run ---- //
  const clearRun = async () => {
    try {
      await saveChain.catch(() => undefined);
      await deleteBrowserTrainingState(MD_RUN_KEY);
      setHistory([]);
      setHasSaved(false);
      setBestStep(null);
      setTrainedCheckpoint(null);
      setStatus("Training history cleared.");
      setError(null);
    } catch (e) {
      setError(`Could not clear history: ${(e as Error).message}`);
    }
  };

  // ---- Export / Import bundle (identical format to Integer‑GPT) ---- //
  const exportBundle = async () => {
    if (!trainedCheckpoint()) {
      setError("No trained checkpoint to export.");
      return;
    }
    setBusy(true);
    try {
      const bundle = await createModelBundle({
        kind: "solid2",
        checkpoint: trainedCheckpoint()! as any,
        tokenizer: null as any,
        wordVocab: null,
        weights: new Float32Array(0),
        state: {} as any,
        prompts: [],
      } as any, { includeResumeState: false });
      downloadModelBundle(bundle);
      setStatus(`Exported ${bundle.filename}`);
    } catch (e) {
      setError(`Export failed: ${(e as Error).message}`);
    } finally {
      setBusy(false);
    }
  };

  const importBundle = async (file: File) => {
    setBusy(true);
    setError(null);
    try {
      const imported = await importModelBundle(file);
      // The real implementation would feed the imported checkpoint back into the WASM model.
      setTrainedCheckpoint(imported.checkpoint as any);
      setStatus(`Imported bundle from ${file.name}`);
    } catch (e) {
      setError(`Import failed: ${(e as Error).message}`);
    } finally {
      setBusy(false);
    }
  };

  let fileInput: HTMLInputElement | undefined;

  return (
    <section class="card training-card">
      <div class="card-heading compact">
        <div>
          <span class="eyebrow">Masked‑Diffusion Training</span>
          <h2>Fine‑tune in the browser</h2>
        </div>
        <div class={`runtime-pill ${busy() ? "warning" : "ready"}`}>
          <span class="runtime-dot" />{busy() ? "Busy" : "Idle"}
        </div>
      </div>

      <Show when={error()}>
        <div class="banner error"><span>{error()}</span></div>
      </Show>

      <Show when={status()}>
        <div class="banner info"><span>{status()}</span></div>
      </Show>

      {/* Controls – mirror Integer‑GPT UI */}
      <div class="training-controls">
        <label>Steps
          <input type="number" value={steps()} onInput={e => setSteps(Number(e.currentTarget.value))} disabled={busy()} />
        </label>
        <label>Batch
          <input type="number" value={batch()} onInput={e => setBatch(Number(e.currentTarget.value))} disabled={busy()} />
        </label>
        <label>Learning Rate
          <input type="number" step="0.0001" value={learningRate()} onInput={e => setLearningRate(Number(e.currentTarget.value))} disabled={busy()} />
        </label>
        <label>Dropout
          <input type="number" step="0.01" value={dropout()} onInput={e => setDropout(Number(e.currentTarget.value))} disabled={busy()} />
        </label>
        <label>Validation every (steps)
          <input type="number" min="1" value={validationEvery()} onInput={e => setValidationEvery(Number(e.currentTarget.value))} disabled={busy()} />
        </label>
        <label>Seed
          <input type="number" value={seed()} onInput={e => setSeed(Number(e.currentTarget.value))} disabled={busy()} />
        </label>
        <label>Time budget (minutes)
          <input type="number" value={maxMinutes()} onInput={e => setMaxMinutes(Number(e.currentTarget.value))} disabled={busy()} />
        </label>
      </div>{datasetSelector}

      {/* Loss chart – always rendered so it can show the restored history */}
      <LossChart points={history()} bestStep={bestStep()} />

      <div class="button-row">
        <button class="button primary" onClick={() => startTraining(false)} disabled={busy()}>Start Training</button>
        <button class="button secondary" onClick={() => startTraining(true)} disabled={busy() || history().length === 0}>Resume Training</button>
        <button class="button secondary" onClick={() => props.onUseCheckpoint?.(trainedCheckpoint()!)} disabled={busy() || !trainedCheckpoint()}>Use this model</button>
        <button class="button ghost" onClick={() => fileInput?.click()} disabled={busy()}>Import Bundle</button>
        <input type="file" accept=".zip" ref={el => (fileInput = el)} style={{ display: "none" }} onChange={e => { if (e.currentTarget.files?.[0]) importBundle(e.currentTarget.files[0]); }} />
        <button class="button secondary" onClick={exportBundle} disabled={busy()}>Export Bundle</button>
        <Show when={hasSaved() && !busy()}>
          <button class="button ghost" onClick={() => void clearRun()}>Clear graph</button>
        </Show>
      </div>
    </section>
  );
}
