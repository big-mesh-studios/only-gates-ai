import { createSignal, createMemo, Show, For, onCleanup } from "solid-js";
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
      <label style={{ marginRight: "1rem" }}>
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
  const bestStep = createMemo(() => {
    const h = history();
    if (h.length === 0) return null;
    let best = h[0];
    for (const p of h) if (p.loss < best.loss) best = p;
    return best.step;
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
  // ---- Stub training entry point ---- //
  // In a full implementation this function would instantiate a TF.js model that
  // mirrors the decoder architecture, apply random masks to the Luau corpus and
  // run a gradient‑descent loop. For now we only provide UI feedback.
  const startTraining = async (resume = false) => {
    if (!model()) await loadModel();
    setBusy(true);
    setError(null);
    const initial = resume ? trainedCheckpoint() : null;
    if (resume && !initial) {
      setError("No checkpoint to resume from.");
      setBusy(false);
      return;
    }
    setStatus(`Training… step 0/${steps()}`);
    const startTime = Date.now();
    const startIdx = initial ? (initial as any).step + 1 : 1;
    for (let i = startIdx; i <= steps(); i++) {
      const loss = Math.max(0, 5 * Math.exp(-i / steps() * 3) + Math.random() * 0.2);
      const point: TrainingPoint = {
        step: i,
        loss,
        smoothedLoss: loss,
        lr: learningRate(),
        elapsedMs: Date.now() - startTime,
      };
      // Add validation loss snapshot according to user frequency
      if (validationEvery() > 0 && i % validationEvery() === 0) {
        // Mock validation loss: slightly higher variance
        const valLoss = Math.max(0, loss * (0.9 + Math.random() * 0.2));
        (point as any).valLoss = valLoss;
      }
      setHistory((prev) => [...prev, point]);
      setStatus(`Training… step ${i}/${steps()}`);
      await new Promise((r) => setTimeout(r, 0));
      if (!busy()) break;
    }
    const dummyCheckpoint: TinyGptCheckpoint = { step: steps() } as any;
    setTrainedCheckpoint(dummyCheckpoint);
    setStatus("Training completed.");
    setBusy(false);
  };

  // UI control for validation frequency (added after other controls)




  // ---- Export / Import bundle (identical format to Integer‑GPT) ---- //
  const exportBundle = async () => {
    if (!trainedCheckpoint()) {
      setError("No trained checkpoint to export.");
      return;
    }
    setBusy(true);
    try {
      const bundle = await createModelBundle({
        checkpoint: trainedCheckpoint()!,
        tokenizer: null as any,
        weights: null as any,
        manifest: { model: { name: "masked-diffusion-luau", parameterCount: 0, vocabSize: 0, context: 0 } },
      }, { includeResumeState: false });
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

      {/* Loss chart */}
      <Show when={history().length > 0}>
        <LossChart points={history()} bestStep={bestStep()} />
      </Show>

      <div class="button-row">
        <button class="button primary" onClick={() => startTraining(false)} disabled={busy()}>Start Training</button>
        <button class="button secondary" onClick={() => startTraining(true)} disabled={busy() || !trainedCheckpoint()}>Resume Training</button>
        <button class="button secondary" onClick={() => props.onUseCheckpoint?.(trainedCheckpoint())} disabled={busy() || !trainedCheckpoint()}>Use this model</button>
        <button class="button ghost" onClick={() => fileInput?.click()} disabled={busy()}>Import Bundle</button>
        <input type="file" accept=".zip" ref={el => (fileInput = el)} style={{ display: "none" }} onChange={e => { if (e.currentTarget.files?.[0]) importBundle(e.currentTarget.files[0]); }} />
        <button class="button secondary" onClick={exportBundle} disabled={busy()}>Export Bundle</button>
        <button class="button warning" onClick={async () => {
          try {
            const dir = await (navigator as any).storage.getDirectory();
            await dir.removeEntry('masked-diffusion.wasm', { recursive: true });
            setStatus('Cache cleared');
          } catch (e) {
            setError(`Clear cache failed: ${(e as Error).message}`);
          }
        }} disabled={busy()}>Clear Cache</button>      </div>
    </section>
  );
}
