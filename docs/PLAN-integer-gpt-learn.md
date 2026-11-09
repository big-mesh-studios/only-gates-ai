# Plan: Integer GPT learning site + resumable community training

Status: agreed direction, implementation in progress.

## 1. Objective

Add a static `/learn/` site to the only-gates-ai web app that teaches how
Integer GPT works under the hood, explains the path from integer inference to
pure logic gates to an FPGA, and rallies contributors to donate training
compute. Make the training scripts resumable so free Colab sessions add up
instead of starting over.

## 2. Route and architecture

- Route: `/learn/` served as `apps/only-gates-ai/dist/learn/index.html`.
- GitHub Pages hosts static files, so section navigation uses hash fragments
  (`/learn/#integers`) and there is no path-based client router. A real
  `learn/index.html` means no rewrite rule and no 404 on deep links.
- Separate workspace app: `apps/only-gates-ai-learn` (`@only-gates-ai/learn`).
- Solid 2 static pattern copied from `big-mesh-studios/apps/place-docs` and
  `apps/homepage`:
  - `@solidjs/vite-plugin` with `solid({ ssr: true, solid: { hydratable: false } })`
  - `vite build --ssr src/entry-server.tsx --outDir .ssr`
  - `scripts/prerender.ts` calls `renderToString`, inlines CSS, writes one HTML
    file.
- No hydration and no client JavaScript.
- Build assembly: `scripts/assemble-learn.mjs` copies the prerendered learn
  output into `apps/only-gates-ai/dist/learn/`, so the GitHub Pages publish
  directory stays `apps/only-gates-ai/dist`.
- Root scripts:
  - `dev:learn`
  - `build:learn`
  - `build:site` = app build + learn build + assemble
- Pages workflow switches from `pnpm build:app` to `pnpm build:site`.

Links between the SPA and the static site:

- App header / Integer GPT panel: "How it works" → `learn/`.
- Learn page: "Open the demo" → `../`.
- Dev: app on `:5174`, learn on `:5175`, absolute localhost links.

## 3. Page content

One long-form page with a sticky anchor nav:

| Section | Content |
| --- | --- |
| Hero | "A GPT that runs on integers", status chips, demo + help-train CTAs |
| One-minute version | Training uses floats and gradients; inference is repeated arithmetic and can be integer-only |
| GPT anatomy | tokenizer → embeddings → transformer blocks → logits → sampler, using real Alice and Solid 2 configs |
| Why integers | int8 weights, Q16 activations, int32 accumulation, int64 requantization, determinism |
| Integer toolbox | RMSNorm, GELU LUT, DI-Exp softmax, attention quantization, xorshift sampling |
| Float → integer | train float → export → quantize at load → integer forward → validate |
| Integers → gates | fixed-point ops lower to adders, comparators, shifts, muxes, ROM/LUTs |
| Why FPGA | deterministic latency, low power, offline, pipelining; also fixed model and memory limits |
| Roadmap | integer WASM shipped; replaceable checkpoints; one block lowered; full netlist; FPGA prototype |
| Help train | compute, code, hardware tracks; free Colab steps; mission board |
| FAQ | GPU needed? bit-identical? is it on FPGA yet? |

Recurring honesty box:

> Today: integer-only inference in WebAssembly. Prototype: the gate compiler for
> the hard-mode symbol classifier. Roadmap: gate-lowered GPT inference and an
> FPGA implementation.

Required inline SVG diagrams:

1. Training vs inference lanes.
2. Transformer block with tensor shapes.
3. Precision map (int8 / Q16 / int32 / int64 / Q15 / Q12).
4. Quantization pipeline.
5. Lowering stack: float model → integer graph → gate netlist → bitstream.
6. FPGA floorplan.

All SVG components use `role="img"`, `<title>`, `<desc>`, a responsive
`viewBox`, and a visible caption.

Grounding sources:

- `src/tiny_gpt_quantize.c` — only place float is allowed.
- `src/tiny_gpt_infer.c` — integer-only forward pass.
- `src/int_math.h` — Q16/Q15/Q12 helpers, integer sqrt, DI-Exp.
- `src/tiny_gpt.h` — storage formats and GELU LUT.
- `packages/only-gates-ai-compiler/src/gates.ts` — 16-gate vocabulary.
- `apps/only-gates-ai/src/lib/tinyGpt.ts` — float vs int8 comparison.
- `apps/only-gates-ai/src/lib/tinyGptSolid2.ts` — integer WASM loader.
- `scripts/train_tiny_gpt_solid2_colab.py` — training path.
- `data/solid2/` — docs/Q&A corpus.

Anchor numbers:

- Alice demo: 62,448 params, 512 vocab, 16 context, 48 dim, 2 layers, 2 heads,
  96 MLP.
- Solid 2 target: 264,528 params, 4,096 BPE vocab, 256 context, 48 dim,
  3 layers, 4 heads, 96 MLP.
- int8 weights are roughly 4× smaller than float32.

Artifact caveat: Solid 2 `weights.bin` and `tokenizer.json` are gitignored and
absent. The page presents Alice as the built-in working example and Solid 2 as
the training target unless release-hosted artifacts are added.

## 4. Resumable training (Phase 0.5)

### Current gaps

`scripts/train_tiny_gpt_solid2_colab.py` retrains the tokenizer every run,
creates fresh AdamW/LambdaLR for LM and QA, and saves only `best-qa.pt` and
`final-state.pt`. No step, epoch, RNG state, or data fingerprint is stored.

### File layout

```text
<out>/latest.pt                  full training state (atomic replace)
<out>/checkpoints/step-N.pt      rolling snapshots (keep N)
<out>/best-qa.pt                 best QA weights
<out>/tokenizer.json             written once, reused on resume
<out>/metrics.jsonl              append-only metrics
<out>/run.json                   manifest and resume history
<out>/final-state.pt             exported final weights
```

### Checkpoint schema

```python
{
  "format": "only-gates-train-state",
  "version": 1,
  "stage": "lm" | "qa",
  "lm": {"step": 4900, "steps": 5000},
  "qa": {
    "epoch": 37,
    "cursor": 128,
    "perm": [...],
    "best_val": 1.4231,
    "best_epoch": 34,
  },
  "model": model.state_dict(),
  "optimizer": opt.state_dict(),
  "scheduler": sch.state_dict(),
  "rng": {
    "python": random.getstate(),
    "numpy": np.random.get_state(),
    "torch": torch.get_rng_state(),
    "cuda": torch.cuda.get_rng_state_all(),
  },
  "config": {...},
  "fingerprint": {
    "docs": "...", "qa": "...", "stories": "...",
    "lm_ids": "...", "tokenizer": "...", "seed": 1337,
  },
  "tokenizer_json": "...",
  "history": {"lm_loss": [...], "qa": [...]},
  "wall_seconds": 8123.4,
  "environment": {"python": "...", "torch": "...", "device": "..."},
}
```

Key rules:

- Save the tokenizer before LM training and embed it in the checkpoint. Never
  retrain BPE on resume.
- Save the QA epoch permutation and cursor so a mid-epoch resume continues the
  same example order.
- Fingerprint docs, QA data, stories, tokenized stream, tokenizer, and seed.
  Refuse resume on mismatch.
- Atomic write via `latest.pt.tmp` then `os.replace`.
- Keep `latest.pt` plus N rolling snapshots.

### CLI

```text
--resume                 load <out>/latest.pt if present
--resume-from PATH       explicit checkpoint
--fresh                  ignore existing checkpoints
--save-every N           default 100 optimizer steps
--max-minutes N          graceful time-budget stop
--max-hours N            alias
--keep-checkpoints N     default 3
--stage lm|qa|all        default all
--data-cache             cache tokenized lm_ids
--export-only            rebuild exported artifacts from latest state
```

Behavior:

1. Build corpus/tokenizer only when no usable checkpoint exists.
2. Verify config, seed, tokenizer hash, data hashes, and planned step counts.
3. Skip LM when the checkpoint is in QA; skip QA when complete.
4. Restore model, optimizer, scheduler, RNG, and QA cursor.
5. Save on budget stop, `SIGINT`, and `SIGTERM`; hard disconnects are covered
   by `--save-every`.
6. Exit 0 after a budget stop and print the exact resume command.
7. Append metrics to `metrics.jsonl` on every save.

### Determinism

Restoring state gives correct continuation, not guaranteed bit-identical output
across GPUs, CUDA versions, or CPU/GPU. Document that:

- same hardware + seed + PyTorch version: close to identical;
- different hardware: same data order and schedule, statistically equivalent;
- CPU tests assert exact equality on a tiny config.

### Tests

- Interrupt a 20-step CPU run after 10, resume, compare with uninterrupted run.
- Resume across the LM → QA boundary; assert LM is not retrained.
- Resume mid-QA-epoch; assert example order matches `perm` / `cursor`.
- Config and seed mismatches refuse resume.
- Tokenizer hash is unchanged across resume.
- `--max-minutes` exits 0 and prints a resume command.
- `--smoke --resume` completes without retraining the tokenizer.

### JS trainer

`scripts/train-tiny-gpt.mjs` stays a dependency-free reference implementation
and is documented as non-resumable for now. The PyTorch trainer is the
community compute path.

## 5. Community enablement

Add:

- `notebooks/train_tiny_gpt_solid2_colab.ipynb` with first-session and
  next-session cells, Drive mount, smoke test, and resume command.
- `.github/ISSUE_TEMPLATE/compute-pledge.yml`
- `.github/ISSUE_TEMPLATE/training-result.yml`
- `.github/ISSUE_TEMPLATE/hardware-benchmark.yml`
- `docs/COMPUTE.md` with artifact naming, evaluation protocol, checkpoint
  handoff, credit policy, and licensing.

Free Colab flow:

```bash
# first session
python scripts/train_tiny_gpt_solid2_colab.py \
  --out /content/drive/MyDrive/tiny-gpt-solid2 \
  --lm-steps 5000 --qa-epochs 100 \
  --save-every 100 --max-minutes 300

# later session
python scripts/train_tiny_gpt_solid2_colab.py \
  --out /content/drive/MyDrive/tiny-gpt-solid2 \
  --lm-steps 5000 --qa-epochs 100 \
  --save-every 100 --max-minutes 300 \
  --resume
```

Site copy: free sessions are short; checkpoints to Drive let GPU time add up.

Training-result template asks for checkpoint SHA-256, sessions used, steps or
epochs completed, hardware, wall time, and whether the run resumed.

## 6. Implementation phases

1. **Phase 0.5 — resumable training**
   - Refactor `train_lm` / `train_qa` around a training state.
   - Checkpoint schema, atomic saves, rolling snapshots, metrics, manifest.
   - Tokenizer and data reuse with fingerprint verification.
   - Budget stop, signal handling, CLI flags.
   - CPU resume tests.

2. **Phase 1 — site MVP**
   - Scaffold `apps/only-gates-ai-learn`.
   - `entry-server.tsx`, `prerender.ts`, styles, content module.
   - All sections and six diagrams.
   - Assembly script, root scripts, Pages workflow.
   - "How it works" links in the SPA.

3. **Phase 2 — community enablement**
   - Colab notebook, issue templates, `docs/COMPUTE.md`, mission board.

4. **Phase 3 — polish**
   - OG image, canonical URL, sitemap/robots, print styles, diagram
     transcripts.

## 7. Verification

- `pnpm --filter @only-gates-ai/learn typecheck`
- `pnpm build:site` produces `apps/only-gates-ai/dist/learn/index.html`
- Built learn page contains no unexpected `<script>` tags
- Anchors resolve; relative links exist
- Informative SVGs have `role="img"`, `<title>`, `<desc>`
- Keyboard navigation and focus states work
- Responsive at 320 / 768 / 1280 px
- Print layout readable
- Existing app tests and `pnpm typecheck:app` still pass
- Resume tests listed in section 4 pass
