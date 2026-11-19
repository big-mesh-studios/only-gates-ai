# Masked-diffusion Luau model

This directory holds the model for the **Masked diffusion** tab. The app loads
these files lazily the first time the tab is opened:

- `checkpoint.json` — model config, parameter count, mask id, denoising steps,
  prompt examples, and training metadata;
- `tokenizer.json` — the Luau byte-level BPE tokenizer with `<|mask|>` appended
  at the end of the vocabulary (id 1,721 in the deployed 1,722-token
  vocabulary);
- `weights.bin` — float32 little-endian weights in `tiny_gpt_create` order.

The checkpoint is produced by
`scripts/train_masked_diffusion_luau_colab.py`:

```sh
python scripts/train_masked_diffusion_luau_colab.py \
  --tasks data/luau-tasks \
  --out masked-diffusion-luau \
  --context 128 --dim 384 --layers 6 --heads 6 --mlp 1152 \
  --epochs 40 --batch 24 --lr 3e-4 --steps 16 --max-minutes 180
```

The trainer also writes `latest.pt` (resumable training state) and `README.md`
in the output directory. Copy only the three inference artifacts above into
this folder when publishing.

## Architecture

The denoiser is a bidirectional Tiny GPT with the same packed layout as the
causal Luau model: pre-norm RMSNorm, GELU-tanh, no biases, tied input/output
embeddings, and learned absolute positions. Every position attends to every
other position in the sequence.

Training masks a random fraction of the answer region and supervises only the
masked, non-pad positions. Inference starts with the whole answer region set to
`<|mask|>`. Each C denoising step runs one full bidirectional pass and fills the
leftmost `ceil(remaining / steps_left)` masked positions. Filling left to right
keeps variable names consistent across an answer, and the UI paints between
steps so the token strip and preview update live.
