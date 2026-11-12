import type { TinyGptConfig } from "./tinyGpt";

/*
 * Shared constants for the deployed Solid 2 demo and its browser trainer.
 *
 * Keeping these in one module avoids a circular import between the inference
 * panel and the data loader, and keeps `tinyGptSolid2.ts` free of
 * `import.meta.env` so its standalone regression test can transpile it.
 */

export const SOLID2_MODEL_BASE = `${import.meta.env.BASE_URL}models/tiny-gpt-solid2/`;

export const SOLID2_DATA_BASE =
  "https://raw.githubusercontent.com/big-mesh-studios/only-gates-ai/main/data/solid2/";

export const SOLID2_TRAIN_CONFIG: TinyGptConfig = {
  vocabSize: 4096,
  context: 256,
  dim: 128,
  layers: 4,
  heads: 4,
  mlp: 512,
};

export function solid2ParameterCount(config: TinyGptConfig): number {
  const { vocabSize, context, dim, layers, mlp } = config;
  return (
    vocabSize * dim +
    context * dim +
    dim +
    layers * (2 * dim + 4 * dim * dim + 2 * dim * mlp)
  );
}
