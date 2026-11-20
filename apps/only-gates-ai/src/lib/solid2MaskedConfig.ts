import type { TinyGptConfig } from "./tinyGpt";

export const SOLID2_MASKED_MODEL_BASE = `${import.meta.env.BASE_URL}models/solid2-masked-diffusion/`;

export const SOLID2_MASKED_DATA_BASE =
  "https://raw.githubusercontent.com/clinuxrulz/only-gates-ai/main/data/solid2/";

export const SOLID2_MASKED_TRAIN_CONFIG: TinyGptConfig = {
  vocabSize: 4097,
  context: 256,
  dim: 384,
  layers: 6,
  heads: 6,
  mlp: 1152,
};

export type Solid2MaskedMode = "solid2";

export interface Solid2MaskedPreset {
  readonly label: string;
  readonly mode: Solid2MaskedMode;
  readonly prompt: string;
}

export const SOLID2_MASKED_PRESETS: readonly Solid2MaskedPreset[] = [
  {
    label: "Split effects",
    mode: "solid2",
    prompt:
      "What are split effects in Solid 2 and why were createEffect split into compute and apply functions?",
  },
  {
    label: "Migrate createEffect",
    mode: "solid2",
    prompt:
      "How do I migrate a Solid 1 createEffect to the Solid 2 split-effect format?",
  },
  {
    label: "Create effect signature",
    mode: "solid2",
    prompt:
      "What is the new createEffect API in Solid 2 that takes two functions (compute, apply)?",
  },
  {
    label: "Signals derived",
    mode: "solid2",
    prompt:
      "Explain how derive works in Solid 2 and how it differs from computed signals in 1.x.",
  },
  {
    label: "Stores and ownership",
    mode: "solid2",
    prompt:
      "What are the rules for writing to signals under owned scope in Solid 2?",
  },
  {
    label: "Control flow",
    mode: "solid2",
    prompt:
      "How does Solid 2 handle control flow like if/else and loops in component bodies?",
  },
];

export function solid2MaskedParameterCount(config: TinyGptConfig): number {
  const { vocabSize, context, dim, layers, mlp } = config;
  return (
    vocabSize * dim +
    context * dim +
    dim +
    layers * (2 * dim + 4 * dim * dim + 2 * dim * mlp)
  );
}