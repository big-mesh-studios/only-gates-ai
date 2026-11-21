import type { TinyGptConfig } from "./tinyGpt";

/*
 * Configuration for the dedicated Luau tab.
 *
 * The tab loads its own bundle from `public/models/tiny-gpt-luau/`, independent
 * of the Solid 2 and Custom Docs models. The checkpoint is produced by
 * `scripts/train_tiny_gpt_luau_colab.py`.
 */

export const LUAU_MODEL_BASE = `${import.meta.env.BASE_URL}models/tiny-gpt-luau/`;

export const LUAU_TRAIN_CONFIG: TinyGptConfig = {
  vocabSize: 1721,
  context: 256,
  dim: 256,
  layers: 6,
  heads: 8,
  mlp: 1024,
};

export type LuauMode = "write" | "ask";

export interface LuauPreset {
  readonly label: string;
  readonly mode: LuauMode;
  readonly prompt: string;
}

/*
 * Keep these prompts close to the canonical prompts used by
 * `scripts/luau_tasks.py` and the FAQ headings in `luau-faq.md`.
 */
export const LUAU_PRESETS: readonly LuauPreset[] = [
  {
    label: "Sum first 50",
    mode: "write",
    prompt:
      "Write a for loop that sums the first 50 elements of the array `values` and prints the result in Luau. The array has at least 50 elements.",
  },
  {
    label: "Print 1 to 20",
    mode: "write",
    prompt:
      "Write a for loop in Luau that runs from 1 to 20 and prints every number.",
  },
  {
    label: "Count above 80",
    mode: "write",
    prompt:
      "Write a for loop that counts how many elements of `scores` are greater than 80, then prints the count in Luau.",
  },
  {
    label: "Sum above threshold",
    mode: "write",
    prompt:
      "Write a for loop in Luau that sums the elements of `values` greater than 10 and prints the total.",
  },
  {
    label: "Count and sum",
    mode: "write",
    prompt:
      "Write a for loop in Luau that counts how many elements of `values` are greater than 10 and also sums those elements, printing the count then the total.",
  },
  {
    label: "Running min and max",
    mode: "write",
    prompt:
      "Write a for loop that finds both the smallest and largest value in `values` and prints them on separate lines in Luau.",
  },
  {
    label: "Search first index",
    mode: "write",
    prompt:
      "Given `values = { 4, 7, 2, 7, 9 }`, write a for loop in Luau that finds the first index where the value is 7 and prints it (or 0 if not found).",
  },
  {
    label: "Transform and total",
    mode: "write",
    prompt:
      "Given `values = { 1, 2, 3, 4 }`, write Luau code that builds a new table `doubled` with every value doubled, then prints the total of `doubled`.",
  },
  {
    label: "Nested loop sum",
    mode: "write",
    prompt:
      "Given `matrix = { { 1, 2 }, { 3, 4 } }`, write nested for loops in Luau that sum every element of `matrix` and print the total.",
  },
  {
    label: "Add function",
    mode: "write",
    prompt:
      "Write a local function named `add` in Luau that takes `a` and `b` and returns their sum. Then print `add(2, 3)`.",
  },
  {
    label: "Function returns sum",
    mode: "write",
    prompt:
      "Write a local function named `total` in Luau that takes `nums` and returns the sum of its elements. Then print `total(values)`.",
  },
  {
    label: "Function count above",
    mode: "write",
    prompt:
      "Write a local function named `countAbove` in Luau that takes `nums` and `limit` and returns how many elements of `nums` are greater than `limit`. Then print `countAbove(values, 10)`.",
  },
  {
    label: "Iterate table",
    mode: "write",
    prompt:
      'Given the array `items = { "sword", "shield", "potion" }`, write a loop in Luau that prints each index and value.',
  },
  {
    label: "While sum",
    mode: "write",
    prompt:
      "Use a while loop in Luau to add the numbers 1 through 10 and print the total.",
  },
  {
    label: "Question style",
    mode: "write",
    prompt:
      "How do I sum the first 50 elements of the array `values` in Luau?",
  },
  {
    label: "For loop question",
    mode: "ask",
    prompt: "How do you do a for loop in Luau?",
  },
  {
    label: "Table question",
    mode: "ask",
    prompt: "How do you create a table in Luau?",
  },
  {
    label: "Type annotations",
    mode: "ask",
    prompt: "How do you add type annotations in Luau?",
  },
];

export function luauParameterCount(config: TinyGptConfig): number {
  const { vocabSize, context, dim, layers, mlp } = config;
  return (
    vocabSize * dim +
    context * dim +
    dim +
    layers * (2 * dim + 4 * dim * dim + 2 * dim * mlp)
  );
}
