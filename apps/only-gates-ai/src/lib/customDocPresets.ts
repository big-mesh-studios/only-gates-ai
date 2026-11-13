/*
 * Curated document sets for the custom browser trainer.
 *
 * Raw GitHub URLs are used because they are stable, plain text, and served with
 * `Access-Control-Allow-Origin: *` so the browser can fetch them directly.
 */

export interface CustomDocPreset {
  readonly id: string;
  readonly label: string;
  readonly description: string;
  readonly license: string;
  readonly urls: readonly string[];
}

const LUAU_DOCS_BASE =
  "https://raw.githubusercontent.com/luau-lang/site/master/src/content/docs/";
const LUAU_FAQ =
  "https://raw.githubusercontent.com/clinuxrulz/only-gates-ai/main/apps/only-gates-ai/src/assets/luau-faq.md";

export const CUSTOM_DOC_PRESETS: readonly CustomDocPreset[] = [
  {
    id: "luau",
    label: "Load Luau docs example",
    description:
      "Official Luau language, library, type-system, lint, and performance docs, plus a short natural-language Luau quick-answer guide.",
    license: "MIT (luau-lang/site); repository quick answers (MIT)",
    urls: [
      LUAU_FAQ,
      `${LUAU_DOCS_BASE}getting-started/intro.md`,
      `${LUAU_DOCS_BASE}getting-started/syntax.md`,
      `${LUAU_DOCS_BASE}getting-started/compatibility.md`,
      `${LUAU_DOCS_BASE}getting-started/lint.md`,
      `${LUAU_DOCS_BASE}reference/library.md`,
      `${LUAU_DOCS_BASE}reference/types-library.md`,
      `${LUAU_DOCS_BASE}reference/attributes.md`,
      `${LUAU_DOCS_BASE}guides/performance.md`,
      `${LUAU_DOCS_BASE}guides/sandbox.md`,
      `${LUAU_DOCS_BASE}types/overview.md`,
      `${LUAU_DOCS_BASE}types/basic-types.md`,
      `${LUAU_DOCS_BASE}types/tables.md`,
      `${LUAU_DOCS_BASE}types/generics.md`,
      `${LUAU_DOCS_BASE}types/unions-and-intersections.md`,
      `${LUAU_DOCS_BASE}types/refinements.md`,
      `${LUAU_DOCS_BASE}types/object-oriented-programs.md`,
      `${LUAU_DOCS_BASE}types/type-functions.md`,
    ],
  },
  {
    id: "luau-quick",
    label: "Load Luau quick answers",
    description:
      "A small natural-question Luau Q&A set with control structures, basic syntax, and tables. Tuned for simple questions and the 1.34M-parameter model.",
    license: "MIT (luau-lang/site); repository quick answers (MIT)",
    urls: [
      LUAU_FAQ,
      `${LUAU_DOCS_BASE}getting-started/intro.md`,
      `${LUAU_DOCS_BASE}getting-started/syntax.md`,
      `${LUAU_DOCS_BASE}types/tables.md`,
    ],
  },
];
