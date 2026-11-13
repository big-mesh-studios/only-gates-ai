import luauFaq from "../assets/luau-faq.md?raw";

/*
 * Generates a Google Colab notebook for the Custom Docs (BPE) trainer.
 *
 * The notebook is built from the URLs and training settings currently in the
 * browser panel. It runs scripts/train_tiny_gpt_custom_colab.py and downloads
 * an inference-only model bundle that the existing "Import model (.zip)" flow
 * can load.
 */

export interface CustomTrainingNotebookOptions {
  readonly urls: readonly string[];
  readonly assistant: boolean;
  readonly steps: number;
  readonly batch: number;
  readonly learningRate: number;
  readonly dropout: number;
  readonly seed: number;
  /** 0 means no time budget. */
  readonly maxMinutes: number;
  readonly warmStart: boolean;
  readonly repoUrl?: string;
  /** Colab output directory. Defaults to Drive/MyDrive/tiny-gpt-custom. */
  readonly outputDir?: string;
}

export interface GeneratedCustomNotebook {
  readonly filename: string;
  readonly json: string;
}

const MAX_URLS = 256;
const DEFAULT_REPO = "https://github.com/clinuxrulz/only-gates-ai.git";

function source(text: string): string[] {
  const lines = text.replace(/\r\n/g, "\n").split("\n");
  return lines.map((line, index) =>
    index === lines.length - 1 ? line : `${line}\n`,
  );
}

function markdownCell(text: string): Record<string, unknown> {
  return {
    cell_type: "markdown",
    metadata: {},
    source: source(text),
  };
}

function codeCell(text: string): Record<string, unknown> {
  return {
    cell_type: "code",
    execution_count: null,
    metadata: {},
    outputs: [],
    source: source(text),
  };
}

function shellCommand(parts: readonly string[]): string {
  return parts.join(" \\\n  ");
}

function slugFor(urls: readonly string[]): string {
  const first = urls[0];
  if (!first) return "docs";
  try {
    const url = new URL(first);
    if (url.hostname === "raw.githubusercontent.com") {
      const parts = url.pathname.split("/").filter(Boolean);
      const slug = parts.slice(0, 2).join("-");
      if (slug) return slug.replace(/[^a-z0-9-]+/gi, "-").slice(0, 48);
    }
    return url.hostname
      .replace(/^www\./, "")
      .replace(/[^a-z0-9-]+/gi, "-")
      .replace(/^-+|-+$/g, "")
      .slice(0, 48) || "docs";
  } catch {
    return "docs";
  }
}

export function buildCustomTrainingNotebook(
  options: CustomTrainingNotebookOptions,
): GeneratedCustomNotebook {
  const urls = options.urls.map((url) => url.trim()).filter(Boolean);
  if (urls.length === 0) {
    throw new Error("Add at least one document URL before creating a notebook.");
  }
  if (urls.length > MAX_URLS) {
    throw new Error(`At most ${MAX_URLS} document URLs are supported.`);
  }
  if (!Number.isFinite(options.steps) || options.steps < 1) {
    throw new Error("Training steps must be a positive number.");
  }
  if (!Number.isFinite(options.batch) || options.batch < 1) {
    throw new Error("Batch size must be a positive number.");
  }
  if (!Number.isFinite(options.learningRate) || options.learningRate <= 0) {
    throw new Error("Learning rate must be a positive number.");
  }
  if (!Number.isFinite(options.seed)) {
    throw new Error("Seed must be a finite number.");
  }

  const repoUrl = options.repoUrl ?? DEFAULT_REPO;
  const outputDir = options.outputDir ?? "/content/drive/MyDrive/tiny-gpt-custom";
  // The Luau preset ships a natural-question FAQ from this app's own assets.
  // Write it into Colab instead of depending on the file being published on
  // the repository's main branch.
  const faqSuffix = "/apps/only-gates-ai/src/assets/luau-faq.md";
  const usesBundledFaq = urls.some((url) => url.endsWith(faqSuffix));
  const notebookUrls = urls.map((url) =>
    url.endsWith(faqSuffix) ? "file:///content/luau-faq.md" : url,
  );
  const urlLines = notebookUrls
    .map((url) => `  ${JSON.stringify(url)},`)
    .join("\n");
  const assistantFlag = options.assistant ? ["--assistant"] : [];
  const warmStartFlag = options.warmStart ? ["--warm-start"] : [];
  const timeBudget =
    options.maxMinutes > 0 ? [`--max-minutes ${options.maxMinutes}`] : [];
  const modeLabel = options.assistant
    ? "assistant (Q&A) mode"
    : "plain document continuation";

  const intro = `# Integer GPT — Custom Docs training on Colab

This notebook trains a custom decoder-only Integer GPT model on the documents
listed below. Training uses floating point on a Colab GPU; the downloadable
bundle is quantized to int8/Q16 by the app.

**Before you start:** open **Runtime → Change runtime type → T4 GPU**.

Generated from the app with:
- ${urls.length} document URL${urls.length === 1 ? "" : "s"}
- ${modeLabel}
- ${options.steps.toLocaleString()} steps, batch ${options.batch}, learning rate ${options.learningRate}
- warm start: ${options.warmStart ? "on" : "off"}

The download cell saves \`only-gates-custom-bundle.zip\`. Import it with
**Integer GPT → Custom Docs (BPE) → Import model (.zip)**.

Assistant mode builds its answers from Markdown headings. Headings written as
natural questions (for example, \`## How do you do a for loop in Luau?\`) make
the model respond better to the way you actually ask questions.

The documents keep their own licenses; this notebook does not relicense them.`;

  const cloneRepo = `from pathlib import Path

REPO = Path('/content/only-gates-ai')
if not REPO.exists():
    !git clone --depth 1 ${repoUrl} {REPO}
%cd {REPO}
!git pull --ff-only || true`;

  const urlCell = `from pathlib import Path

URLS = [
${urlLines}
]

URLS_FILE = Path('/content/only-gates-custom-urls.txt')
URLS_FILE.write_text("\\n".join(URLS) + "\\n", encoding='utf-8')
print(f'{len(URLS)} document URLs written to {URLS_FILE}')`;

  const inspectCommand = shellCommand([
    "!python scripts/train_tiny_gpt_custom_colab.py",
    "--urls-file {URLS_FILE}",
    ...(options.assistant ? ["--assistant"] : []),
    "--inspect-only",
  ]);

  const trainParts = [
    "!python scripts/train_tiny_gpt_custom_colab.py",
    "--urls-file {URLS_FILE}",
    "--out {OUT}",
    `--steps ${Math.round(options.steps)}`,
    `--batch ${Math.round(options.batch)}`,
    `--lr ${options.learningRate}`,
    `--dropout ${options.dropout}`,
    `--seed ${Math.round(options.seed)}`,
    ...assistantFlag,
    ...warmStartFlag,
    ...timeBudget,
    "--save-every 100",
    "--val-every 100",
    "--export-final",
    "--bundle-out /content/only-gates-custom-bundle.zip",
    "--resume",
  ];

  const notebook = {
    nbformat: 4,
    nbformat_minor: 0,
    metadata: {
      accelerator: "GPU",
      colab: { provenance: [] },
      kernelspec: { display_name: "Python 3", name: "python3" },
      language_info: { name: "python" },
    },
    cells: [
      markdownCell(intro),
      codeCell(`from pathlib import Path

try:
    from google.colab import drive
    drive.mount('/content/drive')
    OUT = Path(${JSON.stringify(outputDir)})
except ImportError:
    OUT = Path('/content/tiny-gpt-custom')

OUT.mkdir(parents=True, exist_ok=True)
print('Training output:', OUT)`),
      codeCell(cloneRepo),
      ...(usesBundledFaq
        ? [
            codeCell(`from pathlib import Path

FAQ_PATH = Path('/content/luau-faq.md')
FAQ_PATH.write_text(${JSON.stringify(luauFaq)}, encoding='utf-8')
print('wrote', FAQ_PATH)`),
          ]
        : []),
      markdownCell(`## 1. Document URLs

These are the normalized URLs from the app. Edit the list if you want to add
or remove documents before training.`),
      codeCell(urlCell),
      codeCell(`# The trainer uses the deployed BPE tokenizer; regex improves
# Unicode pre-tokenization parity with the browser.
try:
    import regex  # noqa: F401
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'regex'])`),
      markdownCell(`## 2. Check the URLs

This fetches and tokenizes the documents without training, so bad links or
empty pages fail quickly.`),
      codeCell(inspectCommand),
      markdownCell(`## 3. Train or resume

Rerunning this cell resumes from the last checkpoint. Increase \`--steps\` for a
better model; 3,000–5,000 steps is a reasonable T4 starting point. The time
budget comes from the app's **Time budget** field; raise or remove
\`--max-minutes\` for a longer run. The cell exports the final weights
(\`--export-final\`) because small Q&A models often overfit their instructions;
an early best-validation checkpoint can underfit them. Add \`--fresh\` only
when you intentionally changed the URLs or settings.`),
      codeCell(shellCommand(trainParts)),
      markdownCell(`## 4. Download the bundle

The training cell always exports the current model, even if it stopped early.
Import the downloaded ZIP with **Integer GPT → Custom Docs (BPE) →
Import model (.zip)**.`),
      codeCell(`import hashlib
from pathlib import Path
from google.colab import files

bundle = Path('/content/only-gates-custom-bundle.zip')
if bundle.exists():
    data = bundle.read_bytes()
    print(f'{bundle} · {len(data):,} bytes')
    print('sha256:', hashlib.sha256(data).hexdigest())
    files.download(str(bundle))
else:
    print('No bundle yet. Run the training cell first.')`),
      markdownCell(`## Notes

- The output directory is on Google Drive by default, so a dropped session can
  resume from the same checkpoint.
- The bundle is inference-only; it does not include optimizer or resume state.
- The trainer prints a loss line every 10 steps and a validation loss every 100
  steps.
- If the session ends, rerun the training cell. It uses \`--resume\`.`),
    ],
  };

  return {
    filename: `only-gates-custom-${slugFor(urls)}-colab.ipynb`,
    json: JSON.stringify(notebook, null, 2),
  };
}

export function downloadCustomNotebook(notebook: GeneratedCustomNotebook): void {
  const blob = new Blob([notebook.json], {
    type: "application/x-ipynb+json",
  });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = notebook.filename;
  anchor.rel = "noopener";
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}
