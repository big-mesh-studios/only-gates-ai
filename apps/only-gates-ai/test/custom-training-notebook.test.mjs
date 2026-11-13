import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import test from "node:test";

import { createServer } from "vite";

/*
 * Colab notebook generator tests.
 *
 * These stay at the JSON level: the notebook is a portable artifact, so the
 * important contract is nbformat validity plus the URLs/settings embedded in
 * the generated shell cells.
 */

const appRoot = fileURLToPath(new URL("..", import.meta.url));

async function createTestServer() {
  return createServer({
    root: appRoot,
    appType: "custom",
    logLevel: "silent",
    server: { middlewareMode: true },
    optimizeDeps: { noDiscovery: true },
  });
}

function cellText(cell) {
  return Array.isArray(cell.source) ? cell.source.join("") : cell.source;
}

test("custom notebook embeds URLs, settings, and the bundle command", async () => {
  const server = await createTestServer();
  try {
    const notebook = await server.ssrLoadModule(
      "/src/lib/customTrainingNotebook.ts",
    );
    const generated = notebook.buildCustomTrainingNotebook({
      urls: [
        "https://raw.githubusercontent.com/luau-lang/site/master/src/content/docs/getting-started/intro.md",
        "https://example.com/docs/api.html?version=2&lang=en",
      ],
      assistant: true,
      steps: 600,
      batch: 4,
      learningRate: 0.0004,
      dropout: 0.1,
      seed: 1337,
      maxMinutes: 30,
      warmStart: true,
    });

    assert.match(generated.filename, /^only-gates-custom-.*\.ipynb$/);
    const parsed = JSON.parse(generated.json);
    assert.equal(parsed.nbformat, 4);
    assert.equal(parsed.nbformat_minor, 0);
    assert.ok(parsed.cells.length > 5);

    const allCode = parsed.cells
      .filter((cell) => cell.cell_type === "code")
      .map(cellText)
      .join("\n");
    const allText = parsed.cells.map(cellText).join("\n");

    assert.ok(allText.includes("https://example.com/docs/api.html?version=2&lang=en"));
    assert.ok(allText.includes("luau-lang/site"));
    assert.match(allCode, /--steps 600/);
    assert.match(allCode, /--batch 4/);
    assert.match(allCode, /--lr 0\.0004/);
    assert.match(allCode, /--dropout 0\.1/);
    assert.match(allCode, /--seed 1337/);
    assert.match(allCode, /--max-minutes 30/);
    assert.match(allCode, /--assistant/);
    assert.match(allCode, /--warm-start/);
    assert.match(allCode, /--export-final/);
    assert.match(allCode, /--inspect-only/);
    assert.match(allCode, /--bundle-out \/content\/only-gates-custom-bundle\.zip/);
    assert.match(allCode, /--resume/);

    const urlsCell = parsed.cells.find((cell) =>
      cellText(cell).includes("URLS = ["),
    );
    assert.ok(urlsCell, "notebook should contain an editable URL list");
    assert.ok(cellText(urlsCell).includes("https://example.com/docs/api.html?version=2&lang=en"));
  } finally {
    await server.close();
  }
});

test("custom notebook omits assistant/warm-start flags when disabled", async () => {
  const server = await createTestServer();
  try {
    const notebook = await server.ssrLoadModule(
      "/src/lib/customTrainingNotebook.ts",
    );
    const generated = notebook.buildCustomTrainingNotebook({
      urls: ["https://example.com/guide.md"],
      assistant: false,
      steps: 100,
      batch: 2,
      learningRate: 0.001,
      dropout: 0,
      seed: 7,
      maxMinutes: 0,
      warmStart: false,
    });
    const parsed = JSON.parse(generated.json);
    const code = parsed.cells
      .filter((cell) => cell.cell_type === "code")
      .map(cellText)
      .join("\n");
    assert.doesNotMatch(code, /--assistant/);
    assert.doesNotMatch(code, /--warm-start/);
    assert.doesNotMatch(code, /--max-minutes/);
  } finally {
    await server.close();
  }
});

test("custom notebook embeds the Luau FAQ locally", async () => {
  const server = await createTestServer();
  try {
    const notebook = await server.ssrLoadModule(
      "/src/lib/customTrainingNotebook.ts",
    );
    const generated = notebook.buildCustomTrainingNotebook({
      urls: [
        "https://raw.githubusercontent.com/clinuxrulz/only-gates-ai/main/apps/only-gates-ai/src/assets/luau-faq.md",
        "https://example.com/guide.md",
      ],
      assistant: true,
      steps: 100,
      batch: 2,
      learningRate: 0.001,
      dropout: 0,
      seed: 7,
      maxMinutes: 0,
      warmStart: false,
    });
    const parsed = JSON.parse(generated.json);
    const code = parsed.cells
      .filter((cell) => cell.cell_type === "code")
      .map(cellText)
      .join("\n");
    assert.match(code, /FAQ_PATH = Path\('\/content\/luau-faq\.md'\)/);
    assert.match(code, /How do you do a for loop in Luau\?/);
    assert.match(code, /file:\/\/\/content\/luau-faq\.md/);
    assert.doesNotMatch(
      code,
      /raw\.githubusercontent\.com\/clinuxrulz\/only-gates-ai\/main\/apps\/only-gates-ai\/src\/assets\/luau-faq\.md/,
    );
  } finally {
    await server.close();
  }
});

test("custom notebook requires at least one URL", async () => {
  const server = await createTestServer();
  try {
    const notebook = await server.ssrLoadModule(
      "/src/lib/customTrainingNotebook.ts",
    );
    assert.throws(
      () =>
        notebook.buildCustomTrainingNotebook({
          urls: ["   "],
          assistant: false,
          steps: 100,
          batch: 2,
          learningRate: 0.001,
          dropout: 0,
          seed: 7,
          maxMinutes: 0,
          warmStart: false,
        }),
      /Add at least one document URL/,
    );
  } finally {
    await server.close();
  }
});
