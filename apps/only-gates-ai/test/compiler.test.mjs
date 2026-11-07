import assert from "node:assert/strict";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import test from "node:test";

import ts from "typescript";
import { createServer } from "vite";

const appRoot = fileURLToPath(new URL("..", import.meta.url));

async function importModule(code) {
  const compiled = ts.transpileModule(code, {
    compilerOptions: {
      target: ts.ScriptTarget.ES2022,
      module: ts.ModuleKind.ESNext,
    },
    fileName: "qa-model.ts",
  });
  const directory = await mkdtemp(join(tmpdir(), "qa-vite-"));
  const file = join(directory, "qa-model.mjs");
  try {
    await writeFile(file, compiled.outputText);
    return await import(pathToFileURL(file).href);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
}

test("the Vite compiler path emits a working standalone module", async () => {
  const server = await createServer({
    root: appRoot,
    appType: "custom",
    logLevel: "silent",
    server: { middlewareMode: true },
    optimizeDeps: { noDiscovery: true },
  });

  try {
    const compiler = await server.ssrLoadModule("/src/lib/qaCompiler.ts");
    const corpus = await server.ssrLoadModule("/src/assets/solid2-qa.ts");
    const compiled = compiler.compileQuestionAnswerer({
      documents: corpus.SOLID2_QA_DOCUMENTS,
      name: "Solid 2.0 Q&A",
    });

    assert.equal(compiled.code.includes("const LEXICON: QaLexicon"), true);
    assert.equal(compiled.code.includes("function createQuestionAnswerer"), true);
    assert.equal(/^\s*import\s/m.test(compiled.code), false);

    const generated = await importModule(compiled.code);

    const answer = generated.ask("How do split effects work?");
    assert.equal(answer?.matched, true);
    assert.equal(answer?.documentId, "effects-split");
    assert.equal(answer?.confidence, 1);
    assert.equal(generated.STATS.concepts, corpus.SOLID2_QA_DOCUMENTS.length);
  } finally {
    await server.close();
  }
});
