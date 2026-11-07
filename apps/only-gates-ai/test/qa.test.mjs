import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import test from "node:test";

import ts from "typescript";

import {
  SOLID2_QA_DOCUMENTS,
  SOLID2_QA_SUGGESTIONS,
} from "../src/assets/solid2-qa.ts";
import {
  BUNDLED_QA_LEXICON,
  bundledQaLexiconFingerprint,
} from "../src/lib/qaAssets.ts";
import { compileQuestionAnswererCode } from "../src/lib/qaCodegen.ts";
import { parseQaModelFile, serializeQaModelFile } from "../src/lib/qaModel.ts";
import {
  createAnswererFromModelWithLexicon,
  createQuestionAnswerer,
} from "../src/lib/qaRuntime.ts";

const appDirectory = new URL("..", import.meta.url);
const fixture = JSON.parse(
  await readFile(
    new URL("test/fixtures/solid2-qa-results.json", appDirectory),
    "utf8",
  ),
);
const runtimeSource = await readFile(
  new URL("src/lib/qaRuntime.ts", appDirectory),
  "utf8",
);

function appAnswerer() {
  return createQuestionAnswerer({
    documents: SOLID2_QA_DOCUMENTS,
    lexicon: BUNDLED_QA_LEXICON,
  });
}

async function importGeneratedModule(code) {
  const compiled = ts.transpileModule(code, {
    compilerOptions: {
      target: ts.ScriptTarget.ES2022,
      module: ts.ModuleKind.ESNext,
    },
    fileName: "qa-model.ts",
  });
  const directory = await mkdtemp(join(tmpdir(), "qa-model-"));
  const file = join(directory, "qa-model.mjs");
  try {
    await writeFile(file, compiled.outputText);
    return await import(pathToFileURL(file).href);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
}

test("the shared engine reproduces the golden Solid 2.0 results", () => {
  const answerer = appAnswerer();

  assert.deepEqual(answerer.stats, fixture.stats);
  assert.deepEqual(answerer.suggestions, fixture.suggestions);
  assert.equal(answerer.questions.length, fixture.variants);

  for (const { query, result } of fixture.results) {
    assert.deepEqual(answerer.ask(query), result, `query: ${query}`);
  }
});

test("generated TypeScript matches the in-app answerer exactly", async () => {
  const answerer = appAnswerer();
  const compiled = compileQuestionAnswererCode({
    documents: SOLID2_QA_DOCUMENTS,
    suggestions: answerer.suggestions,
    name: "Solid 2.0 Q&A",
    lexicon: BUNDLED_QA_LEXICON,
    runtimeSource,
  });
  const generated = await importGeneratedModule(compiled.code);

  assert.deepEqual(generated.STATS, answerer.stats);
  assert.deepEqual(generated.QUESTIONS, answerer.questions);
  assert.deepEqual(generated.SUGGESTIONS, answerer.suggestions);
  assert.equal(compiled.lexiconWordCount, BUNDLED_QA_LEXICON.words.length);
  assert.equal(/^\s*import\s/m.test(compiled.code), false);
  assert.equal(
    /\bwindow\b|\bindexedDB\b|\bHTMLDialogElement\b/.test(runtimeSource),
    false,
  );
  assert.deepEqual(generated.MODEL.lexicon, {
    bits: BUNDLED_QA_LEXICON.bits,
    words: BUNDLED_QA_LEXICON.words.length,
    fingerprint: bundledQaLexiconFingerprint(),
  });

  for (const { query, result } of fixture.results) {
    assert.deepEqual(generated.ask(query), result, `query: ${query}`);
  }
});

test("the generated module typechecks under strict settings", async () => {
  const compiled = compileQuestionAnswererCode({
    documents: SOLID2_QA_DOCUMENTS,
    suggestions: SOLID2_QA_SUGGESTIONS,
    name: "Solid 2.0 Q&A",
    lexicon: BUNDLED_QA_LEXICON,
    runtimeSource,
  });
  const directory = await mkdtemp(join(tmpdir(), "qa-typecheck-"));
  const file = join(directory, "qa-model.ts");
  try {
    await writeFile(file, compiled.code);
    const program = ts.createProgram([file], {
      target: ts.ScriptTarget.ES2022,
      module: ts.ModuleKind.ESNext,
      moduleResolution: ts.ModuleResolutionKind.Bundler,
      lib: ["lib.es2022.d.ts"],
      strict: true,
      noUncheckedIndexedAccess: true,
      noEmit: true,
      skipLibCheck: true,
    });
    const diagnostics = ts.getPreEmitDiagnostics(program);
    assert.deepEqual(
      diagnostics.map((diagnostic) =>
        ts.flattenDiagnosticMessageText(diagnostic.messageText, "\n"),
      ),
      [],
    );
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test("model JSON round-trips through validation and rebuilds the model", () => {
  const answerer = appAnswerer();
  const model = serializeQaModelFile(SOLID2_QA_DOCUMENTS, {
    name: "Solid 2.0 Q&A",
    suggestions: SOLID2_QA_SUGGESTIONS,
    lexicon: BUNDLED_QA_LEXICON,
    stats: answerer.stats,
    createdAt: "2026-01-01T00:00:00.000Z",
    updatedAt: "2026-01-02T00:00:00.000Z",
  });

  assert.match(model.lexicon.fingerprint, /^fnv1a-[0-9a-f]{8}$/);
  assert.equal(model.lexicon.words, BUNDLED_QA_LEXICON.words.length);

  const parsed = parseQaModelFile(JSON.parse(JSON.stringify(model)));
  assert.deepEqual(parsed.documents, SOLID2_QA_DOCUMENTS);
  assert.equal(parsed.name, "Solid 2.0 Q&A");
  assert.equal(parsed.createdAt, "2026-01-01T00:00:00.000Z");

  const rebuilt = createAnswererFromModelWithLexicon(parsed, BUNDLED_QA_LEXICON);
  assert.deepEqual(rebuilt.stats, answerer.stats);
  assert.deepEqual(rebuilt.ask("How do split effects work?"), answerer.ask("How do split effects work?"));
});

test("a model with a different lexicon fingerprint is rejected by default", () => {
  const model = serializeQaModelFile(SOLID2_QA_DOCUMENTS, {
    name: "Solid 2.0 Q&A",
    lexicon: BUNDLED_QA_LEXICON,
  });
  const mismatched = {
    ...model,
    lexicon: { ...model.lexicon, fingerprint: "fnv1a-00000000" },
  };

  assert.throws(
    () => createAnswererFromModelWithLexicon(mismatched, BUNDLED_QA_LEXICON),
    /different word-vector lexicon/,
  );
});

test("createAnswerer extends a compiled model at runtime", async () => {
  const compiled = compileQuestionAnswererCode({
    documents: SOLID2_QA_DOCUMENTS,
    suggestions: SOLID2_QA_SUGGESTIONS,
    name: "Solid 2.0 Q&A",
    lexicon: BUNDLED_QA_LEXICON,
    runtimeSource,
  });
  const generated = await importGeneratedModule(compiled.code);

  const extra = {
    id: "custom-concept",
    questions: [
      "What is the zygomorphic flux capacitor?",
      "How does the zygomorphic flux capacitor work?",
    ],
    answer: "It stores a made-up answer for the runtime extension test.",
    tags: ["test"],
  };
  const extended = generated.createAnswerer([
    ...generated.DOCUMENTS,
    extra,
  ]);

  assert.equal(extended.stats.concepts, generated.DOCUMENTS.length + 1);
  const answer = extended.ask("zygomorphic flux capacitor");
  assert.equal(answer?.answer, extra.answer);
  assert.equal(answer?.matched, true);
  assert.equal(answer?.documentId, "custom-concept");

  assert.equal(
    extended.ask("How do split effects work?")?.answer,
    generated.ask("How do split effects work?")?.answer,
  );
});

test("generated data escapes HTML and line separators", async () => {
  const documents = [
    {
      id: "escaping",
      questions: ["How do I escape </script> and separators?"],
      answer: "Literal: </script> \u2028 \u2029 and `backticks`.",
    },
  ];
  const compiled = compileQuestionAnswererCode({
    documents,
    name: "Escaping",
    lexicon: BUNDLED_QA_LEXICON,
    runtimeSource,
  });

  assert.equal(compiled.code.includes("</script>"), false);
  assert.equal(compiled.code.includes("\u2028"), false);
  assert.equal(compiled.code.includes("\u2029"), false);

  const generated = await importGeneratedModule(compiled.code);
  assert.equal(generated.DOCUMENTS[0].answer, documents[0].answer);
  assert.equal(generated.DOCUMENTS[0].questions[0], documents[0].questions[0]);
});
