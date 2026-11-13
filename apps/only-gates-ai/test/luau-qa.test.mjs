import assert from "node:assert/strict";
import test from "node:test";

import {
  LUAU_QA_DOCUMENTS,
  LUAU_QA_SUGGESTIONS,
} from "../src/assets/luau-qa.ts";

const appDirectory = new URL("..", import.meta.url);

test("Luau Q&A corpus has unique concept IDs", () => {
  const ids = LUAU_QA_DOCUMENTS.map((d) => d.id);
  const unique = new Set(ids);
  assert.strictEqual(unique.size, ids.length, "All document IDs must be unique");
});

test("Luau Q&A corpus documents have non-empty questions", () => {
  for (const doc of LUAU_QA_DOCUMENTS) {
    assert.ok(doc.questions.length > 0, `${doc.id} must have at least one question`);
  }
});

test("Luau Q&A corpus documents have non-empty answers", () => {
  for (const doc of LUAU_QA_DOCUMENTS) {
    assert.ok(doc.answer.length > 0, `${doc.id} must have a non-empty answer`);
  }
});

test("Luau Q&A corpus sources start with https://luau.org/", () => {
  for (const doc of LUAU_QA_DOCUMENTS) {
    assert.ok(
      doc.source.startsWith("https://luau.org/"),
      `${doc.id} source must start with https://luau.org/, got: ${doc.source}`,
    );
  }
});

test("Luau Q&A corpus suggestions are non-empty strings", () => {
  for (const suggestion of LUAU_QA_SUGGESTIONS) {
    assert.ok(
      typeof suggestion === "string" && suggestion.length > 0,
      `Each suggestion must be a non-empty string`,
    );
  }
});

test("Every suggestion matches at least one corpus question", () => {
  const allQuestions = LUAU_QA_DOCUMENTS.flatMap((doc) => doc.questions);
  for (const suggestion of LUAU_QA_SUGGESTIONS) {
    const matched = allQuestions.some((q) =>
      q.toLowerCase().includes(suggestion.toLowerCase()),
    );
    assert.ok(
      matched,
      `Suggestion "${suggestion}" must match at least one corpus question`,
    );
  }
});

test("Every concept's first question has a non-empty answer", () => {
  for (const doc of LUAU_QA_DOCUMENTS) {
    const firstQuestion = doc.questions[0];
    assert.ok(
      firstQuestion.length > 0,
      `${doc.id} first question must be non-empty`,
    );
    assert.ok(
      doc.answer.length > 0,
      `${doc.id} answer must be non-empty so the first question can self-retrieve`,
    );
  }
});

test("Luau Q&A corpus covers syntax concepts", () => {
  const ids = LUAU_QA_DOCUMENTS.map((d) => d.id);
  const syntaxRelated = ids.filter(
    (id) =>
      id.includes("continue") ||
      id.includes("compound") ||
      id.includes("const") ||
      id.includes("if-expr") ||
      id.includes("generalized") ||
      id.includes("iteration") ||
      id.includes("string-interp") ||
      id.includes("floor-div") ||
      id.includes("attributes") ||
      id.includes("string-escape") ||
      id.includes("number-literal"),
  );
  assert.ok(syntaxRelated.length > 0, "Corpus must include syntax concepts");
});

test("Luau Q&A corpus covers type system concepts", () => {
  const ids = LUAU_QA_DOCUMENTS.map((d) => d.id);
  const typeRelated = ids.filter(
    (id) =>
      id.includes("gradual") ||
      id.includes("structural") ||
      id.includes("type-annot") ||
      id.includes("type-cast") ||
      id.includes("type-alias") ||
      id.includes("typeof") ||
      id.includes("primitive") ||
      id.includes("any-unknown") ||
      id.includes("singleton") ||
      id.includes("function-type") ||
      id.includes("type-pack") ||
      id.includes("generic") ||
      id.includes("refinement") ||
      id.includes("oop") ||
      id.includes("type-func") ||
      id.includes("require") ||
      id.includes("alias"),
  );
  assert.ok(typeRelated.length > 0, "Corpus must include type system concepts");
});

test("Luau Q&A corpus covers library concepts", () => {
  const ids = LUAU_QA_DOCUMENTS.map((d) => d.id);
  const libraryRelated = ids.filter(
    (id) =>
      id.includes("math-lib") ||
      id.includes("table-lib") ||
      id.includes("string-split") ||
      id.includes("buffer") ||
      id.includes("vector") ||
      id.includes("bit32") ||
      id.includes("utf8") ||
      id.includes("coroutine") ||
      id.includes("os-debug") ||
      id.includes("missing") ||
      id.includes("native") ||
      id.includes("fastcall") ||
      id.includes("table-perf") ||
      id.includes("fenv"),
  );
  assert.ok(libraryRelated.length > 0, "Corpus must include library concepts");
});

test("Luau Q&A corpus covers performance/compatibility concepts", () => {
  const ids = LUAU_QA_DOCUMENTS.map((d) => d.id);
  const perfRelated = ids.filter(
    (id) =>
      id.includes("gc") ||
      id.includes("linter") ||
      id.includes("lint-list") ||
      id.includes("lua-version") ||
      id.includes("lua-diff") ||
      id.includes("impl-limits") ||
      id.includes("sandbox"),
  );
  assert.ok(perfRelated.length > 0, "Corpus must include performance/compatibility concepts");
});

test("Luau Q&A corpus has reasonable question count per document (3-20)", () => {
  for (const doc of LUAU_QA_DOCUMENTS) {
    assert.ok(
      doc.questions.length >= 3 && doc.questions.length <= 20,
      `${doc.id} should have between 3 and 20 questions, got ${doc.questions.length}`,
    );
  }
});

test("Luau Q&A corpus answers reference specific doc pages (have #anchor)", () => {
  for (const doc of LUAU_QA_DOCUMENTS) {
    assert.ok(
      doc.source.includes("#"),
      `${doc.id} source must reference a specific page section with #anchor`,
    );
  }
});

test("Luau Q&A corpus has concepts spanning the expected domains", () => {
  const ids = LUAU_QA_DOCUMENTS.map((d) => d.id);
  const expectedIds = new Set([
    "luau-overview",
    "running-luau",
    "type-check-directives",
    "string-escapes",
    "number-literals",
    "continue-statement",
    "compound-assignment",
    "const-bindings",
    "if-expressions",
    "generalized-iteration",
    "iter-metamethod",
    "string-interpolation",
    "floor-division",
    "attributes",
    "gradual-typing",
    "structural-typing",
    "type-annotations",
    "type-casts",
    "type-aliases",
    "typeof-types",
    "primitive-types",
    "any-unknown-never",
    "singleton-types",
    "function-types",
    "variadic-type-packs",
    "table-types",
    "unions-and-optional",
    "intersection-types",
    "tagged-unions",
    "generics",
    "generic-instantiation",
    "type-refinements",
    "oop-typing",
    "type-functions-and-types-library",
    "require-typing",
    "math-library",
    "table-library",
    "table-freeze-clone",
    "string-split",
    "buffer-library",
    "vector-library",
    "bit32-library",
    "utf8-library",
    "coroutine-library",
    "os-debug-library",
    "missing-libraries",
    "native-codegen",
    "fastcalls-and-imports",
    "table-performance",
    "fenv-loadstring",
    "gc-allocation",
    "linter-overview",
    "lint-list",
    "lua-version-support",
    "lua-differences",
    "implementation-limits",
    "sandboxing",
  ]);
  const missing = [...expectedIds].filter((id) => !ids.includes(id));
  assert.ok(
    missing.length === 0,
    `Missing expected concept IDs: ${missing.join(", ")}`,
  );
});

console.log("All Luau Q&A corpus integrity tests passed!");