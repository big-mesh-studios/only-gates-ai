import {
  buildQaModelFile,
  QA_ENGINE_NAME,
  QA_ENGINE_VERSION,
  QA_MODEL_FORMAT,
  QA_MODEL_VERSION,
  type QaDocument,
  type QaModelFile,
  type QaModelFileOptions,
} from "./qaRuntime.ts";

const MAX_DOCUMENTS = 2000;
const MAX_QUESTIONS_PER_DOCUMENT = 500;
const MAX_QUESTION_LENGTH = 2000;
const MAX_ANSWER_LENGTH = 100_000;
const MAX_NAME_LENGTH = 200;
const MAX_TAGS = 100;
const MAX_KEYWORDS = 100;

export function serializeQaModelFile(
  documents: readonly QaDocument[],
  options: QaModelFileOptions,
): QaModelFile {
  return buildQaModelFile(documents, options);
}

function asRecord(value: unknown, context: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error(`${context} must be an object.`);
  }
  return value as Record<string, unknown>;
}

function asArray(value: unknown, context: string): readonly unknown[] {
  if (!Array.isArray(value)) {
    throw new Error(`${context} must be an array.`);
  }
  return value;
}

function asString(value: unknown, context: string): string {
  if (typeof value !== "string") {
    throw new Error(`${context} must be text.`);
  }
  return value;
}

function asNonEmptyString(value: unknown, context: string): string {
  const text = asString(value, context).trim();
  if (text.length === 0) {
    throw new Error(`${context} cannot be empty.`);
  }
  return text;
}

function asOptionalString(
  value: unknown,
  context: string,
): string | undefined {
  if (value === undefined) {
    return undefined;
  }
  return asString(value, context);
}

function asPositiveInteger(value: unknown, context: string): number {
  if (typeof value !== "number" || !Number.isInteger(value) || value <= 0) {
    throw new Error(`${context} must be a positive whole number.`);
  }
  return value;
}

function asNonNegativeInteger(value: unknown, context: string): number {
  if (typeof value !== "number" || !Number.isInteger(value) || value < 0) {
    throw new Error(`${context} must be a non-negative whole number.`);
  }
  return value;
}

function asStringArray(
  value: unknown,
  context: string,
  limit: number,
  maxLength: number,
): string[] {
  const entries = asArray(value, context);
  if (entries.length > limit) {
    throw new Error(`${context} has ${entries.length} entries; the limit is ${limit}.`);
  }
  return entries.map((entry, index) => {
    const text = asNonEmptyString(entry, `${context} entry ${index + 1}`);
    if (text.length > maxLength) {
      throw new Error(
        `${context} entry ${index + 1} is longer than ${maxLength} characters.`,
      );
    }
    return text;
  });
}

function validateSource(value: unknown, context: string): string | undefined {
  const text = asOptionalString(value, context);
  if (text === undefined || text.trim().length === 0) {
    return undefined;
  }
  let url: URL;
  try {
    url = new URL(text.trim());
  } catch {
    throw new Error(`${context} must be a valid URL.`);
  }
  if (url.protocol !== "http:" && url.protocol !== "https:") {
    throw new Error(`${context} must use http or https.`);
  }
  return url.toString();
}

function parseDocument(value: unknown, index: number): QaDocument {
  const context = `Concept ${index + 1}`;
  const record = asRecord(value, context);
  const id = asNonEmptyString(record.id, `${context} id`);
  if (id.length > 200) {
    throw new Error(`${context} id is longer than 200 characters.`);
  }

  const questions = asStringArray(
    record.questions,
    `${context} questions`,
    MAX_QUESTIONS_PER_DOCUMENT,
    MAX_QUESTION_LENGTH,
  );
  if (questions.length === 0) {
    throw new Error(`${context} needs at least one question.`);
  }

  const answer = asNonEmptyString(record.answer, `${context} answer`);
  if (answer.length > MAX_ANSWER_LENGTH) {
    throw new Error(
      `${context} answer is longer than ${MAX_ANSWER_LENGTH} characters.`,
    );
  }

  const source = validateSource(record.source, `${context} source`);
  const tags =
    record.tags === undefined
      ? undefined
      : asStringArray(record.tags, `${context} tags`, MAX_TAGS, 100);
  const keywords =
    record.keywords === undefined
      ? undefined
      : asStringArray(
          record.keywords,
          `${context} keywords`,
          MAX_KEYWORDS,
          100,
        );

  return {
    id,
    questions,
    answer,
    source,
    tags,
    keywords,
  };
}

/**
 * Validates a QA model file imported from a user. The model is deliberately
 * small: documents plus metadata. Vectors, IDF weights, and entry indexes are
 * rebuilt deterministically by the engine.
 */
export function parseQaModelFile(value: unknown): QaModelFile {
  const record = asRecord(value, "The QA model");
  if (record.format !== QA_MODEL_FORMAT) {
    throw new Error("This is not a Gate Forest QA model.");
  }
  if (record.version !== QA_MODEL_VERSION) {
    throw new Error(
      `Unsupported QA model version: ${String(record.version)}. Recompile it with this app version.`,
    );
  }

  const name = asNonEmptyString(record.name, "The model name");
  if (name.length > MAX_NAME_LENGTH) {
    throw new Error(
      `The model name is longer than ${MAX_NAME_LENGTH} characters.`,
    );
  }

  const engine = asRecord(record.engine, "The engine");
  const engineName = asNonEmptyString(engine.name, "The engine name");
  if (engineName !== QA_ENGINE_NAME) {
    throw new Error(`Unsupported QA engine: ${engineName}.`);
  }
  const engineVersion = asPositiveInteger(engine.version, "The engine version");
  if (engineVersion !== QA_ENGINE_VERSION) {
    throw new Error(
      `Unsupported QA engine version: ${String(engineVersion)}. Recompile the model with this app version.`,
    );
  }

  const lexicon = asRecord(record.lexicon, "The lexicon");
  const bits = asPositiveInteger(lexicon.bits, "The lexicon bit width");
  if (bits % 32 !== 0) {
    throw new Error("The lexicon bit width must be a multiple of 32.");
  }
  const words = asNonNegativeInteger(lexicon.words, "The lexicon word count");
  const fingerprint = asNonEmptyString(
    lexicon.fingerprint,
    "The lexicon fingerprint",
  );

  const documents = asArray(record.documents, "The model documents");
  if (documents.length === 0) {
    throw new Error("The model has no concepts.");
  }
  if (documents.length > MAX_DOCUMENTS) {
    throw new Error(
      `The model has ${documents.length} concepts; the limit is ${MAX_DOCUMENTS}.`,
    );
  }

  const parsedDocuments = documents.map((document, index) =>
    parseDocument(document, index),
  );
  const seenIds = new Set<string>();
  for (const document of parsedDocuments) {
    if (seenIds.has(document.id)) {
      throw new Error(`The model has duplicate concept id "${document.id}".`);
    }
    seenIds.add(document.id);
  }

  const suggestions =
    record.suggestions === undefined
      ? undefined
      : asStringArray(
          record.suggestions,
          "The model suggestions",
          MAX_DOCUMENTS,
          MAX_QUESTION_LENGTH,
        );

  return {
    format: QA_MODEL_FORMAT,
    version: QA_MODEL_VERSION,
    name,
    createdAt: asOptionalString(record.createdAt, "The created time"),
    updatedAt: asOptionalString(record.updatedAt, "The updated time"),
    engine: {
      name: QA_ENGINE_NAME,
      version: QA_ENGINE_VERSION,
    },
    lexicon: {
      bits,
      words,
      fingerprint,
    },
    documents: parsedDocuments,
    suggestions,
  };
}
