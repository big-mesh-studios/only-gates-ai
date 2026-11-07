/*
 * Portable question-answering runtime.
 *
 * This file is intentionally dependency-free and environment-agnostic. The app
 * uses it directly, and the compiler inlines the whole file into generated
 * TypeScript modules. Keeping one implementation is what makes the generated
 * model byte-for-byte equivalent to the in-app model.
 *
 * QA is retrieval, not generation. A question is tokenized into content words,
 * those words are mapped to 64-bit vectors, and every indexed question is
 * scored with:
 *
 *   confidence = 0.8 * idf-weighted token overlap
 *              + 0.2 * bit similarity
 *
 * The best match above the threshold supplies the answer. There is no learned
 * network to compile here, just a small deterministic scorer.
 */

export const QA_ENGINE_NAME = "qa-retrieval" as const;
export const QA_ENGINE_VERSION = 1 as const;
export const QA_MODEL_FORMAT = "only-gates-qa-model" as const;
export const QA_MODEL_VERSION = 1 as const;

export interface Token {
  readonly text: string;
  readonly start: number;
  readonly end: number;
}

export interface QaDocument {
  readonly id: string;
  readonly questions: readonly string[];
  readonly answer: string;
  readonly source?: string;
  readonly tags?: readonly string[];
  readonly keywords?: readonly string[];
}

export interface QaPair {
  readonly question: string;
  readonly answer: string;
}

export interface QaParseResult {
  readonly pairs: readonly QaPair[];
  readonly errors: readonly string[];
}

export interface QaStats {
  readonly pairs: number;
  readonly concepts: number;
  readonly variants: number;
  readonly questionWords: number;
  readonly vectorBits: number;
  readonly coverage: number;
}

export interface QaMatch {
  readonly question: string;
  readonly confidence: number;
}

export interface QaAnswer extends QaMatch {
  readonly answer: string;
  readonly matched: boolean;
  readonly candidates: readonly QaMatch[];
  readonly documentId: string;
  readonly source?: string;
  readonly tags: readonly string[];
}

export interface QaLexicon {
  readonly bits: number;
  readonly words: readonly string[];
  /** One standard base64 string per word, encoding `bits / 8` little-endian bytes. */
  readonly vectors: readonly string[];
}

export interface QaModelLexiconInfo {
  readonly bits: number;
  readonly words: number;
  readonly fingerprint: string;
}

export interface QaModelInfo {
  readonly format: typeof QA_MODEL_FORMAT;
  readonly version: typeof QA_MODEL_VERSION;
  readonly name: string;
  readonly createdAt?: string;
  readonly updatedAt?: string;
  readonly engine: {
    readonly name: typeof QA_ENGINE_NAME;
    readonly version: number;
  };
  readonly lexicon: QaModelLexiconInfo;
  readonly stats?: QaStats;
}

export interface QaModelFile extends QaModelInfo {
  readonly documents: readonly QaDocument[];
  readonly suggestions?: readonly string[];
}

export interface QaAnswerer {
  readonly stats: QaStats;
  readonly questions: readonly string[];
  readonly suggestions: readonly string[];
  ask(text: string): QaAnswer | null;
}

export interface QaRuntimeConfig {
  readonly documents: readonly QaDocument[];
  readonly suggestions?: readonly string[];
  readonly lexicon: QaLexicon;
}

export interface QaModelLoadOptions {
  /**
   * Loading a model built with a different word-vector lexicon changes the
   * scores. Exact parity is the default, so this must be explicit.
   */
  readonly allowLexiconMismatch?: boolean;
}

export interface QaModelFileOptions {
  readonly name: string;
  readonly lexicon: QaLexicon;
  readonly suggestions?: readonly string[];
  readonly stats?: QaStats;
  readonly createdAt?: string;
  readonly updatedAt?: string;
}

const TOKEN_PATTERN = /[A-Za-z0-9']+/g;

export function tokenize(text: string): Token[] {
  const tokens: Token[] = [];
  TOKEN_PATTERN.lastIndex = 0;
  let match: RegExpExecArray | null;
  while ((match = TOKEN_PATTERN.exec(text)) !== null) {
    const value = match[0]!;
    tokens.push({
      text: value,
      start: match.index,
      end: match.index + value.length,
    });
  }
  return tokens;
}

/*
 * The answers are written with small fenced code blocks. Rendering them as
 * plain text would show the backticks, so split the answer into prose and code
 * segments. This is deliberately not a full Markdown parser.
 */
export interface QaAnswerSegment {
  readonly kind: "text" | "code";
  readonly text: string;
  readonly language?: string;
}

export function splitAnswer(text: string): QaAnswerSegment[] {
  const segments: QaAnswerSegment[] = [];
  const fence = /```([^\n`]*)\n([\s\S]*?)```/g;
  let cursor = 0;
  let match: RegExpExecArray | null;

  while ((match = fence.exec(text)) !== null) {
    const prose = text.slice(cursor, match.index).trim();
    if (prose.length > 0) {
      segments.push({ kind: "text", text: prose });
    }
    const language = (match[1] ?? "").trim();
    segments.push({
      kind: "code",
      text: (match[2] ?? "").replace(/\n$/, ""),
      language,
    });
    cursor = match.index + match[0].length;
  }

  const tail = text.slice(cursor).trim();
  if (tail.length > 0) {
    segments.push({ kind: "text", text: tail });
  }

  return segments.length > 0 ? segments : [{ kind: "text", text }];
}

export function splitInlineCode(text: string): QaAnswerSegment[] {
  const parts: QaAnswerSegment[] = [];
  const code = /`([^`\n]+)`/g;
  let cursor = 0;
  let match: RegExpExecArray | null;

  while ((match = code.exec(text)) !== null) {
    if (match.index > cursor) {
      parts.push({ kind: "text", text: text.slice(cursor, match.index) });
    }
    parts.push({ kind: "code", text: match[1] ?? "" });
    cursor = match.index + match[0].length;
  }
  if (cursor < text.length) {
    parts.push({ kind: "text", text: text.slice(cursor) });
  }

  return parts.length > 0 ? parts : [{ kind: "text", text }];
}

/**
 * Parse Q&A training text. Supported shapes:
 *
 *   question => answer            (one line)
 *   question                      (blank line between pairs)
 *   answer
 *   Q: question                   (Q/A labels also work)
 *   A: answer
 */
export function parseQaPairs(text: string): QaParseResult {
  const pairs: QaPair[] = [];
  const errors: string[] = [];
  const blocks = text
    .replace(/\r\n?/g, "\n")
    .split(/\n\s*\n/)
    .map((block) => block.trim())
    .filter((block) => block.length > 0);

  blocks.forEach((block, index) => {
    const lines = block
      .split("\n")
      .map((line) => line.trim())
      .filter((line) => line.length > 0);
    if (lines.length === 0) {
      return;
    }

    const arrowPattern = /^(.+?)\s*(?:=>|->)\s*(.+)$/;
    const arrows = lines.map((line) => arrowPattern.exec(line));
    if (arrows.every((match) => match !== null)) {
      for (const arrow of arrows) {
        const question = arrow![1]!.trim();
        const answer = arrow![2]!.trim();
        if (question.length === 0 || answer.length === 0) {
          errors.push(`Pair ${index + 1} needs both a question and an answer.`);
          return;
        }
        pairs.push({ question, answer });
      }
      return;
    }

    if (arrows[0] !== null) {
      const question = arrows[0]![1]!.trim();
      const answer = [arrows[0]![2]!.trim(), ...lines.slice(1)]
        .join(" ")
        .trim();
      if (question.length === 0 || answer.length === 0) {
        errors.push(`Pair ${index + 1} needs both a question and an answer.`);
        return;
      }
      pairs.push({ question, answer });
      return;
    }

    const answerIndex = lines.findIndex((line) =>
      /^(?:a|answer)\s*[:.)-]\s*/i.test(line),
    );
    if (answerIndex > 0) {
      const question = stripQaMarker("q", lines.slice(0, answerIndex).join(" "));
      const answer = lines
        .slice(answerIndex)
        .map((line, position) =>
          position === 0 ? stripQaMarker("a", line) : line,
        )
        .join(" ")
        .trim();
      if (question.length === 0 || answer.length === 0) {
        errors.push(`Pair ${index + 1} needs both a question and an answer.`);
        return;
      }
      pairs.push({ question, answer });
      return;
    }

    if (lines.length >= 2) {
      const question = stripQaMarker("q", lines[0]!);
      const answer = lines.slice(1).join(" ").trim();
      if (question.length === 0 || answer.length === 0) {
        errors.push(`Pair ${index + 1} needs both a question and an answer.`);
        return;
      }
      pairs.push({ question, answer });
      return;
    }

    errors.push(
      `Pair ${index + 1}: put the answer on the next line, use "question => answer", or label it with "A:".`,
    );
  });

  return { pairs, errors };
}

function stripQaMarker(kind: "q" | "a", line: string): string {
  const pattern =
    kind === "q"
      ? /^(?:q|question)\s*[:.)-]\s*/i
      : /^(?:a|answer)\s*[:.)-]\s*/i;
  return line.replace(pattern, "").trim();
}

const QA_STOP_WORDS = new Set([
  "a",
  "an",
  "and",
  "are",
  "as",
  "at",
  "be",
  "but",
  "by",
  "can",
  "could",
  "did",
  "do",
  "does",
  "for",
  "from",
  "had",
  "has",
  "have",
  "he",
  "her",
  "his",
  "how",
  "i",
  "if",
  "in",
  "into",
  "is",
  "it",
  "its",
  "me",
  "my",
  "of",
  "on",
  "or",
  "our",
  "she",
  "should",
  "so",
  "that",
  "the",
  "their",
  "them",
  "then",
  "there",
  "these",
  "they",
  "this",
  "those",
  "to",
  "was",
  "we",
  "were",
  "what",
  "when",
  "where",
  "which",
  "who",
  "why",
  "will",
  "with",
  "would",
  "you",
  "your",
  // Solid 2 questions often include these, but they do not help distinguish
  // one concept from another.
  "about",
  "after",
  "available",
  "before",
  "between",
  "change",
  "changed",
  "changes",
  "difference",
  "different",
  "example",
  "examples",
  "explain",
  "happen",
  "happened",
  "happens",
  "mean",
  "meaning",
  "means",
  "need",
  "needs",
  "new",
  "old",
  "replace",
  "replaced",
  "replacement",
  "show",
  "solid",
  "still",
  "tell",
  "use",
  "used",
  "uses",
  "using",
  "version",
  "want",
  "wants",
  "way",
  "ways",
  "work",
  "working",
  "works",
  "2",
  "0",
  "rc",
]);

/*
 * A few 1.x -> 2.0 renames are useful as query expansions. Adding the
 * counterpart to both the query and the indexed question lets "onMount"
 * find "onSettled" without making the whole index fuzzy.
 */
const QA_ALIASES: ReadonlyMap<string, readonly string[]> = new Map([
  ["onmount", ["onsettled"]],
  ["onsettled", ["onmount"]],
  ["createcomputed", ["createeffect", "creatememo"]],
  ["creatememo", ["createcomputed"]],
  ["batch", ["flush"]],
  ["flush", ["batch"]],
  ["unwrap", ["snapshot"]],
  ["snapshot", ["unwrap"]],
  ["mergeprops", ["merge"]],
  ["merge", ["mergeprops"]],
  ["splitprops", ["omit"]],
  ["omit", ["splitprops"]],
  ["suspense", ["loading"]],
  ["loading", ["suspense"]],
  ["errorboundary", ["errored"]],
  ["errored", ["errorboundary"]],
  ["suspenselist", ["reveal"]],
  ["reveal", ["suspenselist"]],
  ["initialvalue", ["prev"]],
  ["prev", ["initialvalue"]],
  ["once", ["untrack"]],
  ["untrack", ["once"]],
  ["createresource", ["creatememo", "fetch"]],
  ["createmutable", ["createstore"]],
  ["produce", ["draft"]],
]);

function singularize(token: string): string {
  if (token.length <= 3) {
    return token;
  }
  if (token.endsWith("ies") && token.length > 4) {
    return `${token.slice(0, -3)}y`;
  }
  if (/(?:ses|xes|zes|ches|shes)$/.test(token)) {
    return token.slice(0, -2);
  }
  if (token.endsWith("s") && !token.endsWith("ss")) {
    return token.slice(0, -1);
  }
  return token;
}

export function contentTokens(text: string): string[] {
  const rawTokens = tokenize(text).map((token) => token.text);
  const expanded: string[] = [];

  for (const raw of rawTokens) {
    const lower = raw.toLowerCase().replace(/^'+|'+$/g, "");
    if (lower.length === 0) {
      continue;
    }

    const candidates = new Set<string>([lower, singularize(lower)]);
    if (/[a-z][A-Z]/.test(raw)) {
      for (const part of raw.split(/(?=[A-Z])/)) {
        const normalized = part.toLowerCase();
        if (normalized.length > 0) {
          candidates.add(normalized);
        }
      }
    }
    if (raw.includes("_")) {
      for (const part of raw.split("_")) {
        const normalized = part.toLowerCase();
        if (normalized.length > 0) {
          candidates.add(normalized);
        }
      }
    }

    for (const candidate of candidates) {
      if (QA_STOP_WORDS.has(candidate)) {
        continue;
      }
      expanded.push(candidate);
      for (const alias of QA_ALIASES.get(candidate) ?? []) {
        expanded.push(alias);
      }
    }
  }

  if (expanded.length > 0) {
    return expanded;
  }
  return rawTokens.map((token) => token.toLowerCase()).filter(Boolean);
}

function popcount32(value: number): number {
  let x = value >>> 0;
  x -= (x >>> 1) & 0x55555555;
  x = (x & 0x33333333) + ((x >>> 2) & 0x33333333);
  x = (x + (x >>> 4)) & 0x0f0f0f0f;
  return Math.imul(x, 0x01010101) >>> 24;
}

const BASE64_ALPHABET =
  "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
const BASE64_LOOKUP = (() => {
  const table = new Int16Array(128);
  table.fill(-1);
  for (let index = 0; index < BASE64_ALPHABET.length; index++) {
    table[BASE64_ALPHABET.charCodeAt(index)] = index;
  }
  return table;
})();

function decodeBase64(value: string): Uint8Array {
  let padding = 0;
  while (
    padding < 2 &&
    value.length > padding &&
    value.charCodeAt(value.length - 1 - padding) === 61
  ) {
    padding++;
  }

  const byteLength = Math.floor((value.length * 3) / 4) - padding;
  const bytes = new Uint8Array(byteLength);
  let buffer = 0;
  let bits = 0;
  let offset = 0;

  for (let index = 0; index < value.length - padding; index++) {
    const code = value.charCodeAt(index);
    const digit = code < 128 ? BASE64_LOOKUP[code]! : -1;
    if (digit < 0) {
      throw new Error("The word-vector lexicon contains invalid base64 data.");
    }
    buffer = (buffer << 6) | digit;
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      bytes[offset++] = (buffer >>> bits) & 0xff;
    }
  }

  if (offset !== byteLength) {
    throw new Error("The word-vector lexicon contains invalid base64 data.");
  }
  return bytes;
}

function decodeVector(encoded: string, vectorWords: number): Uint32Array {
  const bytes = decodeBase64(encoded);
  const expected = vectorWords * 4;
  if (bytes.length !== expected) {
    throw new Error(
      `A word vector has ${bytes.length} bytes, but the lexicon needs ${expected}.`,
    );
  }

  const vector = new Uint32Array(vectorWords);
  for (let index = 0; index < bytes.length; index++) {
    vector[index >> 2] =
      vector[index >> 2]! | (bytes[index]! << ((index & 3) * 8));
  }
  return vector;
}

function hashVector(word: string, vectorWords: number): Uint32Array {
  const vector = new Uint32Array(vectorWords);
  let hash = 0x811c9dc5;
  for (let wordIndex = 0; wordIndex < vectorWords; wordIndex++) {
    for (let index = 0; index < word.length; index++) {
      hash ^= word.charCodeAt(index);
      hash = Math.imul(hash, 0x01000193);
    }
    hash ^= 0x9e3779b9 + wordIndex;
    hash = Math.imul(hash, 0x85ebca6b);
    vector[wordIndex] = hash >>> 0;
  }
  if (vector.every((value) => value === 0)) {
    vector[0] = 1;
  }
  return vector;
}

function majorityVector(
  vectors: readonly Uint32Array[],
  vectorWords: number,
): Uint32Array {
  const out = new Uint32Array(vectorWords);
  if (vectors.length === 0) {
    return out;
  }
  if (vectors.length === 1) {
    out.set(vectors[0]!);
    return out;
  }

  const threshold = vectors.length / 2;
  const latest = vectors[vectors.length - 1]!;
  for (let word = 0; word < vectorWords; word++) {
    let result = 0;
    for (let bit = 0; bit < 32; bit++) {
      let ones = 0;
      for (const vector of vectors) {
        if (((vector[word]! >>> bit) & 1) === 1) {
          ones++;
        }
      }
      const latestBit = (latest[word]! >>> bit) & 1;
      if (ones > threshold || (ones === threshold && latestBit === 1)) {
        result |= 1 << bit;
      }
    }
    out[word] = result >>> 0;
  }
  return out;
}

function scoreKey(
  query: Uint32Array,
  key: Uint32Array,
  vectorWords: number,
): number {
  let matches = 0;
  for (let word = 0; word < vectorWords; word++) {
    matches += popcount32(~(query[word]! ^ key[word]!));
  }
  return 2 * matches - vectorWords * 32;
}

interface PreparedLexicon {
  readonly bits: number;
  readonly vectorWords: number;
  readonly index: ReadonlyMap<string, Uint32Array>;
}

const lexiconCache = new WeakMap<QaLexicon, PreparedLexicon>();

function prepareLexicon(lexicon: QaLexicon): PreparedLexicon {
  const cached = lexiconCache.get(lexicon);
  if (cached !== undefined) {
    return cached;
  }

  if (
    !Number.isSafeInteger(lexicon.bits) ||
    lexicon.bits <= 0 ||
    lexicon.bits % 32 !== 0
  ) {
    throw new Error("The word-vector lexicon needs a positive bit width that is a multiple of 32.");
  }
  if (lexicon.words.length !== lexicon.vectors.length) {
    throw new Error(
      `The word-vector lexicon has ${lexicon.words.length} words but ${lexicon.vectors.length} vectors.`,
    );
  }

  const vectorWords = lexicon.bits / 32;
  const index = new Map<string, Uint32Array>();
  for (let position = 0; position < lexicon.words.length; position++) {
    const word = lexicon.words[position]!;
    if (typeof word !== "string" || word.length === 0) {
      throw new Error(`Word ${position + 1} in the lexicon is empty.`);
    }
    index.set(
      word.toLowerCase(),
      decodeVector(lexicon.vectors[position]!, vectorWords),
    );
  }

  const prepared: PreparedLexicon = {
    bits: lexicon.bits,
    vectorWords,
    index,
  };
  lexiconCache.set(lexicon, prepared);
  return prepared;
}

function getWordVector(word: string, prepared: PreparedLexicon): Uint32Array {
  const lower = word.toLowerCase();
  const bundled = prepared.index.get(lower);
  return bundled ?? hashVector(lower, prepared.vectorWords);
}

function questionVector(
  tokens: readonly string[],
  prepared: PreparedLexicon,
): Uint32Array {
  return majorityVector(
    tokens.map((token) => getWordVector(token, prepared)),
    prepared.vectorWords,
  );
}

function weightedOverlap(
  query: ReadonlySet<string>,
  entry: ReadonlySet<string>,
  idf: ReadonlyMap<string, number>,
): number {
  let shared = 0;
  let union = 0;
  for (const token of query) {
    const weight = idf.get(token) ?? 1;
    union += weight;
    if (entry.has(token)) {
      shared += weight;
    }
  }
  for (const token of entry) {
    if (!query.has(token)) {
      union += idf.get(token) ?? 1;
    }
  }
  return union === 0 ? 0 : shared / union;
}

interface QaEntry {
  readonly question: string;
  readonly answer: string;
  readonly tokens: ReadonlySet<string>;
  readonly vector: Uint32Array;
  readonly documentId: string;
  readonly source?: string;
  readonly tags: readonly string[];
}

interface PreparedDocument {
  readonly document: QaDocument;
  readonly variantTokens: readonly (readonly string[])[];
  readonly union: ReadonlySet<string>;
}

/**
 * Builds a self-contained answerer from source documents and a word-vector
 * lexicon. Everything the answerer needs is derived here, so a generated
 * module only has to embed the documents, the lexicon, and this runtime.
 */
export function createQuestionAnswerer(config: QaRuntimeConfig): QaAnswerer {
  if (config.documents.length === 0) {
    throw new Error("Add at least one question and answer pair.");
  }

  const prepared = prepareLexicon(config.lexicon);
  const preparedDocuments: PreparedDocument[] = config.documents.map(
    (document) => {
      if (document.questions.length === 0) {
        throw new Error(`"${document.id}" needs at least one question.`);
      }
      const variantTokens = document.questions.map((question) =>
        contentTokens(question),
      );
      const keywordTokens = (document.keywords ?? []).flatMap((keyword) =>
        contentTokens(keyword),
      );
      const union = new Set<string>();
      for (const tokens of variantTokens) {
        for (const token of tokens) {
          union.add(token);
        }
      }
      for (const token of keywordTokens) {
        union.add(token);
      }
      return { document, variantTokens, union };
    },
  );

  const documentFrequency = new Map<string, number>();
  for (const { union } of preparedDocuments) {
    for (const token of union) {
      documentFrequency.set(token, (documentFrequency.get(token) ?? 0) + 1);
    }
  }

  const total = preparedDocuments.length;
  const idf = new Map<string, number>();
  for (const [token, count] of documentFrequency) {
    idf.set(token, Math.log((total + 1) / (count + 1)) + 1);
  }

  let questionWords = 0;
  let bundledWords = 0;
  const entries: QaEntry[] = [];

  for (const { document, variantTokens } of preparedDocuments) {
    document.questions.forEach((question, index) => {
      const tokens = variantTokens[index] ?? [];
      if (tokens.length === 0) {
        return;
      }
      questionWords += tokens.length;
      for (const token of tokens) {
        if (prepared.index.has(token)) {
          bundledWords++;
        }
      }
      entries.push({
        question,
        answer: document.answer,
        tokens: new Set(tokens),
        vector: questionVector([...new Set(tokens)], prepared),
        documentId: document.id,
        source: document.source,
        tags: document.tags ?? [],
      });
    });
  }

  if (entries.length === 0) {
    throw new Error("Add at least one question and answer pair.");
  }

  const suggestions =
    config.suggestions ??
    config.documents
      .map((document) => document.questions[0])
      .filter(
        (question): question is string =>
          question !== undefined && question.length > 0,
      );

  const stats: QaStats = {
    pairs: entries.length,
    concepts: config.documents.length,
    variants: entries.length,
    questionWords,
    vectorBits: prepared.bits,
    coverage: questionWords === 0 ? 0 : bundledWords / questionWords,
  };

  function ask(text: string): QaAnswer | null {
    const tokens = contentTokens(text);
    if (tokens.length === 0 || entries.length === 0) {
      return null;
    }

    const unique = new Set(tokens);
    const queryVector = questionVector([...unique], prepared);
    const scored = entries
      .map((entry, index) => {
        const overlap = weightedOverlap(unique, entry.tokens, idf);
        const matches =
          (scoreKey(queryVector, entry.vector, prepared.vectorWords) +
            prepared.bits) /
          2;
        const similarity = matches / prepared.bits;
        return {
          index,
          question: entry.question,
          confidence: 0.8 * overlap + 0.2 * similarity,
        };
      })
      .sort((a, b) => b.confidence - a.confidence);

    const best = scored[0]!;
    const entry = entries[best.index]!;
    let shared = 0;
    for (const token of unique) {
      if (entry.tokens.has(token)) {
        shared++;
      }
    }
    const matched =
      best.confidence >= 0.34 && (shared > 0 || best.confidence >= 0.72);

    return {
      answer: entry.answer,
      question: entry.question,
      confidence: best.confidence,
      matched,
      candidates: scored.slice(0, 3).map((candidate) => ({
        question: candidate.question,
        confidence: candidate.confidence,
      })),
      documentId: entry.documentId,
      source: entry.source,
      tags: entry.tags,
    };
  }

  return {
    stats,
    questions: entries.map((entry) => entry.question),
    suggestions,
    ask,
  };
}

/** Stable FNV-1a fingerprint of the word list and vector data. */
export function qaLexiconFingerprint(lexicon: QaLexicon): string {
  let hash = 0x811c9dc5;
  const mix = (value: string): void => {
    for (let index = 0; index < value.length; index++) {
      hash ^= value.charCodeAt(index);
      hash = Math.imul(hash, 0x01000193);
    }
  };

  mix(`bits:${lexicon.bits}\n`);
  mix(`words:${lexicon.words.length}\n`);
  for (let index = 0; index < lexicon.words.length; index++) {
    mix(lexicon.words[index]!);
    mix(":");
    mix(lexicon.vectors[index]!);
    mix("\n");
  }
  return `fnv1a-${(hash >>> 0).toString(16).padStart(8, "0")}`;
}

export function buildQaModelInfo(options: QaModelFileOptions): QaModelInfo {
  return {
    format: QA_MODEL_FORMAT,
    version: QA_MODEL_VERSION,
    name: options.name,
    createdAt: options.createdAt,
    updatedAt: options.updatedAt,
    engine: {
      name: QA_ENGINE_NAME,
      version: QA_ENGINE_VERSION,
    },
    lexicon: {
      bits: options.lexicon.bits,
      words: options.lexicon.words.length,
      fingerprint: qaLexiconFingerprint(options.lexicon),
    },
    stats: options.stats,
  };
}

export function buildQaModelFile(
  documents: readonly QaDocument[],
  options: QaModelFileOptions,
): QaModelFile {
  return {
    ...buildQaModelInfo(options),
    documents: [...documents],
    suggestions:
      options.suggestions === undefined ? undefined : [...options.suggestions],
  };
}

/**
 * Loads a model file with a specific lexicon. Generated modules call this with
 * their embedded lexicon; the app calls it with the bundled lexicon.
 */
export function createAnswererFromModelWithLexicon(
  model: QaModelFile,
  lexicon: QaLexicon,
  options: QaModelLoadOptions = {},
): QaAnswerer {
  if (model === null || typeof model !== "object") {
    throw new Error("The QA model must be an object.");
  }
  if (model.format !== QA_MODEL_FORMAT) {
    throw new Error("This is not a Gate Forest QA model.");
  }
  if (model.version !== QA_MODEL_VERSION) {
    throw new Error(
      `Unsupported QA model version: ${String(model.version)}. Recompile it with this app version.`,
    );
  }
  if (
    model.engine?.name !== QA_ENGINE_NAME ||
    model.engine.version !== QA_ENGINE_VERSION
  ) {
    throw new Error(
      "This QA model was built with a different retrieval engine version.",
    );
  }
  if (!Array.isArray(model.documents)) {
    throw new Error("The QA model has no documents.");
  }

  if (!options.allowLexiconMismatch) {
    const fingerprint = qaLexiconFingerprint(lexicon);
    if (
      model.lexicon?.bits !== lexicon.bits ||
      model.lexicon.words !== lexicon.words.length ||
      model.lexicon.fingerprint !== fingerprint
    ) {
      throw new Error(
        "This QA model was built with a different word-vector lexicon. Recompile it with this app version, or pass { allowLexiconMismatch: true } to load it anyway.",
      );
    }
  }

  return createQuestionAnswerer({
    documents: model.documents,
    suggestions: model.suggestions,
    lexicon,
  });
}
