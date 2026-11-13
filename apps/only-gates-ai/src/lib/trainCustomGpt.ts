import { createTokenStreamCache } from "./tokenStreamCache";
import { SOLID2_MODEL_BASE, SOLID2_TRAIN_CONFIG, solid2ParameterCount } from "./solid2Config";
import {
  BPE_SPECIAL,
  loadBpeTokenizer,
  type BpeTokenizer,
} from "./tinyGptBpe";
import {
  createSolid2WasmFromData,
  type Solid2Checkpoint,
  type Solid2WasmModel,
} from "./tinyGptSolid2";
import { packTinyGptWeights } from "./tinyGptWasm";
import { loadBundledSolid2Model } from "./trainSolid2Gpt";
import {
  hashString,
  trainTokenGpt,
  yieldToBrowser,
  type BrowserTrainingBackend,
  type BrowserTrainingProgress,
  type BrowserTrainingResult,
  type BrowserTrainingState,
  type TokenTrainingData,
} from "./trainTinyGpt";

/*
 * Browser-side training on a user-supplied list of Markdown documents.
 *
 * The model uses the deployed Solid 2 architecture and BPE tokenizer, so the
 * integer-only WASM engine can load the result unchanged. Only the corpus is
 * different: every URL is fetched, Markdown is used directly, HTML is reduced
 * to readable text, and the result is tokenized with the same 4096-token
 * vocabulary.
 */

const tokenCache = createTokenStreamCache("gate-forest-custom-training-data");
const TOKEN_SCHEME_VERSION = 3;
const VALIDATION_FRACTION = 0.1;
const MAX_PROMPTS = 5;
const QA_REPEATS = 3;
const QA_VALIDATION_EVERY = 10;

export type CustomDocKind = "markdown" | "html";

export interface CustomDoc {
  readonly url: string;
  readonly text: string;
  readonly kind?: CustomDocKind;
}

export interface CustomTrainingDataInfo {
  readonly documentCount: number;
  readonly documentChars: number;
  readonly htmlDocuments: number;
  readonly paragraphs: number;
  readonly trainingTokens: number;
  readonly validationTokens: number;
  readonly tokenizerVocab: number;
  readonly cached: boolean;
  readonly warmStart: boolean;
  /** True when the corpus was converted into question/answer pairs. */
  readonly assistant: boolean;
  readonly qaPairs: number;
}

export interface CustomTrainingData {
  readonly tokenizer: BpeTokenizer;
  readonly trainingData: TokenTrainingData;
  readonly prompts: readonly string[];
  readonly info: CustomTrainingDataInfo;
}

export interface CustomDataOptions {
  readonly urls: readonly string[];
  readonly tokenizerUrl?: string;
  /** Base URL for the deployed tokenizer. Defaults to `SOLID2_MODEL_BASE`. */
  readonly modelBaseUrl?: string;
  /**
   * Builds question/answer pairs from Markdown headings and trains with the
   * chat special tokens instead of treating the docs as plain text.
   */
  readonly assistant?: boolean;
  readonly onStatus?: (status: string) => void;
  readonly shouldCancel?: () => boolean;
}

export interface CustomTrainingOptions extends CustomDataOptions {
  readonly steps?: number;
  readonly batch?: number;
  readonly lr?: number;
  readonly seed?: number;
  readonly dropout?: number;
  readonly maxMinutes?: number;
  readonly backend?: "auto" | BrowserTrainingBackend;
  readonly initialState?: BrowserTrainingState | null;
  /**
   * Initializes from the bundled Solid 2 checkpoint instead of random weights.
   * The architecture and tokenizer are identical; only the corpus changes.
   * Ignored when resuming a saved run (`initialState` wins).
   */
  readonly warmStart?: boolean;
  readonly onProgress?: (progress: BrowserTrainingProgress) => void;
  readonly onCheckpoint?: (state: BrowserTrainingState) => void;
  readonly yieldEvery?: number;
  readonly saveEvery?: number;
}

export interface CustomTrainingResult {
  readonly result: BrowserTrainingResult;
  readonly checkpoint: Solid2Checkpoint;
  readonly tokenizer: BpeTokenizer;
  /** Packed in `tiny_gpt_create` order, ready for the WASM quantizer. */
  readonly weights: Float32Array;
  readonly data: CustomTrainingDataInfo;
  readonly prompts: readonly string[];
}

/* ------------------------------------------------------------------ */
/* URL handling                                                       */
/* ------------------------------------------------------------------ */

/**
 * Normalizes GitHub "blob" links to their raw form. Raw hosts already serve
 * plain text with CORS enabled, which the browser fetch needs.
 */
function normalizeDocUrl(value: string): string {
  let url: URL;
  try {
    url = new URL(value);
  } catch {
    throw new Error(`Not a valid URL: ${value}`);
  }
  if (url.protocol !== "http:" && url.protocol !== "https:") {
    throw new Error(`Only http(s) URLs are supported: ${value}`);
  }
  if (url.hostname === "github.com" || url.hostname === "www.github.com") {
    const parts = url.pathname.split("/").filter(Boolean);
    if (parts.length >= 5 && parts[2] === "blob") {
      const [owner, repo, , ...rest] = parts;
      return `https://raw.githubusercontent.com/${owner}/${repo}/${rest.join("/")}`;
    }
  }
  return url.toString();
}

/** Parses a newline-separated list of document URLs. */
export function parseDocUrls(input: string | readonly string[]): string[] {
  const seen = new Set<string>();
  const urls: string[] = [];
  const lines = typeof input === "string" ? input.split(/\r?\n/) : input;
  for (const line of lines) {
    const value = line.trim();
    if (value.length === 0 || value.startsWith("#")) continue;
    const normalized = normalizeDocUrl(value);
    if (seen.has(normalized)) continue;
    seen.add(normalized);
    urls.push(normalized);
  }
  return urls;
}

const HTML_EXTENSIONS = new Set([".html", ".htm", ".xhtml"]);
const MARKDOWN_EXTENSIONS = new Set([".md", ".markdown", ".mdown", ".txt"]);
const HTML_CONTENT_TYPES = new Set(["text/html", "application/xhtml+xml"]);
const MARKDOWN_CONTENT_TYPES = new Set([
  "text/markdown",
  "text/plain",
  "text/x-markdown",
]);
const BINARY_CONTENT_TYPES = new Set([
  "application/pdf",
  "application/zip",
  "application/gzip",
  "application/octet-stream",
]);

/**
 * Decides whether a fetched response is HTML, plain text, or not trainable.
 * Returns null for obviously binary content.
 */
export function detectDocumentKind(
  url: string,
  contentType: string | null,
  text: string,
): CustomDocKind | null {
  const type = (contentType ?? "").split(";")[0]!.trim().toLowerCase();
  let extension = "";
  try {
    const path = new URL(url).pathname.toLowerCase();
    const dot = path.lastIndexOf(".");
    if (dot >= 0) extension = path.slice(dot);
  } catch {
    // The URL was already validated; an empty extension simply falls through.
  }

  const head = text.slice(0, 2048).trimStart();
  const looksHtml =
    /^(?:<!--[\s\S]*?-->\s*)?(?:<!doctype\s+html|<html[\s>]|<head[\s>]|<body[\s>])/i.test(
      head,
    );

  if (
    HTML_CONTENT_TYPES.has(type) ||
    HTML_EXTENSIONS.has(extension) ||
    looksHtml
  ) {
    return "html";
  }
  if (
    MARKDOWN_CONTENT_TYPES.has(type) ||
    type.startsWith("text/") ||
    MARKDOWN_EXTENSIONS.has(extension)
  ) {
    return "markdown";
  }
  if (BINARY_CONTENT_TYPES.has(type)) {
    if (type === "application/octet-stream" && !text.includes("\u0000")) {
      return "markdown";
    }
    return null;
  }
  if (text.includes("\u0000")) return null;
  return "markdown";
}

const SKIPPED_TAGS = new Set([
  "script",
  "style",
  "noscript",
  "template",
  "svg",
  "canvas",
  "iframe",
  "form",
  "button",
  "input",
  "select",
  "textarea",
  "nav",
  "aside",
]);
const BLOCK_TAGS = new Set([
  "address",
  "article",
  "blockquote",
  "details",
  "dialog",
  "div",
  "dl",
  "fieldset",
  "figcaption",
  "figure",
  "main",
  "p",
  "section",
  "summary",
  "ul",
  "ol",
  "li",
  "h1",
  "h2",
  "h3",
  "h4",
  "h5",
  "h6",
  "pre",
  "hr",
]);
const NOISE_NAME =
  /(?:^|[\s_-])(?:nav|navbar|navigation|sidebar|side-bar|menu|breadcrumb|footer|toc|table-of-contents|pagination|skip-link|social|share)(?:$|[\s_-])/i;

/**
 * Converts a fetched HTML document into Markdown-ish text.
 *
 * DOMParser is used instead of string/regular-expression stripping because
 * text such as `<Component />` inside a code sample must survive untouched.
 * Scripts are parsed but never executed.
 */
export function htmlToMarkdown(html: string): string {
  if (typeof DOMParser === "undefined") {
    throw new Error("HTML extraction is only available in the browser.");
  }
  const doc = new DOMParser().parseFromString(html, "text/html");

  doc
    .querySelectorAll(
      "script, style, noscript, template, svg, canvas, iframe, form, button, " +
        "input, select, textarea, nav, aside, body > header, body > footer, " +
        "[role='navigation'], [role='banner'], [role='contentinfo'], " +
        "[aria-hidden='true'], [hidden]",
    )
    .forEach((element) => element.remove());

  for (const element of Array.from(doc.querySelectorAll("[class], [id]"))) {
    const name = `${element.getAttribute("class") ?? ""} ${
      element.getAttribute("id") ?? ""
    }`;
    if (NOISE_NAME.test(name)) element.remove();
  }

  const candidates = [
    doc.querySelector("main"),
    doc.querySelector("[role='main']"),
    doc.querySelector("article"),
  ].filter((element): element is Element => element !== null);
  const root =
    candidates.sort(
      (left, right) =>
        (right.textContent?.length ?? 0) - (left.textContent?.length ?? 0),
    )[0] ??
    doc.body ??
    doc.documentElement;
  if (!root) return "";
  if (root === doc.body) {
    root.querySelectorAll("header, footer").forEach((element) => element.remove());
  }

  const blocks: string[] = [];
  const push = (value: string): void => {
    const text = value.replace(/\n{3,}/g, "\n\n").trim();
    if (text.length > 0) blocks.push(text);
  };

  const inline = (node: Node): string => {
    if (node.nodeType === 3) return node.nodeValue ?? "";
    if (node.nodeType !== 1) return "";
    const element = node as Element;
    const tag = element.tagName.toLowerCase();
    if (SKIPPED_TAGS.has(tag)) return "";
    if (tag === "br") return "\n";
    if (tag === "img") return element.getAttribute("alt") ?? "";
    if (tag === "code") {
      const code = (element.textContent ?? "").replace(/\s+/g, " ").trim();
      if (code.length === 0) return "";
      const fence = code.includes("`") ? "``" : "`";
      return `${fence}${code}${fence}`;
    }
    let value = "";
    for (const child of Array.from(element.childNodes)) {
      value += inline(child);
    }
    return value;
  };

  const normalizeInline = (node: Node): string =>
    inline(node)
      .replace(/[ \t\r\f\v]+/g, " ")
      .replace(/ *\n */g, "\n")
      .trim();
  const singleLine = (node: Node): string =>
    normalizeInline(node).replace(/\s*\n\s*/g, " ").replace(/\s+/g, " ").trim();

  const walk = (node: Node): void => {
    if (node.nodeType === 3) {
      const text = normalizeInline(node);
      if (text.length > 0) push(text);
      return;
    }
    if (node.nodeType !== 1) return;
    const element = node as Element;
    const tag = element.tagName.toLowerCase();
    if (SKIPPED_TAGS.has(tag)) return;

    if (tag === "pre") {
      const raw = (element.textContent ?? "")
        .replace(/\r\n/g, "\n")
        .replace(/^\n+|\n+$/g, "");
      if (raw.length === 0) return;
      const classNames = element.querySelector("code")?.getAttribute("class") ?? "";
      const language = /(?:^|\s)language-([\w-]+)/.exec(classNames)?.[1] ?? "";
      const fence = raw.includes("```") ? "~~~~" : "```";
      push(`${fence}${language}\n${raw}\n${fence}`);
      return;
    }
    if (/^h[1-6]$/.test(tag)) {
      const text = singleLine(element);
      if (text.length > 0) push(`${"#".repeat(Number(tag[1]))} ${text}`);
      return;
    }
    if (tag === "li") {
      const text = singleLine(element);
      if (text.length > 0) push(`- ${text}`);
      return;
    }
    if (tag === "blockquote") {
      const text = normalizeInline(element);
      if (text.length > 0) {
        push(
          text
            .split("\n")
            .map((line) => `> ${line}`)
            .join("\n"),
        );
      }
      return;
    }
    if (tag === "table") {
      for (const row of Array.from(element.querySelectorAll("tr"))) {
        const cells = Array.from(row.children)
          .filter((cell) => /^(td|th)$/i.test(cell.tagName))
          .map((cell) => singleLine(cell));
        if (cells.some((cell) => cell.length > 0)) {
          push(cells.join(" | "));
        }
      }
      return;
    }
    if (tag === "hr") {
      push("---");
      return;
    }
    if (BLOCK_TAGS.has(tag)) {
      const hasBlockChild = Array.from(element.children).some((child) =>
        BLOCK_TAGS.has(child.tagName.toLowerCase()),
      );
      if (hasBlockChild) {
        for (const child of Array.from(element.childNodes)) walk(child);
      } else {
        const text = singleLine(element);
        if (text.length > 0) push(text);
      }
      return;
    }
    for (const child of Array.from(element.childNodes)) walk(child);
  };

  for (const child of Array.from(root.childNodes)) walk(child);
  return blocks.join("\n\n");
}

/** Removes leading YAML front matter used by many documentation sites. */
export function stripFrontMatter(text: string): string {
  return text.replace(/^\uFEFF?---\r?\n[\s\S]*?\r?\n---[ \t]*(?:\r?\n|$)/, "");
}

async function fetchCustomDocs(
  urls: readonly string[],
  options: CustomDataOptions,
): Promise<CustomDoc[]> {
  const docs: CustomDoc[] = [];
  for (let index = 0; index < urls.length; index++) {
    if (options.shouldCancel?.()) {
      throw new Error("Training was cancelled.");
    }
    const url = urls[index]!;
    options.onStatus?.(
      `Downloading document ${index + 1} of ${urls.length} (${url})...`,
    );
    let response: Response;
    try {
      response = await fetch(url);
    } catch (reason) {
      throw new Error(
        `Could not fetch ${url}. Check that the URL is public and allows ` +
          `cross-origin reads (CORS). ${(reason as Error).message}`,
      );
    }
    if (!response.ok) {
      throw new Error(`fetch failed: ${response.status} ${url}`);
    }
    const raw = await response.text();
    const kind = detectDocumentKind(
      url,
      response.headers.get("content-type"),
      raw,
    );
    if (kind === null) {
      throw new Error(
        `${url} is not a text, Markdown, or HTML document ` +
          `(content type: ${response.headers.get("content-type") ?? "unknown"}).`,
      );
    }
    let text = raw;
    if (kind === "html") {
      options.onStatus?.(`Extracting readable text from ${url}...`);
      text = htmlToMarkdown(raw);
      if (text.trim().length === 0) {
        throw new Error(
          `${url} has no readable text. The page may be rendered by JavaScript ` +
            `or block automated clients.`,
        );
      }
    } else {
      text = stripFrontMatter(raw);
    }
    if (text.trim().length === 0) {
      throw new Error(`${url} is empty.`);
    }
    docs.push({ url, text, kind });
    await yieldToBrowser();
  }
  return docs;
}

/* ------------------------------------------------------------------ */
/* Tokenization                                                       */
/* ------------------------------------------------------------------ */

function appendIds(target: number[], ids: ArrayLike<number>): void {
  for (let index = 0; index < ids.length; index++) {
    target.push(ids[index]!);
  }
}

function countParagraphs(docs: readonly CustomDoc[]): number {
  let count = 0;
  for (const doc of docs) {
    for (const paragraph of doc.text.split(/\n\s*\n/)) {
      if (paragraph.trim().length > 0) count++;
    }
  }
  return count;
}

function stripMarkdown(value: string): string {
  return value
    .replace(/!\[([^\]]*)\]\([^)]*\)/g, "$1")
    .replace(/\[([^\]]+)\]\([^)]*\)/g, "$1")
    .replace(/[*_`~]/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

function cleanMarkdownLinks(value: string): string {
  return value
    .replace(/!\[([^\]]*)\]\([^)]*\)/g, "$1")
    .replace(/\[([^\]]+)\]\([^)]*\)/g, "$1");
}

/** Derives short prompts from Markdown headings, falling back to first lines. */
export function extractCustomPrompts(
  docs: readonly CustomDoc[],
  limit = MAX_PROMPTS,
): string[] {
  const prompts: string[] = [];
  const seen = new Set<string>();
  const add = (value: string): void => {
    const prompt = stripMarkdown(value).slice(0, 120).trim();
    if (prompt.length < 4 || seen.has(prompt)) return;
    seen.add(prompt);
    prompts.push(prompt);
  };

  for (const doc of docs) {
    const headings = doc.text.matchAll(/^#{1,6}[ \t]+(.+?)[ \t]*#*[ \t]*$/gm);
    for (const heading of headings) {
      add(heading[1] ?? "");
      if (prompts.length >= limit) return prompts;
    }
  }
  if (prompts.length === 0) {
    for (const doc of docs) {
      const firstLine = doc.text
        .split(/\r?\n/)
        .map((line) => line.trim())
        .find((line) => line.length > 0);
      if (firstLine) add(firstLine);
      if (prompts.length >= limit) break;
    }
  }
  return prompts;
}

export interface CustomQaPair {
  readonly question: string;
  readonly answer: string;
}

interface MarkdownSection {
  readonly title: string;
  readonly body: string;
}

/**
 * Splits Markdown into heading-delimited sections. Headings inside fenced code
 * blocks are ignored, so Luau comments that start with `#` are not mistaken for
 * document structure.
 */
function extractMarkdownSections(doc: CustomDoc): MarkdownSection[] {
  const text = stripFrontMatter(doc.text);
  const sections: MarkdownSection[] = [];
  const lines = text.split(/\r?\n/);
  let title: string | null = null;
  let body: string[] = [];
  let fence: string | null = null;

  const flush = () => {
    if (title === null) return;
    const text = body.join("\n").trim();
    if (text.length >= 20) sections.push({ title, body: text });
  };

  for (const line of lines) {
    const trimmed = line.trimStart();
    const fenceMatch = /^(```+|~~~+)/.exec(trimmed);
    if (fenceMatch) {
      if (fence === null) {
        fence = fenceMatch[1]!;
      } else if (trimmed.startsWith(fence)) {
        fence = null;
      }
    }

    if (fence === null) {
      const heading = /^#{1,6}[ \t]+(.+?)[ \t]*#*[ \t]*$/.exec(line);
      if (heading) {
        flush();
        title = stripMarkdown(heading[1] ?? "");
        body = [];
        continue;
      }
    }

    if (title !== null) body.push(line);
  }
  flush();

  if (sections.length === 0) {
    const text = stripFrontMatter(doc.text).trim();
    if (text.length === 0) return [];
    const firstLine =
      text
        .split(/\r?\n/)
        .map((line) => stripMarkdown(line))
        .find((line) => line.length > 0) ?? "Overview";
    return [{ title: firstLine.slice(0, 80), body: text }];
  }
  return sections;
}

/** Builds chat-formatted question/answer examples from Markdown headings. */
export function buildCustomQaPairs(
  docs: readonly CustomDoc[],
): CustomQaPair[] {
  const pairs: CustomQaPair[] = [];
  const seen = new Set<string>();
  for (const doc of docs) {
    for (const section of extractMarkdownSections(doc)) {
      const title = section.title.replace(/[.!?]+$/, "").trim();
      if (title.length < 2) continue;
      const alreadyQuestion =
        /\?$/.test(section.title) ||
        /^(?:what|how|why|when|where|which|who|can|does|do|is|are|should|will)\b/i.test(
          title,
        );
      const questions = alreadyQuestion
        ? [`${title}?`]
        : [
            `What is ${title}?`,
            `How do I use ${title}?`,
            `Explain ${title}.`,
          ];
      const answer = cleanMarkdownLinks(section.body);
      for (const question of questions) {
        const key = question.toLowerCase();
        if (seen.has(key)) continue;
        seen.add(key);
        pairs.push({ question, answer });
      }
    }
  }
  return pairs;
}

interface CustomTokenStream {
  readonly trainingTokens: Uint16Array;
  readonly validationTokens: Uint16Array;
}

function specialIds(tokenizer: BpeTokenizer): {
  readonly user: number;
  readonly assistant: number;
  readonly end: number;
} {
  const user = tokenizer.idFor(BPE_SPECIAL.user);
  const assistant = tokenizer.idFor(BPE_SPECIAL.assistant);
  const end = tokenizer.idFor(BPE_SPECIAL.end);
  if (user === undefined || assistant === undefined || end === undefined) {
    throw new Error("The tokenizer is missing the chat special tokens.");
  }
  return { user, assistant, end };
}

function encodeQaPair(
  tokenizer: BpeTokenizer,
  pair: CustomQaPair,
  ids: { readonly user: number; readonly assistant: number; readonly end: number },
): number[] {
  const tokens: number[] = [ids.user];
  appendIds(tokens, tokenizer.encodeOrdinary(pair.question));
  tokens.push(ids.assistant);
  appendIds(tokens, tokenizer.encodeOrdinary(pair.answer));
  tokens.push(ids.end);
  return tokens;
}

export interface CustomQaSplit {
  readonly training: readonly CustomQaPair[];
  readonly validation: readonly CustomQaPair[];
}

/**
 * Splits Q&A pairs by answer group so validation never shares an answer with
 * training. Splitting individual question variants would let "What is X?" sit
 * in validation while "How do I use X?" trains on the same answer.
 */
export function splitCustomQaPairs(
  pairs: readonly CustomQaPair[],
): CustomQaSplit {
  const answerOrder: string[] = [];
  const seen = new Set<string>();
  for (const pair of pairs) {
    if (seen.has(pair.answer)) continue;
    seen.add(pair.answer);
    answerOrder.push(pair.answer);
  }
  const validationAnswers = new Set(
    answerOrder.filter(
      (_answer, index) => (index + 1) % QA_VALIDATION_EVERY === 0,
    ),
  );
  return {
    training: pairs.filter((pair) => !validationAnswers.has(pair.answer)),
    validation: pairs.filter((pair) => validationAnswers.has(pair.answer)),
  };
}

async function buildCustomQaTokenStream(
  tokenizer: BpeTokenizer,
  pairs: readonly CustomQaPair[],
  context: number,
  onProgress: (fraction: number) => void,
  shouldCancel?: () => boolean,
): Promise<CustomTokenStream> {
  const ids = specialIds(tokenizer);
  const split = splitCustomQaPairs(pairs);
  const source = split.training.length > 0 ? split.training : pairs;

  const trainingTokens: number[] = [];
  const total = source.length * QA_REPEATS;
  let completed = 0;
  let lastPercent = -1;
  for (let pass = 0; pass < QA_REPEATS; pass++) {
    for (const pair of source) {
      appendIds(trainingTokens, encodeQaPair(tokenizer, pair, ids));
      completed += 1;
      if (completed % 8 === 0) {
        const fraction = total > 0 ? completed / total : 1;
        const percent = Math.floor(fraction * 100);
        if (percent !== lastPercent) {
          lastPercent = percent;
          onProgress(fraction);
        }
        await yieldToBrowser();
        if (shouldCancel?.()) {
          throw new Error("Training was cancelled.");
        }
      }
    }
  }
  onProgress(1);

  const validationTokens: number[] = [];
  for (const pair of split.validation) {
    appendIds(validationTokens, encodeQaPair(tokenizer, pair, ids));
  }

  const minTraining = context + 2;
  if (trainingTokens.length < minTraining) {
    throw new Error(
      `The documentation produced only ${trainingTokens.length.toLocaleString()} ` +
        `training tokens, but this model needs at least ${minTraining.toLocaleString()}. ` +
        `Add more documentation.`,
    );
  }

  return {
    trainingTokens: Uint16Array.from(trainingTokens),
    validationTokens: Uint16Array.from(validationTokens),
  };
}

async function buildCustomTokenStream(
  tokenizer: BpeTokenizer,
  docs: readonly CustomDoc[],
  context: number,
  onProgress: (fraction: number) => void,
  shouldCancel?: () => boolean,
): Promise<CustomTokenStream> {
  const endId = tokenizer.idFor(BPE_SPECIAL.end);
  if (endId === undefined) {
    throw new Error("The tokenizer is missing its end-of-document token.");
  }

  const chunks: string[] = [];
  for (const doc of docs) {
    for (const paragraph of doc.text.split(/\n\s*\n/)) {
      const text = paragraph.trim();
      if (text.length > 0) chunks.push(text);
    }
  }
  if (chunks.length === 0) {
    throw new Error("The documents have no text to train on.");
  }

  const tokens: number[] = [];
  let lastPercent = -1;
  for (let index = 0; index < chunks.length; index++) {
    appendIds(tokens, tokenizer.encodeOrdinary(chunks[index]!));
    tokens.push(endId);
    if (index % 8 === 0) {
      const fraction = index / chunks.length;
      const percent = Math.floor(fraction * 100);
      if (percent !== lastPercent) {
        lastPercent = percent;
        onProgress(fraction);
      }
      await yieldToBrowser();
      if (shouldCancel?.()) {
        throw new Error("Training was cancelled.");
      }
    }
  }
  onProgress(1);

  const minTraining = context + 2;
  if (tokens.length < minTraining) {
    throw new Error(
      `The documents are too short: ${tokens.length.toLocaleString()} tokens, ` +
        `but this model needs at least ${minTraining.toLocaleString()}. ` +
        `Add more Markdown files or content.`,
    );
  }

  // Hold out the newest ~10% of the stream, but keep a full training window.
  // A short corpus may end up with no validation tokens; trainTokenGpt treats
  // that as "no validation result" instead of failing.
  let split = Math.floor(tokens.length * (1 - VALIDATION_FRACTION));
  if (tokens.length - split < minTraining) {
    split = Math.max(minTraining, tokens.length - minTraining);
  }
  if (split > tokens.length) split = tokens.length;

  return {
    trainingTokens: Uint16Array.from(tokens.slice(0, split)),
    validationTokens: Uint16Array.from(tokens.slice(split)),
  };
}

/* ------------------------------------------------------------------ */
/* Data loading                                                       */
/* ------------------------------------------------------------------ */

function modelBase(options: CustomDataOptions): string {
  return (options.modelBaseUrl ?? SOLID2_MODEL_BASE).replace(/\/?$/, "/");
}

export async function loadCustomTrainingData(
  options: CustomDataOptions,
): Promise<CustomTrainingData> {
  const urls = parseDocUrls(options.urls);
  if (urls.length === 0) {
    throw new Error("Add at least one document URL.");
  }

  const tokenizerUrl =
    options.tokenizerUrl ?? `${modelBase(options)}tokenizer.json`;
  options.onStatus?.("Loading the deployed Solid 2 BPE tokenizer...");
  const tokenizer = await loadBpeTokenizer(tokenizerUrl);

  const docs = await fetchCustomDocs(urls, options);
  const documentChars = docs.reduce((total, doc) => total + doc.text.length, 0);
  if (documentChars === 0) {
    throw new Error("The downloaded documents are empty.");
  }

  const assistant = options.assistant === true;
  const qaPairs = assistant ? buildCustomQaPairs(docs) : [];
  if (assistant && qaPairs.length === 0) {
    throw new Error(
      "Assistant (Q&A) mode needs Markdown headings to build questions. " +
        "Uncheck it for plain-text training, or add heading-based documents.",
    );
  }

  const dataHash = hashString(
    docs.map((doc) => `${doc.url}\u0000${doc.text}`).join("\u0001"),
  );
  const tokenizerHash = hashString(JSON.stringify(tokenizer.toJSON()));
  const mode = assistant ? "qa" : "docs";
  const cacheKey = `v${TOKEN_SCHEME_VERSION}:${mode}:${tokenizerHash}:${dataHash}`;

  let stream: CustomTokenStream;
  let cached = false;
  const hit = await tokenCache.read(cacheKey);
  if (hit) {
    stream = {
      trainingTokens: hit.trainingTokens,
      validationTokens: hit.validationTokens,
    };
    cached = true;
    options.onStatus?.(
      `Loaded the cached token stream (${hit.trainingTokens.length.toLocaleString()} training tokens).`,
    );
  } else {
    if (assistant) {
      options.onStatus?.(
        `Building ${qaPairs.length.toLocaleString()} question/answer examples from Markdown headings...`,
      );
      stream = await buildCustomQaTokenStream(
        tokenizer,
        qaPairs,
        SOLID2_TRAIN_CONFIG.context,
        (fraction) =>
          options.onStatus?.(
            `Tokenizing question/answer examples (${Math.round(fraction * 100)}%)...`,
          ),
        options.shouldCancel,
      );
    } else {
      options.onStatus?.(
        `Tokenizing ${urls.length} document${urls.length === 1 ? "" : "s"} with the BPE tokenizer...`,
      );
      stream = await buildCustomTokenStream(
        tokenizer,
        docs,
        SOLID2_TRAIN_CONFIG.context,
        (fraction) =>
          options.onStatus?.(`Tokenizing documents (${Math.round(fraction * 100)}%)...`),
        options.shouldCancel,
      );
    }
    options.onStatus?.("Caching the token stream in IndexedDB...");
    await tokenCache.write({
      key: cacheKey,
      trainingTokens: stream.trainingTokens,
      validationTokens: stream.validationTokens,
      savedAt: Date.now(),
    });
  }

  let prompts: string[];
  if (assistant) {
    // One suggestion per section, rather than three variants of the first one.
    prompts = [];
    const seenAnswers = new Set<string>();
    for (const pair of qaPairs) {
      if (seenAnswers.has(pair.answer)) continue;
      seenAnswers.add(pair.answer);
      prompts.push(pair.question);
      if (prompts.length >= MAX_PROMPTS) break;
    }
  } else {
    prompts = extractCustomPrompts(docs);
  }
  const corpusHash = hashString(
    `${TOKEN_SCHEME_VERSION}:${mode}:${tokenizerHash}:${dataHash}`,
  );
  const info: CustomTrainingDataInfo = {
    documentCount: docs.length,
    documentChars,
    htmlDocuments: docs.filter((doc) => doc.kind === "html").length,
    paragraphs: countParagraphs(docs),
    trainingTokens: stream.trainingTokens.length,
    validationTokens: stream.validationTokens.length,
    tokenizerVocab: tokenizer.vocab.length,
    cached,
    warmStart: false,
    assistant,
    qaPairs: qaPairs.length,
  };

  const htmlCount = docs.filter((doc) => doc.kind === "html").length;
  options.onStatus?.(
    `Ready: ${stream.trainingTokens.length.toLocaleString()} training tokens from ` +
      `${docs.length} document${docs.length === 1 ? "" : "s"}` +
      `${assistant ? ` and ${qaPairs.length.toLocaleString()} Q&A examples` : ""}` +
      `${htmlCount > 0 ? ` (${htmlCount} HTML page${htmlCount === 1 ? "" : "s"} converted)` : ""}` +
      `${cached ? " (cached)" : ""}.`,
  );

  return {
    tokenizer,
    prompts,
    info,
    trainingData: {
      vocab: tokenizer.vocab,
      trainingTokens: stream.trainingTokens,
      validationTokens: stream.validationTokens,
      corpusHash,
      corpusSize: stream.trainingTokens.length,
      corpusLabel: "corpusTokens",
    },
  };
}

/* ------------------------------------------------------------------ */
/* Training and export                                                */
/* ------------------------------------------------------------------ */

export async function trainCustomGpt(
  options: CustomTrainingOptions = { urls: [] },
): Promise<CustomTrainingResult> {
  const data = await loadCustomTrainingData(options);
  const urls = parseDocUrls(options.urls);
  const assistant = data.info.assistant;

  let initialWeights: Float32Array | undefined;
  let warmStarted = false;
  if (options.warmStart && !options.initialState) {
    options.onStatus?.(
      "Loading the bundled Solid 2 checkpoint for a warm start...",
    );
    const bundled = await loadBundledSolid2Model(modelBase(options));
    const expected = solid2ParameterCount(bundled.checkpoint.config);
    if (bundled.weights.length !== expected) {
      throw new Error(
        `Bundled weights.bin has ${bundled.weights.length.toLocaleString()} values but ` +
          `the checkpoint config needs ${expected.toLocaleString()}.`,
      );
    }
    if (bundled.checkpoint.config.vocabSize !== data.trainingData.vocab.length) {
      throw new Error(
        "The bundled Solid 2 vocabulary does not match the BPE tokenizer.",
      );
    }
    initialWeights = bundled.weights;
    warmStarted = true;
    options.onStatus?.(
      `Warm-starting from the bundled model (${expected.toLocaleString()} params).`,
    );
  } else {
    options.onStatus?.(
      "Selecting the TensorFlow.js backend and starting training...",
    );
  }

  const attachSources = (state: BrowserTrainingState): BrowserTrainingState => ({
    ...state,
    sourceUrls: urls,
    assistant,
  });

  const result = await trainTokenGpt({
    data: data.trainingData,
    config: SOLID2_TRAIN_CONFIG,
    defaultConfig: SOLID2_TRAIN_CONFIG,
    steps: options.steps ?? 600,
    batch: options.batch ?? 4,
    lr: options.lr ?? 4e-4,
    seed: options.seed ?? 1337,
    dropout: options.dropout ?? 0.1,
    maxMinutes: options.maxMinutes ?? 30,
    backend: options.backend,
    initialState: options.initialState,
    initialWeights,
    onProgress: options.onProgress,
    onCheckpoint: options.onCheckpoint
      ? (state) => options.onCheckpoint!(attachSources(state))
      : undefined,
    shouldCancel: options.shouldCancel,
    yieldEvery: options.yieldEvery ?? 2,
    saveEvery: options.saveEvery ?? 100,
    kind: "custom",
    // Instruction-following models memorize the Q&A pairs. With an honest
    // answer-group holdout, best-validation can select a much too early step.
    exportFinal: true,
    modelMeta: {
      name: assistant
        ? "tiny-gpt-custom-qa-browser"
        : "tiny-gpt-custom-browser",
      source:
        "Custom Markdown, text, and HTML documents supplied by the user: " +
        `${urls.slice(0, 8).join(", ")}` +
        (urls.length > 8 ? ` and ${urls.length - 8} more` : "") +
        ".",
      license:
        "Upstream documents retain their own licenses; model weights trained " +
        "in the browser by this repository.",
    },
  });

  const state = attachSources(result.state);
  const checkpoint: Solid2Checkpoint = {
    format: "only-gates-tiny-gpt-custom",
    version: 1,
    config: result.checkpoint.config,
    parameterCount: solid2ParameterCount(result.checkpoint.config),
    tokenizer: "tokenizer.json",
    meta: {
      name: result.checkpoint.meta.name,
      source: result.checkpoint.meta.source,
      license: result.checkpoint.meta.license,
      training: {
        ...result.checkpoint.meta.training,
        assistant,
      },
    },
    samples: [...result.checkpoint.samples],
    prompts: [...data.prompts],
  };

  return {
    result: { ...result, state },
    checkpoint,
    tokenizer: data.tokenizer,
    weights: packTinyGptWeights(result.checkpoint),
    data: { ...data.info, warmStart: warmStarted },
    prompts: data.prompts,
  };
}

/** Builds the integer-only WASM model from a completed custom training run. */
export function buildCustomWasmFromTraining(
  result: CustomTrainingResult,
): Promise<Solid2WasmModel> {
  return createSolid2WasmFromData(
    result.checkpoint,
    result.tokenizer,
    result.weights,
  );
}
