/*
 * Byte-level BPE tokenizer.
 *
 * The vocabulary and merges are produced once by scripts/train-bpe.mjs and
 * shared by the PyTorch trainer, the browser demo, and the tests. Byte-level
 * BPE keeps arbitrary code, casing, whitespace, and Unicode round-trippable;
 * word-level tokenizers cannot represent `createEffect`, JSX, or indentation
 * faithfully.
 *
 * Special tokens are matched before pre-tokenization, so they always map to a
 * single id even if the same text appears inside normal prose.
 */

export const BPE_SPECIAL_TOKENS = [
  "<|pad|>",
  "<|user|>",
  "<|assistant|>",
  "<|end|>",
  "<|facts|>",
] as const;

export type BpeSpecialToken = (typeof BPE_SPECIAL_TOKENS)[number];

export interface BpeModel {
  readonly version: 1;
  readonly specialTokens: readonly string[];
  readonly vocab: readonly string[];
  readonly merges: readonly (readonly [string, string])[];
}

export interface BpeTrainOptions {
  readonly vocabSize?: number;
  readonly specialTokens?: readonly string[];
  readonly minFrequency?: number;
  /** Called with progress messages; useful for the training script. */
  readonly onProgress?: (message: string) => void;
}

const BYTE_TO_CHAR = (() => {
  const bytes: number[] = [];
  for (let i = 33; i <= 126; i++) bytes.push(i);
  for (let i = 161; i <= 172; i++) bytes.push(i);
  for (let i = 174; i <= 255; i++) bytes.push(i);
  const codePoints = bytes.slice();
  let extra = 0;
  for (let b = 0; b < 256; b++) {
    if (!bytes.includes(b)) {
      bytes.push(b);
      codePoints.push(256 + extra);
      extra++;
    }
  }
  const map = new Array<string>(256);
  for (let i = 0; i < bytes.length; i++) {
    map[bytes[i]!] = String.fromCodePoint(codePoints[i]!);
  }
  return map;
})();

const CHAR_TO_BYTE = (() => {
  const map = new Map<string, number>();
  for (let b = 0; b < 256; b++) {
    map.set(BYTE_TO_CHAR[b]!, b);
  }
  return map;
})();

/*
 * GPT-2 style pre-tokenization. It keeps a word, a number, a run of
 * punctuation, or a run of whitespace together. The leading-space branch is
 * important for code, where indentation belongs to the following token.
 */
const PRE_TOKENIZE = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

const PAIR_STRIDE = 1 << 16;

function pairKey(left: number, right: number): number {
  return left * PAIR_STRIDE + right;
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function preTokenize(text: string): string[] {
  return text.match(PRE_TOKENIZE) ?? [];
}

function bytesToSymbols(text: string): string[] {
  const bytes = new TextEncoder().encode(text);
  const symbols = new Array<string>(bytes.length);
  for (let i = 0; i < bytes.length; i++) {
    symbols[i] = BYTE_TO_CHAR[bytes[i]!]!;
  }
  return symbols;
}

/** A packed byte-level BPE tokenizer. */
export class ByteLevelBpe {
  readonly vocab: readonly string[];
  readonly merges: readonly (readonly [string, string])[];
  readonly specialTokens: readonly string[];
  private readonly tokenToId = new Map<string, number>();
  private readonly mergeRanks = new Map<string, number>();
  private readonly specialToId = new Map<string, number>();
  private readonly specialPattern: RegExp | null;
  private readonly pieceCache = new Map<string, number[]>();

  constructor(model: BpeModel) {
    this.vocab = model.vocab;
    this.merges = model.merges;
    this.specialTokens = model.specialTokens;
    model.vocab.forEach((token, id) => {
      // The first occurrence wins. Special tokens are unique in practice.
      if (!this.tokenToId.has(token)) this.tokenToId.set(token, id);
    });
    model.merges.forEach(([left, right], rank) => {
      this.mergeRanks.set(`${left}\u0000${right}`, rank);
    });
    model.specialTokens.forEach((token, index) => {
      const id = this.tokenToId.get(token);
      if (id !== undefined) this.specialToId.set(token, id);
      else this.specialToId.set(token, index);
    });
    if (model.specialTokens.length > 0) {
      const alternatives = model.specialTokens
        .slice()
        .sort((a, b) => b.length - a.length)
        .map(escapeRegExp)
        .join("|");
      this.specialPattern = new RegExp(`(${alternatives})`, "gu");
    } else {
      this.specialPattern = null;
    }
  }

  get size(): number {
    return this.vocab.length;
  }

  idFor(token: string): number | undefined {
    return this.tokenToId.get(token);
  }

  tokenFor(id: number): string | undefined {
    return this.vocab[id];
  }

  /** Encodes text, preserving special tokens as single ids. */
  encode(text: string): number[] {
    if (text.length === 0) return [];
    if (this.specialPattern === null) return this.encodeOrdinary(text);
    const ids: number[] = [];
    let cursor = 0;
    this.specialPattern.lastIndex = 0;
    for (;;) {
      const match = this.specialPattern.exec(text);
      if (match === null) break;
      const start = match.index;
      if (start > cursor) {
        ids.push(...this.encodeOrdinary(text.slice(cursor, start)));
      }
      const special = match[0];
      const id = this.specialToId.get(special);
      if (id === undefined) {
        ids.push(...this.encodeOrdinary(special));
      } else {
        ids.push(id);
      }
      cursor = start + special.length;
    }
    if (cursor < text.length) {
      ids.push(...this.encodeOrdinary(text.slice(cursor)));
    }
    return ids;
  }

  /** Encodes ordinary text without looking for special tokens. */
  encodeOrdinary(text: string): number[] {
    const ids: number[] = [];
    for (const piece of preTokenize(text)) {
      const cached = this.pieceCache.get(piece);
      if (cached !== undefined) {
        ids.push(...cached);
        continue;
      }
      const symbols = bytesToSymbols(piece);
      this.mergeSymbols(symbols);
      const pieceIds: number[] = [];
      for (const symbol of symbols) {
        const id = this.tokenToId.get(symbol);
        if (id === undefined) {
          // A byte-level vocabulary always contains every single byte, so
          // this only happens for a corrupt model.
          throw new Error(`tokenizer is missing symbol ${JSON.stringify(symbol)}`);
        }
        pieceIds.push(id);
      }
      if (this.pieceCache.size < 100_000) {
        this.pieceCache.set(piece, pieceIds);
      }
      ids.push(...pieceIds);
    }
    return ids;
  }

  /** Decodes ids back to text. Unknown ids are rendered as U+FFFD. */
  decode(ids: readonly number[]): string {
    const specialIds = new Set(this.specialToId.values());
    const bytes: number[] = [];
    let output = "";
    const flush = () => {
      if (bytes.length === 0) return;
      output += new TextDecoder().decode(new Uint8Array(bytes));
      bytes.length = 0;
    };
    for (const id of ids) {
      if (specialIds.has(id)) {
        flush();
        output += this.vocab[id] ?? "\uFFFD";
        continue;
      }
      const token = this.vocab[id];
      if (token === undefined) {
        bytes.push(0xef, 0xbf, 0xbd);
        continue;
      }
      for (const char of token) {
        const byte = CHAR_TO_BYTE.get(char);
        if (byte === undefined) {
          bytes.push(0xef, 0xbf, 0xbd);
        } else {
          bytes.push(byte);
        }
      }
    }
    flush();
    return output;
  }

  toJSON(): BpeModel {
    return {
      version: 1,
      specialTokens: this.specialTokens,
      vocab: this.vocab,
      merges: this.merges,
    };
  }

  private mergeSymbols(symbols: string[]): void {
    while (symbols.length > 1) {
      let bestRank = Number.POSITIVE_INFINITY;
      let bestIndex = -1;
      for (let i = 0; i < symbols.length - 1; i++) {
        const rank = this.mergeRanks.get(`${symbols[i]!}\u0000${symbols[i + 1]!}`);
        if (rank !== undefined && rank < bestRank) {
          bestRank = rank;
          bestIndex = i;
        }
      }
      if (bestIndex < 0) return;
      const merged = symbols[bestIndex]! + symbols[bestIndex + 1]!;
      symbols.splice(bestIndex, 2, merged);
    }
  }
}

/** Parses a tokenizer JSON document produced by trainBpe or train-bpe.mjs. */
export function parseBpeModel(json: string | BpeModel): ByteLevelBpe {
  const model = (typeof json === "string" ? JSON.parse(json) : json) as BpeModel;
  if (model.version !== 1) {
    throw new Error(`unsupported tokenizer version ${String((model as { version?: unknown }).version)}`);
  }
  return new ByteLevelBpe(model);
}

/*
 * Training is intentionally dependency-free. It counts unique pre-tokens, then
 * repeatedly merges the most frequent adjacent pair. Only words that contain
 * the chosen pair are rebuilt, which keeps the cost proportional to the number
 * of merge occurrences instead of the whole corpus.
 */
export function trainByteLevelBpe(corpus: string, options: BpeTrainOptions = {}): BpeModel {
  const specialTokens = (options.specialTokens ?? BPE_SPECIAL_TOKENS) as readonly string[];
  const vocabSize = options.vocabSize ?? 4096;
  const minFrequency = options.minFrequency ?? 2;
  const progress = options.onProgress ?? (() => {});
  const baseVocab = [...specialTokens];
  for (let b = 0; b < 256; b++) baseVocab.push(BYTE_TO_CHAR[b]!);
  for (const token of specialTokens) {
    if (baseVocab.indexOf(token) !== baseVocab.lastIndexOf(token)) {
      throw new Error(`duplicate special token ${token}`);
    }
  }
  const vocab = baseVocab.slice();
  const tokenToId = new Map<string, number>();
  vocab.forEach((token, id) => tokenToId.set(token, id));
  const words: number[][] = [];
  const frequencies: number[] = [];
  const wordIndex = new Map<string, number>();
  for (const piece of preTokenize(corpus)) {
    const key = piece;
    const existing = wordIndex.get(key);
    if (existing !== undefined) {
      frequencies[existing]!++;
      continue;
    }
    const symbols = bytesToSymbols(piece).map((symbol) => {
      const id = tokenToId.get(symbol);
      if (id === undefined) throw new Error("missing base byte token");
      return id;
    });
    wordIndex.set(key, words.length);
    words.push(symbols);
    frequencies.push(1);
  }
  progress(`tokenizer: ${words.length} unique pre-tokens, ${vocab.length} base tokens`);
  const pairCounts = new Map<number, number>();
  const pairWords = new Map<number, Set<number>>();
  const mergePairs: [number, number][] = [];
  const addPair = (left: number, right: number, word: number, count: number) => {
    const key = pairKey(left, right);
    pairCounts.set(key, (pairCounts.get(key) ?? 0) + count);
    let set = pairWords.get(key);
    if (set === undefined) {
      set = new Set();
      pairWords.set(key, set);
    }
    set.add(word);
  };
  const removePair = (left: number, right: number, word: number, count: number) => {
    const key = pairKey(left, right);
    const next = (pairCounts.get(key) ?? 0) - count;
    if (next > 0) {
      pairCounts.set(key, next);
    } else {
      pairCounts.delete(key);
    }
    const set = pairWords.get(key);
    if (set !== undefined) {
      set.delete(word);
      if (set.size === 0) pairWords.delete(key);
    }
  };
  for (let w = 0; w < words.length; w++) {
    const word = words[w]!;
    const count = frequencies[w]!;
    for (let i = 0; i < word.length - 1; i++) {
      addPair(word[i]!, word[i + 1]!, w, count);
    }
  }
  while (vocab.length < vocabSize) {
    let bestPair = -1;
    let bestCount = 0;
    for (const [key, count] of pairCounts) {
      if (count > bestCount || (count === bestCount && key < bestPair)) {
        bestPair = key;
        bestCount = count;
      }
    }
    if (bestPair < 0 || bestCount < minFrequency) break;
    const left = Math.floor(bestPair / PAIR_STRIDE);
    const right = bestPair % PAIR_STRIDE;
    const merged = vocab[left]! + vocab[right]!;
    const newId = vocab.length;
    vocab.push(merged);
    tokenToId.set(merged, newId);
    mergePairs.push([left, right]);
    const affected = Array.from(pairWords.get(bestPair) ?? []);
    for (const w of affected) {
      const word = words[w]!;
      const count = frequencies[w]!;
      for (let i = 0; i < word.length - 1; i++) {
        removePair(word[i]!, word[i + 1]!, w, count);
      }
    }
    for (const w of affected) {
      const word = words[w]!;
      const rebuilt: number[] = [];
      for (let i = 0; i < word.length; ) {
        if (i + 1 < word.length && word[i] === left && word[i + 1] === right) {
          rebuilt.push(newId);
          i += 2;
        } else {
          rebuilt.push(word[i]!);
          i++;
        }
      }
      words[w] = rebuilt;
    }
    for (const w of affected) {
      const word = words[w]!;
      const count = frequencies[w]!;
      for (let i = 0; i < word.length - 1; i++) {
        addPair(word[i]!, word[i + 1]!, w, count);
      }
    }
    if ((vocab.length - baseVocab.length) % 256 === 0) {
      progress(
        `tokenizer: ${vocab.length - baseVocab.length} merges, ` + `best pair count ${bestCount}`,
      );
    }
  }
  const merges = mergePairs.map(([left, right]) => [vocab[left]!, vocab[right]!] as const);
  return {
    version: 1,
    specialTokens,
    vocab,
    merges,
  };
}

/** Convenience wrapper used by the browser loader. */
export function createBpeTokenizer(model: BpeModel): ByteLevelBpe {
  return new ByteLevelBpe(model);
}