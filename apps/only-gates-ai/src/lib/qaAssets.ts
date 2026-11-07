import {
  WORD_VECTOR_BITS,
  WORD_VECTOR_DATA,
  WORD_VECTOR_WORDS,
} from "../assets/word-vectors.ts";
import { qaLexiconFingerprint, type QaLexicon } from "./qaRuntime.ts";

/** The bundled Word2Bits subset used by the in-app QA model. */
export const BUNDLED_QA_LEXICON: QaLexicon = {
  bits: WORD_VECTOR_BITS,
  words: WORD_VECTOR_WORDS,
  vectors: WORD_VECTOR_DATA,
};

let cachedFingerprint: string | null = null;

export function bundledQaLexiconFingerprint(): string {
  if (cachedFingerprint === null) {
    cachedFingerprint = qaLexiconFingerprint(BUNDLED_QA_LEXICON);
  }
  return cachedFingerprint;
}
