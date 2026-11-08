import { ByteLevelBpe, parseBpeModel, type BpeModel } from "@only-gates-ai/core";

export const BPE_SPECIAL = {
  pad: "<|pad|>",
  user: "<|user|>",
  assistant: "<|assistant|>",
  end: "<|end|>",
  facts: "<|facts|>",
} as const;

export type BpeTokenizer = ByteLevelBpe;

export async function loadBpeTokenizer(url: string): Promise<ByteLevelBpe> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`tokenizer fetch failed: ${res.status} ${url}`);
  const json = (await res.json()) as BpeModel;
  return parseBpeModel(json);
}

export function formatChatPrompt(tokenizer: ByteLevelBpe, question: string): string {
  // Encode with special tokens preserved.
  return `${BPE_SPECIAL.user}${question}${BPE_SPECIAL.assistant}`;
}

export function encodeChatPrompt(tokenizer: ByteLevelBpe, question: string): number[] {
  return tokenizer.encode(formatChatPrompt(tokenizer, question));
}

export function decodeGenerated(
  tokenizer: ByteLevelBpe,
  ids: readonly number[],
): string {
  const endId = tokenizer.idFor(BPE_SPECIAL.end);
  const padId = tokenizer.idFor(BPE_SPECIAL.pad);
  const userId = tokenizer.idFor(BPE_SPECIAL.user);
  const asstId = tokenizer.idFor(BPE_SPECIAL.assistant);
  const factsId = tokenizer.idFor(BPE_SPECIAL.facts);
  const filtered = ids.filter(
    (id) => id !== endId && id !== padId && id !== userId && id !== asstId && id !== factsId,
  );
  return tokenizer.decode(filtered);
}

export function isStopToken(tokenizer: ByteLevelBpe, id: number): boolean {
  return id === tokenizer.idFor(BPE_SPECIAL.end) || id === tokenizer.idFor(BPE_SPECIAL.pad);
}
