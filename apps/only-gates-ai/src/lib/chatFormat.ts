/*
 * Tiny Markdown helpers for chat answers.
 *
 * Answers are written with fenced and inline code. Rendering them as plain
 * text would show the backticks, so split the answer into prose and code
 * segments. This is deliberately not a full Markdown parser.
 */

export interface AnswerSegment {
  readonly kind: "text" | "code";
  readonly text: string;
  readonly language?: string;
}

export function splitAnswer(text: string): AnswerSegment[] {
  const segments: AnswerSegment[] = [];
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

export function splitInlineCode(text: string): AnswerSegment[] {
  const parts: AnswerSegment[] = [];
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
