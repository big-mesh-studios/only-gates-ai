#!/usr/bin/env python3
"""Train a custom-docs Integer GPT model on Colab and export an importable ZIP.

This is the Colab counterpart to the browser's **Custom Docs (BPE)** trainer.
It fetches the URLs supplied by the app, tokenizes them with the deployed
Solid 2 BPE tokenizer, trains the same tiny GPT architecture with PyTorch, and
writes an inference-only ``only-gates-model-bundle`` ZIP for the app's
**Import model (.zip)** flow.

The model/tokenizer/packing helpers live in
``scripts/train_tiny_gpt_solid2_colab.py``; this file adds the custom URL
corpus, Q&A mode, and bundle export.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import signal
import sys
import time
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import model_bundle  # noqa: E402
import train_tiny_gpt_solid2_colab as base  # noqa: E402

CHECKPOINT_FORMAT = "only-gates-custom-train"
CHECKPOINT_VERSION = 1
TOKEN_SCHEME_VERSION = 3
VALIDATION_FRACTION = 0.1
MAX_PROMPTS = 5
QA_REPEATS = 3
QA_VALIDATION_EVERY = 10
MAX_SOURCE_URLS = 256
DEFAULT_TOKENIZER_URL = (
    "https://raw.githubusercontent.com/clinuxrulz/only-gates-ai/main/"
    "apps/only-gates-ai/public/models/tiny-gpt-solid2/tokenizer.json"
)
SPECIAL_USER = "<|user|>"
SPECIAL_ASSISTANT = "<|assistant|>"
SPECIAL_END = "<|end|>"


# ---------------------------------------------------------------------------
# URLs and document extraction
# ---------------------------------------------------------------------------


def normalize_url(value: str) -> str:
    """Normalize GitHub blob URLs to raw URLs, matching the browser loader."""
    value = value.strip()
    if not value:
        raise ValueError("empty URL")
    # Colab notebooks can write a local Markdown companion file (for example
    # the Luau FAQ) and pass it as a file:// URL.
    if value.startswith("file://"):
        return value
    if not (value.startswith("http://") or value.startswith("https://")):
        raise ValueError(f"Only http(s) or file URLs are supported: {value}")
    if "github.com/" in value:
        parsed = re.match(r"https?://(?:www\.)?github\.com/([^/]+)/([^/]+)/blob/(.+)", value)
        if parsed:
            owner, repo, rest = parsed.groups()
            return f"https://raw.githubusercontent.com/{owner}/{repo}/{rest}"
    return value


def parse_urls(urls_file: Optional[Path], extra: Sequence[str]) -> List[str]:
    values: List[str] = []
    if urls_file is not None:
        if not urls_file.exists():
            raise SystemExit(f"URL file not found: {urls_file}")
        values.extend(urls_file.read_text(encoding="utf-8").splitlines())
    values.extend(extra)
    seen = set()
    urls: List[str] = []
    for line in values:
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        normalized = normalize_url(text)
        if normalized in seen:
            continue
        seen.add(normalized)
        urls.append(normalized)
    if not urls:
        raise SystemExit("Add at least one document URL (--urls-file or --url).")
    if len(urls) > MAX_SOURCE_URLS:
        raise SystemExit(f"At most {MAX_SOURCE_URLS} document URLs are supported.")
    return urls


HTML_EXTENSIONS = {".html", ".htm", ".xhtml"}
MARKDOWN_EXTENSIONS = {".md", ".markdown", ".mdown", ".txt"}
HTML_CONTENT_TYPES = {"text/html", "application/xhtml+xml"}
MARKDOWN_CONTENT_TYPES = {"text/markdown", "text/plain", "text/x-markdown"}
BINARY_CONTENT_TYPES = {
    "application/pdf",
    "application/zip",
    "application/gzip",
    "application/octet-stream",
}


def detect_document_kind(url: str, content_type: Optional[str], text: str) -> Optional[str]:
    content_type = (content_type or "").split(";")[0].strip().lower()
    path = url.split("?", 1)[0].lower()
    dot = path.rfind(".")
    extension = path[dot:] if dot >= 0 else ""
    head = text.lstrip()[:2048]
    looks_html = bool(
        re.match(
            r"^(?:<!--[\s\S]*?-->\s*)?(?:<!doctype\s+html|<html[\s>]|<head[\s>]|<body[\s>])",
            head,
            re.IGNORECASE,
        )
    )
    if content_type in HTML_CONTENT_TYPES or extension in HTML_EXTENSIONS or looks_html:
        return "html"
    if (
        content_type in MARKDOWN_CONTENT_TYPES
        or content_type.startswith("text/")
        or extension in MARKDOWN_EXTENSIONS
    ):
        return "markdown"
    if content_type in BINARY_CONTENT_TYPES:
        if content_type == "application/octet-stream" and "\x00" not in text:
            return "markdown"
        return None
    if "\x00" in text:
        return None
    return "markdown"


def strip_front_matter(text: str) -> str:
    return re.sub(r"^\ufeff?---\r?\n[\s\S]*?\r?\n---[ \t]*(?:\r?\n|$)", "", text, count=1)


def _fetch_url(url: str, attempts: int = 4) -> Tuple[bytes, Optional[str]]:
    last: Optional[Exception] = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "only-gates-ai-custom-trainer"},
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read(), response.headers.get("Content-Type")
        except Exception as exc:  # noqa: BLE001
            last = exc
            if attempt + 1 < attempts:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"fetch failed {url}: {last}")


def _decode_document(data: bytes, content_type: Optional[str]) -> str:
    charset = "utf-8"
    match = re.search(r"charset=([\w.-]+)", content_type or "", re.IGNORECASE)
    if match:
        charset = match.group(1)
    try:
        return data.decode(charset, errors="replace")
    except LookupError:
        return data.decode("utf-8", errors="replace")


class _Node:
    __slots__ = ("tag", "attrs", "children", "text")

    def __init__(self, tag: Optional[str], attrs: Mapping[str, str], text: Optional[str] = None):
        self.tag = tag
        self.attrs = dict(attrs)
        self.children: List["_Node"] = []
        self.text = text


class _TreeParser(HTMLParser):
    VOID_TAGS = {
        "area", "base", "br", "col", "embed", "hr", "img", "input",
        "link", "meta", "param", "source", "track", "wbr",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = _Node("#root", {})
        self.stack: List[_Node] = [self.root]

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        node = _Node(tag.lower(), {key: value or "" for key, value in attrs})
        self.stack[-1].children.append(node)
        if node.tag not in self.VOID_TAGS:
            self.stack.append(node)

    def handle_startendtag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        node = _Node(tag.lower(), {key: value or "" for key, value in attrs})
        self.stack[-1].children.append(node)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        for index in range(len(self.stack) - 1, 0, -1):
            if self.stack[index].tag == tag:
                del self.stack[index:]
                return

    def handle_data(self, data: str) -> None:
        if data:
            self.stack[-1].children.append(_Node(None, {}, data))


SKIPPED_TAGS = {
    "script", "style", "noscript", "template", "svg", "canvas", "iframe",
    "form", "button", "input", "select", "textarea", "nav", "aside",
    "header", "footer",
}
BLOCK_TAGS = {
    "address", "article", "blockquote", "details", "dialog", "div", "dl",
    "fieldset", "figcaption", "figure", "main", "p", "section", "summary",
    "ul", "ol", "li", "h1", "h2", "h3", "h4", "h5", "h6", "pre", "hr",
}
NOISE_NAME = re.compile(
    r"(?:^|[\s_-])(?:nav|navbar|navigation|sidebar|side-bar|menu|breadcrumb|"
    r"footer|toc|table-of-contents|pagination|skip-link|social|share)"
    r"(?:$|[\s_-])",
    re.IGNORECASE,
)
VOID_HTML_TAGS = _TreeParser.VOID_TAGS


def _text_content(node: _Node) -> str:
    if node.tag is None:
        return node.text or ""
    return "".join(_text_content(child) for child in node.children)


def _is_noise(node: _Node) -> bool:
    if node.tag in SKIPPED_TAGS:
        return True
    role = node.attrs.get("role", "").lower()
    if role in {"navigation", "banner", "contentinfo"}:
        return True
    if node.attrs.get("aria-hidden", "").lower() == "true" or "hidden" in node.attrs:
        return True
    name = f"{node.attrs.get('class', '')} {node.attrs.get('id', '')}"
    return bool(NOISE_NAME.search(name))


def _prune(node: _Node) -> None:
    kept: List[_Node] = []
    for child in node.children:
        if child.tag is not None and _is_noise(child):
            continue
        _prune(child)
        kept.append(child)
    node.children = kept


def _find_nodes(node: _Node, predicate) -> List[_Node]:
    found = [node] if predicate(node) else []
    for child in node.children:
        found.extend(_find_nodes(child, predicate))
    return found


def _descendants(node: _Node, tag: str) -> Iterable[_Node]:
    for child in node.children:
        if child.tag == tag:
            yield child
        yield from _descendants(child, tag)


def _inline(node: _Node) -> str:
    if node.tag is None:
        return node.text or ""
    if node.tag in SKIPPED_TAGS:
        return ""
    if node.tag == "br":
        return "\n"
    if node.tag == "img":
        return node.attrs.get("alt", "")
    if node.tag == "code":
        code = re.sub(r"\s+", " ", _text_content(node)).strip()
        if not code:
            return ""
        fence = "``" if "`" in code else "`"
        return f"{fence}{code}{fence}"
    return "".join(_inline(child) for child in node.children)


def _normalize_inline(node: _Node) -> str:
    value = _inline(node)
    value = re.sub(r"[ \t\r\f\v]+", " ", value)
    value = re.sub(r" *\n *", "\n", value)
    return value.strip()


def _single_line(node: _Node) -> str:
    return re.sub(r"\s+", " ", _normalize_inline(node)).strip()


def html_to_markdown(html: str) -> str:
    """Port of the browser's HTML-to-Markdown extraction rules."""
    parser = _TreeParser()
    parser.feed(html)
    parser.close()
    root = parser.root
    _prune(root)

    candidates = _find_nodes(
        root,
        lambda node: node.tag in {"main", "article"}
        or node.attrs.get("role", "").lower() == "main",
    )
    body = next((node for node in _find_nodes(root, lambda n: n.tag == "body")), None)
    if candidates:
        candidates.sort(key=lambda node: len(_text_content(node)), reverse=True)
        root = candidates[0]
    elif body is not None:
        root = body

    blocks: List[str] = []

    def push(value: str) -> None:
        text = re.sub(r"\n{3,}", "\n\n", value).strip()
        if text:
            blocks.append(text)

    def walk(node: _Node) -> None:
        if node.tag is None:
            text = _normalize_inline(node)
            if text:
                push(text)
            return
        tag = node.tag
        if tag in SKIPPED_TAGS:
            return
        if tag == "pre":
            raw = _text_content(node).replace("\r\n", "\n").strip("\n")
            if not raw:
                return
            language = ""
            code_node = next((child for child in _descendants(node, "code")), None)
            if code_node is not None:
                match = re.search(r"(?:^|\s)language-([\w-]+)", code_node.attrs.get("class", ""))
                if match:
                    language = match.group(1)
            fence = "~~~~" if "```" in raw else "```"
            push(f"{fence}{language}\n{raw}\n{fence}")
            return
        if re.match(r"h[1-6]$", tag):
            text = _single_line(node)
            if text:
                push(f"{'#' * int(tag[1])} {text}")
            return
        if tag == "li":
            text = _single_line(node)
            if text:
                push(f"- {text}")
            return
        if tag == "blockquote":
            text = _normalize_inline(node)
            if text:
                push("\n".join(f"> {line}" for line in text.splitlines()))
            return
        if tag == "table":
            for row in _descendants(node, "tr"):
                cells = [
                    _single_line(cell)
                    for cell in row.children
                    if cell.tag in {"td", "th"}
                ]
                if any(cells):
                    push(" | ".join(cells))
            return
        if tag == "hr":
            push("---")
            return
        if tag in BLOCK_TAGS:
            has_block_child = any(
                child.tag in BLOCK_TAGS for child in node.children
            )
            if has_block_child:
                for child in node.children:
                    walk(child)
            else:
                text = _single_line(node)
                if text:
                    push(text)
            return
        for child in node.children:
            walk(child)

    for child in root.children:
        walk(child)
    return "\n\n".join(blocks)


def load_documents(urls: Sequence[str]) -> List[Dict[str, str]]:
    docs: List[Dict[str, str]] = []
    for index, url in enumerate(urls):
        print(f"fetch {index + 1}/{len(urls)}: {url}", flush=True)
        data, content_type = _fetch_url(url)
        raw = _decode_document(data, content_type)
        kind = detect_document_kind(url, content_type, raw)
        if kind is None:
            raise SystemExit(
                f"{url} is not a text, Markdown, or HTML document "
                f"(content type: {content_type or 'unknown'})."
            )
        if kind == "html":
            text = html_to_markdown(raw)
            if not text.strip():
                raise SystemExit(
                    f"{url} has no readable text. It may be rendered by JavaScript "
                    "or block automated clients."
                )
        else:
            text = strip_front_matter(raw)
            if not text.strip():
                raise SystemExit(f"{url} is empty.")
        docs.append({"url": url, "kind": kind, "text": text})
        print(f"  {kind}, {len(text):,} chars", flush=True)
    return docs


# ---------------------------------------------------------------------------
# Tokenizer and custom-doc examples
# ---------------------------------------------------------------------------


def _local_model_dir() -> Path:
    return SCRIPT_DIR.parent / "apps" / "only-gates-ai" / "public" / "models" / "tiny-gpt-solid2"


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_tokenizer_model(args: argparse.Namespace) -> Dict[str, Any]:
    if args.tokenizer is not None:
        tokenizer_model = _read_json(Path(args.tokenizer))
        source = str(args.tokenizer)
    else:
        local = _local_model_dir() / "tokenizer.json"
        if args.tokenizer_url == DEFAULT_TOKENIZER_URL and local.exists():
            tokenizer_model = _read_json(local)
            source = str(local)
        else:
            print(f"fetch tokenizer: {args.tokenizer_url}", flush=True)
            data, content_type = _fetch_url(args.tokenizer_url)
            tokenizer_model = json.loads(_decode_document(data, content_type))
            source = args.tokenizer_url
    if not isinstance(tokenizer_model, dict):
        raise SystemExit("tokenizer.json must contain a JSON object")
    for key in ("version", "specialTokens", "vocab", "merges"):
        if key not in tokenizer_model:
            raise SystemExit(f"tokenizer.json is missing {key!r}")
    vocab = tokenizer_model.get("vocab", [])
    if len(vocab) != args.vocab:
        raise SystemExit(
            f"tokenizer vocabulary has {len(vocab):,} tokens but --vocab is "
            f"{args.vocab:,}; use the deployed 4096-token tokenizer."
        )
    tokenizer = base.BpeTokenizer(tokenizer_model)
    for special in (SPECIAL_USER, SPECIAL_ASSISTANT, SPECIAL_END):
        if tokenizer.token_to_id.get(special) is None:
            raise SystemExit(f"tokenizer is missing the {special} special token")
    print(f"tokenizer: {len(vocab):,} tokens from {source}", flush=True)
    return tokenizer_model


def _special_ids(tokenizer: base.BpeTokenizer) -> Dict[str, int]:
    return {
        "user": tokenizer.token_to_id[SPECIAL_USER],
        "assistant": tokenizer.token_to_id[SPECIAL_ASSISTANT],
        "end": tokenizer.token_to_id[SPECIAL_END],
    }


def strip_markdown(value: str) -> str:
    value = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", value)
    value = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", value)
    value = re.sub(r"[*_`~]", "", value)
    return re.sub(r"\s+", " ", value).strip()


def clean_markdown_links(value: str) -> str:
    value = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", value)
    return re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", value)


def extract_sections(doc: Mapping[str, str]) -> List[Dict[str, str]]:
    text = strip_front_matter(doc["text"])
    sections: List[Dict[str, str]] = []
    title: Optional[str] = None
    body: List[str] = []
    fence: Optional[str] = None

    def flush() -> None:
        if title is None:
            return
        value = "\n".join(body).strip()
        if len(value) >= 20:
            sections.append({"title": title, "body": value})

    for line in text.splitlines():
        trimmed = line.lstrip()
        fence_match = re.match(r"^(```+|~~~+)", trimmed)
        if fence_match:
            if fence is None:
                fence = fence_match.group(1)
            elif trimmed.startswith(fence):
                fence = None
        if fence is None:
            heading = re.match(r"^#{1,6}[ \t]+(.+?)[ \t]*#*[ \t]*$", line)
            if heading:
                flush()
                title = strip_markdown(heading.group(1))
                body = []
                continue
        if title is not None:
            body.append(line)
    flush()

    if not sections:
        text = strip_front_matter(doc["text"]).strip()
        if not text:
            return []
        first = next(
            (strip_markdown(line) for line in text.splitlines() if strip_markdown(line)),
            "Overview",
        )
        return [{"title": first[:80], "body": text}]
    return sections


def build_qa_pairs(docs: Sequence[Mapping[str, str]]) -> List[Dict[str, str]]:
    pairs: List[Dict[str, str]] = []
    seen = set()
    for doc in docs:
        for section in extract_sections(doc):
            title = re.sub(r"[.!?]+$", "", section["title"]).strip()
            if len(title) < 2:
                continue
            already_question = bool(
                re.search(r"\?$", section["title"])
                or re.match(
                    r"^(?:what|how|why|when|where|which|who|can|does|do|is|are|should|will)\b",
                    title,
                    re.IGNORECASE,
                )
            )
            questions = (
                [f"{title}?"]
                if already_question
                else [
                    f"What is {title}?",
                    f"How do I use {title}?",
                    f"Explain {title}.",
                ]
            )
            answer = clean_markdown_links(section["body"])
            for question in questions:
                key = question.lower()
                if key in seen:
                    continue
                seen.add(key)
                pairs.append({"question": question, "answer": answer})
    return pairs


def extract_prompts(
    docs: Sequence[Mapping[str, str]],
    pairs: Sequence[Mapping[str, str]],
    assistant: bool,
) -> List[str]:
    prompts: List[str] = []
    seen = set()

    def add(value: str) -> None:
        prompt = strip_markdown(value)[:120].strip()
        if len(prompt) < 4 or prompt in seen:
            return
        seen.add(prompt)
        prompts.append(prompt)

    if assistant:
        seen_answers = set()
        for pair in pairs:
            answer = pair["answer"]
            if answer in seen_answers:
                continue
            seen_answers.add(answer)
            add(pair["question"])
            if len(prompts) >= MAX_PROMPTS:
                break
        return prompts

    for doc in docs:
        for heading in re.finditer(
            r"^#{1,6}[ \t]+(.+?)[ \t]*#*[ \t]*$",
            doc["text"],
            re.MULTILINE,
        ):
            add(heading.group(1))
            if len(prompts) >= MAX_PROMPTS:
                return prompts
    if not prompts:
        for doc in docs:
            first = next(
                (line.strip() for line in doc["text"].splitlines() if line.strip()),
                "",
            )
            if first:
                add(first)
            if len(prompts) >= MAX_PROMPTS:
                break
    return prompts


def encode_qa_pair(
    tokenizer: base.BpeTokenizer,
    pair: Mapping[str, str],
    ids: Mapping[str, int],
) -> List[int]:
    tokens = [ids["user"]]
    tokens.extend(tokenizer.encode_ordinary(pair["question"]))
    tokens.append(ids["assistant"])
    tokens.extend(tokenizer.encode_ordinary(pair["answer"]))
    tokens.append(ids["end"])
    return tokens


def build_qa_stream(
    tokenizer: base.BpeTokenizer,
    pairs: Sequence[Mapping[str, str]],
    context: int,
) -> Tuple[np.ndarray, np.ndarray]:
    ids = _special_ids(tokenizer)
    # Hold out whole answer groups. Splitting individual question variants
    # leaks the same answer into both training and validation.
    answer_order: List[str] = []
    by_answer: Dict[str, List[Dict[str, str]]] = {}
    for pair in pairs:
        answer = pair["answer"]
        if answer not in by_answer:
            by_answer[answer] = []
            answer_order.append(answer)
        by_answer[answer].append(dict(pair))
    validation_answers = {
        answer
        for index, answer in enumerate(answer_order)
        if (index + 1) % QA_VALIDATION_EVERY == 0
    }
    validation = [pair for pair in pairs if pair["answer"] in validation_answers]
    training = [pair for pair in pairs if pair["answer"] not in validation_answers]
    source = training if training else list(pairs)
    training_tokens: List[int] = []
    for _ in range(QA_REPEATS):
        for pair in source:
            training_tokens.extend(encode_qa_pair(tokenizer, pair, ids))
    validation_tokens: List[int] = []
    for pair in validation:
        validation_tokens.extend(encode_qa_pair(tokenizer, pair, ids))
    minimum = context + 2
    if len(training_tokens) < minimum:
        raise SystemExit(
            f"The documentation produced only {len(training_tokens):,} training tokens, "
            f"but this model needs at least {minimum:,}. Add more documentation."
        )
    return (
        np.asarray(training_tokens, dtype=np.int32),
        np.asarray(validation_tokens, dtype=np.int32),
    )


def build_docs_stream(
    tokenizer: base.BpeTokenizer,
    docs: Sequence[Mapping[str, str]],
    context: int,
) -> Tuple[np.ndarray, np.ndarray]:
    end_id = tokenizer.token_to_id[SPECIAL_END]
    chunks: List[str] = []
    for doc in docs:
        for paragraph in re.split(r"\n\s*\n", doc["text"]):
            text = paragraph.strip()
            if text:
                chunks.append(text)
    if not chunks:
        raise SystemExit("The documents have no text to train on.")
    tokens: List[int] = []
    for chunk in chunks:
        tokens.extend(tokenizer.encode_ordinary(chunk))
        tokens.append(end_id)
    minimum = context + 2
    if len(tokens) < minimum:
        raise SystemExit(
            f"The documents are too short: {len(tokens):,} tokens, but this model "
            f"needs at least {minimum:,}. Add more Markdown files or content."
        )
    split = int(len(tokens) * (1 - VALIDATION_FRACTION))
    if len(tokens) - split < minimum:
        split = max(minimum, len(tokens) - minimum)
    split = min(split, len(tokens))
    return (
        np.asarray(tokens[:split], dtype=np.int32),
        np.asarray(tokens[split:], dtype=np.int32),
    )


def make_corpus(
    tokenizer: base.BpeTokenizer,
    docs: Sequence[Mapping[str, str]],
    context: int,
    assistant: bool,
) -> Tuple[np.ndarray, np.ndarray, List[str], int]:
    pairs = build_qa_pairs(docs) if assistant else []
    if assistant and not pairs:
        raise SystemExit(
            "Assistant (Q&A) mode needs Markdown headings to build questions. "
            "Use --no-assistant for plain-text training, or add heading-based documents."
        )
    if assistant:
        training, validation = build_qa_stream(tokenizer, pairs, context)
    else:
        training, validation = build_docs_stream(tokenizer, docs, context)
    prompts = extract_prompts(docs, pairs, assistant)
    return training, validation, prompts, len(pairs)


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------


def make_config(args: argparse.Namespace) -> Dict[str, int]:
    config = {
        "vocabSize": args.vocab,
        "context": args.context,
        "dim": args.dim,
        "layers": args.layers,
        "heads": args.heads,
        "mlp": args.mlp,
    }
    if config["dim"] % config["heads"] != 0:
        raise SystemExit("--dim must be divisible by --heads")
    return config


def unpack_weights(model: torch.nn.Module, packed: np.ndarray) -> None:
    """Fill a TinyGPT model from packed ``tiny_gpt_create`` float32 weights."""
    vocab, context, dim, layers, mlp = (
        model.vocab_size, model.context, model.dim, model.layers_n, model.mlp_n
    )
    offset = 0

    def take(count: int) -> torch.Tensor:
        nonlocal offset
        chunk = packed[offset : offset + count]
        if len(chunk) != count:
            raise ValueError("packed weights are truncated")
        offset += count
        return torch.from_numpy(chunk.copy())

    with torch.no_grad():
        model.token_emb.weight.copy_(take(vocab * dim).view(vocab, dim))
        model.pos_emb.weight.copy_(take(context * dim).view(context, dim))
        model.g_final.gain.copy_(take(dim))
        for block in model.blocks:
            block.g1.gain.copy_(take(dim))
            block.wqkv.weight.copy_(take(dim * dim * 3).view(dim, dim * 3).T)
            block.wo.weight.copy_(take(dim * dim).view(dim, dim).T)
            block.g2.gain.copy_(take(dim))
            block.w1.weight.copy_(take(dim * mlp).view(dim, mlp).T)
            block.w2.weight.copy_(take(mlp * dim).view(mlp, dim).T)
    if offset != packed.size:
        raise ValueError("packed weights have trailing values")


def warm_start_from_bundled(model: torch.nn.Module, config: Mapping[str, int], path: Path) -> None:
    checkpoint_path = path / "checkpoint.json"
    weights_path = path / "weights.bin"
    if not checkpoint_path.exists() or not weights_path.exists():
        raise SystemExit(
            f"Warm start needs checkpoint.json and weights.bin in {path}. "
            "Clone the repository or pass --warm-start-dir."
        )
    checkpoint = _read_json(checkpoint_path)
    bundled = checkpoint.get("config", {})
    for key in ("vocabSize", "context", "dim", "layers", "heads", "mlp"):
        if int(bundled.get(key, -1)) != int(config[key]):
            raise SystemExit(
                f"Warm start config mismatch for {key}: bundled={bundled.get(key)!r} "
                f"current={config[key]!r}. Use --fresh without --warm-start."
            )
    packed = np.fromfile(weights_path, dtype="<f4")
    expected = model_bundle.parameter_count(config)
    if packed.size != expected:
        raise SystemExit(
            f"bundled weights.bin has {packed.size:,} values but the config needs "
            f"{expected:,}."
        )
    unpack_weights(model, packed)
    print(f"warm-started from {weights_path}", flush=True)


def model_state_cpu(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def snapshot_state(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    total_steps: int,
    best_step: int,
    best_loss: float,
    best_model: Optional[Dict[str, torch.Tensor]],
    history: Sequence[Mapping[str, Any]],
    config: Mapping[str, int],
    seed: int,
    batch: int,
    lr: float,
    dropout: float,
    mode: str,
    urls: Sequence[str],
    url_hash: str,
    corpus_hash: str,
    tokenizer_model: Mapping[str, Any],
    token_file: str,
    prompts: Sequence[str],
    elapsed_seconds: float,
) -> Dict[str, Any]:
    return {
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        "config": dict(config),
        "seed": seed,
        "steps": total_steps,
        "batch": batch,
        "lr": lr,
        "dropout": dropout,
        "mode": mode,
        "source_urls": list(urls),
        "url_hash": url_hash,
        "corpus_hash": corpus_hash,
        "tokenizer_model": tokenizer_model,
        "prompts": list(prompts),
        "step": step,
        "model": model_state_cpu(model),
        "optimizer": optimizer.state_dict(),
        "best_step": best_step,
        "best_loss": best_loss,
        "best_model": best_model,
        "history": list(history),
        "token_file": token_file,
        "elapsed_seconds": elapsed_seconds,
        "rng": base.capture_rng(),
    }


def save_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    base.atomic_torch_save(path, dict(payload))


def load_checkpoint(path: Path) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise SystemExit(f"{path} is not a {CHECKPOINT_FORMAT} checkpoint")
    if int(payload.get("version", -1)) != CHECKPOINT_VERSION:
        raise SystemExit(f"{path} has unsupported checkpoint version {payload.get('version')}")
    return payload


def load_token_cache(
    out_dir: Path, checkpoint: Mapping[str, Any]
) -> Tuple[np.ndarray, np.ndarray]:
    token_file = str(checkpoint.get("token_file", "tokens.npz"))
    path = out_dir / token_file
    if path.exists():
        data = np.load(path)
        return (
            np.asarray(data["train"], dtype=np.int32),
            np.asarray(data["validation"], dtype=np.int32),
        )
    # Backward compatibility with checkpoints that embedded the arrays.
    if "train_tokens" in checkpoint and "validation_tokens" in checkpoint:
        return (
            np.asarray(checkpoint["train_tokens"], dtype=np.int32),
            np.asarray(checkpoint["validation_tokens"], dtype=np.int32),
        )
    raise SystemExit(f"missing token cache {path}; run with --fresh to rebuild it")


def token_windows(tokens: np.ndarray, context: int) -> torch.Tensor:
    if len(tokens) < context + 1:
        return torch.empty((0, context + 1), dtype=torch.long)
    array = torch.tensor(tokens, dtype=torch.long)
    return array.unfold(0, context + 1, 1)


def eval_loss(
    model: torch.nn.Module,
    windows: torch.Tensor,
    context: int,
    batch: int,
    device: torch.device,
    max_batches: int = 64,
) -> float:
    if windows.numel() == 0:
        return float("nan")
    count = windows.size(0)
    batches = min(max_batches, max(1, math.ceil(count / batch)))
    indices = np.linspace(0, count - 1, batches * batch).astype(np.int64)
    model.eval()
    total = 0.0
    seen = 0
    with torch.no_grad():
        for start in range(0, len(indices), batch):
            chosen = torch.from_numpy(indices[start : start + batch])
            chunk = windows[chosen].to(device)
            _, loss = model(chunk[:, :-1], chunk[:, 1:])
            if math.isfinite(float(loss)):
                total += float(loss) * len(chosen)
                seen += len(chosen)
    model.train()
    return total / seen if seen else float("nan")


# ---------------------------------------------------------------------------
# Samples and export
# ---------------------------------------------------------------------------


def generate_samples(
    model: torch.nn.Module,
    tokenizer: base.BpeTokenizer,
    prompts: Sequence[str],
    assistant: bool,
    device: torch.device,
    max_new: int = 120,
) -> List[Dict[str, str]]:
    if not prompts:
        return []
    ids = _special_ids(tokenizer)
    skip_ids = {
        tokenizer.token_to_id[special]
        for special in tokenizer.special_tokens
        if tokenizer.token_to_id.get(special) is not None
    }
    samples: List[Dict[str, str]] = []
    model.eval()
    for prompt in prompts[:MAX_PROMPTS]:
        if assistant:
            prompt_ids = [ids["user"], *tokenizer.encode_ordinary(prompt), ids["assistant"]]
        else:
            prompt_ids = tokenizer.encode_ordinary(prompt)
        if not prompt_ids:
            continue
        tokens = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        generated = model.generate(
            tokens,
            max_new=max_new,
            temperature=0.7,
            top_k=40,
            eos_id=ids["end"],
        )[0].tolist()
        pieces: List[str] = []
        for token_id in generated[len(prompt_ids) :]:
            if token_id == ids["end"]:
                break
            if token_id in skip_ids:
                continue
            if 0 <= token_id < len(tokenizer.vocab):
                pieces.append(tokenizer.vocab[token_id])
        decoded = bytearray()
        for char in "".join(pieces):
            byte = base.BYTE_DECODER.get(char)
            if byte is not None:
                decoded.append(byte)
        text = decoded.decode("utf-8", errors="replace")
        samples.append({"prompt": prompt, "text": text})
        print(f"sample> {prompt} => {text[:140]}", flush=True)
    return samples


def export_bundle(
    *,
    out_dir: Path,
    bundle_path: Path,
    config: Mapping[str, int],
    tokenizer_model: Mapping[str, Any],
    model: torch.nn.Module,
    prompts: Sequence[str],
    samples: Sequence[Mapping[str, str]],
    assistant: bool,
    urls: Sequence[str],
    corpus_hash: str,
    step: int,
    best_step: int,
    best_loss: float,
    elapsed_seconds: float,
    history: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
) -> Path:
    packed = base.pack_weights(model)
    base.verify_packed(model, packed, {**config})
    weights = packed.numpy().astype("<f4", copy=False).tobytes()
    source = "Custom documents fetched from: " + ", ".join(urls[:8])
    if len(urls) > 8:
        source += f" and {len(urls) - 8} more"
    source += "."
    license_text = (
        "Upstream documents retain their own licenses; model weights trained "
        "by this notebook with only-gates-ai."
    )
    training = {
        "steps": int(step),
        "batch": args.batch,
        "learningRate": args.lr,
        "dropout": args.dropout,
        "seed": args.seed,
        "bestStep": int(best_step),
        "bestLoss": best_loss if math.isfinite(best_loss) else None,
        "elapsedMs": int(max(0.0, elapsed_seconds) * 1000),
        "corpusHash": corpus_hash,
        "sourceUrls": list(urls),
    }
    written = model_bundle.write_model_bundle(
        out_dir,
        bundle_path,
        config=config,
        tokenizer_model=tokenizer_model,
        weights=weights,
        name=args.name,
        source=source,
        license=license_text,
        prompts=prompts,
        samples=samples,
        training=training,
        history=history,
        assistant=assistant,
    )
    print(f"\nwrote {written} ({written.stat().st_size:,} bytes)", flush=True)
    print(
        "Import it in Integer GPT -> Custom Docs (BPE) -> Import model (.zip).",
        flush=True,
    )
    return written


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--out", "--output-dir", dest="out", type=Path, default=Path("tiny-gpt-custom-out"))
    parser.add_argument("--urls-file", type=Path, default=None, help="Text file with one URL per line")
    parser.add_argument("--url", action="append", default=[], help="Document URL (repeatable)")
    parser.add_argument("--tokenizer", type=Path, default=None, help="Local tokenizer.json override")
    parser.add_argument("--tokenizer-url", default=DEFAULT_TOKENIZER_URL)
    parser.add_argument("--assistant", action="store_true", help="Build Q&A examples from Markdown headings")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-minutes", type=float, default=0.0)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--val-every", type=int, default=100)
    parser.add_argument("--vocab", type=int, default=4096)
    parser.add_argument("--context", "--ctx", dest="context", type=int, default=256)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--mlp", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--name", default="tiny-gpt-custom-colab")
    parser.add_argument("--bundle-out", type=Path, default=None)
    parser.add_argument("--warm-start", action="store_true")
    parser.add_argument("--warm-start-dir", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument(
        "--export-final",
        action="store_true",
        help=(
            "Export the final training weights instead of the best-validation "
            "checkpoint. Small Q&A models often overfit their instruction data; "
            "the final weights answer the trained questions much better."
        ),
    )
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.context, args.dim, args.layers, args.heads, args.mlp = 64, 32, 1, 2, 64
        args.steps, args.batch = 2, 2
        args.val_every, args.save_every = 1, 1
        args.warm_start = False
        print("SMOKE mode: tiny config and two steps", flush=True)
    if args.steps < 1:
        raise SystemExit("--steps must be at least 1")
    if args.batch < 1:
        raise SystemExit("--batch must be at least 1")
    if args.max_minutes < 0:
        raise SystemExit("--max-minutes must be non-negative")

    signal.signal(signal.SIGINT, base.request_stop)
    signal.signal(signal.SIGTERM, base.request_stop)

    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.out / "latest.pt"
    bundle_path = args.bundle_out or (args.out / "only-gates-custom-bundle.zip")
    config = make_config(args)
    assistant = bool(args.assistant)
    mode = "qa" if assistant else "docs"

    checkpoint: Optional[Dict[str, Any]] = None
    if not args.fresh and (args.resume or args.export_only) and checkpoint_path.exists():
        checkpoint = load_checkpoint(checkpoint_path)
        print(f"loaded checkpoint {checkpoint_path} at step {checkpoint.get('step', 0)}", flush=True)
    elif args.export_only:
        raise SystemExit("--export-only needs an existing latest.pt; run training first")

    if args.export_only and checkpoint is not None:
        config = dict(checkpoint["config"])
        assistant = checkpoint.get("mode") == "qa"
        mode = "qa" if assistant else "docs"
        args.batch = int(checkpoint.get("batch", args.batch))
        args.lr = float(checkpoint.get("lr", args.lr))
        args.dropout = float(checkpoint.get("dropout", args.dropout))
        args.seed = int(checkpoint.get("seed", args.seed))
        tokenizer_model = checkpoint["tokenizer_model"]
        tokenizer = base.BpeTokenizer(tokenizer_model)
        dropout = args.dropout
        model = base.TinyGPT(
            config["vocabSize"], config["context"], config["dim"],
            config["layers"], config["heads"], config["mlp"], dropout,
        ).to(torch.device(args.device))
        if args.export_final:
            state = checkpoint["model"]
        else:
            state = checkpoint.get("best_model") or checkpoint["model"]
        model.load_state_dict(state)
        prompts = checkpoint.get("prompts", [])
        samples = generate_samples(model, tokenizer, prompts, assistant, torch.device(args.device))
        export_bundle(
            out_dir=args.out,
            bundle_path=bundle_path,
            config=config,
            tokenizer_model=tokenizer_model,
            model=model,
            prompts=prompts,
            samples=samples,
            assistant=assistant,
            urls=checkpoint.get("source_urls", []),
            corpus_hash=checkpoint.get("corpus_hash", ""),
            step=int(checkpoint.get("step", 0)),
            best_step=(
                int(checkpoint.get("step", 0))
                if args.export_final
                else int(checkpoint.get("best_step", 0))
            ),
            best_loss=(
                float("nan")
                if args.export_final
                else float(checkpoint.get("best_loss", float("inf")))
            ),
            elapsed_seconds=float(checkpoint.get("elapsed_seconds", 0.0)),
            history=checkpoint.get("history", []),
            args=args,
        )
        return

    urls = parse_urls(args.urls_file, args.url)
    url_hash = base.sha256_json(list(urls))

    if checkpoint is not None:
        if checkpoint.get("mode") != mode:
            raise SystemExit("checkpoint mode differs; use --fresh to change --assistant")
        if checkpoint.get("url_hash") != url_hash:
            raise SystemExit("the URL list changed since the checkpoint; use --fresh to start a new run")
        if checkpoint.get("config") != config:
            raise SystemExit("checkpoint config differs; use --fresh to change the model config")
        if int(checkpoint.get("seed", args.seed)) != args.seed:
            raise SystemExit("checkpoint seed differs; use --fresh to change --seed")
        tokenizer_model = checkpoint["tokenizer_model"]
        tokenizer = base.BpeTokenizer(tokenizer_model)
        train_tokens, validation_tokens = load_token_cache(args.out, checkpoint)
        if not (args.out / "tokens.npz").exists():
            np.savez_compressed(
                args.out / "tokens.npz",
                train=train_tokens,
                validation=validation_tokens,
            )
        prompts = list(checkpoint.get("prompts", []))
        corpus_hash = str(checkpoint.get("corpus_hash", ""))
    else:
        tokenizer_model = load_tokenizer_model(args)
        tokenizer = base.BpeTokenizer(tokenizer_model)
        docs = load_documents(urls)
        train_tokens, validation_tokens, prompts, qa_pairs = make_corpus(
            tokenizer, docs, config["context"], assistant
        )
        token_stream_hash = hashlib.sha256(
            train_tokens.tobytes() + validation_tokens.tobytes()
        ).hexdigest()
        corpus_hash = base.sha256_json(
            {
                "version": TOKEN_SCHEME_VERSION,
                "mode": mode,
                "urls": list(urls),
                "tokenizer": base.sha256_json(tokenizer_model),
                "stream": token_stream_hash,
            }
        )
        print(
            f"corpus: {len(train_tokens):,} training tokens, "
            f"{len(validation_tokens):,} validation tokens, "
            f"{qa_pairs:,} Q&A pairs, {len(prompts)} prompts",
            flush=True,
        )

    if args.inspect_only:
        print("inspect-only: no training started", flush=True)
        return

    if checkpoint is None:
        np.savez_compressed(
            args.out / "tokens.npz",
            train=train_tokens,
            validation=validation_tokens,
        )

    device = torch.device(args.device)
    model = base.TinyGPT(
        config["vocabSize"], config["context"], config["dim"],
        config["layers"], config["heads"], config["mlp"], args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01
    )

    start_step = 0
    total_steps = args.steps
    history: List[Dict[str, Any]] = []
    best_step = 0
    best_loss = float("inf")
    best_model: Optional[Dict[str, torch.Tensor]] = None
    elapsed_base = 0.0

    if checkpoint is not None:
        total_steps = max(int(checkpoint.get("steps", args.steps)), args.steps)
        start_step = int(checkpoint.get("step", 0))
        history = list(checkpoint.get("history", []))
        best_step = int(checkpoint.get("best_step", 0))
        best_loss = float(checkpoint.get("best_loss", float("inf")))
        best_model = checkpoint.get("best_model")
        elapsed_base = float(checkpoint.get("elapsed_seconds", 0.0))
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        base.restore_rng(checkpoint.get("rng"))
        print(f"resuming at step {start_step}/{total_steps}", flush=True)
    else:
        if args.warm_start:
            warm_dir = args.warm_start_dir or _local_model_dir()
            warm_start_from_bundled(model, config, warm_dir)

    windows = token_windows(train_tokens, config["context"])
    validation_windows = token_windows(validation_tokens, config["context"])
    if windows.size(0) == 0:
        raise SystemExit("The training stream is too short for the model context.")

    session_start = time.monotonic()
    deadline = session_start + args.max_minutes * 60.0 if args.max_minutes > 0 else None
    stopped = False
    last_step = start_step
    smooth: Optional[float] = None
    if history:
        smooth = history[-1].get("smoothedLoss")

    def elapsed_seconds() -> float:
        return elapsed_base + (time.monotonic() - session_start)

    def save(step: int) -> None:
        save_checkpoint(
            checkpoint_path,
            snapshot_state(
                model,
                optimizer,
                step=step,
                total_steps=total_steps,
                best_step=best_step,
                best_loss=best_loss,
                best_model=best_model,
                history=history,
                config=config,
                seed=args.seed,
                batch=args.batch,
                lr=args.lr,
                dropout=args.dropout,
                mode=mode,
                urls=urls,
                url_hash=url_hash,
                corpus_hash=corpus_hash,
                tokenizer_model=tokenizer_model,
                token_file="tokens.npz",
                prompts=prompts,
                elapsed_seconds=elapsed_seconds(),
            ),
        )

    print(
        f"train {start_step + 1}..{total_steps} batch={args.batch} lr={args.lr} "
        f"ctx={config['context']} params={model_bundle.parameter_count(config):,} "
        f"device={device}",
        flush=True,
    )
    model.train()
    for step in range(start_step + 1, total_steps + 1):
        if base.STOP_REQUESTED or (deadline is not None and time.monotonic() >= deadline):
            stopped = True
            break
        lr = args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / total_steps)))
        for group in optimizer.param_groups:
            group["lr"] = lr
        indices = torch.randint(0, windows.size(0), (args.batch,))
        chunk = windows[indices].to(device)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(chunk[:, :-1], chunk[:, 1:])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_value = float(loss.detach())
        if not math.isfinite(loss_value):
            raise SystemExit(f"non-finite loss at step {step}")
        smooth = loss_value if smooth is None else smooth * 0.95 + loss_value * 0.05
        val_loss: Optional[float] = None
        if step % max(1, args.val_every) == 0 or step == total_steps:
            value = eval_loss(model, validation_windows, config["context"], args.batch, device)
            if math.isfinite(value):
                val_loss = value
                if value < best_loss:
                    best_loss = value
                    best_step = step
                    best_model = model_state_cpu(model)
        point = {
            "step": step,
            "loss": loss_value,
            "smoothedLoss": smooth,
            "lr": lr,
            "elapsedMs": elapsed_seconds() * 1000.0,
        }
        if val_loss is not None:
            point["valLoss"] = val_loss
        history.append(point)
        last_step = step
        if step % 10 == 0 or step == total_steps:
            print(
                f"step {step}/{total_steps} loss {loss_value:.4f} "
                f"smoothed {smooth:.4f}"
                + (f" val {val_loss:.4f}" if val_loss is not None else "")
                + f" lr {lr:.2e} {elapsed_seconds():.1f}s",
                flush=True,
            )
        if args.save_every > 0 and step % args.save_every == 0 and step < total_steps:
            save(step)

    save(last_step)
    if stopped:
        print(
            f"training stopped at step {last_step}/{total_steps}; "
            f"rerun with --resume to continue",
            flush=True,
        )

    if best_model is not None and not args.export_final:
        model.load_state_dict(best_model)
    samples = generate_samples(model, tokenizer, prompts, assistant, device)
    export_bundle(
        out_dir=args.out,
        bundle_path=bundle_path,
        config=config,
        tokenizer_model=tokenizer_model,
        model=model,
        prompts=prompts,
        samples=samples,
        assistant=assistant,
        urls=urls,
        corpus_hash=corpus_hash,
        step=last_step,
        best_step=last_step if args.export_final else best_step,
        best_loss=float("nan") if args.export_final else best_loss,
        elapsed_seconds=elapsed_seconds(),
        history=history,
        args=args,
    )


if __name__ == "__main__":
    main()
