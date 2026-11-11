#!/usr/bin/env python3
"""
Train a ~1M-parameter integer-friendly Tiny GPT on Solid 2 docs + curated Q&A,
exporting weights compatible with only-gates-ai's C/WASM integer inference.

Designed to run on a fresh Colab VM via the Termux/Colab CLI:
  colab run --gpu T4 --timeout 7200 scripts/train_tiny_gpt_solid2_colab.py -- \\
    --out /content/tiny-gpt-solid2

Free sessions are short, so the trainer checkpoints full model, optimizer,
scheduler, RNG and data-position state to <out>/latest.pt. Rerun the same
command with --resume to continue; add --max-minutes to stop gracefully before
the session ends:
  python scripts/train_tiny_gpt_solid2_colab.py --out DIR --max-minutes 300
  python scripts/train_tiny_gpt_solid2_colab.py --out DIR --max-minutes 300 --resume

It is also runnable locally (needs: torch, numpy; optional: regex for best BPE).

What it does:
  1. Fetches pinned Solid 2.0 docs (solidjs/solid@next, blob-SHA verified).
  2. Loads the curated Solid 2 Q&A corpus (embedded + optional local override).
  3. Trains a byte-level BPE tokenizer (4096 vocab) and writes tokenizer.json
     in the format consumed by @only-gates-ai/core ByteLevelBpe.
  4. Trains a decoder-only transformer (pre-norm RMSNorm, GELU-tanh, learned
     positions, no biases, tied embeddings) on docs+Q&A chat examples.
  5. Exports weights.bin (float32, exact tiny_gpt_create packing order),
     checkpoint.json, and tokenizer.json.

Float math is used only for training; the browser/WASM path quantizes to
int8/Q16 integer inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import signal
import sys
import time
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Corpus (pinned Solid 2 docs) + supplemental (HackMD) + stories + embedded Q&A
# ---------------------------------------------------------------------------

DOCS_COMMIT = "8138dabec4fc6514a44e540f2dacf630d51a9a67"
DOCS = [
    ("01-reactivity-batching-effects.md", "8fa6483be18380a959daad26df11337d5e6b9201"),
    ("02-signals-derived-ownership.md", "f2865cfae544e3d8d0729dfe1b439b0f6590aebc"),
    ("03-control-flow.md", "38c0ec42a98ca3802ba00ba7ebaabdda9d713074"),
    ("04-stores.md", "09effefcecae24948a61c01b963413918d26e896"),
    ("05-async-data.md", "2a83aa834e675e28c4b12b2296841d76faf02787"),
    ("06-actions-optimistic.md", "3d15321ed570b38da5d65a2b8daa2f5219d9ed25"),
    ("07-dom.md", "bdd88a7c66a11e4731da7204a2d72c3ff607dcbb"),
    ("08-dev-diagnostics.md", "9f24bce00336f4586649679881467e9fd489cc6d"),
    ("09-typescript-jsx.md", "cc580c4edadd2aa4b83c61ee07f0b5a86b09cd9b"),
    ("10-server-functions.md", "c101c0b862686b1977b7925f600453d7647140ba"),
    ("11-server-components.md", "3d1104c5ea9c097bdeb63997b54b0cdf8530c91d"),
    ("12-ssr-http.md", "0b829db0ca3d87843f24091f71a3bcb197d395b2"),
    ("MIGRATION.md", "7a943c8d423cb33e9b3d8c2feb59f37cb318137d"),
    ("README.md", "d36e8ded6a6f341a7c168e9613fd02ee4429255c"),
]

# Supplemental Solid 2 material (fetched when local snapshot not present). HackMD
# `/download` returns markdown. These are proposal/notes-era; pinned official docs
# above remain authoritative for API details.
SUPPLEMENTAL_DOCS = [
    (
        "supplemental/solid-2-proposed-api-changes-hackmd.md",
        "https://hackmd.io/@0u1u3zEAQAO0iYWVAStEvw/SyXYy2swbg/download",
        "Solid 2.0 Proposed API Changes (HackMD, Ryan Carniato) - supplemental/proposal context",
    ),
    (
        "supplemental/ai-readiness-kitchen-sink-hackmd.md",
        "https://hackmd.io/Zs3fzRfOSku0lm8LgzD-BQ/download",
        "Solid 2 AI readiness / kitchen-sink findings (HackMD) - patterns and pitfalls",
    ),
]

SPECIAL_TOKENS = ["<|pad|>", "<|user|>", "<|assistant|>", "<|end|>", "<|facts|>"]
PAD_ID, USER_ID, ASSISTANT_ID, END_ID, FACTS_ID = 0, 1, 2, 3, 4

# Curated Q&A (compact). Keep in sync with data/solid2/qa.json / solid2-qa.ts.
QA_DATA: List[Dict] = json.loads(
    r'''[{"id":"effects-split","questions":["What are split effects?","What is a split effect in Solid?","Why are effects split in Solid 2?","Why did Solid 2 split createEffect?","Why does createEffect take two functions?","What does createEffect(compute, apply) mean?","How do split effects work?","What is the difference between compute and apply in createEffect?","Solid 2 createEffect signature","What is the new createEffect API?"],"answer":"In Solid 2, `createEffect` takes two functions. The compute function reads reactive values and returns a value; Solid records exactly what it read. The apply function receives that value and performs the side effect. Apply runs after all compute phases in the batch, so dependencies are known before any side effect runs. The apply function can return a cleanup function. This replaces the 1.x `on` helper and makes async and boundary behaviour predictable."},{"id":"effects-migrate","questions":["How do I migrate createEffect from Solid 1?","How do I rewrite an effect for Solid 2?","What is the new createEffect syntax?","How do I use createEffect in Solid 2?","Solid 1 effect to Solid 2","createEffect before and after Solid 2","How do I update my effects for Solid 2?","Convert a 1.x effect to a split effect"],"answer":"Move the reactive read into the first function and the side effect into the second.\n\n```js\n// Solid 1\ncreateEffect(() => {\n  el.title = name();\n});\n\n// Solid 2\ncreateEffect(\n  () => name(),\n  value => {\n    el.title = value;\n  },\n);\n```\n\nThe compute function should only read reactive values. Put DOM work, subscriptions, and other side effects in the apply function."},{"id":"effects-cleanup","questions":["How do I clean up an effect in Solid 2?","Where does onCleanup go in a split effect?","How do I return cleanup from createEffect?","Solid 2 effect cleanup","How do I cancel an interval in an effect?","How do I remove an event listener in an effect?","Does createEffect support cleanup?","How do I stop an effect from leaking?"],"answer":"Return a cleanup function from the apply phase. Solid calls it before the next apply and when the owner is disposed.\n\n```js\ncreateEffect(\n  () => name(),\n  value => {\n    const id = setInterval(() => console.log(value), 1000);\n    return () => clearInterval(id);\n  },\n);\n```\n\n`onCleanup` still exists for cleanup inside computations, but effect cleanup normally belongs on the apply side."},{"id":"effects-on-removed","questions":["What happened to on() in Solid 2?","Is the on helper still available?","How do I watch one specific signal in Solid 2?","How do I replace on() in Solid 2?","Solid 2 on() removed","How do I run an effect only when one signal changes?","How do I declare explicit dependencies in Solid 2?","Do I still need on()?"],"answer":"`on` is no longer needed because the compute phase is the explicit dependency declaration.\n\n```js\ncreateEffect(() => count(), value => {\n  console.log(value);\n});\n\n// Multiple dependencies\ncreateEffect(\n  () => [a(), b()],\n  ([a, b]) => console.log(a, b),\n);\n```\n\nTo skip the initial run, pass `{ defer: true }` as the third argument."},{"id":"effects-defer","questions":["How do I skip the first run of an effect in Solid 2?","How do I use defer with createEffect?","How do I run an effect only after the initial render?","Solid 2 effect should not run on mount","How do I avoid the initial effect run?","What replaced on(..., { defer: true })?","createEffect defer option"],"answer":"Pass `{ defer: true }` as the third argument:\n\n```js\ncreateEffect(\n  () => count(),\n  value => {\n    console.log(value);\n  },\n  { defer: true },\n);\n```\n\nThe apply phase is skipped on the initial run and starts after the dependency changes. This replaces `on(..., { defer: true })` from Solid 1."},{"id":"effects-prev","questions":["What happened to the initial value of createEffect?","How do I get the previous value in createEffect?","Solid 2 createEffect initial value","How do I use the prev argument in createEffect?","How do I set a default previous value?","Why is the second createEffect argument not an initial value?","createEffect prev parameter"],"answer":"The 1.x `initialValue` argument is gone. The compute function receives the previous value (`undefined` on the first run), and the apply function receives `(value, prev)`.\n\n```js\ncreateEffect(\n  (prev = 0) => count(),\n  (value, prev) => {\n    console.log(\"changed from\", prev, \"to\", value);\n  },\n);\n```\n\n`createMemo` also treats its second argument as options now, not an initial value."},{"id":"effects-createcomputed","questions":["What replaced createComputed?","Is createComputed still available in Solid 2?","How do I migrate createComputed?","CreateComputed removal in Solid 2","What is the alternative to createComputed?","How do I do derived write-back in Solid 2?","Solid 2 createComputed replacement"],"answer":"`createComputed` is removed. Pick a replacement based on intent:\n\n- Use `createMemo` for a read-only derived value.\n- Use split `createEffect(compute, apply)` for a side effect on change.\n- Use function-form `createSignal(fn)` or `createStore(fn)` for derived state that also has a setter.\n\nFor example, `createComputed(() => setDoubled(count() * 2))` becomes `const doubled = createMemo(() => count() * 2)`."},{"id":"effects-onmount","questions":["What replaced onMount?","Is onMount still available in Solid 2?","How do I run code after mount in Solid 2?","What is onSettled?","onMount vs onSettled","How do I migrate onMount?","How do I run something once after render in Solid 2?"],"answer":"`onMount` is replaced by `onSettled`. It runs when the current activity settles, such as after mount, can read reactive values, and can return a cleanup function.\n\n```js\nonSettled(() => {\n  measureLayout();\n  const onResize = () => measureLayout();\n  window.addEventListener(\"resize\", onResize);\n  return () => window.removeEventListener(\"resize\", onResize);\n});\n```\n\nFor a one-time read that should not track, use `untrack`."},{"id":"effects-error","questions":["How do I handle errors in createEffect?","How do I catch an async error in an effect?","Solid 2 effect error callback","What is the EffectBundle in Solid 2?","How do I use the createEffect error option?","How do I handle a failed fetch in an effect?","createEffect error handling"],"answer":"Pass an `EffectBundle` object as the second argument instead of a bare apply function:\n\n```js\ncreateEffect(\n  () => fetchData(id()),\n  {\n    effect: data => renderData(data),\n    error: (err, cleanup) => {\n      console.error(\"Fetch failed:\", err);\n      cleanup();\n    },\n  },\n);\n```\n\nThe `error` handler catches errors from async computations in the reactive graph. For component-level error UI, use an `Errored` boundary."},{"id":"effects-stores","questions":["How do I read a store in a split effect?","Why do I get a strict read warning with a store in an effect?","How do I use a store proxy in createEffect?","How do I deep track a store in an effect?","Should I use snapshot inside an effect?","Store in the createEffect compute phase","Solid 2 STRICT_READ_UNTRACKED"],"answer":"The apply phase runs untracked, so don't pass a store proxy through and read its properties there. Extract the values you need in compute and pass plain data:\n\n```js\ncreateEffect(\n  () => ({ name: store.user.name, age: store.user.age }),\n  value => sendAnalytics(value.name, value.age),\n);\n```\n\nFor deep tracking, `deep(store)` returns a plain snapshot and tracks every nested property. For a copy without subscribing, use `snapshot(store)`."},{"id":"batching-flush","questions":["Why does my signal not update immediately in Solid 2?","Why does a setter still read the old value?","What is flush in Solid 2?","How do I read the DOM right after setting state?","Signal updates feel delayed in Solid 2","How does microtask batching work?","How do I use flush(fn)?","Why is count() still old after setCount?"],"answer":"Solid 2 batches writes on a microtask. After `setCount(1)`, `count()` still returns the old value until the batch flushes. Call `flush()` to apply pending updates immediately:\n\n```js\nsetSubmitted(true);\nflush();\ninputRef.focus();\n```\n\n`flush(fn)` runs the callback in a synchronous flush scope and applies its writes before returning. Use `flush` sparingly: tests, imperative interop, and reading the DOM right after a write."},{"id":"batching-batch-removed","questions":["What happened to batch in Solid 2?","Is batch still available?","What replaces batch?","How do I batch updates in Solid 2?","Do I need to call batch?","Solid 2 batch removal"],"answer":"`batch` is removed because all writes are batched by default on a microtask. Just perform the writes; they apply together after the batch. If you need them applied synchronously, call `flush()` after the writes. `flush(fn)` applies writes made inside the callback before it returns."},{"id":"writes-owned-scope","questions":["Why does writing a signal in an effect throw in Solid 2?","Can I set a signal inside createEffect?","What is a reactive write in owned scope?","Solid 2 write inside memo error","What is the ownedWrite option?","Why does setting state in a component body throw?","How do I update a signal from an effect?"],"answer":"Writing to a signal or store inside a reactive scope (an effect, memo, or component body) throws in development. Move the write to an event handler or `onSettled`. `untrack` does not exempt a write: it stops tracking, not ownership.\n\nFor a rare internal signal that must be written in scope, opt in narrowly:\n\n```js\nconst [ref, setRef] = createSignal(null, { ownedWrite: true });\n```\n\nDo not use `ownedWrite` for application state. Derive the value with `createMemo` instead of writing it back."},{"id":"top-level-reads","questions":["What is the top-level reactive read warning in Solid 2?","Why does destructuring props warn in Solid 2?","How do I fix an untracked reactive read warning?","Solid 2 props destructuring warning","How do I read props once in Solid 2?","Component body signal read warning","Why does reading a signal in a Show callback warn?"],"answer":"In a component body, a top-level reactive read warns in development because it can capture a dependency accidentally. Read inside JSX or a reactive scope such as `createMemo` or `createEffect`, or make a one-time read explicit with `untrack`:\n\n```js\nconst title = untrack(() => props.title);\n```\n\nAvoid destructuring reactive props at the top level. Keep the props object and read `props.title` in JSX so the read stays tracked."},{"id":"untrack-once","questions":["What replaced the /*@once*/ JSX marker?","How do I read a value once in Solid 2 JSX?","Solid 2 once comment removed","How do I use untrack in Solid 2?","What is the replacement for /*@once*/?","How do I set an initial input value in Solid 2?","How do I opt out of reactivity for one read?"],"answer":"`/*@once*/` is no longer part of Solid's public JSX model. If a value should update, leave the read reactive in JSX. For an initial DOM value, use the platform default prop, such as `defaultValue={props.initialValue}`. If you truly need a one-time JavaScript read, use `untrack(() => props.value)` outside JSX."},{"id":"create-root-ownership","questions":["Is createRoot owned by its parent in Solid 2?","How does nested createRoot disposal work?","How do I make a detached root in Solid 2?","What is runWithOwner(null)?","Solid 2 createRoot lifetime","How do I dispose a nested root?","Why is my createRoot disposed with the component?"],"answer":"In Solid 2, a `createRoot` created inside an existing owner is owned by that parent and is disposed with it. Its own `dispose()` still works for earlier teardown. If you need the Solid 1 behaviour where the root lives until you dispose it, detach explicitly:\n\n```js\nconst dispose = runWithOwner(null, () =>\n  createRoot(dispose => {\n    onCleanup(() => controller.destroy());\n    return dispose;\n  }),\n);\n```"},{"id":"create-signal-fn","questions":["What is createSignal with a function?","How do I make a writable memo in Solid 2?","How do I create derived state with a setter?","What is createSignal(fn)?","How do I replace createComputed write-back?","Solid 2 writable derived signal","How do I derive a signal from props?"],"answer":"`createSignal(fn, options?)` creates a writable derived signal. The function computes the value and receives the previous value; the setter still writes like a normal signal. When a source changes, the function re-runs and receives your write as `prev`.\n\n```js\nconst [value, setValue] = createSignal(() => props.something);\n```\n\nThis replaces many `createComputed` write-back patterns: read the source in the initializer, keep the setter for local writes."},{"id":"create-store-fn","questions":["What is createStore with a function?","How do I create a derived store in Solid 2?","What is createStore(fn)?","How do I make a projection store?","What replaced createSelector?","Solid 2 derived store","How do I derive a store from reactive state?"],"answer":"`createStore(fn, seed, options?)` creates a derived store. The function receives a mutable draft and updates it based on reactive sources; the second argument is the seed/backing value, not a memo-style initial value.\n\n```js\nconst [selected, setSelected] = createStore(draft => {\n  const id = selectedId();\n  draft[id] = true;\n  if (draft._prev != null) delete draft[draft._prev];\n  draft._prev = id;\n}, {});\n```\n\nIt is the store analogue of a writable derived signal and supports selector-like patterns without notifying everything."},{"id":"create-memo-options","questions":["What is the second argument of createMemo in Solid 2?","Did createMemo lose its initial value?","How do I make a lazy memo in Solid 2?","What is the createMemo lazy option?","What is the unobserved callback?","How do I stop a memo from computing until it is read?","Solid 2 memo options"],"answer":"In Solid 2, the second argument to `createMemo` is an options object, not an initial value.\n\n```js\nconst expensive = createMemo(\n  () => heavyComputation(source()),\n  { lazy: true },\n);\n```\n\nWith `lazy: true`, the first computation waits until the memo is read, and the memo autodisposes when it loses all subscribers. `createSignal` and `createMemo` also accept an `unobserved` callback for cleanup when the last subscriber goes away."},{"id":"effect-vs-memo","questions":["Should I use createEffect or createMemo?","What is the difference between an effect and a memo?","When should I use createEffect?","Can I set a signal inside a memo?","Effect vs derived value in Solid 2","Why should effects not write signals?","Why does my effect run even when the value is the same?"],"answer":"Use `createMemo` for derived values and `createEffect` for side effects that leave the reactive system, such as DOM work, subscriptions, or logging. Effects should not write signals; derive the value instead.\n\nA memo caches and compares its result. An effect runs whenever its compute phase re-runs, even if the returned value is equal. To skip equal values, put a memo in front and read the memo in the effect."},{"id":"on-settled","questions":["What is onSettled in Solid 2?","When does onSettled run?","Can onSettled return cleanup?","Can I create primitives inside onSettled?","onSettled vs onMount","How do I run code when the activity settles?","What replaces onMount in Solid 2?"],"answer":"`onSettled` runs when the current activity settles, such as after mount. It replaces `onMount`, can read reactive values, and can return a cleanup function.\n\n```js\nonSettled(() => {\n  const value = count();\n  doSomething(value);\n  return () => cleanup();\n});\n```\n\nUnlike other tracked scopes, it cannot create nested primitives. That is a breaking change from Solid 1. `onCleanup` remains for cleanup inside computations."},{"id":"create-render-effect","questions":["What is createRenderEffect in Solid 2?","What is the difference between createRenderEffect and createEffect?","How do I run an effect synchronously during render?","Solid 2 render phase effect","When should I use createRenderEffect?","How do I bind a DOM property synchronously?"],"answer":"`createRenderEffect` uses the same split compute/apply pattern as `createEffect`, but it runs synchronously during the render phase instead of being batched to the next microtask. Use it for DOM-level bindings that must update immediately during rendering. Prefer `createEffect` for application-level side effects."},{"id":"async-resource-removed","questions":["What replaced createResource?","Is createResource gone in Solid 2?","How do I fetch data in Solid 2?","How do I load async data in Solid 2?","What is the createResource alternative?","How do I handle a loading state in Solid 2?","Solid 2 data fetching","How do I fetch with an async memo?"],"answer":"`createResource` is removed. Fetch data with an async computation, usually an async `createMemo`, and wrap the consumers in `Loading`:\n\n```js\nconst user = createMemo(() => fetchUser(id()));\n```\n\n```jsx\n<Loading fallback={<Spinner />}>\n  <Profile user={user()} />\n</Loading>\n```\n\nUse `isPending(() => user())` for an \"updating…\" indicator while an input change is in flight."},{"id":"async-loading","questions":["What is Loading in Solid 2?","What replaced Suspense?","How do I show a loading spinner in Solid 2?","Loading vs Suspense","How do I wait for async data before rendering?","What is the Loading fallback?","How do I coordinate multiple loading boundaries?"],"answer":"`Loading` replaces `Suspense` for the initial \"not ready yet\" UI. Wrap the component that reads an async computation:\n\n```jsx\n<Loading fallback={<Spinner />}>\n  <Profile user={user()} />\n</Loading>\n```\n\n`Loading` shows its fallback while the content is not ready. Use `isPending` for a subtle indicator during a revalidation, and `Reveal` to coordinate sibling `Loading` boundaries."},{"id":"async-is-pending","questions":["What is isPending in Solid 2?","What is the difference between Loading and isPending?","How do I show an updating indicator?","isPending vs Loading","How do I detect a pending update?","When is isPending true?","Why is isPending false after refresh?"],"answer":"`isPending(fn)` returns true while a change to the reactive expression in `fn` is in flight. It is for \"a change is on the way\" indicators, not the initial fallback. A bare `refresh()` is silent; to make a refetch pending, use `affects(...)` or an optimistic flag. Use `Loading` for initial readiness and `isPending` for in-flight updates."}]'''
)


def git_blob_sha(text: str) -> str:
    data = text.encode("utf-8")
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def fetch_text(url: str, attempts: int = 4) -> str:
    last = None
    for i in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "only-gates-ai-trainer"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read().decode("utf-8")
        except Exception as e:  # noqa: BLE001
            last = e
            if i + 1 < attempts:
                time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"fetch failed {url}: {last}")


def _script_dir() -> Optional[Path]:
    try:
        return Path(__file__).resolve().parent  # type: ignore[name-defined]
    except Exception:
        return None

def _repo_data_dir() -> Optional[Path]:
    sd = _script_dir()
    if sd:
        p = sd.parent / "data" / "solid2"
        if p.is_dir():
            return p
    for p in [Path("data/solid2"), Path("/content/data/solid2")]:
        if p.is_dir():
            return p
    return None

def load_docs(docs_dir: Optional[Path]) -> List[Tuple[str, str]]:
    # auto-detect local repo docs when run inside the repo
    candidates = []
    if docs_dir:
        candidates.append(docs_dir)
    candidates += [Path("data/solid2/docs"), Path("/content/data/solid2/docs")]
    sd = _script_dir()
    if sd:
        candidates.append(sd.parent / "data" / "solid2" / "docs")
    local = next((p for p in candidates if p and p.is_dir() and (p / "MIGRATION.md").exists()), None)
    docs: List[Tuple[str, str]] = []
    for name, sha in DOCS:
        text = None
        if local:
            p = local / name
            if p.exists():
                text = p.read_text(encoding="utf-8")
        if text is None:
            url = (
                f"https://raw.githubusercontent.com/solidjs/solid/{DOCS_COMMIT}"
                f"/documentation/solid-2.0/{name}"
            )
            print(f"fetch {name}", flush=True)
            text = fetch_text(url)
        actual = git_blob_sha(text)
        if actual != sha:
            raise RuntimeError(f"{name}: blob sha mismatch {actual} != {sha}")
        docs.append((name, text))
    if local:
        print(f"loaded {len(docs)} pinned docs from {local}", flush=True)

    # Supplemental material (HackMD / extra notes). Prefer local snapshots.
    repo = _repo_data_dir()
    for name, url, note in SUPPLEMENTAL_DOCS:
        text = None
        if repo:
            p = repo / name
            if p.exists():
                text = p.read_text(encoding="utf-8")
                print(f"loaded supplemental {name} ({len(text)} chars)", flush=True)
        if text is None:
            try:
                print(f"fetch supplemental {name}", flush=True)
                text = fetch_text(url)
            except Exception as e:  # noqa: BLE001
                print(f"WARNING: skipping supplemental {name}: {e}", flush=True)
                continue
        docs.append((f"[SUPPLEMENTAL: {note}]\n{name}", text))
    return docs

def load_stories(stories_path: Optional[Path], stories_chars: int = 0, stories_url: Optional[str] = None) -> str:
    candidates = []
    if stories_path:
        candidates.append(stories_path)
    repo = _repo_data_dir()
    if repo:
        candidates.append(repo / "tinystories.txt")
    sd = _script_dir()
    if sd:
        candidates.append(sd.parent / "data" / "solid2" / "tinystories.txt")
    candidates += [Path("data/solid2/tinystories.txt"), Path("/content/tinystories.txt"), Path("/content/data/solid2/tinystories.txt")]
    for p in candidates:
        try:
            if p and p.exists() and p.is_file():
                text = p.read_text(encoding="utf-8", errors="replace")
                if stories_chars and len(text) > stories_chars:
                    # cut on a paragraph boundary when possible
                    cut = text.rfind("\n\n", 0, stories_chars)
                    text = text[: cut if cut > stories_chars // 2 else stories_chars]
                print(f"loaded stories {p} ({len(text)} chars)", flush=True)
                return text
        except Exception as e:  # noqa: BLE001
            print(f"WARNING: could not read stories {p}: {e}", flush=True)
    if stories_url:
        try:
            print(f"fetch stories {stories_url}", flush=True)
            text = fetch_text(stories_url)
            if stories_chars and len(text) > stories_chars:
                cut = text.rfind("\n\n", 0, stories_chars)
                text = text[: cut if cut > stories_chars // 2 else stories_chars]
            print(f"loaded stories url ({len(text)} chars)", flush=True)
            return text
        except Exception as e:  # noqa: BLE001
            print(f"WARNING: could not fetch stories {stories_url}: {e}", flush=True)
    print("WARNING: no stories corpus found (pass --stories/--stories-url or place data/solid2/tinystories.txt); continuing without general-language data", flush=True)
    return ""


def load_qa(qa_path: Optional[Path]) -> List[Dict]:
    candidates = []
    if qa_path: candidates.append(qa_path)
    sd = _script_dir()
    candidates += [Path("data/solid2/qa.json"), Path("data/solid2/qa-compact.json"), Path("/content/data/solid2/qa.json")]
    if sd:
        candidates.append(sd.parent / "data" / "solid2" / "qa.json")
    for p in candidates:
        try:
            if p and p.exists():
                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data, dict) and "documents" in data:
                    return data["documents"]
                if isinstance(data, list):
                    return data
        except Exception:
            pass
    return QA_DATA
# ---------------------------------------------------------------------------
# Byte-level BPE (GPT-2 alphabet), producing ByteLevelBpe-compatible JSON:
#   { "version": 1, "specialTokens": [...], "vocab": [...], "merges": [[l,r], ...] }
# ---------------------------------------------------------------------------

def bytes_to_unicode() -> Dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}

BYTE_ENCODER = bytes_to_unicode()
BYTE_DECODER = {v: k for k, v in BYTE_ENCODER.items()}

import re as _stdre
try:
    import regex as _regex
    PRE_TOKENIZE_RE = _regex.compile(
        r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
except Exception:
    PRE_TOKENIZE_RE = _stdre.compile(
        r"""'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+"""
    )

PAIR_STRIDE = 1 << 16

def pre_tokenize(text: str) -> List[str]:
    return PRE_TOKENIZE_RE.findall(text)

def text_to_symbols(text: str) -> Tuple[int, ...]:
    return tuple(BYTE_ENCODER[b] for b in text.encode("utf-8"))

def get_pairs(symbols: Sequence) -> Dict[Tuple, int]:
    counts: Dict[Tuple, int] = {}
    prev = symbols[0]
    for ch in symbols[1:]:
        pair = (prev, ch)
        counts[pair] = counts.get(pair, 0) + 1
        prev = ch
    return counts

def merge_symbols(symbols: Tuple, pair: Tuple, merged: str) -> Tuple:
    out = []
    i = 0
    while i < len(symbols):
        j = symbols.index(pair[0], i) if pair[0] in symbols[i:] else -1
        if j < 0 or j >= len(symbols) - 1:
            out.extend(symbols[i:])
            break
        out.extend(symbols[i:j])
        if symbols[j] == pair[0] and symbols[j + 1] == pair[1]:
            out.append(merged)
            i = j + 2
        else:
            out.append(symbols[j])
            i = j + 1
    return tuple(out)

class BpeTokenizer:
    def __init__(self, model: Dict):
        self.vocab = model["vocab"]
        self.merges = [tuple(m) for m in model["merges"]]
        self.special_tokens = model["specialTokens"]
        self.token_to_id = {}
        for i, t in enumerate(self.vocab):
            if t not in self.token_to_id:
                self.token_to_id[t] = i
        self.merge_ranks = {"\0".join(m): i for i, m in enumerate(self.merges)}
        self.special_to_id = {t: self.token_to_id.get(t, i) for i, t in enumerate(self.special_tokens)}
        self.cache: Dict[str, List[int]] = {}

    def _bpe(self, token: str) -> List[str]:
        symbols = list(text_to_symbols(token))
        while len(symbols) > 1:
            best = None
            best_rank = 10**18
            for i in range(len(symbols) - 1):
                rank = self.merge_ranks.get(symbols[i] + "\0" + symbols[i + 1])
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best = i
            if best is None:
                break
            merged = symbols[best] + symbols[best + 1]
            symbols[best : best + 2] = [merged]
        return symbols

    def encode_ordinary(self, text: str) -> List[int]:
        ids: List[int] = []
        for piece in pre_tokenize(text):
            cached = self.cache.get(piece)
            if cached is not None:
                ids.extend(cached)
                continue
            piece_ids = []
            for sym in self._bpe(piece):
                tid = self.token_to_id.get(sym)
                if tid is None:
                    raise ValueError(f"missing symbol {sym!r}")
                piece_ids.append(tid)
            if len(self.cache) < 100000:
                self.cache[piece] = piece_ids
            ids.extend(piece_ids)
        return ids

    def encode(self, text: str) -> List[int]:
        if not self.special_tokens:
            return self.encode_ordinary(text)
        # greedy special token split, longest first
        specials = sorted(self.special_tokens, key=len, reverse=True)
        out: List[int] = []
        i = 0
        while i < len(text):
            matched = None
            for s in specials:
                if text.startswith(s, i):
                    matched = s
                    break
            if matched:
                sid = self.special_to_id.get(matched)
                if sid is not None:
                    out.append(sid)
                else:
                    out.extend(self.encode_ordinary(matched))
                i += len(matched)
            else:
                # accumulate until next special
                j = i + 1
                while j <= len(text):
                    if any(text.startswith(s, j) for s in specials):
                        break
                    j += 1
                out.extend(self.encode_ordinary(text[i:j]))
                i = j
        return out

def train_bpe(corpus: str, vocab_size: int = 4096, min_frequency: int = 2) -> Dict:
    # base vocab: specials + 256 bytes
    vocab = list(SPECIAL_TOKENS)
    for b in range(256):
        vocab.append(BYTE_ENCODER[b])
    token_to_id = {t: i for i, t in enumerate(vocab)}

    # count unique pre-tokens
    word_freq: Dict[Tuple, int] = {}
    for piece in pre_tokenize(corpus):
        syms = text_to_symbols(piece)
        word_freq[syms] = word_freq.get(syms, 0) + 1
    words = list(word_freq.keys())
    freqs = [word_freq[w] for w in words]
    print(f"tokenizer: {len(words)} unique pre-tokens, base {len(vocab)}", flush=True)

    merges: List[List[str]] = []
    while len(vocab) < vocab_size:
        pair_counts: Dict[Tuple, int] = {}
        for syms, f in zip(words, freqs):
            if len(syms) < 2 or f <= 0:
                continue
            local = get_pairs(syms)
            for p, c in local.items():
                pair_counts[p] = pair_counts.get(p, 0) + c * f
        if not pair_counts:
            break
        # max by count, then lexicographic for determinism
        best = max(pair_counts.items(), key=lambda kv: (kv[1], kv[0][0], kv[0][1]))
        if best[1] < min_frequency:
            break
        pair = best[0]
        merged = pair[0] + pair[1]
        merges.append([pair[0], pair[1]])
        if merged not in token_to_id:
            token_to_id[merged] = len(vocab)
            vocab.append(merged)
        # apply merge to words containing pair (rebuild all - simple/safe for small corpus)
        new_words = []
        for syms in words:
            if len(syms) < 2:
                new_words.append(syms)
                continue
            out = []
            i = 0
            while i < len(syms):
                if i < len(syms) - 1 and syms[i] == pair[0] and syms[i + 1] == pair[1]:
                    out.append(merged)
                    i += 2
                else:
                    out.append(syms[i])
                    i += 1
            new_words.append(tuple(out))
        words = new_words
        if len(vocab) % 256 == 0 or len(vocab) == vocab_size:
            print(f"  vocab {len(vocab)} merges {len(merges)}", flush=True)

    return {"version": 1, "specialTokens": SPECIAL_TOKENS, "vocab": vocab, "merges": merges}
# ---------------------------------------------------------------------------
# Model (matches C inference: pre-norm RMSNorm, GELU-tanh, no biases, tied emb)
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(dim))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-5) * self.gain

class Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp: int, dropout: float):
        super().__init__()
        if dim % heads: raise ValueError("dim % heads != 0")
        self.heads, self.head_dim = heads, dim // heads
        self.g1, self.g2 = RMSNorm(dim), RMSNorm(dim)
        self.wqkv = nn.Linear(dim, dim * 3, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.w1 = nn.Linear(dim, mlp, bias=False)
        self.w2 = nn.Linear(mlp, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        h = self.g1(x)
        qkv = self.wqkv(h).view(B, T, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)
        scores = scores.masked_fill(mask, float("-inf"))
        x = x + self.dropout(self.wo((torch.softmax(scores, -1) @ v).transpose(1, 2).reshape(B, T, D)))
        h = self.g2(x)
        x = x + self.dropout(self.w2(F.gelu(self.w1(h), approximate="tanh")))
        return x

class TinyGPT(nn.Module):
    def __init__(self, vocab: int, ctx: int, dim: int, layers: int, heads: int, mlp: int, dropout: float):
        super().__init__()
        self.vocab_size, self.context, self.dim, self.layers_n, self.heads, self.mlp_n = vocab, ctx, dim, layers, heads, mlp
        self.token_emb = nn.Embedding(vocab, dim)
        self.pos_emb = nn.Embedding(ctx, dim)
        self.blocks = nn.ModuleList([Block(dim, heads, mlp, dropout) for _ in range(layers)])
        self.g_final = RMSNorm(dim)
        self.apply(self._init)
        for b in self.blocks:
            nn.init.normal_(b.wo.weight, std=0.02 / math.sqrt(2 * max(1, layers)))
            nn.init.normal_(b.w2.weight, std=0.02 / math.sqrt(2 * max(1, layers)))
    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear): nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.Embedding): nn.init.normal_(m.weight, 0.0, 0.02)
    def forward(self, tokens, targets=None):
        B, T = tokens.shape
        if T > self.context: raise ValueError("seq > context")
        x = self.token_emb(tokens) + self.pos_emb(torch.arange(T, device=tokens.device))
        for b in self.blocks: x = b(x)
        logits = self.g_final(x) @ self.token_emb.weight.T
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1), ignore_index=-100)
        return logits, loss
    @torch.no_grad()
    def generate(self, tokens, max_new, temperature=0.8, top_k=40, eos_id=None):
        self.eval()
        for _ in range(max_new):
            logits, _ = self(tokens[:, -self.context:])
            logits = logits[:, -1, :] / max(temperature, 1e-4)
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < v[:, -1:], float("-inf"))
            nxt = torch.multinomial(torch.softmax(logits, -1), 1)
            tokens = torch.cat([tokens, nxt], 1)
            if eos_id is not None and int(nxt.item()) == eos_id: break
        return tokens

def parameter_count(vocab, ctx, dim, layers, mlp):
    return vocab*dim + ctx*dim + dim + layers*(2*dim + 4*dim*dim + 2*dim*mlp)

# ---------------------------------------------------------------------------
# Packing (exact tiny_gpt_create order) + verification
# ---------------------------------------------------------------------------

def pack_weights(model: TinyGPT) -> torch.Tensor:
    parts = [model.token_emb.weight.detach().cpu().flatten(),
             model.pos_emb.weight.detach().cpu().flatten(),
             model.g_final.gain.detach().cpu().flatten()]
    for b in model.blocks:
        parts += [b.g1.gain.detach().cpu().flatten(),
                  b.wqkv.weight.detach().cpu().T.flatten(),
                  b.wo.weight.detach().cpu().T.flatten(),
                  b.g2.gain.detach().cpu().flatten(),
                  b.w1.weight.detach().cpu().T.flatten(),
                  b.w2.weight.detach().cpu().T.flatten()]
    return torch.cat(parts).to(torch.float32)

def verify_packed(model: TinyGPT, packed: torch.Tensor, cfg: Dict) -> None:
    V,C,D,L,M = cfg["vocabSize"],cfg["context"],cfg["dim"],cfg["layers"],cfg["mlp"]
    expect = parameter_count(V,C,D,L,M)
    if packed.numel() != expect: raise ValueError(f"packed {packed.numel()} != {expect}")
    off=0
    def take(n): 
        nonlocal off; v=packed[off:off+n]; off+=n; return v
    assert torch.equal(take(V*D).view(V,D), model.token_emb.weight.detach().cpu())
    assert torch.equal(take(C*D).view(C,D), model.pos_emb.weight.detach().cpu())
    assert torch.equal(take(D), model.g_final.gain.detach().cpu())
    for b in model.blocks:
        assert torch.equal(take(D), b.g1.gain.detach().cpu())
        assert torch.equal(take(D*D*3).view(D,D*3), b.wqkv.weight.detach().cpu().T)
        assert torch.equal(take(D*D).view(D,D), b.wo.weight.detach().cpu().T)
        assert torch.equal(take(D), b.g2.gain.detach().cpu())
        assert torch.equal(take(D*M).view(D,M), b.w1.weight.detach().cpu().T)
        assert torch.equal(take(M*D).view(M,D), b.w2.weight.detach().cpu().T)
    assert off == packed.numel()
# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def build_corpus(docs, qa, stories: str = ""):
    parts = []
    for name, text in docs:
        parts.append(f"# {name}\n\n{text}")
    doc_text = "\n\n".join(parts)
    qa_pairs = "\n".join(f"<|user|>{q}<|assistant|>{d['answer']}<|end|>" for d in qa for q in d['questions'])
    qa_plain = "\n\n".join(f"Q: {q}\nA: {d['answer']}" for d in qa for q in d['questions'])
    # tokenizer corpus: domain + QA + optional general stories (helps byte patterns/coherence)
    tok_parts = [doc_text, doc_text, qa_pairs, qa_pairs, qa_plain, qa_plain]
    if stories:
        tok_parts.append(stories)
    tok_corpus = "\n\n".join(tok_parts)
    # LM text: interleave general stories (language modeling) with Solid 2 domain + chat QA.
    # Keep chat-formatted QA prominent so the model learns the inference format.
    lm_parts = [doc_text, qa_pairs, qa_pairs, qa_pairs]
    if stories:
        lm_parts = [stories, doc_text, stories, qa_pairs, qa_pairs, qa_pairs, stories, doc_text, qa_pairs]
    lm_text = "\n\n".join(lm_parts)
    return tok_corpus, lm_text

def make_qa_examples(tokenizer: BpeTokenizer, qa, ctx: int):
    # hold out every 5th concept id for val
    ids = sorted({d["id"] for d in qa})
    val_ids = {x for i, x in enumerate(ids) if i % 5 == 0}
    def pack(doc, q):
        question = tokenizer.encode_ordinary(q)
        answer = tokenizer.encode_ordinary(doc["answer"])
        fixed = 3  # user, assistant, end
        if len(answer) + fixed > ctx:
            answer = answer[: max(1, ctx - fixed)]
        budget = ctx - fixed - len(answer)
        if len(question) > budget:
            question = question[-budget:] if budget > 0 else []
        seq = [USER_ID] + question + [ASSISTANT_ID] + answer + [END_ID]
        prompt_len = len(seq) - len(answer) - 1
        # The model computes same-position cross-entropy, so position i must be
        # trained to predict seq[i + 1]. Start at the assistant token so the
        # model learns to emit the first answer token, and end at the last token
        # before <|end|> so it learns to stop.
        labels = [-100] * len(seq)
        for i in range(prompt_len - 1, len(seq) - 1):
            labels[i] = seq[i + 1]
        # pad
        pad = ctx - len(seq)
        if pad > 0:
            seq = seq + [PAD_ID] * pad
            labels = labels + [-100] * pad
        return seq[:ctx], labels[:ctx]
    train, val = [], []
    for d in qa:
        target = val if d["id"] in val_ids else train
        for q in d["questions"]:
            target.append(pack(d, q))
    return train, val, len(val_ids), len(ids) - len(val_ids)

def tokenize_lm(tokenizer: BpeTokenizer, text: str):
    # split by blank lines to insert <|end|> boundaries
    ids: List[int] = []
    for chunk in text.split("\n\n"):
        t = chunk.strip()
        if not t: continue
        ids.extend(tokenizer.encode_ordinary(t))
        ids.append(END_ID)
    return ids

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def lr_lambda(step, total, warmup):
    if step < warmup: return (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

CHECKPOINT_FORMAT = "only-gates-train-state"
CHECKPOINT_VERSION = 1
STOP_REQUESTED = False


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(value) -> str:
    return sha256_text(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False))


def sha256_ids(ids: Sequence[int]) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(ids, dtype=np.int32).tobytes())
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_torch_save(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def prune_snapshots(directory: Path, keep: int) -> None:
    if keep <= 0:
        return
    files = sorted(directory.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in files[keep:]:
        try:
            old.unlink()
        except OSError:
            pass


def capture_rng() -> Dict:
    rng = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        rng["cuda"] = torch.cuda.get_rng_state_all()
    return rng


def restore_rng(rng: Optional[Dict]) -> None:
    if not rng:
        return
    try:
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        if torch.cuda.is_available() and rng.get("cuda"):
            if len(rng["cuda"]) == torch.cuda.device_count():
                torch.cuda.set_rng_state_all(rng["cuda"])
            else:
                print("WARNING: CUDA device count changed; skipping CUDA RNG restore", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: could not restore RNG state: {exc}", flush=True)


def request_stop(signum, _frame) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"\nsignal {signum}: finishing the current step, then saving", flush=True)


def finite_or_none(value: float):
    return float(value) if math.isfinite(value) else None


def make_config(args) -> Dict:
    return {
        "vocabSize": args.vocab,
        "context": args.context,
        "dim": args.dim,
        "layers": args.layers,
        "heads": args.heads,
        "mlp": args.mlp,
        "dropout": args.dropout,
        "qaOversample": args.qa_oversample,
    }


def config_diff(old: Dict, new: Dict) -> List[str]:
    keys = sorted(set(old) | set(new))
    return [f"{k}: checkpoint={old.get(k)!r} current={new.get(k)!r}"
            for k in keys if old.get(k) != new.get(k)]


def train_lm(model, tokens, steps, ctx, batch, lr, warmup, wd, clip, device, label,
             opt, sch, start_step=0, loss_sum=0.0, loss_count=0,
             save_every=0, on_save=None, should_stop=None):
    if steps <= 0 or tokens is None or len(tokens) < ctx + 2:
        return float("nan"), True, loss_sum, loss_count, start_step
    model.train()
    arr = torch.tensor(tokens, dtype=torch.long)
    windows = arr.unfold(0, ctx + 1, 1)
    n = windows.size(0)
    t0 = time.time()
    finished = True
    last_step = start_step
    for step in range(start_step, steps):
        b = windows[torch.randint(0, n, (batch,))].to(device)
        opt.zero_grad(set_to_none=True)
        _, loss = model(b[:, :-1], b[:, 1:])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step(); sch.step()
        loss_sum += float(loss.detach()); loss_count += 1
        last_step = step + 1
        if step % 50 == 0 or step == steps - 1:
            print(f"{label} {step}/{steps} loss {float(loss.detach()):.4f} lr {sch.get_last_lr()[0]:.2e} {time.time()-t0:.1f}s", flush=True)
        if on_save and save_every > 0 and last_step % save_every == 0 and last_step < steps:
            on_save(last_step, loss_sum, loss_count)
        if should_stop and should_stop():
            if on_save:
                on_save(last_step, loss_sum, loss_count)
            finished = False
            break
    return loss_sum / max(1, loss_count), finished, loss_sum, loss_count, last_step

@torch.no_grad()
def eval_qa(model, data, batch, device):
    if not data: return float("nan")
    model.eval(); total=n=0
    for i in range(0, len(data), batch):
        chunk = data[i:i+batch]
        x = torch.tensor([c[0] for c in chunk], dtype=torch.long, device=device)
        y = torch.tensor([c[1] for c in chunk], dtype=torch.long, device=device)
        _, loss = model(x, y)
        total += float(loss) * len(chunk); n += len(chunk)
    return total / max(1,n)

def train_qa(model, train_data, val_data, epochs, batch, lr, warmup, wd, clip, device, out_dir,
             opt, sch, start_epoch=0, start_cursor=0, start_perm=None,
             best=float("inf"), best_ep=0, step=0,
             save_every=0, on_save=None, should_stop=None):
    if not train_data or epochs <= 0:
        return best, best_ep, True, step
    finished = True
    for ep in range(start_epoch, epochs):
        model.train()
        if ep == start_epoch and start_perm:
            perm = list(start_perm)
        else:
            perm = list(range(len(train_data)))
            random.shuffle(perm)
        begin = start_cursor if ep == start_epoch else 0
        run = n = 0
        for i in range(begin, len(perm), batch):
            idx = perm[i:i+batch]
            x = torch.tensor([train_data[j][0] for j in idx], dtype=torch.long, device=device)
            y = torch.tensor([train_data[j][1] for j in idx], dtype=torch.long, device=device)
            opt.zero_grad(set_to_none=True)
            _, loss = model(x,y); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step(); sch.step()
            run += float(loss.detach())*len(idx); n+=len(idx); step+=1
            cursor = i + len(idx)
            if on_save and save_every > 0 and step % save_every == 0:
                on_save(ep, cursor, perm, best, best_ep, step)
            if should_stop and should_stop():
                if on_save:
                    on_save(ep, cursor, perm, best, best_ep, step)
                finished = False
                break
        if not finished:
            break
        v = eval_qa(model, val_data, batch, device)
        print(f"qa epoch {ep+1}/{epochs} train {run/max(1,n):.4f} val {v:.4f}", flush=True)
        if math.isfinite(v) and v < best:
            best=v; best_ep=ep+1
            torch.save(model.state_dict(), out_dir/"best-qa.pt")
        if on_save:
            on_save(ep + 1, 0, None, best, best_ep, step)
        start_perm = None
        start_cursor = 0
    if math.isfinite(best) and (out_dir/"best-qa.pt").exists():
        model.load_state_dict(torch.load(out_dir/"best-qa.pt", map_location=device))
    return best, best_ep, finished, step

def generate_samples(model, tokenizer: BpeTokenizer, prompts, device, max_new=120):
    model.eval(); out=[]
    for p in prompts:
        ids = [USER_ID] + tokenizer.encode_ordinary(p) + [ASSISTANT_ID]
        x = torch.tensor([ids], dtype=torch.long, device=device)
        y = model.generate(x, max_new=max_new, temperature=0.7, top_k=40, eos_id=END_ID)[0].tolist()
        gen = y[len(ids):]
        # decode (skip specials in output except end marker handling)
        text = []
        for tid in gen:
            if tid == END_ID: break
            if tid in (PAD_ID, USER_ID, ASSISTANT_ID, FACTS_ID): continue
            tok = tokenizer.vocab[tid] if tid < len(tokenizer.vocab) else ""
            text.append(tok)
        # byte-level decode
        bs = bytearray()
        for ch in "".join(text):
            b = BYTE_DECODER.get(ch)
            if b is not None: bs.append(b)
        try: dec = bs.decode("utf-8", errors="replace")
        except: dec = "".join(text)
        out.append({"prompt": p, "text": dec})
        print(f"sample> {p} => {dec[:140]}", flush=True)
    return out

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", "--output-dir", dest="out", type=Path, default=Path("tiny-gpt-solid2-out"))
    ap.add_argument("--docs-dir", type=Path, default=None, help="Local dir with pinned Solid 2 .md files (else fetched)")
    ap.add_argument("--qa-json", type=Path, default=None, help="Path to qa.json or qa-compact list (else embedded)")
    ap.add_argument("--stories", type=Path, default=None, help="Optional general-language stories text (e.g. data/solid2/tinystories.txt) to improve coherence")
    ap.add_argument("--stories-url", type=str, default=None, help="URL to fetch stories text from (used if local file not found)")
    ap.add_argument("--stories-chars", type=int, default=900_000, help="Max characters of stories to include (0 = all)")
    ap.add_argument("--no-stories", action="store_true", help="Disable stories even if a default file is found")
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--context", "--ctx", dest="context", type=int, default=256)
    ap.add_argument("--dim", type=int, default=48)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--mlp", type=int, default=96)
    ap.add_argument("--batch", "--batch-size", dest="batch", type=int, default=16)
    ap.add_argument("--lm-steps", type=int, default=5000, help="LM pretraining steps over stories+docs+QA chat text (more = better coherence)")
    ap.add_argument("--qa-epochs", type=int, default=100, help="QA chat finetune epochs (best val checkpoint restored)")
    ap.add_argument("--qa-oversample", type=int, default=8, help="Times to repeat chat-formatted QA in the LM stream")
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--qa-lr", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--target-params", type=int, default=1_000_000)
    ap.add_argument("--smoke", action="store_true", help="Tiny fast run for pipeline verification")
    ap.add_argument("--resume", action="store_true", help="Continue from <out>/latest.pt when it exists")
    ap.add_argument("--resume-from", type=Path, default=None, help="Explicit training-state checkpoint to continue")
    ap.add_argument("--fresh", action="store_true", help="Ignore existing checkpoints and start over")
    ap.add_argument("--save-every", type=int, default=100, help="Save a checkpoint every N optimizer steps (0 disables step saves)")
    ap.add_argument("--max-minutes", type=float, default=0.0, help="Stop gracefully after this many minutes (0 = no limit)")
    ap.add_argument("--max-hours", type=float, default=0.0, help="Stop gracefully after this many hours (0 = no limit)")
    ap.add_argument("--keep-checkpoints", type=int, default=3, help="Rolling checkpoint snapshots to keep (0 disables)")
    ap.add_argument("--stage", choices=["all", "lm", "qa"], default="all", help="Run both stages, only LM pretraining, or only QA finetuning")
    ap.add_argument("--data-cache", action="store_true", help="Reuse <out>/lm_ids.npy when present")
    ap.add_argument("--export-only", action="store_true", help="Rebuild exported artifacts from latest.pt without training")
    args = ap.parse_args()
    if args.smoke:
        args.vocab, args.context, args.dim, args.layers, args.heads, args.mlp = 512, 64, 32, 1, 2, 64
        args.lm_steps, args.qa_epochs, args.batch = 2, 1, 2
        print("SMOKE mode: tiny config/steps", flush=True)
    if args.max_minutes < 0 or args.max_hours < 0:
        raise SystemExit("--max-minutes and --max-hours must be non-negative")

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    budget_seconds = args.max_minutes * 60 + args.max_hours * 3600
    session_start = time.monotonic()
    deadline = session_start + budget_seconds if budget_seconds > 0 else None

    def should_stop() -> bool:
        return STOP_REQUESTED or (deadline is not None and time.monotonic() >= deadline)

    device = torch.device(args.device)
    cfg = make_config(args)
    params = parameter_count(args.vocab, args.context, args.dim, args.layers, args.mlp)
    print(f"config vocab={args.vocab} ctx={args.context} dim={args.dim} layers={args.layers} heads={args.heads} mlp={args.mlp}")
    print(f"parameters {params:,} (target ~{args.target_params:,}, delta {params-args.target_params:+,}) device={device}")
    if args.dim % args.heads != 0:
        raise SystemExit("dim must be divisible by heads")

    # 1) training state. Resuming is opt-in so a stale directory cannot surprise a run.
    ckpt = None
    checkpoint_path = args.resume_from or (args.out / "latest.pt")
    if not args.fresh and (args.resume or args.resume_from is not None or args.export_only):
        if checkpoint_path.exists():
            print(f"resuming from {checkpoint_path}", flush=True)
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        elif args.resume_from is not None:
            raise SystemExit(f"checkpoint not found: {checkpoint_path}")
        else:
            print(f"no checkpoint at {checkpoint_path}; starting fresh", flush=True)

    if args.export_only and ckpt is None:
        raise SystemExit("--export-only needs an existing checkpoint; run training first")

    if ckpt is not None:
        if ckpt.get("format") != CHECKPOINT_FORMAT:
            raise SystemExit(f"{checkpoint_path}: not a {CHECKPOINT_FORMAT} checkpoint")
        if int(ckpt.get("version", -1)) != CHECKPOINT_VERSION:
            raise SystemExit(f"{checkpoint_path}: unsupported checkpoint version {ckpt.get('version')}")
        diffs = config_diff(ckpt.get("config", {}), cfg)
        if diffs:
            raise SystemExit("checkpoint config differs from this run:\n  " + "\n  ".join(diffs) + "\nUse --fresh to start over.")
        if int(ckpt.get("seed", args.seed)) != args.seed:
            raise SystemExit(f"checkpoint seed {ckpt.get('seed')} != --seed {args.seed}; use --fresh to change it")

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    # 2) corpus + tokenizer. A resumed run reuses the tokenizer stored in the
    # checkpoint; retraining BPE could renumber every token and corrupt the run.
    qa = load_qa(args.qa_json)
    qa_hash = sha256_json(qa)
    docs = None
    stories = None
    lm_text = None
    if ckpt is not None:
        tok_model = ckpt.get("tokenizer_model")
        if not tok_model:
            raise SystemExit("checkpoint has no tokenizer_model; cannot resume")
        print("loaded tokenizer from checkpoint", flush=True)
    else:
        docs = load_docs(args.docs_dir)
        stories = "" if args.no_stories else load_stories(args.stories, args.stories_chars, args.stories_url)
        print(f"docs {len(docs)} qa concepts {len(qa)} stories chars {len(stories)}", flush=True)
        tok_corpus, lm_text = build_corpus(docs, qa, stories)
        print("training BPE tokenizer...", flush=True)
        tok_model = train_bpe(tok_corpus, vocab_size=args.vocab, min_frequency=2)
        atomic_write_text(args.out / "tokenizer.json", json.dumps(tok_model, ensure_ascii=False) + "\n")
        print(f"tokenizer vocab {len(tok_model['vocab'])} merges {len(tok_model['merges'])}", flush=True)
        tokenizer = BpeTokenizer(tok_model)
        rt = "createEffect(() => name(), (v) => { el.title = v; });"
        dec = "".join(tokenizer.vocab[i] if i < len(tokenizer.vocab) else "" for i in tokenizer.encode_ordinary(rt))
        bs = bytearray(); [bs.append(BYTE_DECODER[c]) for c in dec if c in BYTE_DECODER]
        if bs.decode("utf-8", errors="ignore") != rt:
            print("WARNING: tokenizer round-trip mismatch for sample", flush=True)
    tokenizer = BpeTokenizer(tok_model)

    # 3) data. LM ids are cached so a resumed Colab session does not need to
    # rebuild the corpus, but the cached stream is checked against the fingerprint.
    qa_train, qa_val, n_val, n_train = make_qa_examples(tokenizer, qa, args.context)
    cache_path = args.out / "lm_ids.npy"
    lm_ids = None
    if ckpt is not None and (args.data_cache or cache_path.exists()):
        try:
            lm_ids = np.load(cache_path).astype(np.int64).tolist()
            print(f"loaded cached lm ids {len(lm_ids)}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: could not read {cache_path}: {exc}", flush=True)
    if lm_ids is None and lm_text is None:
        print("rebuilding the token stream to verify the checkpoint fingerprint...", flush=True)
        docs = load_docs(args.docs_dir)
        stories = "" if args.no_stories else load_stories(args.stories, args.stories_chars, args.stories_url)
        print(f"docs {len(docs)} qa concepts {len(qa)} stories chars {len(stories)}", flush=True)
        _, lm_text = build_corpus(docs, qa, stories)
    if lm_ids is None:
        lm_ids = tokenize_lm(tokenizer, lm_text)
        chat_ids = []
        qa_rep = max(0, args.qa_oversample)
        for _ in range(qa_rep):
            for seq, _lab in qa_train:
                chat_ids.extend([t for t in seq if t != PAD_ID])
        lm_ids = lm_ids + chat_ids
        np.save(cache_path, np.asarray(lm_ids, dtype=np.int32))
        print(f"wrote token-stream cache {cache_path}", flush=True)
    print(f"lm tokens {len(lm_ids)} qa train {len(qa_train)} val {len(qa_val)} (concepts train={n_train} val={n_val})", flush=True)

    fingerprint = {
        "qa": qa_hash,
        "tokenizer": sha256_json(tok_model),
        "lm_ids": sha256_ids(lm_ids),
        "seed": args.seed,
    }
    if docs is not None:
        fingerprint["docs"] = sha256_text("\n\0".join(f"{name}\n{text}" for name, text in docs))
    if stories is not None:
        fingerprint["stories"] = sha256_text(stories)

    if ckpt is not None:
        old = ckpt.get("fingerprint", {})
        # When the cached token stream let us skip docs and stories, carry the
        # original hashes forward so run.json still describes the whole run.
        for key in ("docs", "stories"):
            if key not in fingerprint and key in old:
                fingerprint[key] = old[key]
        required = {"qa", "tokenizer", "lm_ids", "seed"}
        missing = sorted(required - set(old))
        mismatches = [f"{k}: checkpoint={old[k]} current={fingerprint[k]}"
                      for k in old if k in fingerprint and old[k] != fingerprint[k]]
        if missing or mismatches:
            detail = mismatches or [f"missing fingerprint field: {k}" for k in missing]
            raise SystemExit("checkpoint data mismatch:\n  " + "\n  ".join(detail) + "\nUse --fresh to start a new run.")

    # 4) model + checkpoint writer
    model = TinyGPT(args.vocab, args.context, args.dim, args.layers, args.heads, args.mlp, args.dropout).to(device)
    if ckpt is not None:
        model.load_state_dict(ckpt["model"])

    snap_dir = args.out / "checkpoints"
    wall_base = float(ckpt.get("wall_seconds", 0.0)) if ckpt else 0.0
    snap_index = int(ckpt.get("snapshot_index", 0)) if ckpt else 0
    resume_count = (int(ckpt.get("resume_count", 0)) + 1) if ckpt else 0
    stories_chars = len(stories) if stories is not None else int(ckpt.get("stories_chars", 0) if ckpt else 0)
    env = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": str(device),
    }
    if torch.cuda.is_available():
        env["gpu"] = torch.cuda.get_device_name(0)

    def write_state(stage_name: str, optimizer=None, scheduler=None, model_state=None, **extra) -> None:
        nonlocal snap_index
        payload = {
            "format": CHECKPOINT_FORMAT,
            "version": CHECKPOINT_VERSION,
            "stage": stage_name,
            "config": cfg,
            "fingerprint": fingerprint,
            "tokenizer_model": tok_model,
            "seed": args.seed,
            "resume_count": resume_count,
            "environment": env,
            "stories_chars": stories_chars,
            "wall_seconds": wall_base + (time.monotonic() - session_start),
            "model": model_state if model_state is not None else model.state_dict(),
            "rng": capture_rng(),
        }
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        if scheduler is not None:
            payload["scheduler"] = scheduler.state_dict()
        payload.update(extra)
        atomic_torch_save(args.out / "latest.pt", payload)
        if args.keep_checkpoints > 0:
            snap_dir.mkdir(parents=True, exist_ok=True)
            snap_index += 1
            atomic_torch_save(snap_dir / f"{stage_name}-{snap_index:07d}.pt", payload)
            prune_snapshots(snap_dir, args.keep_checkpoints)
        manifest = {
            "format": payload["format"],
            "version": payload["version"],
            "stage": stage_name,
            "config": cfg,
            "fingerprint": fingerprint,
            "seed": args.seed,
            "resumeCount": resume_count,
            "environment": env,
            "wallSeconds": round(payload["wall_seconds"], 1),
            "lm": {"step": payload.get("lm_step"), "steps": payload.get("lm_steps"), "loss": payload.get("lm_loss")},
            "qa": {"epoch": payload.get("qa_epoch"), "cursor": payload.get("qa_cursor"),
                   "bestVal": payload.get("qa_best"), "bestEpoch": payload.get("qa_best_epoch"),
                   "step": payload.get("qa_step")},
        }
        atomic_write_text(args.out / "run.json", json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    def log_metric(record: Dict) -> None:
        item = dict(record)
        item["time"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        item["wallSeconds"] = round(wall_base + (time.monotonic() - session_start), 1)
        with open(args.out / "metrics.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    stage = ckpt.get("stage", "lm") if ckpt is not None else ("lm" if args.stage != "qa" else "qa")
    if args.export_only:
        stage = "done"
    if stage == "lm" and args.stage == "qa":
        raise SystemExit("cannot start QA: LM pretraining in the checkpoint is not complete. Use --stage all.")

    lm_step = int(ckpt.get("lm_step", 0)) if ckpt else 0
    lm_loss_sum = float(ckpt.get("lm_loss_sum", 0.0)) if ckpt else 0.0
    lm_loss_count = int(ckpt.get("lm_loss_count", 0)) if ckpt else 0
    lm_loss = (float(ckpt["lm_loss"]) if ckpt and ckpt.get("lm_loss") is not None else float("nan"))
    qa_epoch = int(ckpt.get("qa_epoch", 0)) if ckpt else 0
    qa_cursor = int(ckpt.get("qa_cursor", 0)) if ckpt else 0
    qa_perm = ckpt.get("qa_perm") if ckpt else None
    qa_best = (float(ckpt["qa_best"]) if ckpt and ckpt.get("qa_best") is not None else float("inf"))
    qa_best_epoch = int(ckpt.get("qa_best_epoch", 0)) if ckpt else 0
    qa_step = int(ckpt.get("qa_step", 0)) if ckpt else 0

    # 5) LM stage
    if stage == "lm":
        if args.lm_steps <= 0 or len(lm_ids) <= args.context + 2:
            lm_loss = float("nan")
            stage = "lm-done" if args.stage == "lm" else "qa"
            write_state(stage, lm_step=lm_step, lm_steps=args.lm_steps,
                        lm_loss=None, lm_loss_sum=lm_loss_sum, lm_loss_count=lm_loss_count,
                        qa_epoch=0, qa_cursor=0, qa_perm=None,
                        qa_best=None, qa_best_epoch=0, qa_step=0)
            if args.stage == "lm":
                print("LM stage skipped (no steps or data).", flush=True)
                return
        else:
            print(f"LM pretrain {args.lm_steps} steps (starting at {lm_step})...", flush=True)
            lm_opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
            lm_sch = torch.optim.lr_scheduler.LambdaLR(lm_opt, lambda s: lr_lambda(s, args.lm_steps, 100))
            if ckpt is not None and ckpt.get("stage") == "lm":
                if ckpt.get("optimizer") and ckpt.get("scheduler"):
                    lm_opt.load_state_dict(ckpt["optimizer"])
                    lm_sch.load_state_dict(ckpt["scheduler"])
                restore_rng(ckpt.get("rng"))

            def save_lm(done, loss_sum, loss_count):
                mean = loss_sum / max(1, loss_count)
                write_state("lm", optimizer=lm_opt, scheduler=lm_sch,
                            lm_step=done, lm_steps=args.lm_steps,
                            lm_loss=finite_or_none(mean),
                            lm_loss_sum=loss_sum, lm_loss_count=loss_count,
                            qa_epoch=0, qa_cursor=0, qa_perm=None,
                            qa_best=None, qa_best_epoch=0, qa_step=0)
                log_metric({"stage": "lm", "step": done, "steps": args.lm_steps,
                            "loss": finite_or_none(mean), "lr": lm_sch.get_last_lr()[0]})

            lm_loss, finished, lm_loss_sum, lm_loss_count, lm_step = train_lm(
                model, lm_ids, args.lm_steps, args.context, args.batch,
                args.lr, 100, 0.01, 1.0, device, "lm",
                lm_opt, lm_sch, start_step=lm_step,
                loss_sum=lm_loss_sum, loss_count=lm_loss_count,
                save_every=args.save_every, on_save=save_lm, should_stop=should_stop)
            print(f"lm mean loss {lm_loss:.4f}", flush=True)
            if not finished:
                print(f"stopped at LM step {lm_step}/{args.lm_steps}. Resume with --resume.", flush=True)
                return
            stage = "lm-done" if args.stage == "lm" else "qa"
            write_state(stage, lm_step=lm_step, lm_steps=args.lm_steps,
                        lm_loss=finite_or_none(lm_loss),
                        lm_loss_sum=lm_loss_sum, lm_loss_count=lm_loss_count,
                        qa_epoch=0, qa_cursor=0, qa_perm=None,
                        qa_best=None, qa_best_epoch=0, qa_step=0)
            if args.stage == "lm":
                print("LM stage complete. Rerun with --stage all to finetune QA.", flush=True)
                return

    # 6) QA stage
    if stage in ("lm-done", "qa") and args.stage != "lm":
        if args.qa_epochs <= 0 or not qa_train:
            stage = "done"
        else:
            total_steps = max(1, args.qa_epochs * ((len(qa_train) + args.batch - 1) // args.batch))
            qa_opt = torch.optim.AdamW(model.parameters(), lr=args.qa_lr, betas=(0.9, 0.95), weight_decay=0.01)
            qa_sch = torch.optim.lr_scheduler.LambdaLR(qa_opt, lambda s: lr_lambda(s, total_steps, 20))
            if ckpt is not None and ckpt.get("stage") == "qa":
                if ckpt.get("optimizer") and ckpt.get("scheduler"):
                    qa_opt.load_state_dict(ckpt["optimizer"])
                    qa_sch.load_state_dict(ckpt["scheduler"])
                restore_rng(ckpt.get("rng"))
            elif ckpt is not None and ckpt.get("stage") == "lm-done":
                # Continue the RNG stream from the end of LM when a run was split
                # into separate --stage lm and --stage all invocations.
                restore_rng(ckpt.get("rng"))
            print(f"QA finetune {args.qa_epochs} epochs (starting at epoch {qa_epoch}, cursor {qa_cursor})...", flush=True)

            def save_qa(ep, cursor, perm, best, best_ep, step):
                write_state("qa", optimizer=qa_opt, scheduler=qa_sch,
                            lm_step=lm_step, lm_steps=args.lm_steps,
                            lm_loss=finite_or_none(lm_loss),
                            lm_loss_sum=lm_loss_sum, lm_loss_count=lm_loss_count,
                            qa_epoch=ep, qa_cursor=cursor, qa_perm=perm,
                            qa_best=finite_or_none(best), qa_best_epoch=best_ep, qa_step=step)
                log_metric({"stage": "qa", "epoch": ep, "cursor": cursor,
                            "best_val": finite_or_none(best), "best_epoch": best_ep,
                            "step": step, "lr": qa_sch.get_last_lr()[0]})

            qa_best, qa_best_epoch, finished, qa_step = train_qa(
                model, qa_train, qa_val, args.qa_epochs, args.batch,
                args.qa_lr, 20, 0.01, 1.0, device, args.out,
                qa_opt, qa_sch, start_epoch=qa_epoch, start_cursor=qa_cursor,
                start_perm=qa_perm, best=qa_best, best_ep=qa_best_epoch, step=qa_step,
                save_every=args.save_every, on_save=save_qa, should_stop=should_stop)
            print(f"qa best val {qa_best:.4f} @ epoch {qa_best_epoch}", flush=True)
            if not finished:
                print(f"stopped during QA epoch {qa_epoch}/{args.qa_epochs}. Resume with --resume.", flush=True)
                return
            stage = "done"
            write_state("done", lm_step=lm_step, lm_steps=args.lm_steps,
                        lm_loss=finite_or_none(lm_loss), lm_loss_sum=lm_loss_sum,
                        lm_loss_count=lm_loss_count, qa_epoch=args.qa_epochs,
                        qa_cursor=0, qa_perm=None, qa_best=finite_or_none(qa_best),
                        qa_best_epoch=qa_best_epoch, qa_step=qa_step)

    if stage != "done":
        print("training stopped before export; rerun with --resume", flush=True)
        return

    # 7) samples + export
    prompts = ["What are split effects?", "How do I migrate createEffect from Solid 1?",
               "What replaced createResource?", "Why does my signal not update immediately in Solid 2?"]
    samples = generate_samples(model, tokenizer, prompts, device)

    torch.save(model.state_dict(), args.out / "final-state.pt")
    packed = pack_weights(model)
    export_cfg = {k: cfg[k] for k in ("vocabSize", "context", "dim", "layers", "heads", "mlp")}
    verify_packed(model, packed, export_cfg)
    (args.out / "weights.bin").write_bytes(packed.numpy().astype("<f4", copy=False).tobytes())
    (args.out / "tokenizer.json").write_text(json.dumps(tok_model, ensure_ascii=False) + "\n", encoding="utf-8")
    source_parts = ["Pinned Solid 2.0 documentation (MIT, solidjs/solid@next)", "supplemental Solid 2 HackMD notes (proposal/pattern context)", "curated Solid 2 Q&A in this repository"]
    if stories_chars:
        source_parts.append("general-language short stories (coherence)")
    checkpoint = {
        "format": "only-gates-tiny-gpt",
        "version": 1,
        "config": export_cfg,
        "parameterCount": int(params),
        "tokenizer": "tokenizer.json",
        "meta": {
            "name": "tiny-gpt-solid2",
            "source": "; ".join(source_parts) + ".",
            "license": "Model weights generated by this repository; upstream sources retain their own licenses/attribution.",
            "training": {
                "lmSteps": args.lm_steps, "qaEpochs": args.qa_epochs,
                "qaBestVal": finite_or_none(qa_best), "qaBestEpoch": qa_best_epoch,
                "lmMeanLoss": finite_or_none(lm_loss),
                "context": args.context, "dim": args.dim, "layers": args.layers,
                "heads": args.heads, "mlp": args.mlp, "vocabSize": args.vocab,
                "storiesChars": stories_chars,
            },
        },
        "samples": samples,
        "prompts": ["What are split effects?", "How do I migrate createEffect from Solid 1?",
                    "What replaced createResource?", "What is flush in Solid 2?",
                    "How do I handle loading state in Solid 2?"],
    }
    (args.out / "checkpoint.json").write_text(json.dumps(checkpoint, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out/'weights.bin'} ({(args.out/'weights.bin').stat().st_size} bytes)")
    print(f"wrote {args.out/'checkpoint.json'} and tokenizer.json")
    print(f"parameterCount={params} (target ~1M). To use in web app: copy weights.bin, tokenizer.json, checkpoint.json to apps/only-gates-ai/public/models/tiny-gpt-solid2/")

if __name__ == "__main__":
    main()
