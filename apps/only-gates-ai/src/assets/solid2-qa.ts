import type { QaDocument } from "../lib/qaRuntime.ts";

/*
 * A curated Solid 2.0 Q&A corpus.
 *
 * Answers are written for this demo and checked against the official
 * Solid 2.0 RFCs and migration guide. Each document links back to the
 * source section. The questions are deliberately varied: retrieval needs
 * to recognise the same concept from a heading, a migration question, or
 * a first-time-user phrasing.
 *
 * Sources (pinned):
 *   https://github.com/solidjs/solid/tree/next/documentation/solid-2.0
 */

const MIGRATION =
  "https://github.com/solidjs/solid/blob/7a943c8d423cb33e9b3d8c2feb59f37cb318137d/documentation/solid-2.0/MIGRATION.md";
const RFC_REACTIVITY =
  "https://github.com/solidjs/solid/blob/8fa6483be18380a959daad26df11337d5e6b9201/documentation/solid-2.0/01-reactivity-batching-effects.md";
const RFC_SIGNALS =
  "https://github.com/solidjs/solid/blob/f2865cfae544e3d8d0729dfe1b439b0f6590aebc/documentation/solid-2.0/02-signals-derived-ownership.md";

export const SOLID2_QA_DOCUMENTS: readonly QaDocument[] = [
  {
    id: "effects-split",
    questions: [
      "What are split effects?",
      "What is a split effect in Solid?",
      "Why are effects split in Solid 2?",
      "Why did Solid 2 split createEffect?",
      "Why does createEffect take two functions?",
      "What does createEffect(compute, apply) mean?",
      "How do split effects work?",
      "What is the difference between compute and apply in createEffect?",
      "Solid 2 createEffect signature",
      "What is the new createEffect API?",
    ],
    answer: `In Solid 2, \`createEffect\` takes two functions. The compute function reads reactive values and returns a value; Solid records exactly what it read. The apply function receives that value and performs the side effect. Apply runs after all compute phases in the batch, so dependencies are known before any side effect runs. The apply function can return a cleanup function. This replaces the 1.x \`on\` helper and makes async and boundary behaviour predictable.`,
    source: `${RFC_REACTIVITY}#split-tracking-from-effect`,
    tags: ["effects", "reactivity"],
    keywords: ["createEffect", "compute", "apply", "split effect"],
  },
  {
    id: "effects-migrate",
    questions: [
      "How do I migrate createEffect from Solid 1?",
      "How do I rewrite an effect for Solid 2?",
      "What is the new createEffect syntax?",
      "How do I use createEffect in Solid 2?",
      "Solid 1 effect to Solid 2",
      "createEffect before and after Solid 2",
      "How do I update my effects for Solid 2?",
      "Convert a 1.x effect to a split effect",
    ],
    answer: `Move the reactive read into the first function and the side effect into the second.

\`\`\`js
// Solid 1
createEffect(() => {
  el.title = name();
});

// Solid 2
createEffect(
  () => name(),
  value => {
    el.title = value;
  },
);
\`\`\`

The compute function should only read reactive values. Put DOM work, subscriptions, and other side effects in the apply function.`,
    source: `${MIGRATION}#effects-lifecycle-and-cleanup`,
    tags: ["effects", "migration"],
    keywords: ["createEffect", "migrate"],
  },
  {
    id: "effects-cleanup",
    questions: [
      "How do I clean up an effect in Solid 2?",
      "Where does onCleanup go in a split effect?",
      "How do I return cleanup from createEffect?",
      "Solid 2 effect cleanup",
      "How do I cancel an interval in an effect?",
      "How do I remove an event listener in an effect?",
      "Does createEffect support cleanup?",
      "How do I stop an effect from leaking?",
    ],
    answer: `Return a cleanup function from the apply phase. Solid calls it before the next apply and when the owner is disposed.

\`\`\`js
createEffect(
  () => name(),
  value => {
    const id = setInterval(() => console.log(value), 1000);
    return () => clearInterval(id);
  },
);
\`\`\`

\`onCleanup\` still exists for cleanup inside computations, but effect cleanup normally belongs on the apply side.`,
    source: `${MIGRATION}#effects-lifecycle-and-cleanup`,
    tags: ["effects", "cleanup"],
    keywords: ["createEffect", "cleanup", "onCleanup"],
  },
  {
    id: "effects-on-removed",
    questions: [
      "What happened to on() in Solid 2?",
      "Is the on helper still available?",
      "How do I watch one specific signal in Solid 2?",
      "How do I replace on() in Solid 2?",
      "Solid 2 on() removed",
      "How do I run an effect only when one signal changes?",
      "How do I declare explicit dependencies in Solid 2?",
      "Do I still need on()?",
    ],
    answer: `\`on\` is no longer needed because the compute phase is the explicit dependency declaration.

\`\`\`js
createEffect(() => count(), value => {
  console.log(value);
});

// Multiple dependencies
createEffect(
  () => [a(), b()],
  ([a, b]) => console.log(a, b),
);
\`\`\`

To skip the initial run, pass \`{ defer: true }\` as the third argument.`,
    source: `${MIGRATION}#on-helper--split-effects`,
    tags: ["effects", "migration"],
    keywords: ["createEffect", "on", "dependencies"],
  },
  {
    id: "effects-defer",
    questions: [
      "How do I skip the first run of an effect in Solid 2?",
      "How do I use defer with createEffect?",
      "How do I run an effect only after the initial render?",
      "Solid 2 effect should not run on mount",
      "How do I avoid the initial effect run?",
      "What replaced on(..., { defer: true })?",
      "createEffect defer option",
    ],
    answer: `Pass \`{ defer: true }\` as the third argument:

\`\`\`js
createEffect(
  () => count(),
  value => {
    console.log(value);
  },
  { defer: true },
);
\`\`\`

The apply phase is skipped on the initial run and starts after the dependency changes. This replaces \`on(..., { defer: true })\` from Solid 1.`,
    source: `${MIGRATION}#on-helper--split-effects`,
    tags: ["effects", "migration"],
    keywords: ["createEffect", "defer"],
  },
  {
    id: "effects-prev",
    questions: [
      "What happened to the initial value of createEffect?",
      "How do I get the previous value in createEffect?",
      "Solid 2 createEffect initial value",
      "How do I use the prev argument in createEffect?",
      "How do I set a default previous value?",
      "Why is the second createEffect argument not an initial value?",
      "createEffect prev parameter",
    ],
    answer: `The 1.x \`initialValue\` argument is gone. The compute function receives the previous value (\`undefined\` on the first run), and the apply function receives \`(value, prev)\`.

\`\`\`js
createEffect(
  (prev = 0) => count(),
  (value, prev) => {
    console.log("changed from", prev, "to", value);
  },
);
\`\`\`

\`createMemo\` also treats its second argument as options now, not an initial value.`,
    source: `${MIGRATION}#effects-lifecycle-and-cleanup`,
    tags: ["effects", "migration"],
    keywords: ["createEffect", "prev", "initialValue"],
  },
  {
    id: "effects-createcomputed",
    questions: [
      "What replaced createComputed?",
      "Is createComputed still available in Solid 2?",
      "How do I migrate createComputed?",
      "CreateComputed removal in Solid 2",
      "What is the alternative to createComputed?",
      "How do I do derived write-back in Solid 2?",
      "Solid 2 createComputed replacement",
    ],
    answer: `\`createComputed\` is removed. Pick a replacement based on intent:

- Use \`createMemo\` for a read-only derived value.
- Use split \`createEffect(compute, apply)\` for a side effect on change.
- Use function-form \`createSignal(fn)\` or \`createStore(fn)\` for derived state that also has a setter.

For example, \`createComputed(() => setDoubled(count() * 2))\` becomes \`const doubled = createMemo(() => count() * 2)\`.`,
    source: `${MIGRATION}#createcomputed--creatememo-createeffect-or-derived-createsignal`,
    tags: ["reactivity", "migration"],
    keywords: ["createComputed", "createMemo", "derived"],
  },
  {
    id: "effects-onmount",
    questions: [
      "What replaced onMount?",
      "Is onMount still available in Solid 2?",
      "How do I run code after mount in Solid 2?",
      "What is onSettled?",
      "onMount vs onSettled",
      "How do I migrate onMount?",
      "How do I run something once after render in Solid 2?",
    ],
    answer: `\`onMount\` is replaced by \`onSettled\`. It runs when the current activity settles, such as after mount, can read reactive values, and can return a cleanup function.

\`\`\`js
onSettled(() => {
  measureLayout();
  const onResize = () => measureLayout();
  window.addEventListener("resize", onResize);
  return () => window.removeEventListener("resize", onResize);
});
\`\`\`

For a one-time read that should not track, use \`untrack\`.`,
    source: `${MIGRATION}#effects-lifecycle-and-cleanup`,
    tags: ["lifecycle", "migration"],
    keywords: ["onMount", "onSettled"],
  },
  {
    id: "effects-error",
    questions: [
      "How do I handle errors in createEffect?",
      "How do I catch an async error in an effect?",
      "Solid 2 effect error callback",
      "What is the EffectBundle in Solid 2?",
      "How do I use the createEffect error option?",
      "How do I handle a failed fetch in an effect?",
      "createEffect error handling",
    ],
    answer: `Pass an \`EffectBundle\` object as the second argument instead of a bare apply function:

\`\`\`js
createEffect(
  () => fetchData(id()),
  {
    effect: data => renderData(data),
    error: (err, cleanup) => {
      console.error("Fetch failed:", err);
      cleanup();
    },
  },
);
\`\`\`

The \`error\` handler catches errors from async computations in the reactive graph. For component-level error UI, use an \`Errored\` boundary.`,
    source: `${RFC_REACTIVITY}#createeffect-error-handling`,
    tags: ["effects", "errors"],
    keywords: ["createEffect", "error", "Errored"],
  },
  {
    id: "effects-stores",
    questions: [
      "How do I read a store in a split effect?",
      "Why do I get a strict read warning with a store in an effect?",
      "How do I use a store proxy in createEffect?",
      "How do I deep track a store in an effect?",
      "Should I use snapshot inside an effect?",
      "Store in the createEffect compute phase",
      "Solid 2 STRICT_READ_UNTRACKED",
    ],
    answer: `The apply phase runs untracked, so don't pass a store proxy through and read its properties there. Extract the values you need in compute and pass plain data:

\`\`\`js
createEffect(
  () => ({ name: store.user.name, age: store.user.age }),
  value => sendAnalytics(value.name, value.age),
);
\`\`\`

For deep tracking, \`deep(store)\` returns a plain snapshot and tracks every nested property. For a copy without subscribing, use \`snapshot(store)\`.`,
    source: `${RFC_REACTIVITY}#stores-in-the-compute-phase`,
    tags: ["effects", "stores"],
    keywords: ["createEffect", "store", "deep", "snapshot"],
  },
  {
    id: "batching-flush",
    questions: [
      "Why does my signal not update immediately in Solid 2?",
      "Why does a setter still read the old value?",
      "What is flush in Solid 2?",
      "How do I read the DOM right after setting state?",
      "Signal updates feel delayed in Solid 2",
      "How does microtask batching work?",
      "How do I use flush(fn)?",
      "Why is count() still old after setCount?",
    ],
    answer: `Solid 2 batches writes on a microtask. After \`setCount(1)\`, \`count()\` still returns the old value until the batch flushes. Call \`flush()\` to apply pending updates immediately:

\`\`\`js
setSubmitted(true);
flush();
inputRef.focus();
\`\`\`

\`flush(fn)\` runs the callback in a synchronous flush scope and applies its writes before returning. Use \`flush\` sparingly: tests, imperative interop, and reading the DOM right after a write.`,
    source: `${RFC_REACTIVITY}#flush-and-microtask-batching`,
    tags: ["reactivity", "batching"],
    keywords: ["flush", "batch", "microtask"],
  },
  {
    id: "batching-batch-removed",
    questions: [
      "What happened to batch in Solid 2?",
      "Is batch still available?",
      "What replaces batch?",
      "How do I batch updates in Solid 2?",
      "Do I need to call batch?",
      "Solid 2 batch removal",
    ],
    answer: `\`batch\` is removed because all writes are batched by default on a microtask. Just perform the writes; they apply together after the batch. If you need them applied synchronously, call \`flush()\` after the writes. \`flush(fn)\` applies writes made inside the callback before it returns.`,
    source: `${MIGRATION}#batch--default-microtask-batching--flush`,
    tags: ["reactivity", "migration"],
    keywords: ["batch", "flush"],
  },
  {
    id: "writes-owned-scope",
    questions: [
      "Why does writing a signal in an effect throw in Solid 2?",
      "Can I set a signal inside createEffect?",
      "What is a reactive write in owned scope?",
      "Solid 2 write inside memo error",
      "What is the ownedWrite option?",
      "Why does setting state in a component body throw?",
      "How do I update a signal from an effect?",
    ],
    answer: `Writing to a signal or store inside a reactive scope (an effect, memo, or component body) throws in development. Move the write to an event handler or \`onSettled\`. \`untrack\` does not exempt a write: it stops tracking, not ownership.

For a rare internal signal that must be written in scope, opt in narrowly:

\`\`\`js
const [ref, setRef] = createSignal(null, { ownedWrite: true });
\`\`\`

Do not use \`ownedWrite\` for application state. Derive the value with \`createMemo\` instead of writing it back.`,
    source: `${RFC_REACTIVITY}#no-writes-under-owned-scope`,
    tags: ["reactivity", "errors"],
    keywords: ["ownedWrite", "write", "memo", "effect"],
  },
  {
    id: "top-level-reads",
    questions: [
      "What is the top-level reactive read warning in Solid 2?",
      "Why does destructuring props warn in Solid 2?",
      "How do I fix an untracked reactive read warning?",
      "Solid 2 props destructuring warning",
      "How do I read props once in Solid 2?",
      "Component body signal read warning",
      "Why does reading a signal in a Show callback warn?",
    ],
    answer: `In a component body, a top-level reactive read warns in development because it can capture a dependency accidentally. Read inside JSX or a reactive scope such as \`createMemo\` or \`createEffect\`, or make a one-time read explicit with \`untrack\`:

\`\`\`js
const title = untrack(() => props.title);
\`\`\`

Avoid destructuring reactive props at the top level. Keep the props object and read \`props.title\` in JSX so the read stays tracked.`,
    source: `${RFC_REACTIVITY}#strict-top-level-access-new-in-20`,
    tags: ["reactivity", "diagnostics"],
    keywords: ["untrack", "props", "warning", "top-level"],
  },
  {
    id: "untrack-once",
    questions: [
      "What replaced the /*@once*/ JSX marker?",
      "How do I read a value once in Solid 2 JSX?",
      "Solid 2 once comment removed",
      "How do I use untrack in Solid 2?",
      "What is the replacement for /*@once*/?",
      "How do I set an initial input value in Solid 2?",
      "How do I opt out of reactivity for one read?",
    ],
    answer: `\`/*@once*/\` is no longer part of Solid's public JSX model. If a value should update, leave the read reactive in JSX. For an initial DOM value, use the platform default prop, such as \`defaultValue={props.initialValue}\`. If you truly need a one-time JavaScript read, use \`untrack(() => props.value)\` outside JSX.`,
    source: `${MIGRATION}#once-jsx-marker`,
    tags: ["reactivity", "migration", "jsx"],
    keywords: ["untrack", "once", "defaultValue"],
  },
  {
    id: "create-root-ownership",
    questions: [
      "Is createRoot owned by its parent in Solid 2?",
      "How does nested createRoot disposal work?",
      "How do I make a detached root in Solid 2?",
      "What is runWithOwner(null)?",
      "Solid 2 createRoot lifetime",
      "How do I dispose a nested root?",
      "Why is my createRoot disposed with the component?",
    ],
    answer: `In Solid 2, a \`createRoot\` created inside an existing owner is owned by that parent and is disposed with it. Its own \`dispose()\` still works for earlier teardown. If you need the Solid 1 behaviour where the root lives until you dispose it, detach explicitly:

\`\`\`js
const dispose = runWithOwner(null, () =>
  createRoot(dispose => {
    onCleanup(() => controller.destroy());
    return dispose;
  }),
);
\`\`\``,
    source: `${RFC_SIGNALS}#ownership-createroot-is-owned-by-the-parent-by-default`,
    tags: ["ownership", "migration"],
    keywords: ["createRoot", "runWithOwner", "dispose"],
  },
  {
    id: "create-signal-fn",
    questions: [
      "What is createSignal with a function?",
      "How do I make a writable memo in Solid 2?",
      "How do I create derived state with a setter?",
      "What is createSignal(fn)?",
      "How do I replace createComputed write-back?",
      "Solid 2 writable derived signal",
      "How do I derive a signal from props?",
    ],
    answer: `\`createSignal(fn, options?)\` creates a writable derived signal. The function computes the value and receives the previous value; the setter still writes like a normal signal. When a source changes, the function re-runs and receives your write as \`prev\`.

\`\`\`js
const [value, setValue] = createSignal(() => props.something);
\`\`\`

This replaces many \`createComputed\` write-back patterns: read the source in the initializer, keep the setter for local writes.`,
    source: `${RFC_SIGNALS}#function-form-createsignal-writable-memo`,
    tags: ["reactivity", "signals"],
    keywords: ["createSignal", "derived", "writable"],
  },
  {
    id: "create-store-fn",
    questions: [
      "What is createStore with a function?",
      "How do I create a derived store in Solid 2?",
      "What is createStore(fn)?",
      "How do I make a projection store?",
      "What replaced createSelector?",
      "Solid 2 derived store",
      "How do I derive a store from reactive state?",
    ],
    answer: `\`createStore(fn, seed, options?)\` creates a derived store. The function receives a mutable draft and updates it based on reactive sources; the second argument is the seed/backing value, not a memo-style initial value.

\`\`\`js
const [selected, setSelected] = createStore(draft => {
  const id = selectedId();
  draft[id] = true;
  if (draft._prev != null) delete draft[draft._prev];
  draft._prev = id;
}, {});
\`\`\`

It is the store analogue of a writable derived signal and supports selector-like patterns without notifying everything.`,
    source: `${RFC_SIGNALS}#function-form-createstore-derivedprojection-store`,
    tags: ["stores", "reactivity"],
    keywords: ["createStore", "projection", "createSelector"],
  },
  {
    id: "create-memo-options",
    questions: [
      "What is the second argument of createMemo in Solid 2?",
      "Did createMemo lose its initial value?",
      "How do I make a lazy memo in Solid 2?",
      "What is the createMemo lazy option?",
      "What is the unobserved callback?",
      "How do I stop a memo from computing until it is read?",
      "Solid 2 memo options",
    ],
    answer: `In Solid 2, the second argument to \`createMemo\` is an options object, not an initial value.

\`\`\`js
const expensive = createMemo(
  () => heavyComputation(source()),
  { lazy: true },
);
\`\`\`

With \`lazy: true\`, the first computation waits until the memo is read, and the memo autodisposes when it loses all subscribers. \`createSignal\` and \`createMemo\` also accept an \`unobserved\` callback for cleanup when the last subscriber goes away.`,
    source: `${RFC_REACTIVITY}#lazy-memos`,
    tags: ["reactivity", "memos"],
    keywords: ["createMemo", "lazy", "unobserved"],
  },
  {
    id: "effect-vs-memo",
    questions: [
      "Should I use createEffect or createMemo?",
      "What is the difference between an effect and a memo?",
      "When should I use createEffect?",
      "Can I set a signal inside a memo?",
      "Effect vs derived value in Solid 2",
      "Why should effects not write signals?",
      "Why does my effect run even when the value is the same?",
    ],
    answer: `Use \`createMemo\` for derived values and \`createEffect\` for side effects that leave the reactive system, such as DOM work, subscriptions, or logging. Effects should not write signals; derive the value instead.

A memo caches and compares its result. An effect runs whenever its compute phase re-runs, even if the returned value is equal. To skip equal values, put a memo in front and read the memo in the effect.`,
    source: `${RFC_REACTIVITY}#split-tracking-from-effect`,
    tags: ["effects", "memos"],
    keywords: ["createEffect", "createMemo", "derived"],
  },
  {
    id: "on-settled",
    questions: [
      "What is onSettled in Solid 2?",
      "When does onSettled run?",
      "Can onSettled return cleanup?",
      "Can I create primitives inside onSettled?",
      "onSettled vs onMount",
      "How do I run code when the activity settles?",
      "What replaces onMount in Solid 2?",
    ],
    answer: `\`onSettled\` runs when the current activity settles, such as after mount. It replaces \`onMount\`, can read reactive values, and can return a cleanup function.

\`\`\`js
onSettled(() => {
  const value = count();
  doSomething(value);
  return () => cleanup();
});
\`\`\`

Unlike other tracked scopes, it cannot create nested primitives. That is a breaking change from Solid 1. \`onCleanup\` remains for cleanup inside computations.`,
    source: `${RFC_REACTIVITY}#createtrackedeffect-and-onsettled`,
    tags: ["lifecycle", "effects"],
    keywords: ["onSettled", "onMount", "cleanup"],
  },
  {
    id: "create-render-effect",
    questions: [
      "What is createRenderEffect in Solid 2?",
      "What is the difference between createRenderEffect and createEffect?",
      "How do I run an effect synchronously during render?",
      "Solid 2 render phase effect",
      "When should I use createRenderEffect?",
      "How do I bind a DOM property synchronously?",
    ],
    answer: `\`createRenderEffect\` uses the same split compute/apply pattern as \`createEffect\`, but it runs synchronously during the render phase instead of being batched to the next microtask. Use it for DOM-level bindings that must update immediately during rendering. Prefer \`createEffect\` for application-level side effects.`,
    source: `${RFC_REACTIVITY}#createrendereffect-render-phase-split-effect`,
    tags: ["effects", "dom"],
    keywords: ["createRenderEffect", "createEffect", "render"],
  },
  {
    id: "async-resource-removed",
    questions: [
      "What replaced createResource?",
      "Is createResource gone in Solid 2?",
      "How do I fetch data in Solid 2?",
      "How do I load async data in Solid 2?",
      "What is the createResource alternative?",
      "How do I handle a loading state in Solid 2?",
      "Solid 2 data fetching",
      "How do I fetch with an async memo?",
    ],
    answer: `\`createResource\` is removed. Fetch data with an async computation, usually an async \`createMemo\`, and wrap the consumers in \`Loading\`:

\`\`\`js
const user = createMemo(() => fetchUser(id()));
\`\`\`

\`\`\`jsx
<Loading fallback={<Spinner />}>
  <Profile user={user()} />
</Loading>
\`\`\`

Use \`isPending(() => user())\` for an "updating…" indicator while an input change is in flight.`,
    source: `${MIGRATION}#createresource--async-computations--loading`,
    tags: ["async", "migration"],
    keywords: ["createResource", "createMemo", "fetch", "Loading"],
  },
  {
    id: "async-loading",
    questions: [
      "What is Loading in Solid 2?",
      "What replaced Suspense?",
      "How do I show a loading spinner in Solid 2?",
      "Loading vs Suspense",
      "How do I wait for async data before rendering?",
      "What is the Loading fallback?",
      "How do I coordinate multiple loading boundaries?",
    ],
    answer: `\`Loading\` replaces \`Suspense\` for the initial "not ready yet" UI. Wrap the component that reads an async computation:

\`\`\`jsx
<Loading fallback={<Spinner />}>
  <Profile user={user()} />
</Loading>
\`\`\`

\`Loading\` shows its fallback while the content is not ready. Use \`isPending\` for a subtle indicator during a revalidation, and \`Reveal\` to coordinate sibling \`Loading\` boundaries.`,
    source: `${MIGRATION}#suspense--loading`,
    tags: ["async", "loading"],
    keywords: ["Loading", "Suspense", "fallback"],
  },
  {
    id: "async-is-pending",
    questions: [
      "What is isPending in Solid 2?",
      "What is the difference between Loading and isPending?",
      "How do I show an updating indicator?",
      "isPending vs Loading",
      "How do I detect a pending update?",
      "When is isPending true?",
      "Why is isPending false after refresh?",
    ],
    answer: `\`isPending(fn)\` returns true while a change to the reactive expression in \`fn\` is in flight. It is for "a change is on the way" indicators, not the initial fallback. A bare \`refresh()\` is silent; to make a refetch pending, use \`affects(...)\` or an optimistic flag. Use \`Loading\` for initial readiness and \`isPending\` for in-flight updates.`,
    source: `${MIGRATION}#initial-loading-vs-in-flight-changes-loading-vs-ispending`,
    tags: ["async", "loading"],
    keywords: ["isPending", "Loading", "refresh", "affects"],
  },
];

export const SOLID2_QA_SUGGESTIONS: readonly string[] = [
  "What are split effects?",
  "How do I migrate createEffect from Solid 1?",
  "How do I clean up an effect?",
  "What happened to on()?",
  "What replaced createResource?",
  "How do I show a loading indicator?",
];
