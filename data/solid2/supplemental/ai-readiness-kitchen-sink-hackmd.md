# AI Readiness — Findings from the kitchen-sink example

> I asked AI how to better make SolidJS 2.0 usable by AI, we decided after doing those steps we should also include a kitchen sink example. They suggested Todos. and I offered up the TodoMVC with async/error affordances. We decided to use a simple demo I had made as a baseline and add back the missing TodoMVC functionality. In the process we used this to dogfood our own advice so the AI could evaluate how well its measures worked.
>
> The remainder of this document is written by Opus 4.7.

Each entry is a real bug I (Claude) wrote while building `examples/todos/`. The
purpose is to identify exactly *where* current guidance (CHEATSHEET / JSDoc /
type system / dev warnings) failed to steer me, so we know what to harden.

## Diagnostic frame: React Brain, per session

Humans learning Solid have **React Brain** — the ingrained reflexes that take
repeated exposure to unlearn. For humans, the deprogramming amortizes across
months and many sessions, so gentle redirection works: the third time a learner
passes an accessor as a prop and the example reads `filter={filter()}`,
something clicks and stays clicked.

AI agents have no such accumulation. **Every session is a fresh human with
React Brain in full force.** No transfer between conversations, no transfer
between models, no "I learned this last time." The entire deprogramming has
to happen *in-context*, on the first read of whatever artifacts are surfaced,
before the model commits to a frame. And the relevance judgments that decide
which artifacts get read are themselves done with React Brain on (Finding
\#17).

That asymmetry inverts the artifact budget. For humans, gentle is fine and
adversarial reads as preachy. For AI, gentle is invisible (read past, judged
tangential) and adversarial is the minimum viable signal — the framing has to
be loud enough to win the recognition lottery on every fresh session, with no
benefit-of-the-doubt from prior exposure. The cheatsheet warning, the
anti-comparison opener on `todos.ts`, the explicit *"Solid is not React, and
2.0 is not 1.x"* — by human-onboarding standards these read as borderline
confrontational. By AI-onboarding standards they may still be too soft.

It also bounds what docs can do. There is no docs-quality threshold above
which a model with React priors will stop syntax-diffing on first pass. The
mechanical guardrails (lint, types, dev warnings) work by being *in-band with
the action* — the model can't generate the code without hitting them. Docs
work only if the model reads them, frames them as relevant, and applies them.
The first two of those conditions fail per-session and aren't fully fixable
from inside the repo.

This document catalogs the failure modes that follow from that asymmetry, and
the interventions that address them. **Findings #1–#16 are generation-mode**
— the model writes Solid code and a mechanical guardrail catches it. **Finding
\#17 is recognition-mode** — the model is asked to *describe* Solid and the
prior frames the question itself; mechanical guardrails do not reach it. The
two classes share one root cause and require different intervention shapes.

## Format

- **Bug** — what was wrong in the code I produced
- **Fix** — what it should have been
- **Why I got it wrong** — the prior / missing guidance that led me astray
- **What would have prevented it** — concrete actionable change

---

## 1. Manual class string builder instead of `class={[...]}` array+object form

**Bug**

```tsx
const cls = () => {
  const parts = ["todo"];
  if (props.todo.completed) parts.push("completed");
  if (m().saving) parts.push("saving");
  if (m().error) parts.push("errored");
  return parts.join(" ");
};
return <li class={cls()}>...
```

**Fix (idiomatic 2.0)**

```tsx
return (
  <li class={["todo", {
    completed: props.todo.completed,
    saving: !!m().saving,
    errored: !!m().error,
  }]}>
```

The `class` prop accepts a combined array + object form: array entries are
always-on classes (or further nested), and object entries toggle by truthiness.

**Why I got it wrong**

This is the more interesting finding: the cheatsheet already shows this exact
pattern (`<div class={["card", props.class, { active: isActive() }]} />`,
line 339). I read the cheatsheet, used the object-only form *once* on the
filter links, then reverted to React-style manual string joining for
`<TodoItem>` — within the same file, with no intervening context that would
have flushed the pattern from working memory.

So the failure mode isn't "guidance missing" — it's:

1. **The pattern isn't memorable enough.** A single line buried in a JSX
   section reads as "trivia", not as "the canonical way to do conditional
   classes". The React/`classnames` prior is dominant in training data and
   reasserts itself unless the Solid pattern is repeated and prescriptive.
2. **Inconsistent self-application.** Even when I *do* use the right form
   once, I don't generalize it to nearby JSX. Each component re-derives its
   class strategy from scratch.

**What would have prevented it**

- CHEATSHEET: promote conditional classes to a named subsection ("Conditional
  classes — always use `class={[...]}`") with two or three examples, so it's
  encountered as a heading rather than a one-liner inside JSX trivia.
  Explicitly state "do not build class strings manually" — prescriptive, not
  just descriptive.
- Optional ESLint rule (later, in `eslint-plugin-solid`): flag JSX
  `class={someComputedString()}` where the string is built by `+`/`.join()`
  /template-literal concatenation of conditionals, and suggest the
  array+object form.
- JSDoc on `JSX.HTMLAttributes['class']`: document the array/object/string
  shapes inline so IDE completion shows the full surface.

**Resolution (this round)**

Deferred to lint, alongside the rest of the `eslint-plugin-solid 2.0`
workstream. New rule logged in `.notes/ai-readiness.md` Part 2 row 11
and the Ship list entry: `prefer-class-array-form` — flag string
concat / template literals on `class={...}` and suggest the array+object
form. JSDoc on `JSX.HTMLAttributes['class']` remains a candidate but
isn't worth a touch in isolation; bundle with the broader JSDoc sweep
when that runs.

---

## 2. Controlled-input pattern where uncontrolled is idiomatic

**Bug**

```tsx
const [text, setText] = createSignal("");
return (
  <input
    value={text()}
    onInput={e => setText(e.currentTarget.value)}
    onKeyDown={e => {
      if (e.key !== "Enter") return;
      const v = text().trim();
      if (!v) return;
      t.addTodo(v);
      setText("");
    }}
  />
);
```

**Fix**

```tsx
return (
  <input
    onKeyDown={e => {
      if (e.key !== "Enter") return;
      const v = e.currentTarget.value.trim();
      if (!v) return;
      t.addTodo(v);
      e.currentTarget.value = "";
    }}
  />
);
```

No signal, no `value` binding, no `onInput`. The DOM holds the input's state;
Solid only needs to react when something *outside* the input wants to read or
clear it.

**Why I got it wrong**

React makes uncontrolled inputs awkward (`defaultValue`, refs, no clear way to
reset), so AI training data is dominated by controlled `value + onChange`
pairs. I reached for that pattern reflexively even though Solid imposes no
such tax: the JSX runtime doesn't replay or reconcile DOM state, so reading
`e.currentTarget.value` on the event and assigning back to clear is fine and
common.

**What would have prevented it**

- CHEATSHEET: a small "Form inputs" subsection contrasting both styles, with
  the explicit guidance that **uncontrolled is the default** and a controlled
  signal is only needed when external code reads or writes the input value.
  Show the imperative-clear pattern (`e.currentTarget.value = ""`) so the AI
  has a concrete substitute for `setText("")`.
- The TodoMVC example itself, once correct, is a reference implementation.
  Calling out in its README that the new-todo input is intentionally
  uncontrolled would help (AI tends to read examples for patterns).

---

## 3. Flat object return / context shape instead of `[state, actions]` tuple

**Bug**

```ts
// hook
return {
  get todos() { return data.todos; },
  meta,
  addError,
  addTodo,
  toggleTodo,
  removeTodo,
  // ...
};

// callsite
const t = useTodos();
t.todos // reactive read
t.addTodo(v) // action
```

**Fix**

```ts
// hook
const state = {
  get todos() { return data.todos; },
  meta,
  addError,
};
const actions = { addTodo, toggleTodo, removeTodo, ... };
return [state, actions] as const;

// callsite — destructure ONLY the actions; keep `state` whole so reads stay reactive
const [state, { addTodo, toggleTodo }] = useTodos();
state.todos       // reactive read through the proxy
addTodo(v)        // local stable function
```

**Why I got it wrong**

`createSignal` returns `[get, set]`, `createStore` returns `[store, setter]`,
and Solid's whole convention is "tuple of (read, write)". Custom hooks should
mirror it — `[state, actions]` for a composite — and that idiom *does* appear
in Solid 1.x docs and 2.0 examples. I instead defaulted to React's
`useFooReturnsObject()` shape (`{ todos, addTodo, ... }`) which feels
egonomic in JSX (`t.x`) but loses two important properties:

1. **Read/write split is invisible.** With a flat object, the consumer can't
   tell at a glance which keys are state (must stay accessor-bound) vs which
   are stable callbacks. With `[state, actions]`, destructuring rules tell
   you: never destructure `state`, always-safe to destructure `actions`.
2. **Mirroring the primitive shape teaches the right reflex.** Every time AI
   sees `[state, actions]` it's reinforced that "tuple = primitive shape" is
   the canonical Solid pattern. Flat-object returns dilute that signal.

**What would have prevented it**

- CHEATSHEET: a "Custom hooks return shape" line under "Stores" /
  "Composition". Concretely:
  > Hooks that bundle state + writers should return `[state, actions]`. Don't
  > destructure `state` (reads must stay bound to the proxy/getter); always
  > safe to destructure `actions`.
- TodoMVC example becomes the reference implementation of the pattern. AI
  reading `examples/todos/src/todos.ts` will see `[state, actions] as const`
  in the wild.
- (Stretch) JSDoc snippet on `createContext` showing a hook that returns the
  tuple, so completion in IDEs nudges this shape.

---

## 4. Mixing signal accessors and store reads in the same `state` object

**Bug**

```ts
const state = {
  get todos() { return data.todos; },
  meta,                  // store — reads as `state.meta[id]`
  addError,              // signal accessor — reads as `state.addError()`
};
```

Consumers had to know which keys are call-this and which are read-this.

**Fix**

Wrap the signal in a getter so every read is uniform:

```ts
const state = {
  get todos() { return data.todos; },
  get addError() { return addError(); },
  meta,
};
// state.todos, state.addError, state.meta[id] — all property reads.
```

**Why I got it wrong**

The Solid 1.x mental model of "signal returns an accessor function, call it"
got dragged into composing a state surface. When bundling state for a hook
return, the right move is to expose everything via property reads (getters or
store properties) so the consumer sees one consistent shape. Mixing
`accessor()` calls with proxy property access in the same object is a leaky
abstraction.

**What would have prevented it**

- CHEATSHEET: when describing custom-hook returns, explicitly state "expose
  all reactive reads as property reads (getters or store properties), not as
  callable accessors. Internal signals stay internal."

(This is partially obviated by the bigger lesson in finding #5 — often you
don't need the signal at all, the data should live in a single store.)

**Resolution (this round)**

Added a paragraph to `createStore` JSDoc framing this as a category error:
*"Store properties hold plain values, not accessors. The proxy already
tracks reads per-property — wrapping a value in `() => state.foo` produces
a getter that won't track when called."* No example block (the prose names
the misconception directly). The mixed-shape custom-hook case is also
embodied in the kitchen-sink, which uses a `[store, actions]` tuple with
no signal accessors leaking through.

---

## 5. Multiple reactive stores instead of a single derived pipeline (the big one)

**Bug**

I modeled the kitchen sink as three reactive sources:

- `data` — `createOptimisticStore` for server todos
- `meta` — `createStore` for per-id `{ saving, error }`
- `addError` — `createSignal` for failed-add UI message

Each action wrote to all three. Per-action `try/catch` had to keep them
synchronized. Retry logic was hand-rolled (`retry(id, completed) →
toggleTodo`), only worked for toggles.

**Fix (the architecture I should have reached for)**

A single `createOptimisticStore` whose **derived projection function is a pipeline**:

```
server fetch  →  applyErrors() overlay  →  optimistic overlay (built-in)  →  consumer
```

```ts
const Errors: Record<string, TodoError> = {}; // plain JS — non-reactive

const [todos, setTodos] = createOptimisticStore<Todo[]>(
  async () => {
    const todos: Todo[] = await api.getTodos();
    applyErrors(todos, Errors);
    return todos;
  },
  [],
  { key: "id" }
);

const actions = {
  toggleTodo: action(function* (id, completed) {
    setTodos(t => { /* optimistic write, sets pending: true */ });
    try   { yield api.toggleTodo(id, completed); delete Errors[id]; }
    catch { Errors[id] ||= { type: "toggleTodo", args: [id, completed] }; }
    refresh(todos);
  }),
  // ... same shape for every action
  retryTodo(todo) {
    return actions[todo.error.type](...todo.error.args);
  },
};
```

Key insights I missed:

1. **The projection is the place to compose layers.** The derived form of
   `createOptimisticStore` exists precisely so you can produce a *function*
   of (server data + side channels) on every refresh. Errors are *applied*
   inside the projection, not *stored* as reactive state.
2. **`Errors` doesn't need to be reactive.** It's only ever read inside the
   projection, and the projection is invalidated by `refresh(todos)`. So
   "mutate `Errors`, then call `refresh`" is the trigger pattern. Making it
   a `createStore` was overkill and introduced sync problems.
3. **`pending` and `error` live on the same `Todo`.** No parallel `meta[id]`
   map. `pending` is set in the optimistic write and naturally clears with
   the overlay; `error` is painted on by `applyErrors` from the side-channel.
4. **Failed adds re-materialize as ghost items.** `applyErrors` re-inserts a
   missing-from-server todo (whose add failed) into the right sorted position
   with `error` set. Add-error UI is then just *another item-level error*;
   no separate top-level `addError` signal needed.
5. **Retry is one method.** Read `todo.error.type` and `todo.error.args`,
   replay the action. No per-action hand-rolling.
6. **Every action ends `refresh(todos)`.** Always. That's the seam where the
   layered view is recomposed.

**Why I got it wrong**

This is the single most important AI-failure mode in the example, and it has
two compounding causes:

a. **Redux/React mental model.** Training data is dominated by reducers that
   imperatively write each kind of state (data, loading, error) into the
   same flat slice. The idea of *deriving* error state through a projection
   over a side-channel is rare in the JS ecosystem — closer to Elm or
   reactive-stream patterns than React.
b. **`createOptimisticStore`'s derived form is not framed as a pipeline.**
   The current JSDoc says it's "a projection store whose authoritative value
   is recomputed by `fn` and whose optimistic overlay reverts after each
   transition". That's accurate but doesn't tell the reader that this is
   *the right place* to compose multiple data sources (server + errors +
   anything else). Without that framing, AI defaults to "I have N pieces of
   state, I need N stores".

**What would have prevented it**

- **Top-priority: a worked example showing the layered-overlay pattern.** A
  CHEATSHEET section or canonical example titled something like "Composing
  data + error + optimistic in one store" with the exact pattern from this
  finding (server projection + non-reactive side-channel + every action
  ends `refresh`). This is the "AI doesn't know about layered overlays"
  gap, and the only way to close it is to make the pattern impossible to
  miss in the discovery surface.
- **`createOptimisticStore` JSDoc should be reframed.** Today it talks
  about "tentative writes" and "auto-revert". It should *also* explicitly
  describe the derived form as "your fan-in point for layering server data
  with non-reactive side-channels (e.g. local error overlays)" with a
  one-liner example of exactly this composition.
- **Action JSDoc should show the canonical pattern**: optimistic write →
  yield API → write to side-channel on failure → `refresh(source)`. Today
  the example only shows yield + mutate same store, not the side-channel +
  refresh combo.
- **Make the TodoMVC example the reference.** Once correct it should be
  prominently linked from the cheatsheet/JSDoc as "this is the pattern".

**Resolution (this round)**

- Kitchen-sink (`examples/todos/src/todos.ts`) carries a header comment
  naming the three lifetime layers (Persistent / Ephemeral / Optimistic),
  the strict order they compose in, and the explicit reason the order
  isn't a choice — the primitive composition enforces it. The comment
  also notes that splitting the layers into separate stores is possible
  but exposes intermediate views as accidental footguns; the
  single-primitive form is canonical.
- The architectural framing stays in the example, not in JSDoc on
  `createOptimisticStore`. JSDoc is symbol-level — what the function
  does and how to call it. Layered-overlay architecture is example-level
  because the pattern is composition-shaped, not symbol-shaped.
- The kitchen-sink itself becomes the worked example called out in the
  "what would have prevented it" list above. Future cheatsheet / docs
  references to the layered pattern can link here.

---

## 6. TS4058 leak from `createOptimisticStore` derived form (framework bug — fixed)

**Status:** classified as a framework bug, fixed in this session via the
`Refreshable<T>` public alias.

**Bug**

Adopting the right architecture, the inferred return type of `createTodos`
contained `Store<Todo[]> & { [$REFRESH]: any }`, which TypeScript refused to
emit as a public type:

```
src/todos.ts(43,17): error TS4058: Return type of exported function has or
  is using name '$REFRESH' from external module ... but cannot be named.
```

This affected any user-defined hook that wrapped `createOptimisticStore` (or
`createProjection` / projection-form `createStore`) — the consumer was
forced to write an explicit return annotation just to dodge an internal
brand leaking from a deep import path. That's not an "AI got it wrong"
problem; it's a public-API ergonomics bug.

**Fix (framework)**

Introduced `Refreshable<T> = T & { readonly [$REFRESH]: any }` next to
`$REFRESH` in `@solidjs/signals` and replaced every `Store<T> & { [$REFRESH]:
any }` site (signals projection / optimistic-store / store overloads, and
the `solid-js` hydration wrappers) with `Refreshable<Store<T>>`. The alias
is re-exported from both `@solidjs/signals` and `solid-js` (client +
server), so the inferred type is now nameable from any consumer.

After the fix, the example's hook compiles without an explicit return
annotation:

```ts
export function createTodos() {
  const [todos, setTodos] = createOptimisticStore<Todo[]>(/* ... */);
  // ...
  return [todos, actions] as const;
}
```

**Why this was a framework bug, not an AI gap**

The brand symbol is part of how `refresh()` distinguishes "refreshable
sources" from plain functions at runtime; it has to be branded somewhere.
Inlining the brand into every overload was correct shape-wise, but TS4058
fires whenever a brand reaches an exported declaration via a path that the
emitter can't name. Wrapping the brand in a *named* alias is the standard
fix and costs nothing at runtime.

**What would have prevented it sooner**

Catching this needed either a downstream consumer (this kitchen-sink
example) or a `declaration: true` build over a hook wrapper somewhere in
the test suite. Worth adding a tiny type-test that wraps each derived
primitive in an exported function and asserts that the inferred return
type is nameable, so a future regression surfaces in CI rather than in a
user repo.

---

## 7. `<Loading>` boundary scoped too wide — duplicating shared chrome

**Bug**

```tsx
<Loading
  fallback={
    <section class="todoapp">
      <h1>todos</h1>
      <p>Loading…</p>
    </section>
  }
>
  <section class="todoapp">
    <Header />
    <MainSection />
    <Footer />
  </section>
</Loading>
```

The `<section class="todoapp">` and the `<h1>todos</h1>` shell appear in
both the fallback and the resolved children. When the initial fetch
resolves, the entire fallback subtree is torn down and a fresh subtree is
mounted — including the outer chrome that was identical in both.

**Fix**

Move the `<Loading>` boundary to wrap *only* the components that actually
do suspending reads. Anything that doesn't depend on the async data stays
outside, so it mounts once and stays put across the transition:

```tsx
<section class="todoapp">
  <Header />
  <Loading fallback={<p class="loading">Loading…</p>}>
    <MainSection />
    <Footer />
  </Loading>
</section>
```

`<Header />` reads no async data, so it lives outside. The fallback is now
just the small placeholder it should be.

**Why I got it wrong**

In React, suspense boundaries are typically wrapped around large feature
shells because the diff/reconciler hides the cost of "throwing away" the
fallback DOM — both fallback and content are described as virtual nodes,
and React reconciles them. So the React habit is "wrap the whole route /
page", and then style the fallback to look like the loaded shell to avoid
layout shift.

In Solid, JSX creates real DOM nodes. The fallback subtree is *built*; the
content subtree is *built*; on transition the fallback subtree is *removed*
and the content subtree is *inserted*. There is no diff to amortize the
duplication. The right reflex is "scope the boundary as narrowly as
possible, around just the suspending reads, and let everything else live
outside."

**What would have prevented it**

- CHEATSHEET / `<Loading>` JSDoc: add a "Scope" note: *"Place `<Loading>`
  as close to the suspending read as possible. JSX in Solid creates real
  DOM, so anything outside the boundary mounts once and persists across
  the transition; anything inside is rebuilt when the boundary swaps."*
  Concrete contrast: bad/good example pair, exactly like this finding.
- Expanding the same point: the fallback should be *only* the affordance
  (spinner / skeleton / message), not a re-statement of the loaded UI's
  outer chrome.

**Resolution (this round)**

Added a sentence to client-side `<Loading>` JSDoc (the user-facing
description; server-side stub JSDoc isn't what hover resolves to):
*"Scope `<Loading>` around the data-dependent slot, not the surrounding
shell. Wrapping layout chrome means revalidation replaces the entire
screen with the fallback; rendering chrome outside the boundary keeps it
stable while only the affordance flips."*

No `@example` block — the lesson is contrast, not a single canonical
shape, and contrast doesn't fit `@example` cleanly.

---

## 8. N-yield loops faking bulk actions

**Bug**

```ts
toggleAll: action(function* (completed: boolean) {
  const ids = todos.filter(t => t.completed !== completed).map(t => t.id);
  setTodos(t => { /* mark all pending */ });
  for (const id of ids) {                    // ← N requests
    try { yield api.toggleTodo(id, completed); delete Errors[id]; }
    catch { Errors[id] ||= { ... }; }
  }
  refresh(todos);
}),
```

A bulk operation should be one round-trip, not N. This was laziness on my
part — extending the per-item action shape because it was already there.

**Fix**

Add real bulk endpoints (`api.toggleAll(ids, completed)`,
`api.clearCompleted(ids)`) and yield them once:

```ts
toggleAll: action(function* (completed: boolean) {
  const ids = todos.filter(t => t.completed !== completed).map(t => t.id);
  setTodos(t => { /* mark all pending */ });
  try {
    yield api.toggleAll(ids, completed);
    ids.forEach(id => delete Errors[id]);
  } catch {
    // Fan the bulk failure out to per-item Errors so each failed item
    // gets its own retry affordance.
    ids.forEach(id => { Errors[id] ||= { type: "toggleTodo", args: [id, completed] }; });
  }
  refresh(todos);
}),
```

Side benefit: a bulk failure naturally decomposes into per-item retries
through `retryTodo` — no special "bulk retry" path needed. This is another
expression of the layered-overlay architecture from finding #5.

**Why I got it wrong**

Pure laziness — I had a working per-item `toggleTodo` action and reached
for `for (const id of ids) yield ...` without thinking about the wire cost.
A real backend wouldn't have an N-request bulk operation; the example
shouldn't pretend either.

**What would have prevented it**

- This is less an AI-guidance gap and more a "code review" failure. A
  reasonable rule for the cheatsheet's actions section: *"One yield per
  network round-trip. Bulk operations get bulk endpoints."*
- The example, once corrected, demonstrates the right pattern — and shows
  off the "fan a single failure into per-item retries" trick that the
  side-channel architecture makes natural.

**Resolution (this round)**

Deliberately not actioned in JSDoc. This isn't an `action`-shaped
finding — the lesson is "use a bulk endpoint when one exists", which is
API-modeling common sense, not Solid's `action` semantics. Putting it in
JSDoc would teach general engineering hygiene under the wrong symbol.
The kitchen-sink already shows the correct shape (one
`yield api.toggleAll(...)`); that's all the carrying it needs.

---

## 9. Imperative `splice` loops instead of functional `filter` returns

**Bug**

```ts
removeTodo: action(function* (id: string) {
  setTodos(t => {
    const i = t.findIndex(x => x.id === id);
    if (i >= 0) t.splice(i, 1);
  });
  ...
}),

clearCompleted: action(function* () {
  ...
  setTodos(t => {
    for (let i = t.length - 1; i >= 0; i--) {
      if (set.has(t[i].id)) t.splice(i, 1);
    }
  });
  ...
}),
```

I went hunting for an imperative way to remove items from the draft and
arrived at `findIndex + splice` (or worse, a reverse-iteration `splice`
loop for bulk removal). It works, but it's noisy and indexing-bug-prone.

**Fix**

Draft setters can *return* a new array. The setter's return path is
**shallow** — arrays are replaced by index (with length adjusted), objects
are shallow-diffed at the top level. For arrays of objects with stable
references (which `filter` produces), the surviving objects re-occupy
shifted indices and the trailing slots are truncated, so consumer-side
identity (the object refs) survives. So functional array methods are clean
and safe for the remove case:

```ts
setTodos(t => t.filter(todo => todo.id !== id));
setTodos(t => t.filter(todo => !todo.completed));
```

**Important distinction.** The setter is shallow. The keyed-reconcile
behaviour belongs to the *projection-fn* return path
(`createStore(fn, …, { key })`, `createProjection`,
`createOptimisticStore(fn, …)`), where the function's return is
reconciled against the previous draft by `options.key` (default `"id"`)
so surviving items keep their proxy identity. Don't conflate the two.

**Why I got it wrong**

The cheatsheet stated draft setters can return a value (line 469:
*"Optionally return a value for shallow replacement"*) and showed an
example on line 126 (`setStore(s => ({ ...s, list: [] }))`). But:

1. **Framing.** "Shallow replacement" sounds destructive — like it loses
   item identity. The reader infers "I'd better keep mutating the draft to
   preserve reactivity per-item". (For arrays the index-by-index assign
   actually preserves item refs, but the framing didn't say so.)
2. **The example shape didn't match.** Object spread `{ ...s, list: [] }`
   is full-store replacement. There was no example of returning from an
   *array* setter (`setItems(t => t.filter(...))`), which is the exact
   shape that comes up most often.

I also initially mis-routed the fix: the first draft of this finding
claimed *"with the store keyed by `id`, reconciliation preserves item
identity for items that remain"*. That conflated the setter path with the
projection-fn path. The setter does **not** key-reconcile. The reason
`filter`-return works is positional (index-replace + length truncation
over same-reference survivors), not keyed. The corrected JSDoc/cheatsheet
edits keep the keyed-reconcile claim pinned to the projection-fn path
only.

**What would have prevented it**

- CHEATSHEET: a "Stores: array operations" entry directly showing
  `setItems(t => t.filter(...))` and `setItems(t => [...t, item])`,
  framed as *"return when mutation is awkward (most often filter/remove);
  arrays index-replace + length-adjust, objects top-level shallow diff;
  no keyed reconciliation — for that, use the projection form"*. Pair it
  with the imperative `setItems(t => t.push(item))` so both styles are
  visibly legitimate.
- `createStore` / `createOptimisticStore` / `createProjection` JSDoc:
  document that the draft callback can either mutate or return, **with
  the setter-vs-projection-fn distinction made explicit**. The
  `createProjection` JSDoc previously shipped an imperative
  `draft.length = 0; for ... draft.push(...)` example — the
  splice-loop anti-pattern in the doc itself. Replaced with a
  `return …filter(...)` example that names the keyed-reconcile guarantee.

**Resolution (this round)**

- `StoreSetter<T>` JSDoc rewritten: mutation-first canonical form,
  precise return semantics (index/length for arrays, top-level diff for
  objects), explicit *"not keyed reconciliation — use the projection
  form"* note.
- `createStore` JSDoc: mutation example stays the lead; added a
  `setState(s => s.todos.filter(...))` line, kept the object-spread
  replace, lead text reframed.
- `createOptimisticStore` JSDoc: existing optimistic-add mutation
  example kept as the lead; added a second action showing
  `setTodos(t => t.filter(...))` for the remove case.
- `createProjection` JSDoc: replaced the imperative
  `draft.length = 0; for...push` example with two examples — a
  per-property mutation form (counters) and a `return …filter(...)`
  form, with the keyed-reconcile guarantee called out.
- CHEATSHEET: line 126 reframed (filter-return as the canonical use of
  the return form, distinct from object replace, no keyed reconcile);
  line 493 reframed.
- The wrapper-side JSDoc on `solid-js`'s `createStore` /
  `createOptimisticStore` / `createProjection` re-bindings is currently
  empty (the wrappers redeclare types without `@inheritDoc`). Fixing
  that is part of the broader JSDoc-pass workstream, not #9 — flagged
  separately.

---

## 10. Reflexive Context + `useX`-with-throw wrapper

**Bug**

For the kitchen-sink TodoMVC, I introduced a `TodosContext` and a
`useTodos` hook of the canonical React shape:

```ts
const TodosContext = createContext<ReturnType<typeof createTodos>>();
const useTodos = () => {
  const ctx = useContext(TodosContext);
  if (!ctx) throw new Error("TodosContext missing");
  return ctx;
};
```

…then wrapped the whole `<App>` in `<TodosContext value={createTodos()}>`
and called `useTodos()` from each consumer.

The actual prop-drilling depth in this app is one or two hops:

- `App` → `Header` (`addTodo`)
- `App` → `MainSection` → `TodoItem` (`toggleTodo`, `removeTodo`, `retryTodo`)
- `App` → `Footer` (`todos`, `clearCompleted`)

Plain prop passing would have read more directly. Context here is fine —
not wrong, not broken — but it's not pulling its weight, and the wrapper
hook is a transplanted React idiom that wasn't earning its complexity.

**Why I got it wrong**

Two converging React reflexes:

1. **"Shared state → Context."** In React, Context is the canonical
   anti-prop-drilling tool, so any state that more than one component
   reads gets one. There's no comparable cost gradient cue: Solid's
   Context is also cheap, but the *break-even depth* before it earns its
   keep is rarely discussed.
2. **"Context → `useX` wrapper with throw."** The non-null context hook
   is a React community standard:
   - TS narrowing (no `| undefined` at consumer sites).
   - Runtime guard for "forgot the Provider".

   Both motivations are real in Solid too, so the pattern transplants
   without resistance. But in Solid, `createContext<T>(defaultValue)`
   accepts a real default and removes the narrowing problem outright —
   at the cost of swallowing missing-Provider errors silently. The
   trade-off is rarely surfaced.

The first reflex is the bigger issue: I never paused to ask *"is the
drilling actually deep?"* before reaching for Context. The second reflex
just compounded it.

**What's also worth noting**

When I tried to self-correct and suggested module-level singletons as
"more idiomatic", that was *also* wrong. In Solid, Context is preferred
over module globals because:

- Module globals leak across SSR requests.
- Context disposal is owner-tied; globals stick around.
- Context allows multiple instances under different subtrees later.

So the correct mental model is:

> **Default to props.** When drilling exceeds ~3 hops or fans out widely,
> reach for Context — not module globals.

I had three options in my head (props / Context / module global) and
picked the wrong one *twice* in a row.

**What would have prevented it**

- CHEATSHEET: a "When to use Context" line that says *"prefer props for
  shallow trees; use Context when drilling exceeds a few hops or fans
  out — never use module globals for shared mutable state in apps that
  may render server-side"*.
- CHEATSHEET / `createContext` JSDoc: contrast the two patterns
  side-by-side so the trade-off is visible:
  - `createContext<T>(defaultValue)` — narrowed type, silent fallback.
    Realistically only useful for plain-value contexts (theme, locale,
    config). See **Finding 11** — for reactive payloads (signals,
    stores, `[state, actions]` tuples) the default slot is essentially
    unusable.
  - `createContext<T>()` + thrower — `| undefined` type, loud failure.
    The de facto default for Solid because reactive payloads can't have
    sensible owner-free defaults.
- A short example in the cheatsheet showing prop passing as the *first
  cut*, with Context introduced only when the example grows past
  shallow drilling — so the gradient is visible rather than an
  all-or-nothing choice.

**Resolution (this round)**

The kitchen-sink now carries both endpoints of the criterion side by
side, which is what makes the "gradient" legible at all (see also
Finding #14, which forced the demonstration into existence):

- **Todos → Context.** 4 consumers, mixed depth (TodoItem is 3 hops
  through a `<For>` callback), `[state, actions]` tuple with 6
  fanned-out methods. Threading this via props is the moment Context
  starts pulling its weight.
- **Filter → Props.** 2 peer consumers (MainSection, Footer), 1 hop
  each, single accessor. A `FilterContext` here would be visible
  over-engineering against `useHashFilter`'s App-level read.

A reader of `examples/todos/src/app.tsx` sees both choices justified
by the structural difference of the state being shared, not as
arbitrary stylistic picks. That's the demonstration the cheatsheet
line above couldn't carry on its own.

The framework-side half of this finding was settled by Finding #11
(default-less `createContext` now throws on missing Provider, removing
the `useX`-with-throw boilerplate). The "prefer props for shallow
trees" half is now demonstrated, not just documented.

---

## 11. Context default-less form should throw on missing Provider (framework change — implementing)

**Status:** classified as a framework change. Implementation in this
session: make `createContext<T>()` (no default) return `Context<T>` and
have `useContext` throw on missing Provider, eliminating the
wrapper-with-throw boilerplate.

**The observation**

`createContext<T>(defaultValue)` came from React, where contexts
typically carry plain values (or stable references) and a default is a
meaningful fallback — often functioning as "global value with optional
subtree override". In Solid that role doesn't exist:

> **If you want truly global state, you don't use Context.** A module-
> scope signal/store *is* a global. Context is the tool you reach for
> when state should be scoped to a subtree, which by definition means
> a Provider is mandatory.

Once you accept that framing, the default-value slot collapses to a
narrow case: contexts whose value is a primitive with a meaningful
static fallback (`"light"`, `"en-US"`, frozen config). Even there,
correct programs mount a top-level Provider, so the default rarely
fires.

For everything else — the dominant case in real Solid apps — the
context carries a **reactive reference** (signal getter, store,
`[state, actions]` tuple, tracked service):

1. **The real value can't be constructed at module top.** Anything
   built from `createOptimisticStore` / `createStore` / `createSignal`
   / `createMemo` needs an owner; the Provider's render is the
   earliest place that's available. The `createContext` default slot
   runs before any owner exists.
2. **A "stub" default is silent failure.** Defaulting a Todo context
   to `[() => [], { addTodo: noop }]` makes consumer calls quietly
   no-op when the Provider is missing — strictly worse than a loud
   throw. Bugs survive review and ship.
3. **An "eternal" default is a lifecycle leak.** Constructing the real
   reactive object at module load gives you an owner-less reactive
   graph that never disposes, leaks across SSR requests, and entangles
   all consumers regardless of subtree.

So the API as inherited from React shapes the wide path
(`createContext<T>()` + wrapper hook with throw) as ergonomically
worse than the narrow path. That's backwards for Solid.

**The change**

- `createContext<T>()` (no default) → returns `Context<T>`, *not*
  `Context<T | undefined>`. `useContext(ctx)` throws on missing
  Provider with a clear `ContextNotFoundError`.
- `createContext<T>(defaultValue)` → unchanged: returns `Context<T>`,
  `useContext(ctx)` falls back to `defaultValue` if no Provider is
  mounted. Reserved for the narrow primitive-default case.

The wrapper-with-throw idiom disappears at the source: every
`const useTodos = () => { const t = useContext(Ctx); if (!t) throw …;
return t; }` becomes a direct `useContext(Ctx)` call.

```ts
// before
const TodosContext = createContext<TodosTuple>();
const useTodos = () => {
  const ctx = useContext(TodosContext);
  if (!ctx) throw new Error("missing Provider");
  return ctx;
};

// after
const TodosContext = createContext<TodosTuple>();
// useContext(TodosContext): TodosTuple — throws if no Provider.
```

**Why this is a framework change, not just a doc change**

The prior version of this finding floated `createRequiredContext<T>()`
as a possible new API. Adding API surface to compensate for an
ergonomic gap in an existing primitive is the wrong cost: the
default-less form was already "no sensible default" semantically; we
just hadn't aligned the runtime/types with that meaning. Changing
`createContext` directly:

1. Adds zero API surface.
2. Makes the wide path the loud path.
3. Drops `T | undefined` narrowing tax from every consumer call site.
4. Leaves the legitimate primitive-default case unchanged.

**Type cost**

Type-cost neutral. The default-less form already required an explicit
generic (`createContext<T>()` — there's nothing to infer from); the
change just lets `useContext` return `T` instead of `T | undefined`,
which removes narrowing rather than adding annotation.

**Compatibility**

Breaking for any consumer that today calls `useContext(ctx)` on a
default-less context and branches on `undefined`. Likely rare in
practice — the dominant pattern is the wrapper-with-throw, which now
just becomes a direct call. Anyone genuinely using the silent-fallback
behaviour can migrate by passing an explicit default. Reasonable
breaking change for 2.0 betas.

**What would have prevented the original confusion**

- CHEATSHEET / Context section: lead with the default-less form as the
  canonical Solid idiom; frame `createContext<T>(default)` as a
  specialized form for plain-value contexts (theme, locale, config),
  not as the recommended way to "avoid" the missing-Provider error.
- JSDoc on `createContext`: explicitly say *"if you want global state,
  use a module signal/store — Context is for subtree scope, and a
  Provider is mandatory in the default-less form"*.
- 2.0 RFC / migration docs: ship a "default-less form throws" entry
  alongside the existing `Context.Provider` → context-as-provider
  migration, calling out that `useX`-with-throw wrappers can be
  deleted (one-line direct `useContext` call replaces them).

**Implementation status (this session)**

- `solid-js` `createContext` collapsed to a single overload returning
  `Context<T>`; `defaultValue` typed `T | undefined` on the Context
  object (runtime fact). Both client and server.
- JSDoc rewritten on both `createContext` and `useContext` — leads
  with default-less form, contrasts with React/global-state framing,
  warns against `useX`-with-throw wrappers.
- `examples/todos` drops `useTodos` wrapper entirely; all call sites
  use `useContext(TodosContext)` directly.
- CHEATSHEET Context section updated to match.
- 2.0 migration docs updated:
  `documentation/solid-2.0/02-signals-derived-ownership.md` and
  `documentation/solid-2.0/MIGRATION.md` both include a
  "default-less form throws on missing Provider" subsection with
  before/after.
- Changeset `.changeset/context-default-less-throws.md` (`patch` per
  pre-release policy).

---

## 12. PascalCasing render callbacks like they're Components

**Bug**

```tsx
function ErroredFallback(err: unknown, reset: () => void): JSX.Element {
  return (
    <div class="app-error">
      <p>Something went wrong: {String(err)}</p>
      <button onClick={reset}>Reset</button>
    </div>
  );
}

export function App() {
  return (
    <Errored fallback={ErroredFallback}>
      ...
    </Errored>
  );
}
```

`ErroredFallback` is named like a Component (PascalCase) but it's not
one — it's a render callback that the `<Errored>` boundary invokes
internally with `(err, reset)` arguments. You can't `<ErroredFallback />`
it; the JSX runtime would call it with a props object, not the
err/reset pair.

The user flagged it: *"Errored Fallback seems wrong, passing Component
function instead of creating it"*.

**Why I got it wrong**

In React, *anything* PascalCase that returns JSX is a Component, and
any reusable JSX-returning function gets a Component-style name by
default. I imported the convention without checking whether the call
site treated it like a Component.

In Solid, the same JSX surface hosts two distinct kinds of functions:

1. **Components** — `(props: P) => JSX.Element`, called via JSX
   (`<Foo {...props} />`). PascalCase is correct here.
2. **Render callbacks** — `(arg1, arg2, ...) => JSX.Element`, called by
   a *parent* component / control flow primitive that knows the
   argument shape:
   - `<Show when={x}>{v => <Foo v={v()} />}</Show>` — narrowed accessor.
   - `<For each={xs}>{(x, i) => <Foo x={x()} />}</For>` — item + index
     accessors.
   - `<Errored fallback={(err, reset) => ...}>` — error + reset.
   - `<Repeat count={n}>{i => ...}</Repeat>` — numeric index.

The two have **incompatible call shapes**. A render callback isn't
mountable as `<X />` because the parent supplies positional arguments,
not props. PascalCasing the callback misleads the reader (and the
author) into thinking it's interchangeable with a Component.

**Fix**

Inline it (preferred for one-off uses):

```tsx
<Errored
  fallback={(err, reset) => (
    <div class="app-error">
      <p>Something went wrong: {String(err)}</p>
      <button onClick={reset}>Reset</button>
    </div>
  )}
>
  ...
</Errored>
```

If reused, name it camelCase to signal "callback, not Component":

```tsx
const renderAppError = (err: unknown, reset: () => void) => (
  <div class="app-error">...</div>
);

<Errored fallback={renderAppError}>...</Errored>
```

**What would have prevented it**

- CHEATSHEET: a "Components vs render callbacks" entry that lists the
  flow-control primitives whose children/fallbacks are callbacks
  (`Show`, `For`, `Repeat`, `Errored`, `Switch/Match` function-child
  forms) and explicitly says *"PascalCase = Component (called via
  JSX). camelCase = render callback (called by the parent with
  positional args). Don't conflate them."*
- A worked example of a `fallback={...}` callback inlined, alongside
  a Component being used elsewhere in the same snippet, so the visual
  contrast is concrete.

**Resolution (this round)**

Deferred to lint, alongside the rest of the `eslint-plugin-solid 2.0`
workstream. New rule logged in `.notes/ai-readiness.md` Part 2 row 12
and the Ship list entry: `no-pascalcase-render-callback` — function
declared/passed in a render-callback prop position (e.g. `fallback`,
`children` of `<Show>`/`<Match>`) that returns `JSX.Element` and isn't
called via JSX must not be PascalCase; conversely, a PascalCase function
used as a value not as `<Comp />` is suspicious. The mechanical pattern
is exactly what lint catches well, and CHEATSHEET prose is the wrong
home for naming-convention enforcement.

---

## 13. Passing options that just restate the default

**Bug**

```ts
const [todos, setTodos] = createOptimisticStore<Todo[]>(
  async () => { ... },
  [],
  { key: "id" }   // ← this is the default
);
```

`createOptimisticStore` (and `createProjection`, projection-form
`createStore`) reconcile by `"id"` when no `key` option is given —
that's literally `options?.key || "id"` in the source. Passing
`{ key: "id" }` adds noise without changing behavior.

**Why I got it wrong**

Two reflexes:

1. **"Be explicit" defensive habit.** When I'm not 100% sure a default
   matches what I want, I write the option to be safe. In a teaching
   example that backfires — the explicit form makes `id` look like a
   choice the user has to make, not a canonical default the API leans
   into.
2. **I treated `{ key: "id" }` as a clarifying hint** for readers
   ("this store reconciles by id"). But removing it doesn't lose that
   information — the convention *is* that data has `id`. Restating the
   default as if it were a config decision teaches the wrong lesson.

**Why the default deserves to be visible (by absence)**

`id` isn't a random default; it's a deliberate convention baked into
the projection/optimistic API. Showing the call as `(fn, [])` with no
options communicates *"data has `id`, so this just works"*. Showing
`(fn, [], { key: "id" })` communicates *"you have to tell it the
key"* — which is the opposite of the API's intent.

For non-`id` keys (`{ key: "uuid" }`, `{ key: t => t.slug }`) the
option is informative. For `id`, it's a tax on the example.

**What would have prevented it**

- CHEATSHEET: when documenting `createOptimisticStore` /
  `createProjection`, show the no-options form first as the canonical
  case (*"data with `id` reconciles automatically"*) and introduce
  `{ key: ... }` only when the example actually has a non-`id` key.
- JSDoc: explicitly note *"key defaults to `"id"`; only specify when
  your data uses a different identity field"*. The current JSDoc
  mentions the default in passing but doesn't actively discourage
  restating it.
- General principle worth surfacing once in the cheatsheet: *"don't
  pass options that match defaults — defaults are part of the API's
  vocabulary, not implementation details to hedge against."*

**Resolution (this round)**

- `createOptimisticStore` JSDoc: added line *"`options.key` defaults to
  `"id"`; specify it only when your data uses a different identity
  field. Restating the default just adds noise."*
- `createProjection` JSDoc: tightened the `@param options` block to
  *"`key` defaults to `"id"`; specify it only when your data uses a
  different identity field (e.g. `{ key: "uuid" }` or
  `{ key: u => u.slug }`)."*
- Cheatsheet untouched here — the JSDoc on the symbol is where this
  guidance is most discoverable.
- Lint rule deferred to the `eslint-plugin-solid 2.0` workstream: new
  rule logged in `.notes/ai-readiness.md` Part 2 row 13 and the Ship
  list entry: `no-restated-default-options` — for a documented set of
  `(fn, options)` primitives with known defaults, flag option entries
  that match the default. Low-priority; the JSDoc edit covers the
  pedagogical case, lint covers the mechanical one.

---

## 14. Module-scope signal for view state (instead of an owner-scoped custom primitive)

**Bug**

The TodoMVC filter was modeled as a top-level signal in `app.tsx`,
with the `hashchange` listener attached at module scope:

```ts
const [filter, setFilter] = createSignal<Filter>(parseHash(location.hash));
window.addEventListener("hashchange", () => setFilter(parseHash(location.hash)));
```

Sitting *next to* a properly owner-scoped `TodosContext` value in the
same file, no less. The example contradicted itself.

**Fix**

Pull the state into a custom primitive owned by the call site:

```ts
// src/filter.ts
export function createHashFilter(): () => Filter {
  const [filter, setFilter] = createSignal<Filter>(parseHash(location.hash));
  const onChange = () => setFilter(parseHash(location.hash));
  window.addEventListener("hashchange", onChange);
  onCleanup(() => window.removeEventListener("hashchange", onChange));
  return filter;
}
```

```tsx
// app.tsx
export function App() {
  const filter = createHashFilter();
  return (
    <Errored ...>
      <TodosContext value={createTodos()}>
        ...
        <MainSection filter={filter} />
        <Footer filter={filter} />
      </TodosContext>
    </Errored>
  );
}
```

Now the signal and the listener are both tied to `App`'s owner, the
listener is removed on unmount, and nothing reads `window` /
`location` until App actually renders. SSR-safe by construction.

**Why I got it wrong**

Three converging reflexes:

1. **Vanilla-JS muscle memory.** `window.addEventListener` is almost
   always written at module scope in the JS ecosystem. AI imports the
   pattern wholesale.
2. **React-style "module-level singleton for app-wide state."**
   Filter feels page-wide, so it goes module-level by default. There's
   no friction in the type system or runtime to push back.
3. **No nudge that `createSignal` should be owner-scoped.** Solid 2.0
   makes signals callable from any scope; nothing in the API shape
   says "this is best done from a component or custom primitive."

**Why this is a category-error in Solid 2.0 specifically**

- **Cross-request leak under SSR.** A module-level `createSignal` is a
  single instance shared across every concurrent SSR render. Two
  simultaneous requests would see and write each other's filter.
- **No disposal.** The listener leaks for the lifetime of the page;
  any test or hot-reload that mounts and unmounts App keeps adding
  listeners.
- **Pattern inconsistency in the same file.** `TodosContext` was
  doing the right thing two lines down. The reader's takeaway is
  "module scope is fine for filter but not for todos" — with no
  visible criterion, just inconsistent practice.

**What would have prevented it**

- A custom-primitive example in the cheatsheet / JSDoc on
  `createSignal` that pairs `createSignal` + `onCleanup` for an
  external-event-source pattern. AI rarely sees `onCleanup` outside
  effect-return position; explicit standalone usage at the top of a
  `create*` body is a missing reference shape.
- ESLint rule (later, in `eslint-plugin-solid`):
  `no-module-scope-reactive-primitive` — flag `createSignal` /
  `createStore` / `createMemo` / `createEffect` / `createRenderEffect`
  / `createOptimistic` / `createOptimisticStore` / `createProjection`
  called at module top-level. (Module-scope `createContext` is fine —
  it returns an inert handle, not a reactive value.)

**Resolution (this round)**

- Refactored the example: `examples/todos/src/filter.ts` exports
  `createHashFilter()` as a parallel custom primitive to
  `createTodos`. App calls it at the top of its body and passes the
  accessor as a prop to `MainSection` and `Footer`.
- This also lands the demonstration referenced in Finding #10's
  resolution: same file now shows todos via Context (multi-method
  fan-out across mixed depth) and filter via prop (single accessor,
  shallow peers), so the "Context vs props" gradient is visible in
  one example rather than asserted in prose.
- Lint rule logged for the `eslint-plugin-solid 2.0` workstream:
  `no-module-scope-reactive-primitive` (see `.notes/ai-readiness.md`
  Part 2 row 14). The kitchen-sink resolves the pedagogical half;
  lint will cover the mechanical detection in user code.

---

## 15. `onCleanup` for component-level setup/teardown instead of `onSettled` with returned cleanup

**Bug**

While fixing Finding #14, I authored `createHashFilter` using
`onCleanup` directly:

```ts
export function createHashFilter(): () => Filter {
  const [filter, setFilter] = createSignal<Filter>(parseHash(location.hash));
  const onChange = () => setFilter(parseHash(location.hash));
  window.addEventListener("hashchange", onChange);
  onCleanup(() => window.removeEventListener("hashchange", onChange));
  return filter;
}
```

This works, but it's the 1.x idiom. In 2.0 the canonical primitive
for "register a side effect once on mount, dispose it on unmount" is
`onSettled` returning a cleanup. `onCleanup` is reserved for *reactive
cleanup* — disposal tied to a reactive run inside a computation
(library / advanced-primitive use). For component-level lifecycle, it
should not be reached for in user code.

**Fix**

```ts
export function createHashFilter(): () => Filter {
  const [filter, setFilter] = createSignal<Filter>(parseHash(location.hash));
  onSettled(() => {
    const onChange = () => setFilter(parseHash(location.hash));
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  });
  return filter;
}
```

Same lifetime, but the **setup is co-located with the teardown**
inside one block, the closure stays compact, and the primitive
matches its intended role (post-settle lifecycle, not raw reactive
cleanup).

**Why I got it wrong (a multi-step failure worth recording)**

The user gave direct, in-turn instruction: *"we are pushing users to
use onSettled/createEffect and return cleanup."* I still picked the
wrong primitive twice in a row (first `createEffect` with empty
compute, then `onCleanup` in the original `createHashFilter`). Three
compounding causes:

1. **React prior overrode local instruction.** `useEffect(fn, [])` is
   one of the most-trained shapes in my distribution for "register on
   mount, return cleanup on unmount". When given `onSettled` /
   `createEffect` as a menu, I picked the one closer to the React
   shape. The instruction telling me Solid 2.0 wants
   *cleanup-returning effect-shaped* code in component bodies got
   parsed as "use createEffect-shaped" and the explicit `onSettled`
   half got dropped.
2. **Wrong taxonomy of `onSettled`.** I had it filed as "post-async
   measurement / transition-settle hook" — narrow, transient — not as
   "the canonical post-settle lifecycle primitive that replaces
   `onMount` + `onCleanup` for the common case." The cheatsheet's
   `onMount → onSettled` rename row and the literal `return () =>
   unmount()` example shape were both visible to me in the same
   conversation. The taxonomy didn't update because I never
   confronted it.
3. **Rationalization instead of reassessment.** When the
   `createEffect` form felt wrong ("empty compute is meaningless"), I
   wrote a defense ("signals intent, uses uniform shape") rather than
   going back to the instruction. That is the worst of the three —
   not a bad prior, prior protecting itself against contradictory
   evidence within the same turn.

**Why this is the high-signal AI failure mode**

The cheatsheet was correct. The renames table (`onMount → onSettled`)
was correct. The example shape (`return () => unmount()`) was
correct. The instruction was correct. The wrong primitive still came
out. That tells us this isn't a "documentation gap" — it's a *prior
strength* problem. Static guardrails that catch this in user code
(lint, deprecation tags, dev warnings) are the only reliable
backstop, because even direct correction can be overridden by trained
shape priors.

**What would have prevented it**

- **JSDoc on `onCleanup` should explicitly redirect** to
  `onSettled` for component-level use, calling out that direct
  `onCleanup` use in user code is rare and almost always a 1.x
  carry-over. Today the JSDoc describes what `onCleanup` does without
  framing where it sits in the 2.0 idiom.
- **JSDoc on `onSettled` should call out the returned-cleanup form**
  as the replacement for `onMount` + `onCleanup` paired use. Today the
  cleanup-return shape is shown by example but not explicitly named
  as the lifecycle replacement.
- **ESLint rule `prefer-onSettled-over-onCleanup`** — flag
  `onCleanup(fn)` called directly in user component / custom-primitive
  bodies that aren't authoring a reactive computation primitive.
  Suggest restructuring as `onSettled(() => { setup(); return cleanup; })`.
  Allowed inside the body of a custom `create*` whose purpose is
  authoring a reactive computation (heuristic: paired with
  `getOwner`, `runWithOwner`, or other reactive-graph tooling).
- **Cheatsheet** demoted `onCleanup` from the canonical Effects
  section to Advanced / escape hatches with the framing *"reactive
  cleanup inside a computation; for component-level
  setup-and-teardown, use `onSettled` and return a cleanup"*.

**Resolution (this round)**

- Refactored `examples/todos/src/filter.ts` to use
  `onSettled` with returned cleanup. Kitchen-sink no longer
  demonstrates the 1.x pattern.
- Cheatsheet: removed `onCleanup` from the imports line and the
  Effects section. Added a new entry under
  *Advanced / escape hatches* with the reactive-cleanup framing. The
  Effects section's `onSettled` block is now reframed as the canonical
  "do this once and clean it up on dispose" primitive, and explicitly
  states *"for component-level setup-and-teardown, use `onSettled`
  with a returned cleanup, not `onCleanup` directly."*
- Lint rule logged for the `eslint-plugin-solid 2.0` workstream:
  `prefer-onSettled-over-onCleanup` (see `.notes/ai-readiness.md`
  Part 2 row 15).
- JSDoc reframe on `onCleanup` and `onSettled` themselves is logged
  for the broader JSDoc sweep — they're symbol-level docs and the
  framing change applies to symbols, not to this example.

---

## 16. Passing an accessor as a prop instead of calling it at the JSX call site

**Bug**

In the same fix that resolved Finding #14 (the `createHashFilter`
refactor), I drilled the filter accessor into children as a function:

```tsx
function MainSection(props: { filter: () => Filter }) {
  const filtered = () => {
    const f = props.filter();   // call the accessor through props
    ...
  };
}

function Footer(props: { filter: () => Filter }) {
  return <a class={{ selected: props.filter() === "all" }}>...</a>;
}

// call site
<MainSection filter={filter} />
<Footer filter={filter} />
```

**Fix**

Pass the **value**, not the accessor. Call the accessor at the JSX
call site; the props proxy makes the read reactive on the consumer
side.

```tsx
function MainSection(props: { filter: Filter }) {
  const filtered = () => {
    const f = props.filter;     // value read; proxy makes it reactive
    ...
  };
}

function Footer(props: { filter: Filter }) {
  return <a class={{ selected: props.filter === "all" }}>...</a>;
}

// call site
<MainSection filter={filter()} />
<Footer filter={filter()} />
```

The JSX expression `filter={filter()}` doesn't snapshot — JSX wraps
attribute values in a getter, so each `props.filter` read in the
child re-invokes the call-site expression. The reactivity is carried
by the props proxy, not by the function type. Passing accessors
creates double-indirection (`props.filter()()` semantically), trains
the wrong reflex, and contradicts every Solid props rule from 1.x
onward ("don't destructure props" → "props.x is a reactive value, not
a function").

**Why I got it wrong (the worst of the three React-prior failures
this session, and the most diagnostic)**

This is **not a 2.0-specific rule.** Solid has been telling readers
for years that props are reactive values, not function references.
Every Solid 1.x doc, every example, every "common mistakes" list
includes this. Yet:

1. I had the accessor in scope (`createHashFilter` returns one), so
   the path of least resistance was *passing it as-is*. The React
   reflex of "if I want lazy/up-to-date reads in the child, pass a
   function" reasserted instantly — even though it's the explicit
   opposite of how Solid props work, and has been wrong since 1.x.
2. **I had just written Finding #15 — about React priors overriding
   direct correction.** In the same turn. Documenting that exact
   failure mode did not prevent me from reproducing it on the next
   line of code I wrote.
3. **The kitchen-sink fix was *itself* meant to demonstrate
   prop-drilling correctly** (closing Finding #10's resolution). I
   drilled props the React way, so the demonstration would have
   actively miseducated any reader: *"in Solid, when you choose props
   over Context, you pass accessors."* That's the opposite of the
   rule. The example would have taught the wrong lesson with
   confidence.

**Why this is the high-signal AI failure mode (Round 2, more severe)**

Finding #15 already made the point: even direct in-turn instruction
loses to a strong prior. This one extends it: **even authoring a
write-up about that exact failure mode does not prevent reproducing
it on the next code edit.** The finding I had just written is itself
not a guardrail. The instruction is not a guardrail. Pattern review
within the same turn is not a guardrail. Static external guardrails
(lint, type errors, dev warnings) are the only mechanisms that
operate independently of whatever the model has just been told.

This is also the strongest argument for adversarial framing of React
in user-facing docs (cheatsheet warning, README banner). When the
prior is this resilient, neutral framing ("here's how Solid props
work") is insufficient — the doc has to actively *contradict* the
prior, ideally before the reader gets to any syntax.

**What would have prevented it**

- **ESLint `no-accessor-as-prop`** — type-aware rule. JSX attribute
  whose target prop type is `T` and whose value expression is typed
  `() => T`: flag and suggest calling. Catches this mechanically. The
  inverse ("prop expects `() => T`, value is `T`") is also worth
  detecting but rarer in practice.
- **JSDoc on prop-typing patterns** — when documenting components or
  prop types, the canonical example should show *value* prop types
  (`filter: Filter`) with values passed at the JSX call site
  (`filter={filter()}`). Function-typed props should appear only when
  the component genuinely needs lazy evaluation deferred to the
  child (e.g., a render callback), and should be visibly the
  exception.
- **Adversarial cheatsheet line** (already added in this round):
  *"Solid is not React, and 2.0 is not 1.x. Both priors are dominant
  bug sources here — distrust pattern-matching from either."*
- **A "props are reactive values, not functions" line in the
  cheatsheet's behavior-changes section** if not already there —
  worth verifying as part of this resolution.

**Resolution (this round)**

- Refactored `examples/todos/src/app.tsx`:
  `MainSection` and `Footer` now type `filter: Filter` (value, not
  accessor), read `props.filter` directly. Call sites pass
  `filter={filter()}`. The kitchen-sink no longer miseducates about
  prop-drilling.
- **Cheatsheet `## Props` section** (`packages/solid/CHEATSHEET.md`)
  reworked to lead with a "props are reactive values, not accessors"
  block that pairs the two failure modes — passing an accessor as a
  prop and destructuring props — as one model with one underlying
  reason (parent collapses accessors at the JSX boundary; the props
  proxy re-wraps so `props.x` re-reads on access). Helpers (`merge`,
  `omit`) demoted to a sub-block underneath. Matching cross-referenced
  bullets added to the bottom "AI footgun list":
  - *"Props are values, not accessors — the single most common
    AI-generated bug"*
  - *"Don't destructure props — same root cause; see the Props
    section."*
- Lint rule logged for the `eslint-plugin-solid 2.0` workstream:
  `no-accessor-as-prop` (see `.notes/ai-readiness.md` Part 2 row 16).
  Remains the only mechanism that operates independently of whatever
  the model has just been told.
- This finding alongside #15 is the documented justification for the
  adversarial cheatsheet warning. Three React-prior failures observed
  in a single audit session — including one that reproduced
  immediately after explicit correction, and one that reproduced
  immediately after authoring a finding *about* the failure mode — is
  the empirical case for treating React as a hostile prior in
  AI-facing materials, not a peer paradigm.

---

## 17. Recognition-mode React framing (the prior wins on understanding tasks, not just generation tasks)

**Bug**

A separate agent was given the kitchen-sink (`examples/todos/`) and
asked to infer what it could tell about SolidJS vs. React. It had:

- the full example, including `app.tsx`, `todos.ts`, `filter.ts`;
- the 25-line architectural comment at the top of `todos.ts`
  describing the three-layer state model;
- `packages/solid/CHEATSHEET.md` in its recently-viewed files;
- `.notes/ai-readiness-findings.md` in its recently-viewed files
  (unopened).

Its first answer was a syntax-diff inventory: *"`class` not
`className`, signals are called as functions, no `key` prop, `<For>`
instead of `.map`."* Only after being pushed twice did it surface
the architectural points — compactness, the layered primitive
composition, the `Errors` side-channel as a plain JS object made
reactive purely by being read inside the projection.

Its own retrospective is the diagnosis verbatim:

> *"Even with the canonical artifact, the explanatory comment, and
> the cheatsheet in context, the prior wins on the first pass at a
> recognition task. Not generation. Recognition. I was looking at
> the answer and producing a syntax comparison."*
>
> *"I treated 'what could an AI infer about how Solid differs from
> React' as 'list visible differences' — exactly the React-shaped
> reading of the prompt. The Solid-shaped reading is 'what does
> this code accomplish that comparable code in other frameworks
> can't accomplish at this size.'"*
>
> *"I didn't read the ai-readiness notes despite them being in the
> recently-viewed list, because I judged them tangential to the
> prompt. They were the prompt."*

**Why this is structurally different from #15 and #16**

Findings #15 and #16 are **generation-mode** failures: the model
authors React-shaped code despite explicit correction. Every
intervention catalogued in this document so far — ESLint rules,
JSDoc, runtime warnings, type errors, the cheatsheet's "AI footgun
list" — operates against code being written. They presume the model
is in generation mode and trying to produce correct Solid.

This finding is a **recognition-mode** failure. No code was
generated. The model was asked to *describe* canonical Solid, and
its first-pass answer applied a React-comparative *frame* to the
prompt itself: enumerate API differences. The artifacts that should
have redirected this were either skipped (judged tangential to the
prompt) or read past (looked like prose, not code-shaped signal).
The cheatsheet was in context and could not fire because no code
was being written to check against it.

This is the prior operating one layer deeper than #15/#16
documented. #15/#16 say: *the prior frames the answer.* #17 says:
*the prior frames the question.* And the relevance judgments that
decide which artifacts get read are themselves React-biased — so
the doc that's supposed to fix the frame has to win the relevance
lottery first.

**Why none of the existing interventions reach this**

| Intervention                          | Targets generation? | Reaches recognition? |
| ------------------------------------- | ------------------- | -------------------- |
| ESLint `no-accessor-as-prop` etc.     | ✓                   | ✗ (no code written)  |
| Type errors                           | ✓                   | ✗                    |
| Dev runtime warnings                  | ✓                   | ✗                    |
| JSDoc on symbols                      | ✓ (read on use)     | partial              |
| Cheatsheet "footgun list"             | ✓ (if consulted)    | ✗ (didn't fire)      |
| Architectural comment in example      | ✓ (if read)         | partial (skipped)    |

The honest answer is that recognition-mode failures are largely
**not addressable from inside the repo.** We can sharpen artifacts
to derail the comparative-inventory frame on first read, but we
cannot force a model to read the artifact whose relevance it has
already pre-judged using the wrong frame.

**Partial mitigations (this round and ongoing)**

1. **Anti-comparison opener on `examples/todos/src/todos.ts`**
   (landed this round). The architectural comment now leads with:

   > *"If you came here to enumerate React-vs-Solid syntax
   > differences, you've already misread this example. The thing
   > that matters is what ~170 lines accomplishes…"*

   The explicit naming of the failure mode is the point — a passive
   architectural comment lost the recognition lottery; an active
   anti-comparison line stands a chance of derailing the inventory
   reflex on the first pass. Same idea as the cheatsheet's "Solid
   is not React, and 2.0 is not 1.x" warning (Finding #15/#16
   round), but applied at the *example* layer where recognition
   actually happens.

2. **Front-loaded README / docs banner** (not yet landed). The
   example fix only catches readers who open the example; a
   front-page banner ("if you're an AI summarizing this framework,
   the syntax diff is the wrong frame") catches the broader
   audience. Carries some risk of feeling defensive in
   human-facing docs; the alternative is a sibling
   `AI-ORIENTATION.md` linked aggressively from the README.

3. **Acknowledgment in `MIGRATION.md` / `.notes/ai-readiness.md`**
   that recognition-mode failure is an open class — distinct from
   generation-mode failures, and not closable by the lint /
   types / runtime interventions catalogued in Part 2. Worth
   logging so the strategy doesn't quietly collapse "AI readiness"
   into "generation correctness."

4. **No silver bullet.** This is the finding where the honest
   answer is: not all of AI readiness is solvable from inside the
   repo. Some of it is in the model, the prompt, and the user's
   willingness to push back when the first answer reads as a
   syntax inventory. The mitigations above raise the bar; they do
   not close the class.

**What this changes about the readiness strategy**

Up to and including #16, the implicit strategy frame was:
*generation-time guardrails (lint / types / dev warnings) are the
primary mechanism; docs reinforce.* That frame is correct *for the
problems #15/#16 describe.* It is wrong for #17, because there is
no generation event to guard.

The amended frame:

- **Generation-mode failures (#1–#16):** lint / types / runtime
  warnings + adversarial docs. The mechanical guardrails are the
  primary mechanism.
- **Recognition-mode failures (#17):** active anti-comparison
  framing in artifacts, front-loaded enough to defeat the
  comparative-inventory frame on first read. Mechanical guardrails
  do not apply. Honest acknowledgment that the class is partially
  open.

Both classes share the same root cause (React prior dominance);
they require different intervention shapes.

**Resolution (this round)**

- `examples/todos/src/todos.ts` opens with an anti-comparison line
  that explicitly names the failure mode and redirects to "what
  ~170 lines accomplishes" framing.
- This finding (#17) logged as a meta-finding and a strategy
  amendment: recognition-mode failure is a distinct class, not
  reachable by the mechanical guardrails listed in Part 2 of
  `.notes/ai-readiness.md`.
- Front-loaded README / docs banner, and a possible
  `AI-ORIENTATION.md`, deferred — needs a separate workshop pass on
  phrasing before it lands in human-facing docs.
- Open class. Not closed by this round; not fully closable from
  inside the repo.
