# Solid 2.0 Proposed API Changes

## Experience

### 1. No writes under Owned Scope

Because of split effects I think it is finally safe to disable writes under reactive scope. There are a few exceptions which can be overriden by setting pureWrite option on the signal but for the most part well will Error/Warn when someone tries to write a Signal in a tracked context.

### 2. Stricter `untrack`

Top-level in components when accessing a reactive value we warn unless the developer explicitly wraps the access in an `untrack`. Admittedly wrapping with `untrack` won't help with async but maybe it is the right direction so people don't accidentally do it. We could also in dev detect up component throwing and error out to tell the person they are using top-level async. There are varying degrees here but we should consider being "stricter".

With derived Signals we should basically never initiate signals from other reactive values that aren't derived.

### 3. `flush`

The default mechanism for batching will be on the next microtask. So yes we will work like Vue/Svelte etc as in the reactive graph is immediately askable after a set, but you need to call `flush` if you want to read from the DOM. Unlike those we will not be forcing things async, and `flush` will just drain the queue immediately.

But it does mean for stuff like focus management etc.. you will need to call this in your event handlers. This also means `batch` will no longer be needed.

## Control Flow

### 1. Combined For/Index - single signature control flow

Honestly it doesn't add much overhead and will simplify everything. It's a breaking change so it's perfect for 2.0. The `value` and the `index` will now be accessors which will make life a lot simpler. For will now have a `keyed` prop which will be true by default. If set to false it will act like Index does today.. and if set to a function that returns the key it will act like Keyed does in many other frameworks.

```jsx
// normal
<For each={todos()}>{
  todo => <Todo todo={todo()} />
}</For>

// equivalent to Index today
<For each={todos()} keyed={false}>{
  todo => <Todo todo={todo()} />
}</For>

// key by something custom
<For each={todos()} keyed={item => item.id}>{
  todo => <Todo todo={todo()} />
}</For>
```

Also Show and Switch/Match will use functions for both keyed/non-keyed forms. Show still defaults to `false`
```jsx
<Show when={user()}>{user => <User user={user()} />}</Show>

<Show when={user()} keyed>{user => <User user={user()} />}</Show>
```

### 2. Repeat

We introduce a new list control flow to handle ranges. Primarily useful for non-keyed store scenarios as it doesn't receive a list so it does no diffing as items are added.

```jsx
<Repeat count={todos.length}>{
  i => <Todo todo={todos[i]}/>
}</Repeat>
```
Can also be used for windowed operation:
```jsx
<Repeat count={50} from={offset()}>{
  i => <Todo todo={todos[i]} />              
}</Repeat>
```
This is super efficient way to handle things like large chats or like 3D models (I used this for the first time in the 3D spinning cube demo from 5 years ago).

### 3. Loading

Suspense will be called Loading now since it never returns to fallback. It is an initial load affordance and can be nested to break out of waiting for async on new navigation the same way Suspense did.

```jsx
<Loading fallback="loading...">
  <MyStuff />
</Loading>
```

### 4. Errored

Updated version of `ErrorBoundary`. Shorter name and also better retry functionality. We build it in and it will attempt to pull down the graph to fix the error, including refetching any errored async.

```jsx
<Errored fallback={(err, retry) => <button onClick={retry}>Retry</button>}>
  <MyStuff />
</Errored>
```

## Signals

### 1. Automatic microtask batching (flush)

Effects will not be processed to the microtask queue with the option to flush immediately with `flush`. Removes the need for `batch`.

### 2.`createComputed` deprecated in favor of derived base primitives

In hopes of promoting more derived approaches for consistency we are looking to create mutable wrappers over computed values as a replacement for writing back to Signals. This can be very useful for things like overriding parent props, or creating temporary writeable layers over async data. The reactive computation acts as a reset.

In addition to the APIs they have today Signals/Stores would also support:

```jsx
const [name, setName] = createSignal(() => props.name);

const [store, setStore] = createStore(user => { user.name = props.name }, {});
```

In the example above name would either be `props.name` or the value you have set with `setName`. The last set wins. If they are the same value they won't notify by default as with other primitives.

There are some efficiency challenges here to avoid creating unnecessary computations but a node recycling approach similar to S.js should probably do the trick.

## Stores

### 1. Store use mutable `produce` by default

```jsx!
const [store, setStore] = createStore({ list: [] })

setStore(s => {
  s.greeting = "hi";
  s.list.push("value");
})

// legacy can be done:
setStore(path('list', (list) => [...list, "value"]))
```

Returning a value from the setter function will do a shallow diff. More ergonomic way to remove items from a top level array or remove all properties from an object.

### 2. New Derived Store primitive `createProjection`

This would allow dynamic derived capabilities driven by mutation. Hope to replace `createSelector`.

```jsx!
const selected = createProjection(s => {
  const sId = selectedId();
  s[sId] = true;
  if (s.previous != null) delete s[s.previous];
  s.previous = sId;
}, {})

<tr class={selected[row.id] ? "selected" : ""}></tr>
```
In so on `selectId` change instead of notifying every row it would only notify up to 2 rows.

Projections have other uses as the basis for derivedStores for things like data fetching with granular updates. Since any data returned from the projection function will automatically reconcile. Basically if you mutate you handle your own diffing and if you return the new value the whole things gets diffed.

```jsx!
const usersStore = createProjection(() => getUsers, [])
```

### 3. `mergeProps` renamed to `merge` to not treat `undefined` as not `in`

Renaming `mergeProps` to `merge` since it applies to `props` and `stores` alike. This a general helper and will be treated as such.

Currently we treat `undefined` as special. Like a pass through. Now that stores properly handle removal we should reflect that here as well. That means `undefined` value on a key will override key on a previous object.

### 4. `splitProps` replaced by `omit`

Splitting props unnecessarily creates additional objects. It can be convenient but it also can be expensive and leads to proxy de-optimization (ie.. proxy wrapping proxy etc). Instead the opposite of `merge` should be `omit`. We only focus on the last object with the missing properties:

```jsx!
const withoutAorB = omit({ a: 1, b: 2, c: 3}, "a", "b");

withoutAorB.c // 3
"a" in withoutAorB // false
```
You can always just referencing the original source object is can be easily done and at worse wrap it in an thunk accessor.

### 5. `deep` helper

Given the nature of splitting effect tracking we now likely need a way to deeply observe. A new primitive that does optimal deep tracking will be introduced so you can:

```js
const [store] = createStore({});

createEffect(() => deep(store), value => {...})
```

## Async

### 1. All computations are async capable (no more createResource)

```js
const user = createMemo(() => fetchUser(props.id));
```

When you return a Promise or Async Iterable any computation will be able to handle the async in a non-nullable way. Ie.. `Promise<number>` becomes `Accessor<number>`. It does this by throwing along the dependency graph when unsettled. This doesn't cause components to re-run just halting of graph propagation.

And unlike 1.0 we only register the read for boundaries in the final render effect where it is consumed.

We are missing key properties of createResource. As we are missing `state`, `error`, `loading`, `latest`, `mutate`, and `reload`.

To handle this it will be primitive based introduced in the next sections.

### 2. `isPending`
Pending can be queried at an expression level via `isPending`:

```jsx
const isDataPending = () => isPending(() => users() || posts())
```
It is a replacement for `.loading` and works anywhere along the dependency graph. Note it will always return false on initial creation. The expectations is to use Loading boundaries for that case.

### 3. `pending`

Useful for trying to get the future value of the expression. However, it might also return the stale value if it isn't available yet.

```ts
const [userId, setUserId] = createSignal(1)
const user: Accessor<User> = createMemo(() => fetchUser(userId()))
const pendingUserId: Accessor<User | undefined> = () => pending(userId)
```

### 4. `transition` changes

Mostly we will be supporting multiple unentangled transitions in flight at the same time. We will use a combination of the synchronous process and graph dependencies to decide what is entangled.

This means no transition wrappers like `useTranstion/startTransition`. Everything is a transition. While this typically would be expensive, we've pioneered a new approach that doesn't require forking the whole graph and instead relies on deferred value commit and deferred disposal to accomplish this.

### 5. `action`

We still need a wrapper though to handle async mutations to collect the mutation and connect it back into the graph. For that we have a generator based API (since who knows when AsyncContext will be available):

```js
const myAction = action(function*() {
  setOptimistic(s => { s.push(newTodo) })
  yield db.addTodo(newTodo);
  todos.refresh();
})
```

People might note that generators are notoriously bad with TS but there are a few options here. First you can use an async generator:

```js
const myAction = action(async function*() {
  setOptimistic(s => { s.push(newTodo) });
  // correct types on res
  const res = await db.addTodo(newTodo);
  yield; // remember to call yield to resume the action
  todos.refresh();
})
```
But I anticipate there likely will be patterns around this for better ergonomics that can use generator helpers like:

```js
const myAction = action(function*() {
  setOptimistic(s => { s.push(newTodo) })
  // correct types on res
  const [res, err] = yield* attempt(db.addTodo(newTodo));
  todos.refresh();
})
```

### 6. `createOptimistic`/`createOptimisticStore`

Instead of a core primitive to handle mutations we will reuse transitions and introduce the ability to inject optimistic state. At it's core it has the same API as `createSignal`/`createStore` it just has special behavior that causes it to only be set during transitions and resets to its source when it completes.

```js
const [optimistic, setOptimistic] = createOptimisticStore(() => db.getTodos();

// later
action(function*() {
  setOptimistic(s => { s.push(newTodo) })
  yield db.addTodo(newTodo);
  todos.refresh();
})
```

## Effects

### 1. Split tracking from effect

Not a decision that comes lightly as people probably will resist this but the implications are pretty large when it comes to async models, resumability, etc... 

```jsx
createEffect(
  prev => count(),
  (value, prev) => {
    console.log(`The count is`, value);
    return () => console.log("cleaning up");
  }
);
```
It means we will be able to run the front half of all effects before running any of the backside. This will give us knowledge of all dependencies before running any side effects.

In so like top-level in components, reads from signals will warn in dev and should be avoided. Async makes this a matter of best practice.

`createRenderEffect` will similarly be split. The difference is it will wrap its tracking part in `latest` so it will tear. `createEffect` while it will be scheduled the same way will not impact Suspense or ErrorBoundaries and is considered outside of rendering.

For that reason `creatEffect` will also accept an object instead of a function to manage errors coming from the reactive graph (like async fetching errors):

```jsx
createEffect(
  prev => count(), 
  {
    effect(value, prev) {
      console.log(`The count is`, value);
      return () => console.log("cleaning up");
    },
    error(error, cleanup) {
    
    }
  }
);
```


### 2. createTrackedEffect

We will still have single callback effect but it will no longer be the recommended path. It will exist for special cases and it will have warnings since it may re-run in async situations. We will wrap it in `latest` to reduce that happening but it is possible if any async value is read in it and not in any rendering that it will run more than once initially (throwing when it hits the not ready value).

Unlike before it will not be reducing and it will return a cleanup function:

```js
createTrackedEffect(() => {
  console.log(`The count is`, value());
  return () => console.log("cleaning up");
});
```
`onCleanup` will still work but we positioning it as more of an advanced primitive for reactive lifecycle cleanup.

### 3. `onMount` => `onSettled`

Similarly `onSettked` will just be an `untracked` version of `createTrackedEffect`:
```js
onSettled(() => {
  console.log(`The count is`, value());
  return () => console.log("cleaning up");
});
```
This is important because it gives us a way to catch async inside mount functions since it can run again if it throws.

`onSettled` also can be used in event handlers or non-reactive contexts to schedule work to be done when the current activity is fully settled.

## Ownership

### 1. `createRoot` dispose by parent Root by default

It is a minor change but it sets a better pattern. Basically the idea is to hoist ownership as the pattern instead of having no ownership. No ownership can still be done manually by `runWithOwner` and passing a `null`.

## Context

### 1. Context.Provider is no longer necessary

You now can use the Context directly as the Provider component instead of using `.Provider`
```js
const ThemeContext = createContext("light");

<ThemeContext value="dark">
  <Component />
</ThemeContext>
```

## DOM Expressions

### 1. Follow HTML standards by default

Right now many attributes/properties can be set multiple ways according to the types which makes it difficult to build on top of.

In 2.0 we will favor setting attributes over properties in almost all cases. The reason is it plays nicer with web components and removes a lot of the special cases especially when considering SSR. We will also be following HTML lower casing rather than camel casing for this reason for all built in attributes. This also means we no longer need namespaces for `attr:` or `bool:`.

Note:

1. Event handlers will stay camelCase as they aren't exactly equivalent to native ones and this accents the `on` modifier
2. `value`, `selected`, `checked`, `muted` ie, the default attributes are still handled as props to avoid unnecessary confusion.

### 2. Enhanced class prop

We are getting rid of `classList` by merging it into `class`. We will also take this opportunity to add support for arrays for combining classes. Ie think `clsx` built in with a few inline optimizations.

So the class prop can be any of string, object, or array of string or object. This does impact composition but you can always use an array to merge things in:

```js
function MyComponent(props) {
  return <div class={["local-class", props.class]} />
}
```

### 3. Consistent boolean handling

Using boolean literal values will add remove an attribute not set it to string. For psuedo boolean that require "true" you will need to pass it as a string. Types will change to reflect this.

## Removals

All most removed APIs will be able to approximated that we can ship a `@solidjs/legacy` package to support any that haven't been marked as deprecated during 1.0's lifetime.

1. createComputed

This will be replaced by other primitives. A computation that writes Signals that runs during pure time is dangerous not to mention harder to place in a lazy world.

2. batch

`flush` will handle the use cases.

3. onError/catchError
4. createSelector
5. createResource
6. Index
7. createMutable to Solid Primitives
8. classList
9. oncapture:
10. scheduler/createDeferred to Solid Primitives
11. splitProps - use `omit`
12. mergeProps - renamed to `merge`
13. `on` helper - not necessary with split effects
14. useTransition
15. startTransition

