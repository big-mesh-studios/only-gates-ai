import { For } from "solid-js";

import { lineCount } from "./code";

export function Chapter(props: {
  id: string;
  number: string;
  kicker: string;
  title: string;
  children: any;
}) {
  return (
    <section class="chapter" id={props.id}>
      <div class="chapter-head">
        <span class="chapter-number" aria-hidden="true">
          {props.number}
        </span>
        <div>
          <p class="kicker">{props.kicker}</p>
          <h2>{props.title}</h2>
        </div>
      </div>
      {props.children}
    </section>
  );
}

export function Checkpoint(props: { title?: string; children: any }) {
  return (
    <aside class="checkpoint" aria-label={props.title ?? "Checkpoint"}>
      <strong>{props.title ?? "Checkpoint"}</strong>
      {props.children}
    </aside>
  );
}

export function Bug(props: { children: any }) {
  return (
    <aside class="bug" aria-label="Common bug">
      <strong>Common bug</strong>
      {props.children}
    </aside>
  );
}

export function Note(props: { children: any }) {
  return (
    <aside class="note-box" aria-label="Note">
      {props.children}
    </aside>
  );
}

export function Code(props: {
  code: string;
  lang?: string;
  file?: string;
  label?: string;
  collapsible?: boolean;
  open?: boolean;
}) {
  const count = () => lineCount(props.code);
  const caption = () =>
    props.label ?? props.file ?? props.lang ?? "source";

  const block = (
    <pre
      class="snippet-code"
      tabindex="0"
      role="region"
      aria-label={`${caption()}, ${count()} lines`}
    >
      <code>{props.code}</code>
    </pre>
  );

  if (props.collapsible) {
    return (
      <details class="code-details" open={props.open}>
        <summary>
          <span>{caption()}</span>
          <span class="code-meta">
            {props.lang ? `${props.lang} · ` : ""}
            {count()} lines
          </span>
        </summary>
        {block}
      </details>
    );
  }

  return (
    <figure class="snippet">
      <figcaption class="snippet-head">
        <span>{caption()}</span>
        {props.lang ? <span class="code-meta">{props.lang}</span> : null}
      </figcaption>
      {block}
    </figure>
  );
}

export function SourceLink(props: { href: string; children: any }) {
  return (
    <a class="source-link" href={props.href}>
      {props.children} →
    </a>
  );
}

export function DataTable(props: {
  caption: string;
  head: string[];
  rows: (string | number)[][];
}) {
  return (
    <div class="table-wrap">
      <table>
        <caption>{props.caption}</caption>
        <thead>
          <tr>
            <For each={props.head}>{(cell) => <th scope="col">{cell}</th>}</For>
          </tr>
        </thead>
        <tbody>
          <For each={props.rows}>
            {(row) => (
              <tr>
                <For each={row}>
                  {(cell, index) =>
                    index() === 0 ? (
                      <th scope="row">{cell}</th>
                    ) : (
                      <td>{cell}</td>
                    )
                  }
                </For>
              </tr>
            )}
          </For>
        </tbody>
      </table>
    </div>
  );
}
