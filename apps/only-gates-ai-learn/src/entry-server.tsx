import { renderToString } from "@solidjs/web";

import { App } from "./App";
import { DeepDiveApp } from "./deep-dive/App";

/** The field notes' markup, as the prerender writes it into the document. */
export function renderLearn(): string {
  return renderToString(() => <App />);
}

/** The deep-dive tutorial's markup. */
export function renderDeepDive(): string {
  return renderToString(() => <DeepDiveApp />);
}
