import { renderToString } from "@solidjs/web";
import { App } from "./App";

/** The page's markup, as the prerender writes it into the document. */
export function render(): string {
  return renderToString(() => <App />);
}
