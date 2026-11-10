import { render } from "@solidjs/web";

import { DeepDiveApp } from "./deep-dive/App";
import "./styles.css";
import "./deep-dive/styles.css";

const root = document.getElementById("root");
if (!root) {
  throw new Error("Missing #root element.");
}

render(() => <DeepDiveApp />, root);
