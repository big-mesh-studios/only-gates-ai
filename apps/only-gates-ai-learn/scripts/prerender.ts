// @ts-nocheck
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");

// The page is compiled for the server into a folder outside the source tree, so
// it is loaded by address once it exists rather than imported by name.
const bundle = pathToFileURL(join(root, ".ssr/entry-server.js")).href;
const { render } = (await import(bundle)) as { render: () => string };

// Nothing on this page reacts. The stylesheet travels inside the document, so
// the whole page is one request and one file. Assembly copies dist/ into the
// main app's output at learn/.
const styles = await readFile(join(root, "src/styles.css"), "utf8");
const body = render();

const title = "How Integer GPT works — only-gates-ai";
const description =
  "Integer-only GPT inference, the fixed-point toolbox, and the path from arithmetic to logic gates and an FPGA. Includes free Colab training instructions.";

const document = `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta name="color-scheme" content="dark" />
    <title>${title}</title>
    <meta name="description" content="${description}" />
    <meta property="og:type" content="article" />
    <meta property="og:title" content="${title}" />
    <meta property="og:description" content="${description}" />
    <link
      rel="icon"
      href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><rect width='32' height='32' rx='7' fill='%237ee787'/><text x='16' y='23' font-size='19' text-anchor='middle' fill='%2306210d' font-family='monospace'>⌁</text></svg>"
    />
    <style>
${styles.trimEnd()}
    </style>
  </head>
  <body>
${body}
  </body>
</html>
`;

await mkdir(join(root, "dist"), { recursive: true });
await writeFile(join(root, "dist/index.html"), document);
console.log("wrote dist/index.html");
