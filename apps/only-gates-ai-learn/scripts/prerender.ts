// @ts-nocheck
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");

// The pages are compiled for the server into a folder outside the source tree,
// so they are loaded by address once the bundle exists rather than imported by
// name.
const bundle = pathToFileURL(join(root, ".ssr/entry-server.js")).href;
const { renderLearn, renderDeepDive } = (await import(bundle)) as {
  renderLearn: () => string;
  renderDeepDive: () => string;
};

const [baseStyles, deepDiveStyles] = await Promise.all([
  readFile(join(root, "src/styles.css"), "utf8"),
  readFile(join(root, "src/deep-dive/styles.css"), "utf8"),
]);

const icon =
  "data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><rect width='32' height='32' rx='7' fill='%237ee787'/><text x='16' y='23' font-size='19' text-anchor='middle' fill='%2306210d' font-family='monospace'>⌁</text></svg>";

/**
 * Nothing on either page reacts. The stylesheet travels inside the document,
 * so each page is one request and one file. Assembly copies dist/ into the
 * main app's output at learn/.
 */
function document({ title, description, styles, body }) {
  return `<!doctype html>
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
    <link rel="icon" href="${icon}" />
    <style>
${styles.trimEnd()}
    </style>
  </head>
  <body>
${body}
  </body>
</html>
`;
}

const learnTitle = "How Integer GPT works — only-gates-ai";
const learnDescription =
  "Integer-only GPT inference, the fixed-point toolbox, and the path from arithmetic to logic gates and an FPGA. Includes free Colab training instructions.";

const deepDiveTitle = "Build Integer GPT from scratch — only-gates-ai";
const deepDiveDescription =
  "A from-zero tutorial: what a GPT is, what each transformer part is for, how to implement the float forward pass, then how to replace it with integer-only inference: int8 quantization, Q16 activations, integer matmul, RMSNorm, a GELU lookup table, causal attention, DI-Exp softmax, sampling, and the float-to-integer export contract.";

const pages = [
  {
    path: join(root, "dist/index.html"),
    title: learnTitle,
    description: learnDescription,
    styles: baseStyles,
    body: renderLearn(),
  },
  {
    path: join(root, "dist/deep-dive/index.html"),
    title: deepDiveTitle,
    description: deepDiveDescription,
    styles: `${baseStyles}\n${deepDiveStyles}`,
    body: renderDeepDive(),
  },
];

for (const page of pages) {
  await mkdir(dirname(page.path), { recursive: true });
  await writeFile(page.path, document(page));
  console.log(`wrote ${page.path.slice(root.length + 1)}`);
}
