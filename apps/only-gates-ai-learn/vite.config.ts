import { defineConfig } from "vite";
import solid from "@solidjs/vite-plugin";

// This app is served from /learn/ inside the main app's GitHub Pages output.
// A relative base keeps the page working under any project path and lets the
// few links in the page stay relative. Nothing here hydrates: the pages are
// rendered once by scripts/prerender.ts and shipped as static HTML.
export default defineConfig({
  base: "./",
  plugins: [solid({ ssr: true, solid: { hydratable: false } })],
  build: {
    target: "esnext",
  },
  server: {
    port: 5175,
    // The deep-dive page imports the production C sources with ?raw from the
    // repository root, outside this app's root. The SSR build reads them
    // directly; this allow-list is only for the dev server.
    fs: {
      allow: ["..", "../..", "../../.."],
    },
  },
});
