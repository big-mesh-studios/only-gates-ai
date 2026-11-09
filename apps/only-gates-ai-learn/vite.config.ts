import { defineConfig } from "vite";
import solid from "@solidjs/vite-plugin";

// This app is served from /learn/ inside the main app's GitHub Pages output.
// A relative base keeps the page working under any project path and lets the
// few links in the page stay relative. Nothing here hydrates: the page is
// rendered once by scripts/prerender.ts and shipped as a single HTML file.
export default defineConfig({
  base: "./",
  plugins: [solid({ ssr: true, solid: { hydratable: false } })],
  build: {
    target: "esnext",
  },
  server: {
    port: 5175,
  },
});
