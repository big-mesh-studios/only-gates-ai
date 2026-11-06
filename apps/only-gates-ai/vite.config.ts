import { defineConfig } from "vite";
import solid from "@solidjs/vite-plugin";

export default defineConfig({
  plugins: [solid()],
  optimizeDeps: {
    exclude: ["@only-gates-ai/core"],
  },
  // A relative base makes the build work from any path it is served under:
  // GitHub Pages project sites (`<user>.github.io/<repo>/`), custom domains,
  // and `vite preview` all work without rebuilding with a different prefix.
  base: "./",
  build: {
    target: "esnext",
  },
  server: {
    port: 5174,
  },
});
