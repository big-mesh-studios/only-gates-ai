import { defineConfig } from "vite";
import solid from "@solidjs/vite-plugin";

export default defineConfig({
  plugins: [solid()],
  optimizeDeps: {
    exclude: ["@only-gates-ai/core"],
  },
  build: {
    target: "esnext",
  },
  server: {
    port: 5174,
  },
});
