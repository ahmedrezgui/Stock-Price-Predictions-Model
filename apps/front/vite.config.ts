import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      "/market-api": {
        target: "http://127.0.0.1:8002",
        rewrite: (p) => p.replace(/^\/market-api/, ""),
      },
      "/assistant-api": {
        target: "http://127.0.0.1:8001",
        rewrite: (p) => p.replace(/^\/assistant-api/, ""),
        timeout: 300000,
        proxyTimeout: 300000,
      },
    },
  },
});
