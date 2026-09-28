import { sites } from "@openai/sites-vite-plugin";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const isCodexSeatbeltSandbox = process.env.CODEX_SANDBOX === "seatbelt";

export default defineConfig(({ command }) => ({
  server: {
    port: 5173,
    strictPort: true,
    watch: {
      ignored: ["**/jobs/**", "**/outputs/**", "**/backend/**", "**/.uvicorn.log"],
      ...(isCodexSeatbeltSandbox ? { useFsEvents: false, usePolling: true } : {}),
    },
  },
  plugins: [react(), ...(command === "build" ? [sites()] : [])],
}));
