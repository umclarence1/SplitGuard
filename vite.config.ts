import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  server: {
    port: 5173,
    strictPort: true,
    watch: {
      ignored: ["**/jobs/**", "**/outputs/**", "**/backend/**", "**/.uvicorn.log"],
    },
  },
  plugins: [react()],
});
