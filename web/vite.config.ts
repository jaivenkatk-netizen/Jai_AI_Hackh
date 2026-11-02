import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    // The React app talks to FastAPI through this proxy, so the same relative
    // URLs ("/api/kpis") work in dev and in a built bundle. Flipping between
    // fixtures and the live API is then one flag in src/lib/api.ts, not a
    // rewrite of every fetch call.
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/ws': {
        target: 'ws://127.0.0.1:8000',
        ws: true,
      },
    },
  },
})
