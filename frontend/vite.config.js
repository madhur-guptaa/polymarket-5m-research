import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// Dev server proxies API + WebSocket to the FastAPI server on :8000, so the
// browser talks same-origin (no CORS, no hardcoded IPs). Override the backend
// target with VITE_BACKEND if it runs elsewhere.
const backend = process.env.VITE_BACKEND || 'http://localhost:8000'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      '/api': { target: backend, changeOrigin: true },
      '/ws': { target: backend.replace(/^http/, 'ws'), ws: true, changeOrigin: true },
    },
  },
})
