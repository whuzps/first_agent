import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const BACKEND = 'http://localhost:8000'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5174,
    proxy: {
      '/api':      { target: BACKEND, changeOrigin: true },
      '/auth':     { target: BACKEND, changeOrigin: true },
      '/chat':     { target: BACKEND, changeOrigin: true },
      '/greet':    { target: BACKEND, changeOrigin: true },
      '/health':   { target: BACKEND, changeOrigin: true },
      '/models':   { target: BACKEND, changeOrigin: true },
      '/sessions': { target: BACKEND, changeOrigin: true },
      '/suggest':  { target: BACKEND, changeOrigin: true },
    },
  },
})
