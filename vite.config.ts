import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { resolve } from 'path'

export default defineConfig({
  plugins: [react()],
  root: resolve(__dirname, 'frontend'),
  server: {
    port: 5173,
    open: true,
    host: '127.0.0.1'
  },
  build: {
    outDir: resolve(__dirname, 'dist')
  }
})
