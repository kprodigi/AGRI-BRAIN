// frontend/vite.config.js
import { defineConfig } from 'vite'
import os from 'node:os'
import path from 'node:path'

const cacheRoot =
  process.env.VITE_CACHE_DIR ||
  path.join(os.tmpdir(), 'vite-cache', 'agri-brain-frontend')

export default defineConfig({
  cacheDir: cacheRoot,

  resolve: {
    alias: {
      '@': path.resolve(import.meta.dirname, './src'),
    },
  },

  server: {
    host: true,
    watch: {
      usePolling: true,
      interval: 200,
      awaitWriteFinish: { stabilityThreshold: 500, pollInterval: 200 },
      ignored: ['**/.git/**', '**/dist/**', '**/node_modules/**']
    }
  },

  optimizeDeps: {
    force: true
  },

  build: {
    outDir: 'dist',
    emptyOutDir: true,
    rollupOptions: {
      output: {
        manualChunks: {
          'react-vendor': ['react', 'react-dom', 'react-router-dom'],
          'recharts': ['recharts'],
          'framer': ['framer-motion'],
          'radix': [
            '@radix-ui/react-dialog', '@radix-ui/react-dropdown-menu',
            '@radix-ui/react-tooltip', '@radix-ui/react-select',
            '@radix-ui/react-tabs', '@radix-ui/react-popover',
          ],
        },
      },
    },
  }
})
