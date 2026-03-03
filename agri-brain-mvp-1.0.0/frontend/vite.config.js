// frontend/vite.config.js
import { defineConfig } from 'vite'
import os from 'node:os'
import path from 'node:path'

// Put Vite's cache in a non-synced temp dir (outside Dropbox)
// You can override with env var: VITE_CACHE_DIR="C:\\path\\to\\somewhere"
const cacheRoot =
  process.env.VITE_CACHE_DIR ||
  path.join(os.tmpdir(), 'vite-cache', 'agri-brain-frontend')

export default defineConfig({
  cacheDir: cacheRoot,

  server: {
    host: true,
    // Polling avoids rename races on networked/synced filesystems
    watch: {
      usePolling: true,
      interval: 200,
      awaitWriteFinish: { stabilityThreshold: 500, pollInterval: 200 },
      ignored: ['**/.git/**', '**/dist/**', '**/node_modules/**']
    }
  },

  // Ensure fresh prebundles go into the temp cache dir
  optimizeDeps: {
    force: true
  },

  build: {
    outDir: 'dist',
    emptyOutDir: true
  }
})
