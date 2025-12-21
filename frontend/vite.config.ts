import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/image-labeling': {
        target: 'http://localhost:8000', // Backend url
        changeOrigin: true,
      },
      '/experiment': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        secure: false,
      },
      // This redirects all /images calls to the Python backend
      '/images': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        secure: false,
      }
    },
  },
});
