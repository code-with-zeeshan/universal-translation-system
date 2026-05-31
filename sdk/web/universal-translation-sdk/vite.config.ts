import { defineConfig } from 'vite';
import { resolve } from 'path';

export default defineConfig({
  server: {
    port: 3000,
    open: true,
    cors: true
  },
  
  resolve: {
    alias: {
      '@': resolve(__dirname, './src')
    }
  },
  
  build: {
    lib: {
      entry: resolve(__dirname, 'src/index.ts'),
      name: 'UniversalTranslation',
      formats: ['es', 'umd'],
      fileName: (format) => `universal-translation.${format}.js`
    },
    rollupOptions: {
      external: ['react', 'react-dom', 'onnxruntime-web'],
      output: {
        globals: {
          react: 'React',
          'react-dom': 'ReactDOM',
          'onnxruntime-web': 'ort'
        }
      }
    },
    sourcemap: true,
    minify: 'terser'
  },
  
  optimizeDeps: {
    include: ['onnxruntime-web']
  }
});