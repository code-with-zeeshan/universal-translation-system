// Minimal Express server to serve example with proper WASM headers
const express = require('express');
const path = require('path');
const app = express();
const root = path.join(__dirname, '..');

// Serve dist with COOP/COEP and CORS for wasm
app.use((req, res, next) => {
  res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
  res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
  res.setHeader('Access-Control-Allow-Origin', '*'); // tighten in prod
  next();
});

app.use('/dist', express.static(path.join(root, 'dist'), {
  setHeaders: (res, filePath) => {
    if (filePath.endsWith('.wasm')) {
      res.setHeader('Content-Type', 'application/wasm');
      res.setHeader('Cache-Control', 'public, max-age=604800, immutable');
    }
  }
}));

app.use('/', express.static(path.join(root, 'example')));

const port = process.env.PORT || 3000;
app.listen(port, () => {
  console.log(`Example server running at http://localhost:${port}`);
  console.log('Make sure you have run: npm run build');
});