// scripts/build-wasm.js
const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

// Ensure output directory exists
const outputDir = path.resolve(__dirname, '../public/wasm');
if (!fs.existsSync(outputDir)) {
  fs.mkdirSync(outputDir, { recursive: true });
}

// Build command
const buildCmd = `emcc 
  -O3 
  -s WASM=1 
  -s ALLOW_MEMORY_GROWTH=1
  -s MODULARIZE=1
  -s EXPORT_NAME="createWasmEncoder"
  -s EXPORTED_RUNTIME_METHODS='["cwrap", "setValue", "getValue"]'
  -s EXPORTED_FUNCTIONS='["_malloc", "_free"]'
  -I../../../encoder_core/include
  src/wasm/encoder.cpp
  -o public/wasm/encoder.js`;

console.log('Building WebAssembly module...');
try {
  execSync(buildCmd, { stdio: 'inherit' });
  console.log('WebAssembly build successful!');
} catch (error) {
  console.error('WebAssembly build failed:', error);
  process.exit(1);
}