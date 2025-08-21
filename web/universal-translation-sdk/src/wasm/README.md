# WebAssembly Encoder for Universal Translation System

This directory contains the WebAssembly implementation of the Universal Translation System encoder. The WebAssembly encoder provides a lightweight, client-side alternative to the ONNX-based encoder, allowing for faster encoding with less network traffic.

## Features

- **Lightweight**: The WebAssembly encoder is much smaller than the full ONNX model
- **Fast**: Native code execution for better performance
- **Offline Capable**: Can work without an internet connection once loaded
- **Fallback Support**: Automatically falls back to ONNX or cloud encoding if WebAssembly fails

## Building

To build the WebAssembly encoder, you need to have Emscripten installed. Then run:

```bash
npm run build:wasm
```

This will compile the C++ code in this directory to WebAssembly and place the output in the `public/wasm` directory.

## Usage

The WebAssembly encoder is automatically used by the Universal Translation SDK when available. You can control its usage with the following options:

```typescript
import { TranslationEncoder } from '@universal-translation/web-sdk';

const encoder = new TranslationEncoder({
  useWasmEncoder: true,  // Enable WebAssembly encoder (default: true)
  wasmEncoderPath: '/wasm/encoder.js',  // Path to WebAssembly module
  enableFallback: true  // Enable fallback to ONNX or cloud (default: true)
});

// Check if WebAssembly encoder is available
const hasWasm = encoder.hasWasmEncoder();
console.log(`WebAssembly encoder available: ${hasWasm}`);

// Use the encoder (will use WebAssembly if available)
const encoded = await encoder.encode('Hello world', 'en', 'es');
```

## Implementation Details

The WebAssembly encoder is implemented in C++ using Emscripten to compile to WebAssembly. It provides the following functionality:

- Text encoding to embeddings
- Vocabulary management
- Embedding compression

The encoder uses a simplified algorithm compared to the full ONNX model, but provides sufficient quality for most use cases.

## Fallback Mechanism

If the WebAssembly encoder fails to load or encode text, the SDK will automatically fall back to:

1. ONNX-based encoding (if available)
2. Cloud-based encoding (if enabled)

This ensures that the translation process continues to work even if WebAssembly is not supported or encounters an error.

## Demo

A demo of the WebAssembly encoder is available in the `example/wasm-demo.html` file. To run it:

1. Build the SDK: `npm run build`
2. Serve the example directory: `npm run example`
3. Open `http://localhost:3000/wasm-demo.html` in your browser

## Performance Considerations

The WebAssembly encoder is optimized for performance, but there are a few things to keep in mind:

- The first encoding operation may be slower due to WebAssembly compilation
- Performance varies across browsers and devices
- For best performance, preload the WebAssembly module before encoding

## Browser Compatibility

The WebAssembly encoder works in all modern browsers that support WebAssembly, including:

- Chrome 57+
- Firefox 53+
- Safari 11+
- Edge 16+

For older browsers, the SDK will automatically fall back to ONNX or cloud encoding.