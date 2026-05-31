# UniversalTranslationSDK for Web

A Web SDK for the Universal Translation System, supporting WASM-based edge encoding and coordinator integration.

## Features
- Edge encoding (WASM), cloud decoding (privacy-preserving)
- WebAssembly implementation for faster encoding
- Dynamic vocabulary packs
- Coordinator integration for load balancing

## Quick Start

1) Install and build:
```bash
npm install
npm run build
```

2) Use in your app:
```ts
import { TranslationClient } from '@universal-translation/web-sdk';
const client = new TranslationClient({
  decoderUrl: 'http://localhost:8002/api/decode',
  useWasm: true,
});
const result = await client.translate({ text: 'Hello world', sourceLang: 'en', targetLang: 'es' });
```

## WASM Headers
Serve `dist/wasm/` with COOP/COEP headers (see `example/server.js`):
- `Cross-Origin-Opener-Policy: same-origin`
- `Cross-Origin-Embedder-Policy: require-corp`

## Documentation
- See [docs/SDK_INTEGRATION.md](../../docs/SDK_INTEGRATION.md) and [docs/ARCHITECTURE.md](../../docs/ARCHITECTURE.md)
- WASM details: [src/wasm/README.md](src/wasm/README.md)

---

For more, see the main repo.
