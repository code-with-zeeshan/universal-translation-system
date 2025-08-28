# UniversalTranslationSDK for Web

A Web SDK for the Universal Translation System, supporting config-driven language management, dynamic vocabulary packs, and seamless integration with the advanced coordinator and monitoring system.

## Features
- Config-driven language and vocabulary management (see `data/config.yaml`)
- Edge encoding (where supported), cloud decoding (privacy-preserving)
- **WebAssembly implementation** for faster, more efficient encoding
- Dynamic vocabulary packs (download only what you need)
- Coordinator integration for load balancing and health checks
- Prometheus metrics for monitoring
- Easy integration with web apps (React, Vue, etc.)

## Quick Start

1) Install and build
```bash
npm install
npm run build
```

2) Run example
- Vite dev server:
```bash
npm run example
# Open http://localhost:3000
```
- Express server with WASM headers:
```bash
node example/server.js
# Open http://localhost:3000
```

3) Initialize and use in your app
```ts
import { TranslationClient } from '@universal-translation/web-sdk';
const client = new TranslationClient({
  decoderUrl: 'http://localhost:8002/api/decode',
  useWasm: true,
});
const result = await client.translate({ text: 'Hello world', sourceLang: 'en', targetLang: 'es' });
```

4) Monitor and manage
- Use the coordinator status endpoint and dashboard
- Prometheus metrics are available for all translation requests

## Adding New Languages
- Update `data/config.yaml` and run the pipeline to add new languages
- Vocabulary packs are managed automatically

## Documentation
- See [docs/SDK_INTEGRATION.md](../../docs/SDK_INTEGRATION.md) and [docs/ARCHITECTURE.md](../../docs/ARCHITECTURE.md)
- Publishing steps: [docs/SDK_PUBLISHING.md](../../docs/SDK_PUBLISHING.md)
- For WebAssembly implementation details, see [src/wasm/README.md](src/wasm/README.md)

## Monitoring
- All requests and node health are visible in the coordinator dashboard
- Prometheus metrics available for advanced analytics

---

For more, see the main repo and the coordinator dashboard.