# UniversalTranslationSDK for Web

A Web SDK for the Universal Translation System with edge encoding, coordinator integration, and automatic local decoder discovery.

## Features
- **Edge encoding** via WASM/ONNX (privacy-preserving: raw text never leaves device)
- **Coordinator-aware**: single decoder → direct call, multiple decoders → proxy through coordinator
- **Local decoder preference**: auto-discovers `localhost:8000/8080/9000`, falls back to cloud
- **Auto-updating encoder**: checks HF Hub for newer `encoder.onnx` at init
- **Dynamic vocabulary packs** from HF Hub with CDN fallback

## Quick Start

```ts
import { TranslationClient } from '@universal-translation/web-sdk';

// Auto-detect: local decoder → coordinator → cloud fallback
const client = new TranslationClient({
  coordinatorUrl: 'http://coordinator:5100',
  useWasm: true,
});

// Or force local decoder for privacy:
const client = new TranslationClient({
  localDecoderUrl: 'http://localhost:8000',
  preferLocal: true,
});

const result = await client.translate({
  text: 'Hello world',
  sourceLang: 'en',
  targetLang: 'es'
});
```

## Options

| Option | Default | Description |
|---|---|---|
| `decoderUrl` | env `DECODER_API_URL` | Direct decoder endpoint |
| `coordinatorUrl` | env `COORDINATOR_API_URL` | Coordinator for multi-decoder routing |
| `localDecoderUrl` | — | Local decoder (privacy mode) |
| `preferLocal` | `true` | Prefer local decoder over cloud |
| `useWasm` | `true` | Enable WASM local encoding |

## WASM Headers
Serve `dist/wasm/` with COOP/COEP headers:
- `Cross-Origin-Opener-Policy: same-origin`
- `Cross-Origin-Embedder-Policy: require-corp`

See `example/server.js` for a working server.

## Documentation
- [SDK Integration](../../docs/SDK_INTEGRATION.md)
- [Architecture](../../docs/ARCHITECTURE.md)
- [WASM details](src/wasm/README.md)
