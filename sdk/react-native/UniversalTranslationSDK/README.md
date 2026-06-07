# UniversalTranslationSDK for React Native

A React Native SDK for the Universal Translation System with coordinator-aware routing, local decoder preference, and auto-updating encoder.

## Features
- **Edge encoding** via native iOS/Android modules (privacy-preserving)
- **Coordinator-aware**: single decoder → direct, multiple → proxy through coordinator
- **Local decoder preference**: auto-discovers local decoder, falls back to cloud
- **Auto-updating encoder**: checks HF Hub for newer model at init
- **Dynamic vocabulary packs** with retry logic

## Quick Start

```tsx
import { TranslationClient } from '@universal-translation/react-native-sdk';

// Auto-routes: local decoder → coordinator → cloud fallback
const client = new TranslationClient({
  coordinatorUrl: 'http://coordinator:5100',
});

// Or force privacy mode (local decoder only):
const client = new TranslationClient({
  localDecoderUrl: 'http://localhost:8000',
  preferLocal: true,
});

const result = await client.translate({
  text: 'Hello world',
  sourceLang: 'en',
  targetLang: 'es',
});
```

## Options

| Option | Default | Description |
|---|---|---|
| `decoderUrl` | env `DECODER_API_URL` | Direct decoder endpoint |
| `coordinatorUrl` | env `COORDINATOR_API_URL` | Coordinator for multi-decoder routing |
| `localDecoderUrl` | — | Local decoder (privacy mode) |
| `preferLocal` | `true` | Prefer local decoder over cloud |

## Installation

```bash
npm install @universal-translation/react-native-sdk
cd ios && pod install && cd ..
```

## Documentation
- [SDK Integration](../../docs/SDK_INTEGRATION.md)
- [Architecture](../../docs/ARCHITECTURE.md)
- [Publishing](../../docs/SDK_PUBLISHING.md)
