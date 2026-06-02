# UniversalTranslationSDK for React Native

A React Native SDK for the Universal Translation System, supporting dynamic vocabulary packs and coordinator integration.

## Features
- Edge encoding, cloud decoding (privacy-preserving)
- Dynamic vocabulary packs
- Coordinator integration for load balancing

## Quick Start

1) Install:
```bash
npm install @universal-translation/react-native-sdk
```

2) iOS pods:
```bash
cd ios && pod install && cd ..
```

3) Use:
```tsx
import { TranslationClient } from '@universal-translation/react-native-sdk';

const client = new TranslationClient({
  decoderUrl: 'http://localhost:5100/api/decode',
  apiKey: process.env.API_KEY,
});

const result = await client.translate({ text: 'Hello world', sourceLang: 'en', targetLang: 'es' });
```

## Documentation
- See [docs/SDK_INTEGRATION.md](../../docs/SDK_INTEGRATION.md) and [docs/ARCHITECTURE.md](../../docs/ARCHITECTURE.md)

---

For more, see the main repo.
