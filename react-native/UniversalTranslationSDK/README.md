# UniversalTranslationSDK for React Native

A React Native SDK for the Universal Translation System, supporting config-driven language management, dynamic vocabulary packs, and seamless integration with the advanced coordinator and monitoring system.

## Features
- Config-driven language and vocabulary management (see `data/config.yaml`)
- Edge encoding (where supported), cloud decoding (privacy-preserving)
- Dynamic vocabulary packs (download only what you need)
- Coordinator integration for load balancing and health checks
- Prometheus metrics for monitoring
- Easy integration with React Native apps

## Quick Start

1) Install the package
```bash
npm install @universal-translation/react-native-sdk
# or
yarn add @universal-translation/react-native-sdk
```

2) iOS pods
```bash
cd ios && pod install && cd ..
```

3) Initialize and use (Coordinator binary endpoint)
```tsx
import { TranslationClient } from '@universal-translation/react-native-sdk';

const client = new TranslationClient({
  decoderUrl: 'http://localhost:8002/api/decode',
  apiKey: process.env.API_KEY,
});

const result = await client.translate({ text: 'Hello world', sourceLang: 'en', targetLang: 'es' });
```

4) Linking notes
- Autolinking works for RN 0.60+.
- If adding a native encoder module, ensure Android NDK/CMake and iOS Podspec are configured.

5) Monitoring
- Use coordinator `/api/status` and Prometheus metrics.

## Adding New Languages
- Update `data/config.yaml` and run the pipeline to add new languages
- Vocabulary packs are managed automatically

## Documentation
- See [docs/SDK_INTEGRATION.md](../../docs/SDK_INTEGRATION.md) and [docs/ARCHITECTURE.md](../../docs/ARCHITECTURE.md)

## Monitoring
- All requests and node health are visible in the coordinator dashboard
- Prometheus metrics available for advanced analytics

---

For more, see the main repo and the coordinator dashboard.