# UniversalTranslationSDK for Web

A Web SDK for the Universal Translation System, supporting config-driven language management, dynamic vocabulary packs, and seamless integration with the advanced coordinator and monitoring system.

## Features
- Config-driven language and vocabulary management (see `data/config.yaml`)
- Edge encoding (where supported), cloud decoding (privacy-preserving)
- Dynamic vocabulary packs (download only what you need)
- Coordinator integration for load balancing and health checks
- Prometheus metrics for monitoring
- Easy integration with web apps (React, Vue, etc.)

## Quick Start

1. Add the SDK to your project:
   - Install via npm or yarn
   - Configure the decoder/coordinator endpoint in your app

2. Initialize and use:
```js
import { TranslationClient } from 'universal-translation-sdk';
const client = new TranslationClient({ decoderUrl: 'https://coordinator.example.com/decode' });
const result = await client.translate({ text: 'Hello world', sourceLang: 'en', targetLang: 'es' });
```

3. Monitor and manage:
- Use the coordinator dashboard to view node health, load, and analytics
- Prometheus metrics are available for all translation requests

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