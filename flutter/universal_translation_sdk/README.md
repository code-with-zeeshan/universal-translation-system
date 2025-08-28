# UniversalTranslationSDK for Flutter

A cross-platform Flutter SDK for the Universal Translation System, supporting config-driven language management, dynamic vocabulary packs, and seamless integration with the advanced coordinator and monitoring system.

## Features
- Config-driven language and vocabulary management (see `data/config.yaml`)
- Edge encoding, cloud decoding (privacy-preserving)
- Dynamic vocabulary packs (download only what you need)
- Coordinator integration for load balancing and health checks
- Prometheus metrics for monitoring
- Easy integration with Flutter apps (Android, iOS, desktop)

## Quick Start

1) Install the package
```yaml
# pubspec.yaml
dependencies:
  universal_translation_sdk:
    path: ../../flutter/universal_translation_sdk
```

2) Platform setup
- iOS: ensure Podfile uses iOS 13.0+ and run `pod install`.
- Android: minSdkVersion 23+ recommended.

3) Initialize and use (Coordinator binary endpoint)
```dart
import 'package:universal_translation_sdk/universal_translation_sdk.dart';
import 'package:http/http.dart' as http;

final encoder = TranslationEncoder();
await encoder.initialize();
await encoder.loadVocabulary('en', 'es');
final encoded = await encoder.encode(text: 'Hello, world!', sourceLang: 'en', targetLang: 'es');

final resp = await http.post(
  Uri.parse('http://localhost:8002/api/decode'),
  headers: {
    'Content-Type': 'application/octet-stream',
    'X-Source-Language': 'en',
    'X-Target-Language': 'es',
  },
  body: encoded,
);
```

4) Monitoring
- Use coordinator `/api/status` and Prometheus metrics.

## Adding New Languages
- Update `data/config.yaml` and run the pipeline to add new languages
- Vocabulary packs are managed automatically

## Documentation
- See [docs/SDK_INTEGRATION.md](../../docs/SDK_INTEGRATION.md) and [docs/ARCHITECTURE.md](../../docs/ARCHITECTURE.md)
- Publishing steps: [docs/SDK_PUBLISHING.md](../../docs/SDK_PUBLISHING.md)

## Monitoring
- All requests and node health are visible in the coordinator dashboard
- Prometheus metrics available for advanced analytics

---

For more, see the main repo and the coordinator dashboard.