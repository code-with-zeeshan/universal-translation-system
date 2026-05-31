# UniversalTranslationSDK for Android

A native Android SDK for the Universal Translation System, supporting config-driven language management, dynamic vocabulary packs, and integration with the coordinator.

## Features
- Edge encoding, cloud decoding (privacy-preserving)
- Dynamic vocabulary packs (download only what you need)
- Coordinator integration for load balancing
- Easy integration with Android apps

## Quick Start

1) Add the SDK to your project (see `docs/SDK_PUBLISHING.md` for publishing options):
```gradle
dependencies { implementation 'com.example:universal-translation-sdk:1.0.0' }
```

2) Initialize and use:
```kotlin
val encoder = TranslationEncoder(context)
encoder.loadVocabulary("en", "es")
val encoded = encoder.encode("Hello world", "en", "es")

val req = okhttp3.Request.Builder()
  .url("http://localhost:8002/api/decode")
  .addHeader("Content-Type", "application/octet-stream")
  .addHeader("X-Source-Language", "en")
  .addHeader("X-Target-Language", "es")
  .post(okhttp3.RequestBody.create(null, encoded))
  .build()
```

## Documentation
- See [docs/SDK_INTEGRATION.md](../../docs/SDK_INTEGRATION.md) and [docs/ARCHITECTURE.md](../../docs/ARCHITECTURE.md)
- Publishing steps: [docs/SDK_PUBLISHING.md](../../docs/SDK_PUBLISHING.md)

---

For more, see the main repo and the coordinator dashboard.
