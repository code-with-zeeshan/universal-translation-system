# UniversalTranslationSDK for Android

A native Android SDK for the Universal Translation System with coordinator-aware routing and auto-updating encoder.

## Features
- **Edge encoding** via JNI (privacy-preserving)
- **Coordinator-aware**: single decoder → direct, multiple → proxy through coordinator
- **Auto-updating encoder**: checks HF Hub for newer model at init
- **Dynamic vocabulary packs** from HF Hub with CDN fallback

## Quick Start

```kotlin
// Direct decoder (single node)
val client = TranslationClient(context, decoderUrl = "http://decoder:8000")

// With coordinator (auto-routes: single node direct, multi-node via coordinator)
val client = TranslationClient(context,
    decoderUrl = "http://decoder:8000",
    coordinatorUrl = "http://coordinator:5100"
)

// Translate
val result = client.translate("Hello world", "en", "es")
```

## Constructor

```kotlin
TranslationClient(
    context: Context,
    decoderUrl: String = "https://api.yourdomain.com/decode",
    coordinatorUrl: String? = null  // enables multi-decoder routing
)
```

## Documentation
- [SDK Integration](../../docs/SDK_INTEGRATION.md)
- [Architecture](../../docs/ARCHITECTURE.md)
- [Publishing](../../docs/SDK_PUBLISHING.md)
