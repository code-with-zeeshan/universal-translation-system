# UniversalTranslationSDK for iOS

A native iOS SDK for the Universal Translation System with coordinator-aware routing, batch translation, and auto-updating encoder.

## Features
- **Edge encoding** via CoreML (privacy-preserving)
- **Coordinator-aware**: single decoder → direct, multiple → proxy through coordinator
- **Auto-updating encoder**: checks HF Hub for newer model at init
- **Background translation** and batch processing
- **Dynamic vocabulary packs** with LRU cache

## Quick Start

```swift
import UniversalTranslationSDK

// Direct decoder
let client = try TranslationClient(decoderURL: "http://decoder:8000")

// With coordinator (auto-routes based on pool size)
let client = try TranslationClient(
    decoderURL: "http://decoder:8000",
    coordinatorURL: "http://coordinator:5100"
)

// Translate
let response = try await client.translate(text: "Hello world", from: "en", to: "es")

// Batch translate
let results = try await client.translateBatch(
    texts: ["Hello", "World"],
    from: "en", to: "es"
)
```

## Constructor

```swift
TranslationClient(
    decoderURL: String = "https://api.yourdomain.com/decode",
    coordinatorURL: String? = nil
)
```

## Installation

### CocoaPods
```ruby
pod 'UniversalTranslationSDK', :path => '../sdk/ios/UniversalTranslationSDK'
```

### Swift Package Manager
Add via Xcode: File → Add Packages → `https://github.com/yourusername/universal-translation-system`

## Documentation
- [SDK Integration](../../docs/SDK_INTEGRATION.md)
- [Architecture](../../docs/ARCHITECTURE.md)
- [Publishing](../../docs/SDK_PUBLISHING.md)
