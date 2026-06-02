# UniversalTranslationSDK for iOS

A native iOS SDK for the Universal Translation System, supporting config-driven language management, dynamic vocabulary packs, and integration with the coordinator.

## Features
- Edge encoding, cloud decoding (privacy-preserving)
- Dynamic vocabulary packs
- Coordinator integration for load balancing
- Easy integration with iOS apps

## Quick Start

1) Install via CocoaPods:
```ruby
pod 'UniversalTranslationSDK', :path => '../sdk/ios/UniversalTranslationSDK'
```

Or via Swift Package Manager.

2) Initialize and use:
```swift
import UniversalTranslationSDK

let encoder = TranslationEncoder()
try encoder.loadVocabulary(source: "en", target: "es")
let encoded = try encoder.encode(text: "Hello world", source: "en", target: "es")

let url = URL(string: "http://localhost:5100/api/decode")!
var req = URLRequest(url: url)
req.httpMethod = "POST"
req.addValue("application/octet-stream", forHTTPHeaderField: "Content-Type")
req.addValue("en", forHTTPHeaderField: "X-Source-Language")
req.addValue("es", forHTTPHeaderField: "X-Target-Language")
req.httpBody = encoded
```

## Documentation
- See [docs/SDK_INTEGRATION.md](../../docs/SDK_INTEGRATION.md) and [docs/ARCHITECTURE.md](../../docs/ARCHITECTURE.md)

---

For more, see the main repo.
