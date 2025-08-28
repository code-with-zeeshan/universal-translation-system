# UniversalTranslationSDK for iOS

A native iOS SDK for the Universal Translation System, supporting config-driven language management, dynamic vocabulary packs, and seamless integration with the advanced coordinator and monitoring system.

## Features
- Config-driven language and vocabulary management (see `data/config.yaml`)
- Edge encoding, cloud decoding (privacy-preserving)
- Dynamic vocabulary packs (download only what you need)
- Coordinator integration for load balancing and health checks
- Prometheus metrics for monitoring
- Easy integration with iOS apps

## Quick Start

1) Install via CocoaPods (dev)
```ruby
# Podfile (app)
platform :ios, '13.0'
use_frameworks!

target 'YourApp' do
  pod 'UniversalTranslationSDK', :path => '../../ios/UniversalTranslationSDK'
end
```
```bash
cd ios && pod install && cd ..
```

Or, install via Swift Package Manager by adding the repo URL in Xcode > Package Dependencies.

2) Initialize and use (Coordinator binary endpoint)
```swift
import UniversalTranslationSDK

let encoder = TranslationEncoder()
try encoder.loadVocabulary(source: "en", target: "es")
let encoded = try encoder.encode(text: "Hello world", source: "en", target: "es")
```

Binary POST to Coordinator
```swift
import Foundation

let url = URL(string: "http://localhost:8002/api/decode")!
var req = URLRequest(url: url)
req.httpMethod = "POST"
req.addValue("application/octet-stream", forHTTPHeaderField: "Content-Type")
req.addValue("en", forHTTPHeaderField: "X-Source-Language")
req.addValue("es", forHTTPHeaderField: "X-Target-Language")
req.httpBody = encoded

let task = URLSession.shared.dataTask(with: req) { data, resp, err in
  // handle response
}
task.resume()
```

3) Monitoring
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
