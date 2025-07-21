# Universal Translation SDK for iOS

[![Platform](https://img.shields.io/badge/platform-iOS%20%7C%20macOS%20%7C%20tvOS%20%7C%20watchOS-lightgrey.svg)]()
[![Swift Version](https://img.shields.io/badge/Swift-5.7+-orange.svg)]()
[![License](https://img.shields.io/badge/license-MIT-blue.svg)]()

The Universal Translation SDK provides on-device encoding and cloud-based decoding for fast, efficient multilingual translation on Apple platforms.

## Features

- 🚀 **On-device encoding** using CoreML with Neural Engine optimization
- 🌍 **20+ languages** supported with dynamic vocabulary loading
- 📦 **Compact models** (~125MB INT8 quantized)
- 🔄 **Background translation** support
- 📱 **SwiftUI components** included
- 🔒 **Privacy-focused** - text processing happens on-device
- ⚡ **Offline capability** for downloaded language packs

## Requirements

- iOS 15.0+ / macOS 12.0+ / tvOS 15.0+ / watchOS 8.0+
- Xcode 14.0+
- Swift 5.7+

## Installation

### Swift Package Manager

Add the following to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/yourusername/universal-translation-system.git", from: "1.0.0")
]
```

Or in Xcode:
1. File → Add Package Dependencies
2. Enter: `https://github.com/yourusername/universal-translation-system.git`
3. Select "Up to Next Major Version" with "1.0.0"

### CocoaPods

Add to your `Podfile`:

```ruby
pod 'UniversalTranslationSDK', '~> 1.0'
```

Then run:
```bash
pod install
```

## Quick Start

### Basic Translation

```swift
import UniversalTranslationSDK

// Initialize the client
let client = try TranslationClient(decoderURL: "https://api.yourdomain.com/decode")

// Translate text
let response = try await client.translate(
    text: "Hello, world!",
    from: "en",
    to: "es"
)
print(response.translation) // "¡Hola, mundo!"
```

### SwiftUI Integration

```swift
import SwiftUI
import UniversalTranslationSDK

@main
struct MyApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
                .translationEnabled() // Enable translation globally
        }
    }
}

// In your views
struct ContentView: View {
    @Environment(\.translationClient) var translator
    
    var body: some View {
        VStack {
            // Quick translation button
            QuickTranslationButton(
                text: "Hello world",
                sourceLang: "en",
                targetLang: "es"
            )
            
            // Full translation view
            TranslationView()
        }
    }
}
```

### Batch Translation

```swift
let texts = ["Hello", "How are you?", "Good morning"]
let responses = try await client.translateBatch(
    texts: texts,
    from: "en",
    to: "fr"
)
```

### Translation with Options

```swift
let options = TranslationOptions(
    formality: .formal,
    domain: .business,
    preserveFormatting: true
)

let response = try await client.translate(
    text: businessEmail,
    from: "en",
    to: "de",
    options: options
)
```

## Advanced Usage

### Background Translation

Enable background translation for better performance:

```swift
// In your app delegate or scene
await client.enableBackgroundTranslation()
```

Add to your `Info.plist`:
```xml
<key>BGTaskSchedulerPermittedIdentifiers</key>
<array>
    <string>com.universal.translation.process</string>
</array>
<key>UIBackgroundModes</key>
<array>
    <string>fetch</string>
    <string>processing</string>
</array>
```

### Offline Translation

Download language packs for offline use:

```swift
// Download specific language pairs
await client.downloadOfflineModels(for: ["en", "es", "fr"])

// Check offline availability
if client.isOfflineAvailable(source: "en", target: "es") {
    // Can translate without internet
}
```

### Custom Vocabulary Management

```swift
// Prefetch vocabularies based on user preferences
await client.vocabularyManager.prefetchVocabulariesForUserLanguages()

// Get current memory usage
let memoryMB = await client.getMemoryUsage() / 1_048_576
print("Using \(memoryMB)MB of memory")
```

### Performance Monitoring

```swift
// Track translation performance
let metrics = await client.encoder.getPerformanceMetrics()
for (operation, timings) in metrics {
    let average = timings.reduce(0, +) / Double(timings.count)
    print("\(operation): \(average * 1000)ms average")
}
```

## Language Support

The SDK supports the following languages:

| Code | Language | Script | RTL |
|------|----------|--------|-----|
| en | English | Latin | No |
| es | Spanish | Latin | No |
| fr | French | Latin | No |
| de | German | Latin | No |
| zh | Chinese | Han | No |
| ja | Japanese | Kana/Kanji | No |
| ko | Korean | Hangul | No |
| ar | Arabic | Arabic | Yes |
| hi | Hindi | Devanagari | No |
| ru | Russian | Cyrillic | No |
| pt | Portuguese | Latin | No |
| it | Italian | Latin | No |
| tr | Turkish | Latin | No |
| th | Thai | Thai | No |
| vi | Vietnamese | Latin | No |
| pl | Polish | Latin | No |
| uk | Ukrainian | Cyrillic | No |
| nl | Dutch | Latin | No |
| id | Indonesian | Latin | No |
| sv | Swedish | Latin | No |

## Architecture

The SDK uses a hybrid architecture:

1. **On-device Encoder**: Processes text locally using CoreML
2. **Vocabulary Packs**: Dynamic loading based on language pairs
3. **Cloud Decoder**: Handles translation using optimized models
4. **Compression**: LZ4 compression for network efficiency

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Text      │────▶│   Encoder    │────▶│  Compressed │
│   Input     │     │  (CoreML)    │     │   Output    │
└─────────────┘     └──────────────┘     └──────┬───────┘
                                                 │
                           Network               ▼
                    ┌──────────────────────────────┐
                    │                              │
                    ▼                              │
┌─────────────┐     ┌──────────────┐     ┌───────┴─────┐
│ Translation │◀────│   Decoder    │◀────│   Cloud     │
│   Output    │     │  (Server)    │     │   API       │
└─────────────┘     └──────────────┘     └─────────────┘
```

## Error Handling

The SDK provides detailed error types:

```swift
do {
    let response = try await client.translate(text: text, from: "en", to: "es")
} catch TranslationError.networkError(let error) {
    // Handle network issues
} catch TranslationError.unsupportedLanguage(let lang) {
    // Handle unsupported language
} catch TranslationError.vocabularyNotLoaded {
    // Vocabulary needs to be downloaded
} catch {
    // Handle other errors
}
```

## Privacy & Security

- Text is processed on-device before transmission
- Only compressed embeddings are sent to the server
- No raw text is stored or transmitted
- Supports App Transport Security (ATS)
- Vocabulary packs are downloaded over HTTPS

## Troubleshooting

### Common Issues

1. **Model not found**
   - Ensure the CoreML model is included in your app bundle
   - Check that the model file has the correct target membership

2. **High memory usage**
   - Limit concurrent translations
   - Clear cache regularly: `client.clearCache()`
   - Use edge vocabulary packs for reduced memory

3. **Slow first translation**
   - This is normal - the model needs to warm up
   - Consider pre-warming: `await client.encoder.initialize()`

### Debug Logging

Enable detailed logging:

```swift
// In your app's initialization
UniversalTranslationSDK.enableDebugLogging()
```

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for details.

## License

This SDK is available under the MIT license. See [LICENSE](../../LICENSE) for details.

## Support

- 📧 Email: support@universaltranslation.com
- 🐛 Issues: [GitHub Issues](https://github.com/code-with-zeeshan/universal-translation-system/issues)
- 📖 Docs: [Full Documentation](https://docs.universaltranslation.com)
