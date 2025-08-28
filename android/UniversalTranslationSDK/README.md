# UniversalTranslationSDK for Android

A native Android SDK for the Universal Translation System, supporting config-driven language management, dynamic vocabulary packs, and seamless integration with the advanced coordinator and monitoring system.

## Features
- Config-driven language and vocabulary management (see `data/config.yaml`)
- Edge encoding, cloud decoding (privacy-preserving)
- Dynamic vocabulary packs (download only what you need)
- Coordinator integration for load balancing and health checks
- Prometheus metrics for monitoring
- Easy integration with Android apps

## Quick Start

1) Add the SDK to your project
- Option A: Local Maven (development)
```bash
# From repo root
./gradlew :android:UniversalTranslationSDK:publishToMavenLocal
```
```gradle
// app/build.gradle
repositories { mavenLocal(); google(); mavenCentral() }
dependencies { implementation 'com.example:universal-translation-sdk:1.0.0' }
```
- Option B: Direct module include (settings.gradle)
```gradle
include(':UniversalTranslationSDK')
project(':UniversalTranslationSDK').projectDir = new File(rootDir, '../android/UniversalTranslationSDK')
```
- Option C: AAR
Place the AAR in `app/libs` and add `implementation files('libs/universal-translation-sdk.aar')`.

2) Configure Gradle
```gradle
// app/build.gradle
android {
  defaultConfig { minSdkVersion 23 }
}
dependencies { implementation "com.squareup.okhttp3:okhttp:4.12.0" }
```

3) Initialize and use
```kotlin
val encoder = TranslationEncoder(context)
encoder.loadVocabulary("en", "es")
val encoded = encoder.encode("Hello world", "en", "es")
```

4) Binary POST to Coordinator
```kotlin
val req = okhttp3.Request.Builder()
  .url("http://localhost:8002/api/decode")
  .addHeader("Content-Type", "application/octet-stream")
  .addHeader("X-Source-Language", "en")
  .addHeader("X-Target-Language", "es")
  .post(okhttp3.RequestBody.create(null, encoded))
  .build()
```

5) Monitoring
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