# Universal Translation SDK for Android

The Universal Translation SDK is a robust Android library designed for real-time text translation, combining on-device encoding with cloud-based decoding. It supports multiple languages, efficient vocabulary management, and background downloading capabilities, making it ideal for integrating translation functionality into Android applications.

## Features

- **On-Device Encoding**: Utilizes ONNX Runtime for efficient local text encoding.
- **Cloud-Based Decoding**: Sends encoded data to a cloud server for translation decoding.
- **Vocabulary Management**: Handles language-specific vocabulary packs with LRU caching and memory-mapped file loading.
- **Background Downloads**: Uses WorkManager to manage vocabulary pack downloads with network constraints.
- **Analytics Tracking**: Logs translation performance and errors for monitoring and optimization.
- **Native Integration**: Leverages JNI for native C++ encoding operations.
- **Material Design UI**: Includes a sample UI for language selection and translation display.
- **Performance Monitoring**: Tracks encoding and translation metrics for performance analysis.

## Project Structure

```
universal-translation-system/android/
└── UniversalTranslationSDK
    │   build.gradle
    │   proguard-rules.pro
    │
    └── src
        └── main
            │   AndroidManifest.xml
            │
            ├── cpp
            │   │   CMakeLists.txt
            │   │   jni_wrapper.cpp
            │
            ├── java
            │   │   com/universaltranslation/encoder
            │   │       MainActivity.kt
            │   │       TranslationClient.kt
            │   │       TranslationEncoder.kt
            │   │       VocabularyPackManager.kt
            │
            └── res
                ├── drawable
                │   │   ic_swap.xml
                │
                ├── layout
                │   │   activity_main.xml
                │
                ├── values
                │   │   colors.xml
                │   │   strings.xml
                │   │   themes.xml
                │
                └── xml
                    │   file_paths.xml
```

## Prerequisites

- **Android Studio**: Latest stable version recommended.
- **Android SDK**: Minimum SDK 21, Target SDK 34.
- **Kotlin**: Version 1.8 or higher.
- **CMake**: Version 3.22.1 or higher for native code compilation.
- **Dependencies**: ONNX Runtime, OkHttp, Gson, WorkManager, and AndroidX libraries.
- **Internet Connection**: Required for vocabulary downloads and cloud-based decoding.

## Installation

1. **Add the SDK to Your Project**:
   - Clone the repository or include the SDK as a module in your Android project.
   - Update your app's `build.gradle`:

     ```gradle
     implementation project(':UniversalTranslationSDK')
     ```

2. **Configure Native Libraries**:
   - Ensure the `jniLibs` directory includes ONNX Runtime libraries for supported ABIs (`armeabi-v7a`, `arm64-v8a`, `x86`, `x86_64`).
   - The `CMakeLists.txt` file manages native code compilation.

3. **Add Permissions**:
   - Include the following in your `AndroidManifest.xml`:

     ```xml
     <uses-permission android:name="android.permission.INTERNET" />
     <uses-permission android:name="android.permission.ACCESS_NETWORK_STATE" />
     ```

## Usage

### Initialization

Initialize the `TranslationClient` and `EnhancedVocabularyPackManager` in your activity or service:

```kotlin
val translationClient = TranslationClient(context)
val vocabManager = EnhancedVocabularyPackManager(context)

// Preload languages
val languages = setOf("en", "es", "fr")
vocabManager.downloadPacksForLanguages(languages)
```

### Performing a Translation

Translate text using the `translate` function:

```kotlin
lifecycleScope.launch {
    when (val result = translationClient.translate("Hello, world!", "en", "es")) {
        is TranslationResult.Success -> {
            println("Translation: ${result.translation}")
        }
        is TranslationResult.Error -> {
            println("Error: ${result.message}")
        }
    }
}
```

### Monitoring Download Progress

Observe vocabulary download progress using WorkManager's `LiveData`:

```kotlin
vocabManager.getDownloadProgress().observe(this) { workInfos ->
    workInfos.forEach { workInfo ->
        when (workInfo.state) {
            WorkInfo.State.RUNNING -> {
                val progress = workInfo.progress.getInt("progress", 0)
                println("Download progress: $progress%")
            }
            WorkInfo.State.SUCCEEDED -> println("Download completed")
            WorkInfo.State.FAILED -> println("Download failed")
            else -> {}
        }
    }
}
```

### Cleaning Up

Ensure proper cleanup to avoid memory leaks:

```kotlin
override fun onDestroy() {
    super.onDestroy()
    translationClient.destroy()
    vocabManager.cancelAllDownloads()
}
```

## Key Components

### TranslationClient.kt
- Coordinates local encoding and cloud-based decoding.
- Handles HTTP requests to the translation server.
- Tracks analytics for translation success and errors.

### TranslationEncoder.kt
- Manages local encoding using native C++ libraries via ONNX Runtime.
- Loads and caches vocabulary packs.
- Monitors performance metrics using `PerformanceMonitor`.

### VocabularyPackManager.kt
- Manages vocabulary pack downloads using WorkManager.
- Implements efficient loading with memory-mapped files for large vocabularies (>5MB).
- Uses an LRU cache for quick access to frequently used packs.

### MainActivity.kt
- Provides a sample UI with language spinners, input/output text fields, and a translate button.
- Demonstrates SDK integration with Material Design components.

### jni_wrapper.cpp
- JNI interface for native encoding operations.
- Integrates with ONNX Runtime for model execution.

### AndroidManifest.xml
- Defines necessary permissions and application components.
- Configures file provider for vocabulary storage.

### Resource Files
- **activity_main.xml**: Defines the UI layout with Material Design components, including spinners, text fields, and buttons.
- **strings.xml**: Contains language names, codes, and UI strings.
- **colors.xml**: Defines color resources for the UI theme.
- **themes.xml**: Configures Material Design themes for light and dark modes.
- **ic_swap.xml**: Provides a vector drawable for the language swap button.
- **file_paths.xml**: Configures file provider paths for vocabulary storage.

## Supported Languages

The SDK supports the following language packs:

| Pack Name    | Languages Supported                     |
|--------------|----------------------------------------|
| Latin        | English (en), Spanish (es), French (fr), German (de), Italian (it), Portuguese (pt) |
| CJK          | Chinese (zh), Japanese (ja), Korean (ko) |
| Arabic       | Arabic (ar)                            |
| Devanagari   | Hindi (hi)                             |
| Cyrillic     | Russian (ru), Ukrainian (uk)           |

## Configuration

### Build Configuration
- **build.gradle**:
  - Specifies dependencies (ONNX Runtime, OkHttp, Gson, WorkManager, etc.).
  - Configures Kotlin and CMake for native code compilation.
  - Sets up Maven publishing for SDK distribution.

- **proguard-rules.pro**:
  - Ensures native methods and data classes are preserved.
  - Configures rules for ONNX Runtime, OkHttp, Gson, and Kotlin coroutines.

### Native Code
- **CMakeLists.txt**:
  - Configures the build for the native `universal_encoder` library.
  - Links against ONNX Runtime and other dependencies (lz4, msgpackc).

### Network Configuration
- The default decoder URL is set to `https://api.yourdomain.com/decode`.
- Update the `decoderUrl` in `TranslationClient` to your server endpoint.

## ProGuard Rules

The `proguard-rules.pro` file includes rules to prevent the stripping of critical classes and methods, including:
- Native methods
- SDK classes (`com.universaltranslation.encoder.*`)
- ONNX Runtime and Gson classes
- Kotlin coroutine classes

## UI Customization

To customize the UI, modify the following resource files:
- **activity_main.xml**: Adjust the layout structure and Material Design components.
- **colors.xml**: Update color values for branding.
- **themes.xml**: Customize Material Design themes.
- **strings.xml**: Modify language names, UI text, or add new languages.

## Testing

- **Unit Tests**: Use JUnit and Mockito for unit testing (`testImplementation`).
- **Instrumentation Tests**: Use Espresso for UI testing (`androidTestImplementation`).
- **Coroutine Testing**: Use `kotlinx-coroutines-test` for testing coroutines.

Run tests with:

```bash
./gradlew test
./grad gradlew connectedAndroidTest
```

## Publishing

The SDK is configured for Maven publishing. To publish to a Maven repository:

```bash
./gradlew publish
```

Update the `groupId`, `artifactId`, and `version` in `build.gradle` as needed:

```gradle
groupId = 'com.universaltranslation'
artifactId = 'encoder-sdk'
version = '1.0.0'
```

## Troubleshooting

- **Native Library Loading Error**:
  - Ensure `libuniversal_encoder.so` and `libonnxruntime.so` are included in `jniLibs` for all supported ABIs.
  - Verify `System.loadLibrary("universal_encoder")` in `TranslationEncoder.kt`.

- **Network Errors**:
  - Check the `decoderUrl` configuration.
  - Ensure internet permissions are granted.
  - Verify network connectivity using `ConnectivityManager`.

- **Vocabulary Download Failures**:
  - Confirm the `downloadUrl` in `VocabularyManager.kt`.
  - Check WorkManager constraints (`NetworkType.UNMETERED` for large downloads).
  - Monitor logs using `Logcat` with the tag `VocabularyDownloadWorker`.

- **Performance Issues**:
  - Review `PerformanceMonitor` metrics using `translationClient.getPerformanceMetrics()`.
  - Optimize large vocabulary loading by adjusting `LARGE_FILE_THRESHOLD` in `VocabularyPackManager.kt`.

## Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m 'Add your feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Create a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.