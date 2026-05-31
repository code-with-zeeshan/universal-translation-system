# SDK Publishing and Linking Guide

This guide covers publishing the Android and iOS SDKs, and linking steps for React Native.

All SDKs reside under the `sdk/` directory:
- `sdk/android/UniversalTranslationSDK/`
- `sdk/ios/UniversalTranslationSDK/`
- `sdk/flutter/universal_translation_sdk/`
- `sdk/react-native/UniversalTranslationSDK/`
- `sdk/web/universal-translation-sdk/`

---

## Android: Gradle (Maven Publishing)

### 1) Add Maven Publishing
```gradle
// sdk/android/UniversalTranslationSDK/build.gradle
plugins {
  id 'com.android.library'
  id 'maven-publish'
}

publishing {
  publications {
    release(MavenPublication) {
      from components.release
      groupId = 'com.example'
      artifactId = 'universal-translation-sdk'
      version = '1.0.0'
      pom {
        name = 'Universal Translation SDK (Android)'
        description = 'Android encoder client for Universal Translation System'
        licenses { license { name = 'Apache-2.0' } }
      }
    }
  }
  repositories {
    mavenLocal()
  }
}
```

### 2) Build & publish
```bash
cd sdk/android/UniversalTranslationSDK
./gradlew assembleRelease publishToMavenLocal
```

---

## iOS: CocoaPods Podspec & SPM

### 1) Podspec
```ruby
# sdk/ios/UniversalTranslationSDK/UniversalTranslationSDK.podspec
Pod::Spec.new do |s|
  s.name             = 'UniversalTranslationSDK'
  s.version          = '1.0.0'
  s.summary          = 'iOS encoder client for Universal Translation System'
  s.license          = { :type => 'Apache-2.0', :file => 'LICENSE' }
  s.homepage         = 'https://github.com/code-with-zeeshan/universal-translation-system'
  s.author           = { 'Code with Zeeshan' => 'dev@code-with-zeeshan.com' }
  s.source           = { :git => 'https://github.com/code-with-zeeshan/universal-translation-system.git', :tag => s.version }
  s.platform         = :ios, '13.0'
  s.swift_version    = '5.7'
  s.source_files     = 'Sources/**/*.{swift,h,mm}'
  s.public_header_files = 'Sources/**/*.h'
end
```

---

## Web SDK: npm Publishing

```bash
cd sdk/web/universal-translation-sdk
npm install
npm run build
npm publish --access public
```

---

## React Native: Linking Notes
- Autolinking works for RN 0.60+.
- iOS: `cd ios && pod install` after adding the RN package.

---

## Versioning & CI
- Semantic versioning (`MAJOR.MINOR.PATCH`).
- Automate publishing with GitHub Actions (`.github/workflows/sdk-publish.yml`, `web-npm-publish.yml`).
- Naming convention:
  - Web: `@your-org/universal-translation-sdk`
  - React Native: `@your-org/universal-translation-sdk-rn`
  - Android (Maven): `com.yourorg:universal-translation-sdk`
  - iOS: `UniversalTranslationSDK`
