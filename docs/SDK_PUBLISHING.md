# SDK Publishing and Linking Guide

This guide covers publishing the Android and iOS SDKs, and linking steps for React Native.

---

## Android: Gradle (Maven Publishing)

### 1) Add Maven Publishing to your SDK module
```gradle
// sdk-module/build.gradle
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
        licenses {
          license {
            name = 'Apache-2.0'
            url = 'https://www.apache.org/licenses/LICENSE-2.0'
          }
        }
      }
    }
  }
  repositories {
    mavenLocal() // for local testing
    // maven {
    //   url = uri("https://maven.your-org.com/releases")
    //   credentials { username = findProperty('mavenUser'); password = findProperty('mavenPass') }
    // }
  }
}
```

### 2) Build & publish locally
```bash
./gradlew :sdk-module:assembleRelease :sdk-module:publishToMavenLocal
```

### 3) Consume from an app
```gradle
repositories {
  mavenLocal()
  mavenCentral()
}

dependencies {
  implementation 'com.example:universal-translation-sdk:1.0.0'
}
```

---

## iOS: CocoaPods Podspec & SPM

### 1) Podspec template
```ruby
Pod::Spec.new do |s|
  s.name             = 'UniversalTranslationSDK'
  s.version          = '1.0.0'
  s.summary          = 'iOS encoder client for Universal Translation System'
  s.license          = { :type => 'Apache-2.0', :file => 'LICENSE' }
  s.homepage         = 'https://github.com/your-org/universal-translation-system'
  s.author           = { 'Your Org' => 'dev@your-org.com' }
  s.source           = { :git => 'https://github.com/your-org/universal-translation-system.git', :tag => s.version }
  s.platform         = :ios, '13.0'
  s.swift_version    = '5.7'
  s.source_files     = 'ios/UniversalTranslationSDK/Sources/**/*.{swift,h,mm}'
  s.public_header_files = 'ios/UniversalTranslationSDK/Sources/**/*.h'
  s.requires_arc     = true
  s.frameworks       = 'Foundation'

  # If linking C++ encoder core
  # s.vendored_libraries = 'ios/UniversalTranslationSDK/Libraries/libuniversal_encoder_core.a'
end
```

### 2) Publish
- Push to a spec repo (internal or CocoaPods trunk)
- Or consume via `:path` during development:
```ruby
pod 'UniversalTranslationSDK', :path => '../ios/UniversalTranslationSDK'
```

### 3) Swift Package Manager
- Tag Git repo (e.g., `1.0.0`) and add it in Xcode > Package Dependencies.

---

## Web SDK: npm Publishing

### Package scope and name
- Prefer a scoped name for organization packages: `@your-org/universal-translation-sdk`
- Ensure `name`, `version`, `license`, `repository`, and `homepage` are set in `web/universal-translation-sdk/package.json`

### Build & prepare
```bash
cd web/universal-translation-sdk
npm install
npm run build:wasm   # builds wasm if applicable
npm run build        # builds dist/
```

### Pre-publish checks
- `.npmignore` or `files` field includes only necessary artifacts (dist/, README, LICENSE, package.json, typings)
- `main/module/types` fields point to built outputs (e.g., dist/index.cjs, dist/index.esm.js, dist/index.d.ts)
- Version bumped in package.json

### Login & publish
```bash
npm login
npm publish --access public  # for scoped public packages
```

### Tagging & dist-tags
```bash
npm version patch            # or minor/major
npm publish --tag next       # for prereleases
```

### Consuming
```bash
npm install @your-org/universal-translation-sdk
```

## React Native: Linking Notes

- Autolinking works for RN 0.60+ when the library is correctly configured.
- iOS: run `cd ios && pod install` after adding the RN package.
- Android: confirm your root `settings.gradle` includes the RN library if not autolinked.

### Manual linking fallback
```bash
npx react-native config # inspect config
```
- If using a native encoder module, ensure:
  - Android: `CMakeLists.txt`/`Android.mk` and `.so`/AAR are included.
  - iOS: Podspec exposes headers and any vendored libs.

---

## Versioning & CI
- Use semantic versioning (`MAJOR.MINOR.PATCH`).
- Automate publishing in CI with protected tags and credentials in secrets.
- Sign artifacts if required by your repository.

## Naming & Scope Notes
- Keep a consistent naming scheme across platforms:
  - Web: `@your-org/universal-translation-sdk`
  - React Native: `@your-org/universal-translation-sdk-rn`
  - Android (Maven): `com.yourorg:universal-translation-sdk`
  - iOS (CocoaPods/SPM): `UniversalTranslationSDK`
- Avoid breaking changes without major version bumps.
- Reserve the same organization scope across registries (npm, Maven, CocoaPods trunk/Specs) to prevent squatters.