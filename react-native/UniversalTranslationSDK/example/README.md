# UniversalTranslationSDK Example (React Native)

Quick steps to run the example app using the local SDK in this monorepo.

## Prerequisites
- Node.js 18+
- Yarn or npm
- Watchman (macOS)
- Xcode (iOS) or Android Studio (Android)
- CocoaPods (`sudo gem install cocoapods`) for iOS

## 1. Install deps
```bash
# From repo root or inside the example folder
cd react-native/UniversalTranslationSDK/example
npm install  # or yarn install
```

## 2. Link local SDK (monorepo)
No publish needed — the example uses the local package via workspace/relative path.
If you need to rebuild TS/Native code of the SDK, run inside SDK root:
```bash
cd ../
npm install
npm run prepare
```

## 3. iOS
```bash
cd ios
pod install
cd ..
# Start Metro
npm run start
# Run app
npm run ios
```

## 4. Android
```bash
# Start Metro
npm run start
# Run app
npm run android
```

## 5. Environment
Create a `.env` in the example folder if required by your setup:
```
COORDINATOR_URL=https://coord.example.com
API_KEY=...
```

## 6. Troubleshooting
- iOS build errors: run `cd ios && pod repo update && pod install`.
- Metro cache: `npm start -- --reset-cache`.
- Android Gradle clean: `cd android && ./gradlew clean`.
- Ensure the SDK TS build output is present if the example imports built artifacts.