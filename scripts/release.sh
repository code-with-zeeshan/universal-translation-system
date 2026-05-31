#!/bin/bash
# Universal Translation System Release Script
set -euo pipefail

VERSION=${1:-}

if [ -z "$VERSION" ]; then
    echo "Usage: ./release.sh <version>"
    exit 1
fi

echo "🚀 Preparing release $VERSION..."

# Update versions
python scripts/version_manager.py release "$VERSION"

# Run tests
echo "🧪 Running tests..."
pytest tests/

# Build Android (if gradlew exists)
if [ -d "sdk/android/UniversalTranslationSDK" ] && [ -f "sdk/android/UniversalTranslationSDK/gradlew" ]; then
  echo "📱 Building Android SDK..."
  pushd sdk/android/UniversalTranslationSDK >/dev/null
  ./gradlew clean build
  popd >/dev/null
else
  echo "ℹ️ Android gradle wrapper not found; skipping Android build."
fi

# Validate iOS (if swift available)
if command -v swift >/dev/null 2>&1; then
  if [ -d "sdk/ios/UniversalTranslationSDK" ]; then
    echo "📱 Validating iOS SDK..."
    pushd sdk/ios/UniversalTranslationSDK >/dev/null
    swift build
    popd >/dev/null
  fi
else
  echo "⚠️ swift not available; skipping iOS validation."
fi

# Update changelog reminder
echo "📝 Don't forget to update CHANGELOG.md!"

# Create tag
echo "🏷️  Creating git tag..."
git add -A
if ! git diff --cached --quiet; then
  git commit -m "chore: release v$VERSION"
else
  echo "ℹ️ No changes to commit."
fi
git tag -a "v$VERSION" -m "Release version $VERSION"

echo "✅ Release prepared!"
echo "📤 Run 'git push && git push --tags' to publish"