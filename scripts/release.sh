#!/bin/bash
# Universal Translation System Release Script

VERSION=$1

if [ -z "$VERSION" ]; then
    echo "Usage: ./release.sh <version>"
    exit 1
fi

echo "🚀 Preparing release $VERSION..."

# Update versions
python scripts/version_manager.py release $VERSION

# Run tests
echo "🧪 Running tests..."
pytest tests/

# Build Android
echo "📱 Building Android SDK..."
cd android/UniversalTranslationSDK
./gradlew clean build
cd ../..

# Validate iOS
echo "📱 Validating iOS SDK..."
cd ios/UniversalTranslationSDK
swift build
cd ../..

# Update changelog
echo "📝 Don't forget to update CHANGELOG.md!"

# Create tag
echo "🏷️  Creating git tag..."
git add -A
git commit -m "chore: release v$VERSION"
git tag -a "v$VERSION" -m "Release version $VERSION"

echo "✅ Release prepared!"
echo "📤 Run 'git push && git push --tags' to publish"