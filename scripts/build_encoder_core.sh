#!/bin/bash
set -e

# scripts/build_encoder_core.sh - Build the native C++ encoder core for all platforms
# Usage: bash scripts/build_encoder_core.sh [--linux] [--macos] [--android] [--ios] [--all]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ENCODER_DIR="$PROJECT_DIR/runtime/encoder_core"
OUTPUT_DIR="$ENCODER_DIR/output"

NPROC=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

detect_platform() {
    case "$(uname -s)" in
        Linux*)  echo "linux" ;;
        Darwin*) echo "macos" ;;
        *)       echo "unknown" ;;
    esac
}

HOST_PLATFORM=$(detect_platform)

check_cmake() {
    if ! command -v cmake &>/dev/null; then
        echo "ERROR: cmake is not installed. Install it (e.g., apt install cmake, brew install cmake)."
        exit 1
    fi
}

build_linux() {
    echo "=== Building for Linux (x86_64) ==="
    check_cmake

    local BUILD_DIR="$ENCODER_DIR/build_linux"
    local TARGET_OUTPUT="$OUTPUT_DIR/linux"

    mkdir -p "$BUILD_DIR" "$TARGET_OUTPUT"

    cmake -S "$ENCODER_DIR" -B "$BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON \
        -DBUILD_TESTS=OFF \
        -DBUILD_EXAMPLES=OFF

    make -C "$BUILD_DIR" -j"$NPROC"

    cp "$BUILD_DIR"/libuniversal_encoder_core.* "$TARGET_OUTPUT/" 2>/dev/null || \
    cp "$BUILD_DIR"/libuniversal_encoder_core* "$TARGET_OUTPUT/" 2>/dev/null || \
    { echo "ERROR: Library not found in build_linux output."; exit 1; }

    echo "OK: Linux libraries -> $TARGET_OUTPUT"
}

build_macos() {
    echo "=== Building for macOS (x86_64 + arm64 universal) ==="
    check_cmake

    local BUILD_DIR_X64="$ENCODER_DIR/build_macos_x86_64"
    local BUILD_DIR_ARM="$ENCODER_DIR/build_macos_arm64"
    local BUILD_DIR_UNIVERSAL="$ENCODER_DIR/build_macos_universal"
    local TARGET_OUTPUT="$OUTPUT_DIR/macos"

    mkdir -p "$BUILD_DIR_X64" "$BUILD_DIR_ARM" "$BUILD_DIR_UNIVERSAL" "$TARGET_OUTPUT"

    echo "--- Building x86_64 slice ---"
    cmake -S "$ENCODER_DIR" -B "$BUILD_DIR_X64" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_OSX_ARCHITECTURES=x86_64 \
        -DBUILD_SHARED_LIBS=ON \
        -DBUILD_TESTS=OFF \
        -DBUILD_EXAMPLES=OFF
    make -C "$BUILD_DIR_X64" -j"$NPROC"

    echo "--- Building arm64 slice ---"
    cmake -S "$ENCODER_DIR" -B "$BUILD_DIR_ARM" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_OSX_ARCHITECTURES=arm64 \
        -DBUILD_SHARED_LIBS=ON \
        -DBUILD_TESTS=OFF \
        -DBUILD_EXAMPLES=OFF
    make -C "$BUILD_DIR_ARM" -j"$NPROC"

    echo "--- Creating universal binary ---"
    local X64_LIB=$(find "$BUILD_DIR_X64" -name "libuniversal_encoder_core*.dylib" | head -1)
    local ARM_LIB=$(find "$BUILD_DIR_ARM" -name "libuniversal_encoder_core*.dylib" | head -1)

    if [ -z "$X64_LIB" ] || [ -z "$ARM_LIB" ]; then
        echo "ERROR: Missing architecture slices."
        exit 1
    fi

    lipo -create "$X64_LIB" "$ARM_LIB" -output "$TARGET_OUTPUT/libuniversal_encoder_core.dylib"
    lipo -info "$TARGET_OUTPUT/libuniversal_encoder_core.dylib"

    echo "OK: macOS universal library -> $TARGET_OUTPUT"
}

build_android() {
    echo "=== Building for Android (arm64-v8a) ==="
    check_cmake

    local NDK_HOME="${ANDROID_NDK_HOME:-${ANDROID_NDK:-$ANDROID_HOME/ndk/$(ls -1 $ANDROID_HOME/ndk 2>/dev/null | sort -V | tail -1)}}"

    if [ ! -d "$NDK_HOME" ]; then
        echo "ERROR: Android NDK not found."
        echo "  Set ANDROID_NDK_HOME, ANDROID_NDK, or ANDROID_HOME."
        echo "  Or install via sdkmanager: sdkmanager 'ndk;25.2.9519653'"
        exit 1
    fi

    local TOOLCHAIN_FILE="$NDK_HOME/build/cmake/android.toolchain.cmake"
    if [ ! -f "$TOOLCHAIN_FILE" ]; then
        echo "ERROR: Android toolchain not found at $TOOLCHAIN_FILE"
        exit 1
    fi

    local BUILD_DIR="$ENCODER_DIR/build_android"
    local TARGET_OUTPUT="$OUTPUT_DIR/android"

    mkdir -p "$BUILD_DIR" "$TARGET_OUTPUT"

    cmake -S "$ENCODER_DIR" -B "$BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN_FILE" \
        -DANDROID_ABI=arm64-v8a \
        -DANDROID_PLATFORM=android-24 \
        -DANDROID=ON \
        -DBUILD_SHARED_LIBS=ON \
        -DBUILD_TESTS=OFF \
        -DBUILD_EXAMPLES=OFF

    make -C "$BUILD_DIR" -j"$NPROC"

    cp "$BUILD_DIR"/libuniversal_encoder_core.* "$TARGET_OUTPUT/" 2>/dev/null || \
    cp "$BUILD_DIR"/libuniversal_encoder_core* "$TARGET_OUTPUT/" 2>/dev/null || \
    { echo "ERROR: Library not found in build_android output."; exit 1; }

    echo "OK: Android libraries -> $TARGET_OUTPUT"
}

build_ios() {
    echo "=== Building for iOS (arm64) ==="
    check_cmake

    local BUILD_DIR="$ENCODER_DIR/build_ios"
    local TARGET_OUTPUT="$OUTPUT_DIR/ios"

    mkdir -p "$BUILD_DIR" "$TARGET_OUTPUT"

    cmake -S "$ENCODER_DIR" -B "$BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_SYSTEM_NAME=iOS \
        -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0 \
        -DIOS=ON \
        -DBUILD_SHARED_LIBS=ON \
        -DBUILD_TESTS=OFF \
        -DBUILD_EXAMPLES=OFF

    make -C "$BUILD_DIR" -j"$NPROC"

    cp "$BUILD_DIR"/libuniversal_encoder_core.* "$TARGET_OUTPUT/" 2>/dev/null || \
    cp "$BUILD_DIR"/libuniversal_encoder_core* "$TARGET_OUTPUT/" 2>/dev/null || \
    { echo "ERROR: Library not found in build_ios output."; exit 1; }

    echo "OK: iOS libraries -> $TARGET_OUTPUT"
}

build_all() {
    build_linux
    if [ "$HOST_PLATFORM" = "macos" ]; then
        build_macos
        build_ios
    fi
    build_android
}

show_help() {
    cat <<EOF
Build the native C++ encoder core for all platforms.

Usage: bash scripts/build_encoder_core.sh [OPTION]

Options:
  --linux          Build for Linux (x86_64)
  --macos          Build for macOS (universal x86_64 + arm64)
  --android        Build for Android (arm64-v8a via NDK)
  --ios            Build for iOS (arm64)
  --all            Build all available platforms
  --help           Show this help message

Environment:
  ANDROID_NDK_HOME   Path to Android NDK (required for --android)
  ANDROID_NDK        Alternative NDK path variable

Output directory: runtime/encoder_core/output/<platform>/
EOF
}

if [ $# -eq 0 ]; then
    show_help
    exit 0
fi

for arg in "$@"; do
    case "$arg" in
        --linux)   build_linux ;;
        --macos)   build_macos ;;
        --android) build_android ;;
        --ios)     build_ios ;;
        --all)     build_all ;;
        --help)    show_help; exit 0 ;;
        *)
            echo "Unknown option: $arg"
            show_help
            exit 1
            ;;
    esac
done
