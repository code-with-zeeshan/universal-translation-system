#!/bin/bash
set -e

# scripts/install.sh - Universal Translation System installation
# Usage: bash scripts/install.sh [--train] [--serve] [--coordinator] [--dev] [--encoder-core] [--all]
#
# Linux (Lightning AI Studio / cloud):   bash scripts/install.sh --train
# macOS:                                  bash scripts/install.sh --serve
# Full installation:                      bash scripts/install.sh --all
#
# Cross-compilation notes:
#   Android: Use NDK r25+ with CMake toolchain file:
#     cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
#           -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-24 ..
#   iOS: Use Xcode with -DCMAKE_SYSTEM_NAME=iOS and -DCMAKE_OSX_SYSROOT=iphoneos

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_DIR/venv"

# Parse args
INSTALL_TRAIN=false
INSTALL_SERVE=false
INSTALL_COORDINATOR=false
INSTALL_DEV=false
INSTALL_ENCODER_CORE=false

for arg in "$@"; do
    case $arg in
        --train) INSTALL_TRAIN=true ;;
        --serve) INSTALL_SERVE=true ;;
        --coordinator) INSTALL_COORDINATOR=true ;;
        --dev) INSTALL_DEV=true ;;
        --encoder-core) INSTALL_ENCODER_CORE=true ;;
        --all) INSTALL_TRAIN=true; INSTALL_SERVE=true; INSTALL_COORDINATOR=true; INSTALL_DEV=true ;;
        --help)
            echo "Usage: bash scripts/install.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --train          Install training dependencies"
            echo "  --serve          Install serving dependencies"
            echo "  --coordinator    Install coordinator dependencies"
            echo "  --dev            Install dev/test dependencies"
            echo "  --encoder-core   Build encoder C++ core with CMake"
            echo "  --all            Install everything"
            echo ""
            echo "Examples:"
            echo "  bash scripts/install.sh --train              # Lightning AI Studio (training only)"
            echo "  bash scripts/install.sh --serve              # Cloud decoder server"
            echo "  bash scripts/install.sh --all                # Full installation"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Usage: bash scripts/install.sh [--train] [--serve] [--coordinator] [--dev] [--encoder-core] [--all]"
            exit 1
            ;;
    esac
done

# If no flags given, install base only
INSTALL_BASE=true

echo "============================================"
echo " Universal Translation System Installer"
echo "============================================"
echo ""

# ---- OS Detection ----
OS="$(uname -s)"
ARCH="$(uname -m)"
echo "OS: $OS $ARCH"

# ---- Python version check ----
echo ""
echo "Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 not found. Install Python 3.9+ first."
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || { [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 9 ]; }; then
    echo "ERROR: Python 3.9+ required, found $PYTHON_VERSION"
    exit 1
fi
echo "  Python $PYTHON_VERSION — OK"

# ---- Virtual environment ----
echo ""
echo "Setting up virtual environment..."
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    echo "  Created venv at $VENV_DIR"
else
    echo "  venv already exists at $VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

pip install --upgrade pip --quiet
echo "  pip upgraded"

# ---- Install requirements ----
echo ""
echo "Installing Python dependencies..."

REQUIREMENTS_DIR="$PROJECT_DIR/requirements"
INSTALLED=()

pip install -r "$REQUIREMENTS_DIR/base.txt" --quiet
INSTALLED+=("base.txt")

if [ "$INSTALL_TRAIN" = true ]; then
    pip install -r "$REQUIREMENTS_DIR/train.txt" --quiet
    INSTALLED+=("train.txt")
fi

if [ "$INSTALL_SERVE" = true ]; then
    pip install -r "$REQUIREMENTS_DIR/serve.txt" --quiet
    INSTALLED+=("serve.txt")
fi

if [ "$INSTALL_COORDINATOR" = true ]; then
    pip install -r "$REQUIREMENTS_DIR/coordinator.txt" --quiet
    INSTALLED+=("coordinator.txt")
fi

if [ "$INSTALL_DEV" = true ]; then
    pip install -r "$REQUIREMENTS_DIR/dev.txt" --quiet
    INSTALLED+=("dev.txt")
    # Install pre-commit hooks
    if [ -f "$PROJECT_DIR/.pre-commit-config.yaml" ]; then
        pre-commit install 2>/dev/null || true
    fi
fi

echo "  Installed: ${INSTALLED[*]}"

# ---- CUDA check ----
echo ""
echo "Checking CUDA availability..."
if command -v nvidia-smi &> /dev/null; then
    DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo "unknown")
    echo "  CUDA detected — driver version: $DRIVER"
    # Check torch CUDA
    python3 -c "import torch; print(f'  torch.cuda.is_available: {torch.cuda.is_available()}')" 2>/dev/null || echo "  torch not yet installed or no CUDA support"
else
    echo "  CUDA not detected (nvidia-smi not found). GPU training will not be available."
fi

# ---- Encoder C++ core ----
if [ "$INSTALL_ENCODER_CORE" = true ]; then
    echo ""
    echo "Building encoder C++ core..."
    ENCODER_DIR="$PROJECT_DIR/runtime/encoder_core"
    BUILD_DIR="$ENCODER_DIR/build"

    if ! command -v cmake &> /dev/null; then
        echo "  ERROR: cmake not found. Install CMake 3.14+ first."
        echo "  Ubuntu/Debian: sudo apt-get install -y cmake build-essential"
        echo "  macOS: brew install cmake"
        exit 1
    fi

    CMAKE_VERSION=$(cmake --version 2>&1 | head -1 | awk '{print $3}')
    echo "  cmake $CMAKE_VERSION — OK"

    # Check for C++ compiler
    if ! command -v g++ &> /dev/null && ! command -v clang++ &> /dev/null; then
        echo "  ERROR: No C++ compiler found (g++ or clang++)."
        echo "  Ubuntu/Debian: sudo apt-get install -y build-essential"
        echo "  macOS: xcode-select --install"
        exit 1
    fi

    mkdir -p "$BUILD_DIR"
    cmake -S "$ENCODER_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
    cmake --build "$BUILD_DIR" --parallel "$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"

    echo "  Encoder C++ core built at $BUILD_DIR"
    INSTALLED+=("encoder-core (C++ build)")
fi

# ---- Summary ----
echo ""
echo "============================================"
echo " Installation complete!"
echo "============================================"
echo ""
echo "  Python:      $PYTHON_VERSION"
echo "  venv:        $VENV_DIR"
echo "  Installed:   ${INSTALLED[*]}"
echo ""

if command -v nvidia-smi &> /dev/null; then
    echo "  GPU:         Available"
else
    echo "  GPU:         Not detected"
fi

echo ""
echo "Activate venv:  source $VENV_DIR/bin/activate"
echo "Run tests:      python3 -m pytest tests/"
echo ""

# Cross-compilation reminders
if [ "$INSTALL_ENCODER_CORE" = true ] && { [ "$OS" = "Linux" ] || [ "$OS" = "Darwin" ]; }; then
    echo "---"
    echo "Cross-compilation for Android/iOS:"
    echo "  Android:"
    echo "    cmake -S encoder_core -B encoder_core/build_android \\"
    echo "      -DCMAKE_TOOLCHAIN_FILE=\$ANDROID_NDK/build/cmake/android.toolchain.cmake \\"
    echo "      -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-24"
    echo "  iOS:"
    echo "    cmake -S encoder_core -B encoder_core/build_ios \\"
    echo "      -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_SYSROOT=iphoneos \\"
    echo "      -DCMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED=NO"
    echo ""
fi
