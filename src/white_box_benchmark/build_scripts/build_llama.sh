#!/bin/bash

set -e  # Exit on error

REPO_URL="https://github.com/VakantieModus/llama.cpp.git"
BRANCH="extend-eval-callback-for-extracting-specific-layers"
BASE_DIR="src/white_box_benchmark"
CLONE_DIR="$BASE_DIR/llama.cpp"

# Ensure base directory exists
mkdir -p "$BASE_DIR"

if [ ! -d "$CLONE_DIR" ]; then
    echo "Cloning repository into $CLONE_DIR..."
    git clone "$REPO_URL" "$CLONE_DIR"
    cd "$CLONE_DIR"
    git checkout "$BRANCH"
else
    echo "Repository already exists at $CLONE_DIR. Updating..."
    cd "$CLONE_DIR"
    git fetch origin
    git checkout "$BRANCH"
    git pull origin "$BRANCH"
fi

# Clean build directory to avoid mixing CPU/GPU builds
echo "Cleaning previous build..."
rm -rf build
mkdir build
cd build

export PATH=/usr/local/cuda/bin:$PATH
export CUDACXX=/usr/local/cuda/bin/nvcc

echo "Running CMake configuration with cuBLAS (GPU) support..."
cmake .. -DGGML_CUDA=ON -DLLAMA_CURL=OFF

echo "Building the project..."
cmake --build .
