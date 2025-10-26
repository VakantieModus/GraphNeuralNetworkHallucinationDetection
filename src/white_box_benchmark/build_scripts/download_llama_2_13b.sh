#!/bin/bash

set -e  # Exit on any error

# === CONFIGURABLE PARAMETERS ===
MODEL_REPO="TheBloke/Llama-2-13B-GGUF"
MODEL_FILE="llama-2-13b.Q8_0.gguf"
TARGET_DIR="src/white_box_benchmark/llama_cpp_models"

# === INTERNALS ===
MODEL_NAME=$(basename "$MODEL_REPO")
DEST_PATH="$TARGET_DIR/${MODEL_NAME}_${MODEL_FILE}"
URL="https://huggingface.co/${MODEL_REPO}/resolve/main/${MODEL_FILE}"

# === CREATE TARGET DIRECTORY IF NEEDED ===
mkdir -p "$TARGET_DIR"

# === DOWNLOAD MODEL ===
echo "Downloading model from $URL..."
curl -L "$URL" --output "$DEST_PATH"

echo "Model saved to $DEST_PATH"
