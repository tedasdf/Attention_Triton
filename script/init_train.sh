#!/bin/bash

# Configuration
IMAGE_NAME="ntp-train"
DOCKERFILE_PATH=".devcontainer/Dockerfile"
STORAGE_PATH="/home/$(whoami)/ai_storage"

# --- NEW: Check for API Key ---
# If a command line argument is provided, use it. Otherwise, use the env var.
WANDB_KEY=${1:-$WANDB_API_KEY}

if [ -z "$WANDB_KEY" ]; then
    echo "⚠️ Warning: No WANDB_API_KEY provided. Training might fail or run offline."
fi

echo "🔍 Step 1: Checking Docker Environment..."

# 1. Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Error: Docker is not running."
    exit 1
fi

# 2. Build the image (Removing the 'if exists' check)
# It is better to run 'build' every time. If nothing changed, 
# Docker's layer caching makes this take 0.5 seconds anyway.
echo "🏗️  Ensuring image '$IMAGE_NAME' is up to date..."
docker build -t $IMAGE_NAME -f $DOCKERFILE_PATH .

echo "🧪 Step 2: Validating GPU & Integrity..."

# Check GPU connectivity inside the container
if docker run --rm --gpus all $IMAGE_NAME nvidia-smi > /dev/null 2>&1; then
    echo "✅ GPU Passthrough OK."
else
    echo "❌ Error: Docker cannot see your 3090. Check nvidia-container-toolkit."
    exit 1
fi

echo "🚀 Step 3: Launching Training..."

# 4. Execute with Environment Variables and Mounts
# We use -e to pass the key you have in your local terminal session
docker run --rm --gpus all \
    -e WANDB_API_KEY="$WANDB_KEY" \
    -v "$(pwd):/app" \
    -v "$STORAGE_PATH:/storage" \
    $IMAGE_NAME \
    python main/train.py \
    --data_dir /storage/datasets \
    --output_dir /storage/checkpoints

# 5. Cleanup dangling images to save space on your 3090
docker image prune -f