#!/bin/bash

# Configuration
IMAGE_NAME="ntp-train"
DOCKERFILE_PATH=".devcontainer/Dockerfile"
# Ensure the script knows where your storage is
STORAGE_PATH="/home/$(whoami)/ai_storage"

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
    -e WANDB_API_KEY=$WANDB_API_KEY \
    -v "$(pwd):/app" \
    -v "$STORAGE_PATH:/storage" \
    $IMAGE_NAME \
    python main/train.py \
    --smoke-test \
    --data-dir /storage/datasets \
    --output-dir /storage/checkpoints

# 5. Cleanup dangling images to save space on your 3090
docker image prune -f