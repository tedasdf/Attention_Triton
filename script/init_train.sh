#!/bin/bash

# Configuration
IMAGE_NAME="ntp-train"
DOCKERFILE_PATH=".devcontainer/Dockerfile"

echo "🔍 Step 1: Checking Docker Environment..."

# 1. Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Error: Docker is not running. Please start Docker Desktop."
    exit 1
fi

# 2. Check if the image exists, if not, build it
if [[ "$(docker images -q $IMAGE_NAME 2> /dev/null)" == "" ]]; then
    echo "🏗️  Image '$IMAGE_NAME' not found. Building now..."
    docker build -t $IMAGE_NAME -f $DOCKERFILE_PATH .
else
    echo "✅ Image '$IMAGE_NAME' already exists."
fi

echo "🧪 Step 2: Validating Container Integrity..."

# 3. Check if the container can run a simple import check
# This ensures dependencies (torch, transformers) are actually installed
if docker run --rm $IMAGE_NAME python -c "import torch; print(f'Torch {torch.__version__} OK')" > /dev/null 2>&1; then
    echo "✅ Dependency check passed."
else
    echo "❌ Error: Container internal check failed. Rebuilding..."
    docker build --no-cache -t $IMAGE_NAME -f $DOCKERFILE_PATH .
fi

echo "🚀 Step 3: Launching Smoke Test..."

# 4. Execute the Smoke Test with Bind Mount
# Using $(pwd) for compatibility with Bash-on-Windows
docker run --rm \
    -v "$(pwd):/app" \
    $IMAGE_NAME \
    python main/train.py --smoke-test