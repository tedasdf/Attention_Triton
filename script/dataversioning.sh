#!/bin/bash

# 1. Check if a filename was provided
if [ -z "$1" ]; then
    echo "Usage: ./dvc_push_data.sh <path_to_file>"
    exit 1
fi

FILE_PATH=$1

# 2. Check if the file actually exists
if [ ! -f "$FILE_PATH" ]; then
    echo "Error: File $FILE_PATH not found!"
    exit 1
fi

echo "🚀 Processing $FILE_PATH..."

# 3. Add to DVC
dvc add "$FILE_PATH"

# 4. Push to S3
echo "☁️ Pushing to S3 (Sydney)..."
dvc push

# 5. Git Automation
# We extract just the .dvc filename (e.g., data/corpus.txt.dvc)
DVC_FILE="$FILE_PATH.dvc"

echo "📝 Committing metadata to Git..."
git add "$DVC_FILE" .gitignore
git commit -m "MLOps: Updated data version for $FILE_PATH"

# --- ADD THE TAGGING LOGIC HERE ---
DATA_VERSION="v$(date +%Y%m%d%H%M)"
git tag -a "$DATA_VERSION" -m "Data version for $FILE_PATH"
echo "✅ Tagged commit as $DATA_VERSION"
# ----------------------------------

echo "🚀 Push these changes to GitHub to trigger your pipeline!"