#!/bin/bash

# 1. Break the loop: If this commit was made by the runner, stop.
if [[ $(git log -1 --pretty=%B) == *"[skip ci]"* ]]; then
    echo "This was a bot commit. Stopping to prevent infinite loops."
    exit 0
fi

# 2. Check if the metadata file exists
if [ -f "run_metadata.env" ]; then
    echo "🏆 Better model detected. Syncing to S3..."
    
    source run_metadata.env
    
    # 3. DVC sync (Assumes your S3 remote is already configured on the 3090)
    dvc add model.pt
    dvc push
    
    # 4. Configure Git Identity (Local to this run)
    git config user.name "3090-MLOps-Bot"
    git config user.email "bot@your-domain.com"
    
    # 5. Git Tagging
    TAG_NAME="registry/${WANDB_RUN_NAME}/${MLFLOW_RUN_ID}"
    git tag -a "$TAG_NAME" -m "Performance Loss: $VAL_LOSS"
    git push origin "$TAG_NAME"
    
    # 6. Push the .dvc pointer with [skip ci] to prevent loops
    git add model.pt.dvc run_metadata.env
    git commit -m "chore: update champion model to $WANDB_RUN_NAME [skip ci]"
    git push origin main
else
    echo "stats: No improvement or smoke test. Skipping registry update."
fi