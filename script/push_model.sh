#!/bin/bash

# 2. Check if the metadata file was created (only happens if it was a 'Better' model)
if [ -f "run_metadata.env" ]; then
    echo "🏆 Better model detected. Syncing to S3..."
    
    # Load the variables (WANDB_RUN_NAME, MLFLOW_RUN_ID, VAL_LOSS)
    source run_metadata.env
    
    # 3. DVC sync
    dvc add model.pt
    dvc push
    
    # 4. Git Tagging with BOTH IDs for maximum traceability
    TAG_NAME="registry/${WANDB_RUN_NAME}/${MLFLOW_RUN_ID}"
    
    git tag -a "$TAG_NAME" -m "Performance Loss: $VAL_LOSS"
    git push origin --tags
    
    # 5. Push the .dvc pointer so Git knows which hash to use
    git add model.pt.dvc run_metadata.env
    git commit -m "chore: update champion model to $WANDB_RUN_NAME [skip ci]"
    git push
else
    echo "stats: No improvement or smoke test. Skipping registry update."
fi