#!/usr/bin/env bash
set -euo pipefail



REMOTE="mys3remote"
INPUT_PATH="/home/fypits25/Documents/ted_backyard/actions-runner/_work/Attention_Triton/Attention_Triton/dataset/processed_datasets/unified_python"
OUTPUT_PATH="/home/fypits25/Documents/ted_backyard/actions-runner/_work/Attention_Triton/Attention_Triton/dataset/preprocessed_datasets_v1"

echo "📥 DVC pull: $INPUT_PATH (remote: $REMOTE)"
dvc pull "$INPUT_PATH" -r "$REMOTE"

echo "⚙️  Running pipeline..."
python dataset_pipeline/pipeline.py --config "/home/fypits25/Documents/ted_backyard/actions-runner/_work/Attention_Triton/Attention_Triton/dataset_pipeline/config/pipeline.yaml" --input_dir "$INPUT_PATH" --output_dir "$OUTPUT_PATH"

echo "➕ Tracking output with DVC: $OUTPUT_PATH"
dvc add "$OUTPUT_PATH"

echo "☁️ Pushing data to S3 via DVC (remote: $REMOTE)"
dvc push -r "$REMOTE"

echo "📝 Committing DVC metadata to Git..."
git add *.dvc .gitignore dvc.yaml dvc.lock

# Only commit if there are staged changes
if git diff --cached --quiet; then
  echo "ℹ️  No DVC metadata changes to commit."
else
  git commit -m "Data: update processed snapshot ($OUTPUT_PATH)"
  git push
fi

echo "✅ Done."