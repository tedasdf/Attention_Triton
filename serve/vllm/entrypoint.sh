#!/bin/bash
set -e

vllm serve "$VLLM_MODEL" \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype "${VLLM_DTYPE:-auto}" \
  --max-model-len "${VLLM_MAX_MODEL_LEN:-2048}"