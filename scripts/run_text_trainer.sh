#!/bin/bash
set -e
python3 /workspace/scripts/text_trainer.py "$@"

# Upload to HuggingFace if credentials are provided
if [[ -n "$HUGGINGFACE_TOKEN" && -n "$HUGGINGFACE_USERNAME" ]]; then
    echo "Uploading model to HuggingFace..."
    python3 /workspace/trainer/utils/hf_upload.py
else
    echo "Skipping HuggingFace upload (HUGGINGFACE_TOKEN or HUGGINGFACE_USERNAME not set)"
fi