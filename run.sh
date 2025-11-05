#!/usr/bin/env bash

# Usage: ./run_model.sh <DEVICE_ID> <MODEL_NAME>
# Example: ./run_model.sh 15 llama-3

# Capture command-line arguments
DEVICE_ID=$1
MODEL_NAME=$2

# You can add optional checks to ensure both arguments are provided
if [ -z "$DEVICE_ID" ] || [ -z "$MODEL_NAME" ]; then
  echo "Usage: $0 <DEVICE_ID> <MODEL_NAME>"
  exit 1
fi

# Now set the environment variable and run your Python script
CUDA_VISIBLE_DEVICES="${DEVICE_ID}" python main_copy_replaced_words.py \
    -d gsm_8k \
    -s test \
    -m "${MODEL_NAME}" \
    -o /culture-evaluation/output/new_prompt_english/
