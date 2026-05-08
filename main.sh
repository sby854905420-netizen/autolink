#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Experiment controls
DATASET_NAME="Spider2"
HF_MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
TOP_N="100"
SENTENCE_TRANSFORMER_MODEL="BAAI/bge-large-en-v1.5"

export SENTENCE_TRANSFORMER_MODEL

python "$PROJECT_ROOT/Run_local/main.py" \
    --dataset_name "$DATASET_NAME" \
    --hf_model_name "$HF_MODEL_NAME" \
    --top_n "$TOP_N"
