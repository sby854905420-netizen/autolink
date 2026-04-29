#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Experiment controls
DATASET_NAME="MMQA"
OLLAMA_MODEL="ministral-3:14b"
OLLAMA_BASE_URL="http://127.0.0.1:11434"

TOP_N="100"
NUM_THREADS="1"
SENTENCE_TRANSFORMER_MODEL="BAAI/bge-large-en-v1.5"

export OLLAMA_MODEL
export OLLAMA_BASE_URL
export SENTENCE_TRANSFORMER_MODEL

python "$PROJECT_ROOT/Run_local/main.py" \
    --dataset_name "$DATASET_NAME" \
    --ollama_model "$OLLAMA_MODEL" \
    --ollama_base_url "$OLLAMA_BASE_URL" \
    --top_n "$TOP_N" \
    --num_threads "$NUM_THREADS"
