#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Experiment controls
DATASET_NAME="${DATASET_NAME:-Spider2}"
BATCH_SIZE="${EMBED_BATCH_SIZE:-512}"
SENTENCE_TRANSFORMER_MODEL="${SENTENCE_TRANSFORMER_MODEL:-BAAI/bge-large-en-v1.5}"

export SENTENCE_TRANSFORMER_MODEL

python "$PROJECT_ROOT/Run_local/build_embeddings.py" \
    --dataset_name "$DATASET_NAME" \
    --batch_size "$BATCH_SIZE"
