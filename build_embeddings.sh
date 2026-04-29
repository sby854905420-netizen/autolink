#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Experiment controls
SENTENCE_TRANSFORMER_MODEL="BAAI/bge-large-en-v1.5"

export SENTENCE_TRANSFORMER_MODEL

python "$PROJECT_ROOT/Run_local/build_embeddings.py" \
    --dataset_name MMQA \
    --batch_size 1024
