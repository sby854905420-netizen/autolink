#!/usr/bin/env bash
set -e

# You can switch the dataset name here.
DATASET_NAME="${1:-${DATASET_NAME:-mmqa}}"

case "$DATASET_NAME" in
    mmqa|mmqa_smoke)
        python generate_docs.py
        python embedding_docs.py
        ;;
    *)
        echo "Unsupported dataset for local embedding build: $DATASET_NAME" >&2
        exit 1
        ;;
esac
