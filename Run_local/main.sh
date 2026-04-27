#!/usr/bin/env bash
set -e

# You can switch the local Ollama model and dataset here.
OLLAMA_MODEL="${OLLAMA_MODEL:-ministral-3:14b}"
DATASET_NAME="${1:-${DATASET_NAME:-mmqa}}"

export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434}"
export OLLAMA_MODEL

LOG_PATH="${LOG_PATH:-log_${DATASET_NAME}_global_topn100}"
TOP_N="${TOP_N:-100}"
NUM_THREADS="${NUM_THREADS:-1}"

rm -f "$LOG_PATH/cost.json" "$LOG_PATH/cost.json.lock"

python retrieve_topk_schema.py --log_path "$LOG_PATH" --top_n "$TOP_N" --dataset_name "$DATASET_NAME"
python add_id.py --log_path "$LOG_PATH" --dataset_name "$DATASET_NAME"
python generate_schema.py --log_path "$LOG_PATH" --is_initial --dataset_name "$DATASET_NAME"
python complete_schema.py --log_path "$LOG_PATH" --dataset_name "$DATASET_NAME" --num_threads "$NUM_THREADS"
python postprocess.py --log_path "$LOG_PATH" --dataset_name "$DATASET_NAME"
python generate_schema.py --log_path "$LOG_PATH" --dataset_name "$DATASET_NAME"
