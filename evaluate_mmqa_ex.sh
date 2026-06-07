#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="${ROOT_DIR}/Run_local/evaluate_ex.py"
OUTPUT_DIR="${ROOT_DIR}/Evaluation"

if [[ ! -f "${EVAL_SCRIPT}" ]]; then
  echo "EX evaluator not found: ${EVAL_SCRIPT}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

mapfile -d '' SQL_RESULTS < <(
  find "${ROOT_DIR}/Log/MMQA" \
    -path "*/sql_results/sql_generation_MMQA_*.json" \
    -type f \
    ! -name "*_ex.json" \
    -print0 | sort -z
)

if [[ "${#SQL_RESULTS[@]}" -eq 0 ]]; then
  echo "No MMQA sql_generation result files found under ${ROOT_DIR}/Log/MMQA" >&2
  exit 1
fi

echo "Found ${#SQL_RESULTS[@]} MMQA SQL result file(s)."

for sql_result in "${SQL_RESULTS[@]}"; do
  rel_path="${sql_result#"${ROOT_DIR}/Log/MMQA/"}"
  model_name="${rel_path%%/*}"
  rest="${rel_path#*/}"
  run_id="${rest%%/*}"
  sql_basename="$(basename "${sql_result}" .json)"
  output_path="${OUTPUT_DIR}/MMQA_${model_name}_${run_id}_${sql_basename}_ex.json"
  echo "Evaluating EX: ${sql_result}"
  python3 "${EVAL_SCRIPT}" \
    --dataset_name MMQA \
    --sql_results "${sql_result}" \
    --output_path "${output_path}"
done

echo "MMQA EX evaluation finished."
