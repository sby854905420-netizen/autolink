mkdir -p Evaluation/ex_results/MMQA

for f in Log/sql_results/MMQA/*/sql_generation_MMQA_*.json; do
  model="$(basename "$(dirname "$f")")"
  out="Evaluation/ex_results/MMQA/${model}_$(basename "${f%.json}")_ex.json"

  python Run_local/evaluate_ex.py \
    --dataset_name MMQA \
    --sql_results "$f" \
    --output_path "$out"
done