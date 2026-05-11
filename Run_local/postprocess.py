import os
import json
from tqdm import tqdm
import argparse
import glob

from cost_tool import SampleCostRecorder
from progress_logger import get_progress_logger, progress_bar, setup_progress_logging
from utils import DEFAULT_DATASET_NAME, is_dataset_instance

def extract_description(description_text):
    lines = description_text.strip().split("\n")
    
    for line in lines:
        if line.startswith("description:"):
            return line[len("description:"):].strip()

    return ""                
        
def merge(log_path, is_preprocess=False, dataset_name: str = DEFAULT_DATASET_NAME):
    setup_progress_logging(log_path)
    logger = get_progress_logger(__name__)
    cost_output_path = os.path.join(log_path, "cost.json")
    if is_preprocess:
        with open(f"{log_path}/unfilled_pre_rule.json", "r", encoding="utf-8") as f:
            initial_candidates = json.load(f)
    else:
        with open(f"{log_path}/initial_candidates.json", "r", encoding="utf-8") as f:
            initial_candidates = json.load(f)
        
    logger.info(
        "Merging candidates started | dataset=%s samples=%s is_preprocess=%s",
        dataset_name,
        len(initial_candidates),
        is_preprocess,
    )

    for instance_id, schema_info in progress_bar(
        initial_candidates.items(),
        desc="merge step2 candidates",
        total=len(initial_candidates),
        unit="sample",
    ):
        if os.path.exists(f"{log_path}/candidates/{instance_id}.json"):
            with open(f"{log_path}/candidates/{instance_id}.json", "r", encoding="utf-8") as f:
                step2_candidates = json.load(f)
                
            step2_schema = step2_candidates[instance_id]
            schema_info["column_candidates"] += step2_schema["column_candidates"]
            schema_info["table_candidates"] += step2_schema["table_candidates"]
            schema_info["column_types"] += step2_schema["column_types"]
            schema_info["column_values"] += step2_schema["column_values"]
            schema_info["descriptions"] += step2_schema["descriptions"]

    with open(f"{log_path}/unfilled_schema.json", "w", encoding="utf-8") as f:
        json.dump(initial_candidates, f, indent=4, ensure_ascii=False)
    
    with open(f"{log_path}/unfilled_schema.json", "r", encoding="utf-8") as f:
        initial_candidates = json.load(f)
        
    final_schemas =  {}
        
    for instance_id, schema_info in progress_bar(
        initial_candidates.items(),
        desc="fill merged candidates",
        total=len(initial_candidates),
        unit="sample",
    ):
        with SampleCostRecorder(
            sample_id=instance_id,
            output_path=cost_output_path,
        ):
            final_schemas[instance_id] = {}
            
            db_name = schema_info["db_name"]
            
            filled_table_candidates = []
            filled_column_candidates = []
            
            table_candidates = schema_info["table_candidates"]
            column_candidates = schema_info["column_candidates"]
            
            if instance_id.startswith("bq") or instance_id.startswith("ga"):

                db_path = f"resource/databases/bigquery/{db_name}"
                json_files = glob.glob(os.path.join(db_path, "**", "*.json"), recursive=True)
                
                for table_candidate, column_candidate in zip(table_candidates, column_candidates):
                    table_name = table_candidate.split(".")[-1]
                    found_path = None
                    for json_file in json_files:
                        if table_name.strip().lower() in json_file.strip().lower():
                            found_path = json_file
                            break
                    if not found_path:
                        print(f"Warning: Instance_id {instance_id}, Table {table_name} not found in BigQuery database {db_name}.")
                        continue
                    with open(found_path, "r", encoding="utf-8") as f:
                        table_info = json.load(f)
                    nested_column_names = table_info["nested_column_names"]
                    
                    is_nested = False
                    for nested_column in nested_column_names:
                        if "." in nested_column:
                            is_nested = True
                            break
                    if not is_nested:
                        filled_column_candidates.append(column_candidate)
                        filled_table_candidates.append(table_candidate)
                    else:
                        for nested_column in nested_column_names:
                            if "." in nested_column:
                                first_parts = nested_column.split(".")[0]
                                if first_parts.lower() == column_candidate.lower():
                                    filled_column_candidates.append(nested_column)
                                    filled_table_candidates.append(table_candidate)
                            elif nested_column == column_candidate:
                                filled_column_candidates.append(nested_column)
                                filled_table_candidates.append(table_candidate)
                                
                                
            elif instance_id.startswith("sf"):
                for table_candidate, column_candidate in zip(table_candidates, column_candidates):
                    filled_column_candidates.append(column_candidate)
                    filled_table_candidates.append(table_candidate)
                
            elif instance_id.startswith("local") or is_dataset_instance(instance_id, dataset_name):
                for table_candidate, column_candidate in zip(table_candidates, column_candidates):
                    filled_column_candidates.append(column_candidate)
                    filled_table_candidates.append(table_candidate)
                
            final_schemas[instance_id]["table_candidates"] = filled_table_candidates
            final_schemas[instance_id]["column_candidates"] = filled_column_candidates
        
    with open(f"{log_path}/merge_candidates.json", "w") as f:
        json.dump(final_schemas, f, indent=4, ensure_ascii=False)
    logger.info(
        "Merging candidates finished | dataset=%s samples=%s output=%s",
        dataset_name,
        len(final_schemas),
        os.path.join(log_path, "merge_candidates.json"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_path', type=str, default="log_mmqa_global_topn100")
    parser.add_argument('--dataset_name', type=str, default=DEFAULT_DATASET_NAME)
    args = parser.parse_args()
    logger = get_progress_logger(__name__)
    logger.info("Merging candidate schemas...")
    merge(log_path=args.log_path, is_preprocess=True, dataset_name=args.dataset_name)
    logger.info("Merging completed.")
