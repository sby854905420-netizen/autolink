import json
import os
import argparse
import glob

from cost_tool import SampleCostRecorder
from progress_logger import get_progress_logger, progress_bar, setup_progress_logging
from utils import DEFAULT_DATASET_NAME, determine_embedding_path, is_dataset_instance, load_dataset_data


def fill_rule(initial_candidates, dataset_name: str = DEFAULT_DATASET_NAME):
    spider2_data = load_dataset_data(dataset_name)

    final_schemas =  {}
    
    for instance_id, schema_info in initial_candidates.items():
        final_schemas[instance_id] = {}
        
        db_name = spider2_data[instance_id]["db_name"]
        
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
        
    return final_schemas

def add_pre_rule(log_path, dataset_name: str = DEFAULT_DATASET_NAME):
    setup_progress_logging(log_path)
    logger = get_progress_logger(__name__)
    cache_path = os.path.join(log_path, "cache")
    status_path = os.path.join(log_path, "status")
    cost_output_path = os.path.join(log_path, "cost.json")
    spider2_data = load_dataset_data(dataset_name)
    
    with open(f"{log_path}/initial_candidates.json", "r", encoding="utf-8") as f:
        initial_candidates = json.load(f)
        
    logger.info(
        "Adding id/name/code columns started | dataset=%s samples=%s",
        dataset_name,
        len(initial_candidates),
    )
    add_id_candidates = {}
    
    for instance_id, schema_info in progress_bar(
        initial_candidates.items(),
        desc="add id/name/code",
        total=len(initial_candidates),
        unit="sample",
    ):
        with SampleCostRecorder(
            sample_id=instance_id,
            output_path=cost_output_path,
        ):
            db_name = schema_info["db_name"]
            question = schema_info["question"]
            table_candidates = schema_info["table_candidates"]
            column_candidates = schema_info["column_candidates"]
            add_id_candidates[instance_id] = {
                "question":question,
                "db_name": db_name,
                "table_candidates": table_candidates,
                "column_candidates": column_candidates,
                "column_types": schema_info["column_types"].copy(),
                "column_values": schema_info["column_values"].copy(),
                "descriptions": schema_info["descriptions"].copy(),
            }
            
            embedding_path = determine_embedding_path(instance_id, dataset_name)
            
            with open(f"{status_path}/{instance_id}.json", "r", encoding="utf-8") as f:
                status = json.load(f)
            
            with open(f"{cache_path}/{instance_id}.json", "r", encoding="utf-8") as f:
                cache = json.load(f)
            
            if status["is_complete"]:
                continue
                
            seen_tables = []
            cache_updated = False
            status_updated = False
            
            for table in table_candidates:
                if table not in seen_tables:
                    seen_tables.append(table)
                else:
                    continue
                    
                with open(f"{embedding_path}/{db_name}/metadata.json", "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                    
                for index, all_columns in enumerate(metadata):
                    if all_columns["table"] != table:
                        continue
                    
                    column =  all_columns["column"]
                    column_type = all_columns["column_type"]
                    column_value = all_columns["column_value"]
                    description = all_columns["description"]
                    
                    if ("id" in column.lower() or "name" in column.lower() or "code" in column.lower()) and index not in cache["used_indices"]:
                        add_id_candidates[instance_id]["table_candidates"].append(table)
                        add_id_candidates[instance_id]["column_candidates"].append(column)
                        add_id_candidates[instance_id]["column_types"].append(column_type)
                        add_id_candidates[instance_id]["column_values"].append(column_value)
                        add_id_candidates[instance_id]["descriptions"].append(description)
                        
                        cache["used_indices"].append(index)
                        cache_updated = True
                        status["used_count"] += 1 
                        status_updated = True

                        if status["used_count"] >= status["total_available"]:
                            status["is_complete"] = True
                            break
                
                if status["is_complete"]:
                    break
            
            if status_updated:
                status["remaining_count"] = status["total_available"] - status["used_count"]
            
            if status_updated: 
                with open(f"{status_path}/{instance_id}.json", "w", encoding="utf-8") as f:
                    json.dump(status, f, ensure_ascii=False, indent=4)
            
            if cache_updated: 
                with open(f"{cache_path}/{instance_id}.json", "w", encoding="utf-8") as f:
                    json.dump(cache, f, ensure_ascii=False, indent=4)
                
    with open(f"{log_path}/unfilled_pre_rule.json", "w", encoding="utf-8") as f:
        json.dump(add_id_candidates, f, ensure_ascii=False, indent=4)
    
    filled_pre_rule = fill_rule(add_id_candidates, dataset_name)
    
    with open(f"{log_path}/filled_pre_rule.json", "w", encoding="utf-8") as f:
        json.dump(filled_pre_rule, f, ensure_ascii=False, indent=4)
    logger.info(
        "Adding id/name/code columns finished | dataset=%s samples=%s output=%s",
        dataset_name,
        len(add_id_candidates),
        os.path.join(log_path, "unfilled_pre_rule.json"),
    )
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_path', type=str, default="log_mmqa_global_topn100")
    parser.add_argument('--dataset_name', type=str, default=DEFAULT_DATASET_NAME)
    args = parser.parse_args()

    add_pre_rule(args.log_path, args.dataset_name)
