import os
import json
import multiprocessing as mp
from tqdm import tqdm
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from cost_tool import SampleCostRecorder
from progress_logger import get_progress_logger, progress_bar, setup_progress_logging
from utils import *
import argparse

def sliding_window_table_match(metadata_table: str, target_table: str) -> bool:
    metadata_parts = metadata_table.lower().split('.')
    target_parts = target_table.lower().split('.')
    
    if len(target_parts) > len(metadata_parts):
        return False
    
    window_size = len(target_parts)
    
    for i in range(len(metadata_parts) - window_size + 1):
        window = metadata_parts[i:i + window_size]
        if window == target_parts:
            return True
    
    return False

def find_with_name(column_name: str, table_name: str, db_name: str, embed_path: str):
    metadata_path = os.path.join(embed_path, db_name, "metadata.json")
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata_mapping = json.load(f)
    
    is_find = False
    result = []
    
    for idx, metadata in enumerate(metadata_mapping):
        if (metadata["column"].lower() == column_name.lower() and 
            sliding_window_table_match(metadata["table"], table_name)):
            is_find = True
            print("Exact match found:", metadata["column"], metadata["table"])
            result.append({
                "index": int(idx),
                "metadata": metadata
            })
    
    if not is_find:
        for idx, metadata in enumerate(metadata_mapping):
            if (metadata["column"].lower() == column_name.lower() and 
                sliding_window_table_match(mask_digits(metadata["table"]), mask_digits(table_name))):
                is_find = True
                print("Partial match found:", metadata["column"], metadata["table"])
                result.append({
                    "index": int(idx),
                    "metadata": metadata
                })
    
    if not is_find:
        max_count=5
        for idx, metadata in enumerate(metadata_mapping):
            if metadata["column"].lower() == column_name.lower():
                is_find = True
                print("Column match found:", metadata["column"], metadata["table"])
                result.append({
                    "index": int(idx),
                    "metadata": metadata
                })
                max_count -= 1
                if max_count <= 0:
                    break
    
    if not is_find:
        return "No matching column found. Please check the column name and table name."
    else:
        return result

def _retrieve_with_device_filtered(question: str, db_name: str, embed_path: str, 
                                 excluded_indices: set, top_k: int = 5, device: str = "cuda:0"):
    from model_manager import model_manager
    index_path = os.path.join(embed_path, db_name, "index.faiss")
    index = faiss.read_index(index_path)
    metadata_path = os.path.join(embed_path, db_name, "metadata.json")
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata_mapping = json.load(f)
    model_manager.load_model(device=device)
    question_embedding = model_manager.encode(question)
    distances, indices = index.search(question_embedding.reshape(1, -1), len(metadata_mapping))
    filtered_results = []
    
    for i in range(len(indices[0])):
        idx = int(indices[0][i])
        if 0 <= idx < len(metadata_mapping) and idx not in excluded_indices:
            metadata = metadata_mapping[idx]
            filtered_results.append({
                "index": idx,
                "distance": float(distances[0][i]),
                "metadata": metadata
            })
            if len(filtered_results) >= top_k:
                break
    
    return filtered_results, len(metadata_mapping)


def load_instance_cache(instance_id: str, cache_dir: str):
    cache_file = os.path.join(cache_dir, f"{instance_id}.json")
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"used_indices": []}


def save_instance_cache(instance_id: str, cache_dir: str, cache_data: dict):
    os.makedirs(cache_dir, exist_ok=True)
    if "used_indices" in cache_data:
        cache_data["used_indices"] = [int(idx) for idx in cache_data["used_indices"]]
    
    cache_file = os.path.join(cache_dir, f"{instance_id}.json")
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, ensure_ascii=False, indent=2)


def load_instance_status(instance_id: str, status_dir: str):
    status_file = os.path.join(status_dir, f"{instance_id}.json")
    if os.path.exists(status_file):
        with open(status_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "is_complete": False,
        "total_available": 0,
        "used_count": 0,
        "remaining_count": 0
    }


def save_instance_status(instance_id: str, status_dir: str, status_data: dict):
    os.makedirs(status_dir, exist_ok=True)

    cleaned_status = {}
    for key, value in status_data.items():
        if isinstance(value, (np.integer, np.int64, np.int32)):
            cleaned_status[key] = int(value)
        elif isinstance(value, (np.floating, np.float64, np.float32)):
            cleaned_status[key] = float(value)
        else:
            cleaned_status[key] = value
    
    status_file = os.path.join(status_dir, f"{instance_id}.json")
    with open(status_file, "w", encoding="utf-8") as f:
        json.dump(cleaned_status, f, ensure_ascii=False, indent=2)


def get_next_k_results(instance_id: str, question: str, db_name: str, embed_path: str, 
                      top_k: int, cache_dir: str, status_dir: str, device: str):
    cache = load_instance_cache(instance_id, cache_dir)
    status = load_instance_status(instance_id, status_dir)
    
    used_indices = set(cache.get("used_indices", []))
    
    if status.get("is_complete", False):
        print(f"Instance {instance_id} retrieve all completed.")
        return [], {}, "All columns in this databases are retrieved. There is no need to retrieve again."
    
    results, total_available = _retrieve_with_device_filtered(
        question=question,
        db_name=db_name, 
        embed_path=embed_path,
        excluded_indices=used_indices,
        top_k=top_k,
        device=device
    )
    
    if status.get("total_available", 0) == 0:
        status["total_available"] = total_available
    
    new_used_indices = [int(result["index"]) for result in results]
    all_used_indices = list(used_indices) + new_used_indices
    
    cache["used_indices"] = all_used_indices
    save_instance_cache(instance_id, cache_dir, cache)
    
    used_count = len(all_used_indices)
    remaining_count = total_available - used_count
    is_complete = len(results) < top_k or remaining_count <= 0
    
    status = {
        "is_complete": is_complete,
        "total_available": int(total_available), 
        "used_count": int(used_count),
        "remaining_count": int(remaining_count)
    }
    save_instance_status(instance_id, status_dir, status)
    
    metadata_mapping = {}
    for result in results:
        metadata_mapping[result["index"]] = result["metadata"]
    
    if is_complete:
        return results, metadata_mapping, "All columns in this databases are retrieved. There is no need to retrieve again."
    else:
        return results, metadata_mapping, ""


def process_batch_with_device(batch_items, device_id, top_k, log_dir, dataset_name):
    logger = get_progress_logger(__name__)
    logger.info(
        "Retrieval worker started | pid=%s gpu=%s samples=%s top_k=%s dataset=%s",
        os.getpid(),
        device_id,
        len(batch_items),
        top_k,
        dataset_name,
    )
    try:
        from model_manager import model_manager
        model_manager.load_model(device=f"cuda:{device_id}")
        memory_info = model_manager.get_memory_usage()
        if memory_info:
            logger.info(
                "Embedding model ready | pid=%s gpu=%s device=%s",
                os.getpid(),
                device_id,
                memory_info["device"],
            )
        else:
            logger.info("Embedding model ready | pid=%s gpu=%s device=cpu", os.getpid(), device_id)
    except Exception as e:
        logger.warning(
            "Embedding model load failed, falling back to CPU | pid=%s gpu=%s error=%s",
            os.getpid(),
            device_id,
            e,
        )
        model_manager.load_model(device="cpu")
    
    batch_results = {}
    
    cache_dir = os.path.join(log_dir, "cache")
    status_dir = os.path.join(log_dir, "status")
    cost_output_path = os.path.join(log_dir, "cost.json")
    
    device = f"cuda:{device_id}"
    
    for instance_id, item in progress_bar(
        batch_items.items(),
        desc=f"retrieve gpu={device_id} pid={os.getpid()}",
        total=len(batch_items),
        unit="sample",
    ):
        with SampleCostRecorder(
            sample_id=instance_id,
            output_path=cost_output_path,
        ):
            question = item["question"]
            db_name = item["db_name"]

            embed_path = determine_embedding_path(instance_id, dataset_name)

            results, metadata_mapping, completion_message = get_next_k_results(
                instance_id=instance_id,
                question=question,
                db_name=db_name,
                embed_path=embed_path,
                top_k=top_k,
                cache_dir=cache_dir,
                status_dir=status_dir,
                device=device
            )

            table_candidates = []
            column_candidates = []
            column_types = []
            descriptions = []
            column_values = []

            for result in results:
                metadata = result["metadata"]

                table = metadata["table"]
                table_candidates.append(table)

                column = metadata["column"]
                column_candidates.append(column)

                column_type = metadata["column_type"]
                column_types.append(column_type)

                column_value = metadata["column_value"]
                column_values.append(column_value)

                description = metadata["description"]
                descriptions.append(description)

            batch_results[instance_id] = {
                "question": question,
                "db_name": db_name,
                "column_candidates": column_candidates,
                "column_types": column_types,
                "column_values": column_values,
                "table_candidates": table_candidates,
                "descriptions": descriptions,
                "retrieved_count": len(results)
            }

    return batch_results


def retrieve_additional(
    instance_id: str,
    question: str,
    additional_k: int,
    log_dir: str,
    dataset_name: str = DEFAULT_DATASET_NAME,
    device: str = "cuda:0",
):
    cache_dir = os.path.join(log_dir, "cache")
    status_dir = os.path.join(log_dir, "status")

    spider2_data = load_dataset_data(dataset_name)
    
    if instance_id not in spider2_data:
        raise ValueError(f"Instance {instance_id} does not exist")
    
    db_name = spider2_data[instance_id]["db_name"]
    embed_path = determine_embedding_path(instance_id, dataset_name)
    
    results, metadata_mapping, completion_message = get_next_k_results(
        instance_id=instance_id,
        question=question,
        db_name=db_name,
        embed_path=embed_path,
        top_k=additional_k,
        cache_dir=cache_dir,
        status_dir=status_dir,
        device=device
    )
    
    formatted_results = []
    for result in results:
        metadata = result["metadata"]
        formatted_results.append({
            "table": metadata["table"],
            "column": metadata["column"],
            "column_type": metadata["column_type"],
            "column_value": metadata["column_value"],
            "description": metadata["description"],
            "distance": result["distance"]
        })
    
    return formatted_results, completion_message


def retrieve(log_dir: str, top_n: int = 50, dataset_name: str = DEFAULT_DATASET_NAME):
    setup_progress_logging(log_dir)
    logger = get_progress_logger(__name__)
    os.makedirs(log_dir, exist_ok=True)

    spider2_data = load_dataset_data(dataset_name)

    instance_ids = list(spider2_data.keys())

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
    num_gpus = len(visible_devices)
    logger.info(
        "Retrieval setup | dataset=%s samples=%s top_n=%s visible_devices=%s",
        dataset_name,
        len(instance_ids),
        top_n,
        ",".join(visible_devices),
    )
    
    batch_size = len(instance_ids) // num_gpus
    if batch_size == 0:
        batch_size = 1

    batches = []
    for i in range(num_gpus):
        start_idx = i * batch_size
        if i == num_gpus - 1: 
            end_idx = len(instance_ids)
        else:
            end_idx = (i + 1) * batch_size
        
        if start_idx < len(instance_ids):
            batch = {instance_id: spider2_data[instance_id] for instance_id in instance_ids[start_idx:end_idx]}
            batches.append((batch, i, top_n, log_dir, dataset_name))

    mp.set_start_method('spawn', force=True)
    
    with mp.Pool(processes=len(batches)) as pool:
        batch_results = pool.starmap(process_batch_with_device, batches)

    all_candidates = {}
    for instance_id in instance_ids:
        for batch_result in batch_results:
            if instance_id in batch_result:
                all_candidates[instance_id] = batch_result[instance_id]
                break

    with open(os.path.join(f"{log_dir}", "initial_candidates.json"), "w", encoding="utf-8") as f:
        json.dump(all_candidates, f, ensure_ascii=False, indent=2)
        
    logger.info(
        "Retrieval completed | dataset=%s candidates=%s output=%s cache=%s status=%s",
        dataset_name,
        len(all_candidates),
        os.path.join(log_dir, "initial_candidates.json"),
        os.path.join(log_dir, "cache"),
        os.path.join(log_dir, "status"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_path', type=str, default="log_mmqa_global_topn100")
    parser.add_argument('--top_n', type=int, default=100)
    parser.add_argument('--dataset_name', type=str, default=DEFAULT_DATASET_NAME)
    args = parser.parse_args()

    retrieve(args.log_path, top_n=args.top_n, dataset_name=args.dataset_name)
