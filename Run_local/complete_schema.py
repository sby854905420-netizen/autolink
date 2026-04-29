import argparse
import json
import multiprocessing as mp
import os
import re
import shutil
import sqlite3
import threading
from urllib import error as urllib_error
from urllib import request as urllib_request

import pandas as pd
from tqdm import tqdm

from cost_tool import SampleCostRecorder
from config import *
from retrieve_topk_schema import get_next_k_results
from utils import *


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
MMQA_SQLITE_DIR = os.path.join(get_mmqa_data_dir(), "Sqlite_database")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "ministral-3:14b")
QUALIFIED_TABLE_PATTERN = re.compile(r'"([A-Za-z0-9_]+)\.([A-Za-z0-9_]+)"')
QUOTED_DB_TABLE_PATTERN = re.compile(r'"([A-Za-z0-9_]+)"\s*\.\s*"([A-Za-z0-9_]+)"')
QUOTED_DB_UNQUOTED_TABLE_PATTERN = re.compile(r'"([A-Za-z0-9_]+)"\s*\.\s*([A-Za-z0-9_]+)\b')
QUOTED_PRAGMA_PATTERN = re.compile(r'"([A-Za-z0-9_]+)"\s*\.\s*pragma_table_info\s*\(', re.IGNORECASE)
UNQUOTED_DB_TABLE_PATTERN = re.compile(r'(?<![\w"])\b([A-Za-z0-9_]+)\.([A-Za-z0-9_]+)\b')

sqlite_lock = threading.Lock()
AVAILABLE_DB_IDS = {
    os.path.splitext(filename)[0]
    for filename in os.listdir(MMQA_SQLITE_DIR)
    if filename.endswith(".sqlite")
}


def chat_with_ollama(messages):
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
    }

    request = urllib_request.Request(
        f"{OLLAMA_BASE_URL}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib_request.urlopen(request, timeout=600) as response:
            response_data = json.loads(response.read().decode("utf-8"))
    except urllib_error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP error {e.code}: {body}") from e
    except urllib_error.URLError as e:
        raise RuntimeError(f"Failed to connect to Ollama at {OLLAMA_BASE_URL}: {e}") from e

    message = response_data.get("message", {})
    content = message.get("content", "")
    if not content:
        raise RuntimeError(f"Ollama returned an empty response: {response_data}")
    return content, response_data


def normalize_attached_table_references(sql: str) -> str:
    return QUALIFIED_TABLE_PATTERN.sub(r'"\1"."\2"', sql)


def extract_referenced_db_ids(sql: str):
    db_ids = set()

    for match in QUOTED_DB_TABLE_PATTERN.finditer(sql):
        db_id = match.group(1)
        if db_id in AVAILABLE_DB_IDS:
            db_ids.add(db_id)

    for match in QUOTED_DB_UNQUOTED_TABLE_PATTERN.finditer(sql):
        db_id = match.group(1)
        if db_id in AVAILABLE_DB_IDS:
            db_ids.add(db_id)

    for match in QUOTED_PRAGMA_PATTERN.finditer(sql):
        db_id = match.group(1)
        if db_id in AVAILABLE_DB_IDS:
            db_ids.add(db_id)

    for match in UNQUOTED_DB_TABLE_PATTERN.finditer(sql):
        db_id = match.group(1)
        if db_id in AVAILABLE_DB_IDS:
            db_ids.add(db_id)

    return sorted(db_ids)


def build_sqlite_connection(db_ids):
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    for db_id in db_ids:
        db_path = os.path.join(MMQA_SQLITE_DIR, f"{db_id}.sqlite")
        conn.execute(f'ATTACH DATABASE ? AS "{db_id}"', (db_path,))
    return conn


def backup_instance_state(instance_id: str, log_path: str):
    cache_dir = os.path.join(log_path, "cache")
    status_dir = os.path.join(log_path, "status")
    backup_dir = os.path.join(log_path, "backup")

    os.makedirs(backup_dir, exist_ok=True)

    cache_file = os.path.join(cache_dir, f"{instance_id}.json")
    status_file = os.path.join(status_dir, f"{instance_id}.json")

    backup_cache_file = os.path.join(backup_dir, f"{instance_id}_cache.json")
    backup_status_file = os.path.join(backup_dir, f"{instance_id}_status.json")

    if not os.path.exists(backup_cache_file) and not os.path.exists(backup_status_file):
        shutil.copy2(cache_file, backup_cache_file)
        shutil.copy2(status_file, backup_status_file)


def restore_instance_state(instance_id: str, log_path: str):
    cache_dir = os.path.join(log_path, "cache")
    status_dir = os.path.join(log_path, "status")
    backup_dir = os.path.join(log_path, "backup")

    backup_cache_file = os.path.join(backup_dir, f"{instance_id}_cache.json")
    backup_status_file = os.path.join(backup_dir, f"{instance_id}_status.json")

    cache_file = os.path.join(cache_dir, f"{instance_id}.json")
    status_file = os.path.join(status_dir, f"{instance_id}.json")

    if os.path.exists(backup_cache_file):
        shutil.copy2(backup_cache_file, cache_file)
    if os.path.exists(backup_status_file):
        shutil.copy2(backup_status_file, status_file)


def thread_safe_sql_execution(instance_id, sql, db_name, dataset_name):
    if instance_id.startswith("local") or is_dataset_instance(instance_id, dataset_name):
        with sqlite_lock:
            return sql_execution(instance_id, sql, db_name, dataset_name)
    return sql_execution(instance_id, sql, db_name, dataset_name)


def sql_execution(instance_id, sql, db_name, dataset_name):
    if not (instance_id.startswith("local") or is_dataset_instance(instance_id, dataset_name)):
        return "error", f"Unsupported instance_id for Run_local: {instance_id}"

    normalized_sql = normalize_attached_table_references(sql)
    referenced_db_ids = extract_referenced_db_ids(normalized_sql)
    if not referenced_db_ids:
        return "error", (
            "No database ids were found in the SQL query. "
            "Use full table names like \"db_id\".\"table_name\" in the global MMQA space."
        )
    if len(referenced_db_ids) > 10:
        return "error", (
            f"The SQL query references {len(referenced_db_ids)} databases, "
            "which exceeds SQLite's ATTACH limit of 10."
        )

    conn = build_sqlite_connection(referenced_db_ids)
    try:
        df = pd.read_sql_query(normalized_sql, conn)
        if df.empty:
            return "empty", "No data found for the specified query."
        return "success", df
    except Exception as e:
        return "error", f"Error occurred while fetching data: {e}"
    finally:
        conn.close()


def remove_column_values(schema_text):
    lines = schema_text.split("\n")
    processed_lines = []

    for line in lines:
        if line.strip().startswith("Column name:"):
            pattern = r"(Column name:.*?Column type:.*?); Column value: \[.*?\]; (Description:.*)"
            match = re.match(pattern, line.strip())

            if match:
                processed_line = f"{match.group(1)}; {match.group(2)}"
                processed_lines.append(processed_line)
            else:
                if "; Column value:" in line and "; Description:" in line:
                    value_start = line.find("; Column value:")
                    desc_start = line.find("; Description:")
                    if value_start != -1 and desc_start != -1 and value_start < desc_start:
                        processed_line = line[:value_start] + line[desc_start:]
                        processed_lines.append(processed_line)
                    else:
                        processed_lines.append(line)
                else:
                    processed_lines.append(line)
        else:
            processed_lines.append(line)

    return "\n".join(processed_lines)


def process_instance_batch(batch_instances, log_path, dataset_name):
    cache_path = os.path.join(log_path, "cache")
    status_path = os.path.join(log_path, "status")
    schema_path = os.path.join(log_path, "schema_prompts")
    model_output_path = os.path.join(log_path, "model_output")
    tool_calls_path = os.path.join(log_path, "tool_calls")
    input_path = os.path.join(log_path, "input")
    candidates_path = os.path.join(log_path, "candidates")
    error_path = os.path.join(log_path, "error")
    cost_output_path = os.path.join(log_path, "cost.json")

    for instance_id, info in tqdm(batch_instances.items(), leave=False, desc=f"Thread {os.getpid()}"):
        with SampleCostRecorder(
            sample_id=instance_id,
            output_path=cost_output_path,
        ) as cost_recorder:
            restore_instance_state(instance_id, log_path)

            each_candidates = {}
            embed_path = determine_embedding_path(instance_id, dataset_name)

            if not (instance_id.startswith("local") or is_dataset_instance(instance_id, dataset_name)):
                raise ValueError(f"Unknown instance ID: {instance_id}")

            documents_path = "documents/localdb.json"
            sql_type = SQLITE
            sql_optimization = SQLITE_DIALECT_OPTIMIZATION

            with open(documents_path, "r", encoding="utf-8") as f:
                documents = json.load(f)

            spider2_data = load_dataset_data(dataset_name)

            if instance_id not in spider2_data:
                raise ValueError(f"Instance ID {instance_id} not found in {get_dataset_file(dataset_name)}")

            question = info["question"]
            db_name = info["db_name"]
            knowledge_data = ""
            db_documents = documents[db_name]

            with open(f"{schema_path}/{instance_id}.txt", "r", encoding="utf-8") as f:
                retrieved_schemas = f.read()

            all_tables = list(db_documents.keys())
            all_calls = {}
            all_model_output = ""
            all_inputs = ""

            system_prompt = SCHEMA_LINKING.format(
                SQL_TYPE=sql_type,
                SQL_OPTIMIZATION=sql_optimization,
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": USER_INPUT.format(
                        RETRIEVED_SCHEMA=json.dumps(retrieved_schemas, ensure_ascii=False),
                        USER_QUESTION=question,
                        EXTERNAL_KNOWLEDGE=knowledge_data,
                        ALL_TABLES=json.dumps(all_tables, ensure_ascii=False),
                    ),
                },
            ]

            is_finished = False
            is_error = False

            column_candidates = []
            table_candidates = []
            column_type_candidates = []
            column_value_candidates = []
            description_candidates = []

            for i in range(10):
                all_inputs += f"Turn {i}\n" + str(messages) + "\n" + "=" * 50 + "\n\n"

                if is_finished or is_error:
                    break

                try:
                    model_output, response_data = chat_with_ollama(messages)
                    cost_recorder.add_response_usage(response_data)
                except Exception as e:
                    model_output = f"Model call failed: {e}"
                    is_error = True

                all_model_output += f"Turn {i}\n{model_output}\n" + "=" * 50 + "\n\n"

                try:
                    full_lines, tool_calls = parse_model_output(model_output)
                except Exception:
                    full_lines = []
                    tool_calls = []
                    is_error = True

                    with open(os.path.join(error_path, instance_id) + ".txt", "w", encoding="utf-8") as f:
                        f.write(model_output)

                func_messages = ""

                for line, func in zip(full_lines, tool_calls):
                    if func["tool"] == "stop":
                        is_finished = True
                        break
                    elif func["tool"] == "schema_retrieval":
                        is_complete = False

                        table = func["table"]
                        column = func["column"]
                        description = func["description"]

                        if table == "" and column == "" and description == "":
                            continue

                        if is_complete:
                            continue

                        func_messages += f"Tool: {line} \n The tool returns the following results:"

                        retrieve_content = (
                            "column name: " + column + "\n"
                            + "table name: " + table + "\n"
                            + "description: " + description
                        )

                        semantic_results, metadata_mapping, text = get_next_k_results(
                            instance_id=instance_id,
                            question=retrieve_content,
                            db_name=db_name,
                            embed_path=embed_path,
                            top_k=3,
                            cache_dir=cache_path,
                            status_dir=status_path,
                            device="cuda:0",
                        )

                        new_results = ""
                        for result in semantic_results:
                            metadata = result["metadata"]
                            table = metadata["table"]
                            column = metadata["column"]
                            description = metadata["description"]
                            column_type = metadata["column_type"]
                            column_value = metadata["column_value"]
                            func_messages += f"{description}\n"
                            new_results += f"{description}\n"
                            column_candidates.append(column)
                            table_candidates.append(table)
                            column_type_candidates.append(column_type)
                            column_value_candidates.append(column_value)
                            description_candidates.append(description)

                        if text:
                            is_complete = True
                            func_messages += f"{text}\n"
                        func["result"] = new_results
                        func_messages += "\n"

                    elif func["tool"] == "sql_execution" or func["tool"] == "sql_draft":
                        query = func["query"]

                        if query == "":
                            continue
                        func_messages += f"Tool: {line} \n The tool returns the following results:\n"
                        exec_status, results = thread_safe_sql_execution(instance_id, query, db_name, dataset_name)
                        func_messages += f"{results}\n\n"
                        func["result"] = str(results)

                all_calls[f"turn_{i}"] = tool_calls
                func_messages += (
                    "\nFor `@sql_execution`, if the results return column names or table names including the missing "
                    "tables or columns you think, in this turn, you must use the @schema_retrieval tool to retrieve "
                    "the missing tables or columns.\nBecause we will use initial schema and the results of "
                    "`@sql_execution` tool as the final schema. Please do not think that the columns obtained by "
                    "@sql_execution will be recalled. Only the columns obtained by `@schema_retrieval` can be "
                    "considered to be recalled correctly.\nPlease also pay attention to the column name like `*id`, "
                    "`*name`, `*text`, `*code` and so on. These columns are often crucial for final SQL construction, "
                    "especially for joins, filtering, and output.\nYou also need to pay attention that Some important "
                    "columns may exist in more than one table, but the initial schema may include only one instance. "
                    "This can cause critical tables to be omitted if you're not careful. Always check whether a column "
                    "name is shared across tables, and whether the other tables containing it also provide relevant "
                    "context for the question."
                )

                messages.append({"role": "assistant", "content": model_output})
                messages.append({"role": "user", "content": func_messages})

            each_candidates[instance_id] = {
                "question": question,
                "db_name": db_name,
                "column_candidates": column_candidates,
                "column_types": column_type_candidates,
                "column_values": column_value_candidates,
                "table_candidates": table_candidates,
                "descriptions": description_candidates,
            }

            if is_error:
                print(f"Error occurred for instance {instance_id}. Skipping...")
                continue

            with open(os.path.join(model_output_path, instance_id) + ".txt", "w", encoding="utf-8") as f:
                f.write(all_model_output)

            with open(os.path.join(tool_calls_path, instance_id) + ".json", "w", encoding="utf-8") as f:
                json.dump(all_calls, f, ensure_ascii=False, indent=2)

            with open(os.path.join(input_path, instance_id) + ".txt", "w", encoding="utf-8") as f:
                f.write(all_inputs)

            with open(os.path.join(candidates_path, instance_id) + ".json", "w", encoding="utf-8") as f:
                json.dump(each_candidates, f, ensure_ascii=False, indent=2)


def complete_schema(log_path, dataset_name, num_threads=3):
    status_dir = os.path.join(log_path, "status")

    model_output_path = os.path.join(log_path, "model_output")
    os.makedirs(model_output_path, exist_ok=True)

    tool_calls_path = os.path.join(log_path, "tool_calls")
    os.makedirs(tool_calls_path, exist_ok=True)

    input_path = os.path.join(log_path, "input")
    os.makedirs(input_path, exist_ok=True)

    candidates_path = os.path.join(log_path, "candidates")
    os.makedirs(candidates_path, exist_ok=True)

    error_path = os.path.join(log_path, "error")
    os.makedirs(error_path, exist_ok=True)

    with open(os.path.join(log_path, "initial_candidates.json"), "r", encoding="utf-8") as f:
        initial_candidates = json.load(f)
    spider2_data = load_dataset_data(dataset_name)

    instance_ids = list(spider2_data.keys())

    print("Backup instance status ...")
    for instance_id in instance_ids:
        backup_instance_state(instance_id, log_path)

    clean_instance_ids = []
    for instance_id in instance_ids:
        if os.path.exists(os.path.join(candidates_path, instance_id) + ".json"):
            continue
        with open(os.path.join(status_dir, instance_id + ".json"), "r", encoding="utf-8") as f:
            cache_data = json.load(f)

        if cache_data.get("is_complete", False):
            continue

        clean_instance_ids.append(instance_id)

    print(f"Unfinished instances: {len(clean_instance_ids)}")

    uncompleted_path = os.path.join(log_path, f"uncompleted_instances_{dataset_name}.txt")
    with open(uncompleted_path, "w", encoding="utf-8") as f:
        for instance_id in clean_instance_ids:
            f.write(instance_id + "\n")

    instance_ids = clean_instance_ids

    batch_size = max(1, len(instance_ids) // num_threads)
    batches = []

    for i in range(0, len(instance_ids), batch_size):
        end_idx = min(i + batch_size, len(instance_ids))
        batch = {instance_id: initial_candidates[instance_id] for instance_id in instance_ids[i:end_idx]}
        batches.append(batch)

    num_threads = min(num_threads, len(batches))

    mp.set_start_method("spawn", force=True)

    processes = []
    for i in range(num_threads):
        if i < len(batches):
            p = mp.Process(
                target=process_instance_batch,
                args=(batches[i], log_path, dataset_name),
            )
            processes.append(p)
            p.start()

    for p in processes:
        p.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", type=str, default="log_mmqa_global_topn100")
    parser.add_argument("--dataset_name", type=str, default=DEFAULT_DATASET_NAME)
    parser.add_argument("--num_threads", type=int, default=1)
    args = parser.parse_args()
    print("Starting schema completion...")
    complete_schema(args.log_path, args.dataset_name, num_threads=max(1, args.num_threads))
    print("Schema completion finished.")
