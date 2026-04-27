import os
import json
import re

DEFAULT_DATASET_NAME = "mmqa"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)


def get_mmqa_data_dir() -> str:
    configured_path = os.environ.get("MMQA_DATA_DIR")
    if configured_path:
        return os.path.abspath(configured_path)

    candidates = [
        os.path.join(PROJECT_ROOT, "Data", "MMQA"),
        os.path.join(PROJECT_ROOT, "MMQA"),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate

    return candidates[0]


def get_dataset_file(dataset_name: str = DEFAULT_DATASET_NAME) -> str:
    return os.path.join(BASE_DIR, f"{dataset_name}_data.json")


def load_dataset_data(dataset_name: str = DEFAULT_DATASET_NAME):
    data_file = get_dataset_file(dataset_name)
    with open(data_file, "r", encoding="utf-8") as f:
        return json.load(f)


def is_dataset_instance(instance_id: str, dataset_name: str = DEFAULT_DATASET_NAME) -> bool:
    return instance_id.startswith(f"{dataset_name}_")


def determine_embedding_path(instance_id: str, dataset_name: str = DEFAULT_DATASET_NAME) -> str:
    base_path = "embeddings"
    if instance_id.startswith("bq") or instance_id.startswith("ga"):
        embed_path = os.path.join(base_path, "bigquery")
    elif instance_id.startswith("sf"):
        embed_path = os.path.join(base_path, "snowflake")
    elif instance_id.startswith("local") or is_dataset_instance(instance_id, dataset_name):
        embed_path = os.path.join(base_path, "localdb")
    else:
        raise ValueError(f"Unknown instance_id: {instance_id}")
    return embed_path

def get_subdir(dir_path):
    subdirs = [
        name
        for name in os.listdir(dir_path)
        if os.path.isdir(os.path.join(dir_path, name))
    ]
    return subdirs

def get_json_files(file_path):
    json_files = []
    for root, dirs, files in os.walk(file_path):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))
    json_files = [json_file.replace("\\", "/") for json_file in json_files]
    return json_files

def remove_digits(table_name: str) -> str:
    return "".join([char for char in table_name if not char.isdigit()])

def parse_model_output(output: str):
    full_lines = []
    tool_calls = []
    
    call_types = ["@schema_retrieval", "@sql_execution", "@sql_draft", "@sql_exploration", "@stop()", "@add_schema"]
    
    lines = output.splitlines()
    lines = [line.strip() for line in lines]
    i = 0
    blocks = []
    
    while i < len(lines):
        line = lines[i]

        if any(line.startswith(call_type) for call_type in call_types):
            stack = []
            block_lines = [line]
            
            open_pos = line.find('(')
            if open_pos != -1:
                stack.append('(')

                for c in line[open_pos+1:]:
                    if c == '(':
                        stack.append('(')
                    elif c == ')':
                        stack.pop()  
                        if not stack:  
                            break
                
                j = i + 1
                while stack and j < len(lines):
                    next_line = lines[j].strip()
                    block_lines.append(next_line)
                    
                    for c in next_line:
                        if c == '(':
                            stack.append('(')
                        elif c == ')':
                            if stack:  
                                stack.pop()
                            if not stack:  
                                break
                    
                    j += 1
                    if not stack:  
                        break
                
                i = j
                blocks.append('\n'.join(block_lines))
            else:
                i += 1
        else:
            i += 1
    
    for block in blocks:
        for call_type in call_types:
            if block.strip().startswith(call_type):
                full_lines.append(block)
                
                if call_type == "@schema_retrieval":
                    table_match = re.search(r'table\s*[:=]\s*["\']([^"\']*)["\']', block)
                    column_match = re.search(r'column\s*[:=]\s*["\']([^"\']*)["\']', block)
                    desc_match = re.search(r'description\s*[:=]\s*["\']([^"\']*)["\']', block)
                    
                    tool_calls.append({
                        "tool": "schema_retrieval",
                        "table": table_match.group(1) if table_match else "",
                        "column": column_match.group(1) if column_match else "",
                        "description": desc_match.group(1) if desc_match else ""
                    })
                
                elif call_type == "@add_schema":
                    table_match = re.search(r'table\s*[:=]\s*["\']([^"\']*)["\']', block)
                    column_match = re.search(r'column\s*[:=]\s*["\']([^"\']*)["\']', block)
                    
                    tool_calls.append({
                        "tool": "add_schema",
                        "table": table_match.group(1) if table_match else "",
                        "column": column_match.group(1) if column_match else ""
                    })
                
                elif call_type == "@sql_execution" or call_type == "@sql_draft":
                    tool_type = call_type[1:]  
                    
                    query_match = re.search(r'query\s*[:=]\s*"""(.*?)"""', block, re.DOTALL)
                    if not query_match:
                        query_match = re.search(r'query\s*[:=]\s*["\']([^"\']*)["\']', block)
                    
                    if not query_match:
                        query_start = re.search(r'query\s*[:=]\s*', block)
                        if query_start:
                            query_text = block[query_start.end():]
                            if query_text.startswith('"') or query_text.startswith("'"):
                                query = query_text[1:-1] if query_text.endswith('"') or query_text.endswith("'") else query_text
                            else:
                                stack = []
                                for i, c in enumerate(query_text):
                                    if c == '(':
                                        stack.append(i)
                                    elif c == ')':
                                        if stack:
                                            stack.pop()
                                        if not stack: 
                                            query = query_text[:i].strip()
                                            break
                                else:
                                    query = query_text.strip()
                        else:
                            query = ""
                    else:
                        query = query_match.group(1)
                    
                    tool_calls.append({
                        "tool": tool_type,
                        "query": query
                    })
                elif call_type == "@stop()":
                    tool_calls.append({
                        "tool": "stop"
                    })
                break 
    
    return full_lines, tool_calls

def mask_digits(table_name: str) -> str:
    """
    Mask digits in the table name with asterisks.
    """
    table_name = re.sub(r'\d', '*', table_name)
    table_name = re.sub(r'\*+', '*', table_name)
    return table_name
