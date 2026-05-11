import os
import json
from tqdm import tqdm
import re
import argparse

from cost_tool import SampleCostRecorder
from progress_logger import get_progress_logger, progress_bar, setup_progress_logging
from utils import (
    DEFAULT_DATASET_NAME,
    get_documents_file,
    is_dataset_instance,
    load_dataset_data,
    resolve_external_knowledge_path,
)

def extract_description(description_text):
    lines = description_text.strip().split("\n")

    for line in lines:
        if line.startswith("description:"):
            return line[len("description:"):].strip()

    return ""

def get_parentheses_content(text: str, start: str = "(", end: str = ")") -> str:
    if not text or start not in text:
        return ""

    stack = []
    start_index = -1

    for i, char in enumerate(text):
        if char == start:
            if not stack:
                start_index = i
            stack.append(char)
        elif char == end and stack:
            stack.pop()
            if not stack:
                return text[start_index:i + 1]

    raise ValueError(f"No matching '{end}' found for '{start}' in the text")


def long_value(values: list):
    for value in values:
        if len(str(value)) > 200:
            return True
    return False


def truncate_nested_dict(obj, max_length=100):
    if isinstance(obj, dict):
        return {key: truncate_nested_dict(value, max_length) for key, value in obj.items()}

    elif isinstance(obj, list):
        return [truncate_nested_dict(item, max_length) for item in obj]

    elif isinstance(obj, str):
        if len(obj) > max_length:
            return obj[:max_length] + "...(truncated)"
        return obj

    else:
        str_obj = str(obj)
        if len(str_obj) > max_length:
            return str_obj[:max_length] + "...(truncated)"
        return str_obj

def process_values(values: list, is_dict: bool = False, is_array: bool = False, is_variant: bool = False,
                   max_length: int = 100):
    if not (is_dict or is_array or is_variant):
        if not long_value(values):
            return values[:3]
        return [str(value)[:250] + "...(truncated)" if len(str(value)) > 250 else str(value) for value in values][:3]
    final_values = []
    if is_dict:
        if values[0] == "None":
            return values[:3]
        for i in range(3):
            dict_content = get_parentheses_content(values[i], "{", "}")

            dict_content = preprocess_json_content(dict_content)

            try:
                dict_content = json.loads(dict_content)
            except Exception as e:
                try:
                    dict_content = fix_malformed_json(dict_content)
                    dict_content = json.loads(dict_content)
                except Exception as e2:
                    final_values.append(str(values[i])[:max_length] + "...(truncated)" if len(str(values[i])) > max_length else str(
                        values[i]))

            truncated_dict = truncate_nested_dict(dict_content, max_length=max_length)
            final_values.append(json.dumps(truncated_dict, ensure_ascii=False))
        return final_values

    if is_array:
        for i in range(3):
            array_content = "[" + get_parentheses_content(values[i], "{", "}") + "]"

            if array_content != "[]":
                array_content = preprocess_json_content(array_content)
            else:
                array_content = preprocess_json_content(values[i])

            truncated_array = truncate_nested_dict(array_content, max_length=max_length)
            final_values.append(json.dumps(truncated_array, ensure_ascii=False))
        return final_values
    
    if is_variant:
        if values == "None":
            return values
        if values == []:
            return values
        if values[0].startswith("{"):
            if values[0] == "None":
                return values[:3]
            for i in range(3):
                dict_content = get_parentheses_content(str(values[i]), "{", "}")

                dict_content = preprocess_json_content(dict_content)
                try:
                    dict_content = json.loads(dict_content)
                except Exception as e:
                    try:
                        dict_content = fix_malformed_json(dict_content)
                        dict_content = json.loads(dict_content)
                    except Exception as e2:
                        with open("error_log.txt", "a") as error_file:
                            error_file.write(f"Error processing dict content: {dict_content[:500]}...\n")
                            error_file.write(f"Error: {str(e2)}\n")
                            print(f"Error processing dict content: {dict_content[:500]}...\n")
                            final_values.append(str(values[i])[:max_length] + "...(truncated)" if len(str(values[i])) > max_length else str(
                            values[i]))

                truncated_dict = truncate_nested_dict(dict_content, max_length=max_length)
                final_values.append(json.dumps(truncated_dict, ensure_ascii=False))
            return final_values
        
        elif values[0].startswith("["):
            for i in range(3):
                array_content = "[" + get_parentheses_content(str(values[i]), "{", "}") + "]"

                if array_content != "[]":
                    array_content = preprocess_json_content(array_content)
                else:
                    array_content = preprocess_json_content(str(values[i]))

                truncated_array = truncate_nested_dict(array_content, max_length=max_length)
                final_values.append(json.dumps(truncated_array, ensure_ascii=False))
            return final_values
        else:
            if not long_value(values):
                return values[:3]
            return [str(value)[:250] + "...(truncated)" if len(str(value)) > 250 else str(value) for value in values][:3]

def preprocess_json_content(content):
    content = content.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')

    content = re.sub(r'\s+', ' ', content)

    content = re.sub(r'array\([^)]*\)', '[]', content)

    content = content.replace('\\"', '"').replace("'", '"').replace("None", "null").replace("True", "true").replace("False", "false")

    content = fix_common_json_issues(content)

    return content


def fix_common_json_issues(content):
    content = re.sub(r'": "([^"]*)"([^,}\]]*)"', r'": "\1\2"', content)
    content = re.sub(r'": \[\]"', r'": []', content)
    content = re.sub(r'"(\d+\.?\d*[eE][+-]?\d+)"', r'\1', content)
    content = re.sub(r'"null"', 'null', content)
    content = re.sub(r'": "(\d+\.?\d*)"([,}\]])', r'": \1\2', content)
    content = re.sub(r'": "(\d+)"([,}\]])', r'": \1\2', content)
    return content


def fix_malformed_json(content):
    if len(content) > 10000: 
        content = content[:10000]
        last_comma = content.rfind(',')
        if last_comma > 0:
            content = content[:last_comma]
        content += "}"
    content = re.sub(r',\s*"[^"]*":\s*"[^"]*$', '', content)
    content = re.sub(r',\s*"[^"]*":\s*[^,}]*$', '', content)

    if content.count('{') > content.count('}'):
        content += '}' * (content.count('{') - content.count('}'))

    if content.count('[') > content.count(']'):
        content += ']' * (content.count('[') - content.count(']'))

    content = re.sub(r',(\s*[}\]])', r'\1', content)

    return content


def safe_json_loads(json_str, max_attempts=3):

    for attempt in range(max_attempts):
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            if attempt == max_attempts - 1:
                try:
                    simplified = extract_basic_info(json_str)
                    return json.loads(simplified)
                except:
                    raise e
            else:
                json_str = progressive_fix(json_str, attempt)


def extract_basic_info(content):
    basic_info = {}
    simple_patterns = [
        r'"([^"]+)":\s*"([^"]*)"',  
        r'"([^"]+)":\s*(\d+\.?\d*)',  
        r'"([^"]+)":\s*(true|false|null)' 
    ]

    for pattern in simple_patterns:
        matches = re.findall(pattern, content)
        for key, value in matches:
            if key not in basic_info:  
                if value in ['true', 'false', 'null']:
                    basic_info[key] = json.loads(value)
                elif value.replace('.', '').replace('-', '').isdigit():
                    basic_info[key] = json.loads(value)
                else:
                    basic_info[key] = value

    return json.dumps(basic_info)


def progressive_fix(content, attempt):
    if attempt == 0:
        return preprocess_json_content(content)
    elif attempt == 1:
        return fix_malformed_json(preprocess_json_content(content))
    else:
        return extract_basic_info(content)

def get_column_type(column_type: str):
    is_dict, is_array, is_variant = False, False, False
    if column_type.startswith("ARRAY<") and column_type.endswith(">"):
        is_array = True
    elif column_type.startswith("STRUCT<") and column_type.endswith(">"):
        is_dict = True
    elif column_type.startswith("VARIANT"):
        is_variant = True

    return is_dict, is_array, is_variant


def generate_schema_prompt(
    log_path: str,
    is_initial: bool = False,
    dataset_name: str = DEFAULT_DATASET_NAME,
):
    setup_progress_logging(log_path)
    logger = get_progress_logger(__name__)
    cost_output_path = os.path.join(log_path, "cost.json")
    if is_initial:
        with open(f"{log_path}/unfilled_pre_rule.json", "r", encoding="utf-8") as f:
            candidates = json.load(f)
        
        os.makedirs(f"{log_path}/schema_prompts", exist_ok=True)
        output_dir = os.path.join(log_path, "schema_prompts")
        prompt_kind = "initial"
    else:
        spider2_data = load_dataset_data(dataset_name)
        with open(f"{log_path}/unfilled_schema.json", "r", encoding="utf-8") as f:
            candidates = json.load(f)
        os.makedirs(f"{log_path}/final_schema_prompts", exist_ok=True)
        output_dir = os.path.join(log_path, "final_schema_prompts")
        prompt_kind = "final"

    logger.info(
        "Schema prompt generation started | dataset=%s kind=%s samples=%s output_dir=%s",
        dataset_name,
        prompt_kind,
        len(candidates),
        output_dir,
    )
    
    with open(get_documents_file(dataset_name), "r", encoding="utf-8") as f:
        localdb_data = json.load(f)

    schema_prompt = ""

    for instance_id, schema_info in progress_bar(
        candidates.items(),
        desc=f"schema prompts ({prompt_kind})",
        total=len(candidates),
        unit="sample",
    ):
        with SampleCostRecorder(
            sample_id=instance_id,
            output_path=cost_output_path,
        ):
            db_name = schema_info["db_name"]

            if instance_id.startswith("local") or is_dataset_instance(instance_id, dataset_name):
                db_data = localdb_data[db_name]
            else:
                raise ValueError(f"Unknown instance ID: {instance_id}")

            column_candidates = schema_info["column_candidates"]
            column_types = schema_info["column_types"]
            column_values = schema_info["column_values"]
            column_candidates = schema_info["column_candidates"]
            table_candidates = schema_info["table_candidates"]
            descriptions = schema_info["descriptions"]

            mapping = {}

            for column_name, column_type, column_value, table_name, description in zip(
                    column_candidates, column_types, column_values, table_candidates, descriptions
            ):
                if table_name not in mapping:
                    mapping[table_name] = {}
                    mapping[table_name]["columns"] = []
                    mapping[table_name]["column_types"] = []
                    mapping[table_name]["column_values"] = []
                    mapping[table_name]["descriptions"] = []

                desc = extract_description(description)
                mapping[table_name]["columns"].append(column_name)
                mapping[table_name]["column_types"].append(column_type)
                mapping[table_name]["column_values"].append(column_value)
                mapping[table_name]["descriptions"].append(desc)

            schema_prompt = ""
            for table, column_info in mapping.items():

                schema_prompt += f"###Table full name: {table}\n[\n"

                columns = column_info["columns"]
                column_types = column_info["column_types"]
                column_values = column_info["column_values"]
                descriptions = column_info["descriptions"]
                similar_tables = db_data[table]["similar_tables"]

                for column_name, column_type, column_value, description in zip(
                        columns, column_types, column_values, descriptions
                ):
                    is_dict, is_array, is_variant = get_column_type(column_type)
                    if not instance_id.startswith("sf"):
                        column_va = process_values(column_value, is_dict=is_dict, is_array=is_array, is_variant=is_variant,
                                                  max_length=100)
                    else:
                        column_va = []
                        if column_value == []:
                            column_va = []
                        else:
                            for i in range(len(column_value)):
                                if len(str(column_value[i])) > 1000:
                                    column_va.append(column_value[i][:1000] + "...(truncated)")
                                else:
                                    column_va.append(column_value[i])
                            column_va = column_va[:3]
                    if description:
                        schema_prompt += f"    {column_name} (Type: {column_type}; Sample values: {column_va}; Description: {description})\n"
                    else:
                        schema_prompt += f"    {column_name} (Type: {column_type}; Sample values: {column_va})\n"
                schema_prompt += "]\n"

                if similar_tables:
                    if instance_id.startswith("bq") or instance_id.startswith("ga"):
                        table_name = [similar_table.split(".")[-1] for similar_table in similar_tables]
                        schema_prompt += f"**Some other tables have the similar structure: [{', '.join(table_name)}]**\n"
                    elif instance_id.startswith("sf"):
                        try:
                            table_name = [".".join(similar_table.split(".")[1:]) for similar_table in similar_tables]
                            schema_prompt += f"**Some other tables have the similar structure: [{', '.join(table_name)}]**\n"
                        except Exception as e:
                            raise e
                    else:
                        schema_prompt += f"**Some other tables have the similar structure: [{', '.join(similar_tables)}]**\n"

                schema_prompt += "\n" + "-" * 50 + "\n\n"
            if is_initial:
                with open(f"{log_path}/schema_prompts/{instance_id}.txt", "w", encoding="utf-8") as f:
                    f.write(schema_prompt)
            else:
                external_text = ""

                if instance_id in spider2_data:
                    ek_file = spider2_data[instance_id].get("external_knowledge", "")
                    if ek_file:
                        ek_path = resolve_external_knowledge_path(dataset_name, ek_file)
                        if ek_path:
                            with open(ek_path, "r", encoding="utf-8") as ef:
                                ek_content = ef.read()

                            external_text = (
                                "External knowledge that might be helpful: \n"
                                + ek_content
                            )
                        else:
                            logger.warning("External knowledge file not found | instance=%s path=%s", instance_id, ek_file)

                full_prompt = schema_prompt + external_text
                with open(f"{log_path}/final_schema_prompts/{instance_id}.txt", "w", encoding="utf-8") as f:
                    f.write(full_prompt)
    logger.info(
        "Schema prompt generation finished | dataset=%s kind=%s samples=%s output_dir=%s",
        dataset_name,
        prompt_kind,
        len(candidates),
        output_dir,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_path', type=str, default="log_mmqa_global_topn100")
    parser.add_argument('--is_initial', action='store_true', default=False)
    parser.add_argument('--dataset_name', type=str, default=DEFAULT_DATASET_NAME)
    args = parser.parse_args()

    generate_schema_prompt(args.log_path, args.is_initial, args.dataset_name)
    if args.is_initial:
        all_prompts = os.listdir(f"{args.log_path}/schema_prompts")
    else:
        all_prompts = os.listdir(f"{args.log_path}/final_schema_prompts")
    logger = get_progress_logger(__name__)
    logger.info("Schema prompts generated successfully | total=%s", len(all_prompts))
