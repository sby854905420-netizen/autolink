from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from cost_tool import SampleCostRecorder, extract_token_count
from llm_backends import default_context_length_for_model, chat_with_model
from progress_logger import get_progress_logger, progress_bar, setup_progress_logging
from utils import (
    DEFAULT_DATASET_NAME,
    PROJECT_ROOT,
    get_dataset_file,
    get_documents_dir,
    get_sql_dialect,
    load_dataset_data,
    require_supported_dataset,
    resolve_external_knowledge_path,
    safe_path_component,
)


DEFAULT_SQL_LLM_NAME = "mistralai/Ministral-3-14B-Instruct-2512"
DEFAULT_SQL_MAX_NEW_TOKENS = 4096
DEFAULT_SQL_PROMPT_PATH = Path(__file__).resolve().parent / "sql_generation.txt"
DEFAULT_SCHEMA_SOURCE_FILE = "merge_candidates.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate answer SQL from AutoLink final schema prompts."
    )
    parser.add_argument("dataset", nargs="?", default=None)
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument(
        "--log_path",
        type=Path,
        default=None,
        help="AutoLink schema-linking run directory. Defaults to the latest run for the fixed SQL model.",
    )
    parser.add_argument(
        "--schema_model_name",
        default=DEFAULT_SQL_LLM_NAME,
        help="Model directory name used to locate schema-linking logs when --log_path is omitted.",
    )
    parser.add_argument("--schema_source_file", default=DEFAULT_SCHEMA_SOURCE_FILE)
    parser.add_argument("--schema_prompts_dir", type=Path, default=None)
    parser.add_argument("--prompt_path", type=Path, default=DEFAULT_SQL_PROMPT_PATH)
    parser.add_argument("--output_path", type=Path, default=None)
    parser.add_argument("--sql_dialect", default=None)
    parser.add_argument("--hf_context_length", type=int, default=None)
    parser.add_argument(
        "--hf_max_new_tokens",
        type=int,
        default=int(os.environ.get("HF_MAX_NEW_TOKENS", str(DEFAULT_SQL_MAX_NEW_TOKENS))),
    )
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--instance_id",
        action="append",
        default=None,
        help="Generate SQL for a single instance id. Can be supplied multiple times.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite --output_path if it already exists instead of resuming/skipping existing ids.",
    )
    return parser.parse_args()


def resolve_dataset_name(args: argparse.Namespace) -> str:
    return require_supported_dataset(
        args.dataset_name or args.dataset or os.environ.get("DATASET_NAME") or DEFAULT_DATASET_NAME
    )


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_json_mapping(path: Path) -> dict[str, Any]:
    value = load_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def find_latest_schema_log_path(dataset_name: str, schema_model_name: str) -> Path:
    model_dir_name = safe_path_component(schema_model_name)
    base_dir = Path(PROJECT_ROOT) / "Log" / dataset_name / model_dir_name
    if not base_dir.is_dir():
        raise FileNotFoundError(f"Schema log model directory not found: {base_dir}")

    candidates = [
        path
        for path in base_dir.iterdir()
        if path.is_dir() and (path / "final_schema_prompts").is_dir()
    ]
    if not candidates:
        raise FileNotFoundError(f"No schema runs with final_schema_prompts found under {base_dir}")

    candidates.sort(key=lambda path: path.name, reverse=True)
    return candidates[0]


def resolve_log_path(args: argparse.Namespace, dataset_name: str) -> Path:
    if args.log_path is not None:
        log_path = args.log_path.resolve()
        if not log_path.is_dir():
            raise FileNotFoundError(f"Log path not found: {log_path}")
        return log_path

    return find_latest_schema_log_path(
        dataset_name=dataset_name,
        schema_model_name=args.schema_model_name,
    )


def resolve_output_path(output_path: Path | None, log_path: Path, dataset_name: str) -> Path:
    if output_path is not None:
        resolved = output_path.resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        return resolved

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = log_path / "sql_results"
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir / f"sql_generation_{dataset_name}_{run_id}.json"


def default_sql_dialect_requirements(dataset_name: str) -> str:
    dialect = get_sql_dialect(dataset_name)
    if dialect == "sqlite":
        return (
            "Use SQLite SQL. The AutoLink MMQA global schema may show tables as "
            "database_id.table_name; reference them with SQLite attached-database "
            "notation such as \"database_id\".\"table_name\" when needed. Do not use "
            "Snowflake-only syntax."
        )
    if dialect == "snowflake":
        return (
            "Use Snowflake SQL. Preserve fully qualified table names exactly as shown, "
            "usually DATABASE.SCHEMA.TABLE. Snowflake features such as CTEs, QUALIFY, "
            "ILIKE, DATEADD, DATEDIFF, TRY_CAST, TO_DATE, TRUE/FALSE boolean literals, "
            "and :: casts are allowed when useful. Do not write SQLite-specific SQL."
        )
    return f"Use {dialect} SQL."


def render_prompt(
    prompt_template: str,
    *,
    schema_text: str,
    question: str,
    hint: str,
    dataset_name: str,
    sql_dialect: str,
) -> str:
    return prompt_template.format(
        DATABASE_SCHEMAS=schema_text,
        QUESTION=question,
        HINT=hint,
        DATASET_NAME=dataset_name,
        SQL_DIALECT=sql_dialect,
    )


def normalize_sql_response(response_text: str) -> str:
    text = response_text.strip()
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()

    fenced_match = re.search(r"```(?:sql)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced_match is not None:
        text = fenced_match.group(1).strip()
    else:
        text = text.replace("```", "").strip()

    try:
        response_json = json.loads(text)
    except json.JSONDecodeError:
        response_json = None
    if isinstance(response_json, Mapping):
        sql_value = response_json.get("sql")
        if isinstance(sql_value, str):
            text = sql_value.strip()

    return re.sub(r"^\s*SQL\s*:\s*", "", text, flags=re.IGNORECASE).strip()


def configure_fixed_hf_model(context_length: int | None, max_new_tokens: int) -> int:
    resolved_context_length = (
        context_length
        if context_length is not None
        else default_context_length_for_model(DEFAULT_SQL_LLM_NAME)
    )

    os.environ["HF_MODEL_NAME"] = DEFAULT_SQL_LLM_NAME
    os.environ["HF_CONTEXT_LENGTH"] = str(resolved_context_length)
    os.environ["HF_MAX_NEW_TOKENS"] = str(max_new_tokens)
    os.environ["HF_DO_SAMPLE"] = "false"
    os.environ["HF_TEMPERATURE"] = "0.0"
    os.environ["HF_TOP_P"] = "1.0"
    return resolved_context_length


def load_dataset_index(dataset_name: str) -> dict[str, dict[str, Any]]:
    data = load_dataset_data(dataset_name)
    if isinstance(data, dict):
        rows = data.items()
    elif isinstance(data, list):
        rows = ((str(row.get("id") or row.get("instance_id")), row) for row in data if isinstance(row, dict))
    else:
        raise ValueError(f"Unsupported dataset data format in {get_dataset_file(dataset_name)}")

    index: dict[str, dict[str, Any]] = {}
    for key, row in rows:
        if not isinstance(row, dict):
            continue
        instance_id = str(row.get("id") or row.get("instance_id") or key).strip()
        if not instance_id:
            continue
        record = dict(row)
        record.setdefault("id", instance_id)
        index[instance_id] = record
    return index


def resolve_instance_ids(
    *,
    explicit_ids: Sequence[str] | None,
    dataset_index: Mapping[str, Mapping[str, Any]],
    schema_source: Mapping[str, Any],
    schema_prompts_dir: Path,
    start_index: int,
    limit: int | None,
) -> list[str]:
    if explicit_ids:
        return [str(instance_id).strip() for instance_id in explicit_ids if str(instance_id).strip()]

    if schema_source:
        source_ids = list(schema_source.keys())
    else:
        source_ids = sorted(path.stem for path in schema_prompts_dir.glob("*.txt"))

    source_id_set = set(source_ids)
    ordered_ids = [instance_id for instance_id in dataset_index.keys() if instance_id in source_id_set]
    ordered_ids.extend(instance_id for instance_id in source_ids if instance_id not in dataset_index)

    selected_ids = ordered_ids[max(0, start_index):]
    if limit is not None:
        selected_ids = selected_ids[: max(0, limit)]
    return selected_ids


def load_external_knowledge(dataset_name: str, source_row: Mapping[str, Any]) -> str:
    external_knowledge = source_row.get("external_knowledge")
    if not external_knowledge:
        return ""

    if isinstance(external_knowledge, str):
        knowledge_path = resolve_external_knowledge_path(dataset_name, external_knowledge)
        if knowledge_path:
            return Path(knowledge_path).read_text(encoding="utf-8").strip()
        return external_knowledge.strip()

    return json.dumps(external_knowledge, ensure_ascii=False)


def resolve_hint(schema_text: str, external_knowledge: str) -> str:
    if "External knowledge that might be helpful:" in schema_text:
        return "No hint"
    return external_knowledge if external_knowledge.strip() else "No hint"


def build_schema_linking_metadata(schema_entry: Any) -> dict[str, Any]:
    if not isinstance(schema_entry, Mapping):
        return {}
    return {
        "table_candidates": schema_entry.get("table_candidates", []),
        "column_candidates": schema_entry.get("column_candidates", []),
    }


def usage_field(response_data: Mapping[str, Any], key: str) -> int:
    usage = response_data.get("usage")
    if not isinstance(usage, Mapping):
        return 0
    value = usage.get(key)
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def write_result_file(
    output_path: Path,
    run_info: dict[str, Any],
    result_records: list[dict[str, Any]],
) -> None:
    output_path.write_text(
        json.dumps(
            {
                "run_info": run_info,
                "results": result_records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def load_existing_results(output_path: Path, overwrite: bool) -> list[dict[str, Any]]:
    if overwrite or not output_path.is_file():
        return []

    value = load_json(output_path)
    if not isinstance(value, Mapping):
        return []
    results = value.get("results", [])
    if not isinstance(results, list):
        return []
    return [record for record in results if isinstance(record, dict)]


def build_failure_record(
    *,
    instance_id: str,
    source_row: Mapping[str, Any],
    schema_entry: Any,
    schema_prompt_path: Path,
    error_message: str,
    elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "id": instance_id,
        "question": source_row.get("question", ""),
        "db_name": source_row.get("db_name"),
        "gold_db_id": source_row.get("db_id"),
        "schema_prompt_path": str(schema_prompt_path),
        "schema_linking": build_schema_linking_metadata(schema_entry),
        "predict_sql": "",
        "status": "failed",
        "error": error_message,
        "efficiency": {
            "elapsed_seconds": round(max(0.0, elapsed_seconds), 6),
            "llm_total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        },
    }


def generate_for_instance(
    *,
    instance_id: str,
    source_row: Mapping[str, Any],
    schema_entry: Any,
    schema_prompt_path: Path,
    prompt_template: str,
    dataset_name: str,
    sql_dialect: str,
    cost_output_path: Path,
) -> dict[str, Any]:
    start_time = time.perf_counter()
    if not schema_prompt_path.is_file():
        return build_failure_record(
            instance_id=instance_id,
            source_row=source_row,
            schema_entry=schema_entry,
            schema_prompt_path=schema_prompt_path,
            error_message=f"Missing final schema prompt: {schema_prompt_path}",
            elapsed_seconds=time.perf_counter() - start_time,
        )

    question = str(source_row.get("question") or "").strip()
    if not question:
        return build_failure_record(
            instance_id=instance_id,
            source_row=source_row,
            schema_entry=schema_entry,
            schema_prompt_path=schema_prompt_path,
            error_message="Missing question in dataset row.",
            elapsed_seconds=time.perf_counter() - start_time,
        )

    schema_text = schema_prompt_path.read_text(encoding="utf-8").strip()
    external_knowledge = load_external_knowledge(dataset_name, source_row)
    hint = resolve_hint(schema_text, external_knowledge)
    prompt = render_prompt(
        prompt_template,
        schema_text=schema_text,
        question=question,
        hint=hint,
        dataset_name=dataset_name,
        sql_dialect=sql_dialect,
    )

    with SampleCostRecorder(sample_id=instance_id, output_path=str(cost_output_path)) as cost_recorder:
        try:
            response_text, response_data = chat_with_model([{"role": "user", "content": prompt}])
            cost_recorder.add_response_usage(response_data)
        except Exception as exc:
            elapsed_seconds = time.perf_counter() - start_time
            return build_failure_record(
                instance_id=instance_id,
                source_row=source_row,
                schema_entry=schema_entry,
                schema_prompt_path=schema_prompt_path,
                error_message=f"Model call failed: {exc}",
                elapsed_seconds=elapsed_seconds,
            )

    normalized_sql = normalize_sql_response(response_text)
    response_data = response_data if isinstance(response_data, Mapping) else {}
    elapsed_seconds = time.perf_counter() - start_time
    result_record: dict[str, Any] = {
        "id": instance_id,
        "question": question,
        "db_name": source_row.get("db_name"),
        "gold_db_id": source_row.get("db_id"),
        "schema_prompt_path": str(schema_prompt_path),
        "schema_linking": build_schema_linking_metadata(schema_entry),
        "predict_sql": normalized_sql,
        "status": "success" if normalized_sql else "empty",
        "efficiency": {
            "elapsed_seconds": round(max(0.0, elapsed_seconds), 6),
            "llm_total_tokens": extract_token_count(response_data),
            "prompt_tokens": usage_field(response_data, "prompt_tokens"),
            "completion_tokens": usage_field(response_data, "completion_tokens"),
        },
    }
    if hint != "No hint":
        result_record["hint"] = hint
    if response_text.strip() and response_text.strip() != normalized_sql:
        result_record["raw_response"] = response_text
    return result_record


def run_sql_generation(args: argparse.Namespace) -> Path:
    dataset_name = resolve_dataset_name(args)
    log_path = resolve_log_path(args, dataset_name)
    schema_prompts_dir = args.schema_prompts_dir or (log_path / "final_schema_prompts")
    schema_prompts_dir = schema_prompts_dir.resolve()
    if not schema_prompts_dir.is_dir():
        raise FileNotFoundError(f"Schema prompts directory not found: {schema_prompts_dir}")

    prompt_path = args.prompt_path.resolve()
    if not prompt_path.is_file():
        raise FileNotFoundError(f"SQL prompt template not found: {prompt_path}")

    context_length = configure_fixed_hf_model(
        context_length=args.hf_context_length,
        max_new_tokens=args.hf_max_new_tokens,
    )
    output_path = resolve_output_path(args.output_path, log_path, dataset_name)
    setup_progress_logging(str(output_path.parent))
    logger = get_progress_logger(__name__)

    schema_source_path = log_path / args.schema_source_file
    schema_source = load_json_mapping(schema_source_path) if schema_source_path.is_file() else {}
    dataset_index = load_dataset_index(dataset_name)
    prompt_template = prompt_path.read_text(encoding="utf-8").strip()
    sql_dialect = args.sql_dialect or default_sql_dialect_requirements(dataset_name)
    result_records = load_existing_results(output_path, overwrite=args.overwrite)
    completed_ids = {
        str(record.get("id"))
        for record in result_records
        if record.get("id") is not None and record.get("status") in {"success", "empty", "failed"}
    }
    instance_ids = resolve_instance_ids(
        explicit_ids=args.instance_id,
        dataset_index=dataset_index,
        schema_source=schema_source,
        schema_prompts_dir=schema_prompts_dir,
        start_index=args.start_index,
        limit=args.limit,
    )
    run_info = {
        "task": "sql_generation",
        "dataset_name": dataset_name,
        "model": DEFAULT_SQL_LLM_NAME,
        "schema_source_model": args.schema_model_name,
        "schema_log_path": str(log_path),
        "schema_source_path": str(schema_source_path) if schema_source_path.is_file() else None,
        "schema_prompts_dir": str(schema_prompts_dir),
        "prompt_template": str(prompt_path),
        "dataset_path": get_dataset_file(dataset_name),
        "documents_dir": get_documents_dir(dataset_name),
        "sql_dialect": sql_dialect,
        "hf_context_length": context_length,
        "hf_max_new_tokens": args.hf_max_new_tokens,
        "start_index": args.start_index,
        "limit": args.limit,
        "instance_ids": args.instance_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    cost_output_path = output_path.parent / "sql_generation_cost.json"

    logger.info(
        "SQL generation started | dataset=%s model=%s samples=%s log_path=%s output=%s",
        dataset_name,
        DEFAULT_SQL_LLM_NAME,
        len(instance_ids),
        log_path,
        output_path,
    )
    write_result_file(output_path, run_info, result_records)

    for instance_id in progress_bar(instance_ids, desc="sql generation", unit="sample"):
        if instance_id in completed_ids:
            continue

        source_row = dataset_index.get(instance_id, {"id": instance_id})
        schema_entry = schema_source.get(instance_id, {})
        schema_prompt_path = schema_prompts_dir / f"{instance_id}.txt"
        result_record = generate_for_instance(
            instance_id=instance_id,
            source_row=source_row,
            schema_entry=schema_entry,
            schema_prompt_path=schema_prompt_path,
            prompt_template=prompt_template,
            dataset_name=dataset_name,
            sql_dialect=sql_dialect,
            cost_output_path=cost_output_path,
        )
        result_records.append(result_record)
        completed_ids.add(instance_id)
        write_result_file(output_path, run_info, result_records)

    logger.info("SQL generation finished | output=%s records=%s", output_path, len(result_records))
    return output_path


def main() -> None:
    args = parse_args()
    try:
        run_sql_generation(args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
