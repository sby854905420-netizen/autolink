from __future__ import annotations

import argparse
import json
import math
import os
import re
import sqlite3
import sys
from collections import Counter
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from utils import (
    get_credential_file,
    get_dataset_data_dir,
    get_execution_backend,
    get_mmqa_data_dir,
    load_dataset_data,
    require_supported_dataset,
)


MMQA_SQLITE_DIR = os.path.join(get_mmqa_data_dir(), "Sqlite_database")
ORDER_BY_PATTERN = re.compile(r"\border\s+by\b", flags=re.IGNORECASE)
QUALIFIED_TABLE_PATTERN = re.compile(r'"([A-Za-z0-9_]+)\.([A-Za-z0-9_]+)"')
QUOTED_DB_TABLE_PATTERN = re.compile(r'"([A-Za-z0-9_]+)"\s*\.\s*"([A-Za-z0-9_]+)"')
QUOTED_DB_UNQUOTED_TABLE_PATTERN = re.compile(r'"([A-Za-z0-9_]+)"\s*\.\s*([A-Za-z0-9_]+)\b')
QUOTED_PRAGMA_PATTERN = re.compile(r'"([A-Za-z0-9_]+)"\s*\.\s*pragma_table_info\s*\(', re.IGNORECASE)
UNQUOTED_DB_TABLE_PATTERN = re.compile(r'(?<![\w"])\b([A-Za-z0-9_]+)\.([A-Za-z0-9_]+)\b')
WRITE_OR_SESSION_KEYWORDS = re.compile(
    r"\b("
    r"INSERT|UPDATE|DELETE|MERGE|CREATE|ALTER|DROP|TRUNCATE|GRANT|REVOKE|"
    r"COPY|PUT|REMOVE|CALL|USE"
    r")\b",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Execution Accuracy (EX) for generated SQL results."
    )
    parser.add_argument("dataset", nargs="?", default=None)
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument(
        "--sql_results",
        type=Path,
        default=None,
        help="Path to sql_generation_*.json. If omitted, --log_path is required.",
    )
    parser.add_argument(
        "--log_path",
        type=Path,
        default=None,
        help="AutoLink run directory; the latest sql_results/sql_generation_*.json is used.",
    )
    parser.add_argument("--output_path", type=Path, default=None)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--instance_id",
        action="append",
        default=None,
        help="Evaluate one instance id. Can be supplied multiple times.",
    )
    parser.add_argument(
        "--keep_records",
        action="store_true",
        help="Deprecated; per-instance EX records are always written.",
    )
    return parser.parse_args()


def resolve_dataset_name(args: argparse.Namespace) -> str:
    return require_supported_dataset(
        args.dataset_name or args.dataset or os.environ.get("DATASET_NAME") or "MMQA"
    )


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_sql_results_path(args: argparse.Namespace) -> Path:
    if args.sql_results is not None:
        path = args.sql_results.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"SQL results file not found: {path}")
        return path

    if args.log_path is None:
        raise ValueError("Provide --sql_results or --log_path.")

    sql_dir = args.log_path.resolve() / "sql_results"
    if not sql_dir.is_dir():
        raise FileNotFoundError(f"SQL results directory not found: {sql_dir}")

    candidates = sorted(sql_dir.glob("sql_generation_*.json"), key=lambda path: path.name, reverse=True)
    candidates = [path for path in candidates if path.name != "sql_generation_cost.json"]
    if not candidates:
        raise FileNotFoundError(f"No sql_generation_*.json files found under {sql_dir}")
    return candidates[0]


def resolve_output_path(output_path: Path | None, sql_results_path: Path) -> Path:
    if output_path is not None:
        resolved = output_path.resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        return resolved
    return sql_results_path.with_name(sql_results_path.stem + "_ex.json")


def normalize_mmqa_id(instance_id: Any) -> str:
    text = str(instance_id).strip()
    if text.startswith("mmqa_"):
        return text
    return f"mmqa_{text}"


def gold_sql_key(dataset_name: str, instance_id: str) -> str:
    if dataset_name.lower() == "mmqa" and instance_id.startswith("mmqa_"):
        return instance_id.removeprefix("mmqa_")
    return instance_id


def load_gold_sql(dataset_name: str) -> dict[str, str]:
    path = Path(get_dataset_data_dir(dataset_name)) / "gold_sql.json"
    value = load_json(path)
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected JSON object in {path}")
    return {str(key): str(sql) for key, sql in value.items()}


def load_dataset_index(dataset_name: str) -> dict[str, dict[str, Any]]:
    data = load_dataset_data(dataset_name)
    if not isinstance(data, Mapping):
        raise ValueError(f"Expected normalized dataset data to be a mapping for {dataset_name}")
    return {str(key): dict(value) for key, value in data.items() if isinstance(value, Mapping)}


def load_prediction_records(sql_results_path: Path, dataset_name: str) -> dict[str, dict[str, Any]]:
    value = load_json(sql_results_path)

    if isinstance(value, Mapping) and isinstance(value.get("results"), list):
        raw_records = value["results"]
    elif isinstance(value, list):
        raw_records = value
    elif isinstance(value, Mapping):
        raw_records = []
        for key, item in value.items():
            if isinstance(item, Mapping):
                record = dict(item)
                record.setdefault("id", key)
                raw_records.append(record)
            elif isinstance(item, str):
                raw_records.append({"id": key, "predict_sql": item, "status": "success"})
    else:
        raise ValueError(f"Unsupported SQL results format in {sql_results_path}")

    records: dict[str, dict[str, Any]] = {}
    for record in raw_records:
        if not isinstance(record, Mapping):
            continue
        instance_id = str(record.get("id") or "").strip()
        if not instance_id:
            continue
        if dataset_name.lower() == "mmqa":
            instance_id = normalize_mmqa_id(instance_id)
        records[instance_id] = dict(record)
    return records


def model_from_log_path(path_value: Any, dataset_name: str) -> str | None:
    if not isinstance(path_value, str) or not path_value.strip():
        return None

    parts = Path(path_value).parts
    for index, part in enumerate(parts):
        if part == "Log" and index + 3 < len(parts) and parts[index + 1] == dataset_name:
            return parts[index + 2]
    return None


def infer_schema_linking_model(sql_results_path: Path, dataset_name: str) -> str:
    path_model = model_from_log_path(str(sql_results_path), dataset_name)
    if path_model:
        return path_model

    value = load_json(sql_results_path)
    run_info = value.get("run_info", {}) if isinstance(value, Mapping) else {}
    if isinstance(run_info, Mapping):
        for key in ("schema_log_path", "schema_source_path"):
            model = model_from_log_path(run_info.get(key), dataset_name)
            if model:
                return model
        schema_source_model = run_info.get("schema_source_model")
        if isinstance(schema_source_model, str) and schema_source_model.strip():
            return schema_source_model.strip()

    return "unknown"


def selected_instance_ids(
    dataset_index: Mapping[str, Any],
    prediction_records: Mapping[str, Any],
    explicit_ids: Sequence[str] | None,
    start_index: int,
    limit: int | None,
) -> list[str]:
    if explicit_ids:
        ids = [normalize_mmqa_id(item) if item.isdigit() else str(item).strip() for item in explicit_ids]
    else:
        ids = [instance_id for instance_id in dataset_index.keys() if instance_id in prediction_records]

    ids = ids[max(0, start_index) :]
    if limit is not None:
        ids = ids[: max(0, limit)]
    return ids


def has_order_by(sql: str) -> bool:
    return ORDER_BY_PATTERN.search(sql or "") is not None


def get_available_sqlite_db_ids() -> set[str]:
    return {
        os.path.splitext(filename)[0]
        for filename in os.listdir(MMQA_SQLITE_DIR)
        if filename.endswith(".sqlite")
    }


def normalize_attached_table_references(sql: str) -> str:
    return QUALIFIED_TABLE_PATTERN.sub(r'"\1"."\2"', sql)


def extract_referenced_db_ids(sql: str) -> list[str]:
    available_db_ids = get_available_sqlite_db_ids()
    db_ids = set()

    for pattern in (
        QUOTED_DB_TABLE_PATTERN,
        QUOTED_DB_UNQUOTED_TABLE_PATTERN,
        QUOTED_PRAGMA_PATTERN,
        UNQUOTED_DB_TABLE_PATTERN,
    ):
        for match in pattern.finditer(sql):
            db_id = match.group(1)
            if db_id in available_db_ids:
                db_ids.add(db_id)

    return sorted(db_ids)


def strip_leading_sql_comments(sql: str) -> str:
    text = sql.strip()
    while True:
        if text.startswith("--"):
            _, _, text = text.partition("\n")
            text = text.strip()
            continue
        if text.startswith("/*"):
            end = text.find("*/")
            if end == -1:
                return ""
            text = text[end + 2 :].strip()
            continue
        return text


def validate_sql(sql: str) -> str:
    statement = strip_leading_sql_comments(sql)
    if not statement:
        raise ValueError("Empty SQL query.")

    statement_without_trailing_semicolon = statement.rstrip().rstrip(";").strip()
    if ";" in statement_without_trailing_semicolon:
        raise ValueError("Only one SQL statement is allowed.")

    first_keyword_match = re.match(r"([A-Za-z]+)", statement_without_trailing_semicolon)
    first_keyword = first_keyword_match.group(1).upper() if first_keyword_match else ""
    if first_keyword not in {"SELECT", "WITH"}:
        raise ValueError("Only read-only SELECT/WITH queries are allowed.")

    if WRITE_OR_SESSION_KEYWORDS.search(statement_without_trailing_semicolon):
        raise ValueError("Write, DDL, session, and administrative statements are not allowed.")

    return statement_without_trailing_semicolon


def load_snowflake_credentials(dataset_name: str) -> dict[str, Any]:
    credential_file = get_credential_file(dataset_name)
    if not credential_file or not os.path.isfile(credential_file):
        return {}

    with open(credential_file, "r", encoding="utf-8") as f:
        return json.load(f)


def snowflake_connection_params(dataset_name: str) -> dict[str, Any]:
    credentials = load_snowflake_credentials(dataset_name)
    user = os.environ.get("SNOWFLAKE_USER") or credentials.get("user") or credentials.get("username")
    password = (
        os.environ.get("SNOWFLAKE_PAT")
        or os.environ.get("SNOWFLAKE_PASSWORD")
        or credentials.get("password")
        or credentials.get("pat")
        or credentials.get("token")
    )
    account = os.environ.get("SNOWFLAKE_ACCOUNT") or credentials.get("account")
    warehouse = os.environ.get("SNOWFLAKE_WAREHOUSE") or credentials.get("warehouse")
    role = os.environ.get("SNOWFLAKE_ROLE") or credentials.get("role")

    missing = [
        name
        for name, value in {
            "account": account,
            "user": user,
            "password/PAT": password,
            "warehouse": warehouse,
            "role": role,
        }.items()
        if not value
    ]
    if missing:
        raise ValueError(
            "Missing Snowflake credential fields: "
            + ", ".join(missing)
            + ". Provide them in snowflake_credential.json or SNOWFLAKE_* environment variables."
        )

    params: dict[str, Any] = {
        "account": account,
        "user": user,
        "password": password,
        "warehouse": warehouse,
        "role": role,
        "login_timeout": int(os.environ.get("SNOWFLAKE_LOGIN_TIMEOUT", "30")),
        "network_timeout": int(os.environ.get("SNOWFLAKE_NETWORK_TIMEOUT", "120")),
        "client_session_keep_alive": False,
        "session_parameters": {
            "QUERY_TAG": os.environ.get("SNOWFLAKE_QUERY_TAG", "autolink_ex_evaluation"),
            "STATEMENT_TIMEOUT_IN_SECONDS": int(os.environ.get("SNOWFLAKE_STATEMENT_TIMEOUT", "120")),
        },
    }

    for optional_key in ("database", "schema"):
        value = os.environ.get(f"SNOWFLAKE_{optional_key.upper()}") or credentials.get(optional_key)
        if value:
            params[optional_key] = value

    return params


def sqlite_db_path(db_id: str) -> Path:
    return Path(MMQA_SQLITE_DIR) / f"{db_id}.sqlite"


def sqlite_fetch(sql: str, gold_db_id: str) -> list[tuple[Any, ...]]:
    statement = validate_sql(normalize_attached_table_references(sql))
    referenced_db_ids = extract_referenced_db_ids(statement)

    if referenced_db_ids:
        if len(referenced_db_ids) > 10:
            raise RuntimeError(f"SQLite query references {len(referenced_db_ids)} databases; attach limit is 10.")
        conn = sqlite3.connect(":memory:")
        try:
            for db_id in referenced_db_ids:
                db_path = sqlite_db_path(db_id)
                if not db_path.is_file():
                    raise FileNotFoundError(f"SQLite database not found: {db_path}")
                conn.execute(f'ATTACH DATABASE ? AS "{db_id}"', (str(db_path),))
            cursor = conn.execute(statement)
            return list(cursor.fetchall())
        finally:
            conn.close()

    db_path = sqlite_db_path(gold_db_id)
    if not db_path.is_file():
        raise FileNotFoundError(f"Gold SQLite database not found: {db_path}")
    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.execute(statement)
        return list(cursor.fetchall())
    finally:
        conn.close()


class SnowflakeExecutor:
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.conn = None

    def __enter__(self) -> "SnowflakeExecutor":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def connect(self):
        if self.conn is None:
            import snowflake.connector

            self.conn = snowflake.connector.connect(**snowflake_connection_params(self.dataset_name))
        return self.conn

    def fetch(self, sql: str) -> list[tuple[Any, ...]]:
        statement = validate_sql(sql)
        cursor = None
        try:
            cursor = self.connect().cursor()
            cursor.execute(statement)
            return list(cursor.fetchall())
        finally:
            if cursor is not None:
                cursor.close()


def normalize_number(value: Any) -> Any:
    try:
        decimal_value = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return str(value)

    if decimal_value.is_nan():
        return ("nan",)
    if decimal_value == decimal_value.to_integral_value():
        return int(decimal_value)
    return ("decimal", format(decimal_value.normalize(), "f"))


def normalize_cell(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float, Decimal)):
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        return normalize_number(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value.hex()
    return str(value)


def normalize_rows(rows: Iterable[Sequence[Any]]) -> list[tuple[Any, ...]]:
    return [tuple(normalize_cell(cell) for cell in row) for row in rows]


def equivalent_results(gold_rows: list[tuple[Any, ...]], pred_rows: list[tuple[Any, ...]], ordered: bool) -> bool:
    normalized_gold = normalize_rows(gold_rows)
    normalized_pred = normalize_rows(pred_rows)
    if ordered:
        return normalized_pred == normalized_gold
    return Counter(normalized_pred) == Counter(normalized_gold)


def execute_query(
    *,
    sql: str,
    dataset_name: str,
    gold_db_id: str,
    snowflake_executor: SnowflakeExecutor | None,
) -> list[tuple[Any, ...]]:
    backend = get_execution_backend(dataset_name)
    if backend == "sqlite":
        return sqlite_fetch(sql, gold_db_id)
    if backend == "snowflake":
        if snowflake_executor is None:
            raise RuntimeError("Snowflake executor is not initialized.")
        return snowflake_executor.fetch(sql)
    raise ValueError(f"Unsupported execution backend for {dataset_name}: {backend}")


def evaluate_instance(
    *,
    instance_id: str,
    dataset_name: str,
    dataset_index: Mapping[str, dict[str, Any]],
    gold_sql_by_id: Mapping[str, str],
    prediction_records: Mapping[str, dict[str, Any]],
    snowflake_executor: SnowflakeExecutor | None,
) -> dict[str, Any]:
    dataset_row = dataset_index.get(instance_id, {})
    gold_db_id = str(dataset_row.get("db_id") or dataset_row.get("gold_db_id") or "").strip()
    gold_sql = gold_sql_by_id.get(gold_sql_key(dataset_name, instance_id))
    pred_record = prediction_records.get(instance_id, {})
    pred_sql = str(pred_record.get("predict_sql") or "").strip()

    result = {
        "id": instance_id,
        "gold_db_id": gold_db_id,
        "ordered": bool(has_order_by(gold_sql or "")),
        "correct": 0,
        "status": "incorrect",
        "gold_row_count": None,
        "pred_row_count": None,
    }

    if not gold_sql:
        result["status"] = "missing_gold_sql"
        return result
    if not pred_sql:
        result["status"] = "missing_pred_sql"
        result["pred_generation_status"] = pred_record.get("status")
        return result

    try:
        gold_rows = execute_query(
            sql=gold_sql,
            dataset_name=dataset_name,
            gold_db_id=gold_db_id,
            snowflake_executor=snowflake_executor,
        )
        result["gold_row_count"] = len(gold_rows)
    except Exception as exc:
        result["status"] = "gold_execution_error"
        result["error"] = str(exc)
        return result

    try:
        pred_rows = execute_query(
            sql=pred_sql,
            dataset_name=dataset_name,
            gold_db_id=gold_db_id,
            snowflake_executor=snowflake_executor,
        )
        result["pred_row_count"] = len(pred_rows)
    except Exception as exc:
        result["status"] = "pred_execution_error"
        result["error"] = str(exc)
        return result

    if equivalent_results(gold_rows, pred_rows, ordered=result["ordered"]):
        result["correct"] = 1
        result["status"] = "correct"
    return result


def compact_error(record: Mapping[str, Any]) -> str:
    status = str(record.get("status") or "incorrect")
    detail = record.get("error")
    if detail:
        return f"{status}: {detail}"
    if status == "incorrect":
        return (
            "result_mismatch"
            f"; gold_row_count={record.get('gold_row_count')}"
            f"; pred_row_count={record.get('pred_row_count')}"
            f"; ordered={record.get('ordered')}"
        )
    return status


def compact_sample_record(record: Mapping[str, Any]) -> dict[str, Any]:
    compact = {
        "id": record.get("id"),
        "EX": int(record.get("correct") or 0),
    }
    if compact["EX"] != 1:
        compact["error"] = compact_error(record)
    return compact


def evaluate_ex(args: argparse.Namespace) -> Path:
    dataset_name = resolve_dataset_name(args)
    sql_results_path = resolve_sql_results_path(args)
    output_path = resolve_output_path(args.output_path, sql_results_path)

    dataset_index = load_dataset_index(dataset_name)
    gold_sql_by_id = load_gold_sql(dataset_name)
    prediction_records = load_prediction_records(sql_results_path, dataset_name)
    instance_ids = selected_instance_ids(
        dataset_index=dataset_index,
        prediction_records=prediction_records,
        explicit_ids=args.instance_id,
        start_index=args.start_index,
        limit=args.limit,
    )

    records: list[dict[str, Any]] = []
    snowflake_context = (
        SnowflakeExecutor(dataset_name)
        if get_execution_backend(dataset_name) == "snowflake"
        else None
    )

    try:
        context_manager = snowflake_context if snowflake_context is not None else _NullContext()
        with context_manager as snowflake_executor:
            for index, instance_id in enumerate(instance_ids, start=1):
                record = evaluate_instance(
                    instance_id=instance_id,
                    dataset_name=dataset_name,
                    dataset_index=dataset_index,
                    gold_sql_by_id=gold_sql_by_id,
                    prediction_records=prediction_records,
                    snowflake_executor=snowflake_executor,
                )
                records.append(record)
                if index % 25 == 0:
                    print(f"evaluated {index}/{len(instance_ids)}", file=sys.stderr)
    finally:
        if snowflake_context is not None:
            snowflake_context.close()

    total = len(records)
    correct = sum(int(record.get("correct") or 0) for record in records)
    output = {
        "num_samples": total,
        "schema_linking_model": infer_schema_linking_model(sql_results_path, dataset_name),
        "EX": correct / total if total else 0.0,
        "samples": [compact_sample_record(record) for record in records],
    }

    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "num_samples": output["num_samples"],
                "schema_linking_model": output["schema_linking_model"],
                "EX": output["EX"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return output_path


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return None


def main() -> None:
    args = parse_args()
    try:
        output_path = evaluate_ex(args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
    print(f"EX evaluation written to {output_path}")


if __name__ == "__main__":
    main()
