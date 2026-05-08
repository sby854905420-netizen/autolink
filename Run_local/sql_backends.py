import json
import os
import re
import sqlite3
import threading
from functools import lru_cache

import pandas as pd

from utils import (
    get_credential_file,
    get_execution_backend,
    get_mmqa_data_dir,
    is_dataset_instance,
)


MMQA_SQLITE_DIR = os.path.join(get_mmqa_data_dir(), "Sqlite_database")
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

sqlite_lock = threading.Lock()


@lru_cache(maxsize=1)
def get_available_sqlite_db_ids() -> set[str]:
    return {
        os.path.splitext(filename)[0]
        for filename in os.listdir(MMQA_SQLITE_DIR)
        if filename.endswith(".sqlite")
    }


def normalize_attached_table_references(sql: str) -> str:
    return QUALIFIED_TABLE_PATTERN.sub(r'"\1"."\2"', sql)


def extract_referenced_db_ids(sql: str):
    available_db_ids = get_available_sqlite_db_ids()
    db_ids = set()

    for match in QUOTED_DB_TABLE_PATTERN.finditer(sql):
        db_id = match.group(1)
        if db_id in available_db_ids:
            db_ids.add(db_id)

    for match in QUOTED_DB_UNQUOTED_TABLE_PATTERN.finditer(sql):
        db_id = match.group(1)
        if db_id in available_db_ids:
            db_ids.add(db_id)

    for match in QUOTED_PRAGMA_PATTERN.finditer(sql):
        db_id = match.group(1)
        if db_id in available_db_ids:
            db_ids.add(db_id)

    for match in UNQUOTED_DB_TABLE_PATTERN.finditer(sql):
        db_id = match.group(1)
        if db_id in available_db_ids:
            db_ids.add(db_id)

    return sorted(db_ids)


def build_sqlite_connection(db_ids):
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    for db_id in db_ids:
        db_path = os.path.join(MMQA_SQLITE_DIR, f"{db_id}.sqlite")
        conn.execute(f'ATTACH DATABASE ? AS "{db_id}"', (db_path,))
    return conn


def execute_sqlite_sql(sql: str):
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


def load_snowflake_credentials(dataset_name: str) -> dict:
    credential_file = get_credential_file(dataset_name)
    if not credential_file or not os.path.isfile(credential_file):
        return {}

    with open(credential_file, "r", encoding="utf-8") as f:
        return json.load(f)


def snowflake_connection_params(dataset_name: str) -> dict:
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

    params = {
        "account": account,
        "user": user,
        "password": password,
        "warehouse": warehouse,
        "role": role,
        "login_timeout": int(os.environ.get("SNOWFLAKE_LOGIN_TIMEOUT", "30")),
        "network_timeout": int(os.environ.get("SNOWFLAKE_NETWORK_TIMEOUT", "120")),
        "client_session_keep_alive": False,
        "session_parameters": {
            "QUERY_TAG": os.environ.get("SNOWFLAKE_QUERY_TAG", "autolink_schema_linking"),
            "STATEMENT_TIMEOUT_IN_SECONDS": int(os.environ.get("SNOWFLAKE_STATEMENT_TIMEOUT", "120")),
        },
    }

    for optional_key in ("database", "schema"):
        value = os.environ.get(f"SNOWFLAKE_{optional_key.upper()}") or credentials.get(optional_key)
        if value:
            params[optional_key] = value

    return params


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


def validate_readonly_query(sql: str):
    statement = strip_leading_sql_comments(sql)
    if not statement:
        raise ValueError("Empty SQL query.")

    statement_without_trailing_semicolon = statement.rstrip().rstrip(";").strip()
    if ";" in statement_without_trailing_semicolon:
        raise ValueError("Only one SQL statement is allowed.")

    first_keyword_match = re.match(r"([A-Za-z]+)", statement_without_trailing_semicolon)
    first_keyword = first_keyword_match.group(1).upper() if first_keyword_match else ""
    if first_keyword not in {"SELECT", "WITH"}:
        raise ValueError("Only read-only SELECT/WITH queries are allowed for Snowflake exploration.")

    if WRITE_OR_SESSION_KEYWORDS.search(statement_without_trailing_semicolon):
        raise ValueError("Write, DDL, session, and administrative statements are not allowed.")

    return statement_without_trailing_semicolon


def limited_snowflake_query(sql: str, max_rows: int) -> str:
    statement = validate_readonly_query(sql)
    return f"SELECT * FROM (\n{statement}\n) AS AUTOLINK_LIMITED_RESULT LIMIT {max_rows}"


def execute_snowflake_sql(sql: str, dataset_name: str):
    import snowflake.connector

    max_rows = int(os.environ.get("SNOWFLAKE_RESULT_MAX_ROWS", "20"))
    query = limited_snowflake_query(sql, max_rows=max_rows)
    conn = None
    cursor = None
    try:
        conn = snowflake.connector.connect(**snowflake_connection_params(dataset_name))
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchmany(max_rows)
        columns = [column[0] for column in cursor.description or []]
        df = pd.DataFrame(rows, columns=columns)
        if df.empty:
            return "empty", "No data found for the specified query."
        return "success", df
    except Exception as e:
        return "error", f"Error occurred while fetching data from Snowflake: {e}"
    finally:
        if cursor is not None:
            cursor.close()
        if conn is not None:
            conn.close()


def sql_execution(instance_id, sql, db_name, dataset_name):
    if not (instance_id.startswith("local") or is_dataset_instance(instance_id, dataset_name)):
        return "error", f"Unsupported instance_id for Run_local: {instance_id}"

    backend = get_execution_backend(dataset_name)
    if backend == "sqlite":
        return execute_sqlite_sql(sql)
    if backend == "snowflake":
        return execute_snowflake_sql(sql, dataset_name)
    return "error", f"Unsupported execution backend for {dataset_name}: {backend}"


def thread_safe_sql_execution(instance_id, sql, db_name, dataset_name):
    if get_execution_backend(dataset_name) == "sqlite":
        with sqlite_lock:
            return sql_execution(instance_id, sql, db_name, dataset_name)
    return sql_execution(instance_id, sql, db_name, dataset_name)
