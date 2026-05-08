import argparse
import json
import os
from typing import Optional

from utils import (
    DEFAULT_DATASET_NAME,
    ensure_dir,
    get_dataset_config,
    get_documents_dir,
    require_schema_file,
)

def get_document_db_name(schema: dict, dataset_name: str) -> str:
    config = get_dataset_config(dataset_name)
    return config.get("global_document_db_name") or schema["db_id"]


def qualify_table_name(db_id: str, table_name: str, dataset_name: str) -> str:
    config = get_dataset_config(dataset_name)
    if not config.get("qualify_table_names", True):
        return table_name
    if table_name.startswith(f"{db_id}."):
        return table_name
    return f"{db_id}.{table_name}"


def generate_documents(dataset_name: str = DEFAULT_DATASET_NAME, output_path: Optional[str] = None):
    schema_path = require_schema_file(dataset_name)
    if output_path is None:
        output_path = get_documents_dir(dataset_name)

    with open(schema_path, "r", encoding="utf-8") as f:
        schemas = json.load(f)

    documents = {}

    for schema in schemas:
        db_id = schema["db_id"]
        document_db_name = get_document_db_name(schema, dataset_name)
        db_documents = documents.setdefault(document_db_name, {})

        table_names = schema["table_names"]
        column_names = schema["column_names"]
        column_types = schema["column_types"]
        column_descriptions = schema["column_descriptions"]
        sample_rows = schema.get("sample_rows") or {}

        tables = {}
        for table_name in table_names:
            qualified_table_name = qualify_table_name(db_id, table_name, dataset_name)
            tables[table_name] = {
                "qualified_table_name": qualified_table_name,
                "columns": [],
                "column_types": [],
                "descriptions": [],
                "sample_rows": sample_rows.get(table_name, []),
            }

        for idx, (table_idx, column_name) in enumerate(column_names):
            if table_idx < 0 or table_idx >= len(table_names):
                continue

            table_name = table_names[table_idx]
            tables[table_name]["columns"].append(column_name)
            tables[table_name]["column_types"].append(column_types[idx] if idx < len(column_types) else "")

            column_desc = column_descriptions[idx] if idx < len(column_descriptions) else ""
            tables[table_name]["descriptions"].append("" if column_desc is None else column_desc)

        for table_name, table_info in tables.items():
            qualified_table_name = table_info["qualified_table_name"]
            db_documents[qualified_table_name] = {
                "similar_tables": [],
                "columns": {},
                "column_types": table_info["column_types"],
                "sample_values": [],
            }

            for column_name in table_info["columns"]:
                column_values = []
                for sample_row in table_info["sample_rows"]:
                    value = sample_row.get(column_name, "") if isinstance(sample_row, dict) else ""
                    column_values.append(str(value))
                db_documents[qualified_table_name]["sample_values"].append(column_values)

            for column_name, column_type, column_desc in zip(
                table_info["columns"],
                table_info["column_types"],
                table_info["descriptions"],
            ):
                desc = (
                    "column name: " + column_name + "\n"
                    + "column type: " + column_type + "\n"
                    + "table name: " + qualified_table_name + "\n"
                    + "description: " + column_desc + "\n"
                )
                db_documents[qualified_table_name]["columns"][column_name] = desc

    ensure_dir(output_path)
    output_file = os.path.join(output_path, "localdb.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(documents, f, indent=4, ensure_ascii=False)

    print(f"Documents saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default=DEFAULT_DATASET_NAME)
    parser.add_argument("--output_path", type=str, default=None)
    args = parser.parse_args()

    print(f"Generate documents for {args.dataset_name}...")
    generate_documents(dataset_name=args.dataset_name, output_path=args.output_path)
