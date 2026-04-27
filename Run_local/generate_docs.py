import json
import os

from utils import get_mmqa_data_dir


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
MMQA_SCHEMA_PATH = os.path.join(get_mmqa_data_dir(), "db_info.json")
GLOBAL_DB_NAME = "mmqa_global"


def qualify_table_name(db_id: str, table_name: str) -> str:
    return f"{db_id}.{table_name}"


def generate_documents(output_path: str = "documents"):
    with open(MMQA_SCHEMA_PATH, "r", encoding="utf-8") as f:
        mmqa_schemas = json.load(f)

    documents = {GLOBAL_DB_NAME: {}}

    for schema in mmqa_schemas:
        db_id = schema["db_id"]
        table_names = schema["table_names"]
        column_names = schema["column_names"]
        column_types = schema["column_types"]
        column_descriptions = schema["column_descriptions"]
        sample_rows = schema["sample_rows"]

        tables = {}
        for table_name in table_names:
            qualified_table_name = qualify_table_name(db_id, table_name)
            tables[table_name] = {
                "qualified_table_name": qualified_table_name,
                "columns": [],
                "column_types": [],
                "descriptions": [],
                "sample_rows": sample_rows.get(table_name, []),
            }

        for idx in range(1, len(column_names)):
            table_idx, column_name = column_names[idx]
            if table_idx < 0:
                continue

            table_name = table_names[table_idx]
            tables[table_name]["columns"].append(column_name)
            tables[table_name]["column_types"].append(column_types[idx])

            column_desc = column_descriptions[idx]
            tables[table_name]["descriptions"].append("" if column_desc is None else column_desc)

        for table_name, table_info in tables.items():
            qualified_table_name = table_info["qualified_table_name"]
            documents[GLOBAL_DB_NAME][qualified_table_name] = {
                "similar_tables": [],
                "columns": {},
                "column_types": table_info["column_types"],
                "sample_values": [],
            }

            for column_name in table_info["columns"]:
                column_values = []
                for sample_row in table_info["sample_rows"]:
                    column_values.append(str(sample_row.get(column_name, "")))
                documents[GLOBAL_DB_NAME][qualified_table_name]["sample_values"].append(column_values)

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
                documents[GLOBAL_DB_NAME][qualified_table_name]["columns"][column_name] = desc

    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, "localdb.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(documents, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    print("Generate documents for MMQA global SQLite space...")
    generate_documents(output_path="documents")
