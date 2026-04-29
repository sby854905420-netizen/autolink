import argparse
import json
import os

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from utils import (
    DEFAULT_DATASET_NAME,
    ensure_dir,
    get_documents_file,
    get_local_embedding_dir,
    require_file,
)


MODEL_NAME = os.environ.get("SENTENCE_TRANSFORMER_MODEL", "BAAI/bge-large-en-v1.5")


def embed_documents(input_file: str, embed_path: str, batch_size: int = 32):
    require_file(input_file, "Document file")
    ensure_dir(embed_path)

    model = SentenceTransformer(MODEL_NAME)

    with open(input_file, "r", encoding="utf-8") as f:
        documents = json.load(f)

    for db_name, tables in tqdm(documents.items()):
        db_dir = os.path.join(embed_path, db_name)
        os.makedirs(db_dir, exist_ok=True)

        all_descriptions = []
        metadata_mapping = []

        for table_name, table_info in tables.items():
            columns = table_info["columns"]
            column_types = table_info["column_types"]
            column_values = table_info["sample_values"]

            if len(columns) != len(column_types) or len(columns) != len(column_values):
                print(f"Warning: Length mismatch in table {table_name} of database {db_name}.")
                print(
                    f"Columns: {len(columns)}, Column Types: {len(column_types)}, "
                    f"Column Values: {len(column_values)}"
                )

            for (column_name, desc), column_type, column_value in zip(
                columns.items(), column_types, column_values
            ):
                all_descriptions.append(desc)
                metadata_mapping.append(
                    {
                        "table": table_name,
                        "column": column_name,
                        "column_type": column_type,
                        "column_value": column_value,
                        "description": desc,
                    }
                )

        db_embeddings = []
        for i in tqdm(range(0, len(all_descriptions), batch_size), desc=f"Embedding {db_name}", leave=False):
            batch_descriptions = all_descriptions[i:i + batch_size]
            batch_embeddings = model.encode(batch_descriptions, convert_to_numpy=True)
            db_embeddings.extend(batch_embeddings)

        if not db_embeddings:
            raise ValueError(f"No descriptions found for database {db_name}.")

        dimension = len(db_embeddings[0])
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(db_embeddings, dtype=np.float32))

        faiss.write_index(index, os.path.join(db_dir, "index.faiss"))
        with open(os.path.join(db_dir, "metadata.json"), "w", encoding="utf-8") as f_meta:
            json.dump(metadata_mapping, f_meta, ensure_ascii=False, indent=2)


def get_default_input_file(dataset_name: str = DEFAULT_DATASET_NAME) -> str:
    return get_documents_file(dataset_name)


def get_default_embed_path(dataset_name: str = DEFAULT_DATASET_NAME) -> str:
    return get_local_embedding_dir(dataset_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default=DEFAULT_DATASET_NAME)
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--embed_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1024)
    args = parser.parse_args()

    input_file = args.input_file or get_default_input_file(args.dataset_name)
    embed_path = args.embed_path or get_default_embed_path(args.dataset_name)

    print("Embedding MMQA global SQLite documents...")
    embed_documents(input_file, embed_path, batch_size=args.batch_size)
    print(f"Embeddings saved to {embed_path}")
