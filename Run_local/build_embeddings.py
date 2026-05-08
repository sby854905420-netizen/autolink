import argparse
import os
import sys

from utils import (
    DEFAULT_DATASET_NAME,
    get_documents_file,
    get_local_embedding_dir,
    require_schema_file,
    require_supported_dataset,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Build local dataset documents and embeddings.")
    parser.add_argument("dataset", nargs="?", default=None)
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--batch_size", type=int, default=int(os.environ.get("EMBED_BATCH_SIZE", "1024")))
    parser.add_argument("--documents_path", default=None)
    parser.add_argument("--embeddings_path", default=None)
    return parser.parse_args()


def resolve_dataset_name(args) -> str:
    return args.dataset_name or args.dataset or os.environ.get("DATASET_NAME") or DEFAULT_DATASET_NAME


def build_embeddings(
    dataset_name: str = DEFAULT_DATASET_NAME,
    batch_size: int = 1024,
    documents_path: str | None = None,
    embeddings_path: str | None = None,
):
    dataset_name = require_supported_dataset(dataset_name)
    require_schema_file(dataset_name)

    from embedding_docs import embed_documents
    from generate_docs import generate_documents

    generate_documents(dataset_name=dataset_name, output_path=documents_path)

    input_file = os.path.join(documents_path, "localdb.json") if documents_path else get_documents_file(dataset_name)
    embed_path = embeddings_path or get_local_embedding_dir(dataset_name)
    embed_documents(input_file, embed_path, batch_size=batch_size)

    print(f"Embedding build completed for {dataset_name}")
    print(f"Documents: {input_file}")
    print(f"Embeddings: {embed_path}")


if __name__ == "__main__":
    args = parse_args()
    try:
        build_embeddings(
            dataset_name=resolve_dataset_name(args),
            batch_size=max(1, args.batch_size),
            documents_path=args.documents_path,
            embeddings_path=args.embeddings_path,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
