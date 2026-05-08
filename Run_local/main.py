import argparse
import os
import sys

from utils import (
    DEFAULT_DATASET_NAME,
    prepare_log_dir,
    require_dataset_file,
    require_local_embedding_index,
    require_supported_dataset,
)
from llm_backends import (
    DEFAULT_HF_CONTEXT_LENGTH,
    DEFAULT_HF_MAX_NEW_TOKENS,
    DEFAULT_HF_MODEL_NAME,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the local AutoLink pipeline.")
    parser.add_argument("dataset", nargs="?", default=None)
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--hf_model_name", default=os.environ.get("HF_MODEL_NAME", DEFAULT_HF_MODEL_NAME))
    parser.add_argument(
        "--hf_context_length",
        type=int,
        default=int(os.environ.get("HF_CONTEXT_LENGTH", str(DEFAULT_HF_CONTEXT_LENGTH))),
    )
    parser.add_argument(
        "--hf_max_new_tokens",
        type=int,
        default=int(os.environ.get("HF_MAX_NEW_TOKENS", str(DEFAULT_HF_MAX_NEW_TOKENS))),
    )
    parser.add_argument("--top_n", type=int, default=int(os.environ.get("TOP_N", "100")))
    parser.add_argument("--num_threads", type=int, default=int(os.environ.get("NUM_THREADS", "1")))
    parser.add_argument("--time_id", default=os.environ.get("TIME_ID"))
    parser.add_argument("--log_path", default=os.environ.get("LOG_PATH"))
    return parser.parse_args()


def resolve_dataset_name(args) -> str:
    return args.dataset_name or args.dataset or os.environ.get("DATASET_NAME") or DEFAULT_DATASET_NAME


def reset_cost_files(log_path: str):
    for filename in ("cost.json", "cost.json.lock"):
        path = os.path.join(log_path, filename)
        if os.path.exists(path):
            os.remove(path)


def run_pipeline(
    dataset_name: str = DEFAULT_DATASET_NAME,
    hf_model_name: str = DEFAULT_HF_MODEL_NAME,
    hf_context_length: int = DEFAULT_HF_CONTEXT_LENGTH,
    hf_max_new_tokens: int = DEFAULT_HF_MAX_NEW_TOKENS,
    top_n: int = 100,
    num_threads: int = 1,
    time_id: str | None = None,
    log_path: str | None = None,
):
    dataset_name = require_supported_dataset(dataset_name)
    require_dataset_file(dataset_name)
    require_local_embedding_index(dataset_name)

    os.environ["HF_MODEL_NAME"] = hf_model_name
    os.environ["HF_CONTEXT_LENGTH"] = str(hf_context_length)
    os.environ["HF_MAX_NEW_TOKENS"] = str(hf_max_new_tokens)

    resolved_log_path = prepare_log_dir(
        log_path=log_path,
        dataset_name=dataset_name,
        model_name=hf_model_name,
        time_id=time_id,
    )
    reset_cost_files(resolved_log_path)

    from add_id import add_pre_rule
    from complete_schema import complete_schema
    from generate_schema import generate_schema_prompt
    from postprocess import merge
    from retrieve_topk_schema import retrieve

    retrieve(resolved_log_path, top_n=max(1, top_n), dataset_name=dataset_name)
    add_pre_rule(resolved_log_path, dataset_name=dataset_name)
    generate_schema_prompt(resolved_log_path, is_initial=True, dataset_name=dataset_name)
    complete_schema(resolved_log_path, dataset_name=dataset_name, num_threads=max(1, num_threads))
    merge(log_path=resolved_log_path, is_preprocess=True, dataset_name=dataset_name)
    generate_schema_prompt(resolved_log_path, is_initial=False, dataset_name=dataset_name)

    print(f"Pipeline completed for {dataset_name}")
    print(f"Log path: {resolved_log_path}")


if __name__ == "__main__":
    args = parse_args()
    try:
        run_pipeline(
            dataset_name=resolve_dataset_name(args),
            hf_model_name=args.hf_model_name,
            hf_context_length=args.hf_context_length,
            hf_max_new_tokens=args.hf_max_new_tokens,
            top_n=args.top_n,
            num_threads=args.num_threads,
            time_id=args.time_id,
            log_path=args.log_path,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
