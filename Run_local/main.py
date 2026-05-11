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
    DEFAULT_HF_MAX_NEW_TOKENS,
    DEFAULT_HF_MODEL_NAME,
    default_context_length_for_model,
)
from progress_logger import log_run_header, log_stage, setup_progress_logging


def optional_env_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return None
    return int(value)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the local AutoLink pipeline.")
    parser.add_argument("dataset", nargs="?", default=None)
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--hf_model_name", default=os.environ.get("HF_MODEL_NAME", DEFAULT_HF_MODEL_NAME))
    parser.add_argument(
        "--hf_context_length",
        type=int,
        default=optional_env_int("HF_CONTEXT_LENGTH"),
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
    hf_context_length: int | None = None,
    hf_max_new_tokens: int = DEFAULT_HF_MAX_NEW_TOKENS,
    top_n: int = 100,
    num_threads: int = 1,
    time_id: str | None = None,
    log_path: str | None = None,
):
    dataset_name = require_supported_dataset(dataset_name)
    require_dataset_file(dataset_name)
    require_local_embedding_index(dataset_name)

    resolved_hf_context_length = (
        hf_context_length
        if hf_context_length is not None
        else default_context_length_for_model(hf_model_name)
    )

    os.environ["HF_MODEL_NAME"] = hf_model_name
    os.environ["HF_CONTEXT_LENGTH"] = str(resolved_hf_context_length)
    os.environ["HF_MAX_NEW_TOKENS"] = str(hf_max_new_tokens)

    resolved_log_path = prepare_log_dir(
        log_path=log_path,
        dataset_name=dataset_name,
        model_name=hf_model_name,
        time_id=time_id,
    )
    logger = setup_progress_logging(resolved_log_path)
    log_run_header(
        logger,
        dataset_name=dataset_name,
        model_name=hf_model_name,
        log_path=resolved_log_path,
        top_n=top_n,
        num_threads=max(1, num_threads),
        hf_context_length=resolved_hf_context_length,
        hf_max_new_tokens=hf_max_new_tokens,
    )
    reset_cost_files(resolved_log_path)

    from add_id import add_pre_rule
    from complete_schema import complete_schema
    from generate_schema import generate_schema_prompt
    from postprocess import merge
    from retrieve_topk_schema import retrieve

    with log_stage(logger, "retrieve top-k schema", dataset=dataset_name, top_n=max(1, top_n)):
        retrieve(resolved_log_path, top_n=max(1, top_n), dataset_name=dataset_name)
    with log_stage(logger, "add id/name/code columns", dataset=dataset_name):
        add_pre_rule(resolved_log_path, dataset_name=dataset_name)
    with log_stage(logger, "generate initial schema prompts", dataset=dataset_name):
        generate_schema_prompt(resolved_log_path, is_initial=True, dataset_name=dataset_name)
    with log_stage(
        logger,
        "complete schema with llm",
        dataset=dataset_name,
        model=hf_model_name,
        num_threads=max(1, num_threads),
    ):
        complete_schema(resolved_log_path, dataset_name=dataset_name, num_threads=max(1, num_threads))
    with log_stage(logger, "merge candidates", dataset=dataset_name):
        merge(log_path=resolved_log_path, is_preprocess=True, dataset_name=dataset_name)
    with log_stage(logger, "generate final schema prompts", dataset=dataset_name):
        generate_schema_prompt(resolved_log_path, is_initial=False, dataset_name=dataset_name)

    logger.info("Pipeline completed | dataset=%s log_path=%s", dataset_name, resolved_log_path)


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
