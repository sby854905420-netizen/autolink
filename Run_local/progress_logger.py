import logging
import time
from contextlib import contextmanager

from tqdm import tqdm


LOGGER_NAME = "autolink"


class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            tqdm.write(self.format(record))
        except Exception:
            self.handleError(record)


def _parse_log_level() -> int:
    import os

    level_name = os.environ.get("AUTOLINK_LOG_LEVEL", "INFO").strip().upper()
    return getattr(logging, level_name, logging.INFO)


def setup_progress_logging(log_path: str | None = None) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(_parse_log_level())
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(process)d | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    for handler in list(logger.handlers):
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
            handler.close()

    if not any(isinstance(handler, TqdmLoggingHandler) for handler in logger.handlers):
        console_handler = TqdmLoggingHandler()
        console_handler.setLevel(_parse_log_level())
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def get_progress_logger(name: str | None = None) -> logging.Logger:
    base_logger = setup_progress_logging()
    if not name:
        return base_logger
    return base_logger.getChild(name)


def format_fields(**fields) -> str:
    visible_fields = {
        key: value
        for key, value in fields.items()
        if value is not None and value != ""
    }
    return " ".join(f"{key}={value}" for key, value in visible_fields.items())


def log_run_header(
    logger: logging.Logger,
    *,
    dataset_name: str,
    model_name: str,
    log_path: str,
    top_n: int,
    num_threads: int,
    hf_context_length: int,
    hf_max_new_tokens: int,
):
    logger.info(
        "Run started | %s",
        format_fields(
            dataset=dataset_name,
            model=model_name,
            top_n=top_n,
            num_threads=num_threads,
            context_length=hf_context_length,
            max_new_tokens=hf_max_new_tokens,
            log_path=log_path,
        ),
    )


@contextmanager
def log_stage(logger: logging.Logger, stage_name: str, **fields):
    field_text = format_fields(**fields)
    suffix = f" | {field_text}" if field_text else ""
    start_time = time.perf_counter()
    logger.info("Stage started: %s%s", stage_name, suffix)
    try:
        yield
    except Exception:
        elapsed = time.perf_counter() - start_time
        logger.exception("Stage failed: %s elapsed=%.2fs%s", stage_name, elapsed, suffix)
        raise
    else:
        elapsed = time.perf_counter() - start_time
        logger.info("Stage finished: %s elapsed=%.2fs%s", stage_name, elapsed, suffix)


def progress_bar(iterable, **kwargs):
    kwargs.setdefault("dynamic_ncols", True)
    kwargs.setdefault("mininterval", 1.0)
    return tqdm(iterable, **kwargs)
