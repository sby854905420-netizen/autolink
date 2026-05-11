import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from typing import Optional

import fcntl


def _safe_int(value) -> int:
    if value is None:
        return 0
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _safe_float(value) -> float:
    if value is None:
        return 0.0
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return 0.0


def extract_token_count(response_data: Optional[dict]) -> int:
    if not isinstance(response_data, dict):
        return 0

    usage = response_data.get("usage")
    if isinstance(usage, dict):
        total_tokens = usage.get("total_tokens")
        if total_tokens is not None:
            return _safe_int(total_tokens)

        prompt_tokens = (
            usage.get("prompt_tokens")
            or usage.get("input_tokens")
        )
        completion_tokens = (
            usage.get("completion_tokens")
            or usage.get("output_tokens")
        )
        return _safe_int(prompt_tokens) + _safe_int(completion_tokens)

    return 0


def _merge_cost_records(previous_record: dict, cost_record: dict) -> dict:
    return {
        "elapsed_seconds": round(
            _safe_float(previous_record.get("elapsed_seconds"))
            + _safe_float(cost_record.get("elapsed_seconds")),
            6,
        ),
        "token_count": (
            _safe_int(previous_record.get("token_count"))
            + _safe_int(cost_record.get("token_count"))
        ),
    }


def update_cost_json(output_path: str, sample_id: str, cost_record: dict):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    lock_path = f"{output_path}.lock"
    with open(lock_path, "a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

        all_records = {}
        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    all_records = json.loads(content)

        previous_record = all_records.get(sample_id, {})
        all_records[sample_id] = _merge_cost_records(previous_record, cost_record)

        fd, temp_path = tempfile.mkstemp(
            dir=output_dir or ".",
            prefix=".cost_",
            suffix=".json",
            text=True,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as temp_file:
                json.dump(all_records, temp_file, ensure_ascii=False, indent=2)
            os.replace(temp_path, output_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


@dataclass
class SampleCostRecorder:
    sample_id: str
    output_path: str
    token_count: int = 0
    start_time: float = field(default_factory=time.perf_counter)
    _saved_record: Optional[dict] = field(default=None, init=False, repr=False)

    def add_tokens(self, token_count: int):
        self.token_count += _safe_int(token_count)

    def add_response_usage(self, response_data: dict):
        self.add_tokens(extract_token_count(response_data))

    def build_record(self) -> dict:
        elapsed_seconds = max(0.0, time.perf_counter() - self.start_time)
        return {
            "elapsed_seconds": round(elapsed_seconds, 6),
            "token_count": self.token_count,
        }

    def save(self) -> dict:
        if self._saved_record is not None:
            return self._saved_record

        cost_record = self.build_record()
        update_cost_json(self.output_path, self.sample_id, cost_record)
        self._saved_record = cost_record
        return cost_record

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.save()
        return False
