import os
import threading
import importlib.util
from dataclasses import dataclass
from typing import Any


DEFAULT_HF_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_HF_CONTEXT_LENGTH = 131072
DEFAULT_HF_MAX_NEW_TOKENS = 2048
DEFAULT_HF_USE_YARN = True
DEFAULT_HF_YARN_FACTOR = 4.0
DEFAULT_HF_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS = 32768


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _resolve_torch_dtype(torch_module, dtype_name: str):
    dtype_name = (dtype_name or "auto").strip().lower()
    if dtype_name == "auto":
        return "auto"
    if dtype_name in {"bf16", "bfloat16"}:
        return torch_module.bfloat16
    if dtype_name in {"fp16", "float16", "half"}:
        return torch_module.float16
    if dtype_name in {"fp32", "float32", "float"}:
        return torch_module.float32
    raise ValueError(f"Unsupported HF_TORCH_DTYPE: {dtype_name}")


def _primary_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return getattr(model, "device", "cpu")


@dataclass
class ChatResult:
    content: str
    response_data: dict[str, Any]


class HFTransformersChatBackend:
    def __init__(self):
        self.model_name = os.environ.get("HF_MODEL_NAME", DEFAULT_HF_MODEL_NAME)
        self.context_length = _env_int("HF_CONTEXT_LENGTH", DEFAULT_HF_CONTEXT_LENGTH)
        self.max_new_tokens = _env_int("HF_MAX_NEW_TOKENS", DEFAULT_HF_MAX_NEW_TOKENS)
        self.device_map = os.environ.get("HF_DEVICE_MAP", "auto")
        self.device = os.environ.get("HF_DEVICE", "").strip()
        self.torch_dtype = os.environ.get("HF_TORCH_DTYPE", "auto")
        self.trust_remote_code = _env_bool("HF_TRUST_REMOTE_CODE", False)
        self.use_yarn = _env_bool("HF_USE_YARN", DEFAULT_HF_USE_YARN)
        self.yarn_factor = _env_float("HF_YARN_FACTOR", DEFAULT_HF_YARN_FACTOR)
        self.yarn_original_max_position_embeddings = _env_int(
            "HF_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS",
            DEFAULT_HF_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS,
        )
        self.do_sample = _env_bool("HF_DO_SAMPLE", False)
        self.temperature = _env_float("HF_TEMPERATURE", 0.0)
        self.top_p = _env_float("HF_TOP_P", 1.0)
        self.attn_implementation = os.environ.get("HF_ATTENTION_IMPLEMENTATION", "").strip()
        self._lock = threading.Lock()
        self._tokenizer = None
        self._model = None
        self._torch = None

    def _load(self):
        with self._lock:
            if self._model is not None and self._tokenizer is not None:
                return

            try:
                import torch
                from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
            except ImportError as exc:
                raise RuntimeError(
                    "Hugging Face backend requires transformers and torch. "
                    "Install them in the autolink environment before running the pipeline."
                ) from exc

            has_accelerate = importlib.util.find_spec("accelerate") is not None
            use_device_map = bool(self.device_map) and self.device_map.lower() != "none"
            if use_device_map and not has_accelerate:
                print(
                    "accelerate is not installed; falling back to a single-device "
                    "Hugging Face load. Install accelerate to use HF_DEVICE_MAP=auto."
                )
                use_device_map = False

            manual_device = self.device
            if not manual_device:
                manual_device = "cuda:0" if torch.cuda.is_available() else "cpu"

            config = AutoConfig.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
            )
            if self.use_yarn:
                rope_parameters = dict(getattr(config, "rope_parameters", {}) or {})
                config.rope_scaling = {
                    "type": "yarn",
                    "factor": self.yarn_factor,
                    "original_max_position_embeddings": self.yarn_original_max_position_embeddings,
                }
                rope_parameters.update(
                    {
                        "rope_type": "yarn",
                        "factor": self.yarn_factor,
                        "original_max_position_embeddings": self.yarn_original_max_position_embeddings,
                    }
                )
                rope_parameters.setdefault(
                    "rope_theta",
                    getattr(config, "rope_theta", None)
                    or getattr(config, "default_rope_theta", 10000.0),
                )
                config.rope_parameters = rope_parameters
                if hasattr(config, "max_position_embeddings"):
                    config.max_position_embeddings = max(
                        int(getattr(config, "max_position_embeddings", 0) or 0),
                        self.context_length,
                    )

            model_kwargs = {
                "config": config,
                "torch_dtype": _resolve_torch_dtype(torch, self.torch_dtype),
                "trust_remote_code": self.trust_remote_code,
            }
            if use_device_map:
                model_kwargs["device_map"] = self.device_map
            if self.attn_implementation:
                model_kwargs["attn_implementation"] = self.attn_implementation

            print(
                "Loading Hugging Face chat model "
                f"{self.model_name} with context_length={self.context_length}, "
                f"yarn={self.use_yarn}, "
                f"device_map={self.device_map if use_device_map else 'disabled'}, "
                f"device={manual_device if not use_device_map else 'auto'}"
            )
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
            )
            self._tokenizer.model_max_length = self.context_length
            self._model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
            if not use_device_map:
                self._model.to(manual_device)
            self._model.eval()
            self._torch = torch

    def chat(self, messages: list[dict[str, str]]) -> ChatResult:
        self._load()
        tokenizer = self._tokenizer
        model = self._model
        torch = self._torch

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = tokenizer(
            [text],
            return_tensors="pt",
            truncation=False,
        )
        prompt_tokens = int(model_inputs["input_ids"].shape[-1])
        if prompt_tokens + self.max_new_tokens > self.context_length:
            raise RuntimeError(
                "Prompt plus generation budget exceeds configured context length: "
                f"{prompt_tokens} + {self.max_new_tokens} > {self.context_length}. "
                "Raise HF_CONTEXT_LENGTH or lower HF_MAX_NEW_TOKENS."
            )

        device = _primary_device(model)
        model_inputs = {key: value.to(device) for key, value in model_inputs.items()}

        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "pad_token_id": tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id,
        }
        if self.do_sample:
            generation_kwargs["temperature"] = self.temperature
            generation_kwargs["top_p"] = self.top_p

        with torch.inference_mode():
            generated_ids = model.generate(**model_inputs, **generation_kwargs)

        output_ids = generated_ids[0][prompt_tokens:]
        content = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        completion_tokens = int(output_ids.shape[-1])
        response_data = {
            "backend": "hf_transformers",
            "model": self.model_name,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "context_length": self.context_length,
            "max_new_tokens": self.max_new_tokens,
            "rope_scaling": {
                "type": "yarn",
                "factor": self.yarn_factor,
                "original_max_position_embeddings": self.yarn_original_max_position_embeddings,
            }
            if self.use_yarn
            else None,
        }
        if not content:
            raise RuntimeError(f"Hugging Face model returned an empty response: {response_data}")
        return ChatResult(content=content, response_data=response_data)


_backend = None
_backend_lock = threading.Lock()


def get_chat_backend():
    global _backend
    with _backend_lock:
        if _backend is not None:
            return _backend

        _backend = HFTransformersChatBackend()
        return _backend


def chat_with_model(messages: list[dict[str, str]]) -> tuple[str, dict[str, Any]]:
    result = get_chat_backend().chat(messages)
    return result.content, result.response_data
