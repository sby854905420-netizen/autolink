import os
import threading
import importlib.util
from dataclasses import dataclass
from typing import Any

from progress_logger import get_progress_logger


DEFAULT_HF_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_QWEN25_CONTEXT_LENGTH = 131072
DEFAULT_MINISTRAL3_CONTEXT_LENGTH = 262144
DEFAULT_HF_CONTEXT_LENGTH = DEFAULT_QWEN25_CONTEXT_LENGTH
DEFAULT_HF_MAX_NEW_TOKENS = 2048
DEFAULT_HF_USE_YARN = True
DEFAULT_HF_YARN_FACTOR = 4.0
DEFAULT_HF_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS = 32768


def _normalise_model_name(model_name: str) -> str:
    return (model_name or "").strip().lower()


def _is_qwen_model(model_name: str, config=None) -> bool:
    model_name = _normalise_model_name(model_name)
    model_type = _normalise_model_name(getattr(config, "model_type", ""))
    return "qwen" in model_name or model_type.startswith("qwen")


def _is_qwen25_model_name(model_name: str) -> bool:
    model_name = _normalise_model_name(model_name)
    return "qwen2.5" in model_name or "qwen-2.5" in model_name


def _is_ministral3_model_name(model_name: str) -> bool:
    model_name = _normalise_model_name(model_name)
    return "ministral-3" in model_name or "ministral3" in model_name


def default_context_length_for_model(model_name: str) -> int:
    if _is_ministral3_model_name(model_name):
        return DEFAULT_MINISTRAL3_CONTEXT_LENGTH
    if _is_qwen25_model_name(model_name):
        return DEFAULT_QWEN25_CONTEXT_LENGTH
    raise ValueError(
        "Unsupported HF_MODEL_NAME. This pipeline currently supports only "
        "Qwen2.5 series models and Ministral-3 series models."
    )


def _is_ministral3_config(config) -> bool:
    model_type = _normalise_model_name(getattr(config, "model_type", ""))
    text_config = getattr(config, "text_config", None)
    text_model_type = _normalise_model_name(getattr(text_config, "model_type", ""))
    architectures = getattr(config, "architectures", []) or []
    architecture_names = {_normalise_model_name(name) for name in architectures}
    return (
        model_type == "mistral3"
        or text_model_type == "ministral3"
        or "mistral3forconditionalgeneration" in architecture_names
    )


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


def _set_tokenizer_context_length(tokenizer, context_length: int):
    try:
        tokenizer.model_max_length = context_length
    except AttributeError:
        pass


def _decode_tokenizer_output(tokenizer, token_ids) -> str:
    try:
        return tokenizer.decode(token_ids, skip_special_tokens=True).strip()
    except TypeError:
        return tokenizer.decode(token_ids).strip()


def _move_model_inputs(model_inputs, device, torch_module):
    moved_inputs = {}
    for key, value in dict(model_inputs).items():
        if not hasattr(value, "to"):
            moved_inputs[key] = value
            continue
        if key == "pixel_values" and torch_module.is_floating_point(value):
            moved_inputs[key] = value.to(device=device, dtype=torch_module.bfloat16)
        else:
            moved_inputs[key] = value.to(device)
    return moved_inputs


def _normalise_text_message_parts(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalised_messages = []
    for message in messages:
        normalised_message = dict(message)
        content = normalised_message.get("content")
        if isinstance(content, str):
            normalised_message["content"] = [{"type": "text", "text": content}]
        normalised_messages.append(normalised_message)
    return normalised_messages


@dataclass
class ChatResult:
    content: str
    response_data: dict[str, Any]


class HFTransformersChatBackend:
    def __init__(self):
        self._logger = get_progress_logger(__name__)
        self.model_name = os.environ.get("HF_MODEL_NAME", DEFAULT_HF_MODEL_NAME)
        self.context_length = _env_int(
            "HF_CONTEXT_LENGTH",
            default_context_length_for_model(self.model_name),
        )
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
        self._backend_name = "hf_transformers"
        self._applied_rope_scaling = None

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
                self._logger.warning(
                    "accelerate is not installed; falling back to a single-device "
                    "Hugging Face load. Install accelerate to use HF_DEVICE_MAP=auto."
                )
                use_device_map = False

            manual_device = self.device
            if not manual_device:
                manual_device = "cuda:0" if torch.cuda.is_available() else "cpu"

            if _is_ministral3_model_name(self.model_name):
                self._load_ministral3(torch, use_device_map, manual_device)
                return

            config = AutoConfig.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
            )
            if _is_ministral3_config(config):
                self._load_ministral3(torch, use_device_map, manual_device)
                return

            apply_yarn = self.use_yarn and _is_qwen_model(self.model_name, config)
            if self.use_yarn and not apply_yarn:
                self._logger.info(
                    "Skipping YaRN override for non-Qwen Hugging Face model "
                    f"{self.model_name}."
                )
            if apply_yarn:
                rope_parameters = dict(getattr(config, "rope_parameters", {}) or {})
                rope_scaling = {
                    "type": "yarn",
                    "factor": self.yarn_factor,
                    "original_max_position_embeddings": self.yarn_original_max_position_embeddings,
                }
                config.rope_scaling = rope_scaling
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
                self._applied_rope_scaling = rope_scaling

            model_kwargs = {
                "config": config,
                "torch_dtype": _resolve_torch_dtype(torch, self.torch_dtype),
                "trust_remote_code": self.trust_remote_code,
            }
            if use_device_map:
                model_kwargs["device_map"] = self.device_map
            if self.attn_implementation:
                model_kwargs["attn_implementation"] = self.attn_implementation

            self._logger.info(
                "Loading Hugging Face chat model "
                f"{self.model_name} with context_length={self.context_length}, "
                f"yarn={bool(self._applied_rope_scaling)}, "
                f"device_map={self.device_map if use_device_map else 'disabled'}, "
                f"device={manual_device if not use_device_map else 'auto'}"
            )
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
            )
            _set_tokenizer_context_length(self._tokenizer, self.context_length)
            self._model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
            if not use_device_map:
                self._model.to(manual_device)
            self._model.eval()
            self._torch = torch

    def _load_ministral3(self, torch, use_device_map: bool, manual_device: str):
        try:
            from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend
        except ImportError as exc:
            raise RuntimeError(
                "Ministral 3 models require a Transformers build with "
                "Mistral3ForConditionalGeneration and MistralCommonBackend, plus "
                "mistral-common>=1.8.6. Install/update these dependencies in the "
                "autolink environment before using "
                f"{self.model_name}."
            ) from exc

        model_kwargs = {}
        torch_dtype = _resolve_torch_dtype(torch, self.torch_dtype)
        if torch_dtype != "auto":
            model_kwargs["torch_dtype"] = torch_dtype
        if use_device_map:
            model_kwargs["device_map"] = self.device_map
        if self.attn_implementation:
            model_kwargs["attn_implementation"] = self.attn_implementation

        self._logger.info(
            "Loading Ministral 3 Hugging Face chat model "
            f"{self.model_name} with context_length={self.context_length}, "
            "yarn=native, "
            f"device_map={self.device_map if use_device_map else 'disabled'}, "
            f"device={manual_device if not use_device_map else 'auto'}"
        )
        self._tokenizer = MistralCommonBackend.from_pretrained(self.model_name)
        _set_tokenizer_context_length(self._tokenizer, self.context_length)
        self._model = Mistral3ForConditionalGeneration.from_pretrained(
            self.model_name,
            **model_kwargs,
        )
        if not use_device_map:
            self._model.to(manual_device)
        self._model.eval()
        self._torch = torch
        self._backend_name = "hf_transformers_ministral3"
        self._applied_rope_scaling = None

    def _tokenize_messages(self, tokenizer, messages: list[dict[str, str]]):
        if self._backend_name == "hf_transformers_ministral3":
            try:
                return tokenizer.apply_chat_template(
                    messages,
                    return_tensors="pt",
                    return_dict=True,
                )
            except (TypeError, ValueError):
                return tokenizer.apply_chat_template(
                    _normalise_text_message_parts(messages),
                    return_tensors="pt",
                    return_dict=True,
                )

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return tokenizer(
            [text],
            return_tensors="pt",
            truncation=False,
        )

    def chat(self, messages: list[dict[str, str]]) -> ChatResult:
        self._load()
        tokenizer = self._tokenizer
        model = self._model
        torch = self._torch

        model_inputs = self._tokenize_messages(tokenizer, messages)
        prompt_tokens = int(model_inputs["input_ids"].shape[-1])
        if prompt_tokens + self.max_new_tokens > self.context_length:
            raise RuntimeError(
                "Prompt plus generation budget exceeds configured context length: "
                f"{prompt_tokens} + {self.max_new_tokens} > {self.context_length}. "
                "Raise HF_CONTEXT_LENGTH or lower HF_MAX_NEW_TOKENS."
            )

        device = _primary_device(model)
        model_inputs = _move_model_inputs(model_inputs, device, torch)

        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
        }
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if pad_token_id is not None or eos_token_id is not None:
            generation_kwargs["pad_token_id"] = (
                pad_token_id if pad_token_id is not None else eos_token_id
            )
        if self.do_sample:
            generation_kwargs["temperature"] = self.temperature
            generation_kwargs["top_p"] = self.top_p
        if self._backend_name == "hf_transformers_ministral3" and "pixel_values" in model_inputs:
            generation_kwargs.setdefault("image_sizes", [model_inputs["pixel_values"].shape[-2:]])

        with torch.inference_mode():
            generated_ids = model.generate(**model_inputs, **generation_kwargs)

        output_ids = generated_ids[0][prompt_tokens:]
        content = _decode_tokenizer_output(tokenizer, output_ids)
        completion_tokens = int(output_ids.shape[-1])
        response_data = {
            "backend": self._backend_name,
            "model": self.model_name,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "context_length": self.context_length,
            "max_new_tokens": self.max_new_tokens,
            "rope_scaling": self._applied_rope_scaling,
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
