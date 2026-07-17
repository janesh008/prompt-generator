"""Local client for Qwen2.5-VL models using transformers.

This module replaces the OpenRouter API dependency with local execution
of Qwen2.5-VL-3B-Instruct.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional

# Lazy loading singletons for model and processor to avoid loading overhead on import
_MODEL = None
_PROCESSOR = None


def get_model_and_processor():
    """Retrieve or initialize the local Qwen model and processor."""
    global _MODEL, _PROCESSOR
    if _MODEL is None or _PROCESSOR is None:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
        print(f"Initializing local model: {MODEL_ID}...")
        _MODEL = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        _PROCESSOR = AutoProcessor.from_pretrained(
            MODEL_ID,
            min_pixels=256 * 28 * 28,
            max_pixels=512 * 28 * 28,
        )
    return _MODEL, _PROCESSOR


class QwenClientError(RuntimeError):
    """Raised when the local Qwen model execution fails."""


@dataclass
class QwenConfig:
    api_key: Optional[str] = None
    model: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1024
    timeout_seconds: int = 180
    max_retries: int = 3
    retry_base_delay_seconds: float = 2.0
    app_referer: Optional[str] = None
    app_title: Optional[str] = None

    @classmethod
    def from_env(cls, **overrides: Any) -> "QwenConfig":
        return cls(
            api_key=overrides.get("api_key"),
            model=overrides.get("model") or "Qwen/Qwen2.5-VL-3B-Instruct",
            temperature=float(
                overrides.get("temperature")
                if overrides.get("temperature") is not None
                else os.getenv("QWEN_TEMPERATURE", "0.7")
            ),
            max_tokens=int(
                overrides.get("max_tokens")
                if overrides.get("max_tokens") is not None
                else os.getenv("QWEN_MAX_TOKENS", "1024")
            ),
        )


class QwenClient:
    def __init__(self, config: Optional[QwenConfig] = None, **overrides: Any) -> None:
        self.config = config or QwenConfig.from_env(**overrides)

    def chat(
        self,
        messages: Iterable[Mapping[str, Any]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Mapping[str, Any]] = None,
        extra_payload: Optional[Mapping[str, Any]] = None,
    ) -> str:
        import torch
        from qwen_vl_utils import process_vision_info

        model, processor = get_model_and_processor()

        msg_list = list(messages)

        # Apply chat template
        text = processor.apply_chat_template(
            msg_list,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Process vision inputs if any images/videos are included in messages
        image_inputs, video_inputs = process_vision_info(msg_list)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        # Move to GPU/device
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Configuration options
        gen_tokens = max_tokens if max_tokens is not None else self.config.max_tokens
        # Fallback to standard token budget if out of range
        if not gen_tokens or gen_tokens <= 0:
            gen_tokens = 1024

        temp = temperature if temperature is not None else self.config.temperature
        do_sample = temp > 0.0

        gen_kwargs = {
            "max_new_tokens": gen_tokens,
            "do_sample": do_sample,
        }
        if do_sample:
            gen_kwargs["temperature"] = temp

        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                **gen_kwargs,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]

        response = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]

        return response.strip()


def set_openrouter_api_key(api_key: Optional[str] = None) -> str:
    """No-op legacy function for compatibility."""
    return api_key or ""


def set_qwen_api_key(api_key: Optional[str] = None) -> str:
    """No-op legacy function for compatibility."""
    return api_key or ""


def quick_openrouter_test(prompt: str = "Reply with OK only.") -> str:
    """Local inference verification test."""
    client = QwenClient()
    return client.chat([{"role": "user", "content": prompt}], max_tokens=20)


def quick_qwen_test(prompt: str = "Reply with OK only.") -> str:
    """Local inference verification test."""
    return quick_openrouter_test(prompt)
