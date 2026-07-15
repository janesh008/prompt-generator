"""Colab-friendly OpenRouter client for Qwen models.

This module uses OpenRouter's OpenAI-compatible chat completions API. Keep the
API key outside the code and pass it through an environment variable, Colab
secret, or runtime prompt.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional

import requests


DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "qwen/qwen3.5-flash"
DEFAULT_APP_REFERER = "https://colab.research.google.com"
DEFAULT_APP_TITLE = "LittleNest Colab Prompt Generator"


class QwenClientError(RuntimeError):
    """Raised when the Qwen API request fails or returns an invalid payload."""


@dataclass
class QwenConfig:
    api_key: Optional[str] = None
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    temperature: float = 0.7
    max_tokens: int = 12000
    timeout_seconds: int = 180
    max_retries: int = 3
    retry_base_delay_seconds: float = 2.0
    app_referer: Optional[str] = DEFAULT_APP_REFERER
    app_title: Optional[str] = DEFAULT_APP_TITLE

    @classmethod
    def from_env(cls, **overrides: Any) -> "QwenConfig":
        return cls(
            api_key=overrides.get("api_key")
            or os.getenv("OPENROUTER_API_KEY")
            or os.getenv("QWEN_API_KEY"),
            model=overrides.get("model")
            or os.getenv("OPENROUTER_MODEL")
            or os.getenv("QWEN_MODEL", DEFAULT_MODEL),
            base_url=overrides.get("base_url")
            or os.getenv("OPENROUTER_BASE_URL")
            or os.getenv("QWEN_BASE_URL", DEFAULT_BASE_URL),
            temperature=float(
                overrides.get("temperature")
                if overrides.get("temperature") is not None
                else os.getenv("OPENROUTER_TEMPERATURE")
                or os.getenv("QWEN_TEMPERATURE", "0.7")
            ),
            max_tokens=int(
                overrides.get("max_tokens")
                if overrides.get("max_tokens") is not None
                else os.getenv("OPENROUTER_MAX_TOKENS")
                or os.getenv("QWEN_MAX_TOKENS", "12000")
            ),
            timeout_seconds=int(overrides.get("timeout_seconds", 180)),
            max_retries=int(overrides.get("max_retries", 3)),
            retry_base_delay_seconds=float(
                overrides.get("retry_base_delay_seconds", 2.0)
            ),
            app_referer=overrides.get("app_referer")
            or os.getenv("OPENROUTER_HTTP_REFERER", DEFAULT_APP_REFERER),
            app_title=overrides.get("app_title")
            or os.getenv("OPENROUTER_APP_TITLE", DEFAULT_APP_TITLE),
        )


class QwenClient:
    def __init__(self, config: Optional[QwenConfig] = None, **overrides: Any) -> None:
        self.config = config or QwenConfig.from_env(**overrides)
        self.base_url = self.config.base_url.rstrip("/")

        if not self.config.api_key:
            raise QwenClientError(
                "Missing OpenRouter API key. Set OPENROUTER_API_KEY or pass api_key=..."
            )

    def chat(
        self,
        messages: Iterable[Mapping[str, str]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Mapping[str, Any]] = None,
        extra_payload: Optional[Mapping[str, Any]] = None,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": self.config.model,
            "messages": list(messages),
            "temperature": self.config.temperature
            if temperature is None
            else temperature,
            "max_tokens": self.config.max_tokens
            if max_tokens is None
            else max_tokens,
        }

        if response_format:
            payload["response_format"] = dict(response_format)

        if extra_payload:
            payload.update(dict(extra_payload))

        data = self._post_chat_completions(payload)
        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content")
        )

        if not content:
            raise QwenClientError(
                f"Qwen response did not include choices[0].message.content: {data}"
            )

        return str(content).strip()

    def _post_chat_completions(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        if self.config.app_referer:
            headers["HTTP-Referer"] = self.config.app_referer
        if self.config.app_title:
            headers["X-OpenRouter-Title"] = self.config.app_title

        last_error: Optional[Exception] = None

        for attempt in range(1, self.config.max_retries + 1):
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=dict(payload),
                    timeout=self.config.timeout_seconds,
                )

                if response.status_code in {429, 500, 502, 503, 504}:
                    raise QwenClientError(
                        f"Temporary OpenRouter API error {response.status_code}: "
                        f"{response.text[:500]}"
                    )

                if not response.ok:
                    raise QwenClientError(
                        f"OpenRouter API error {response.status_code}: {response.text[:1000]}"
                    )

                return response.json()
            except (requests.RequestException, ValueError, QwenClientError) as exc:
                last_error = exc
                if attempt >= self.config.max_retries:
                    break
                time.sleep(self.config.retry_base_delay_seconds * attempt)

        raise QwenClientError(f"OpenRouter request failed after retries: {last_error}")


def set_openrouter_api_key(api_key: Optional[str] = None) -> str:
    """Set OPENROUTER_API_KEY in the current notebook/runtime.

    In Colab, call this with no argument to get a hidden password prompt:

        set_openrouter_api_key()

    Or pass the value directly:

        set_openrouter_api_key("sk-or-...")
    """

    if api_key is None:
        from getpass import getpass

        api_key = getpass("Enter OpenRouter API key: ").strip()

    if not api_key:
        raise QwenClientError("API key cannot be empty.")

    os.environ["OPENROUTER_API_KEY"] = api_key
    return api_key


def set_qwen_api_key(api_key: Optional[str] = None) -> str:
    """Backward-compatible alias for older notebook cells."""

    return set_openrouter_api_key(api_key)


def quick_openrouter_test(prompt: str = "Reply with OK only.") -> str:
    """Small smoke test for Colab after setting OPENROUTER_API_KEY."""

    client = QwenClient()
    return client.chat([{"role": "user", "content": prompt}], max_tokens=20)


def quick_qwen_test(prompt: str = "Reply with OK only.") -> str:
    """Backward-compatible alias for older notebook cells."""

    return quick_openrouter_test(prompt)
