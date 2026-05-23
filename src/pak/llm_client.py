"""LLM inference backend abstraction.

Like Nemotron-Personas-Korea, PAK handles all inference through an
**OpenAI-compatible HTTP** interface. The following backends work with the same code:

- local vLLM      (any OpenAI-compatible server)
- local Ollama    (``ollama serve``, base_url=``http://localhost:11434/v1``)
- NVIDIA NIM      (``https://integrate.api.nvidia.com/v1``)
- OpenRouter      (``https://openrouter.ai/api/v1``)

Anthropic Claude is an optional backend (separate SDK).
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

import httpx

from pak.config import settings

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Result types
# ----------------------------------------------------------------------------


@dataclass
class CompletionUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class CompletionResult:
    text: str
    model: str
    backend: str
    usage: CompletionUsage = field(default_factory=CompletionUsage)
    raw: dict[str, Any] | None = None


# ----------------------------------------------------------------------------
# Clients
# ----------------------------------------------------------------------------


class OpenAICompatibleClient:
    """Shared OpenAI HTTP client for vLLM / Ollama / NIM / OpenRouter."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 120.0,
        backend_label: str = "openai_compatible",
    ) -> None:
        self.base_url = (base_url or settings.llm_base_url).rstrip("/")
        self.api_key = api_key or settings.llm_api_key or "EMPTY"
        self.timeout = timeout
        self.backend_label = backend_label
        self._client = httpx.Client(timeout=timeout)

    def __enter__(self) -> OpenAICompatibleClient:
        return self

    def __exit__(self, *exc: Any) -> None:
        self._client.close()

    def chat(
        self,
        *,
        model: str,
        system: str | None,
        user: str,
        max_tokens: int = 2000,
        temperature: float = 0.7,
        response_format: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> CompletionResult:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if response_format is not None:
            body["response_format"] = response_format
        if extra_body:
            body.update(extra_body)

        resp = self._client.post(url, headers=headers, json=body)
        resp.raise_for_status()
        data = resp.json()
        choice = data["choices"][0]
        text = choice["message"]["content"]
        usage_d = data.get("usage", {}) or {}
        usage = CompletionUsage(
            input_tokens=int(usage_d.get("prompt_tokens", 0) or 0),
            output_tokens=int(usage_d.get("completion_tokens", 0) or 0),
        )
        return CompletionResult(
            text=text,
            model=data.get("model", model),
            backend=self.backend_label,
            usage=usage,
            raw=data,
        )


class OllamaClient:
    """Ollama native API (`/api/chat`). Unlike OpenAI-compat, supports ``think:false``."""

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:11434",
        timeout: float = 600.0,
        backend_label: str = "ollama_native",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.backend_label = backend_label
        self._client = httpx.Client(timeout=timeout)

    def __enter__(self) -> OllamaClient:
        return self

    def __exit__(self, *exc: Any) -> None:
        self._client.close()

    def chat(
        self,
        *,
        model: str,
        system: str | None,
        user: str,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        response_format: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> CompletionResult:
        url = f"{self.base_url}/api/chat"
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": False,  # qwen3 thinking mode off
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                # The PAK prompt is ~3-4k tokens for system+user combined. The default
                # num_ctx (32k) wastes 8x KV cache, so we cap the context to the actual
                # prompt size to free up GPU headroom and gain concurrency throughput.
                "num_ctx": 8192,
            },
        }
        if response_format and response_format.get("type") == "json_object":
            body["format"] = "json"
        if extra_body:
            body.update(extra_body)

        resp = self._client.post(url, json=body)
        resp.raise_for_status()
        data = resp.json()
        text = data.get("message", {}).get("content", "") or ""
        usage = CompletionUsage(
            input_tokens=int(data.get("prompt_eval_count", 0) or 0),
            output_tokens=int(data.get("eval_count", 0) or 0),
        )
        return CompletionResult(
            text=text,
            model=data.get("model", model),
            backend=self.backend_label,
            usage=usage,
            raw=data,
        )


class AnthropicClient:
    """Optional: Anthropic Claude. Paid. Recommended only when budget > 0."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        backend_label: str = "anthropic",
    ) -> None:
        try:
            from anthropic import Anthropic
        except ImportError as exc:  # pragma: no cover
            raise ImportError("anthropic SDK not installed") from exc
        key = api_key or os.environ.get("ANTHROPIC_API_KEY") or settings.anthropic_api_key
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        self._client = Anthropic(api_key=key)
        self.backend_label = backend_label

    def chat(
        self,
        *,
        model: str,
        system: str | None,
        user: str,
        max_tokens: int = 2000,
        temperature: float = 0.7,
        response_format: dict[str, Any] | None = None,  # noqa: ARG002
        extra_body: dict[str, Any] | None = None,  # noqa: ARG002
        cache_system: bool = True,
    ) -> CompletionResult:
        sys_block: list[dict[str, Any]] | str | None
        if system and cache_system:
            sys_block = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        else:
            sys_block = system
        resp = self._client.messages.create(
            model=model,
            system=sys_block if sys_block is not None else "",  # type: ignore[arg-type]
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(
            block.text for block in resp.content if getattr(block, "type", None) == "text"
        )
        usage = CompletionUsage(
            input_tokens=getattr(resp.usage, "input_tokens", 0) or 0,
            output_tokens=getattr(resp.usage, "output_tokens", 0) or 0,
            cache_creation_input_tokens=getattr(resp.usage, "cache_creation_input_tokens", 0) or 0,
            cache_read_input_tokens=getattr(resp.usage, "cache_read_input_tokens", 0) or 0,
        )
        return CompletionResult(
            text=text, model=model, backend=self.backend_label, usage=usage, raw=None
        )


# ----------------------------------------------------------------------------
# Backend factory
# ----------------------------------------------------------------------------


def get_client(backend: str | None = None) -> Any:
    """Automatically select the backend from settings or environment variables."""
    backend = backend or settings.llm_backend
    if backend == "anthropic":
        return AnthropicClient()
    if backend == "ollama_native":
        # Strip /v1 from base_url to use the native API path
        base = settings.llm_base_url.rstrip("/").removesuffix("/v1")
        return OllamaClient(base_url=base)
    return OpenAICompatibleClient(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        backend_label=backend,
    )


# ----------------------------------------------------------------------------
# Cost estimation (for reference — self-hosting is 0)
# ----------------------------------------------------------------------------


# USD per 1M tokens. Based on public pricing from Anthropic, NVIDIA NIM, OpenRouter, etc.
PRICE_TABLE: dict[str, tuple[float, float]] = {
    # model_id: (input_per_1m_usd, output_per_1m_usd)
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-haiku-4-5-20251001": (0.8, 4.0),
    "Qwen/Qwen3.5-9B": (0.0, 0.0),  # self-host 4090
    "Qwen/Qwen3.5-27B-GPTQ-Int4": (0.0, 0.0),  # self-host 4090
    "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4": (0.0, 0.0),  # self-host 4090
    "Qwen/Qwen3.5-397B-A17B": (1.5, 7.0),  # cloud API estimate
    "qwen3:30b-a3b": (0.0, 0.0),  # self-host generator (PAK narratives)
    "qwen/qwen-2.5-72b-instruct:free": (0.0, 0.0),
}


def estimate_cost(
    *,
    n_personas: int,
    avg_input_tokens: int,
    avg_output_tokens: int,
    n_calls_per_persona: int = 1,
    model: str = None,  # type: ignore[assignment]
) -> dict[str, Any]:
    model = model or settings.llm_default_model
    rates = PRICE_TABLE.get(model, (0.0, 0.0))
    total_in = n_personas * avg_input_tokens * n_calls_per_persona
    total_out = n_personas * avg_output_tokens * n_calls_per_persona
    cost = (total_in / 1e6) * rates[0] + (total_out / 1e6) * rates[1]
    return {
        "n_personas": n_personas,
        "model": model,
        "calls_per_persona": n_calls_per_persona,
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "estimated_cost_usd": cost,
        "is_free_self_host": rates == (0.0, 0.0),
    }


# ----------------------------------------------------------------------------
# JSON response parser (extract JSON from chat responses)
# ----------------------------------------------------------------------------

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def extract_json_object(text: str) -> str:
    m = _JSON_FENCE_RE.search(text)
    if m:
        return m.group(1)
    text = text.strip()
    if text.startswith("{"):
        return text
    first = text.find("{")
    last = text.rfind("}")
    if first >= 0 and last > first:
        return text[first : last + 1]
    raise ValueError(f"no JSON object found in response: {text[:200]}")


def parse_json_response(text: str) -> dict[str, Any]:
    return json.loads(extract_json_object(text))
