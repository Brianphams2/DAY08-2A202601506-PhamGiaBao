"""Resolve one OpenAI-compatible LLM provider for generation, HyDE and RAGAS."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any, Mapping


@dataclass(frozen=True)
class LLMProvider:
    """Connection settings shared by every LLM call in the project."""

    name: str
    api_key: str
    base_url: str | None
    model: str


def _valid_key(value: str | None) -> str:
    key = (value or "").strip()
    return "" if not key or "..." in key else key


def get_llm_provider(*, model_override: str | None = None) -> LLMProvider:
    """Return generic provider settings, then fall back to legacy variables.

    ``LLM_*`` supports any OpenAI-compatible endpoint.  The legacy OpenRouter
    and OpenAI variables remain supported so existing installations do not
    break when the project is updated.
    """
    generic_key = _valid_key(os.getenv("LLM_API_KEY"))
    if generic_key:
        base_url = (os.getenv("LLM_BASE_URL") or "").strip() or None
        model = (model_override or os.getenv("LLM_MODEL") or "").strip()
        if not base_url:
            raise RuntimeError("LLM_BASE_URL is required when LLM_API_KEY is configured")
        if not model:
            raise RuntimeError("LLM_MODEL is required when LLM_API_KEY is configured")
        return LLMProvider(
            name=(os.getenv("LLM_PROVIDER") or "openai-compatible").strip(),
            api_key=generic_key,
            base_url=base_url.rstrip("/"),
            model=model,
        )

    openrouter_key = _valid_key(os.getenv("OPENROUTER_API_KEY"))
    if openrouter_key:
        return LLMProvider(
            name="openrouter",
            api_key=openrouter_key,
            base_url="https://openrouter.ai/api/v1",
            model=(model_override or os.getenv("OPENROUTER_MODEL") or "openai/gpt-4o-mini").strip(),
        )

    openai_key = _valid_key(os.getenv("OPENAI_API_KEY"))
    if openai_key:
        return LLMProvider(
            name="openai",
            api_key=openai_key,
            base_url=None,
            model=(model_override or os.getenv("OPENAI_MODEL") or "gpt-4o-mini").strip(),
        )

    raise RuntimeError(
        "Configure LLM_API_KEY/LLM_BASE_URL/LLM_MODEL, OPENROUTER_API_KEY, "
        "or OPENAI_API_KEY in .env"
    )


def llm_is_configured() -> bool:
    """Return whether at least one non-placeholder provider key is present."""
    return any(
        _valid_key(os.getenv(name))
        for name in ("LLM_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY")
    )


def extract_chat_content(response: Any) -> str:
    """Extract assistant text from SDK objects or compatible proxy payloads.

    Some OpenAI-compatible gateways return the assistant text directly as a
    JSON string for long requests instead of returning a typed ChatCompletion.
    Supporting that form here keeps provider quirks out of retrieval/generation
    code while still rejecting empty or structurally invalid payloads.
    """
    if isinstance(response, str):
        value = response.strip()
        if not value:
            return ""
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return value
        return extract_chat_content(decoded) if decoded != response else value

    choices = response.get("choices") if isinstance(response, Mapping) else getattr(response, "choices", None)
    if not choices:
        return ""
    choice = choices[0]
    message = choice.get("message") if isinstance(choice, Mapping) else getattr(choice, "message", None)
    if message is None:
        return ""
    content = message.get("content") if isinstance(message, Mapping) else getattr(message, "content", None)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            text = part.get("text") if isinstance(part, Mapping) else getattr(part, "text", None)
            if isinstance(text, str):
                parts.append(text)
        return "".join(parts).strip()
    return ""
