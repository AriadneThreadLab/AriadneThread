"""Construct the configured LLM provider.

``LLM_PROVIDER`` selects the implementation. ``ollama`` (default) is unchanged:
``OLLAMA_TOOL_MODE`` still defaults to ``prompted``. ``avalai`` prefers native
OpenAI-compatible tool calling unless ``AVALAI_TOOL_MODE=prompted``.
"""

from __future__ import annotations

from app.core.config import Settings
from app.core.errors import ConfigurationError
from app.llm.avalai import AvalAIConfig, AvalAIProvider
from app.llm.contracts import LLMProvider
from app.llm.ollama import OllamaConfig, OllamaProvider, ToolCallMode


def build_ollama_provider(
    settings: Settings,
    *,
    tool_call_mode: ToolCallMode | None = None,
) -> OllamaProvider:
    """Build an :class:`OllamaProvider` from application settings."""
    mode: ToolCallMode = tool_call_mode or settings.ollama_tool_mode
    return OllamaProvider(
        OllamaConfig(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            num_ctx=settings.ollama_num_ctx,
            request_timeout_seconds=settings.ollama_request_timeout_seconds,
            tool_call_mode=mode,
        )
    )


def build_avalai_provider(
    settings: Settings,
    *,
    tool_call_mode: ToolCallMode | None = None,
) -> AvalAIProvider:
    """Build an :class:`AvalAIProvider` from application settings."""
    api_key = (settings.avalai_api_key or "").strip()
    if not api_key:
        raise ConfigurationError("AVALAI_API_KEY is required when LLM_PROVIDER=avalai")
    mode: ToolCallMode = tool_call_mode or settings.avalai_tool_mode
    return AvalAIProvider(
        AvalAIConfig(
            api_key=api_key,
            base_url=settings.avalai_base_url,
            model=settings.avalai_model,
            request_timeout_seconds=settings.avalai_timeout_seconds,
            tool_call_mode=mode,
            max_attempts=settings.avalai_max_attempts,
            retry_backoff_seconds=settings.avalai_retry_backoff_seconds,
        )
    )


def build_llm_provider(
    settings: Settings,
    *,
    tool_call_mode: ToolCallMode | None = None,
) -> LLMProvider:
    """Build the chat provider selected by ``LLM_PROVIDER``."""
    if settings.llm_provider == "avalai":
        return build_avalai_provider(settings, tool_call_mode=tool_call_mode)
    return build_ollama_provider(settings, tool_call_mode=tool_call_mode)
