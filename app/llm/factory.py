"""Construct the configured LLM provider.

``prompted`` is the safe default: ``/api/tags`` for the installed model reports
``completion`` + ``thinking`` without advertising ``tools``. Native mode remains
available for models that support Ollama's tools field.
"""

from __future__ import annotations

from app.core.config import Settings
from app.llm.ollama import OllamaConfig, OllamaProvider, ToolCallMode


def build_ollama_provider(
    settings: Settings,
    *,
    tool_call_mode: ToolCallMode = "prompted",
) -> OllamaProvider:
    """Build an :class:`OllamaProvider` from application settings."""
    return OllamaProvider(
        OllamaConfig(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            num_ctx=settings.ollama_num_ctx,
            request_timeout_seconds=settings.ollama_request_timeout_seconds,
            tool_call_mode=tool_call_mode,
        )
    )
