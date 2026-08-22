"""Live AvalAI smoke tests.

Skipped unless ``RUN_AVALAI_SMOKE=1`` and ``AVALAI_API_KEY`` is present.
Never prints the key, prompts, or hidden reasoning.
"""

from __future__ import annotations

import os

import pytest
from app.core.config import get_settings
from app.llm.contracts import ChatMessage, ToolDefinition
from app.llm.factory import build_avalai_provider

pytestmark = pytest.mark.avalai

_SECRET_TOOL = ToolDefinition(
    name="get_secret_test_value",
    description="Return a server-side diagnostic value. The model must call this tool.",
    parameters_schema={"type": "object", "properties": {}},
)


def _enabled() -> bool:
    return os.environ.get("RUN_AVALAI_SMOKE") == "1"


@pytest.mark.asyncio
async def test_live_avalai_basic_completion():
    if not _enabled():
        pytest.skip("Set RUN_AVALAI_SMOKE=1 to run live AvalAI checks")
    get_settings.cache_clear()
    settings = get_settings()
    if not (settings.avalai_api_key or "").strip():
        pytest.skip("AVALAI_API_KEY is not set")
    provider = build_avalai_provider(settings)
    try:
        response = await provider.chat(
            [
                ChatMessage(
                    role="user",
                    content="Reply with exactly: Ariadne AvalAI connection OK",
                )
            ]
        )
    finally:
        await provider.aclose()
        get_settings.cache_clear()
    assert "Ariadne AvalAI connection OK" in response.content
    assert "<think>" not in response.content
    assert provider.model_name == settings.avalai_model


@pytest.mark.asyncio
async def test_live_avalai_native_tool_call_is_actually_emitted():
    if not _enabled():
        pytest.skip("Set RUN_AVALAI_SMOKE=1 to run live AvalAI checks")
    get_settings.cache_clear()
    settings = get_settings()
    if not (settings.avalai_api_key or "").strip():
        pytest.skip("AVALAI_API_KEY is not set")
    provider = build_avalai_provider(settings, tool_call_mode="native")
    try:
        response = await provider.chat(
            [
                ChatMessage(
                    role="user",
                    content=(
                        "You must call the get_secret_test_value tool. "
                        "Do not invent the value. Use the tool."
                    ),
                )
            ],
            tools=[_SECRET_TOOL],
        )
    finally:
        mode = provider.tool_call_mode
        await provider.aclose()
        get_settings.cache_clear()
    names = [call.name for call in response.tool_calls]
    # Accepting a tools field is not enough; native emission is the signal.
    if mode == "prompted_fallback":
        pytest.skip("AvalAI route rejected native tools; prompted_fallback is active")
    assert "get_secret_test_value" in names
