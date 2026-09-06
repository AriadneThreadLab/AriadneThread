"""Developer diagnostics for the Ollama provider / tool protocol boundary."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import httpx

from app.analytics.contracts import AnalysisPlan
from app.core.config import Settings
from app.llm.avalai import AvalAIConfig, AvalAIProvider, safe_avalai_host
from app.llm.contracts import ChatMessage, ToolDefinition
from app.llm.factory import build_avalai_provider
from app.llm.ollama import OllamaConfig, OllamaProvider
from app.llm.tool_schema import normalize_tool_parameters_schema
from app.osm.query_spec import OsmFeatureQuery
from app.tools.energy_tools import AnalyzeEnergyGridArgs
from app.tools.search_osm_knowledge import SearchOsmKnowledgeArgs
from app.tools.simbench_tools import SimBenchQueryArgs

_SECRET_TOOL_NAME = "get_secret_test_value"
_SECRET_TOOL_VALUE = "ariadne-diagnostic-ok"


@dataclass(frozen=True, slots=True)
class OllamaDiagnosticReport:
    """Structured, credential-free diagnostic summary."""

    ollama_url: str
    model: str
    basic_chat: str
    advertised_capabilities: tuple[str, ...]
    configured_tool_mode: str
    native_tool_request_accepted: str
    native_tool_call_observed: str
    prompted_protocol: str
    registered_tools: tuple[str, ...]
    notes: tuple[str, ...]

    def format_text(self) -> str:
        caps = ", ".join(self.advertised_capabilities) or "(none reported)"
        tools = "\n".join(f"  - {name}" for name in self.registered_tools) or "  (none)"
        notes = "\n".join(f"  - {note}" for note in self.notes) or "  (none)"
        return (
            f"Ollama URL:\n{self.ollama_url}\n\n"
            f"Model:\n{self.model}\n\n"
            f"Basic chat:\n{self.basic_chat}\n\n"
            f"Advertised capabilities:\n{caps}\n\n"
            f"Configured Ariadne tool mode:\n{self.configured_tool_mode}\n\n"
            f"Native tool request accepted:\n{self.native_tool_request_accepted}\n\n"
            f"Native tool call actually observed:\n{self.native_tool_call_observed}\n\n"
            f"Prompted protocol:\n{self.prompted_protocol}\n\n"
            f"Registered tools:\n{tools}\n\n"
            f"Notes:\n{notes}\n"
        )


def registered_tool_definitions() -> list[ToolDefinition]:
    """Build the production tool catalogue schemas without constructing tools."""
    return [
        ToolDefinition(
            name="search_osm_knowledge",
            description="Search OSM Wiki documentation for tagging guidance.",
            parameters_schema=SearchOsmKnowledgeArgs.model_json_schema(),
        ),
        ToolDefinition(
            name="query_osm",
            description="Query live OpenStreetMap features via a structured feature query.",
            parameters_schema=OsmFeatureQuery.model_json_schema(),
        ),
        ToolDefinition(
            name="analyze_features",
            description="Run deterministic spatial analytics over request-scoped OSM datasets.",
            parameters_schema=AnalysisPlan.model_json_schema(),
        ),
        ToolDefinition(
            name="simbench_query",
            description="Load SimBench power-network metadata and map geometries.",
            parameters_schema=SimBenchQueryArgs.model_json_schema(),
        ),
        ToolDefinition(
            name="analyze_energy_grid",
            description="Run GeoLoadST analysis on a SimBench network.",
            parameters_schema=AnalyzeEnergyGridArgs.model_json_schema(),
        ),
    ]


async def run_ollama_diagnostics(settings: Settings) -> OllamaDiagnosticReport:
    """Probe local Ollama; never prints secrets or hidden reasoning."""
    notes: list[str] = []
    host = _safe_base_url(settings.ollama_base_url)
    capabilities = await _read_capabilities(settings)
    if capabilities:
        notes.append(
            "Advertised capabilities describe HTTP/protocol support, not reliable "
            "native tool-call emission."
        )

    basic = await _probe_basic_chat(settings)
    native_http, native_call = await _probe_native_tool_behavior(settings)
    prompted = await _probe_prompted_protocol(settings)
    tools = tuple(tool.name for tool in registered_tool_definitions())

    return OllamaDiagnosticReport(
        ollama_url=host,
        model=settings.ollama_model,
        basic_chat=basic,
        advertised_capabilities=capabilities,
        configured_tool_mode=settings.ollama_tool_mode,
        native_tool_request_accepted=native_http,
        native_tool_call_observed=native_call,
        prompted_protocol=prompted,
        registered_tools=tools,
        notes=tuple(notes),
    )


def _safe_base_url(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port
    return f"{host}:{port}" if port else host


async def _read_capabilities(settings: Settings) -> tuple[str, ...]:
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            response = await client.get(f"{settings.ollama_base_url.rstrip('/')}/api/show")
            # /api/show requires a body on some versions; fall back to tags.
            if response.status_code >= 400:
                response = await client.post(
                    f"{settings.ollama_base_url.rstrip('/')}/api/show",
                    json={"model": settings.ollama_model},
                )
            if response.status_code >= 400:
                return ()
            body = response.json()
    except (httpx.HTTPError, ValueError, TypeError):
        return ()
    caps = body.get("capabilities")
    if isinstance(caps, list):
        return tuple(str(item) for item in caps)
    return ()


async def _probe_basic_chat(settings: Settings) -> str:
    provider = OllamaProvider(
        OllamaConfig(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            num_ctx=settings.ollama_num_ctx,
            request_timeout_seconds=min(settings.ollama_request_timeout_seconds, 60.0),
            tool_call_mode="prompted",
        )
    )
    try:
        response = await provider.chat(
            [ChatMessage(role="user", content="Reply only with OK")],
        )
        if "OK" in response.content.upper() or response.content.strip():
            return "PASS"
        return "FAIL"
    except Exception as exc:
        return f"FAIL ({type(exc).__name__})"
    finally:
        await provider.aclose()


async def _probe_native_tool_behavior(settings: Settings) -> tuple[str, str]:
    """Return (http_accepted, tool_call_observed)."""
    secret_tool = ToolDefinition(
        name=_SECRET_TOOL_NAME,
        description=(
            "Return the server-side diagnostic value. The value is unavailable to "
            "the model and must be retrieved using the tool."
        ),
        parameters_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    )
    payload: dict[str, Any] = {
        "model": settings.ollama_model,
        "messages": [
            {
                "role": "user",
                "content": (
                    "You must call the get_secret_test_value tool. "
                    "Do not invent the value. Use the tool."
                ),
            }
        ],
        "stream": False,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": secret_tool.name,
                    "description": secret_tool.description,
                    "parameters": normalize_tool_parameters_schema(secret_tool.parameters_schema),
                },
            }
        ],
    }
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(90.0)) as client:
            response = await client.post(
                f"{settings.ollama_base_url.rstrip('/')}/api/chat",
                json=payload,
            )
    except httpx.HTTPError:
        return ("FAIL", "NOT TESTED")

    if response.status_code >= 400:
        return ("FAIL", "NOT TESTED")

    try:
        body = response.json()
    except ValueError:
        return ("PASS", "FAIL")
    message = body.get("message") if isinstance(body, dict) else None
    if not isinstance(message, dict):
        return ("PASS", "FAIL")
    # Never inspect or return thinking content.
    calls = message.get("tool_calls")
    observed = isinstance(calls, list) and any(
        isinstance(entry, dict)
        and isinstance(entry.get("function"), dict)
        and entry["function"].get("name") == _SECRET_TOOL_NAME
        for entry in calls
    )
    # Confirm the diagnostic value is not already present without a tool call.
    content = str(message.get("content") or "")
    if _SECRET_TOOL_VALUE in content and not observed:
        return ("PASS", "FAIL")
    return ("PASS", "PASS" if observed else "FAIL")


async def _probe_prompted_protocol(settings: Settings) -> str:
    """Probe parse → fake tool args → correction path with a harmless tool."""
    secret_tool = ToolDefinition(
        name=_SECRET_TOOL_NAME,
        description=(
            "Return the server-side diagnostic value. The value is unavailable "
            "without calling this tool. Arguments: empty object {}."
        ),
        parameters_schema={"type": "object", "properties": {}},
    )
    provider = OllamaProvider(
        OllamaConfig(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            num_ctx=settings.ollama_num_ctx,
            request_timeout_seconds=min(settings.ollama_request_timeout_seconds, 120.0),
            tool_call_mode="prompted",
        )
    )
    try:
        response = await provider.chat(
            [
                ChatMessage(
                    role="user",
                    content=(
                        "Reply with ONLY a JSON tool_calls object that calls "
                        f"{_SECRET_TOOL_NAME} with arguments {{}}."
                    ),
                )
            ],
            tools=[secret_tool],
        )
        if response.protocol_error:
            return f"FAIL (parse: {response.protocol_error})"
        if not response.has_tool_calls:
            if response.content:
                return "FAIL (final_answer instead of tool_calls)"
            return "FAIL (no tool_calls)"
        call = response.tool_calls[0]
        if call.name != _SECRET_TOOL_NAME:
            return f"FAIL (tool selected: {call.name})"
        if call.arguments not in ({},):
            # Empty object is required; anything else fails validation style check.
            return f"FAIL (arguments keys={sorted(call.arguments.keys())})"
        return "PASS"
    except Exception as exc:
        return f"FAIL ({type(exc).__name__}: {exc})"
    finally:
        await provider.aclose()


def dump_normalized_tool_catalogue() -> str:
    """Return compact JSON of normalised tool schemas for offline inspection."""
    specs = [
        {
            "name": tool.name,
            "description": tool.description,
            "parameters": normalize_tool_parameters_schema(tool.parameters_schema),
        }
        for tool in registered_tool_definitions()
    ]
    return json.dumps(specs, ensure_ascii=False, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class AvalAIDiagnosticReport:
    """Credential-free AvalAI / Gemini diagnostic summary."""

    provider: str
    host: str
    model: str
    configured_tool_mode: str
    effective_tool_mode: str
    basic_chat: str
    native_tool_request_accepted: str
    native_tool_call_observed: str
    notes: tuple[str, ...]

    def format_text(self) -> str:
        notes = "\n".join(f"  - {note}" for note in self.notes) or "  (none)"
        return (
            f"Provider:\n{self.provider}\n\n"
            f"Host:\n{self.host}\n\n"
            f"Model:\n{self.model}\n\n"
            f"Configured tool mode:\n{self.configured_tool_mode}\n\n"
            f"Effective tool mode:\n{self.effective_tool_mode}\n\n"
            f"Basic chat:\n{self.basic_chat}\n\n"
            f"Native tool request accepted:\n{self.native_tool_request_accepted}\n\n"
            f"Native tool call actually observed:\n{self.native_tool_call_observed}\n\n"
            f"Notes:\n{notes}\n"
        )


async def run_avalai_diagnostics(settings: Settings) -> AvalAIDiagnosticReport:
    """Probe AvalAI. Never prints the API key or hidden reasoning."""
    notes: list[str] = []
    if not (settings.avalai_api_key or "").strip():
        return AvalAIDiagnosticReport(
            provider="avalai",
            host=safe_avalai_host(settings.avalai_base_url),
            model=settings.avalai_model,
            configured_tool_mode=settings.avalai_tool_mode,
            effective_tool_mode=settings.avalai_tool_mode,
            basic_chat="FAIL (missing AVALAI_API_KEY)",
            native_tool_request_accepted="NOT TESTED",
            native_tool_call_observed="NOT TESTED",
            notes=("Set AVALAI_API_KEY before running this diagnostic.",),
        )

    basic = await _probe_avalai_basic_chat(settings)
    accepted, observed, effective_mode = await _probe_avalai_native_tools(settings)
    if observed != "PASS":
        notes.append("Accepting a tools field is not enough; native tool_calls were not observed.")
        if effective_mode == "prompted_fallback":
            notes.append("Provider fell back explicitly to prompted_fallback.")
    return AvalAIDiagnosticReport(
        provider="avalai",
        host=safe_avalai_host(settings.avalai_base_url),
        model=settings.avalai_model,
        configured_tool_mode=settings.avalai_tool_mode,
        effective_tool_mode=effective_mode,
        basic_chat=basic,
        native_tool_request_accepted=accepted,
        native_tool_call_observed=observed,
        notes=tuple(notes),
    )


def _secret_tool() -> ToolDefinition:
    return ToolDefinition(
        name=_SECRET_TOOL_NAME,
        description=(
            "Return the server-side diagnostic value. The value is unavailable to "
            "the model and must be retrieved using the tool."
        ),
        parameters_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    )


async def _probe_avalai_basic_chat(settings: Settings) -> str:
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
        if "Ariadne AvalAI connection OK" in response.content:
            return "PASS"
        if response.content.strip():
            return "FAIL (unexpected content)"
        return "FAIL (empty content)"
    except Exception as exc:
        return f"FAIL ({type(exc).__name__})"
    finally:
        await provider.aclose()


async def _probe_avalai_native_tools(settings: Settings) -> tuple[str, str, str]:
    """Return (request_accepted, tool_call_observed, effective_mode)."""
    provider = AvalAIProvider(
        AvalAIConfig(
            api_key=(settings.avalai_api_key or "").strip(),
            base_url=settings.avalai_base_url,
            model=settings.avalai_model,
            request_timeout_seconds=min(settings.avalai_timeout_seconds, 60.0),
            tool_call_mode="native",
            max_attempts=1,
        )
    )
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
            tools=[_secret_tool()],
        )
    except Exception as exc:
        await provider.aclose()
        return (f"FAIL ({type(exc).__name__})", "NOT TESTED", provider.tool_call_mode)

    await provider.aclose()
    observed = any(call.name == _SECRET_TOOL_NAME for call in response.tool_calls)
    if _SECRET_TOOL_VALUE in response.content and not observed:
        return ("PASS", "FAIL", provider.tool_call_mode)
    return ("PASS", "PASS" if observed else "FAIL", provider.tool_call_mode)
