"""Application error hierarchy.

Errors carry a stable ``code`` so that API responses and tool observations can
report failures without leaking internal detail.
"""

from __future__ import annotations

import json


class GeoAgentError(Exception):
    """Base class for all application errors."""

    code = "geoagent_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ConfigurationError(GeoAgentError):
    """Invalid or missing configuration."""

    code = "configuration_error"


class ToolError(GeoAgentError):
    """Base class for failures raised while executing a tool."""

    code = "tool_error"


class ToolNotRegisteredError(ToolError):
    """The model requested a tool that is not in the registry."""

    code = "tool_not_registered"


class ToolArgumentError(ToolError):
    """The model supplied arguments that failed validation."""

    code = "tool_argument_error"


class ToolExecutionError(ToolError):
    """The tool ran but failed (upstream error, timeout, oversized response)."""

    code = "tool_execution_error"


class ToolTimeoutError(ToolExecutionError):
    """The tool exceeded its network or execution timeout."""

    code = "tool_timeout"


class LLMError(GeoAgentError):
    """The language model provider failed or returned an unusable response."""

    code = "llm_error"


class LLMTimeoutError(LLMError):
    """The language model call exceeded its timeout."""

    code = "llm_timeout"


class LLMProtocolError(LLMError):
    """The model response could not be parsed into the expected structure."""

    code = "llm_protocol_error"


class LLMToolProtocolError(LLMProtocolError):
    """The model's prompted/native tool-call envelope was invalid."""

    code = "llm_tool_protocol_error"


class OllamaUnreachableError(LLMError):
    """The Ollama HTTP endpoint could not be reached."""

    code = "ollama_unreachable"


class OllamaModelNotFoundError(LLMError):
    """The configured Ollama model tag is not installed."""

    code = "ollama_model_not_found"


class OllamaBadRequestError(LLMError):
    """Ollama rejected the chat request (HTTP 400)."""

    code = "ollama_bad_request"


class OllamaProtocolError(LLMProtocolError):
    """Ollama returned a response that violates the expected chat protocol."""

    code = "ollama_protocol_error"


class OllamaInvalidResponseError(LLMProtocolError):
    """Ollama returned a body that could not be interpreted as a chat reply."""

    code = "ollama_invalid_response"


class AvalAIError(LLMError):
    """AvalAI / OpenAI-compatible provider failed."""

    code = "avalai_error"


class AvalAIAuthenticationError(AvalAIError):
    """AvalAI rejected the API key (HTTP 401/403)."""

    code = "avalai_authentication_error"


class AvalAIBillingError(AvalAIError):
    """AvalAI rejected the request for credit / quota reasons."""

    code = "avalai_billing_error"


class AvalAIRateLimitError(AvalAIError):
    """AvalAI rate-limited the request (HTTP 429)."""

    code = "avalai_rate_limited"


class AvalAIUnavailableError(AvalAIError):
    """AvalAI could not be reached or returned a server error."""

    code = "avalai_unavailable"


class AvalAIInvalidModelError(AvalAIError):
    """The configured AvalAI model id is not available on the route."""

    code = "avalai_invalid_model"


class AvalAIInvalidResponseError(LLMProtocolError):
    """AvalAI returned a body that could not be interpreted as a chat reply."""

    code = "avalai_invalid_response"


class EmbeddingError(GeoAgentError):
    """The embedding provider failed or is unavailable."""

    code = "embedding_error"


class OverpassError(ToolExecutionError):
    """Overpass API request failed."""

    code = "overpass_error"

    def __init__(
        self,
        message: str,
        *,
        upstream_status: int | None = None,
        attempts: int = 1,
    ) -> None:
        super().__init__(message)
        self.upstream_status = upstream_status
        self.attempts = attempts


class OverpassTimeoutError(OverpassError):
    """Overpass timed out (HTTP 504 or client read timeout)."""

    code = "overpass_timeout"


class OverpassRateLimitedError(OverpassError):
    """Overpass rate-limited the request (HTTP 429)."""

    code = "overpass_rate_limited"


class OverpassBadResponseError(OverpassError):
    """Overpass returned an unusable body (empty, HTML, malformed JSON)."""

    code = "overpass_bad_response"


class OverpassUpstreamError(OverpassError):
    """Overpass returned a retryable/non-timeout upstream failure (e.g. 502/503)."""

    code = "overpass_upstream_error"


class OverpassQueryBuildError(GeoAgentError):
    """A validated request could not be turned into a supported Overpass query."""

    code = "overpass_query_build_error"


class PlaceResolutionError(ToolExecutionError):
    """Trusted place resolution failed."""

    code = "place_resolution_error"


class PlaceAmbiguousError(PlaceResolutionError):
    """Place resolution returned multiple incompatible candidates."""

    code = "place_ambiguous"


class ToolNotEligibleError(ToolError):
    """The model requested a tool that is registered but not currently eligible."""

    code = "tool_not_eligible"


class EnergyUnknownCapabilityError(ToolArgumentError):
    """``capability_id`` is not in the GeoLoadST capability registry."""

    code = "energy_unknown_capability"

    def __init__(
        self,
        message: str,
        *,
        received: str = "",
        allowed: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.received = received
        self.allowed_capabilities = allowed
        self.observation = json.dumps(
            {
                "error_code": "unknown_energy_capability",
                "received": received,
                "allowed_capabilities": list(allowed),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )


class EnergyNetworkLoadError(ToolError):
    """SimBench Connector could not load the requested network id."""

    code = "energy_network_load_failed"


class EnergyPluginUnavailableError(ToolError):
    """The GeoLoadST adapter package is not importable."""

    code = "energy_plugin_unavailable"

    def __init__(self, message: str, *, capability_id: str = "") -> None:
        super().__init__(message)
        self.capability_id = capability_id
        self.observation = _energy_observation(
            error_code=self.code,
            message=(
                "The GeoLoadST plugin is unavailable. Do not retry with a different "
                "capability_id; another scientific method would not answer the same question."
            ),
            capability_id=capability_id,
            do_not_replan=True,
        )


class EnergyPluginInternalError(ToolError):
    """Adapter, serialization, or GeoLoadST implementation failed after selection."""

    code = "energy_plugin_internal_error"

    def __init__(self, message: str, *, capability_id: str = "") -> None:
        super().__init__(message)
        self.capability_id = capability_id
        self.observation = _energy_observation(
            error_code=self.code,
            message=(
                "The selected GeoLoadST analysis failed due to an internal plugin or "
                "engine implementation error. Do not retry with a different "
                "capability_id; another scientific method would not answer the same question."
            ),
            capability_id=capability_id,
            do_not_replan=True,
        )


class ChartNormalizationError(ToolError):
    """Chart transformation failed. The scientific analysis itself may still be valid."""

    code = "chart_normalization_error"


class EnergyAnalysisInfeasibleError(ToolError):
    """The selected capability cannot run on the available dataset or binding."""

    code = "energy_analysis_infeasible"

    def __init__(self, message: str, *, capability_id: str = "") -> None:
        super().__init__(message)
        self.capability_id = capability_id
        self.observation = _energy_observation(
            error_code=self.code,
            message=(
                "The selected GeoLoadST capability is not feasible for the available "
                "data or binding. A different registered capability may be appropriate "
                "only if it answers the user's original question."
            ),
            capability_id=capability_id,
            do_not_replan=False,
        )


class EnergyAnalysisFailedError(ToolError):
    """The GeoLoadST adapter ran but did not produce a successful analysis."""

    code = "energy_analysis_failed"

    def __init__(self, message: str, *, capability_id: str = "") -> None:
        super().__init__(message)
        self.capability_id = capability_id
        self.observation = _energy_observation(
            error_code=self.code,
            message=(
                "GeoLoadST analysis failed. Do not invent results and do not switch to "
                "an unrelated scientific capability unless the observation says the "
                "selected method was infeasible."
            ),
            capability_id=capability_id,
            do_not_replan=True,
        )


def _energy_observation(
    *,
    error_code: str,
    message: str,
    capability_id: str = "",
    do_not_replan: bool = False,
) -> str:
    payload: dict[str, object] = {
        "error_code": error_code,
        "message": message,
        "do_not_replan": do_not_replan,
    }
    if capability_id:
        payload["capability_id"] = capability_id
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


class IngestionError(GeoAgentError):
    """OSM documentation ingestion failed."""

    code = "ingestion_error"


class WikiFetchError(IngestionError):
    """Fetching a page from the OSM wiki failed."""

    code = "wiki_fetch_error"


class WikiParseError(IngestionError):
    """A wiki response or article body could not be parsed."""

    code = "wiki_parse_error"


class CorpusWhitelistError(IngestionError):
    """Attempted to ingest a source that is not on the whitelist."""

    code = "corpus_whitelist_error"


class AgentRequestTimeoutError(GeoAgentError):
    """The bounded total agent HTTP request timeout was exceeded."""

    code = "agent_request_timeout"


class DependencyUnavailableError(GeoAgentError):
    """A required runtime dependency failed a readiness or execution check."""

    code = "dependency_unavailable"


class ActiveLearningError(GeoAgentError):
    """Base class for active-learning selection, review and export failures."""

    code = "active_learning_error"


class SanitizationError(ActiveLearningError):
    """Text or payload contained content that must never be retained."""

    code = "active_learning_sanitization_error"


class ReviewTransitionError(ActiveLearningError):
    """A review lifecycle transition is not permitted for this candidate."""

    code = "active_learning_review_transition_error"


class DatasetExportError(ActiveLearningError):
    """A curated dataset could not be exported."""

    code = "active_learning_dataset_export_error"


class IndicatorCatalogError(GeoAgentError):
    """An indicator definition failed catalog validation."""

    code = "indicator_catalog_error"


class UnknownIndicatorError(IndicatorCatalogError):
    """The requested indicator_id is not in the catalog."""

    code = "unknown_indicator"


class IndicatorComputeError(GeoAgentError):
    """An indicator executor could not run on the supplied datasets."""

    code = "indicator_compute_error"


class IndicatorPlanningError(GeoAgentError):
    """OSM data-requirement planning failed before any Overpass call."""

    code = "indicator_planning_error"

    def __init__(self, message: str, *, reason: str) -> None:
        super().__init__(message)
        self.reason = reason


class UnsupportedDataRequirementError(IndicatorPlanningError):
    """A requirement is not an OSM / place / derived-area input this planner supports."""

    code = "unsupported_data_requirement"

    def __init__(self, message: str) -> None:
        super().__init__(message, reason="unsupported_requirement")


class InventedOsmTagError(IndicatorPlanningError):
    """The model (or ungrounded input) tried to supply OSM tags."""

    code = "invented_osm_tag"

    def __init__(self, message: str) -> None:
        super().__init__(message, reason="invented_osm_tag")


class GroundingConflictError(IndicatorPlanningError):
    """RAG-proposed tags disagree with the catalog's declared feature set."""

    code = "grounding_conflict"

    def __init__(self, message: str) -> None:
        super().__init__(message, reason="grounding_conflict")


class RequirementBudgetExceededError(IndicatorPlanningError):
    """Unique OSM retrievals would exceed DatasetRegistry.MAX_DATASETS."""

    code = "requirement_budget_exceeded"

    def __init__(self, message: str) -> None:
        super().__init__(message, reason="requirement_budget_exceeded")
