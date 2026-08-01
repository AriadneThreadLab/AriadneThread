"""Application error hierarchy.

Errors carry a stable ``code`` so that API responses and tool observations can
report failures without leaking internal detail.
"""

from __future__ import annotations


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


class EmbeddingError(GeoAgentError):
    """The embedding provider failed or is unavailable."""

    code = "embedding_error"


class OverpassError(ToolExecutionError):
    """Overpass API request failed."""

    code = "overpass_error"


class OverpassQueryBuildError(GeoAgentError):
    """A validated request could not be turned into a supported Overpass query."""

    code = "overpass_query_build_error"


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
