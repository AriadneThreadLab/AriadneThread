"""``search_osm_knowledge`` - semantic search over the OSM documentation corpus.

This tool returns tagging guidance and Overpass concepts from stored wiki text.
Its observation states explicitly that the result is documentation, so the model
cannot present it as live map data.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.core.errors import EmbeddingError, GeoAgentError, ToolExecutionError
from app.rag.contracts import (
    MAX_RAG_TOP_K,
    OSM_KNOWLEDGE_DOMAIN,
    KnowledgeRetriever,
    RetrievedPassage,
)
from app.rag.retriever import RetrievalError
from app.tools.contracts import ToolOutcome

TOOL_NAME = "search_osm_knowledge"
TOOL_DESCRIPTION = (
    "Search the local OpenStreetMap documentation corpus (OSM wiki) for tagging "
    "conventions, tag meanings and Overpass concepts. Returns documentation "
    "passages, NOT live map features. Use it to decide which OSM tags to query."
)

MAX_TOP_K = MAX_RAG_TOP_K
_OBSERVATION_SNIPPET_CHARS = 240


class SearchOsmKnowledgeArgs(BaseModel):
    """Arguments accepted from the model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str = Field(
        min_length=2,
        max_length=500,
        description="What to look up, e.g. 'which tag marks a public park'.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=MAX_TOP_K,
        description="How many passages to return.",
    )


class SearchOsmKnowledgeResult(BaseModel):
    """Structured result kept for the API response."""

    model_config = ConfigDict(frozen=True)

    query: str
    domain: str = OSM_KNOWLEDGE_DOMAIN
    passages: list[RetrievedPassage]

    @property
    def passage_count(self) -> int:
        return len(self.passages)


def _summarise(passages: list[RetrievedPassage]) -> str:
    if not passages:
        return (
            "No OSM documentation passages matched. Do not guess tags; ask the "
            "user to narrow the request or try different wording."
        )
    lines = [f"Found {len(passages)} OSM documentation passage(s) (not live map data):"]
    for index, passage in enumerate(passages, start=1):
        heading = passage.document_title
        if passage.section:
            heading = f"{heading} > {passage.section}"
        snippet = " ".join(passage.content.split())[:_OBSERVATION_SNIPPET_CHARS]
        lines.append(f"{index}. [{heading}] {snippet}")
    return "\n".join(lines)


class SearchOsmKnowledgeTool:
    """Tool implementation delegating to a :class:`KnowledgeRetriever`."""

    def __init__(self, retriever: KnowledgeRetriever, *, default_top_k: int = 5) -> None:
        self._retriever = retriever
        self._default_top_k = default_top_k

    @property
    def name(self) -> str:
        return TOOL_NAME

    @property
    def description(self) -> str:
        return TOOL_DESCRIPTION

    @property
    def args_model(self) -> type[SearchOsmKnowledgeArgs]:
        return SearchOsmKnowledgeArgs

    async def execute(self, args: SearchOsmKnowledgeArgs) -> ToolOutcome[SearchOsmKnowledgeResult]:
        try:
            passages = await self._retriever.search(
                args.query,
                top_k=args.top_k or self._default_top_k,
                domain=OSM_KNOWLEDGE_DOMAIN,
            )
        except EmbeddingError as exc:
            raise ToolExecutionError(f"knowledge search unavailable: {exc.message}") from exc
        except RetrievalError as exc:
            raise ToolExecutionError(f"knowledge search failed: {exc.message}") from exc
        except GeoAgentError as exc:
            raise ToolExecutionError(f"knowledge search failed: {exc.message}") from exc
        result = SearchOsmKnowledgeResult(query=args.query, passages=passages)
        return ToolOutcome(observation=_summarise(passages), payload=result)
