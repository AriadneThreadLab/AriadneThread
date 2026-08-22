"""``search_osm_knowledge`` - semantic search over the OSM documentation corpus.

This tool returns tagging guidance and Overpass concepts from stored wiki text.
Its observation states explicitly that the result is documentation, so the model
cannot present it as live map data.

The structured payload keeps the full retrieval list for the API/UI. The LLM
observation is a compact top-N grounding summary only.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field

from app.core.errors import EmbeddingError, GeoAgentError, ToolExecutionError
from app.rag.contracts import (
    MAX_RAG_TOP_K,
    OSM_KNOWLEDGE_DOMAIN,
    KnowledgeRetriever,
    RetrievedPassage,
)
from app.rag.retriever import RetrievalError
from app.tools.context import ToolContext
from app.tools.contracts import ToolOutcome

TOOL_NAME = "search_osm_knowledge"
TOOL_DESCRIPTION = (
    "Search local OSM Wiki documentation for tagging guidance. Arguments: "
    "required query (string), optional top_k (integer). Returns documentation "
    "passages only — never live map features. Use for ambiguous concepts before "
    "query_osm; skip when the user already gives an exact tag like leisure=park."
)

MAX_TOP_K = MAX_RAG_TOP_K
#: Compact model-facing grounding only; full passages stay on the payload/UI.
_LLM_OBSERVATION_MAX_PASSAGES = 5
_OBSERVATION_SNIPPET_CHARS = 140
_TAG_TITLE_RE = re.compile(
    r"^Tag:([A-Za-z0-9_:-]+)=([A-Za-z0-9_:-]+)$",
    re.IGNORECASE,
)


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


def _tag_hints(passages: list[RetrievedPassage]) -> list[str]:
    hints: list[str] = []
    seen: set[str] = set()
    for passage in passages:
        match = _TAG_TITLE_RE.match(passage.document_title.strip())
        if match is None:
            continue
        tag = f"{match.group(1)}={match.group(2)}"
        key = tag.lower()
        if key in seen:
            continue
        seen.add(key)
        hints.append(tag)
    return hints


def _summarise_for_llm(passages: list[RetrievedPassage]) -> str:
    """Compact observation for the next planning turn (not the full UI list)."""
    if not passages:
        return (
            "status=empty source=osm_documentation "
            "No OSM documentation passages matched. Do not guess tags; ask the "
            "user to narrow the request or try different wording. "
            "Reply with one JSON protocol object only."
        )
    ranked = sorted(passages, key=lambda item: item.score, reverse=True)
    grounding = ranked[:_LLM_OBSERVATION_MAX_PASSAGES]
    hints = _tag_hints(grounding)
    lines = [
        "status=ok source=osm_documentation",
        "This is TOOL DATA: OSM Wiki documentation only — never live map features.",
        (
            f"Grounding evidence (top {len(grounding)} of {len(passages)} retrieved; "
            "compact for the model):"
        ),
    ]
    for index, passage in enumerate(grounding, start=1):
        heading = passage.document_title
        if passage.section:
            heading = f"{heading} > {passage.section}"
        snippet = " ".join(passage.content.split())[:_OBSERVATION_SNIPPET_CHARS]
        lines.append(f"{index}. [{heading}] score={passage.score:.3f} {snippet}")
    if hints:
        lines.append("Documented OSM tags mentioned in titles: " + ", ".join(hints) + ".")
        if any(hint.lower() == "leisure=park" for hint in hints):
            lines.append(
                "For public parks, prefer leisure=park from this documentation. "
                "Do not invent public_park or amenity=park."
            )
    lines.append(
        "If the user asked for live city features, next call query_osm with place + "
        'tags like [{"key":"leisure","value":"park"}] + limit. '
        "For landmark radius comparisons, resolve_place each landmark first, then "
        "query_osm with place_ref_scope + the user radius_m. "
        "Respond with exactly one JSON object (tool_calls or final_answer)."
    )
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

    async def execute(
        self,
        args: SearchOsmKnowledgeArgs,
        context: ToolContext,
    ) -> ToolOutcome[SearchOsmKnowledgeResult]:
        del context  # request context unused; signature kept for Tool protocol.
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
        return ToolOutcome(observation=_summarise_for_llm(passages), payload=result)
