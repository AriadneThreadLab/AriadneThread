"""``search_solar_knowledge`` — semantic search over solar-PV documentation.

This tool returns methodology, standards, and regulatory notes from the
packaged corpus. Observations state that the result is documentation, never
SolarDSS indicators and never live map data.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.core.errors import EmbeddingError, GeoAgentError, ToolExecutionError
from app.rag.contracts import (
    MAX_RAG_TOP_K,
    SOLAR_PV_KNOWLEDGE_DOMAIN,
    KnowledgeRetriever,
    RetrievedPassage,
)
from app.rag.retriever import RetrievalError
from app.tools.context import ToolContext
from app.tools.contracts import ToolOutcome

TOOL_NAME = "search_solar_knowledge"
TOOL_DESCRIPTION = (
    "Search local solar-PV methodology, panel, installation, grid, permitting, "
    "regulatory, and standards notes. Arguments: required query (string), "
    "optional top_k (integer). Returns documentation only — never SolarDSS "
    "indicator values and never live map features. Use for definitions "
    "(GHI), why slope matters, exclusion-policy meaning, and what regulation "
    "affects a site."
)
SOLAR_KNOWLEDGE_ATTRIBUTION = (
    "Solar PV knowledge notes (CC-BY-4.0) with citations to public methodology "
    "sources. Not SolarDSS indicator values."
)

MAX_TOP_K = MAX_RAG_TOP_K
_LLM_OBSERVATION_MAX_PASSAGES = 5
_OBSERVATION_SNIPPET_CHARS = 160


class SearchSolarKnowledgeArgs(BaseModel):
    """Arguments accepted from the model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str = Field(
        min_length=2,
        max_length=500,
        description="What to look up, e.g. 'What does GHI mean?'.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=MAX_TOP_K,
        description="How many passages to return.",
    )


class SearchSolarKnowledgeResult(BaseModel):
    """Structured result kept for the API response."""

    model_config = ConfigDict(frozen=True)

    query: str
    domain: str = SOLAR_PV_KNOWLEDGE_DOMAIN
    passages: list[RetrievedPassage]

    @property
    def passage_count(self) -> int:
        return len(self.passages)


def _summarise_for_llm(passages: list[RetrievedPassage]) -> str:
    """Compact observation for the next planning turn (not the full UI list)."""
    if not passages:
        return (
            "status=empty source=solar_documentation evidence=knowledge_document "
            "No solar-PV documentation passages matched. Do not invent GHI, "
            "slope, or other SolarDSS indicator values. Reply with one JSON "
            "protocol object only."
        )
    ranked = sorted(passages, key=lambda item: item.score, reverse=True)
    grounding = ranked[:_LLM_OBSERVATION_MAX_PASSAGES]
    lines = [
        "status=ok source=solar_documentation evidence=knowledge_document",
        (
            "This is TOOL DATA: solar-PV knowledge / document evidence only — "
            "never SolarDSS indicators and never live map features."
        ),
        (
            f"Knowledge evidence (top {len(grounding)} of {len(passages)} retrieved; "
            "compact for the model):"
        ),
    ]
    for index, passage in enumerate(grounding, start=1):
        heading = passage.document_title
        if passage.section:
            heading = f"{heading} > {passage.section}"
        kind = passage.source_type or "solar_pv_knowledge"
        snippet = " ".join(passage.content.split())[:_OBSERVATION_SNIPPET_CHARS]
        lines.append(f"{index}. [{heading}] kind={kind} score={passage.score:.3f} {snippet}")
    lines.append(
        "Never use these passages as replacements for deterministic SolarDSS "
        "indicators (GHI, slope, elevation, road/grid distance, land cover, "
        "temperature, yield). In the final answer, label document citations as "
        "KNOWLEDGE / DOCUMENT EVIDENCE and engine outputs as DETERMINISTIC "
        "ANALYTICAL EVIDENCE. "
        "Respond with exactly one JSON object (tool_calls or final_answer)."
    )
    return "\n".join(lines)


class SearchSolarKnowledgeTool:
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
    def args_model(self) -> type[SearchSolarKnowledgeArgs]:
        return SearchSolarKnowledgeArgs

    async def execute(
        self,
        args: SearchSolarKnowledgeArgs,
        context: ToolContext,
    ) -> ToolOutcome[SearchSolarKnowledgeResult]:
        del context
        try:
            passages = await self._retriever.search(
                args.query,
                top_k=args.top_k or self._default_top_k,
                domain=SOLAR_PV_KNOWLEDGE_DOMAIN,
            )
        except EmbeddingError as exc:
            raise ToolExecutionError(f"solar knowledge search unavailable: {exc.message}") from exc
        except RetrievalError as exc:
            raise ToolExecutionError(f"solar knowledge search failed: {exc.message}") from exc
        except GeoAgentError as exc:
            raise ToolExecutionError(f"solar knowledge search failed: {exc.message}") from exc
        result = SearchSolarKnowledgeResult(query=args.query, passages=passages)
        return ToolOutcome(observation=_summarise_for_llm(passages), payload=result)
