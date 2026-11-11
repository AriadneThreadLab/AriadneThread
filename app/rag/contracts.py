"""Knowledge retrieval contracts.

Retrieval answers questions *about* OpenStreetMap tagging and *about* solar-PV
methodology, standards, and regulation. It never returns geographic features
and it never returns SolarDSS indicator values.
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

#: OSM Wiki tagging and Overpass documentation.
OSM_KNOWLEDGE_DOMAIN = "osm_knowledge"

#: Curated solar-PV methodology, standards, and regulatory notes.
SOLAR_PV_KNOWLEDGE_DOMAIN = "solar_pv_knowledge"

ALLOWED_KNOWLEDGE_DOMAINS: frozenset[str] = frozenset(
    {OSM_KNOWLEDGE_DOMAIN, SOLAR_PV_KNOWLEDGE_DOMAIN}
)

#: Hard upper bound for retrieval ``top_k`` (tool schema and retriever agree).
MAX_RAG_TOP_K = 20

EvidenceClass = Literal["deterministic_analytical", "knowledge_document"]
KNOWLEDGE_EVIDENCE_CLASS: Literal["knowledge_document"] = "knowledge_document"


class RetrievedPassage(BaseModel):
    """One scored chunk of documentation (OSM or solar-PV)."""

    model_config = ConfigDict(frozen=True)

    content: str = Field(description="The passage text.")
    document_title: str = Field(description="Title of the source document.")
    section: str | None = Field(default=None, description="Section heading within the document.")
    source_url: str = Field(description="Canonical citation URL of the document.")
    score: float = Field(description="Similarity score; higher is more relevant.")
    document_id: UUID | None = None
    chunk_id: UUID | None = None
    domain: str = OSM_KNOWLEDGE_DOMAIN
    source_type: str | None = Field(
        default=None,
        description="Document kind (osm_wiki, pv_methodology, grid_rules, …).",
    )
    evidence_class: Literal["knowledge_document"] = KNOWLEDGE_EVIDENCE_CLASS


@runtime_checkable
class KnowledgeRetriever(Protocol):
    """Semantic search over the stored OSM documentation corpus."""

    async def search(
        self,
        query: str,
        *,
        top_k: int,
        domain: str = OSM_KNOWLEDGE_DOMAIN,
    ) -> list[RetrievedPassage]:
        """Return the ``top_k`` most similar passages, best first."""
        ...
