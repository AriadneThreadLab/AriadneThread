"""Knowledge retrieval contracts.

Retrieval answers questions *about* OpenStreetMap (which tags describe a public
park, what ``natural=wood`` means, how Overpass QL works). It never returns
geographic features: those come only from the Overpass tool.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

#: The only knowledge domain in this project. Stored on every document so the
#: corpus can be extended later without mixing unrelated material.
OSM_KNOWLEDGE_DOMAIN = "osm_knowledge"

#: Hard upper bound for retrieval ``top_k`` (tool schema and retriever agree).
MAX_RAG_TOP_K = 20


class RetrievedPassage(BaseModel):
    """One scored chunk of OSM documentation."""

    model_config = ConfigDict(frozen=True)

    content: str = Field(description="The passage text.")
    document_title: str = Field(description="Title of the source document.")
    section: str | None = Field(default=None, description="Section heading within the document.")
    source_url: str = Field(description="Canonical OSM wiki URL of the document.")
    score: float = Field(description="Similarity score; higher is more relevant.")
    document_id: UUID | None = None
    chunk_id: UUID | None = None


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
