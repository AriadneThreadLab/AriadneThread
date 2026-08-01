"""pgvector dense retrieval over the OSM knowledge corpus.

Score semantics: cosine *similarity* in roughly ``[0, 1]`` for L2-normalised
BGE-M3 vectors (``similarity = 1 - cosine_distance``). Higher is more relevant.
Results are ordered by similarity descending.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from sqlalchemy import Select, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import EmbeddingError, GeoAgentError
from app.db.models import KnowledgeChunk, KnowledgeDocument
from app.db.session import Database
from app.embeddings.contracts import EmbeddingProvider, Vector
from app.rag.contracts import MAX_RAG_TOP_K, OSM_KNOWLEDGE_DOMAIN, RetrievedPassage


class RetrievalError(GeoAgentError):
    """Semantic retrieval failed."""

    code = "retrieval_error"


@dataclass(frozen=True, slots=True)
class ScoredChunkRow:
    """One candidate row before mapping to :class:`RetrievedPassage`."""

    chunk_id: UUID
    document_id: UUID
    content: str
    section: str | None
    document_title: str
    source_url: str
    cosine_similarity: float


class PassageStore(Protocol):
    """Persistence boundary for similarity search."""

    async def search_similar(
        self,
        query_vector: Vector,
        *,
        domain: str,
        top_k: int,
    ) -> list[ScoredChunkRow]: ...


def validate_top_k(top_k: int, *, maximum: int = MAX_RAG_TOP_K) -> int:
    if top_k < 1:
        raise RetrievalError("top_k must be at least 1")
    if top_k > maximum:
        raise RetrievalError(f"top_k must be <= {maximum}")
    return top_k


def cosine_similarity_from_distance(distance: float) -> float:
    """Convert pgvector cosine distance to similarity (higher is better)."""
    return 1.0 - float(distance)


class PgVectorPassageStore:
    """Cosine-distance search using pgvector over ``knowledge_chunks``."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def search_similar(
        self,
        query_vector: Vector,
        *,
        domain: str,
        top_k: int,
    ) -> list[ScoredChunkRow]:
        distance = KnowledgeChunk.embedding.cosine_distance(query_vector)
        stmt: Select[tuple[UUID, UUID, str, str | None, str, str, float]] = (
            select(
                KnowledgeChunk.id,
                KnowledgeChunk.document_id,
                KnowledgeChunk.content,
                KnowledgeChunk.section,
                KnowledgeDocument.title,
                KnowledgeDocument.source_url,
                distance.label("cosine_distance"),
            )
            .join(KnowledgeDocument, KnowledgeDocument.id == KnowledgeChunk.document_id)
            .where(KnowledgeDocument.domain == domain)
            .where(KnowledgeChunk.embedding.is_not(None))
            .order_by(distance)
            .limit(top_k)
        )
        rows = (await self._session.execute(stmt)).all()
        return [
            ScoredChunkRow(
                chunk_id=row.id,
                document_id=row.document_id,
                content=row.content,
                section=row.section,
                document_title=row.title,
                source_url=row.source_url,
                cosine_similarity=cosine_similarity_from_distance(row.cosine_distance),
            )
            for row in rows
        ]


@dataclass
class InMemoryPassageStore:
    """Offline store for retrieval unit tests."""

    rows: list[ScoredChunkRow]
    domain: str = OSM_KNOWLEDGE_DOMAIN

    async def search_similar(
        self,
        query_vector: Vector,
        *,
        domain: str,
        top_k: int,
    ) -> list[ScoredChunkRow]:
        del query_vector
        if domain != self.domain:
            return []
        ordered = sorted(self.rows, key=lambda row: row.cosine_similarity, reverse=True)
        return ordered[:top_k]


class PgVectorKnowledgeRetriever:
    """Embed a query with BGE-M3 and retrieve OSM documentation passages."""

    def __init__(
        self,
        store: PassageStore,
        embedder: EmbeddingProvider,
        *,
        default_domain: str = OSM_KNOWLEDGE_DOMAIN,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._default_domain = default_domain

    async def search(
        self,
        query: str,
        *,
        top_k: int,
        domain: str = OSM_KNOWLEDGE_DOMAIN,
    ) -> list[RetrievedPassage]:
        text = query.strip()
        if len(text) < 2:
            raise RetrievalError("query must be a non-empty string")
        limit = validate_top_k(top_k)
        effective_domain = domain or self._default_domain
        if effective_domain != OSM_KNOWLEDGE_DOMAIN:
            # MVP corpus is a single domain; reject surprise domains early.
            raise RetrievalError(f"unsupported knowledge domain: {effective_domain}")

        try:
            query_vector = await self._embedder.embed_query(text)
        except EmbeddingError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            raise EmbeddingError(f"query embedding failed: {exc}") from exc

        rows = await self._store.search_similar(query_vector, domain=effective_domain, top_k=limit)
        return [
            RetrievedPassage(
                content=row.content,
                document_title=row.document_title,
                section=row.section,
                source_url=row.source_url,
                score=row.cosine_similarity,
                document_id=row.document_id,
                chunk_id=row.chunk_id,
            )
            for row in rows
        ]


def build_pgvector_retriever(
    database: Database,
    embedder: EmbeddingProvider,
    *,
    session: AsyncSession,
) -> PgVectorKnowledgeRetriever:
    """Build a retriever bound to an open SQLAlchemy session."""
    return PgVectorKnowledgeRetriever(PgVectorPassageStore(session), embedder)


class SessionBoundKnowledgeRetriever:
    """Opens a short-lived DB session for each search call."""

    def __init__(self, database: Database, embedder: EmbeddingProvider) -> None:
        self._database = database
        self._embedder = embedder

    async def search(
        self,
        query: str,
        *,
        top_k: int,
        domain: str = OSM_KNOWLEDGE_DOMAIN,
    ) -> list[RetrievedPassage]:
        async with self._database.session() as session:
            retriever = PgVectorKnowledgeRetriever(PgVectorPassageStore(session), self._embedder)
            return await retriever.search(query, top_k=top_k, domain=domain)
