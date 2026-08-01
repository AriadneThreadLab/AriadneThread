"""Batch embedding of knowledge chunks into pgvector.

Only chunks with ``embedding IS NULL`` are indexed by default. Re-ingestion
replaces chunk rows, so changed content naturally reappears as NULL and is
picked up on the next index run. ``force=True`` re-embeds every chunk.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import EmbeddingError
from app.db.models import EMBEDDING_DIMENSION, KnowledgeChunk, KnowledgeDocument
from app.db.session import Database
from app.embeddings.contracts import EmbeddingProvider, Vector
from app.rag.contracts import OSM_KNOWLEDGE_DOMAIN

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ChunkForIndex:
    """Minimal chunk payload needed to embed and write a vector."""

    chunk_id: UUID
    content: str
    content_hash: str
    has_embedding: bool


@dataclass(frozen=True, slots=True)
class IndexBatchResult:
    """Counters for one indexing run."""

    selected: int = 0
    embedded: int = 0
    skipped: int = 0
    batches: int = 0


class ChunkIndexStore(Protocol):
    """Persistence boundary for indexing (SQLAlchemy or in-memory)."""

    async def list_chunks(
        self, *, domain: str, force: bool, limit: int | None = None
    ) -> list[ChunkForIndex]: ...

    async def save_embeddings(self, updates: list[tuple[UUID, Vector]]) -> None: ...


def validate_embedding_vector(vector: Vector, *, expected_dim: int = EMBEDDING_DIMENSION) -> Vector:
    """Reject vectors that cannot be stored in ``VECTOR(1024)``."""
    if len(vector) != expected_dim:
        raise EmbeddingError(f"Expected {expected_dim}-dim embeddings, got {len(vector)}")
    if not all(isinstance(value, float) and value == value for value in vector):  # NaN check
        raise EmbeddingError("embedding contains non-finite values")
    return vector


async def index_chunks(
    store: ChunkIndexStore,
    embedder: EmbeddingProvider,
    *,
    domain: str = OSM_KNOWLEDGE_DOMAIN,
    batch_size: int = 8,
    force: bool = False,
) -> IndexBatchResult:
    """Embed pending chunks and persist their vectors in batches."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if embedder.dimension != EMBEDDING_DIMENSION:
        raise EmbeddingError(
            f"embedder dimension {embedder.dimension} does not match VECTOR({EMBEDDING_DIMENSION})"
        )

    chunks = await store.list_chunks(domain=domain, force=force)
    if not force:
        pending = [chunk for chunk in chunks if not chunk.has_embedding]
        skipped = len(chunks) - len(pending)
    else:
        pending = list(chunks)
        skipped = 0

    if not pending:
        logger.info("index: nothing to embed (skipped=%s)", skipped)
        return IndexBatchResult(selected=0, embedded=0, skipped=skipped, batches=0)

    embedded = 0
    batches = 0
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        texts = [chunk.content for chunk in batch]
        vectors = await embedder.embed_documents(texts)
        if len(vectors) != len(batch):
            raise EmbeddingError(
                f"embedder returned {len(vectors)} vectors for {len(batch)} chunks"
            )
        updates: list[tuple[UUID, Vector]] = []
        for chunk, vector in zip(batch, vectors, strict=True):
            updates.append((chunk.chunk_id, validate_embedding_vector(vector)))
        await store.save_embeddings(updates)
        embedded += len(updates)
        batches += 1
        logger.info(
            "index batch %s: embedded=%s/%s domain=%s",
            batches,
            embedded,
            len(pending),
            domain,
        )

    return IndexBatchResult(
        selected=len(pending),
        embedded=embedded,
        skipped=skipped,
        batches=batches,
    )


class SqlAlchemyChunkIndexStore:
    """PostgreSQL-backed chunk indexing store."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def list_chunks(
        self, *, domain: str, force: bool, limit: int | None = None
    ) -> list[ChunkForIndex]:
        # Select a boolean rather than the vector itself so skip counts are
        # accurate without pulling 1024-d embeddings into the application.
        has_embedding = KnowledgeChunk.embedding.is_not(None)
        stmt = (
            select(
                KnowledgeChunk.id,
                KnowledgeChunk.content,
                KnowledgeChunk.content_hash,
                has_embedding.label("has_embedding"),
            )
            .join(KnowledgeDocument, KnowledgeDocument.id == KnowledgeChunk.document_id)
            .where(KnowledgeDocument.domain == domain)
            .order_by(KnowledgeChunk.id)
        )
        del force  # filtering by has_embedding happens in index_chunks
        if limit is not None:
            stmt = stmt.limit(limit)
        rows = (await self._session.execute(stmt)).all()
        return [
            ChunkForIndex(
                chunk_id=row.id,
                content=row.content,
                content_hash=row.content_hash,
                has_embedding=bool(row.has_embedding),
            )
            for row in rows
        ]

    async def save_embeddings(self, updates: list[tuple[UUID, Vector]]) -> None:
        for chunk_id, vector in updates:
            await self._session.execute(
                update(KnowledgeChunk).where(KnowledgeChunk.id == chunk_id).values(embedding=vector)
            )
        await self._session.flush()


@dataclass
class InMemoryChunkIndexStore:
    """Offline store for indexing unit tests."""

    chunks: list[ChunkForIndex]
    embeddings: dict[UUID, Vector]

    async def list_chunks(
        self, *, domain: str, force: bool, limit: int | None = None
    ) -> list[ChunkForIndex]:
        del domain, force  # domain filtering is the caller's responsibility in memory tests
        items = list(self.chunks)
        if limit is not None:
            items = items[:limit]
        return items

    async def save_embeddings(self, updates: list[tuple[UUID, Vector]]) -> None:
        refreshed: list[ChunkForIndex] = []
        by_id = {chunk.chunk_id: chunk for chunk in self.chunks}
        for chunk_id, vector in updates:
            self.embeddings[chunk_id] = vector
            current = by_id[chunk_id]
            refreshed.append(
                ChunkForIndex(
                    chunk_id=current.chunk_id,
                    content=current.content,
                    content_hash=current.content_hash,
                    has_embedding=True,
                )
            )
            by_id[chunk_id] = refreshed[-1]
        self.chunks = [by_id[chunk.chunk_id] for chunk in self.chunks]


async def index_osm_knowledge(
    database: Database,
    embedder: EmbeddingProvider,
    *,
    domain: str = OSM_KNOWLEDGE_DOMAIN,
    batch_size: int = 8,
    force: bool = False,
) -> IndexBatchResult:
    """Index pending OSM knowledge chunks inside one database transaction."""
    async with database.session() as session:
        store = SqlAlchemyChunkIndexStore(session)
        return await index_chunks(
            store,
            embedder,
            domain=domain,
            batch_size=batch_size,
            force=force,
        )
