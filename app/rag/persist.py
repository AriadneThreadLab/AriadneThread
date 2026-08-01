"""Idempotent persistence of prepared OSM documentation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.models import KnowledgeChunk, KnowledgeDocument
from app.rag.prepare import PreparedDocument

PersistStatus = Literal["inserted", "updated", "unchanged"]


@dataclass(frozen=True, slots=True)
class ChunkSnapshot:
    """Stored chunk fields needed for change detection."""

    section: str | None
    content: str
    chunk_index: int
    content_hash: str


@dataclass(frozen=True, slots=True)
class DocumentSnapshot:
    """Stored document fields needed for change detection."""

    id: UUID
    content_hash: str
    chunks: tuple[ChunkSnapshot, ...]


@dataclass(frozen=True, slots=True)
class PersistResult:
    """Outcome of persisting one prepared document."""

    status: PersistStatus
    document_id: str
    chunk_count: int
    source_url: str


class KnowledgeRepository(Protocol):
    """Persistence boundary used by ingestion (SQLAlchemy or in-memory)."""

    async def find(self, domain: str, source_url: str) -> DocumentSnapshot | None: ...

    async def insert(self, prepared: PreparedDocument) -> UUID: ...

    async def replace(self, document_id: UUID, prepared: PreparedDocument) -> None: ...


def chunks_match(existing: tuple[ChunkSnapshot, ...], prepared: PreparedDocument) -> bool:
    """Whether stored chunks are identical to the prepared set."""
    if len(existing) != len(prepared.chunks):
        return False
    ordered = sorted(existing, key=lambda chunk: chunk.chunk_index)
    for current, expected in zip(ordered, prepared.chunks, strict=True):
        if (
            current.chunk_index != expected.chunk_index
            or current.content_hash != expected.content_hash
            or current.section != expected.section
            or current.content != expected.content
        ):
            return False
    return True


async def persist_document(
    repository: KnowledgeRepository, prepared: PreparedDocument
) -> PersistResult:
    """Insert or refresh one document and its chunks.

    Unchanged documents (same content hash and identical chunks) are left alone.
    Changed documents replace all chunks so superseded passages cannot remain
    searchable. Embeddings are not written in this phase.
    """
    existing = await repository.find(prepared.domain, prepared.source_url)

    if (
        existing is not None
        and existing.content_hash == prepared.content_hash
        and chunks_match(existing.chunks, prepared)
    ):
        return PersistResult(
            status="unchanged",
            document_id=str(existing.id),
            chunk_count=len(existing.chunks),
            source_url=prepared.source_url,
        )

    if existing is None:
        document_id = await repository.insert(prepared)
        status: PersistStatus = "inserted"
    else:
        document_id = existing.id
        await repository.replace(document_id, prepared)
        status = "updated"

    return PersistResult(
        status=status,
        document_id=str(document_id),
        chunk_count=len(prepared.chunks),
        source_url=prepared.source_url,
    )


class SqlAlchemyKnowledgeRepository:
    """PostgreSQL-backed repository used by the CLI ingestion command."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def find(self, domain: str, source_url: str) -> DocumentSnapshot | None:
        result = await self._session.execute(
            select(KnowledgeDocument)
            .options(selectinload(KnowledgeDocument.chunks))
            .where(
                KnowledgeDocument.domain == domain,
                KnowledgeDocument.source_url == source_url,
            )
        )
        document = result.scalar_one_or_none()
        if document is None:
            return None
        chunks = tuple(
            ChunkSnapshot(
                section=chunk.section,
                content=chunk.content,
                chunk_index=chunk.chunk_index,
                content_hash=chunk.content_hash,
            )
            for chunk in document.chunks
        )
        return DocumentSnapshot(id=document.id, content_hash=document.content_hash, chunks=chunks)

    async def insert(self, prepared: PreparedDocument) -> UUID:
        document = KnowledgeDocument(
            domain=prepared.domain,
            title=prepared.title,
            source_url=prepared.source_url,
            source_type=prepared.source_type,
            license=prepared.license,
            retrieved_at=prepared.retrieved_at,
            content_hash=prepared.content_hash,
        )
        self._session.add(document)
        await self._session.flush()
        self._add_chunks(document.id, prepared)
        await self._session.flush()
        return document.id

    async def replace(self, document_id: UUID, prepared: PreparedDocument) -> None:
        result = await self._session.execute(
            select(KnowledgeDocument)
            .options(selectinload(KnowledgeDocument.chunks))
            .where(KnowledgeDocument.id == document_id)
        )
        document = result.scalar_one()
        document.title = prepared.title
        document.source_type = prepared.source_type
        document.license = prepared.license
        document.retrieved_at = prepared.retrieved_at
        document.content_hash = prepared.content_hash
        document.chunks.clear()
        await self._session.flush()
        self._add_chunks(document.id, prepared)
        await self._session.flush()

    def _add_chunks(self, document_id: UUID, prepared: PreparedDocument) -> None:
        for chunk in prepared.chunks:
            self._session.add(
                KnowledgeChunk(
                    document_id=document_id,
                    section=chunk.section,
                    content=chunk.content,
                    chunk_index=chunk.chunk_index,
                    content_hash=chunk.content_hash,
                    embedding=None,
                )
            )


@dataclass
class InMemoryKnowledgeRepository:
    """Offline repository for unit tests."""

    documents: dict[tuple[str, str], DocumentSnapshot] = field(default_factory=dict)
    bodies: dict[UUID, PreparedDocument] = field(default_factory=dict)

    async def find(self, domain: str, source_url: str) -> DocumentSnapshot | None:
        return self.documents.get((domain, source_url))

    async def insert(self, prepared: PreparedDocument) -> UUID:
        document_id = uuid4()
        self._store(document_id, prepared)
        return document_id

    async def replace(self, document_id: UUID, prepared: PreparedDocument) -> None:
        self._store(document_id, prepared)

    def _store(self, document_id: UUID, prepared: PreparedDocument) -> None:
        snapshot = DocumentSnapshot(
            id=document_id,
            content_hash=prepared.content_hash,
            chunks=tuple(
                ChunkSnapshot(
                    section=chunk.section,
                    content=chunk.content,
                    chunk_index=chunk.chunk_index,
                    content_hash=chunk.content_hash,
                )
                for chunk in prepared.chunks
            ),
        )
        self.documents[(prepared.domain, prepared.source_url)] = snapshot
        self.bodies[document_id] = prepared
