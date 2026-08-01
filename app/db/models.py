"""ORM models for the OSM knowledge corpus.

These tables store *OSM documentation* only. Live geographic features are not
persisted here; they flow through the Overpass tool as GeoJSON in the request
result.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pgvector.sqlalchemy import VECTOR
from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.config import BGE_M3_EMBEDDING_DIM
from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

#: Dense embedding width. Must stay aligned with BGE-M3 and Settings.embedding_dim.
EMBEDDING_DIMENSION = BGE_M3_EMBEDDING_DIM


class KnowledgeDocument(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """One ingested OSM documentation page (or equivalent source)."""

    __tablename__ = "knowledge_documents"
    __table_args__ = (
        # One row per source within a domain: repeated ingestion upserts here
        # rather than inserting duplicates. content_hash then detects changes.
        UniqueConstraint(
            "domain",
            "source_url",
            name="uq_knowledge_documents_domain_source_url",
        ),
        Index("ix_knowledge_documents_domain", "domain"),
        Index("ix_knowledge_documents_content_hash", "content_hash"),
    )

    domain: Mapped[str] = mapped_column(String(64), nullable=False)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    source_url: Mapped[str] = mapped_column(String(1024), nullable=False)
    source_type: Mapped[str] = mapped_column(String(64), nullable=False)
    license: Mapped[str] = mapped_column(String(128), nullable=False)
    retrieved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)

    chunks: Mapped[list[KnowledgeChunk]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="KnowledgeChunk.chunk_index",
        lazy="selectin",
    )


class KnowledgeChunk(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """One heading-aware passage of a knowledge document, optionally embedded."""

    __tablename__ = "knowledge_chunks"
    __table_args__ = (
        # Stable position within a document. Refresh replaces chunks by index
        # (or deletes orphans) without leaving duplicate searchable rows.
        UniqueConstraint(
            "document_id",
            "chunk_index",
            name="uq_knowledge_chunks_document_id_chunk_index",
        ),
        Index("ix_knowledge_chunks_content_hash", "content_hash"),
        Index("ix_knowledge_chunks_document_id", "document_id"),
    )

    document_id: Mapped[UUID] = mapped_column(
        ForeignKey("knowledge_documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    section: Mapped[str | None] = mapped_column(String(512), nullable=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    # Nullable until Phase 3 indexing; dimension is fixed at the BGE-M3 width.
    embedding: Mapped[list[float] | None] = mapped_column(
        VECTOR(EMBEDDING_DIMENSION),
        nullable=True,
    )

    document: Mapped[KnowledgeDocument] = relationship(back_populates="chunks")
