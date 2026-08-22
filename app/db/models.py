"""ORM models for the OSM knowledge corpus and the active-learning store.

The knowledge tables store *OSM documentation* only. Live geographic features
are not persisted; they flow through the Overpass tool as GeoJSON in the
request result.

The active-learning tables are a separate concern from agent execution state:
they hold compact, reviewable summaries of finished runs (plans, tool sequences,
validation outcomes), never live datasets, prompts or model reasoning.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pgvector.sqlalchemy import VECTOR
from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PgUUID
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


class ActiveLearningCandidateRow(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """One finished run retained for human review.

    Compact by design: plans, tool sequences and validation outcomes, never
    GeoJSON payloads, prompts, credentials or model reasoning.
    """

    __tablename__ = "active_learning_candidates"
    __table_args__ = (
        UniqueConstraint("candidate_id", name="uq_active_learning_candidates_candidate_id"),
        Index("ix_active_learning_candidates_request_id", "request_id"),
        Index("ix_active_learning_candidates_task_signature", "task_signature"),
        Index("ix_active_learning_candidates_query_hash", "query_hash"),
        Index("ix_active_learning_candidates_review_status", "review_status"),
        Index("ix_active_learning_candidates_informativeness_score", "informativeness_score"),
    )

    candidate_id: Mapped[str] = mapped_column(String(64), nullable=False)
    request_id: Mapped[str] = mapped_column(String(64), nullable=False)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    user_query: Mapped[str] = mapped_column(Text, nullable=False)
    task_type: Mapped[str] = mapped_column(String(32), nullable=False)
    task_signature: Mapped[str] = mapped_column(String(200), nullable=False, default="")
    query_hash: Mapped[str] = mapped_column(String(64), nullable=False, default="")

    analysis_plan: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    comparison_plan: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    model_output: Mapped[str | None] = mapped_column(Text, nullable=True)
    final_answer: Mapped[str | None] = mapped_column(Text, nullable=True)

    tool_sequence: Mapped[list[Any]] = mapped_column(JSONB, nullable=False, default=list)
    tool_validation_events: Mapped[list[Any]] = mapped_column(JSONB, nullable=False, default=list)
    place_resolution_events: Mapped[list[Any]] = mapped_column(JSONB, nullable=False, default=list)
    rag_evidence_summary: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    metric_selection: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    error_codes: Mapped[list[Any]] = mapped_column(JSONB, nullable=False, default=list)
    warnings: Mapped[list[Any]] = mapped_column(JSONB, nullable=False, default=list)
    external_error_codes: Mapped[list[Any]] = mapped_column(JSONB, nullable=False, default=list)

    selection_reasons: Mapped[list[Any]] = mapped_column(JSONB, nullable=False, default=list)
    informativeness_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    duplicate_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    review_status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")
    review_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewer: Mapped[str | None] = mapped_column(String(64), nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    corrected_output: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    approved_for_training: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    dataset_split: Mapped[str | None] = mapped_column(String(16), nullable=True)

    model_id: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    prompt_version: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    tool_schema_version: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    schema_version: Mapped[str] = mapped_column(String(64), nullable=False)
    outcome_successful: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)


class ActiveLearningReviewRow(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Append-only audit of human review decisions."""

    __tablename__ = "active_learning_reviews"
    __table_args__ = (Index("ix_active_learning_reviews_candidate_id", "candidate_id"),)

    candidate_id: Mapped[str] = mapped_column(
        ForeignKey(
            "active_learning_candidates.candidate_id",
            ondelete="CASCADE",
        ),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    reviewer: Mapped[str | None] = mapped_column(String(64), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    corrected_output: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)


class TrainingDatasetExportRow(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Lineage of one exported, versioned dataset artifact."""

    __tablename__ = "training_dataset_exports"
    __table_args__ = (
        UniqueConstraint(
            "dataset_version",
            "task",
            name="uq_training_dataset_exports_dataset_version_task",
        ),
    )

    dataset_version: Mapped[str] = mapped_column(String(64), nullable=False)
    task: Mapped[str] = mapped_column(String(32), nullable=False)
    schema_version: Mapped[str] = mapped_column(String(64), nullable=False)
    exported_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    candidate_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    split_counts: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    candidate_ids: Mapped[list[Any]] = mapped_column(JSONB, nullable=False, default=list)
    request_ids: Mapped[list[Any]] = mapped_column(JSONB, nullable=False, default=list)
    source_models: Mapped[list[Any]] = mapped_column(JSONB, nullable=False, default=list)
    content_digest: Mapped[str] = mapped_column(String(64), nullable=False)


class ExecutionMemoryRow(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """One completed analytical execution. Never mutated after insert."""

    __tablename__ = "execution_memories"
    __table_args__ = (
        Index("ix_execution_memories_conversation_id", "conversation_id"),
        Index("ix_execution_memories_created_at", "created_at"),
        Index("ix_execution_memories_parent_execution_id", "parent_execution_id"),
        Index("ix_execution_memories_request_id", "request_id"),
    )

    conversation_id: Mapped[str] = mapped_column(String(64), nullable=False)
    request_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    parent_execution_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("execution_memories.id", ondelete="SET NULL"),
        nullable=True,
    )
    original_user_query: Mapped[str] = mapped_column(Text, nullable=False)
    analysis_type: Mapped[str] = mapped_column(String(32), nullable=False)
    feature_concept: Mapped[str] = mapped_column(String(120), nullable=False, default="")
    outcome: Mapped[str] = mapped_column(String(16), nullable=False)
    snapshot: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)

    datasets: Mapped[list[ExecutionMemoryDatasetRow]] = relationship(
        back_populates="execution",
        cascade="all, delete-orphan",
        lazy="selectin",
    )


class ExecutionMemoryDatasetRow(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Persistent OSM dataset for one execution target (GeoJSON + provenance)."""

    __tablename__ = "execution_memory_datasets"
    __table_args__ = (
        Index("ix_execution_memory_datasets_execution_id", "execution_id"),
        Index("ix_execution_memory_datasets_target_stable_id", "target_stable_id"),
    )

    execution_id: Mapped[UUID] = mapped_column(
        ForeignKey("execution_memories.id", ondelete="CASCADE"),
        nullable=False,
    )
    execution_dataset_id: Mapped[UUID] = mapped_column(
        PgUUID(as_uuid=True),
        nullable=False,
        unique=True,
    )
    target_stable_id: Mapped[str] = mapped_column(String(8), nullable=False)
    retrieved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    truncated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    feature_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    provenance: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    feature_collection: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)

    execution: Mapped[ExecutionMemoryRow] = relationship(back_populates="datasets")
