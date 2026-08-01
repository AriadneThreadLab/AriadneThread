"""Create knowledge_documents and knowledge_chunks with PostGIS and pgvector.

Revision ID: 20260801_0001
Revises:
Create Date: 2026-08-01

Enables the postgis and vector extensions (idempotent) and creates the OSM
documentation persistence tables. No spatial feature tables are created: live
GeoJSON remains request-scoped for the MVP.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import VECTOR

revision: str = "20260801_0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

EMBEDDING_DIMENSION = 1024


def upgrade() -> None:
    # Extensions are created in the migration so a fresh database becomes usable
    # with `alembic upgrade head` alone. IF NOT EXISTS keeps re-runs safe.
    op.execute("CREATE EXTENSION IF NOT EXISTS postgis")
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "knowledge_documents",
        sa.Column(
            "id",
            sa.UUID(),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("domain", sa.String(length=64), nullable=False),
        sa.Column("title", sa.String(length=512), nullable=False),
        sa.Column("source_url", sa.String(length=1024), nullable=False),
        sa.Column("source_type", sa.String(length=64), nullable=False),
        sa.Column("license", sa.String(length=128), nullable=False),
        sa.Column("retrieved_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_knowledge_documents")),
        sa.UniqueConstraint(
            "domain",
            "source_url",
            name="uq_knowledge_documents_domain_source_url",
        ),
    )
    op.create_index(
        "ix_knowledge_documents_domain",
        "knowledge_documents",
        ["domain"],
        unique=False,
    )
    op.create_index(
        "ix_knowledge_documents_content_hash",
        "knowledge_documents",
        ["content_hash"],
        unique=False,
    )

    op.create_table(
        "knowledge_chunks",
        sa.Column(
            "id",
            sa.UUID(),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("document_id", sa.UUID(), nullable=False),
        sa.Column("section", sa.String(length=512), nullable=True),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("embedding", VECTOR(EMBEDDING_DIMENSION), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["knowledge_documents.id"],
            name=op.f("fk_knowledge_chunks_document_id_knowledge_documents"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_knowledge_chunks")),
        sa.UniqueConstraint(
            "document_id",
            "chunk_index",
            name="uq_knowledge_chunks_document_id_chunk_index",
        ),
    )
    op.create_index(
        "ix_knowledge_chunks_content_hash",
        "knowledge_chunks",
        ["content_hash"],
        unique=False,
    )
    op.create_index(
        "ix_knowledge_chunks_document_id",
        "knowledge_chunks",
        ["document_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_knowledge_chunks_document_id", table_name="knowledge_chunks")
    op.drop_index("ix_knowledge_chunks_content_hash", table_name="knowledge_chunks")
    op.drop_table("knowledge_chunks")
    op.drop_index("ix_knowledge_documents_content_hash", table_name="knowledge_documents")
    op.drop_index("ix_knowledge_documents_domain", table_name="knowledge_documents")
    op.drop_table("knowledge_documents")
    # Extensions are left in place: other objects (or a DBA) may depend on them.
