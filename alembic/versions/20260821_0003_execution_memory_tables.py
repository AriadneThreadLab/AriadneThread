"""Persistent Execution Memory tables.

Revision ID: 20260821_0003
Revises: 20260816_0002
Create Date: 2026-08-21

Structured analytical snapshots plus per-target OSM datasets. Separate from
active-learning review tables. Insert-only lineage via parent_execution_id.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260821_0003"
down_revision: str | None = "20260816_0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "execution_memories",
        sa.Column(
            "id",
            sa.UUID(),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("conversation_id", sa.String(length=64), nullable=False),
        sa.Column("request_id", sa.String(length=64), nullable=True),
        sa.Column("parent_execution_id", sa.UUID(), nullable=True),
        sa.Column("original_user_query", sa.Text(), nullable=False),
        sa.Column("analysis_type", sa.String(length=32), nullable=False),
        sa.Column("feature_concept", sa.String(length=120), nullable=False),
        sa.Column("outcome", sa.String(length=16), nullable=False),
        sa.Column("snapshot", postgresql.JSONB(), nullable=False),
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
            ["parent_execution_id"],
            ["execution_memories.id"],
            name="fk_execution_memories_parent_execution_id_execution_memories",
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_execution_memories"),
    )
    op.create_index(
        "ix_execution_memories_conversation_id",
        "execution_memories",
        ["conversation_id"],
    )
    op.create_index("ix_execution_memories_created_at", "execution_memories", ["created_at"])
    op.create_index(
        "ix_execution_memories_parent_execution_id",
        "execution_memories",
        ["parent_execution_id"],
    )
    op.create_index("ix_execution_memories_request_id", "execution_memories", ["request_id"])

    op.create_table(
        "execution_memory_datasets",
        sa.Column(
            "id",
            sa.UUID(),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("execution_id", sa.UUID(), nullable=False),
        sa.Column("execution_dataset_id", sa.UUID(), nullable=False),
        sa.Column("target_stable_id", sa.String(length=8), nullable=False),
        sa.Column("retrieved_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("truncated", sa.Boolean(), nullable=False),
        sa.Column("feature_count", sa.Integer(), nullable=False),
        sa.Column("provenance", postgresql.JSONB(), nullable=False),
        sa.Column("feature_collection", postgresql.JSONB(), nullable=False),
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
            ["execution_id"],
            ["execution_memories.id"],
            name="fk_execution_memory_datasets_execution_id_execution_memories",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_execution_memory_datasets"),
        sa.UniqueConstraint(
            "execution_dataset_id",
            name="uq_execution_memory_datasets_execution_dataset_id",
        ),
    )
    op.create_index(
        "ix_execution_memory_datasets_execution_id",
        "execution_memory_datasets",
        ["execution_id"],
    )
    op.create_index(
        "ix_execution_memory_datasets_target_stable_id",
        "execution_memory_datasets",
        ["target_stable_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_execution_memory_datasets_target_stable_id",
        table_name="execution_memory_datasets",
    )
    op.drop_index(
        "ix_execution_memory_datasets_execution_id",
        table_name="execution_memory_datasets",
    )
    op.drop_table("execution_memory_datasets")
    op.drop_index("ix_execution_memories_request_id", table_name="execution_memories")
    op.drop_index("ix_execution_memories_parent_execution_id", table_name="execution_memories")
    op.drop_index("ix_execution_memories_created_at", table_name="execution_memories")
    op.drop_index("ix_execution_memories_conversation_id", table_name="execution_memories")
    op.drop_table("execution_memories")
