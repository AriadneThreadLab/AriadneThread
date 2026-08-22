"""Create the active-learning candidate, review and dataset-export tables.

Revision ID: 20260816_0002
Revises: 20260801_0001
Create Date: 2026-08-16

These tables are deliberately separate from agent execution state. They hold
compact review material - user query, plans, tool sequence, validation
outcomes, review decisions - and never live GeoJSON, prompts, credentials or
model reasoning.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260816_0002"
down_revision: str | None = "20260801_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "active_learning_candidates",
        sa.Column(
            "id",
            sa.UUID(),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("candidate_id", sa.String(length=64), nullable=False),
        sa.Column("request_id", sa.String(length=64), nullable=False),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("user_query", sa.Text(), nullable=False),
        sa.Column("task_type", sa.String(length=32), nullable=False),
        sa.Column("task_signature", sa.String(length=200), nullable=False),
        sa.Column("query_hash", sa.String(length=64), nullable=False),
        sa.Column("analysis_plan", postgresql.JSONB(), nullable=True),
        sa.Column("comparison_plan", postgresql.JSONB(), nullable=True),
        sa.Column("model_output", sa.Text(), nullable=True),
        sa.Column("final_answer", sa.Text(), nullable=True),
        sa.Column("tool_sequence", postgresql.JSONB(), nullable=False),
        sa.Column("tool_validation_events", postgresql.JSONB(), nullable=False),
        sa.Column("place_resolution_events", postgresql.JSONB(), nullable=False),
        sa.Column("rag_evidence_summary", postgresql.JSONB(), nullable=True),
        sa.Column("metric_selection", postgresql.JSONB(), nullable=True),
        sa.Column("error_codes", postgresql.JSONB(), nullable=False),
        sa.Column("warnings", postgresql.JSONB(), nullable=False),
        sa.Column("external_error_codes", postgresql.JSONB(), nullable=False),
        sa.Column("selection_reasons", postgresql.JSONB(), nullable=False),
        sa.Column("informativeness_score", sa.Float(), nullable=False),
        sa.Column("duplicate_count", sa.Integer(), nullable=False),
        sa.Column("review_status", sa.String(length=16), nullable=False),
        sa.Column("review_notes", sa.Text(), nullable=True),
        sa.Column("reviewer", sa.String(length=64), nullable=True),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("corrected_output", postgresql.JSONB(), nullable=True),
        sa.Column("approved_for_training", sa.Boolean(), nullable=False),
        sa.Column("dataset_split", sa.String(length=16), nullable=True),
        sa.Column("model_id", sa.String(length=128), nullable=False),
        sa.Column("prompt_version", sa.String(length=64), nullable=False),
        sa.Column("tool_schema_version", sa.String(length=64), nullable=False),
        sa.Column("schema_version", sa.String(length=64), nullable=False),
        sa.Column("outcome_successful", sa.Boolean(), nullable=False),
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
        sa.PrimaryKeyConstraint("id", name=op.f("pk_active_learning_candidates")),
        sa.UniqueConstraint(
            "candidate_id",
            name="uq_active_learning_candidates_candidate_id",
        ),
    )
    op.create_index(
        "ix_active_learning_candidates_request_id",
        "active_learning_candidates",
        ["request_id"],
        unique=False,
    )
    op.create_index(
        "ix_active_learning_candidates_task_signature",
        "active_learning_candidates",
        ["task_signature"],
        unique=False,
    )
    op.create_index(
        "ix_active_learning_candidates_query_hash",
        "active_learning_candidates",
        ["query_hash"],
        unique=False,
    )
    op.create_index(
        "ix_active_learning_candidates_review_status",
        "active_learning_candidates",
        ["review_status"],
        unique=False,
    )
    op.create_index(
        "ix_active_learning_candidates_informativeness_score",
        "active_learning_candidates",
        ["informativeness_score"],
        unique=False,
    )

    op.create_table(
        "active_learning_reviews",
        sa.Column(
            "id",
            sa.UUID(),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("candidate_id", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("reviewer", sa.String(length=64), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("corrected_output", postgresql.JSONB(), nullable=True),
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
            ["candidate_id"],
            ["active_learning_candidates.candidate_id"],
            name=op.f("fk_active_learning_reviews_candidate_id_active_learning_candidates"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_active_learning_reviews")),
    )
    op.create_index(
        "ix_active_learning_reviews_candidate_id",
        "active_learning_reviews",
        ["candidate_id"],
        unique=False,
    )

    op.create_table(
        "training_dataset_exports",
        sa.Column(
            "id",
            sa.UUID(),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("dataset_version", sa.String(length=64), nullable=False),
        sa.Column("task", sa.String(length=32), nullable=False),
        sa.Column("schema_version", sa.String(length=64), nullable=False),
        sa.Column("exported_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("candidate_count", sa.Integer(), nullable=False),
        sa.Column("split_counts", postgresql.JSONB(), nullable=False),
        sa.Column("candidate_ids", postgresql.JSONB(), nullable=False),
        sa.Column("request_ids", postgresql.JSONB(), nullable=False),
        sa.Column("source_models", postgresql.JSONB(), nullable=False),
        sa.Column("content_digest", sa.String(length=64), nullable=False),
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
        sa.PrimaryKeyConstraint("id", name=op.f("pk_training_dataset_exports")),
        sa.UniqueConstraint(
            "dataset_version",
            "task",
            name="uq_training_dataset_exports_dataset_version_task",
        ),
    )


def downgrade() -> None:
    op.drop_table("training_dataset_exports")
    op.drop_index(
        "ix_active_learning_reviews_candidate_id",
        table_name="active_learning_reviews",
    )
    op.drop_table("active_learning_reviews")
    for index_name in (
        "ix_active_learning_candidates_informativeness_score",
        "ix_active_learning_candidates_review_status",
        "ix_active_learning_candidates_query_hash",
        "ix_active_learning_candidates_task_signature",
        "ix_active_learning_candidates_request_id",
    ):
        op.drop_index(index_name, table_name="active_learning_candidates")
    op.drop_table("active_learning_candidates")
