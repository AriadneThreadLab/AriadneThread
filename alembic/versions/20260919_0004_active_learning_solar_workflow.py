"""Add solar_workflow JSONB to active-learning candidates.

Revision ID: 20260919_0004
Revises: 20260821_0003
Create Date: 2026-09-19

Stores SolarDSS routing snapshots for human review: structured decision,
indicator provenance copies, ranking profile, model planning, expert feedback
and preference labels. Active learning still does not compute or replace
PVGIS, pvlib, terrain, GIS-distance or MCDA values.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260919_0004"
down_revision: str | None = "20260821_0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "active_learning_candidates",
        sa.Column("solar_workflow", postgresql.JSONB(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("active_learning_candidates", "solar_workflow")
