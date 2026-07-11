"""persist background reminder timing-plan state

Revision ID: e4b7d9a1c612
Revises: d2f6a8c1e503
Create Date: 2026-07-11 00:00:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "e4b7d9a1c612"
down_revision = "d2f6a8c1e503"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    columns = {column["name"] for column in sa.inspect(bind).get_columns("reminders")}
    if "timing_plan_status" not in columns:
        op.add_column(
            "reminders",
            sa.Column("timing_plan_status", sa.String(), nullable=False, server_default="pending"),
        )
    if "timing_planned_at" not in columns:
        op.add_column("reminders", sa.Column("timing_planned_at", sa.String(), nullable=True))
    if "timing_plan_reason" not in columns:
        op.add_column("reminders", sa.Column("timing_plan_reason", sa.Text(), nullable=True))


def downgrade() -> None:
    bind = op.get_bind()
    columns = {column["name"] for column in sa.inspect(bind).get_columns("reminders")}
    for column in ("timing_plan_reason", "timing_planned_at", "timing_plan_status"):
        if column in columns:
            op.drop_column("reminders", column)
