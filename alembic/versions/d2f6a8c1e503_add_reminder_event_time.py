"""preserve event time separately from reminder notification time

Revision ID: d2f6a8c1e503
Revises: b6c2e8f9d401
Create Date: 2026-07-11 00:00:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "d2f6a8c1e503"
down_revision = "b6c2e8f9d401"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    columns = {column["name"] for column in sa.inspect(bind).get_columns("reminders")}
    if "event_time" not in columns:
        op.add_column("reminders", sa.Column("event_time", sa.String(), nullable=True))


def downgrade() -> None:
    bind = op.get_bind()
    columns = {column["name"] for column in sa.inspect(bind).get_columns("reminders")}
    if "event_time" in columns:
        op.drop_column("reminders", "event_time")
