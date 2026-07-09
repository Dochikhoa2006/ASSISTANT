"""release 4 operations recurrence

Revision ID: a4c9e7d1b2f6
Revises: f3b7c9d2a610
Create Date: 2026-07-09 00:00:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "a4c9e7d1b2f6"
down_revision = "f3b7c9d2a610"
branch_labels = None
depends_on = None


def _add_column_if_missing(table_name: str, column: sa.Column) -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {col["name"] for col in inspector.get_columns(table_name)}
    if column.name not in columns:
        op.add_column(table_name, column)


def _create_index_if_missing(index_name: str, table_name: str, columns: list[str]) -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    indexes = {idx["name"] for idx in inspector.get_indexes(table_name)}
    if index_name not in indexes:
        op.create_index(index_name, table_name, columns)


def _drop_index_if_present(index_name: str, table_name: str) -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    indexes = {idx["name"] for idx in inspector.get_indexes(table_name)}
    if index_name in indexes:
        op.drop_index(index_name, table_name=table_name)


def upgrade() -> None:
    _add_column_if_missing("reminders", sa.Column("recurrence_rule", sa.Text(), nullable=True))
    _add_column_if_missing("reminders", sa.Column("recurrence_timezone", sa.String(), nullable=True))
    _add_column_if_missing("reminders", sa.Column("next_fire_time", sa.String(), nullable=True))
    _add_column_if_missing("reminders", sa.Column("last_fire_time", sa.String(), nullable=True))
    _add_column_if_missing("reminders", sa.Column("parent_recurring_reminder_id", sa.String(), nullable=True))
    _add_column_if_missing("reminder_notifications", sa.Column("fire_time", sa.String(), nullable=True))

    _create_index_if_missing(
        "idx_reminders_user_status_next_fire",
        "reminders",
        ["user_id", "status", "next_fire_time"],
    )
    _create_index_if_missing(
        "idx_reminders_user_status_time",
        "reminders",
        ["user_id", "status", "reminder_time"],
    )
    _create_index_if_missing(
        "idx_reminders_user_parent_recurring",
        "reminders",
        ["user_id", "parent_recurring_reminder_id"],
    )
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_reminders_due_coalesce
            ON reminders (user_id, status, COALESCE(next_fire_time, reminder_time))
            """
        )


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("DROP INDEX IF EXISTS idx_reminders_due_coalesce")
    _drop_index_if_present("idx_reminders_user_parent_recurring", "reminders")
    _drop_index_if_present("idx_reminders_user_status_next_fire", "reminders")
    # Keep idx_reminders_user_status_time because it existed before Release 4 in SQLite/Postgres metadata.
    for table_name, column_name in (
        ("reminder_notifications", "fire_time"),
        ("reminders", "parent_recurring_reminder_id"),
        ("reminders", "last_fire_time"),
        ("reminders", "next_fire_time"),
        ("reminders", "recurrence_timezone"),
        ("reminders", "recurrence_rule"),
    ):
        try:
            op.drop_column(table_name, column_name)
        except Exception:
            pass
