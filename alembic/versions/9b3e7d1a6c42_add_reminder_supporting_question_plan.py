"""persist autoscan reminder supporting-question plan state

Revision ID: 9b3e7d1a6c42
Revises: c5a4e8d7f219
Create Date: 2026-07-17 00:00:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "9b3e7d1a6c42"
down_revision = "c5a4e8d7f219"
branch_labels = None
depends_on = None


def _columns() -> set[str]:
    return {
        str(column["name"])
        for column in sa.inspect(op.get_bind()).get_columns("reminders")
    }


def upgrade() -> None:
    columns = _columns()
    if "supporting_question_plan_status" not in columns:
        op.add_column(
            "reminders",
            sa.Column(
                "supporting_question_plan_status",
                sa.String(),
                sa.CheckConstraint(
                    "supporting_question_plan_status IN "
                    "('pending', 'planned', 'needs_review')"
                ),
                nullable=False,
                server_default="pending",
            ),
        )
    if "supporting_question_planned_at" not in columns:
        op.add_column(
            "reminders",
            sa.Column("supporting_question_planned_at", sa.String(), nullable=True),
        )
    if "supporting_question_plan_reason" not in columns:
        op.add_column(
            "reminders",
            sa.Column("supporting_question_plan_reason", sa.Text(), nullable=True),
        )
    if "supporting_question_confidence" not in columns:
        op.add_column(
            "reminders",
            sa.Column(
                "supporting_question_confidence",
                sa.Float(),
                sa.CheckConstraint(
                    "supporting_question_confidence IS NULL OR "
                    "(supporting_question_confidence >= 0.0 AND "
                    "supporting_question_confidence <= 1.0)"
                ),
                nullable=True,
            ),
        )

    op.execute(
        """
        UPDATE reminders
        SET supporting_question_plan_status = 'planned',
            supporting_question_planned_at = COALESCE(
                supporting_question_planned_at, updated_at, created_at
            ),
            supporting_question_plan_reason = COALESCE(
                supporting_question_plan_reason,
                'Existing reminder supporting context was preserved.'
            ),
            supporting_question_confidence = COALESCE(
                supporting_question_confidence, 1.0
            )
        WHERE supporting_question_plan_status = 'pending'
          AND (
              TRIM(COALESCE(supporting_question, '')) <> ''
              OR TRIM(COALESCE(supporting_response, '')) <> ''
          )
        """
    )
    indexes = {
        str(index["name"])
        for index in sa.inspect(op.get_bind()).get_indexes("reminders")
    }
    if "idx_reminders_supporting_question_plan" not in indexes:
        op.create_index(
            "idx_reminders_supporting_question_plan",
            "reminders",
            ["status", "supporting_question_plan_status", "reminder_time"],
        )


def downgrade() -> None:
    indexes = {
        str(index["name"])
        for index in sa.inspect(op.get_bind()).get_indexes("reminders")
    }
    if "idx_reminders_supporting_question_plan" in indexes:
        op.drop_index(
            "idx_reminders_supporting_question_plan",
            table_name="reminders",
        )
    columns = _columns()
    for column_name in (
        "supporting_question_confidence",
        "supporting_question_plan_reason",
        "supporting_question_planned_at",
        "supporting_question_plan_status",
    ):
        if column_name in columns:
            op.drop_column("reminders", column_name)
