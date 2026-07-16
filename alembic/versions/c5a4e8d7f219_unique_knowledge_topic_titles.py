"""enforce one exact knowledge topic title per user

Revision ID: c5a4e8d7f219
Revises: f7c3a9e2d614
Create Date: 2026-07-16 00:00:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "c5a4e8d7f219"
down_revision = "f7c3a9e2d614"
branch_labels = None
depends_on = None


def upgrade() -> None:
    migration_context = op.get_context()
    if migration_context.as_sql:
        # Offline PostgreSQL generation cannot inspect live rows. Emit the same
        # fail-closed preflight into the generated migration SQL instead.
        op.execute(
            """
            DO $$
            BEGIN
                IF EXISTS (
                    SELECT 1
                    FROM knowledge_topics
                    GROUP BY user_id, title
                    HAVING COUNT(*) > 1
                ) THEN
                    RAISE EXCEPTION
                        'Cannot add uq_knowledge_topics_user_title: legacy duplicate knowledge topic groups exist.';
                END IF;
            END
            $$
            """
        )
        op.create_unique_constraint(
            "uq_knowledge_topics_user_title",
            "knowledge_topics",
            ["user_id", "title"],
        )
        return

    bind = op.get_bind()
    duplicate_group_count = int(
        bind.execute(
            sa.text(
                """
                SELECT COUNT(*)
                FROM (
                    SELECT user_id, title
                    FROM knowledge_topics
                    GROUP BY user_id, title
                    HAVING COUNT(*) > 1
                ) AS duplicate_topics
                """
            )
        ).scalar_one()
    )
    if duplicate_group_count:
        raise RuntimeError(
            "Cannot add uq_knowledge_topics_user_title: legacy duplicate "
            f"knowledge topic groups exist ({duplicate_group_count})."
        )
    with op.batch_alter_table("knowledge_topics") as batch_op:
        batch_op.create_unique_constraint(
            "uq_knowledge_topics_user_title",
            ["user_id", "title"],
        )


def downgrade() -> None:
    with op.batch_alter_table("knowledge_topics") as batch_op:
        batch_op.drop_constraint(
            "uq_knowledge_topics_user_title",
            type_="unique",
        )
