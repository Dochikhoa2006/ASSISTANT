"""Release 2 production safety

Revision ID: e2a91c7b5d30
Revises: d8f1c2e9a4b7
Create Date: 2026-07-09 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "e2a91c7b5d30"
down_revision: Union[str, Sequence[str], None] = "d8f1c2e9a4b7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        _upgrade_postgres()
    else:
        _upgrade_generic()


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        _downgrade_postgres()
    else:
        _downgrade_generic()


def _upgrade_postgres() -> None:
    op.execute("ALTER TABLE reminders ADD COLUMN IF NOT EXISTS user_timezone VARCHAR NOT NULL DEFAULT 'UTC'")
    op.execute("ALTER TABLE reminders ADD COLUMN IF NOT EXISTS original_time_text TEXT NULL")
    op.execute("ALTER TABLE reminder_notifications ADD COLUMN IF NOT EXISTS delivery_status VARCHAR NOT NULL DEFAULT 'pending'")
    op.execute("ALTER TABLE reminder_notifications ADD COLUMN IF NOT EXISTS delivery_attempts INTEGER NOT NULL DEFAULT 0")
    op.execute("ALTER TABLE reminder_notifications ADD COLUMN IF NOT EXISTS last_delivery_error TEXT NULL")
    op.execute("ALTER TABLE reminder_notifications ADD COLUMN IF NOT EXISTS sent_at VARCHAR NULL")
    op.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint
                WHERE conname = 'ck_reminder_notifications_delivery_status'
            ) THEN
                ALTER TABLE reminder_notifications
                ADD CONSTRAINT ck_reminder_notifications_delivery_status
                CHECK (delivery_status IN ('pending', 'sent', 'failed', 'retrying'));
            END IF;
        END $$;
        """
    )
    op.create_table(
        "mutation_requests",
        sa.Column("request_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("idempotency_key", sa.String(), nullable=False),
        sa.Column("payload_hash", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("stored_response_json", sa.Text(), nullable=True),
        sa.Column("created_at", sa.String(), nullable=False),
        sa.Column("updated_at", sa.String(), nullable=False),
        sa.CheckConstraint("status IN ('in_progress', 'completed', 'failed')", name="ck_mutation_requests_status"),
        sa.PrimaryKeyConstraint("request_id"),
        sa.UniqueConstraint("user_id", "idempotency_key", name="uq_mutation_requests_user_key"),
    )
    op.create_table(
        "pending_action_confirmations",
        sa.Column("confirmation_token", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("action_type", sa.String(), nullable=False),
        sa.Column("target_entity_type", sa.String(), nullable=False),
        sa.Column("target_entity_id", sa.String(), nullable=True),
        sa.Column("proposed_action_json", sa.Text(), nullable=False),
        sa.Column("target_snapshot_json", sa.Text(), nullable=False),
        sa.Column("expires_at", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("created_at", sa.String(), nullable=False),
        sa.Column("confirmed_at", sa.String(), nullable=True),
        sa.CheckConstraint("status IN ('pending', 'confirmed', 'cancelled', 'expired')", name="ck_pending_action_confirmations_status"),
        sa.PrimaryKeyConstraint("confirmation_token"),
    )
    op.execute("CREATE INDEX IF NOT EXISTS idx_mutation_requests_user_key ON mutation_requests(user_id, idempotency_key)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_confirmations_user_status ON pending_action_confirmations(user_id, status)")


def _upgrade_generic() -> None:
    op.add_column("reminders", sa.Column("user_timezone", sa.String(), nullable=False, server_default="UTC"))
    op.add_column("reminders", sa.Column("original_time_text", sa.Text(), nullable=True))
    op.add_column("reminder_notifications", sa.Column("delivery_status", sa.String(), nullable=False, server_default="pending"))
    op.add_column("reminder_notifications", sa.Column("delivery_attempts", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("reminder_notifications", sa.Column("last_delivery_error", sa.Text(), nullable=True))
    op.add_column("reminder_notifications", sa.Column("sent_at", sa.String(), nullable=True))
    op.create_table(
        "mutation_requests",
        sa.Column("request_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("idempotency_key", sa.String(), nullable=False),
        sa.Column("payload_hash", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("stored_response_json", sa.Text(), nullable=True),
        sa.Column("created_at", sa.String(), nullable=False),
        sa.Column("updated_at", sa.String(), nullable=False),
        sa.PrimaryKeyConstraint("request_id"),
        sa.UniqueConstraint("user_id", "idempotency_key", name="uq_mutation_requests_user_key"),
    )
    op.create_table(
        "pending_action_confirmations",
        sa.Column("confirmation_token", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("action_type", sa.String(), nullable=False),
        sa.Column("target_entity_type", sa.String(), nullable=False),
        sa.Column("target_entity_id", sa.String(), nullable=True),
        sa.Column("proposed_action_json", sa.Text(), nullable=False),
        sa.Column("target_snapshot_json", sa.Text(), nullable=False),
        sa.Column("expires_at", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("created_at", sa.String(), nullable=False),
        sa.Column("confirmed_at", sa.String(), nullable=True),
        sa.PrimaryKeyConstraint("confirmation_token"),
    )
    op.create_index("idx_mutation_requests_user_key", "mutation_requests", ["user_id", "idempotency_key"])
    op.create_index("idx_confirmations_user_status", "pending_action_confirmations", ["user_id", "status"])


def _downgrade_postgres() -> None:
    op.execute("DROP INDEX IF EXISTS idx_confirmations_user_status")
    op.execute("DROP INDEX IF EXISTS idx_mutation_requests_user_key")
    op.drop_table("pending_action_confirmations")
    op.drop_table("mutation_requests")
    op.execute("ALTER TABLE reminder_notifications DROP CONSTRAINT IF EXISTS ck_reminder_notifications_delivery_status")
    op.execute("ALTER TABLE reminder_notifications DROP COLUMN IF EXISTS sent_at")
    op.execute("ALTER TABLE reminder_notifications DROP COLUMN IF EXISTS last_delivery_error")
    op.execute("ALTER TABLE reminder_notifications DROP COLUMN IF EXISTS delivery_attempts")
    op.execute("ALTER TABLE reminder_notifications DROP COLUMN IF EXISTS delivery_status")
    op.execute("ALTER TABLE reminders DROP COLUMN IF EXISTS original_time_text")
    op.execute("ALTER TABLE reminders DROP COLUMN IF EXISTS user_timezone")


def _downgrade_generic() -> None:
    op.drop_index("idx_confirmations_user_status", table_name="pending_action_confirmations")
    op.drop_index("idx_mutation_requests_user_key", table_name="mutation_requests")
    op.drop_table("pending_action_confirmations")
    op.drop_table("mutation_requests")
    op.drop_column("reminder_notifications", "sent_at")
    op.drop_column("reminder_notifications", "last_delivery_error")
    op.drop_column("reminder_notifications", "delivery_attempts")
    op.drop_column("reminder_notifications", "delivery_status")
    op.drop_column("reminders", "original_time_text")
    op.drop_column("reminders", "user_timezone")
