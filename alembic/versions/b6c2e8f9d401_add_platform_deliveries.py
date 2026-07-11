"""persist outbound platform delivery metadata

Revision ID: b6c2e8f9d401
Revises: a4c9e7d1b2f6
Create Date: 2026-07-11 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa

revision = "b6c2e8f9d401"
down_revision = "a4c9e7d1b2f6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "platform_deliveries",
        sa.Column("delivery_id", sa.String(), primary_key=True),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("conversation_hop_id", sa.String(), nullable=True),
        sa.Column("channel", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("recipient", sa.String(), nullable=False),
        sa.Column("message_json", sa.Text(), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.String(), nullable=False),
        sa.CheckConstraint("channel IN ('gmail', 'zalo', 'telegram')", name="ck_platform_deliveries_channel"),
    )
    op.create_index("idx_platform_deliveries_user_created", "platform_deliveries", ["user_id", "created_at"])


def downgrade() -> None:
    op.drop_index("idx_platform_deliveries_user_created", table_name="platform_deliveries")
    op.drop_table("platform_deliveries")
