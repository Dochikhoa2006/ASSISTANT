"""persist immutable reminder-notification source-hop bindings

Revision ID: f7c3a9e2d614
Revises: e4b7d9a1c612
Create Date: 2026-07-11 00:00:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "f7c3a9e2d614"
down_revision = "e4b7d9a1c612"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    columns = {column["name"] for column in sa.inspect(bind).get_columns("reminder_notifications")}
    if "source_topic_id" not in columns:
        op.add_column("reminder_notifications", sa.Column("source_topic_id", sa.String(), nullable=True))
    if "source_hop_id" not in columns:
        op.add_column("reminder_notifications", sa.Column("source_hop_id", sa.String(), nullable=True))

    # Snapshot the existing reminder binding before making either source
    # identity immutable. Every new notification is populated by the
    # repository at insert time.
    op.execute(
        """
        UPDATE reminder_notifications AS n
        SET source_topic_id = r.source_topic_id,
            source_hop_id = r.source_hop_id
        FROM reminders AS r
        WHERE n.reminder_id = r.reminder_id
          AND n.user_id = r.user_id
          AND n.source_hop_id IS NULL
        """
    )
    op.execute(
        """
        CREATE OR REPLACE FUNCTION prevent_reminder_source_rebind()
        RETURNS trigger AS $$
        BEGIN
            IF OLD.source_topic_id IS DISTINCT FROM NEW.source_topic_id
               OR OLD.source_hop_id IS DISTINCT FROM NEW.source_hop_id THEN
                RAISE EXCEPTION 'Reminder source conversation binding is immutable';
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        """
    )
    op.execute("DROP TRIGGER IF EXISTS trg_prevent_reminder_source_rebind ON reminders")
    op.execute(
        """
        CREATE TRIGGER trg_prevent_reminder_source_rebind
        BEFORE UPDATE OF source_topic_id, source_hop_id ON reminders
        FOR EACH ROW EXECUTE FUNCTION prevent_reminder_source_rebind()
        """
    )
    op.execute("DROP TRIGGER IF EXISTS trg_prevent_notification_source_rebind ON reminder_notifications")
    op.execute(
        """
        CREATE TRIGGER trg_prevent_notification_source_rebind
        BEFORE UPDATE OF source_topic_id, source_hop_id ON reminder_notifications
        FOR EACH ROW EXECUTE FUNCTION prevent_reminder_source_rebind()
        """
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS trg_prevent_notification_source_rebind ON reminder_notifications")
    op.execute("DROP TRIGGER IF EXISTS trg_prevent_reminder_source_rebind ON reminders")
    op.execute("DROP FUNCTION IF EXISTS prevent_reminder_source_rebind()")
    bind = op.get_bind()
    columns = {column["name"] for column in sa.inspect(bind).get_columns("reminder_notifications")}
    for column in ("source_hop_id", "source_topic_id"):
        if column in columns:
            op.drop_column("reminder_notifications", column)
