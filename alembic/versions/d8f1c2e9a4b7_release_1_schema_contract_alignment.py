"""Release 1 schema contract alignment

Revision ID: d8f1c2e9a4b7
Revises: 68a848528fe7
Create Date: 2026-07-09 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "d8f1c2e9a4b7"
down_revision: Union[str, Sequence[str], None] = "68a848528fe7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


REMINDER_SIGNAL = """
    lower(coalesce(entities_json, '') || ' ' || coalesce(response_type, '') || ' ' || coalesce(raw_response, '')) LIKE '%reminder%'
    OR lower(coalesce(entities_json, '') || ' ' || coalesce(response_type, '') || ' ' || coalesce(raw_response, '')) LIKE '%reminder_id%'
    OR lower(coalesce(entities_json, '') || ' ' || coalesce(response_type, '') || ' ' || coalesce(raw_response, '')) LIKE '%notification_id%'
    OR lower(coalesce(entities_json, '') || ' ' || coalesce(response_type, '') || ' ' || coalesce(raw_response, '')) LIKE '%reminder_action%'
    OR lower(coalesce(entities_json, '') || ' ' || coalesce(response_type, '') || ' ' || coalesce(raw_response, '')) LIKE '%reminder_reply%'
    OR lower(coalesce(entities_json, '') || ' ' || coalesce(response_type, '') || ' ' || coalesce(raw_response, '')) LIKE '%dismissreminder%'
    OR lower(coalesce(entities_json, '') || ' ' || coalesce(response_type, '') || ' ' || coalesce(raw_response, '')) LIKE '%completereminder%'
    OR lower(coalesce(entities_json, '') || ' ' || coalesce(response_type, '') || ' ' || coalesce(raw_response, '')) LIKE '%createreminder%'
    OR lower(coalesce(entities_json, '') || ' ' || coalesce(response_type, '') || ' ' || coalesce(raw_response, '')) LIKE '%turn_off%'
    OR lower(coalesce(entities_json, '') || ' ' || coalesce(response_type, '') || ' ' || coalesce(raw_response, '')) LIKE '%turn_on%'
"""

KNOWLEDGE_SIGNAL = """
    lower(coalesce(entities_json, '') || ' ' || coalesce(raw_response, '')) LIKE '%knowledge%'
    OR lower(coalesce(entities_json, '') || ' ' || coalesce(raw_response, '')) LIKE '%knowledge_chunk%'
    OR lower(coalesce(entities_json, '') || ' ' || coalesce(raw_response, '')) LIKE '%chunk_id%'
    OR lower(coalesce(entities_json, '') || ' ' || coalesce(raw_response, '')) LIKE '%knowledge_action%'
    OR lower(coalesce(entities_json, '') || ' ' || coalesce(raw_response, '')) LIKE '%knowledge_facts%'
    OR lower(coalesce(entities_json, '') || ' ' || coalesce(raw_response, '')) LIKE '%added new knowledge%'
    OR lower(coalesce(entities_json, '') || ' ' || coalesce(raw_response, '')) LIKE '%modified knowledge%'
    OR lower(coalesce(entities_json, '') || ' ' || coalesce(raw_response, '')) LIKE '%deleted knowledge%'
"""


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
    op.execute("ALTER TABLE conversation_hops ADD COLUMN IF NOT EXISTS root_hop_id VARCHAR NULL")
    op.execute("ALTER TABLE conversation_hops ADD COLUMN IF NOT EXISTS depth_from_root INTEGER NOT NULL DEFAULT 0")

    op.execute(
        """
        DO $$
        DECLARE r RECORD;
        BEGIN
            FOR r IN
                SELECT conname
                FROM pg_constraint
                WHERE conrelid = 'conversation_hops'::regclass
                  AND contype = 'c'
            LOOP
                EXECUTE format('ALTER TABLE conversation_hops DROP CONSTRAINT %I', r.conname);
            END LOOP;

            FOR r IN
                SELECT conname
                FROM pg_constraint
                WHERE conrelid = 'reminders'::regclass
                  AND contype = 'c'
            LOOP
                EXECUTE format('ALTER TABLE reminders DROP CONSTRAINT %I', r.conname);
            END LOOP;
        END $$;
        """
    )

    op.execute(
        f"""
        UPDATE conversation_hops
        SET response_type = CASE
            WHEN response_type = 'text' THEN 'normal'
            WHEN response_type = 'rich_card' THEN 'normal'
            WHEN response_type = 'action' AND ({REMINDER_SIGNAL}) THEN 'reminder_action'
            WHEN response_type = 'action' AND ({KNOWLEDGE_SIGNAL}) THEN 'knowledge_action'
            WHEN response_type = 'action' THEN 'normal'
            ELSE response_type
        END
        """
    )
    op.execute(
        f"""
        UPDATE conversation_hops
        SET intent = CASE
            WHEN intent = 'chat' THEN 'general_response'
            WHEN intent = 'search' THEN 'general_response'
            WHEN intent = 'command' AND ({REMINDER_SIGNAL}) THEN 'reminder'
            WHEN intent = 'command' AND response_type IN ('reminder_action', 'reminder_reply') THEN 'reminder'
            WHEN intent = 'command' AND ({KNOWLEDGE_SIGNAL}) THEN 'knowledge_facts'
            WHEN intent = 'command' AND response_type = 'knowledge_action' THEN 'knowledge_facts'
            WHEN intent = 'command' THEN 'general_response'
            ELSE intent
        END
        """
    )
    op.execute("UPDATE reminders SET status = 'notified' WHERE status = 'read'")

    op.execute(
        """
        WITH RECURSIVE hop_tree AS (
            SELECT h.hop_id, NULL::varchar AS root_hop_id, 0 AS depth_from_root
            FROM conversation_hops h
            WHERE COALESCE(h.parent_hop_id, h.previous_hop_id) IS NULL
            UNION ALL
            SELECT child.hop_id,
                   COALESCE(parent.root_hop_id, parent.hop_id) AS root_hop_id,
                   parent.depth_from_root + 1 AS depth_from_root
            FROM conversation_hops child
            JOIN hop_tree parent
              ON parent.hop_id = COALESCE(child.parent_hop_id, child.previous_hop_id)
            WHERE child.hop_id <> parent.hop_id
        )
        UPDATE conversation_hops h
        SET root_hop_id = hop_tree.root_hop_id,
            depth_from_root = hop_tree.depth_from_root,
            branch_id = COALESCE(h.branch_id, hop_tree.root_hop_id, h.hop_id)
        FROM hop_tree
        WHERE h.hop_id = hop_tree.hop_id
        """
    )
    op.execute("UPDATE conversation_hops SET branch_id = COALESCE(branch_id, root_hop_id, hop_id)")

    op.execute(
        """
        ALTER TABLE conversation_hops
        ADD CONSTRAINT ck_conversation_hops_intent
        CHECK (intent IN ('clarification', 'general_response', 'knowledge_facts', 'reminder'))
        """
    )
    op.execute(
        """
        ALTER TABLE conversation_hops
        ADD CONSTRAINT ck_conversation_hops_response_type
        CHECK (response_type IN ('clarification', 'normal', 'knowledge_action', 'reminder_action', 'reminder_reply', 'error', 'safe_noop'))
        """
    )
    op.execute(
        """
        ALTER TABLE reminders
        ADD CONSTRAINT ck_reminders_status
        CHECK (status IN ('scheduled', 'notified', 'cancelled', 'dismissed', 'completed'))
        """
    )

    op.execute("CREATE INDEX IF NOT EXISTS idx_hops_user_root ON conversation_hops(user_id, root_hop_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_hops_user_branch ON conversation_hops(user_id, branch_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_reminders_user_source_hop ON reminders(user_id, source_hop_id)")


def _upgrade_generic() -> None:
    op.add_column("conversation_hops", sa.Column("root_hop_id", sa.String(), nullable=True))
    op.add_column(
        "conversation_hops",
        sa.Column("depth_from_root", sa.Integer(), nullable=False, server_default="0"),
    )
    op.create_index("idx_hops_user_root", "conversation_hops", ["user_id", "root_hop_id"])
    op.create_index("idx_hops_user_branch", "conversation_hops", ["user_id", "branch_id"])
    op.create_index("idx_reminders_user_source_hop", "reminders", ["user_id", "source_hop_id"])


def _downgrade_postgres() -> None:
    op.execute("DROP INDEX IF EXISTS idx_reminders_user_source_hop")
    op.execute("DROP INDEX IF EXISTS idx_hops_user_branch")
    op.execute("DROP INDEX IF EXISTS idx_hops_user_root")
    op.execute("ALTER TABLE reminders DROP CONSTRAINT IF EXISTS ck_reminders_status")
    op.execute("ALTER TABLE conversation_hops DROP CONSTRAINT IF EXISTS ck_conversation_hops_response_type")
    op.execute("ALTER TABLE conversation_hops DROP CONSTRAINT IF EXISTS ck_conversation_hops_intent")
    op.execute(
        """
        UPDATE conversation_hops
        SET response_type = CASE
            WHEN response_type IN ('knowledge_action', 'reminder_action') THEN 'action'
            ELSE 'text'
        END,
            intent = CASE
            WHEN intent = 'reminder' THEN 'command'
            ELSE 'chat'
        END
        """
    )
    op.execute("UPDATE reminders SET status = 'scheduled' WHERE status = 'cancelled'")
    op.execute("ALTER TABLE conversation_hops DROP COLUMN IF EXISTS depth_from_root")
    op.execute("ALTER TABLE conversation_hops DROP COLUMN IF EXISTS root_hop_id")
    op.create_check_constraint(
        "ck_conversation_hops_intent_legacy",
        "conversation_hops",
        "intent IN ('command', 'search', 'chat')",
    )
    op.create_check_constraint(
        "ck_conversation_hops_response_type_legacy",
        "conversation_hops",
        "response_type IN ('text', 'rich_card', 'action')",
    )
    op.create_check_constraint(
        "ck_reminders_status_legacy",
        "reminders",
        "status IN ('scheduled', 'notified', 'dismissed', 'completed')",
    )


def _downgrade_generic() -> None:
    op.drop_index("idx_reminders_user_source_hop", table_name="reminders")
    op.drop_index("idx_hops_user_branch", table_name="conversation_hops")
    op.drop_index("idx_hops_user_root", table_name="conversation_hops")
    op.drop_column("conversation_hops", "depth_from_root")
    op.drop_column("conversation_hops", "root_hop_id")
