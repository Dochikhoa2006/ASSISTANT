"""SQL source-of-truth repository.

The repository keeps SQL writes atomic and records derived-index work in
``indexing_outbox`` inside the same transaction as the source mutation.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import sqlite3
from typing import Any, Iterator
from uuid import uuid4

from .contracts import (
    KnowledgeAction,
    OperationResult,
    OperationStatus,
    OutboxEntityType,
    OutboxOperation,
    ReminderAction,
    ValidatedKnowledgeAction,
    ValidatedReminderAction,
    ActionValidationResult,
    ReminderCandidateSummary,
    RepositoryTransactionResult,
    RepositoryActionResult,
    HopWrite,
    RetrievalResult,
    OutboxIndexPayload,
    IdempotencyClaimResult,
    MutationRequestStatus,
    ConfirmationStatus,
    KnowledgeSourceStatus,
    ArtifactStatus,
    PendingReminderTiming,
    PendingReminderSupportingQuestion,
)
from .errors import (
    RepositoryConflictError,
    KnowledgeConflictError,
    ReminderConflictError,
    RepositoryValidationError,
    RepositoryTransactionError,
)
from .prompts import DEFAULT_PROMPT_REGISTRY
from .repository import AssistantRepository
from .recurrence import calculate_next_fire_time
from .lifecycle import is_artifact_downloadable, is_indexable_conversation_hop, is_indexable_knowledge_chunk
from .metrics import GLOBAL_METRICS
from .conversation_embedding import (
    conversation_hop_embedding_metadata,
    conversation_hop_semantic_payload,
    serialize_conversation_hop,
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id() -> str:
    return uuid4().hex


def content_hash(user_id: str, text: str) -> str:
    digest = sha256()
    digest.update(user_id.encode("utf-8"))
    digest.update(b"\0")
    digest.update(text.encode("utf-8"))
    return digest.hexdigest()


def _optional_datetime(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


class SQLiteRepository(AssistantRepository):
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")

    @classmethod
    def in_memory(cls) -> "SQLiteRepository":
        connection = sqlite3.connect(":memory:", check_same_thread=False)
        connection.row_factory = sqlite3.Row
        return cls(connection)

    @classmethod
    def persistent(
        cls,
        database_path: str,
        *,
        enable_wal: bool,
        busy_timeout_ms: int,
    ) -> "SQLiteRepository":
        # Streamlit preserves ``session_state`` across reruns, but can execute
        # a later rerun on a different script-runner thread.  The repository
        # stored in that state must therefore be usable by the new thread.
        # Each Streamlit session owns its runtime, while independent processes
        # use their own SQLite connections.
        connection = sqlite3.connect(database_path, check_same_thread=False)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(f"PRAGMA busy_timeout = {busy_timeout_ms}")
        if enable_wal:
            connection.execute("PRAGMA journal_mode = WAL")
        return cls(connection)

    def initialize_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS conversation_topics (
                topic_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                title TEXT NOT NULL,
                topic_summary TEXT NOT NULL,
                state_summary TEXT NOT NULL,
                entities_json TEXT NOT NULL,
                last_hop_id TEXT NULL,
                status TEXT NOT NULL CHECK (status IN ('active', 'archived')),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                version INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS conversation_hops (
                hop_id TEXT PRIMARY KEY,
                topic_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                parent_hop_id TEXT NULL,
                previous_hop_id TEXT NULL,
                root_hop_id TEXT NULL,
                branch_id TEXT NOT NULL,
                depth_from_root INTEGER NOT NULL,
                intent TEXT NOT NULL,
                raw_user_query TEXT NOT NULL,
                rewritten_user_query TEXT NOT NULL,
                raw_response TEXT NOT NULL,
                summarized_user_query TEXT NOT NULL,
                summarized_response TEXT NOT NULL,
                state_summary TEXT NOT NULL,
                entities_json TEXT NOT NULL,
                supporting_questions_json TEXT NOT NULL,
                response_type TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                version INTEGER NOT NULL,
                FOREIGN KEY (topic_id) REFERENCES conversation_topics(topic_id),
                FOREIGN KEY (parent_hop_id) REFERENCES conversation_hops(hop_id),
                FOREIGN KEY (previous_hop_id) REFERENCES conversation_hops(hop_id),
                FOREIGN KEY (root_hop_id) REFERENCES conversation_hops(hop_id)
            );

            CREATE TABLE IF NOT EXISTS knowledge_topics (
                knowledge_topic_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                entities_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                version INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS knowledge_sources (
                source_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                file_type TEXT NOT NULL,
                upload_time TEXT NOT NULL,
                processing_status TEXT NOT NULL CHECK (processing_status IN ('pending', 'processing', 'indexed', 'failed', 'deleted')),
                content_hash TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                version INTEGER NOT NULL,
                is_deleted INTEGER NOT NULL DEFAULT 0 CHECK (is_deleted IN (0, 1)),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS knowledge_chunks (
                chunk_id TEXT PRIMARY KEY,
                knowledge_topic_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                source_id TEXT NULL,
                chunk_index INTEGER NOT NULL,
                raw_text TEXT NOT NULL,
                normalized_text TEXT NOT NULL,
                summary TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                is_deleted INTEGER NOT NULL DEFAULT 0 CHECK (is_deleted IN (0, 1)),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                version INTEGER NOT NULL,
                replaces_chunk_id TEXT NULL,
                replaced_by_chunk_id TEXT NULL,
                change_reason TEXT NULL,
                modified_by_user_query TEXT NULL,
                UNIQUE (user_id, knowledge_topic_id, content_hash),
                FOREIGN KEY (knowledge_topic_id) REFERENCES knowledge_topics(knowledge_topic_id),
                FOREIGN KEY (source_id) REFERENCES knowledge_sources(source_id)
            );

            CREATE TABLE IF NOT EXISTS reminders (
                reminder_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                source_topic_id TEXT NULL,
                source_hop_id TEXT NULL,
                reminder_time TEXT NOT NULL,
                event_time TEXT NULL,
                timing_plan_status TEXT NOT NULL DEFAULT 'pending' CHECK (timing_plan_status IN ('pending', 'planned', 'needs_review')),
                timing_planned_at TEXT NULL,
                timing_plan_reason TEXT NULL,
                supporting_question_plan_status TEXT NOT NULL DEFAULT 'pending' CHECK (supporting_question_plan_status IN ('pending', 'planned', 'needs_review')),
                supporting_question_planned_at TEXT NULL,
                supporting_question_plan_reason TEXT NULL,
                supporting_question_confidence REAL NULL CHECK (supporting_question_confidence IS NULL OR (supporting_question_confidence >= 0.0 AND supporting_question_confidence <= 1.0)),
                status TEXT NOT NULL CHECK (status IN ('scheduled', 'notified', 'cancelled', 'dismissed', 'completed')),
                raw_reminder TEXT NOT NULL,
                reminder_summary TEXT NOT NULL,
                subject TEXT NOT NULL,
                supporting_question TEXT,
                supporting_response TEXT,
                user_timezone TEXT NOT NULL DEFAULT 'UTC',
                original_time_text TEXT NULL,
                recurrence_rule TEXT NULL,
                recurrence_timezone TEXT NULL,
                next_fire_time TEXT NULL,
                last_fire_time TEXT NULL,
                parent_recurring_reminder_id TEXT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                version INTEGER NOT NULL,
                FOREIGN KEY (source_topic_id) REFERENCES conversation_topics(topic_id),
                FOREIGN KEY (source_hop_id) REFERENCES conversation_hops(hop_id)
            );

            CREATE TABLE IF NOT EXISTS reminder_notifications (
                notification_id TEXT PRIMARY KEY,
                reminder_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                source_topic_id TEXT NULL,
                source_hop_id TEXT NULL,
                ui_status TEXT NOT NULL CHECK (ui_status IN ('unread', 'read', 'deleted')),
                delivery_status TEXT NOT NULL DEFAULT 'pending' CHECK (delivery_status IN ('pending', 'sent', 'failed', 'retrying')),
                delivery_attempts INTEGER NOT NULL DEFAULT 0,
                last_delivery_error TEXT NULL,
                created_at TEXT NOT NULL,
                read_at TEXT NULL,
                deleted_at TEXT NULL,
                sent_at TEXT NULL,
                fire_time TEXT NULL,
                UNIQUE (reminder_id, user_id, fire_time),
                FOREIGN KEY (reminder_id) REFERENCES reminders(reminder_id)
            );

            CREATE TABLE IF NOT EXISTS mutation_requests (
                request_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                idempotency_key TEXT NOT NULL,
                payload_hash TEXT NOT NULL,
                status TEXT NOT NULL CHECK (status IN ('in_progress', 'completed', 'failed')),
                stored_response_json TEXT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE (user_id, idempotency_key)
            );

            CREATE TABLE IF NOT EXISTS pending_action_confirmations (
                confirmation_token TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                action_type TEXT NOT NULL,
                target_entity_type TEXT NOT NULL,
                target_entity_id TEXT NULL,
                proposed_action_json TEXT NOT NULL,
                target_snapshot_json TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                status TEXT NOT NULL CHECK (status IN ('pending', 'confirmed', 'cancelled', 'expired')),
                created_at TEXT NOT NULL,
                confirmed_at TEXT NULL
            );

            CREATE TABLE IF NOT EXISTS generated_artifacts (
                artifact_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                conversation_hop_id TEXT NULL,
                file_type TEXT NOT NULL CHECK (file_type IN ('xlsx', 'pdf', 'pptx', 'txt', 'csv')),
                filename TEXT NOT NULL,
                storage_path TEXT NOT NULL,
                storage_url TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NULL,
                status TEXT NOT NULL CHECK (status IN ('created', 'deleted', 'failed'))
            );

            CREATE TABLE IF NOT EXISTS platform_deliveries (
                delivery_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                conversation_hop_id TEXT NULL,
                channel TEXT NOT NULL CHECK (channel IN ('gmail', 'zalo', 'telegram')),
                status TEXT NOT NULL,
                recipient TEXT NOT NULL,
                message_json TEXT NOT NULL,
                error_message TEXT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS indexing_outbox (
                job_id TEXT PRIMARY KEY,
                entity_type TEXT NOT NULL CHECK (entity_type IN ('conversation_hop', 'knowledge_chunk')),
                entity_id TEXT NOT NULL,
                operation TEXT NOT NULL CHECK (operation IN ('upsert', 'delete')),
                status TEXT NOT NULL CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
                retry_count INTEGER NOT NULL,
                error_message TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_topics_user_status ON conversation_topics(user_id, status);
            CREATE INDEX IF NOT EXISTS idx_topics_last_hop ON conversation_topics(user_id, last_hop_id);
            CREATE INDEX IF NOT EXISTS idx_hops_topic_prev ON conversation_hops(user_id, topic_id, previous_hop_id);
            CREATE INDEX IF NOT EXISTS idx_hops_topic_parent ON conversation_hops(user_id, topic_id, parent_hop_id);
            CREATE INDEX IF NOT EXISTS idx_chunks_user_topic_deleted ON knowledge_chunks(user_id, knowledge_topic_id, is_deleted);
            CREATE INDEX IF NOT EXISTS idx_sources_user_status ON knowledge_sources(user_id, processing_status);
            CREATE INDEX IF NOT EXISTS idx_sources_user_hash ON knowledge_sources(user_id, content_hash);
            CREATE INDEX IF NOT EXISTS idx_chunks_user_source ON knowledge_chunks(user_id, source_id);
            CREATE INDEX IF NOT EXISTS idx_artifacts_user_status ON generated_artifacts(user_id, status);
            CREATE INDEX IF NOT EXISTS idx_platform_deliveries_user_created ON platform_deliveries(user_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_reminders_user_status_time ON reminders(user_id, status, reminder_time);
            CREATE INDEX IF NOT EXISTS idx_reminders_user_updated ON reminders(user_id, updated_at);
            CREATE INDEX IF NOT EXISTS idx_reminders_user_created ON reminders(user_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_reminders_user_source_hop ON reminders(user_id, source_hop_id);
            CREATE INDEX IF NOT EXISTS idx_mutation_requests_user_key ON mutation_requests(user_id, idempotency_key);
            CREATE INDEX IF NOT EXISTS idx_confirmations_user_status ON pending_action_confirmations(user_id, status);
            """
        )
        self._ensure_release2_sqlite_columns()
        self._ensure_release3_sqlite_columns()
        self._ensure_release4_sqlite_columns()
        self._ensure_release5_reminder_source_bindings()
        self._ensure_release6_reminder_supporting_question_plan()
        self._ensure_unique_knowledge_topic_titles()
        self.connection.commit()

    def _ensure_unique_knowledge_topic_titles(self) -> None:
        duplicate = self.connection.execute(
            """
            SELECT user_id, title, COUNT(*) AS duplicate_count
            FROM knowledge_topics
            GROUP BY user_id, title
            HAVING COUNT(*) > 1
            LIMIT 1
            """
        ).fetchone()
        if duplicate is not None:
            raise RepositoryValidationError(
                "Cannot enforce unique knowledge topic titles because legacy "
                "duplicate (user_id, title) rows exist. Repair them before startup."
            )
        self.connection.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS uq_knowledge_topics_user_title
            ON knowledge_topics(user_id, title)
            """
        )

    def _ensure_release2_sqlite_columns(self) -> None:
        def columns(table_name: str) -> set[str]:
            rows = self.connection.execute(f"PRAGMA table_info({table_name})").fetchall()
            return {str(row["name"]) for row in rows}

        reminder_columns = columns("reminders")
        if "user_timezone" not in reminder_columns:
            self.connection.execute(
                "ALTER TABLE reminders ADD COLUMN user_timezone TEXT NOT NULL DEFAULT 'UTC'"
            )
        if "original_time_text" not in reminder_columns:
            self.connection.execute(
                "ALTER TABLE reminders ADD COLUMN original_time_text TEXT NULL"
            )

        notification_columns = columns("reminder_notifications")
        if "delivery_status" not in notification_columns:
            self.connection.execute(
                "ALTER TABLE reminder_notifications ADD COLUMN delivery_status TEXT NOT NULL DEFAULT 'pending' CHECK (delivery_status IN ('pending', 'sent', 'failed', 'retrying'))"
            )
        if "delivery_attempts" not in notification_columns:
            self.connection.execute(
                "ALTER TABLE reminder_notifications ADD COLUMN delivery_attempts INTEGER NOT NULL DEFAULT 0"
            )
        if "last_delivery_error" not in notification_columns:
            self.connection.execute(
                "ALTER TABLE reminder_notifications ADD COLUMN last_delivery_error TEXT NULL"
            )
        if "sent_at" not in notification_columns:
            self.connection.execute(
                "ALTER TABLE reminder_notifications ADD COLUMN sent_at TEXT NULL"
            )

    def _ensure_release3_sqlite_columns(self) -> None:
        def columns(table_name: str) -> set[str]:
            rows = self.connection.execute(f"PRAGMA table_info({table_name})").fetchall()
            return {str(row["name"]) for row in rows}

        chunk_columns = columns("knowledge_chunks")
        for column_name in (
            "replaces_chunk_id",
            "replaced_by_chunk_id",
            "change_reason",
            "modified_by_user_query",
        ):
            if column_name not in chunk_columns:
                self.connection.execute(
                    f"ALTER TABLE knowledge_chunks ADD COLUMN {column_name} TEXT NULL"
                )

    def _ensure_release4_sqlite_columns(self) -> None:
        def columns(table_name: str) -> set[str]:
            rows = self.connection.execute(f"PRAGMA table_info({table_name})").fetchall()
            return {str(row["name"]) for row in rows}

        reminder_columns = columns("reminders")
        if "event_time" not in reminder_columns:
            self.connection.execute("ALTER TABLE reminders ADD COLUMN event_time TEXT NULL")
            reminder_columns.add("event_time")
        if "timing_plan_status" not in reminder_columns:
            self.connection.execute("ALTER TABLE reminders ADD COLUMN timing_plan_status TEXT NOT NULL DEFAULT 'pending'")
            reminder_columns.add("timing_plan_status")
        if "timing_planned_at" not in reminder_columns:
            self.connection.execute("ALTER TABLE reminders ADD COLUMN timing_planned_at TEXT NULL")
        if "timing_plan_reason" not in reminder_columns:
            self.connection.execute("ALTER TABLE reminders ADD COLUMN timing_plan_reason TEXT NULL")
        for column_name in (
            "recurrence_rule",
            "recurrence_timezone",
            "next_fire_time",
            "last_fire_time",
            "parent_recurring_reminder_id",
        ):
            if column_name not in reminder_columns:
                self.connection.execute(f"ALTER TABLE reminders ADD COLUMN {column_name} TEXT NULL")

        notification_columns = columns("reminder_notifications")
        if "fire_time" not in notification_columns:
            self.connection.execute("ALTER TABLE reminder_notifications ADD COLUMN fire_time TEXT NULL")
            notification_columns.add("fire_time")
        if self._has_legacy_notification_unique_constraint():
            self._rebuild_reminder_notifications_for_fire_time()

        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_reminders_user_status_next_fire ON reminders(user_id, status, next_fire_time)"
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_reminders_user_parent_recurring ON reminders(user_id, parent_recurring_reminder_id)"
        )

    def _ensure_release5_reminder_source_bindings(self) -> None:
        """Persist an immutable source-hop snapshot for every notification."""
        def columns(table_name: str) -> set[str]:
            rows = self.connection.execute(f"PRAGMA table_info({table_name})").fetchall()
            return {str(row["name"]) for row in rows}

        notification_columns = columns("reminder_notifications")
        if "source_topic_id" not in notification_columns:
            self.connection.execute(
                "ALTER TABLE reminder_notifications ADD COLUMN source_topic_id TEXT NULL"
            )
        if "source_hop_id" not in notification_columns:
            self.connection.execute(
                "ALTER TABLE reminder_notifications ADD COLUMN source_hop_id TEXT NULL"
            )

        # Existing notifications acquire the source identity they had when
        # this invariant was introduced. New notifications copy it at insert
        # time, so subsequent reminder changes can never redirect a reply.
        self.connection.execute(
            """
            UPDATE reminder_notifications
            SET source_topic_id = (
                    SELECT r.source_topic_id FROM reminders r
                    WHERE r.reminder_id = reminder_notifications.reminder_id
                      AND r.user_id = reminder_notifications.user_id
                ),
                source_hop_id = (
                    SELECT r.source_hop_id FROM reminders r
                    WHERE r.reminder_id = reminder_notifications.reminder_id
                      AND r.user_id = reminder_notifications.user_id
                )
            WHERE source_topic_id IS NULL AND source_hop_id IS NULL
            """
        )
        self.connection.executescript(
            """
            CREATE TRIGGER IF NOT EXISTS prevent_reminder_source_rebind
            BEFORE UPDATE OF source_topic_id, source_hop_id ON reminders
            FOR EACH ROW
            WHEN OLD.source_topic_id IS NOT NEW.source_topic_id
              OR OLD.source_hop_id IS NOT NEW.source_hop_id
            BEGIN
                SELECT RAISE(ABORT, 'Reminder source conversation binding is immutable');
            END;

            CREATE TRIGGER IF NOT EXISTS prevent_notification_source_rebind
            BEFORE UPDATE OF source_topic_id, source_hop_id ON reminder_notifications
            FOR EACH ROW
            WHEN OLD.source_topic_id IS NOT NEW.source_topic_id
              OR OLD.source_hop_id IS NOT NEW.source_hop_id
            BEGIN
                SELECT RAISE(ABORT, 'Reminder notification source binding is immutable');
            END;
            """
        )

    def _ensure_release6_reminder_supporting_question_plan(self) -> None:
        """Add an independent, durable autoscan state for reminder questions."""
        rows = self.connection.execute("PRAGMA table_info(reminders)").fetchall()
        reminder_columns = {str(row["name"]) for row in rows}
        if "supporting_question_plan_status" not in reminder_columns:
            self.connection.execute(
                "ALTER TABLE reminders ADD COLUMN supporting_question_plan_status "
                "TEXT NOT NULL DEFAULT 'pending' CHECK "
                "(supporting_question_plan_status IN ('pending', 'planned', 'needs_review'))"
            )
        if "supporting_question_planned_at" not in reminder_columns:
            self.connection.execute(
                "ALTER TABLE reminders ADD COLUMN supporting_question_planned_at TEXT NULL"
            )
        if "supporting_question_plan_reason" not in reminder_columns:
            self.connection.execute(
                "ALTER TABLE reminders ADD COLUMN supporting_question_plan_reason TEXT NULL"
            )
        if "supporting_question_confidence" not in reminder_columns:
            self.connection.execute(
                "ALTER TABLE reminders ADD COLUMN supporting_question_confidence "
                "REAL NULL CHECK (supporting_question_confidence IS NULL OR "
                "(supporting_question_confidence >= 0.0 AND supporting_question_confidence <= 1.0))"
            )
        self.connection.execute(
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
        self.connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_reminders_supporting_question_plan
            ON reminders(status, supporting_question_plan_status, reminder_time)
            """
        )

    def _has_legacy_notification_unique_constraint(self) -> bool:
        for index_row in self.connection.execute("PRAGMA index_list(reminder_notifications)").fetchall():
            if not int(index_row["unique"]):
                continue
            index_name = str(index_row["name"])
            cols = [
                str(col["name"])
                for col in self.connection.execute(f"PRAGMA index_info({index_name})").fetchall()
            ]
            if cols == ["reminder_id", "user_id"]:
                return True
        return False

    def _rebuild_reminder_notifications_for_fire_time(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS reminder_notifications_new (
                notification_id TEXT PRIMARY KEY,
                reminder_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                ui_status TEXT NOT NULL CHECK (ui_status IN ('unread', 'read', 'deleted')),
                delivery_status TEXT NOT NULL DEFAULT 'pending' CHECK (delivery_status IN ('pending', 'sent', 'failed', 'retrying')),
                delivery_attempts INTEGER NOT NULL DEFAULT 0,
                last_delivery_error TEXT NULL,
                created_at TEXT NOT NULL,
                read_at TEXT NULL,
                deleted_at TEXT NULL,
                sent_at TEXT NULL,
                fire_time TEXT NULL,
                UNIQUE (reminder_id, user_id, fire_time),
                FOREIGN KEY (reminder_id) REFERENCES reminders(reminder_id)
            );
            INSERT OR IGNORE INTO reminder_notifications_new (
                notification_id, reminder_id, user_id, ui_status, delivery_status,
                delivery_attempts, last_delivery_error, created_at, read_at,
                deleted_at, sent_at, fire_time
            )
            SELECT notification_id, reminder_id, user_id, ui_status,
                   COALESCE(delivery_status, 'pending'),
                   COALESCE(delivery_attempts, 0),
                   last_delivery_error, created_at, read_at, deleted_at, sent_at, fire_time
            FROM reminder_notifications;
            DROP TABLE reminder_notifications;
            ALTER TABLE reminder_notifications_new RENAME TO reminder_notifications;
            """
        )

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Cursor]:
        cursor = self.connection.cursor()
        try:
            cursor.execute("BEGIN IMMEDIATE")
            yield cursor
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise
        finally:
            cursor.close()

    def ensure_topic(
        self,
        cursor: sqlite3.Cursor,
        *,
        user_id: str,
        title: str,
        topic_summary: str = "",
        state_summary: str = "",
        entities: dict[str, Any] | None = None,
    ) -> str:
        topic = cursor.execute(
            """
            SELECT topic_id FROM conversation_topics
            WHERE user_id = ? AND title = ? AND status = 'active'
            ORDER BY updated_at DESC LIMIT 1
            """,
            (user_id, title),
        ).fetchone()
        if topic:
            return str(topic["topic_id"])

        topic_id = new_id()
        timestamp = now_iso()
        cursor.execute(
            """
            INSERT INTO conversation_topics (
                topic_id, user_id, title, topic_summary, state_summary,
                entities_json, last_hop_id, status, created_at, updated_at, version
            ) VALUES (?, ?, ?, ?, ?, ?, NULL, 'active', ?, ?, 1)
            """,
            (
                topic_id,
                user_id,
                title,
                topic_summary,
                state_summary,
                json.dumps(entities or {}),
                timestamp,
                timestamp,
            ),
        )
        return topic_id

    def append_conversation_hop(
        self,
        cursor: sqlite3.Cursor,
        *,
        topic_id: str,
        user_id: str,
        intent: str,
        raw_user_query: str,
        rewritten_user_query: str,
        raw_response: str,
        response_type: str,
        supporting_questions: list[str | dict[str, Any]] | None = None,
        parent_hop_id: str | None = None,
        branch_id: str | None = None,
        entities: dict[str, Any] | None = None,
        state_summary: str = "",
        hop_id: str | None = None,
    ) -> HopWrite:
        topic = cursor.execute(
            """
            SELECT last_hop_id FROM conversation_topics
            WHERE topic_id = ? AND user_id = ? AND status = 'active'
            """,
            (topic_id, user_id),
        ).fetchone()
        if not topic:
            raise ValueError("Active conversation topic not found for user")

        previous_hop_id = topic["last_hop_id"]
        root_hop_id = self._resolve_root_hop(cursor, previous_hop_id, parent_hop_id)
        depth_from_root = self._resolve_depth(cursor, previous_hop_id, parent_hop_id)
        hop_id = hop_id or new_id()
        timestamp = now_iso()
        effective_branch_id = branch_id or root_hop_id or hop_id
        questions_json = json.dumps(supporting_questions or [])
        entities_json = json.dumps(entities or {})
        cursor.execute(
            """
            INSERT INTO conversation_hops (
                hop_id, topic_id, user_id, parent_hop_id, previous_hop_id,
                root_hop_id, branch_id, depth_from_root, intent, raw_user_query,
                rewritten_user_query, raw_response, summarized_user_query,
                summarized_response, state_summary, entities_json,
                supporting_questions_json, response_type, created_at, updated_at, version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
            """,
            (
                hop_id,
                topic_id,
                user_id,
                parent_hop_id,
                previous_hop_id,
                root_hop_id,
                effective_branch_id,
                depth_from_root,
                intent,
                raw_user_query,
                rewritten_user_query,
                raw_response,
                rewritten_user_query,
                raw_response,
                state_summary,
                entities_json,
                questions_json,
                response_type,
                timestamp,
                timestamp,
            ),
        )
        cursor.execute(
            """
            UPDATE conversation_topics
            SET last_hop_id = ?, updated_at = ?, version = version + 1
            WHERE topic_id = ? AND user_id = ?
            """,
            (hop_id, timestamp, topic_id, user_id),
        )
        outbox_job_id = self.insert_outbox_job(
            cursor,
            entity_type=OutboxEntityType.CONVERSATION_HOP,
            entity_id=hop_id,
            operation=OutboxOperation.UPSERT,
        )
        return HopWrite(topic_id=topic_id, hop_id=hop_id, previous_hop_id=previous_hop_id, outbox_job_id=outbox_job_id)

    def merge_conversation_hop_entities(
        self,
        cursor: sqlite3.Cursor,
        *,
        user_id: str,
        hop_id: str,
        entities: dict[str, Any],
    ) -> None:
        row = cursor.execute(
            "SELECT entities_json FROM conversation_hops WHERE hop_id = ? AND user_id = ?",
            (hop_id, user_id),
        ).fetchone()
        if not row:
            raise ValueError("Conversation hop not found for user")
        try:
            current = json.loads(str(row["entities_json"] or "{}"))
        except (TypeError, ValueError, json.JSONDecodeError):
            current = {}
        if not isinstance(current, dict):
            current = {}
        current.update(entities)
        cursor.execute(
            """
            UPDATE conversation_hops
            SET entities_json = ?, updated_at = ?, version = version + 1
            WHERE hop_id = ? AND user_id = ?
            """,
            (json.dumps(current), now_iso(), hop_id, user_id),
        )
        if cursor.rowcount != 1:
            raise ValueError("Conversation hop entities could not be updated")

    def scan_due_reminders(self, *, now_value: str, limit: int = 100) -> list[str]:
        notified: list[str] = []
        with self.transaction() as cursor:
            rows = cursor.execute(
                """
                SELECT reminder_id, user_id, reminder_time, recurrence_rule,
                       recurrence_timezone, next_fire_time, version
                FROM reminders
                -- Deliberately no user_id predicate: one worker must create
                -- durable notifications for every due user, including offline
                -- users who will retrieve them after reconnecting.
                WHERE status = 'scheduled'
                  AND timing_plan_status = 'planned'
                  AND COALESCE(next_fire_time, reminder_time) <= ?
                ORDER BY COALESCE(next_fire_time, reminder_time)
                LIMIT ?
                """,
                (now_value, limit),
            ).fetchall()
            for row in rows:
                fire_time = row["next_fire_time"] or row["reminder_time"]
                self.create_notification_if_absent(
                    cursor,
                    user_id=row["user_id"],
                    reminder_id=row["reminder_id"],
                    fire_time=fire_time,
                )
                if row["recurrence_rule"]:
                    completed_occurrences = int(
                        cursor.execute(
                            """
                            SELECT COUNT(*)
                            FROM reminder_notifications
                            WHERE reminder_id = ? AND user_id = ?
                            """,
                            (row["reminder_id"], row["user_id"]),
                        ).fetchone()[0]
                    )
                    next_fire_time = calculate_next_fire_time(
                        previous_fire_time=fire_time,
                        recurrence_rule=row["recurrence_rule"],
                        recurrence_timezone=row["recurrence_timezone"] or "UTC",
                        completed_occurrences=completed_occurrences,
                    )
                    if next_fire_time:
                        cursor.execute(
                            """
                            UPDATE reminders
                            SET last_fire_time = ?,
                                next_fire_time = ?,
                                status = 'scheduled',
                                updated_at = ?,
                                version = version + 1
                            WHERE reminder_id = ? AND user_id = ?
                              AND status = 'scheduled' AND version = ?
                            """,
                            (
                                fire_time,
                                next_fire_time,
                                now_iso(),
                                row["reminder_id"],
                                row["user_id"],
                                row["version"],
                            ),
                        )
                    else:
                        cursor.execute(
                            """
                            UPDATE reminders
                            SET status = 'completed',
                                last_fire_time = ?,
                                next_fire_time = NULL,
                                updated_at = ?,
                                version = version + 1
                            WHERE reminder_id = ? AND user_id = ?
                              AND status = 'scheduled' AND version = ?
                            """,
                            (
                                fire_time,
                                now_iso(),
                                row["reminder_id"],
                                row["user_id"],
                                row["version"],
                            ),
                        )
                else:
                    cursor.execute(
                        """
                        UPDATE reminders
                        SET status = 'notified',
                            last_fire_time = ?,
                            next_fire_time = NULL,
                            updated_at = ?,
                            version = version + 1
                        WHERE reminder_id = ? AND user_id = ?
                          AND status = 'scheduled' AND version = ?
                        """,
                        (
                            fire_time,
                            now_iso(),
                            row["reminder_id"],
                            row["user_id"],
                            row["version"],
                        ),
                    )
                if cursor.rowcount != 1:
                    raise ReminderConflictError(
                        "Autoscan reminder transition lost its scheduled version."
                    )
                notified.append(str(row["reminder_id"]))
        GLOBAL_METRICS.increment("reminder_scans_total")
        GLOBAL_METRICS.increment("notification_created_total", len(notified))
        return notified

    def list_reminders_requiring_timing(self, *, limit: int = 100) -> list[PendingReminderTiming]:
        """Return unplanned reminders globally; autoscan owns the LLM phase."""
        rows = self.connection.execute(
            """
            SELECT reminder_id, user_id, event_time, reminder_time, subject,
                   raw_reminder, user_timezone, recurrence_rule, version
            FROM reminders
            WHERE status = 'scheduled' AND timing_plan_status = 'pending'
            ORDER BY COALESCE(event_time, reminder_time), reminder_id
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        candidates: list[PendingReminderTiming] = []
        for row in rows:
            source_value = row["event_time"] or row["reminder_time"]
            try:
                source_time = datetime.fromisoformat(source_value)
                if source_time.tzinfo is None:
                    source_time = source_time.replace(tzinfo=timezone.utc)
            except (TypeError, ValueError):
                # A malformed legacy timestamp cannot safely be sent to the
                # planner.  Mark it terminally below through the normal API.
                source_time = datetime.min.replace(tzinfo=timezone.utc)
            candidates.append(PendingReminderTiming(
                reminder_id=row["reminder_id"], user_id=row["user_id"],
                source_time=source_time, subject=row["subject"],
                raw_reminder=row["raw_reminder"],
                user_timezone=row["user_timezone"] or "UTC",
                recurrence_rule=row["recurrence_rule"], version=int(row["version"]),
            ))
        return candidates

    def complete_reminder_timing_plan(
        self, *, reminder_id: str, user_id: str, expected_version: int,
        notification_time: datetime, reason: str,
    ) -> bool:
        notification_iso = notification_time.astimezone(timezone.utc).isoformat()
        with self.transaction() as cursor:
            cursor.execute(
                """
                UPDATE reminders
                SET reminder_time = ?,
                    next_fire_time = CASE WHEN recurrence_rule IS NOT NULL THEN ? ELSE NULL END,
                    timing_plan_status = 'planned', timing_planned_at = ?,
                    timing_plan_reason = ?, updated_at = ?, version = version + 1
                WHERE reminder_id = ? AND user_id = ? AND version = ?
                  AND status = 'scheduled' AND timing_plan_status = 'pending'
                """,
                (notification_iso, notification_iso, now_iso(), reason, now_iso(),
                 reminder_id, user_id, expected_version),
            )
            return cursor.rowcount == 1

    def fail_reminder_timing_plan(
        self, *, reminder_id: str, user_id: str, expected_version: int, reason: str,
    ) -> bool:
        with self.transaction() as cursor:
            cursor.execute(
                """
                UPDATE reminders
                SET timing_plan_status = 'needs_review', timing_planned_at = ?,
                    timing_plan_reason = ?, updated_at = ?, version = version + 1
                WHERE reminder_id = ? AND user_id = ? AND version = ?
                  AND status = 'scheduled' AND timing_plan_status = 'pending'
                """,
                (now_iso(), reason, now_iso(), reminder_id, user_id, expected_version),
            )
            return cursor.rowcount == 1

    def list_reminders_requiring_supporting_question(
        self, *, limit: int = 100
    ) -> list[PendingReminderSupportingQuestion]:
        """Return globally pending question plans without exposing chat history."""
        rows = self.connection.execute(
            """
            SELECT reminder_id, user_id, subject, reminder_summary, raw_reminder,
                   reminder_time, event_time, user_timezone, recurrence_rule, version
            FROM reminders
            WHERE status = 'scheduled'
              AND supporting_question_plan_status = 'pending'
            ORDER BY COALESCE(event_time, reminder_time), reminder_id
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [
            PendingReminderSupportingQuestion(
                reminder_id=str(row["reminder_id"]),
                user_id=str(row["user_id"]),
                subject=str(row["subject"] or ""),
                reminder_summary=str(row["reminder_summary"] or ""),
                raw_reminder=str(row["raw_reminder"] or ""),
                notification_time=_optional_datetime(row["reminder_time"]),
                event_time=_optional_datetime(row["event_time"]),
                user_timezone=str(row["user_timezone"] or "UTC"),
                recurrence_rule=(
                    str(row["recurrence_rule"])
                    if row["recurrence_rule"] is not None
                    else None
                ),
                version=int(row["version"]),
            )
            for row in rows
        ]

    def complete_reminder_supporting_question_plan(
        self,
        *,
        reminder_id: str,
        user_id: str,
        expected_version: int,
        question: str | None,
        confidence: float,
        reason: str,
    ) -> bool:
        timestamp = now_iso()
        with self.transaction() as cursor:
            cursor.execute(
                """
                UPDATE reminders
                SET supporting_question = ?,
                    supporting_question_plan_status = 'planned',
                    supporting_question_planned_at = ?,
                    supporting_question_plan_reason = ?,
                    supporting_question_confidence = ?,
                    updated_at = ?,
                    version = version + 1
                WHERE reminder_id = ? AND user_id = ? AND version = ?
                  AND status = 'scheduled'
                  AND supporting_question_plan_status = 'pending'
                """,
                (
                    question,
                    timestamp,
                    reason,
                    confidence,
                    timestamp,
                    reminder_id,
                    user_id,
                    expected_version,
                ),
            )
            return cursor.rowcount == 1

    def fail_reminder_supporting_question_plan(
        self,
        *,
        reminder_id: str,
        user_id: str,
        expected_version: int,
        reason: str,
    ) -> bool:
        timestamp = now_iso()
        with self.transaction() as cursor:
            cursor.execute(
                """
                UPDATE reminders
                SET supporting_question_plan_status = 'needs_review',
                    supporting_question_planned_at = ?,
                    supporting_question_plan_reason = ?,
                    supporting_question_confidence = NULL,
                    updated_at = ?,
                    version = version + 1
                WHERE reminder_id = ? AND user_id = ? AND version = ?
                  AND status = 'scheduled'
                  AND supporting_question_plan_status = 'pending'
                """,
                (
                    timestamp,
                    reason,
                    timestamp,
                    reminder_id,
                    user_id,
                    expected_version,
                ),
            )
            return cursor.rowcount == 1

    def load_reminder_reply_context(self, *, user_id: str, reminder_id: str, notification_id: str) -> dict[str, Any]:
        res = self.connection.execute(
            """
            SELECT
                r.reminder_id,
                n.notification_id,
                r.subject,
                r.reminder_summary,
                r.raw_reminder,
                r.status AS reminder_status,
                r.reminder_time,
                r.event_time,
                r.user_timezone,
                r.original_time_text,
                r.recurrence_rule,
                r.recurrence_timezone,
                r.next_fire_time,
                r.last_fire_time,
                r.parent_recurring_reminder_id,
                r.timing_plan_status,
                r.supporting_question_plan_status,
                r.supporting_question_confidence,
                r.supporting_question,
                r.supporting_response,
                n.source_topic_id,
                n.source_hop_id,
                n.ui_status AS notification_ui_status,
                n.delivery_status AS notification_delivery_status,
                n.fire_time AS notification_fire_time,
                n.created_at AS notification_created_at,
                h.topic_id,
                h.rewritten_user_query AS source_rewritten_user_query,
                h.raw_response AS source_raw_response,
                h.supporting_questions_json,
                h.response_type AS source_response_type
            FROM reminders r
            JOIN reminder_notifications n ON r.reminder_id = n.reminder_id
            LEFT JOIN conversation_hops h
              ON n.source_hop_id = h.hop_id AND h.user_id = r.user_id
            WHERE r.user_id = ? AND r.reminder_id = ? AND n.notification_id = ?
            """,
            (user_id, reminder_id, notification_id),
        ).fetchone()
        if not res:
            return {}
        return dict(res)

    def claim_outbox_jobs(self, *, max_attempts: int, batch_size: int, retry_cutoff: str) -> list[dict[str, Any]]:
        with self.transaction() as cursor:
            # Atomic update and return for multi-worker safety
            rows = cursor.execute(
                """
                UPDATE indexing_outbox
                SET status = 'processing', updated_at = ?
                WHERE job_id IN (
                    SELECT job_id FROM indexing_outbox
                    WHERE retry_count < ?
                      AND (status = 'pending' OR (status = 'failed' AND updated_at <= ?))
                    ORDER BY created_at
                    LIMIT ?
                )
                RETURNING *
                """,
                (now_iso(), max_attempts, retry_cutoff, batch_size),
            ).fetchall()
            return [dict(r) for r in rows]

    def claim_outbox_jobs_by_ids(
        self,
        *,
        job_ids: list[str],
        max_attempts: int,
    ) -> list[dict[str, Any]]:
        unique_ids = list(dict.fromkeys(job_id for job_id in job_ids if job_id))
        if not unique_ids:
            return []
        placeholders = ",".join("?" for _ in unique_ids)
        with self.transaction() as cursor:
            rows = cursor.execute(
                f"""
                UPDATE indexing_outbox
                SET status = 'processing', updated_at = ?
                WHERE job_id IN ({placeholders})
                  AND retry_count < ?
                  AND status IN ('pending', 'failed')
                RETURNING *
                """,
                (now_iso(), *unique_ids, max_attempts),
            ).fetchall()
            return [dict(row) for row in rows]

    def get_outbox_job_statuses(
        self,
        *,
        job_ids: list[str],
    ) -> dict[str, str]:
        unique_ids = list(dict.fromkeys(job_id for job_id in job_ids if job_id))
        if not unique_ids:
            return {}
        placeholders = ",".join("?" for _ in unique_ids)
        rows = self.connection.execute(
            f"SELECT job_id, status FROM indexing_outbox WHERE job_id IN ({placeholders})",
            tuple(unique_ids),
        ).fetchall()
        return {str(row["job_id"]): str(row["status"]) for row in rows}

    def release_stale_processing_jobs(self, *, max_attempts: int, timeout_cutoff: str) -> None:
        with self.transaction() as cursor:
            cursor.execute(
                """
                UPDATE indexing_outbox
                SET status = 'failed',
                    retry_count = retry_count + 1,
                    error_message = 'Indexing job timed out while processing',
                    updated_at = ?
                WHERE status = 'processing' AND updated_at < ? AND retry_count < ?
                """,
                (now_iso(), timeout_cutoff, max_attempts),
            )

    def mark_outbox_job_completed(self, *, job_id: str) -> None:
        with self.transaction() as cursor:
            cursor.execute(
                "UPDATE indexing_outbox SET status = 'completed', updated_at = ? WHERE job_id = ?",
                (now_iso(), job_id),
            )

    def mark_outbox_job_failed(self, *, job_id: str, error_message: str) -> None:
        with self.transaction() as cursor:
            cursor.execute(
                """
                UPDATE indexing_outbox
                SET status = 'failed',
                    retry_count = retry_count + 1,
                    error_message = ?,
                    updated_at = ?
                WHERE job_id = ?
                """,
                (error_message, now_iso(), job_id),
            )

    def load_outbox_entity(self, *, entity_type: str, entity_id: str) -> OutboxIndexPayload:
        cursor = self.connection.cursor()
        if entity_type == "conversation_hop":
            row = cursor.execute(
                """
                SELECT h.*, t.status AS topic_status,
                       t.topic_summary AS topic_summary,
                       t.state_summary AS topic_state_summary,
                       t.entities_json AS topic_entities_json,
                       t.last_hop_id AS topic_last_hop_id,
                       t.updated_at AS topic_updated_at,
                       t.version AS topic_version
                FROM conversation_hops h
                JOIN conversation_topics t ON t.topic_id = h.topic_id AND t.user_id = h.user_id
                WHERE h.hop_id = ?
                """,
                (entity_id,),
            ).fetchone()
            if not row:
                raise ValueError(f"Conversation hop not found: {entity_id}")
            hop = dict(row)
            if not is_indexable_conversation_hop(hop):
                raise ValueError(f"Conversation hop is not indexable: {entity_id}")
            return OutboxIndexPayload(
                user_id=hop["user_id"],
                entity_type=entity_type,
                entity_id=entity_id,
                text=serialize_conversation_hop(hop),
                metadata=conversation_hop_embedding_metadata(hop),
            )
        elif entity_type == "knowledge_chunk":
            row = cursor.execute(
                """
                SELECT user_id, knowledge_topic_id, chunk_id, source_id, chunk_index,
                       raw_text, content_hash, is_deleted, version, created_at, updated_at
                FROM knowledge_chunks
                WHERE chunk_id = ? AND is_deleted = 0
                """,
                (entity_id,),
            ).fetchone()
            if not row:
                raise ValueError(f"Active knowledge chunk not found: {entity_id}")
            if not is_indexable_knowledge_chunk(dict(row)):
                raise ValueError(f"Knowledge chunk is not indexable: {entity_id}")
            metadata = {
                "user_id": row["user_id"],
                "knowledge_topic_id": row["knowledge_topic_id"],
                "chunk_id": row["chunk_id"],
                "source_id": row["source_id"] or "",
                "chunk_index": int(row["chunk_index"]),
                "content_hash": row["content_hash"],
                "is_deleted": bool(row["is_deleted"]),
                "version": int(row["version"]),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
            return OutboxIndexPayload(
                user_id=row["user_id"],
                entity_type=entity_type,
                entity_id=entity_id,
                text=row["raw_text"],
                metadata=metadata,
            )
        else:
            raise ValueError(f"Unknown entity type: {entity_type}")

    def hydrate_knowledge_retrieval_results(
        self, *, user_id: str, results: list[RetrievalResult]
    ) -> list[RetrievalResult]:
        ids = [r.entity_id for r in results if r.entity_type == "knowledge_chunk"]
        chunk_map = {
            str(row["chunk_id"]): row
            for row in self.get_knowledge_chunks_by_ids(user_id, ids, include_deleted=False)
        }
        hydrated: list[RetrievalResult] = []
        for result in results:
            chunk = chunk_map.get(result.entity_id)
            if not chunk:
                continue
            payload = dict(result.payload)
            payload.update(
                {
                    "user_id": chunk["user_id"],
                    "knowledge_topic_id": chunk["knowledge_topic_id"],
                    "chunk_id": chunk["chunk_id"],
                    "source_id": chunk.get("source_id"),
                    "chunk_index": chunk["chunk_index"],
                    "content_hash": chunk["content_hash"],
                    "is_deleted": bool(chunk["is_deleted"]),
                    "version": chunk["version"],
                    "created_at": chunk["created_at"],
                    "updated_at": chunk["updated_at"],
                    "text": chunk["normalized_text"] or chunk["raw_text"],
                }
            )
            hydrated.append(
                RetrievalResult(
                    entity_type=result.entity_type,
                    entity_id=result.entity_id,
                    source_store_evidence=result.source_store_evidence,
                    rerank_score=result.rerank_score,
                    confidence=result.confidence,
                    validation_status="sql_validated",
                    payload=payload,
                )
            )
        return hydrated

    def hydrate_conversation_retrieval_results(
        self, *, user_id: str, results: list[RetrievalResult]
    ) -> list[RetrievalResult]:
        ids = [r.entity_id for r in results if r.entity_type == "conversation_hop"]
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        rows = self.connection.execute(
            f"""
            SELECT h.*, t.status AS topic_status,
                   t.topic_summary AS topic_summary,
                   t.state_summary AS topic_state_summary,
                   t.entities_json AS topic_entities_json,
                   t.last_hop_id AS topic_last_hop_id,
                   t.updated_at AS topic_updated_at,
                   t.version AS topic_version
            FROM conversation_hops h
            JOIN conversation_topics t ON t.topic_id = h.topic_id AND t.user_id = h.user_id
            WHERE h.user_id = ? AND h.hop_id IN ({placeholders})
            """,
            [user_id, *ids],
        ).fetchall()
        hop_map = {str(row["hop_id"]): dict(row) for row in rows}
        hydrated: list[RetrievalResult] = []
        for result in results:
            hop = hop_map.get(result.entity_id)
            if not hop:
                continue
            semantic_hop = conversation_hop_semantic_payload(hop)
            payload = conversation_hop_semantic_payload(result.payload)
            payload.update(semantic_hop)
            payload["text"] = serialize_conversation_hop(semantic_hop)
            hydrated.append(
                RetrievalResult(
                    entity_type=result.entity_type,
                    entity_id=result.entity_id,
                    source_store_evidence=result.source_store_evidence,
                    rerank_score=result.rerank_score,
                    confidence=result.confidence,
                    validation_status="sql_validated",
                    payload=payload,
                )
            )
        return hydrated

    def list_all_outbox_entities(self) -> list[tuple[str, str]]:
        cursor = self.connection.cursor()
        results = []
        for row in cursor.execute(
            """
            SELECT h.hop_id
            FROM conversation_hops h
            JOIN conversation_topics t ON t.topic_id = h.topic_id AND t.user_id = h.user_id
            WHERE t.status = 'active'
            ORDER BY h.created_at
            """
        ).fetchall():
            results.append(("conversation_hop", row["hop_id"]))
        for row in cursor.execute("SELECT chunk_id FROM knowledge_chunks WHERE is_deleted = 0 ORDER BY created_at").fetchall():
            results.append(("knowledge_chunk", row["chunk_id"]))
        return results

    def add_knowledge_chunk(
        self,
        cursor: sqlite3.Cursor,
        *,
        user_id: str,
        title: str,
        text: str,
        source_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        replaces_chunk_id: str | None = None,
        change_reason: str | None = None,
        modified_by_user_query: str | None = None,
        knowledge_topic_id: str | None = None,
    ) -> tuple[str, str, str | None]:
        if knowledge_topic_id is not None:
            topic = cursor.execute(
                """
                SELECT knowledge_topic_id FROM knowledge_topics
                WHERE user_id = ? AND knowledge_topic_id = ?
                """,
                (user_id, knowledge_topic_id),
            ).fetchone()
            if not topic:
                raise KnowledgeConflictError(
                    "Knowledge topic not found for user"
                )
            topic_id = str(topic["knowledge_topic_id"])
        else:
            topic_id = self.ensure_knowledge_topic(
                cursor,
                user_id=user_id,
                title=title,
            )
        normalized = " ".join(text.split())
        c_hash = content_hash(user_id, normalized)
        
        existing = cursor.execute(
            """
            SELECT chunk_id, is_deleted FROM knowledge_chunks
            WHERE user_id = ? AND knowledge_topic_id = ? AND content_hash = ?
            """,
            (user_id, topic_id, c_hash)
        ).fetchone()
        
        if existing:
            existing_chunk_id = existing["chunk_id"]
            if existing["is_deleted"] == 0:
                # Already active, no-op
                return topic_id, existing_chunk_id, None
            else:
                # Reactivate deleted chunk
                cursor.execute(
                    """
                    UPDATE knowledge_chunks
                    SET is_deleted = 0, updated_at = ?, version = version + 1
                    WHERE chunk_id = ?
                    """,
                    (now_iso(), existing_chunk_id)
                )
                outbox_job_id = self.insert_outbox_job(
                    cursor,
                    entity_type=OutboxEntityType.KNOWLEDGE_CHUNK,
                    entity_id=existing_chunk_id,
                    operation=OutboxOperation.UPSERT,
                )
                return topic_id, existing_chunk_id, outbox_job_id

        row = cursor.execute(
            """
            SELECT COALESCE(MAX(chunk_index), -1) + 1 AS next_index
            FROM knowledge_chunks
            WHERE user_id = ? AND knowledge_topic_id = ?
            """,
            (user_id, topic_id),
        ).fetchone()
        chunk_id = new_id()
        timestamp = now_iso()
        cursor.execute(
            """
            INSERT INTO knowledge_chunks (
                chunk_id, knowledge_topic_id, user_id, source_id, chunk_index,
                raw_text, normalized_text, summary, metadata_json, content_hash,
                is_deleted, created_at, updated_at, version, replaces_chunk_id,
                change_reason, modified_by_user_query
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, 1, ?, ?, ?)
            """,
            (
                chunk_id,
                topic_id,
                user_id,
                source_id,
                int(row["next_index"]),
                text,
                normalized,
                normalized,
                json.dumps(metadata or {}),
                c_hash,
                timestamp,
                timestamp,
                replaces_chunk_id,
                change_reason,
                modified_by_user_query,
            ),
        )
        outbox_job_id = self.insert_outbox_job(
            cursor,
            entity_type=OutboxEntityType.KNOWLEDGE_CHUNK,
            entity_id=chunk_id,
            operation=OutboxOperation.UPSERT,
        )
        return topic_id, chunk_id, outbox_job_id

    def ensure_knowledge_topic(
        self, cursor: sqlite3.Cursor, *, user_id: str, title: str
    ) -> str:
        topic = cursor.execute(
            """
            SELECT knowledge_topic_id FROM knowledge_topics
            WHERE user_id = ? AND title = ?
            ORDER BY updated_at DESC LIMIT 1
            """,
            (user_id, title),
        ).fetchone()
        if topic:
            return str(topic["knowledge_topic_id"])

        topic_id = new_id()
        timestamp = now_iso()
        cursor.execute(
            """
            INSERT INTO knowledge_topics (
                knowledge_topic_id, user_id, title, description, entities_json,
                created_at, updated_at, version
            ) VALUES (?, ?, ?, '', '{}', ?, ?, 1)
            ON CONFLICT(user_id, title) DO NOTHING
            """,
            (topic_id, user_id, title, timestamp, timestamp),
        )
        resolved = cursor.execute(
            """
            SELECT knowledge_topic_id FROM knowledge_topics
            WHERE user_id = ? AND title = ?
            """,
            (user_id, title),
        ).fetchone()
        if not resolved:
            raise KnowledgeConflictError("Knowledge topic could not be resolved")
        return str(resolved["knowledge_topic_id"])

    @staticmethod
    def _touch_knowledge_topic(
        cursor: sqlite3.Cursor,
        *,
        user_id: str,
        knowledge_topic_id: str,
    ) -> None:
        cursor.execute(
            """
            UPDATE knowledge_topics
            SET updated_at = ?, version = version + 1
            WHERE user_id = ? AND knowledge_topic_id = ?
            """,
            (now_iso(), user_id, knowledge_topic_id),
        )
        if cursor.rowcount != 1:
            raise KnowledgeConflictError(
                "Knowledge topic changed before mutation completion"
            )

    def get_knowledge_chunks_by_ids(
        self, user_id: str, chunk_ids: list[str], include_deleted: bool = False
    ) -> list[dict[str, Any]]:
        if not chunk_ids:
            return []
        placeholders = ",".join("?" for _ in chunk_ids)
        query = f"SELECT * FROM knowledge_chunks WHERE user_id = ? AND chunk_id IN ({placeholders})"
        params: list[Any] = [user_id]
        params.extend(chunk_ids)
        if not include_deleted:
            query += " AND is_deleted = 0"
        with self.transaction() as cursor:
            rows = cursor.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    def soft_delete_knowledge_chunk(
        self,
        cursor: sqlite3.Cursor,
        *,
        user_id: str,
        chunk_id: str,
        expected_version: int | None = None,
        change_reason: str | None = None,
        modified_by_user_query: str | None = None,
    ) -> str:
        timestamp = now_iso()
        query = """
            UPDATE knowledge_chunks
            SET is_deleted = 1,
                updated_at = ?,
                version = version + 1,
                change_reason = COALESCE(?, change_reason),
                modified_by_user_query = COALESCE(?, modified_by_user_query)
            WHERE user_id = ? AND chunk_id = ? AND is_deleted = 0
        """
        params: list[Any] = [timestamp, change_reason, modified_by_user_query, user_id, chunk_id]
        if expected_version is not None:
            query += " AND version = ?"
            params.append(expected_version)
            
        cursor.execute(query, tuple(params))
        if cursor.rowcount != 1:
            raise KnowledgeConflictError("Active knowledge chunk not found or version mismatch for user")
        outbox_job_id = self.insert_outbox_job(
            cursor,
            entity_type=OutboxEntityType.KNOWLEDGE_CHUNK,
            entity_id=chunk_id,
            operation=OutboxOperation.DELETE,
        )
        return outbox_job_id

    def create_knowledge_source(
        self,
        cursor: sqlite3.Cursor,
        *,
        user_id: str,
        filename: str,
        file_type: str,
        content_hash: str,
        metadata: dict[str, Any] | None = None,
        processing_status: str = KnowledgeSourceStatus.PENDING.value,
    ) -> str:
        source_id = new_id()
        timestamp = now_iso()
        cursor.execute(
            """
            INSERT INTO knowledge_sources (
                source_id, user_id, filename, file_type, upload_time,
                processing_status, content_hash, metadata_json, version,
                is_deleted, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, 0, ?, ?)
            """,
            (
                source_id,
                user_id,
                filename,
                file_type,
                timestamp,
                processing_status,
                content_hash,
                json.dumps(metadata or {}),
                timestamp,
                timestamp,
            ),
        )
        return source_id

    def update_knowledge_source_status(
        self,
        cursor: sqlite3.Cursor,
        *,
        user_id: str,
        source_id: str,
        processing_status: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if metadata is None:
            cursor.execute(
                """
                UPDATE knowledge_sources
                SET processing_status = ?, updated_at = ?, version = version + 1
                WHERE user_id = ? AND source_id = ?
                """,
                (processing_status, now_iso(), user_id, source_id),
            )
        else:
            cursor.execute(
                """
                UPDATE knowledge_sources
                SET processing_status = ?, metadata_json = ?, updated_at = ?, version = version + 1
                WHERE user_id = ? AND source_id = ?
                """,
                (processing_status, json.dumps(metadata), now_iso(), user_id, source_id),
            )
        if cursor.rowcount != 1:
            raise KnowledgeConflictError("Knowledge source not found for user")

    def list_knowledge_sources(self, *, user_id: str, include_deleted: bool = False) -> list[dict[str, Any]]:
        query = "SELECT * FROM knowledge_sources WHERE user_id = ?"
        params: list[Any] = [user_id]
        if not include_deleted:
            query += " AND is_deleted = 0 AND processing_status != 'deleted'"
        query += " ORDER BY created_at DESC"
        rows = self.connection.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def get_knowledge_source(
        self, *, user_id: str, source_id: str, include_deleted: bool = False
    ) -> dict[str, Any]:
        query = "SELECT * FROM knowledge_sources WHERE user_id = ? AND source_id = ?"
        params: list[Any] = [user_id, source_id]
        if not include_deleted:
            query += " AND is_deleted = 0 AND processing_status != 'deleted'"
        row = self.connection.execute(query, params).fetchone()
        if not row:
            raise ValueError("Knowledge source not found for user")
        return dict(row)

    def soft_delete_knowledge_source(self, *, user_id: str, source_id: str) -> dict[str, Any]:
        timestamp = now_iso()
        outbox_job_ids: list[str] = []
        with self.transaction() as cursor:
            cursor.execute(
                """
                UPDATE knowledge_sources
                SET is_deleted = 1, processing_status = 'deleted', updated_at = ?, version = version + 1
                WHERE user_id = ? AND source_id = ? AND is_deleted = 0
                """,
                (timestamp, user_id, source_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("Knowledge source not found for user")
            rows = cursor.execute(
                """
                SELECT chunk_id FROM knowledge_chunks
                WHERE user_id = ? AND source_id = ? AND is_deleted = 0
                """,
                (user_id, source_id),
            ).fetchall()
            for row in rows:
                outbox_job_ids.append(
                    self.soft_delete_knowledge_chunk(
                        cursor,
                        user_id=user_id,
                        chunk_id=row["chunk_id"],
                        change_reason="source_deleted",
                    )
                )
        payload = self.get_knowledge_source(user_id=user_id, source_id=source_id, include_deleted=True)
        payload["outbox_job_ids"] = outbox_job_ids
        return payload

    def reindex_knowledge_source(self, *, user_id: str, source_id: str) -> list[str]:
        outbox_job_ids: list[str] = []
        with self.transaction() as cursor:
            source = cursor.execute(
                """
                SELECT source_id FROM knowledge_sources
                WHERE user_id = ? AND source_id = ? AND is_deleted = 0 AND processing_status != 'deleted'
                """,
                (user_id, source_id),
            ).fetchone()
            if not source:
                raise ValueError("Knowledge source not found for user")
            rows = cursor.execute(
                """
                SELECT chunk_id FROM knowledge_chunks
                WHERE user_id = ? AND source_id = ? AND is_deleted = 0
                """,
                (user_id, source_id),
            ).fetchall()
            for row in rows:
                outbox_job_ids.append(
                    self.insert_outbox_job(
                        cursor,
                        entity_type=OutboxEntityType.KNOWLEDGE_CHUNK,
                        entity_id=row["chunk_id"],
                        operation=OutboxOperation.UPSERT,
                    )
                )
        return outbox_job_ids

    def list_knowledge_facts(
        self, *, user_id: str, include_deleted: bool = False, source_id: str | None = None
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM knowledge_chunks WHERE user_id = ?"
        params: list[Any] = [user_id]
        if not include_deleted:
            query += " AND is_deleted = 0"
        if source_id is not None:
            query += " AND source_id = ?"
            params.append(source_id)
        query += " ORDER BY created_at DESC"
        return [dict(row) for row in self.connection.execute(query, params).fetchall()]

    def restore_knowledge_chunk(self, *, user_id: str, chunk_id: str) -> str:
        with self.transaction() as cursor:
            row = cursor.execute(
                """
                SELECT c.chunk_id, c.source_id, s.is_deleted AS source_deleted
                FROM knowledge_chunks c
                LEFT JOIN knowledge_sources s ON s.source_id = c.source_id AND s.user_id = c.user_id
                WHERE c.user_id = ? AND c.chunk_id = ? AND c.is_deleted = 1
                """,
                (user_id, chunk_id),
            ).fetchone()
            if not row:
                raise ValueError("Deleted knowledge chunk not found for user")
            if row["source_id"] and row["source_deleted"] == 1:
                raise ValueError("Cannot restore a chunk from a deleted source")
            cursor.execute(
                """
                UPDATE knowledge_chunks
                SET is_deleted = 0, updated_at = ?, version = version + 1
                WHERE user_id = ? AND chunk_id = ?
                """,
                (now_iso(), user_id, chunk_id),
            )
            return self.insert_outbox_job(
                cursor,
                entity_type=OutboxEntityType.KNOWLEDGE_CHUNK,
                entity_id=chunk_id,
                operation=OutboxOperation.UPSERT,
            )

    def update_knowledge_chunk_text(
        self,
        *,
        user_id: str,
        chunk_id: str,
        text: str,
        change_reason: str | None = None,
        modified_by_user_query: str | None = None,
    ) -> dict[str, Any]:
        existing = self.get_knowledge_chunks_by_ids(user_id, [chunk_id], include_deleted=False)
        if not existing:
            raise ValueError("Active knowledge chunk not found for user")
        old = existing[0]
        with self.transaction() as cursor:
            delete_job_id = self.soft_delete_knowledge_chunk(
                cursor,
                user_id=user_id,
                chunk_id=chunk_id,
                expected_version=old.get("version"),
                change_reason=change_reason or "manual_update",
                modified_by_user_query=modified_by_user_query,
            )
            topic_title = cursor.execute(
                "SELECT title FROM knowledge_topics WHERE user_id = ? AND knowledge_topic_id = ?",
                (user_id, old["knowledge_topic_id"]),
            ).fetchone()["title"]
            _, new_chunk_id, upsert_job_id = self.add_knowledge_chunk(
                cursor,
                user_id=user_id,
                title=topic_title,
                text=text,
                source_id=old.get("source_id"),
                metadata=json.loads(old.get("metadata_json") or "{}"),
                replaces_chunk_id=chunk_id,
                change_reason=change_reason,
                modified_by_user_query=modified_by_user_query,
            )
            cursor.execute(
                """
                UPDATE knowledge_chunks
                SET replaced_by_chunk_id = ?
                WHERE user_id = ? AND chunk_id = ?
                """,
                (new_chunk_id, user_id, chunk_id),
            )
        return {
            "old_chunk_id": chunk_id,
            "new_chunk_id": new_chunk_id,
            "outbox_job_ids": [job for job in (delete_job_id, upsert_job_id) if job],
        }

    def create_generated_artifact(
        self,
        *,
        user_id: str,
        conversation_hop_id: str | None,
        file_type: str,
        filename: str,
        storage_path: str,
        storage_url: str,
        metadata: dict[str, Any] | None = None,
        expires_at: str | None = None,
    ) -> dict[str, Any]:
        artifact_id = new_id()
        timestamp = now_iso()
        with self.transaction() as cursor:
            cursor.execute(
                """
                INSERT INTO generated_artifacts (
                    artifact_id, user_id, conversation_hop_id, file_type,
                    filename, storage_path, storage_url, metadata_json,
                    created_at, expires_at, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'created')
                """,
                (
                    artifact_id,
                    user_id,
                    conversation_hop_id,
                    file_type,
                    filename,
                    storage_path,
                    storage_url,
                    json.dumps(metadata or {}),
                    timestamp,
                    expires_at,
                ),
            )
        return self.get_generated_artifact(user_id=user_id, artifact_id=artifact_id)

    def list_generated_artifacts(
        self, *, user_id: str, include_deleted: bool = False
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM generated_artifacts WHERE user_id = ?"
        params: list[Any] = [user_id]
        if not include_deleted:
            query += " AND status = 'created'"
        query += " ORDER BY created_at DESC"
        return [dict(row) for row in self.connection.execute(query, params).fetchall()]

    def get_generated_artifact(
        self, *, user_id: str, artifact_id: str, include_deleted: bool = False
    ) -> dict[str, Any]:
        query = "SELECT * FROM generated_artifacts WHERE user_id = ? AND artifact_id = ?"
        params: list[Any] = [user_id, artifact_id]
        if not include_deleted:
            query += " AND status = 'created'"
        row = self.connection.execute(query, params).fetchone()
        if not row:
            raise ValueError("Artifact not found for user")
        if not include_deleted and not is_artifact_downloadable(dict(row)):
            raise ValueError("Artifact not found for user")
        return dict(row)

    def bind_generated_artifacts(
        self,
        *,
        user_id: str,
        artifact_ids: list[str],
        conversation_hop_id: str,
    ) -> list[str]:
        unique_ids = list(dict.fromkeys(str(value) for value in artifact_ids if value))
        if not unique_ids or not conversation_hop_id:
            return []
        placeholders = ", ".join("?" for _ in unique_ids)
        with self.transaction() as cursor:
            owner = cursor.execute(
                "SELECT 1 FROM conversation_hops WHERE user_id = ? AND hop_id = ?",
                (user_id, conversation_hop_id),
            ).fetchone()
            if not owner:
                raise ValueError("Conversation hop not found for user")
            cursor.execute(
                f"""
                UPDATE generated_artifacts
                SET conversation_hop_id = ?
                WHERE user_id = ?
                  AND artifact_id IN ({placeholders})
                  AND status = 'created'
                  AND (conversation_hop_id IS NULL OR conversation_hop_id = ?)
                """,
                (conversation_hop_id, user_id, *unique_ids, conversation_hop_id),
            )
            rows = cursor.execute(
                f"""
                SELECT artifact_id FROM generated_artifacts
                WHERE user_id = ?
                  AND artifact_id IN ({placeholders})
                  AND conversation_hop_id = ?
                  AND status = 'created'
                """,
                (user_id, *unique_ids, conversation_hop_id),
            ).fetchall()
        bound = {str(row["artifact_id"]) for row in rows}
        return [artifact_id for artifact_id in unique_ids if artifact_id in bound]

    def delete_generated_artifact(self, *, user_id: str, artifact_id: str) -> dict[str, Any]:
        with self.transaction() as cursor:
            cursor.execute(
                """
                UPDATE generated_artifacts
                SET status = 'deleted'
                WHERE user_id = ? AND artifact_id = ? AND status = 'created'
                """,
                (user_id, artifact_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("Artifact not found for user")
        return self.get_generated_artifact(user_id=user_id, artifact_id=artifact_id, include_deleted=True)

    def record_platform_delivery(self, *, user_id: str, conversation_hop_id: str | None, channel: str, status: str, recipient: str, message: dict[str, Any], error_message: str | None = None) -> dict[str, Any]:
        delivery_id = new_id()
        with self.transaction() as cursor:
            cursor.execute(
                """INSERT INTO platform_deliveries (delivery_id, user_id, conversation_hop_id, channel, status, recipient, message_json, error_message, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (delivery_id, user_id, conversation_hop_id, channel, status, recipient, json.dumps(message), error_message, now_iso()),
            )
        return {"delivery_id": delivery_id, "channel": channel, "status": status}

    def add_reminder(
        self,
        cursor: sqlite3.Cursor,
        *,
        user_id: str,
        source_topic_id: str | None,
        source_hop_id: str | None,
        reminder_time: str,
        raw_reminder: str,
        reminder_summary: str,
        subject: str,
        event_time: str | None = None,
        supporting_question: str | None = None,
        supporting_response: str | None = None,
        user_timezone: str = "UTC",
        original_time_text: str | None = None,
        recurrence_rule: str | None = None,
        recurrence_timezone: str | None = None,
        next_fire_time: str | None = None,
        parent_recurring_reminder_id: str | None = None,
    ) -> str:
        if not source_hop_id:
            raise RepositoryValidationError("A reminder must be bound to its source conversation hop")
        source = cursor.execute(
            """
            SELECT topic_id FROM conversation_hops
            WHERE hop_id = ? AND user_id = ?
            """,
            (source_hop_id, user_id),
        ).fetchone()
        if not source:
            raise RepositoryValidationError("Reminder source hop does not belong to this user")
        canonical_topic_id = str(source["topic_id"])
        if source_topic_id and source_topic_id != canonical_topic_id:
            raise RepositoryValidationError("Reminder source topic does not match its source hop")
        source_topic_id = canonical_topic_id
        reminder_id = new_id()
        timestamp = now_iso()
        has_existing_support = bool(
            str(supporting_question or "").strip()
            or str(supporting_response or "").strip()
        )
        support_plan_status = "planned" if has_existing_support else "pending"
        support_planned_at = timestamp if has_existing_support else None
        support_plan_reason = (
            "Existing reminder supporting context was preserved."
            if has_existing_support
            else None
        )
        support_confidence = 1.0 if has_existing_support else None
        cursor.execute(
            """
            INSERT INTO reminders (
                reminder_id, user_id, source_topic_id, source_hop_id,
                reminder_time, event_time, timing_plan_status, timing_planned_at, timing_plan_reason,
                supporting_question_plan_status, supporting_question_planned_at,
                supporting_question_plan_reason, supporting_question_confidence,
                status, raw_reminder, reminder_summary, subject,
                supporting_question, supporting_response, user_timezone,
                original_time_text, recurrence_rule, recurrence_timezone,
                next_fire_time, last_fire_time, parent_recurring_reminder_id,
                created_at, updated_at, version
            ) VALUES (?, ?, ?, ?, ?, ?, 'pending', NULL, NULL, ?, ?, ?, ?,
                      'scheduled', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, 1)
            """,
            (
                reminder_id,
                user_id,
                source_topic_id,
                source_hop_id,
                reminder_time,
                event_time,
                support_plan_status,
                support_planned_at,
                support_plan_reason,
                support_confidence,
                raw_reminder,
                reminder_summary,
                subject,
                supporting_question,
                supporting_response,
                user_timezone,
                original_time_text,
                recurrence_rule,
                recurrence_timezone,
                next_fire_time,
                parent_recurring_reminder_id,
                timestamp,
                timestamp,
            ),
        )
        return reminder_id

    def update_reminder_status(
        self,
        cursor: sqlite3.Cursor,
        *,
        user_id: str,
        reminder_id: str,
        status: str,
        expected_version: int | None = None,
        expected_status: str | None = None,
    ) -> None:
        timestamp = now_iso()
        query = """
            UPDATE reminders
            SET status = ?, updated_at = ?, version = version + 1
            WHERE user_id = ? AND reminder_id = ?
        """
        params: list[Any] = [status, timestamp, user_id, reminder_id]
        
        if expected_version is not None:
            query += " AND version = ?"
            params.append(expected_version)
        if expected_status is not None:
            query += " AND status = ?"
            params.append(expected_status)
            
        cursor.execute(query, tuple(params))
        if cursor.rowcount != 1:
            raise ReminderConflictError("Reminder update failed ownership, version, or status validation")

    def create_notification_if_absent(
        self, cursor: sqlite3.Cursor, *, user_id: str, reminder_id: str, fire_time: str | None = None
    ) -> str:
        current = cursor.execute(
            """
            SELECT notification_id FROM reminder_notifications
            WHERE user_id = ? AND reminder_id = ?
              AND COALESCE(fire_time, '') = COALESCE(?, '')
            """,
            (user_id, reminder_id, fire_time),
        ).fetchone()
        if current:
            return str(current["notification_id"])
        source = cursor.execute(
            """
            SELECT source_topic_id, source_hop_id FROM reminders
            WHERE user_id = ? AND reminder_id = ?
            """,
            (user_id, reminder_id),
        ).fetchone()
        if not source or not source["source_hop_id"]:
            raise RepositoryValidationError("Reminder has no immutable source conversation hop")
        notification_id = new_id()
        timestamp = now_iso()
        cursor.execute(
            """
            INSERT INTO reminder_notifications (
                notification_id, reminder_id, user_id, source_topic_id, source_hop_id, ui_status,
                delivery_status, delivery_attempts, created_at, fire_time
            ) VALUES (?, ?, ?, ?, ?, 'unread', 'pending', 0, ?, ?)
            """,
            (
                notification_id, reminder_id, user_id,
                source["source_topic_id"], source["source_hop_id"],
                timestamp, fire_time,
            ),
        )
        return notification_id

    def insert_outbox_job(
        self,
        cursor: sqlite3.Cursor,
        *,
        entity_type: OutboxEntityType,
        entity_id: str,
        operation: OutboxOperation,
    ) -> str:
        if entity_type.value == "reminder" or entity_type.value == "reminder_notification":
            raise RepositoryValidationError(f"{entity_type.value} rows must not be indexed.")
            
        timestamp = now_iso()
        
        # Check if there is already a pending or processing job for this exact entity and operation
        existing = cursor.execute(
            """
            SELECT job_id FROM indexing_outbox 
            WHERE entity_type = ? AND entity_id = ? AND operation = ? 
              AND status IN ('pending', 'processing')
            LIMIT 1
            """,
            (entity_type.value, entity_id, operation.value)
        ).fetchone()
        
        if existing:
            # Duplicate job prevention: we don't need a new job if one is already queued or working
            return existing["job_id"]
            
        job_id = new_id()
        cursor.execute(
            """
            INSERT INTO indexing_outbox (
                job_id, entity_type, entity_id, operation, status,
                retry_count, error_message, created_at, updated_at
            ) VALUES (?, ?, ?, ?, 'pending', 0, NULL, ?, ?)
            """,
            (job_id, entity_type.value, entity_id, operation.value, timestamp, timestamp),
        )
        return job_id

    def record_action_audit_noop(
        self,
        *,
        user_id: str,
        topic_title: str,
        raw_user_query: str,
        rewritten_user_query: str,
        response_text: str,
        intent: str,
        response_type: str,
        parent_hop_id: str | None = None,
    ) -> RepositoryTransactionResult:
        try:
            with self.transaction() as cursor:
                topic_id = self.ensure_topic(cursor, user_id=user_id, title=topic_title)
                hop = self.append_conversation_hop(
                    cursor,
                    topic_id=topic_id,
                    user_id=user_id,
                    intent=intent,
                    raw_user_query=raw_user_query,
                    rewritten_user_query=rewritten_user_query,
                    raw_response=response_text,
                    response_type=response_type,
                    parent_hop_id=parent_hop_id,
                )
                result = RepositoryActionResult(
                    action_id=new_id(),
                    action_type="noop",
                    status="skipped",
                    domain_entity_type="none",
                    audit_hop_id=hop.hop_id,
                    indexing_outbox_ids=(hop.outbox_job_id,),
                    user_safe_summary="No action was necessary.",
                    reason_summary="The branch detected a safe no-op condition.",
                )
                return RepositoryTransactionResult(
                    committed=True,
                    results=(result,),
                    audit_topic_id=hop.topic_id,
                    audit_hop_id=hop.hop_id,
                    indexing_outbox_ids=(hop.outbox_job_id,),
                )
        except Exception as exc:
            return RepositoryTransactionResult(
                committed=False,
                error_type="RepositoryTransactionError",
                reason_summary="The database transaction failed and was rolled back.",
                results=(),
            )

    def transactional_knowledge_actions(
        self,
        *,
        user_id: str,
        topic_title: str,
        raw_user_query: str,
        rewritten_user_query: str,
        response_text: str,
        actions: list[ValidatedKnowledgeAction],
        parent_hop_id: str | None = None,
    ) -> RepositoryTransactionResult:
        for action in actions:
            if action.validation_result != ActionValidationResult.EXECUTE:
                raise ValueError("Repository methods must receive only executable actions.")
        try:
            with self.transaction() as cursor:
                topic_id = self.ensure_topic(cursor, user_id=user_id, title=topic_title)
                results: list[RepositoryActionResult] = []
                outbox_jobs: list[str] = []
                knowledge_entities: list[dict[str, Any]] = []
                for action in actions:
                    result = self._apply_knowledge_action(cursor, user_id=user_id, action=action)
                    results.append(result)
                    outbox_jobs.extend(result.indexing_outbox_ids)
                    if not result.domain_entity_id:
                        raise KnowledgeConflictError(
                            "Knowledge mutation did not resolve a chunk"
                        )
                    entity_row = cursor.execute(
                        """
                        SELECT knowledge_topic_id FROM knowledge_chunks
                        WHERE user_id = ? AND chunk_id = ?
                        """,
                        (user_id, result.domain_entity_id),
                    ).fetchone()
                    if not entity_row:
                        raise KnowledgeConflictError(
                            "Knowledge mutation chunk was not persisted"
                        )
                    knowledge_entities.append(
                        {
                            "entity_type": "knowledge_chunk",
                            "action": action.action.value,
                            "knowledge_topic_id": str(
                                entity_row["knowledge_topic_id"]
                            ),
                            "knowledge_chunk_id": result.domain_entity_id,
                            "target_chunk_ids": list(action.target_chunk_ids),
                            "status": result.status,
                        }
                    )
                hop = self.append_conversation_hop(
                    cursor,
                    topic_id=topic_id,
                    user_id=user_id,
                    intent="knowledge_facts",
                    raw_user_query=raw_user_query,
                    rewritten_user_query=rewritten_user_query,
                    raw_response=response_text,
                    response_type="knowledge_action",
                    parent_hop_id=parent_hop_id,
                    entities={"knowledge": knowledge_entities},
                )
                outbox_jobs.append(hop.outbox_job_id)
                updated_results = []
                for r in results:
                    updated_results.append(
                        RepositoryActionResult(
                            action_id=r.action_id,
                            action_type=r.action_type,
                            status=r.status,
                            domain_entity_type=r.domain_entity_type,
                            domain_entity_id=r.domain_entity_id,
                            audit_hop_id=hop.hop_id,
                            indexing_outbox_ids=r.indexing_outbox_ids,
                            user_safe_summary=r.user_safe_summary,
                            reason_summary=r.reason_summary,
                        )
                    )
                return RepositoryTransactionResult(
                    committed=True,
                    results=tuple(updated_results),
                    audit_topic_id=hop.topic_id,
                    audit_hop_id=hop.hop_id,
                    indexing_outbox_ids=tuple(outbox_jobs),
                )
        except RepositoryConflictError as exc:
            from .errors import safe_repository_reason
            return RepositoryTransactionResult(
                committed=False,
                error_type=type(exc).__name__,
                reason_summary=safe_repository_reason(exc),
                results=(),
            )
        except Exception as exc:
            return RepositoryTransactionResult(
                committed=False,
                error_type="RepositoryTransactionError",
                reason_summary="The database transaction failed and was rolled back.",
                results=(),
            )

    def transactional_reminder_actions(
        self,
        *,
        user_id: str,
        topic_title: str,
        raw_user_query: str,
        rewritten_user_query: str,
        response_text: str,
        actions: list[ValidatedReminderAction],
        parent_hop_id: str | None = None,
    ) -> RepositoryTransactionResult:
        for action in actions:
            if action.validation_result != ActionValidationResult.EXECUTE:
                raise ValueError("Repository methods must receive only executable actions.")
        try:
            with self.transaction() as cursor:
                topic_id = self.ensure_topic(cursor, user_id=user_id, title=topic_title)
                results: list[RepositoryActionResult] = []
                reminder_entities: list[dict[str, str]] = []
                audit_hop_id = new_id()
                hop = self.append_conversation_hop(
                    cursor,
                    topic_id=topic_id,
                    user_id=user_id,
                    intent="reminder",
                    raw_user_query=raw_user_query,
                    rewritten_user_query=rewritten_user_query,
                    raw_response=response_text,
                    response_type="reminder_action",
                    entities={"reminders": reminder_entities},
                    hop_id=audit_hop_id,
                    parent_hop_id=parent_hop_id,
                )
                for action in actions:
                    result = self._apply_reminder_action(
                        cursor, user_id=user_id, source_topic_id=topic_id, action=action, audit_hop_id=audit_hop_id
                    )
                    results.append(result)
                    if result.domain_entity_id:
                        reminder_entities.append(
                            {
                                "entity_type": "reminder",
                                "reminder_id": result.domain_entity_id,
                                "subject": str(action.subject or action.reminder_summary or ""),
                            }
                        )
                cursor.execute(
                    "UPDATE conversation_hops SET entities_json = ?, updated_at = ? WHERE hop_id = ? AND user_id = ?",
                    (json.dumps({"reminders": reminder_entities}), now_iso(), hop.hop_id, user_id),
                )
                updated_results = []
                for r in results:
                    updated_results.append(
                        RepositoryActionResult(
                            action_id=r.action_id,
                            action_type=r.action_type,
                            status=r.status,
                            domain_entity_type=r.domain_entity_type,
                            domain_entity_id=r.domain_entity_id,
                            audit_hop_id=hop.hop_id,
                            indexing_outbox_ids=r.indexing_outbox_ids,
                            user_safe_summary=r.user_safe_summary,
                            reason_summary=r.reason_summary,
                        )
                    )
                return RepositoryTransactionResult(
                    committed=True,
                    results=tuple(updated_results),
                    audit_topic_id=hop.topic_id,
                    audit_hop_id=hop.hop_id,
                    indexing_outbox_ids=(hop.outbox_job_id,),
                )
        except RepositoryConflictError as exc:
            from .errors import safe_repository_reason
            return RepositoryTransactionResult(
                committed=False,
                error_type=type(exc).__name__,
                reason_summary=safe_repository_reason(exc),
                results=(),
            )
        except Exception as exc:
            return RepositoryTransactionResult(
                committed=False,
                error_type="RepositoryTransactionError",
                reason_summary="The database transaction failed and was rolled back.",
                results=(),
            )

    def _apply_knowledge_action(
        self, cursor: sqlite3.Cursor, *, user_id: str, action: ValidatedKnowledgeAction
    ) -> RepositoryActionResult:
        action_type = action.action
        if action_type is KnowledgeAction.ADD:
            topic_id, chunk_id, outbox_job_id = self.add_knowledge_chunk(
                cursor,
                user_id=user_id,
                title=action.topic_title or "Knowledge",
                text=action.knowledge_text or action.new_text or "",
                source_id=None,
                metadata=None,
            )
            self._touch_knowledge_topic(
                cursor,
                user_id=user_id,
                knowledge_topic_id=topic_id,
            )
            return RepositoryActionResult(
                action_id=new_id(),
                action_type=action_type.value,
                status="committed",
                domain_entity_type="knowledge_chunk",
                domain_entity_id=chunk_id,
                indexing_outbox_ids=(outbox_job_id,) if outbox_job_id else (),
                user_safe_summary="Added new knowledge." if outbox_job_id else "Knowledge was already known.",
            )
        if action_type in {KnowledgeAction.DELETE, KnowledgeAction.MODIFY}:
            chunk_id = action.target_chunk_ids[0]
            target = cursor.execute(
                """
                SELECT knowledge_topic_id, normalized_text FROM knowledge_chunks
                WHERE user_id = ? AND chunk_id = ? AND is_deleted = 0
                """,
                (user_id, chunk_id),
            ).fetchone()
            if not target:
                raise KnowledgeConflictError(
                    "Active knowledge target was not found"
                )
            target_topic_id = str(target["knowledge_topic_id"])
            if action.target_topic_ids and set(action.target_topic_ids) != {
                target_topic_id
            }:
                raise KnowledgeConflictError(
                    "Knowledge target topic no longer matches"
                )
            if action_type is KnowledgeAction.MODIFY:
                replacement_text = " ".join(
                    (action.replacement_text or action.new_text or "").split()
                )
                if not replacement_text:
                    raise KnowledgeConflictError(
                        "Knowledge replacement text is empty"
                    )
                if replacement_text == str(target["normalized_text"]):
                    raise KnowledgeConflictError(
                        "Knowledge replacement already matches the active target"
                    )
                duplicate_replacement = cursor.execute(
                    """
                    SELECT chunk_id FROM knowledge_chunks
                    WHERE user_id = ? AND knowledge_topic_id = ?
                      AND content_hash = ? AND chunk_id != ?
                    LIMIT 1
                    """,
                    (
                        user_id,
                        target_topic_id,
                        content_hash(user_id, replacement_text),
                        chunk_id,
                    ),
                ).fetchone()
                if duplicate_replacement:
                    raise KnowledgeConflictError(
                        "Knowledge replacement already exists in the target topic"
                    )
            expected_version = action.observed_versions.get(chunk_id)
            del_outbox_job_id = self.soft_delete_knowledge_chunk(
                cursor,
                user_id=user_id,
                chunk_id=chunk_id,
                expected_version=expected_version,
                change_reason=action.reason_summary or action.action.value,
                modified_by_user_query=action.target_description,
            )
            if action_type is KnowledgeAction.MODIFY:
                _, new_chunk_id, add_outbox_job_id = self.add_knowledge_chunk(
                    cursor,
                    user_id=user_id,
                    title=action.topic_title or "Knowledge",
                    text=action.replacement_text or action.new_text or "",
                    source_id=None,
                    metadata=None,
                    replaces_chunk_id=chunk_id,
                    change_reason=action.reason_summary or "modify",
                    modified_by_user_query=action.target_description,
                    knowledge_topic_id=target_topic_id,
                )
                cursor.execute(
                    """
                    UPDATE knowledge_chunks
                    SET replaced_by_chunk_id = ?
                    WHERE user_id = ? AND chunk_id = ?
                    """,
                    (new_chunk_id, user_id, chunk_id),
                )
                self._touch_knowledge_topic(
                    cursor,
                    user_id=user_id,
                    knowledge_topic_id=target_topic_id,
                )
                
                outbox_jobs = [del_outbox_job_id]
                if add_outbox_job_id:
                    outbox_jobs.append(add_outbox_job_id)
                    
                return RepositoryActionResult(
                    action_id=new_id(),
                    action_type=action_type.value,
                    status="committed",
                    domain_entity_type="knowledge_chunk",
                    domain_entity_id=new_chunk_id,
                    indexing_outbox_ids=tuple(outbox_jobs),
                    user_safe_summary="Modified knowledge.",
                )
            self._touch_knowledge_topic(
                cursor,
                user_id=user_id,
                knowledge_topic_id=target_topic_id,
            )
            return RepositoryActionResult(
                action_id=new_id(),
                action_type=action_type.value,
                status="committed",
                domain_entity_type="knowledge_chunk",
                domain_entity_id=chunk_id,
                indexing_outbox_ids=(del_outbox_job_id,),
                user_safe_summary="Deleted knowledge.",
            )
        raise ValueError("Unsupported knowledge action")

    def list_reminder_candidates(
        self,
        user_id: str,
        statuses: tuple[str, ...],
        time_window: tuple[datetime, datetime] | None,
        limit: int,
    ) -> list[ReminderCandidateSummary]:
        query = "SELECT reminder_id, subject, reminder_summary, raw_reminder, reminder_time, status, created_at, updated_at, version FROM reminders WHERE user_id = ?"
        params: list[Any] = [user_id]
        
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            query += f" AND status IN ({placeholders})"
            params.extend(statuses)
            
        if time_window:
            query += " AND reminder_time >= ? AND reminder_time <= ?"
            params.extend([time_window[0].isoformat(), time_window[1].isoformat()])
            
        query += " ORDER BY reminder_time DESC LIMIT ?"
        params.append(limit)
        
        rows = self.connection.execute(query, params).fetchall()
        
        candidates = []
        for r in rows:
            r_time = datetime.fromisoformat(r["reminder_time"]) if r["reminder_time"] else None
            created = datetime.fromisoformat(r["created_at"]) if r["created_at"] else datetime.now(timezone.utc)
            updated = datetime.fromisoformat(r["updated_at"]) if r["updated_at"] else None
            candidates.append(
                ReminderCandidateSummary(
                    reminder_id=r["reminder_id"],
                    subject=r["subject"],
                    reminder_summary=r["reminder_summary"],
                    raw_reminder=r["raw_reminder"],
                    reminder_time=r_time,
                    status=r["status"],
                    created_at=created,
                    updated_at=updated,
                    version=r["version"],
                    is_deleted=False,
                )
            )
        return candidates

    def _apply_reminder_action(
        self,
        cursor: sqlite3.Cursor,
        *,
        user_id: str,
        source_topic_id: str,
        action: ValidatedReminderAction,
        audit_hop_id: str,
    ) -> RepositoryActionResult:
        action_type = action.action

        def preserve_explicit_notification_time(reminder_id: str) -> None:
            if action.timing_plan_required:
                return
            timestamp = now_iso()
            cursor.execute(
                """
                UPDATE reminders
                SET timing_plan_status = 'planned', timing_planned_at = ?,
                    timing_plan_reason = ?
                WHERE user_id = ? AND reminder_id = ?
                """,
                (
                    timestamp,
                    "Explicit notification time preserved from validated reminder mutation.",
                    user_id,
                    reminder_id,
                ),
            )
            if cursor.rowcount != 1:
                raise ReminderConflictError(
                    "Explicit reminder timing state could not be preserved"
                )

        if action_type is ReminderAction.ADD:
            reminder_id = self.add_reminder(
                cursor,
                user_id=user_id,
                source_topic_id=source_topic_id,
                source_hop_id=audit_hop_id,
                reminder_time=action.reminder_time.isoformat() if action.reminder_time else "",
                event_time=(
                    action.event_time
                    or (action.reminder_time if action.timing_plan_required else None)
                ).isoformat()
                if (
                    action.event_time
                    or (action.reminder_time if action.timing_plan_required else None)
                )
                else None,
                raw_reminder=action.raw_reminder or "",
                reminder_summary=action.reminder_summary or "",
                subject=action.subject or "",
                supporting_question=action.supporting_question,
                supporting_response=action.supporting_response,
                user_timezone=action.user_timezone or "UTC",
                original_time_text=action.original_time_text,
                recurrence_rule=action.recurrence_rule,
                recurrence_timezone=action.recurrence_timezone or action.user_timezone,
                next_fire_time=action.next_fire_time.isoformat() if action.next_fire_time else (action.reminder_time.isoformat() if action.recurrence_rule and action.reminder_time else None),
                parent_recurring_reminder_id=action.parent_recurring_reminder_id,
            )
            preserve_explicit_notification_time(reminder_id)
            return RepositoryActionResult(
                action_id=new_id(),
                action_type=action_type.value,
                status="committed",
                domain_entity_type="reminder",
                domain_entity_id=reminder_id,
                indexing_outbox_ids=(),
                user_safe_summary=(
                    f"Added reminder '{action.subject}'. "
                    + (
                        "Timing will be finalized by autoscan before it can notify you."
                        if action.timing_plan_required
                        else "The validated notification time was preserved."
                    )
                ),
            )
        elif action_type is ReminderAction.MODIFY:
            reminder_id = action.target_reminder_ids[0]
            replacement_time = action.replacement_time or action.reminder_time or action.observed_reminder_time
            if not replacement_time:
                raise RepositoryValidationError("Replacement reminder time is required")
            original_source = cursor.execute(
                """
                SELECT source_topic_id, source_hop_id FROM reminders
                WHERE user_id = ? AND reminder_id = ?
                """,
                (user_id, reminder_id),
            ).fetchone()
            if not original_source:
                raise ReminderConflictError("Reminder source binding was not found")
            self.update_reminder_status(
                cursor,
                user_id=user_id,
                reminder_id=reminder_id,
                status="cancelled",
                expected_version=action.observed_version,
                expected_status=action.observed_status,
            )
            replacement_summary = (
                action.replacement_summary
                if action.replacement_summary is not None
                else action.reminder_summary
            )
            replacement_recurrence_rule = (
                action.replacement_recurrence_rule
                if action.replacement_recurrence_rule is not None
                else action.recurrence_rule
            )
            replacement_recurrence_timezone = (
                action.replacement_recurrence_timezone
                if action.replacement_recurrence_timezone is not None
                else action.recurrence_timezone or action.user_timezone
            )
            new_reminder_id = self.add_reminder(
                cursor,
                user_id=user_id,
                # A modification produces a successor reminder, but it must
                # keep the original conversation identity used by reminder
                # replies and its supporting question.
                source_topic_id=original_source["source_topic_id"] or source_topic_id,
                source_hop_id=original_source["source_hop_id"] or audit_hop_id,
                reminder_time=replacement_time.isoformat(),
                event_time=(
                    action.event_time
                    or (replacement_time if action.timing_plan_required else None)
                ).isoformat()
                if (
                    action.event_time
                    or (replacement_time if action.timing_plan_required else None)
                )
                else None,
                raw_reminder=action.raw_reminder or "",
                reminder_summary=replacement_summary or "",
                subject=action.replacement_subject or action.subject or "",
                supporting_question=action.supporting_question,
                supporting_response=action.supporting_response,
                user_timezone=action.user_timezone or "UTC",
                original_time_text=action.original_time_text,
                recurrence_rule=replacement_recurrence_rule,
                recurrence_timezone=replacement_recurrence_timezone,
                next_fire_time=action.next_fire_time.isoformat() if action.next_fire_time else (replacement_time.isoformat() if replacement_recurrence_rule else None),
                parent_recurring_reminder_id=reminder_id,
            )
            preserve_explicit_notification_time(new_reminder_id)
            return RepositoryActionResult(
                action_id=new_id(),
                action_type=action_type.value,
                status="committed",
                domain_entity_type="reminder",
                domain_entity_id=new_reminder_id,
                indexing_outbox_ids=(),
                user_safe_summary=(
                    f"Modified reminder '{action.replacement_subject or action.subject}'. "
                    + (
                        "Timing will be recalculated by autoscan."
                        if action.timing_plan_required
                        else "The validated notification time was preserved."
                    )
                ),
            )
        else:
            reminder_id = action.target_reminder_ids[0]
            status_by_action = {
                ReminderAction.TURN_ON: "scheduled",
                ReminderAction.TURN_OFF: "cancelled",
                ReminderAction.DELETE: "dismissed",
            }
            if action_type is ReminderAction.TURN_ON:
                timestamp = now_iso()
                cursor.execute(
                    """
                    UPDATE reminders
                    SET status = 'scheduled', reminder_time = COALESCE(event_time, reminder_time),
                        next_fire_time = CASE WHEN recurrence_rule IS NOT NULL THEN COALESCE(event_time, reminder_time) ELSE NULL END,
                        timing_plan_status = 'pending', timing_planned_at = NULL,
                        timing_plan_reason = NULL, updated_at = ?, version = version + 1
                    WHERE user_id = ? AND reminder_id = ? AND version = ? AND status = ?
                    """,
                    (timestamp, user_id, reminder_id, action.observed_version, action.observed_status),
                )
                if cursor.rowcount != 1:
                    raise ReminderConflictError("Reminder update failed ownership, version, or status validation")
            else:
                self.update_reminder_status(
                    cursor,
                    user_id=user_id,
                    reminder_id=reminder_id,
                    status=status_by_action[action_type],
                    expected_version=action.observed_version,
                    expected_status=action.observed_status,
                )
            return RepositoryActionResult(
                action_id=new_id(),
                action_type=action_type.value,
                status="committed",
                domain_entity_type="reminder",
                domain_entity_id=reminder_id,
                indexing_outbox_ids=(),
                user_safe_summary=(
                    f"Reminder '{action.subject or reminder_id}' is scheduled again; its timing will be recalculated by autoscan."
                    if action_type is ReminderAction.TURN_ON
                    else f"Reminder '{action.subject or reminder_id}' is now {status_by_action[action_type]}."
                ),
            )

    def _resolve_root_hop(
        self, cursor: sqlite3.Cursor, previous_hop_id: str | None, parent_hop_id: str | None
    ) -> str | None:
        reference = parent_hop_id or previous_hop_id
        if not reference:
            return None
        row = cursor.execute(
            "SELECT COALESCE(root_hop_id, hop_id) AS root_hop_id FROM conversation_hops WHERE hop_id = ?",
            (reference,),
        ).fetchone()
        return str(row["root_hop_id"]) if row else None

    def _resolve_depth(
        self, cursor: sqlite3.Cursor, previous_hop_id: str | None, parent_hop_id: str | None
    ) -> int:
        reference = parent_hop_id or previous_hop_id
        if not reference:
            return 0
        row = cursor.execute(
            "SELECT depth_from_root FROM conversation_hops WHERE hop_id = ?",
            (reference,),
        ).fetchone()
        return int(row["depth_from_root"]) + 1 if row else 0

    def table_count(self, table_name: str) -> int:
        if table_name not in {
            "conversation_topics",
            "conversation_hops",
            "knowledge_topics",
            "knowledge_sources",
            "knowledge_chunks",
            "generated_artifacts",
            "reminders",
            "reminder_notifications",
            "indexing_outbox",
            "mutation_requests",
            "pending_action_confirmations",
        }:
            raise ValueError("Unknown table")
        row = self.connection.execute(f"SELECT COUNT(*) AS total FROM {table_name}").fetchone()
        return int(row["total"])

    def list_notifications(self, *, user_id: str, include_deleted: bool = False) -> list[dict[str, Any]]:
        query = """
            SELECT n.notification_id, n.reminder_id, n.user_id, n.ui_status,
                   n.delivery_status, n.delivery_attempts, n.last_delivery_error,
                   n.sent_at, n.fire_time, n.created_at, n.read_at, n.deleted_at,
                   r.reminder_time, r.status AS reminder_status, r.subject, r.reminder_summary
            FROM reminder_notifications n
            JOIN reminders r ON r.reminder_id = n.reminder_id AND r.user_id = n.user_id
            WHERE n.user_id = ?
        """
        params: list[Any] = [user_id]
        if not include_deleted:
            query += " AND n.ui_status != 'deleted'"
        # Sidebar order is nearest due time at the top and farthest overdue
        # time at the bottom; creation time is only a stable tie-breaker.
        query += " ORDER BY COALESCE(n.fire_time, r.reminder_time, n.created_at) DESC, n.created_at DESC"
        rows = self.connection.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def update_notification_ui_status(
        self, *, user_id: str, notification_id: str, ui_status: str
    ) -> dict[str, Any]:
        if ui_status not in {"unread", "read", "deleted"}:
            raise ValueError("Invalid notification UI status")
        timestamp = now_iso()
        with self.transaction() as cursor:
            if ui_status == "read":
                cursor.execute(
                    """
                    UPDATE reminder_notifications
                    SET ui_status = ?, read_at = ?, deleted_at = NULL
                    WHERE user_id = ? AND notification_id = ?
                    """,
                    (ui_status, timestamp, user_id, notification_id),
                )
            elif ui_status == "unread":
                # Re-opening a notification must remove the acknowledgement
                # timestamp, otherwise it is still semantically "read".
                cursor.execute(
                    """
                    UPDATE reminder_notifications
                    SET ui_status = ?, read_at = NULL, deleted_at = NULL
                    WHERE user_id = ? AND notification_id = ?
                    """,
                    (ui_status, user_id, notification_id),
                )
            else:
                cursor.execute(
                    """
                    UPDATE reminder_notifications
                    SET ui_status = ?, deleted_at = ?
                    WHERE user_id = ? AND notification_id = ?
                    """,
                    (ui_status, timestamp, user_id, notification_id),
                )
            if cursor.rowcount != 1:
                raise ValueError("Notification not found for user")
        row = self.connection.execute(
            """
            SELECT notification_id, reminder_id, user_id, ui_status,
                   delivery_status, delivery_attempts, last_delivery_error, sent_at,
                   fire_time, created_at, read_at, deleted_at
            FROM reminder_notifications
            WHERE user_id = ? AND notification_id = ?
            """,
            (user_id, notification_id),
        ).fetchone()
        return dict(row)

    def update_all_notification_ui_status(self, *, user_id: str, ui_status: str) -> int:
        """Atomically mark this user's visible notifications read or unread."""
        if ui_status not in {"unread", "read"}:
            raise ValueError("Bulk notification status must be read or unread")
        with self.transaction() as cursor:
            if ui_status == "read":
                cursor.execute(
                    """
                    UPDATE reminder_notifications
                    SET ui_status = 'read', read_at = ?, deleted_at = NULL
                    WHERE user_id = ? AND ui_status != 'deleted' AND ui_status != 'read'
                    """,
                    (now_iso(), user_id),
                )
            else:
                cursor.execute(
                    """
                    UPDATE reminder_notifications
                    SET ui_status = 'unread', read_at = NULL, deleted_at = NULL
                    WHERE user_id = ? AND ui_status != 'deleted' AND ui_status != 'unread'
                    """,
                    (user_id,),
                )
            return int(cursor.rowcount)

    def mark_notification_delivery_sent(
        self, *, user_id: str, notification_id: str
    ) -> dict[str, Any]:
        timestamp = now_iso()
        with self.transaction() as cursor:
            cursor.execute(
                """
                UPDATE reminder_notifications
                SET delivery_status = 'sent',
                    sent_at = ?,
                    last_delivery_error = NULL
                WHERE user_id = ? AND notification_id = ?
                """,
                (timestamp, user_id, notification_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("Notification not found for user")
        row = self.connection.execute(
            "SELECT * FROM reminder_notifications WHERE user_id = ? AND notification_id = ?",
            (user_id, notification_id),
        ).fetchone()
        return dict(row)

    def mark_notification_delivery_failed(
        self, *, user_id: str, notification_id: str, error_message: str, retrying: bool = False
    ) -> dict[str, Any]:
        status = "retrying" if retrying else "failed"
        with self.transaction() as cursor:
            cursor.execute(
                """
                UPDATE reminder_notifications
                SET delivery_status = ?,
                    delivery_attempts = delivery_attempts + 1,
                    last_delivery_error = ?
                WHERE user_id = ? AND notification_id = ?
                """,
                (status, error_message, user_id, notification_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("Notification not found for user")
        row = self.connection.execute(
            "SELECT * FROM reminder_notifications WHERE user_id = ? AND notification_id = ?",
            (user_id, notification_id),
        ).fetchone()
        return dict(row)

    def claim_idempotency_key(
        self, *, user_id: str, idempotency_key: str, payload_hash: str
    ) -> IdempotencyClaimResult:
        timestamp = now_iso()
        existing = self.connection.execute(
            """
            SELECT request_id, payload_hash, status, stored_response_json
            FROM mutation_requests
            WHERE user_id = ? AND idempotency_key = ?
            """,
            (user_id, idempotency_key),
        ).fetchone()
        if existing:
            if existing["payload_hash"] != payload_hash:
                return IdempotencyClaimResult(
                    status="conflict",
                    request_id=existing["request_id"],
                    reason="idempotency_conflict",
                )
            if existing["status"] == MutationRequestStatus.COMPLETED.value:
                return IdempotencyClaimResult(
                    status="replay",
                    request_id=existing["request_id"],
                    stored_response_json=existing["stored_response_json"],
                )
            if existing["status"] == MutationRequestStatus.IN_PROGRESS.value:
                return IdempotencyClaimResult(
                    status="in_progress",
                    request_id=existing["request_id"],
                    reason="request_in_progress",
                )
            return IdempotencyClaimResult(status="failed_retry", request_id=existing["request_id"])

        request_id = new_id()
        with self.transaction() as cursor:
            cursor.execute(
                """
                INSERT INTO mutation_requests (
                    request_id, user_id, idempotency_key, payload_hash,
                    status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'in_progress', ?, ?)
                """,
                (request_id, user_id, idempotency_key, payload_hash, timestamp, timestamp),
            )
        return IdempotencyClaimResult(status="started", request_id=request_id)

    def complete_idempotency_request(self, *, request_id: str, stored_response_json: str) -> None:
        with self.transaction() as cursor:
            cursor.execute(
                """
                UPDATE mutation_requests
                SET status = 'completed', stored_response_json = ?, updated_at = ?
                WHERE request_id = ?
                """,
                (stored_response_json, now_iso(), request_id),
            )

    def fail_idempotency_request(self, *, request_id: str, error_message: str | None = None) -> None:
        with self.transaction() as cursor:
            cursor.execute(
                """
                UPDATE mutation_requests
                SET status = 'failed', updated_at = ?
                WHERE request_id = ?
                """,
                (now_iso(), request_id),
            )

    def create_pending_confirmation(
        self,
        *,
        user_id: str,
        action_type: str,
        target_entity_type: str,
        target_entity_id: str | None,
        proposed_action: dict[str, Any],
        target_snapshot: dict[str, Any],
        expires_at: str,
    ) -> dict[str, Any]:
        token = new_id()
        timestamp = now_iso()
        with self.transaction() as cursor:
            cursor.execute(
                """
                INSERT INTO pending_action_confirmations (
                    confirmation_token, user_id, action_type, target_entity_type,
                    target_entity_id, proposed_action_json, target_snapshot_json,
                    expires_at, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)
                """,
                (
                    token,
                    user_id,
                    action_type,
                    target_entity_type,
                    target_entity_id,
                    json.dumps(proposed_action, default=str),
                    json.dumps(target_snapshot, default=str),
                    expires_at,
                    timestamp,
                ),
            )
        return {
            "confirmation_token": token,
            "action_type": action_type,
            "target_entity_type": target_entity_type,
            "target_entity_id": target_entity_id,
            "expires_at": expires_at,
            "status": ConfirmationStatus.PENDING.value,
        }

    def load_pending_confirmation(
        self, *, user_id: str, confirmation_token: str, now_value: str
    ) -> dict[str, Any]:
        row = self.connection.execute(
            """
            SELECT * FROM pending_action_confirmations
            WHERE user_id = ? AND confirmation_token = ?
            """,
            (user_id, confirmation_token),
        ).fetchone()
        if not row:
            raise ValueError("Confirmation not found for user")
        payload = dict(row)
        if payload["status"] != ConfirmationStatus.PENDING.value:
            raise ValueError("Confirmation is not pending")
        if payload["expires_at"] <= now_value:
            with self.transaction() as cursor:
                cursor.execute(
                    "UPDATE pending_action_confirmations SET status = 'expired' WHERE confirmation_token = ?",
                    (confirmation_token,),
                )
            raise ValueError("Confirmation has expired")
        payload["proposed_action"] = json.loads(payload["proposed_action_json"])
        payload["target_snapshot"] = json.loads(payload["target_snapshot_json"])
        return payload

    def mark_confirmation_confirmed(self, *, user_id: str, confirmation_token: str) -> None:
        with self.transaction() as cursor:
            cursor.execute(
                """
                UPDATE pending_action_confirmations
                SET status = 'confirmed', confirmed_at = ?
                WHERE user_id = ? AND confirmation_token = ? AND status = 'pending'
                """,
                (now_iso(), user_id, confirmation_token),
            )
            if cursor.rowcount != 1:
                raise ValueError("Confirmation could not be confirmed")

    def list_reminders(
        self,
        *,
        user_id: str,
        status: str | None = None,
        from_time: str | None = None,
        to_time: str | None = None,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM reminders WHERE user_id = ?"
        params: list[Any] = [user_id]
        if status:
            query += " AND status = ?"
            params.append(status)
        if from_time:
            query += " AND reminder_time >= ?"
            params.append(from_time)
        if to_time:
            query += " AND reminder_time <= ?"
            params.append(to_time)
        query += " ORDER BY reminder_time"
        return [dict(row) for row in self.connection.execute(query, params).fetchall()]

    def append_reminder_reply(
        self,
        *,
        user_id: str,
        reminder_id: str,
        notification_id: str,
        reply_text: str,
        response_text: str,
    ) -> HopWrite:
        context = self.load_reminder_reply_context(
            user_id=user_id,
            reminder_id=reminder_id,
            notification_id=notification_id,
        )
        if not context:
            raise ValueError("Reminder reply context not found for user")
        from .reminder_reply import build_reminder_state, reminder_reply_hop_entities

        reply_hop_id = new_id()
        reminder_state = build_reminder_state(
            context,
            reminder_id=reminder_id,
            notification_id=notification_id,
        )
        with self.transaction() as cursor:
            topic_id = context.get("source_topic_id") or self.ensure_topic(
                cursor, user_id=user_id, title="Reminders"
            )
            hop = self.append_conversation_hop(
                cursor,
                topic_id=topic_id,
                user_id=user_id,
                parent_hop_id=context.get("source_hop_id"),
                intent="reminder",
                raw_user_query=reply_text,
                rewritten_user_query=reply_text,
                raw_response=response_text,
                response_type="reminder_reply",
                entities=reminder_reply_hop_entities(
                    reminder_state,
                    reply_hop_id=reply_hop_id,
                ),
                hop_id=reply_hop_id,
            )
        return hop
