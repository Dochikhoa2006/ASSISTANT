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
)
from .errors import (
    RepositoryConflictError,
    KnowledgeConflictError,
    ReminderConflictError,
    RepositoryValidationError,
    RepositoryTransactionError,
)
from .prompts import DEFAULT_PROMPT_REGISTRY


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


@dataclass(frozen=True)
class HopWrite:
    topic_id: str
    hop_id: str
    previous_hop_id: str | None
    outbox_job_id: str


class SQLRepository:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")

    @classmethod
    def in_memory(cls) -> "SQLRepository":
        return cls(sqlite3.connect(":memory:"))

    @classmethod
    def persistent(
        cls,
        database_path: str,
        *,
        enable_wal: bool,
        busy_timeout_ms: int,
    ) -> "SQLRepository":
        connection = sqlite3.connect(database_path)
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
                UNIQUE (user_id, content_hash),
                FOREIGN KEY (knowledge_topic_id) REFERENCES knowledge_topics(knowledge_topic_id)
            );

            CREATE TABLE IF NOT EXISTS reminders (
                reminder_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                source_topic_id TEXT NULL,
                source_hop_id TEXT NULL,
                reminder_time TEXT NOT NULL,
                status TEXT NOT NULL CHECK (status IN ('scheduled', 'notified', 'read', 'dismissed', 'cancelled')),
                raw_reminder TEXT NOT NULL,
                reminder_summary TEXT NOT NULL,
                subject TEXT NOT NULL,
                supporting_question TEXT,
                supporting_response TEXT,
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
                ui_status TEXT NOT NULL CHECK (ui_status IN ('unread', 'read', 'deleted')),
                created_at TEXT NOT NULL,
                read_at TEXT NULL,
                deleted_at TEXT NULL,
                UNIQUE (reminder_id, user_id),
                FOREIGN KEY (reminder_id) REFERENCES reminders(reminder_id)
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
            CREATE INDEX IF NOT EXISTS idx_reminders_user_status_time ON reminders(user_id, status, reminder_time);
            CREATE INDEX IF NOT EXISTS idx_reminders_user_updated ON reminders(user_id, updated_at);
            CREATE INDEX IF NOT EXISTS idx_reminders_user_created ON reminders(user_id, created_at);
            """
        )
        self.connection.commit()

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
        hop_id = new_id()
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
                raw_user_query,
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
        return HopWrite(topic_id, hop_id, previous_hop_id, outbox_job_id)

    def add_knowledge_chunk(
        self,
        cursor: sqlite3.Cursor,
        *,
        user_id: str,
        title: str,
        text: str,
        source_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[str, str]:
        topic_id = self.ensure_knowledge_topic(cursor, user_id=user_id, title=title)
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
        normalized = " ".join(text.split())
        cursor.execute(
            """
            INSERT INTO knowledge_chunks (
                chunk_id, knowledge_topic_id, user_id, source_id, chunk_index,
                raw_text, normalized_text, summary, metadata_json, content_hash,
                is_deleted, created_at, updated_at, version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, 1)
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
                content_hash(user_id, normalized),
                timestamp,
                timestamp,
            ),
        )
        self.insert_outbox_job(
            cursor,
            entity_type=OutboxEntityType.KNOWLEDGE_CHUNK,
            entity_id=chunk_id,
            operation=OutboxOperation.UPSERT,
        )
        return topic_id, chunk_id

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
            """,
            (topic_id, user_id, title, timestamp, timestamp),
        )
        return topic_id

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

    def soft_delete_knowledge_chunk(self, cursor: sqlite3.Cursor, *, user_id: str, chunk_id: str, expected_version: int | None = None) -> None:
        timestamp = now_iso()
        query = """
            UPDATE knowledge_chunks
            SET is_deleted = 1, updated_at = ?, version = version + 1
            WHERE user_id = ? AND chunk_id = ? AND is_deleted = 0
        """
        params: list[Any] = [timestamp, user_id, chunk_id]
        if expected_version is not None:
            query += " AND version = ?"
            params.append(expected_version)
            
        cursor.execute(query, tuple(params))
        if cursor.rowcount != 1:
            raise KnowledgeConflictError("Active knowledge chunk not found or version mismatch for user")
        self.insert_outbox_job(
            cursor,
            entity_type=OutboxEntityType.KNOWLEDGE_CHUNK,
            entity_id=chunk_id,
            operation=OutboxOperation.DELETE,
        )

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
        supporting_question: str | None = None,
        supporting_response: str | None = None,
    ) -> str:
        reminder_id = new_id()
        timestamp = now_iso()
        cursor.execute(
            """
            INSERT INTO reminders (
                reminder_id, user_id, source_topic_id, source_hop_id,
                reminder_time, status, raw_reminder, reminder_summary, subject,
                supporting_question, supporting_response, created_at, updated_at, version
            ) VALUES (?, ?, ?, ?, ?, 'scheduled', ?, ?, ?, ?, ?, ?, ?, 1)
            """,
            (
                reminder_id,
                user_id,
                source_topic_id,
                source_hop_id,
                reminder_time,
                raw_reminder,
                reminder_summary,
                subject,
                supporting_question,
                supporting_response,
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
        self, cursor: sqlite3.Cursor, *, user_id: str, reminder_id: str
    ) -> str:
        current = cursor.execute(
            """
            SELECT notification_id FROM reminder_notifications
            WHERE user_id = ? AND reminder_id = ?
            """,
            (user_id, reminder_id),
        ).fetchone()
        if current:
            return str(current["notification_id"])
        notification_id = new_id()
        timestamp = now_iso()
        cursor.execute(
            """
            INSERT INTO reminder_notifications (
                notification_id, reminder_id, user_id, ui_status, created_at
            ) VALUES (?, ?, ?, 'unread', ?)
            """,
            (notification_id, reminder_id, user_id, timestamp),
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
            
        job_id = new_id()
        timestamp = now_iso()
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
    ) -> RepositoryTransactionResult:
        for action in actions:
            if action.validation_result != ActionValidationResult.EXECUTE:
                raise ValueError("Repository methods must receive only executable actions.")
        try:
            with self.transaction() as cursor:
                topic_id = self.ensure_topic(cursor, user_id=user_id, title=topic_title)
                results: list[RepositoryActionResult] = []
                outbox_jobs: list[str] = []
                for action in actions:
                    result = self._apply_knowledge_action(cursor, user_id=user_id, action=action)
                    results.append(result)
                    outbox_jobs.extend(result.indexing_outbox_ids)
                hop = self.append_conversation_hop(
                    cursor,
                    topic_id=topic_id,
                    user_id=user_id,
                    intent="knowledge_facts",
                    raw_user_query=raw_user_query,
                    rewritten_user_query=rewritten_user_query,
                    raw_response=response_text,
                    response_type="knowledge_action",
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
    ) -> RepositoryTransactionResult:
        for action in actions:
            if action.validation_result != ActionValidationResult.EXECUTE:
                raise ValueError("Repository methods must receive only executable actions.")
        try:
            with self.transaction() as cursor:
                topic_id = self.ensure_topic(cursor, user_id=user_id, title=topic_title)
                results: list[RepositoryActionResult] = []
                reminder_entities: list[dict[str, str]] = []
                for action in actions:
                    result = self._apply_reminder_action(
                        cursor, user_id=user_id, source_topic_id=topic_id, action=action
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
            _, chunk_id = self.add_knowledge_chunk(
                cursor,
                user_id=user_id,
                title=action.topic_title or "Knowledge",
                text=action.knowledge_text or action.new_text or "",
                source_id=None,
                metadata=None,
            )
            return RepositoryActionResult(
                action_id=new_id(),
                action_type=action_type.value,
                status="committed",
                domain_entity_type="knowledge_chunk",
                domain_entity_id=chunk_id,
                indexing_outbox_ids=(),
            )
        if action_type in {KnowledgeAction.DELETE, KnowledgeAction.MODIFY}:
            chunk_id = action.target_chunk_ids[0]
            expected_version = action.observed_versions.get(chunk_id)
            self.soft_delete_knowledge_chunk(cursor, user_id=user_id, chunk_id=chunk_id, expected_version=expected_version)
            if action_type is KnowledgeAction.MODIFY:
                _, new_chunk_id = self.add_knowledge_chunk(
                    cursor,
                    user_id=user_id,
                    title=action.topic_title or "Knowledge",
                    text=action.replacement_text or action.new_text or "",
                    source_id=None,
                    metadata=None,
                )
                return RepositoryActionResult(
                    action_id=new_id(),
                    action_type=action_type.value,
                    status="committed",
                    domain_entity_type="knowledge_chunk",
                    domain_entity_id=new_chunk_id,
                    indexing_outbox_ids=(),
                )
            return RepositoryActionResult(
                action_id=new_id(),
                action_type=action_type.value,
                status="committed",
                domain_entity_type="knowledge_chunk",
                domain_entity_id=chunk_id,
                indexing_outbox_ids=(),
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
    ) -> RepositoryActionResult:
        action_type = action.action
        if action_type is ReminderAction.ADD:
            reminder_id = self.add_reminder(
                cursor,
                user_id=user_id,
                source_topic_id=source_topic_id,
                source_hop_id=None,
                reminder_time=action.reminder_time.isoformat() if action.reminder_time else "",
                raw_reminder=action.raw_reminder or "",
                reminder_summary=action.reminder_summary or "",
                subject=action.subject or "",
            )
            return RepositoryActionResult(
                action_id=new_id(),
                action_type=action_type.value,
                status="committed",
                domain_entity_type="reminder",
                domain_entity_id=reminder_id,
                indexing_outbox_ids=(),
            )
        elif action_type is ReminderAction.MODIFY:
            reminder_id = action.target_reminder_ids[0]
            self.update_reminder_status(
                cursor,
                user_id=user_id,
                reminder_id=reminder_id,
                status="cancelled",
                expected_version=action.observed_version,
                expected_status=action.observed_status,
            )
            new_reminder_id = self.add_reminder(
                cursor,
                user_id=user_id,
                source_topic_id=source_topic_id,
                source_hop_id=None,
                reminder_time=action.replacement_time.isoformat() if action.replacement_time else (action.reminder_time.isoformat() if action.reminder_time else ""),
                raw_reminder=action.raw_reminder or "",
                reminder_summary=action.replacement_summary or action.reminder_summary or "",
                subject=action.replacement_subject or action.subject or "",
            )
            return RepositoryActionResult(
                action_id=new_id(),
                action_type=action_type.value,
                status="committed",
                domain_entity_type="reminder",
                domain_entity_id=new_reminder_id,
                indexing_outbox_ids=(),
            )
        else:
            reminder_id = action.target_reminder_ids[0]
            status_by_action = {
                ReminderAction.TURN_ON: "scheduled",
                ReminderAction.TURN_OFF: "cancelled",
                ReminderAction.DELETE: "dismissed",
            }
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
            "knowledge_chunks",
            "reminders",
            "reminder_notifications",
            "indexing_outbox",
        }:
            raise ValueError("Unknown table")
        row = self.connection.execute(f"SELECT COUNT(*) AS total FROM {table_name}").fetchone()
        return int(row["total"])

    def list_notifications(self, *, user_id: str, include_deleted: bool = False) -> list[dict[str, Any]]:
        query = """
            SELECT n.notification_id, n.reminder_id, n.user_id, n.ui_status,
                   n.created_at, n.read_at, n.deleted_at,
                   r.reminder_time, r.status AS reminder_status, r.subject, r.reminder_summary
            FROM reminder_notifications n
            JOIN reminders r ON r.reminder_id = n.reminder_id AND r.user_id = n.user_id
            WHERE n.user_id = ?
        """
        params: list[Any] = [user_id]
        if not include_deleted:
            query += " AND n.ui_status != 'deleted'"
        query += " ORDER BY n.created_at DESC"
        rows = self.connection.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def update_notification_ui_status(
        self, *, user_id: str, notification_id: str, ui_status: str
    ) -> dict[str, Any]:
        if ui_status not in {"unread", "read", "deleted"}:
            raise ValueError("Invalid notification UI status")
        timestamp = now_iso()
        read_at = timestamp if ui_status == "read" else None
        deleted_at = timestamp if ui_status == "deleted" else None
        with self.transaction() as cursor:
            cursor.execute(
                """
                UPDATE reminder_notifications
                SET ui_status = ?,
                    read_at = COALESCE(?, read_at),
                    deleted_at = COALESCE(?, deleted_at)
                WHERE user_id = ? AND notification_id = ?
                """,
                (ui_status, read_at, deleted_at, user_id, notification_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("Notification not found for user")
        row = self.connection.execute(
            """
            SELECT notification_id, reminder_id, user_id, ui_status, created_at, read_at, deleted_at
            FROM reminder_notifications
            WHERE user_id = ? AND notification_id = ?
            """,
            (user_id, notification_id),
        ).fetchone()
        return dict(row)

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
        context = self.connection.execute(
            """
            SELECT r.source_topic_id, r.source_hop_id, r.subject
            FROM reminders r
            JOIN reminder_notifications n
              ON n.reminder_id = r.reminder_id AND n.user_id = r.user_id
            WHERE r.user_id = ? AND r.reminder_id = ? AND n.notification_id = ?
            """,
            (user_id, reminder_id, notification_id),
        ).fetchone()
        if not context:
            raise ValueError("Reminder reply context not found for user")
        with self.transaction() as cursor:
            topic_id = context["source_topic_id"] or self.ensure_topic(
                cursor, user_id=user_id, title="Reminders"
            )
            hop = self.append_conversation_hop(
                cursor,
                topic_id=topic_id,
                user_id=user_id,
                parent_hop_id=context["source_hop_id"],
                intent="reminder",
                raw_user_query=reply_text,
                rewritten_user_query=reply_text,
                raw_response=response_text,
                response_type="reminder_reply",
                entities={
                    "reminder_id": reminder_id,
                    "notification_id": notification_id,
                    "subject": context["subject"],
                },
            )
        return hop
