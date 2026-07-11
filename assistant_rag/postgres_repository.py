from __future__ import annotations
import os
import json
import uuid
from contextlib import contextmanager
from typing import Any, Iterator, Optional
from datetime import datetime, timezone

from sqlalchemy import create_engine, select, insert, update, delete, and_, or_, func, case
from sqlalchemy.engine import Engine, Connection

from .repository import AssistantRepository
from .contracts import *
from .postgres_schema import *
from .database import content_hash
from .reminder_safety import normalize_subject, token_similarity, utc_minute, within_minutes
from .recurrence import calculate_next_fire_time
from .lifecycle import is_artifact_downloadable, is_indexable_conversation_hop, is_indexable_knowledge_chunk
from .metrics import GLOBAL_METRICS
from .errors import (
    KnowledgeConflictError,
    ReminderConflictError,
    RepositoryConflictError,
    RepositoryValidationError,
)

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def new_id() -> str:
    return uuid.uuid4().hex

class PostgresRepository(AssistantRepository):
    def __init__(self, engine: Engine):
        self.engine = engine
        self._local_conn: Connection | None = None

    @classmethod
    def create(cls, url: str, pool_size: int = 10, max_overflow: int = 20) -> 'PostgresRepository':
        kwargs = {}
        if not url.startswith("sqlite"):
            kwargs = {"pool_size": pool_size, "max_overflow": max_overflow}
        engine = create_engine(url, **kwargs)
        return cls(engine)

    def close(self) -> None:
        self.engine.dispose()

    @classmethod
    def in_memory(cls) -> 'SQLiteRepository':
        raise NotImplementedError("Use SQLiteRepository for in-memory databases")

    @classmethod
    def persistent(cls, database_path: str, *, enable_wal: bool, busy_timeout_ms: int) -> 'SQLiteRepository':
        raise NotImplementedError("Use SQLiteRepository for persistent SQLite databases")

    def initialize_schema(self) -> None:
        metadata.create_all(self.engine)

    @contextmanager
    def transaction(self) -> Iterator[Connection]:
        if self._local_conn is not None:
            with self._local_conn.begin_nested():
                yield self._local_conn
        else:
            with self.engine.begin() as conn:
                yield conn

    def table_count(self, table_name: str) -> int:
        table = metadata.tables[table_name]
        with self.engine.connect() as conn:
            return conn.execute(select(func.count()).select_from(table)).scalar() or 0

    def ensure_topic(
        self,
        cursor: Connection,
        *,
        user_id: str,
        title: str,
        topic_summary: str = "",
        state_summary: str = "",
        entities: dict[str, Any] | None = None,
    ) -> str:
        stmt = select(conversation_topics.c.topic_id).where(
            and_(
                conversation_topics.c.user_id == user_id,
                conversation_topics.c.title == title,
                conversation_topics.c.status == 'active'
            )
        ).order_by(conversation_topics.c.updated_at.desc()).limit(1)
        
        row = cursor.execute(stmt).fetchone()
        if row:
            return str(row[0])
            
        topic_id = new_id()
        timestamp = now_iso()
        cursor.execute(
            insert(conversation_topics).values(
                topic_id=topic_id,
                user_id=user_id,
                title=title,
                topic_summary=topic_summary,
                state_summary=state_summary,
                entities_json=json.dumps(entities or {}),
                last_hop_id=None,
                status='active',
                created_at=timestamp,
                updated_at=timestamp,
                version=1
            )
        )
        return topic_id

    def append_conversation_hop(
        self,
        cursor: Connection,
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
        stmt = select(conversation_topics.c.last_hop_id).where(
            and_(
                conversation_topics.c.topic_id == topic_id,
                conversation_topics.c.user_id == user_id,
                conversation_topics.c.status == 'active'
            )
        )
        row = cursor.execute(stmt).fetchone()
        if not row:
            raise ValueError("Active conversation topic not found for user")
            
        previous_hop_id = row[0]
        root_hop_id = self._resolve_root_hop(cursor, previous_hop_id, parent_hop_id)
        depth_from_root = self._resolve_depth(cursor, previous_hop_id, parent_hop_id)
        actual_hop_id = hop_id or new_id()
        effective_branch_id = branch_id or root_hop_id or actual_hop_id
        timestamp = now_iso()
        
        outbox_job_id = self.insert_outbox_job(
            cursor,
            entity_type=OutboxEntityType.CONVERSATION_HOP,
            entity_id=actual_hop_id,
            operation=OutboxOperation.UPSERT,
        )
        
        cursor.execute(
            insert(conversation_hops).values(
                hop_id=actual_hop_id,
                user_id=user_id,
                topic_id=topic_id,
                intent=intent,
                raw_user_query=raw_user_query,
                rewritten_user_query=rewritten_user_query,
                raw_response=raw_response,
                response_type=response_type,
                supporting_questions_json=json.dumps(supporting_questions or []),
                previous_hop_id=previous_hop_id,
                parent_hop_id=parent_hop_id,
                root_hop_id=root_hop_id,
                branch_id=effective_branch_id,
                depth_from_root=depth_from_root,
                entities_json=json.dumps(entities or {}),
                outbox_job_id=outbox_job_id,
                created_at=timestamp
            )
        )
        
        cursor.execute(
            update(conversation_topics).where(
                conversation_topics.c.topic_id == topic_id
            ).values(
                last_hop_id=actual_hop_id,
                state_summary=state_summary,
                updated_at=timestamp,
                version=conversation_topics.c.version + 1
            )
        )
        
        return HopWrite(
            topic_id=topic_id,
            hop_id=actual_hop_id,
            previous_hop_id=previous_hop_id,
            outbox_job_id=outbox_job_id
        )

    def _resolve_root_hop(
        self, cursor: Connection, previous_hop_id: str | None, parent_hop_id: str | None
    ) -> str | None:
        reference = parent_hop_id or previous_hop_id
        if not reference:
            return None
        row = cursor.execute(
            select(
                func.coalesce(conversation_hops.c.root_hop_id, conversation_hops.c.hop_id)
            ).where(conversation_hops.c.hop_id == reference)
        ).fetchone()
        return str(row[0]) if row else None

    def _resolve_depth(
        self, cursor: Connection, previous_hop_id: str | None, parent_hop_id: str | None
    ) -> int:
        reference = parent_hop_id or previous_hop_id
        if not reference:
            return 0
        row = cursor.execute(
            select(conversation_hops.c.depth_from_root).where(
                conversation_hops.c.hop_id == reference
            )
        ).fetchone()
        return int(row[0]) + 1 if row else 0

    def scan_due_reminders(self, *, now_value: str, limit: int = 100) -> list[str]:
        notified: list[str] = []
        with self.transaction() as cursor:
            due_expr = func.coalesce(reminders.c.next_fire_time, reminders.c.reminder_time)
            rows = cursor.execute(
                select(
                    reminders.c.reminder_id,
                    reminders.c.user_id,
                    reminders.c.reminder_time,
                    reminders.c.recurrence_rule,
                    reminders.c.recurrence_timezone,
                    reminders.c.next_fire_time,
                )
                # Deliberately global across users. The notification row still
                # carries the owning user_id and remains access-controlled.
                .where(and_(
                    reminders.c.status == "scheduled",
                    reminders.c.timing_plan_status == "planned",
                    due_expr <= now_value,
                ))
                .order_by(due_expr)
                .limit(limit)
            ).fetchall()
            for row in rows:
                fire_time = row.next_fire_time or row.reminder_time
                self.create_notification_if_absent(
                    cursor,
                    user_id=row.user_id,
                    reminder_id=row.reminder_id,
                    fire_time=fire_time,
                )
                if row.recurrence_rule:
                    next_fire_time = calculate_next_fire_time(
                        previous_fire_time=fire_time,
                        recurrence_rule=row.recurrence_rule,
                        recurrence_timezone=row.recurrence_timezone or "UTC",
                    )
                    values = {
                        "last_fire_time": fire_time,
                        "next_fire_time": next_fire_time,
                        "updated_at": now_iso(),
                        "version": reminders.c.version + 1,
                    }
                    if next_fire_time is None:
                        values["status"] = "completed"
                    cursor.execute(
                        update(reminders)
                        .where(
                            and_(
                                reminders.c.reminder_id == row.reminder_id,
                                reminders.c.user_id == row.user_id,
                                reminders.c.status == "scheduled",
                            )
                        )
                        .values(**values)
                    )
                else:
                    cursor.execute(
                        update(reminders)
                        .where(
                            and_(
                                reminders.c.reminder_id == row.reminder_id,
                                reminders.c.user_id == row.user_id,
                                reminders.c.status == "scheduled",
                            )
                        )
                        .values(
                            status="notified",
                            last_fire_time=fire_time,
                            updated_at=now_iso(),
                            version=reminders.c.version + 1,
                        )
                    )
                notified.append(str(row.reminder_id))
        GLOBAL_METRICS.increment("reminder_scans_total")
        GLOBAL_METRICS.increment("notification_created_total", len(notified))
        return notified

    def list_reminders_requiring_timing(self, *, limit: int = 100) -> list[PendingReminderTiming]:
        stmt = (
            select(
                reminders.c.reminder_id, reminders.c.user_id, reminders.c.event_time,
                reminders.c.reminder_time, reminders.c.subject, reminders.c.raw_reminder,
                reminders.c.user_timezone, reminders.c.recurrence_rule, reminders.c.version,
            )
            .where(and_(
                reminders.c.status == "scheduled",
                reminders.c.timing_plan_status == "pending",
            ))
            .order_by(func.coalesce(reminders.c.event_time, reminders.c.reminder_time), reminders.c.reminder_id)
            .limit(limit)
        )
        with self.engine.connect() as conn:
            rows = conn.execute(stmt).fetchall()
        candidates: list[PendingReminderTiming] = []
        for row in rows:
            source_value = row.event_time or row.reminder_time
            try:
                source_time = datetime.fromisoformat(source_value)
                if source_time.tzinfo is None:
                    source_time = source_time.replace(tzinfo=timezone.utc)
            except (TypeError, ValueError):
                source_time = datetime.min.replace(tzinfo=timezone.utc)
            candidates.append(PendingReminderTiming(
                reminder_id=str(row.reminder_id), user_id=str(row.user_id),
                source_time=source_time, subject=str(row.subject or ""),
                raw_reminder=str(row.raw_reminder or ""),
                user_timezone=str(row.user_timezone or "UTC"),
                recurrence_rule=row.recurrence_rule, version=int(row.version),
            ))
        return candidates

    def complete_reminder_timing_plan(
        self, *, reminder_id: str, user_id: str, expected_version: int,
        notification_time: datetime, reason: str,
    ) -> bool:
        notification_iso = notification_time.astimezone(timezone.utc).isoformat()
        with self.transaction() as cursor:
            result = cursor.execute(
                update(reminders)
                .where(and_(
                    reminders.c.reminder_id == reminder_id,
                    reminders.c.user_id == user_id,
                    reminders.c.version == expected_version,
                    reminders.c.status == "scheduled",
                    reminders.c.timing_plan_status == "pending",
                ))
                .values(
                    reminder_time=notification_iso,
                    next_fire_time=case(
                        (reminders.c.recurrence_rule.is_not(None), notification_iso),
                        else_=None,
                    ),
                    timing_plan_status="planned", timing_planned_at=now_iso(),
                    timing_plan_reason=reason, updated_at=now_iso(),
                    version=reminders.c.version + 1,
                )
            )
            return result.rowcount == 1

    def fail_reminder_timing_plan(
        self, *, reminder_id: str, user_id: str, expected_version: int, reason: str,
    ) -> bool:
        with self.transaction() as cursor:
            result = cursor.execute(
                update(reminders)
                .where(and_(
                    reminders.c.reminder_id == reminder_id,
                    reminders.c.user_id == user_id,
                    reminders.c.version == expected_version,
                    reminders.c.status == "scheduled",
                    reminders.c.timing_plan_status == "pending",
                ))
                .values(
                    timing_plan_status="needs_review", timing_planned_at=now_iso(),
                    timing_plan_reason=reason, updated_at=now_iso(),
                    version=reminders.c.version + 1,
                )
            )
            return result.rowcount == 1

    def load_reminder_reply_context(self, *, user_id: str, reminder_id: str, notification_id: str) -> dict[str, str | None]:
        stmt = select(
            reminders.c.subject,
            reminders.c.reminder_summary,
            reminders.c.supporting_question,
            reminders.c.supporting_response,
            reminders.c.source_hop_id,
            reminder_notifications.c.ui_status,
            conversation_hops.c.topic_id,
            conversation_hops.c.raw_user_query.label("source_raw_user_query"),
            conversation_hops.c.rewritten_user_query.label("source_rewritten_user_query"),
            conversation_hops.c.raw_response.label("source_raw_response"),
            conversation_hops.c.supporting_questions_json,
            conversation_hops.c.response_type.label("source_response_type"),
        ).select_from(
            reminders.join(
                reminder_notifications,
                reminders.c.reminder_id == reminder_notifications.c.reminder_id
            ).outerjoin(
                conversation_hops,
                and_(
                    reminders.c.source_hop_id == conversation_hops.c.hop_id,
                    reminders.c.user_id == conversation_hops.c.user_id,
                ),
            )
        ).where(
            and_(
                reminders.c.user_id == user_id,
                reminders.c.reminder_id == reminder_id,
                reminder_notifications.c.notification_id == notification_id
            )
        )
        with self.engine.connect() as conn:
            row = conn.execute(stmt).fetchone()
            if not row:
                raise ValueError("Reminder not found or does not belong to user")
            
            return {
                "subject": row[0],
                "reminder_summary": row[1],
                "supporting_question": row[2],
                "supporting_response": row[3],
                "source_hop_id": row[4],
                "ui_status": row[5],
                "topic_id": row[6],
                "source_raw_user_query": row[7],
                "source_rewritten_user_query": row[8],
                "source_raw_response": row[9],
                "supporting_questions_json": row[10],
                "source_response_type": row[11],
            }


    def add_knowledge_chunk(
        self,
        cursor: Connection,
        *,
        user_id: str,
        title: str,
        text: str,
        source_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        replaces_chunk_id: str | None = None,
        change_reason: str | None = None,
        modified_by_user_query: str | None = None,
    ) -> tuple[str, str, str | None]:
        topic_id = self.ensure_knowledge_topic(cursor, user_id=user_id, title=title)
        
        normalized = " ".join(text.split())
        c_hash = content_hash(user_id, normalized)
        
        stmt = select(knowledge_chunks.c.chunk_id, knowledge_chunks.c.is_deleted).where(
            and_(
                knowledge_chunks.c.user_id == user_id,
                knowledge_chunks.c.knowledge_topic_id == topic_id,
                knowledge_chunks.c.content_hash == c_hash,
            )
        )
        existing = cursor.execute(stmt).fetchone()
        
        if existing:
            existing_chunk_id = existing.chunk_id
            if existing.is_deleted == 0:
                return topic_id, existing_chunk_id, None
            else:
                update_stmt = update(knowledge_chunks).where(
                    knowledge_chunks.c.chunk_id == existing_chunk_id
                ).values(
                    is_deleted=0,
                updated_at=now_iso(),
                version=knowledge_chunks.c.version + 1
            )
                cursor.execute(update_stmt)
                outbox_job_id = self.insert_outbox_job(
                    cursor,
                    entity_type=OutboxEntityType.KNOWLEDGE_CHUNK,
                    entity_id=existing_chunk_id,
                    operation=OutboxOperation.UPSERT,
                )
                return topic_id, existing_chunk_id, outbox_job_id
                
        chunk_id = new_id()
        timestamp = now_iso()
        
        outbox_job_id = self.insert_outbox_job(
            cursor,
            entity_type=OutboxEntityType.KNOWLEDGE_CHUNK,
            entity_id=chunk_id,
            operation=OutboxOperation.UPSERT,
        )
        
        cursor.execute(
            insert(knowledge_chunks).values(
                chunk_id=chunk_id,
                knowledge_topic_id=topic_id,
                user_id=user_id,
                source_id=source_id,
                chunk_index=0,
                raw_text=text,
                normalized_text=text,
                summary=text,
                metadata_json=json.dumps(metadata or {}),
                content_hash=c_hash,
                is_deleted=0,
                created_at=timestamp,
                updated_at=timestamp,
                version=1,
                replaces_chunk_id=replaces_chunk_id,
                change_reason=change_reason,
                modified_by_user_query=modified_by_user_query,
            )
        )
        return topic_id, chunk_id, outbox_job_id

    def ensure_knowledge_topic(
        self,
        cursor: Connection,
        *,
        user_id: str,
        title: str,
    ) -> str:
        stmt = select(knowledge_topics.c.knowledge_topic_id).where(
            and_(
                knowledge_topics.c.user_id == user_id,
                knowledge_topics.c.title == title
            )
        ).order_by(knowledge_topics.c.updated_at.desc()).limit(1)
        row = cursor.execute(stmt).fetchone()
        if row:
            return str(row[0])
            
        topic_id = new_id()
        timestamp = now_iso()
        cursor.execute(
            insert(knowledge_topics).values(
                knowledge_topic_id=topic_id,
                user_id=user_id,
                title=title,
                description='',
                entities_json='{}',
                created_at=timestamp,
                updated_at=timestamp,
                version=1
            )
        )
        return topic_id

    def get_knowledge_chunks_by_ids(
        self, user_id: str, chunk_ids: list[str], include_deleted: bool = False
    ) -> list[dict[str, Any]]:
        if not chunk_ids:
            return []
            
        stmt = select(knowledge_chunks).where(
            and_(
                knowledge_chunks.c.user_id == user_id,
                knowledge_chunks.c.chunk_id.in_(chunk_ids)
            )
        )
        if not include_deleted:
            stmt = stmt.where(knowledge_chunks.c.is_deleted == 0)
            
        with self.engine.connect() as conn:
            rows = conn.execute(stmt).fetchall()
            return [dict(r._mapping) for r in rows]

    def soft_delete_knowledge_chunk(
        self,
        cursor: Connection,
        *,
        user_id: str,
        chunk_id: str,
        expected_version: int | None = None,
        change_reason: str | None = None,
        modified_by_user_query: str | None = None,
    ) -> str:
        timestamp = now_iso()
        stmt = update(knowledge_chunks).where(
            and_(
                knowledge_chunks.c.user_id == user_id,
                knowledge_chunks.c.chunk_id == chunk_id,
                knowledge_chunks.c.is_deleted == 0
            )
        )
        if expected_version is not None:
            stmt = stmt.where(knowledge_chunks.c.version == expected_version)
            
        stmt = stmt.values(
            is_deleted=1,
            updated_at=timestamp,
            version=knowledge_chunks.c.version + 1,
            change_reason=func.coalesce(change_reason, knowledge_chunks.c.change_reason),
            modified_by_user_query=func.coalesce(modified_by_user_query, knowledge_chunks.c.modified_by_user_query),
        )
        
        res = cursor.execute(stmt)
        if res.rowcount != 1:
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
        cursor: Connection,
        *,
        user_id: str,
        filename: str,
        file_type: str,
        content_hash: str,
        metadata: dict[str, Any] | None = None,
        processing_status: str = "pending",
    ) -> str:
        source_id = new_id()
        timestamp = now_iso()
        cursor.execute(
            insert(knowledge_sources).values(
                source_id=source_id,
                user_id=user_id,
                filename=filename,
                file_type=file_type,
                upload_time=timestamp,
                processing_status=processing_status,
                content_hash=content_hash,
                metadata_json=json.dumps(metadata or {}),
                version=1,
                is_deleted=0,
                created_at=timestamp,
                updated_at=timestamp,
            )
        )
        return source_id

    def update_knowledge_source_status(
        self,
        cursor: Connection,
        *,
        user_id: str,
        source_id: str,
        processing_status: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        values = {
            "processing_status": processing_status,
            "updated_at": now_iso(),
            "version": knowledge_sources.c.version + 1,
        }
        if metadata is not None:
            values["metadata_json"] = json.dumps(metadata)
        result = cursor.execute(
            update(knowledge_sources)
            .where(and_(knowledge_sources.c.user_id == user_id, knowledge_sources.c.source_id == source_id))
            .values(**values)
        )
        if result.rowcount != 1:
            raise KnowledgeConflictError("Knowledge source not found for user")

    def list_knowledge_sources(self, *, user_id: str, include_deleted: bool = False) -> list[dict[str, Any]]:
        stmt = select(knowledge_sources).where(knowledge_sources.c.user_id == user_id)
        if not include_deleted:
            stmt = stmt.where(and_(knowledge_sources.c.is_deleted == 0, knowledge_sources.c.processing_status != "deleted"))
        stmt = stmt.order_by(knowledge_sources.c.created_at.desc())
        with self.engine.connect() as conn:
            return [dict(row._mapping) for row in conn.execute(stmt).fetchall()]

    def get_knowledge_source(self, *, user_id: str, source_id: str, include_deleted: bool = False) -> dict[str, Any]:
        stmt = select(knowledge_sources).where(
            and_(knowledge_sources.c.user_id == user_id, knowledge_sources.c.source_id == source_id)
        )
        if not include_deleted:
            stmt = stmt.where(and_(knowledge_sources.c.is_deleted == 0, knowledge_sources.c.processing_status != "deleted"))
        with self.engine.connect() as conn:
            row = conn.execute(stmt).fetchone()
        if not row:
            raise ValueError("Knowledge source not found for user")
        return dict(row._mapping)

    def soft_delete_knowledge_source(self, *, user_id: str, source_id: str) -> dict[str, Any]:
        outbox_job_ids: list[str] = []
        with self.transaction() as cursor:
            result = cursor.execute(
                update(knowledge_sources)
                .where(and_(knowledge_sources.c.user_id == user_id, knowledge_sources.c.source_id == source_id, knowledge_sources.c.is_deleted == 0))
                .values(is_deleted=1, processing_status="deleted", updated_at=now_iso(), version=knowledge_sources.c.version + 1)
            )
            if result.rowcount != 1:
                raise ValueError("Knowledge source not found for user")
            rows = cursor.execute(
                select(knowledge_chunks.c.chunk_id).where(
                    and_(
                        knowledge_chunks.c.user_id == user_id,
                        knowledge_chunks.c.source_id == source_id,
                        knowledge_chunks.c.is_deleted == 0,
                    )
                )
            ).fetchall()
            for row in rows:
                outbox_job_ids.append(
                    self.soft_delete_knowledge_chunk(
                        cursor,
                        user_id=user_id,
                        chunk_id=row[0],
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
                select(knowledge_sources.c.source_id).where(
                    and_(
                        knowledge_sources.c.user_id == user_id,
                        knowledge_sources.c.source_id == source_id,
                        knowledge_sources.c.is_deleted == 0,
                        knowledge_sources.c.processing_status != "deleted",
                    )
                )
            ).fetchone()
            if not source:
                raise ValueError("Knowledge source not found for user")
            rows = cursor.execute(
                select(knowledge_chunks.c.chunk_id).where(
                    and_(knowledge_chunks.c.user_id == user_id, knowledge_chunks.c.source_id == source_id, knowledge_chunks.c.is_deleted == 0)
                )
            ).fetchall()
            for row in rows:
                outbox_job_ids.append(
                    self.insert_outbox_job(
                        cursor,
                        entity_type=OutboxEntityType.KNOWLEDGE_CHUNK,
                        entity_id=row[0],
                        operation=OutboxOperation.UPSERT,
                    )
                )
        return outbox_job_ids

    def list_knowledge_facts(
        self, *, user_id: str, include_deleted: bool = False, source_id: str | None = None
    ) -> list[dict[str, Any]]:
        stmt = select(knowledge_chunks).where(knowledge_chunks.c.user_id == user_id)
        if not include_deleted:
            stmt = stmt.where(knowledge_chunks.c.is_deleted == 0)
        if source_id is not None:
            stmt = stmt.where(knowledge_chunks.c.source_id == source_id)
        stmt = stmt.order_by(knowledge_chunks.c.created_at.desc())
        with self.engine.connect() as conn:
            return [dict(row._mapping) for row in conn.execute(stmt).fetchall()]

    def restore_knowledge_chunk(self, *, user_id: str, chunk_id: str) -> str:
        with self.transaction() as cursor:
            row = cursor.execute(
                select(
                    knowledge_chunks.c.chunk_id,
                    knowledge_chunks.c.source_id,
                    knowledge_sources.c.is_deleted,
                )
                .select_from(knowledge_chunks.outerjoin(knowledge_sources, and_(knowledge_sources.c.source_id == knowledge_chunks.c.source_id, knowledge_sources.c.user_id == knowledge_chunks.c.user_id)))
                .where(and_(knowledge_chunks.c.user_id == user_id, knowledge_chunks.c.chunk_id == chunk_id, knowledge_chunks.c.is_deleted == 1))
            ).fetchone()
            if not row:
                raise ValueError("Deleted knowledge chunk not found for user")
            if row[1] and row[2] == 1:
                raise ValueError("Cannot restore a chunk from a deleted source")
            cursor.execute(
                update(knowledge_chunks)
                .where(and_(knowledge_chunks.c.user_id == user_id, knowledge_chunks.c.chunk_id == chunk_id))
                .values(is_deleted=0, updated_at=now_iso(), version=knowledge_chunks.c.version + 1)
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
            topic_row = cursor.execute(
                select(knowledge_topics.c.title).where(
                    and_(knowledge_topics.c.user_id == user_id, knowledge_topics.c.knowledge_topic_id == old["knowledge_topic_id"])
                )
            ).fetchone()
            _, new_chunk_id, upsert_job_id = self.add_knowledge_chunk(
                cursor,
                user_id=user_id,
                title=topic_row[0] if topic_row else "Knowledge",
                text=text,
                source_id=old.get("source_id"),
                metadata=json.loads(old.get("metadata_json") or "{}"),
                replaces_chunk_id=chunk_id,
                change_reason=change_reason,
                modified_by_user_query=modified_by_user_query,
            )
            cursor.execute(
                update(knowledge_chunks)
                .where(and_(knowledge_chunks.c.user_id == user_id, knowledge_chunks.c.chunk_id == chunk_id))
                .values(replaced_by_chunk_id=new_chunk_id)
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
                insert(generated_artifacts).values(
                    artifact_id=artifact_id,
                    user_id=user_id,
                    conversation_hop_id=conversation_hop_id,
                    file_type=file_type,
                    filename=filename,
                    storage_path=storage_path,
                    storage_url=storage_url,
                    metadata_json=json.dumps(metadata or {}),
                    created_at=timestamp,
                    expires_at=expires_at,
                    status=ArtifactStatus.CREATED.value,
                )
            )
        return self.get_generated_artifact(user_id=user_id, artifact_id=artifact_id)

    def list_generated_artifacts(self, *, user_id: str, include_deleted: bool = False) -> list[dict[str, Any]]:
        stmt = select(generated_artifacts).where(generated_artifacts.c.user_id == user_id)
        if not include_deleted:
            stmt = stmt.where(generated_artifacts.c.status == ArtifactStatus.CREATED.value)
        stmt = stmt.order_by(generated_artifacts.c.created_at.desc())
        with self.engine.connect() as conn:
            return [dict(row._mapping) for row in conn.execute(stmt).fetchall()]

    def get_generated_artifact(self, *, user_id: str, artifact_id: str, include_deleted: bool = False) -> dict[str, Any]:
        stmt = select(generated_artifacts).where(
            and_(generated_artifacts.c.user_id == user_id, generated_artifacts.c.artifact_id == artifact_id)
        )
        if not include_deleted:
            stmt = stmt.where(generated_artifacts.c.status == ArtifactStatus.CREATED.value)
        with self.engine.connect() as conn:
            row = conn.execute(stmt).fetchone()
        if not row:
            raise ValueError("Artifact not found for user")
        payload = dict(row._mapping)
        if not include_deleted and not is_artifact_downloadable(payload):
            raise ValueError("Artifact not found for user")
        return payload

    def delete_generated_artifact(self, *, user_id: str, artifact_id: str) -> dict[str, Any]:
        with self.transaction() as cursor:
            result = cursor.execute(
                update(generated_artifacts)
                .where(and_(generated_artifacts.c.user_id == user_id, generated_artifacts.c.artifact_id == artifact_id, generated_artifacts.c.status == ArtifactStatus.CREATED.value))
                .values(status=ArtifactStatus.DELETED.value)
            )
            if result.rowcount != 1:
                raise ValueError("Artifact not found for user")
        return self.get_generated_artifact(user_id=user_id, artifact_id=artifact_id, include_deleted=True)

    def record_platform_delivery(self, *, user_id: str, conversation_hop_id: str | None, channel: str, status: str, recipient: str, message: dict[str, Any], error_message: str | None = None) -> dict[str, Any]:
        delivery_id = new_id()
        with self.transaction() as cursor:
            cursor.execute(insert(platform_deliveries).values(
                delivery_id=delivery_id, user_id=user_id, conversation_hop_id=conversation_hop_id,
                channel=channel, status=status, recipient=recipient, message_json=json.dumps(message),
                error_message=error_message, created_at=now_iso(),
            ))
        return {"delivery_id": delivery_id, "channel": channel, "status": status}

    def add_reminder(
        self,
        cursor: Connection,
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
        reminder_id = new_id()
        timestamp = now_iso()
        cursor.execute(
            insert(reminders).values(
                reminder_id=reminder_id,
                user_id=user_id,
                source_topic_id=source_topic_id,
                source_hop_id=source_hop_id,
                reminder_time=reminder_time,
                event_time=event_time,
                raw_reminder=raw_reminder,
                reminder_summary=reminder_summary,
                subject=subject,
                supporting_question=supporting_question,
                supporting_response=supporting_response,
                user_timezone=user_timezone,
                original_time_text=original_time_text,
                recurrence_rule=recurrence_rule,
                recurrence_timezone=recurrence_timezone,
                next_fire_time=next_fire_time,
                last_fire_time=None,
                parent_recurring_reminder_id=parent_recurring_reminder_id,
                status='scheduled',
                created_at=timestamp,
                updated_at=timestamp,
                version=1
            )
        )
        return reminder_id

    def find_active_reminder_duplicates(
        self,
        *,
        user_id: str,
        subject: str,
        reminder_time: datetime,
        statuses: tuple[str, ...] = ("scheduled", "notified"),
        limit: int = 20,
    ) -> dict[str, Any]:
        rows = self.list_reminder_candidates(
            user_id=user_id,
            statuses=statuses,
            time_window=None,
            limit=limit,
        )
        target_subject = normalize_subject(subject)
        target_minute = utc_minute(reminder_time)
        similar: list[dict[str, Any]] = []
        for row in rows:
            if not row.reminder_time:
                continue
            row_subject = normalize_subject(row.subject or row.reminder_summary or row.raw_reminder)
            if row_subject == target_subject and utc_minute(row.reminder_time) == target_minute:
                return {"type": "exact", "candidate": row.__dict__}
            if within_minutes(row.reminder_time, reminder_time, 30):
                similarity = token_similarity(target_subject, row_subject)
                if similarity >= 0.65:
                    payload = row.__dict__.copy()
                    payload["similarity"] = similarity
                    similar.append(payload)
        if similar:
            similar.sort(key=lambda item: item["similarity"], reverse=True)
            return {"type": "similar", "candidate": similar[0], "candidates": similar}
        return {"type": "none"}

    def update_reminder_status(
        self,
        cursor: Connection,
        *,
        user_id: str,
        reminder_id: str,
        status: str,
        expected_version: int | None = None,
        expected_status: str | None = None,
    ) -> None:
        stmt = update(reminders).where(
            and_(
                reminders.c.user_id == user_id,
                reminders.c.reminder_id == reminder_id
            )
        )
        if expected_version is not None:
            stmt = stmt.where(reminders.c.version == expected_version)
        if expected_status is not None:
            stmt = stmt.where(reminders.c.status == expected_status)
            
        stmt = stmt.values(
            status=status,
            updated_at=now_iso(),
            version=reminders.c.version + 1
        )
        
        res = cursor.execute(stmt)
        if res.rowcount != 1:
            raise ReminderConflictError("Reminder update failed ownership, version, or status validation")

    def create_notification_if_absent(
        self, cursor: Connection, *, user_id: str, reminder_id: str, fire_time: str | None = None
    ) -> str:
        stmt = select(reminder_notifications.c.notification_id).where(
            and_(
                reminder_notifications.c.user_id == user_id,
                reminder_notifications.c.reminder_id == reminder_id,
                func.coalesce(reminder_notifications.c.fire_time, "") == (fire_time or ""),
            )
        )
        row = cursor.execute(stmt).fetchone()
        if row:
            return str(row[0])
            
        notification_id = new_id()
        cursor.execute(
            insert(reminder_notifications).values(
                notification_id=notification_id,
                reminder_id=reminder_id,
                user_id=user_id,
                ui_status='unread',
                delivery_status=NotificationDeliveryStatus.PENDING.value,
                delivery_attempts=0,
                created_at=now_iso(),
                fire_time=fire_time,
            )
        )
        return notification_id

    def insert_outbox_job(
        self,
        cursor: Connection,
        *,
        entity_type: OutboxEntityType,
        entity_id: str,
        operation: OutboxOperation,
    ) -> str:
        if entity_type.value in ('reminder', 'reminder_notification'):
            raise RepositoryValidationError(f"{entity_type.value} rows must not be indexed.")
            
        stmt = select(indexing_outbox.c.job_id).where(
            and_(
                indexing_outbox.c.entity_type == entity_type.value,
                indexing_outbox.c.entity_id == entity_id,
                indexing_outbox.c.operation == operation.value,
                indexing_outbox.c.status.in_(['pending', 'processing'])
            )
        ).limit(1)
        row = cursor.execute(stmt).fetchone()
        if row:
            return str(row[0])
            
        job_id = new_id()
        timestamp = now_iso()
        cursor.execute(
            insert(indexing_outbox).values(
                job_id=job_id,
                entity_type=entity_type.value,
                entity_id=entity_id,
                operation=operation.value,
                status='pending',
                retry_count=0,
                created_at=timestamp,
                updated_at=timestamp
            )
        )
        return job_id

    def record_action_audit_noop(
        self, *, user_id: str, topic_title: str, raw_user_query: str, rewritten_user_query: str, response_text: str, intent: str, response_type: str, parent_hop_id: str | None = None
    ) -> RepositoryTransactionResult:
        with self.transaction() as cursor:
            topic_id = self.ensure_topic(cursor, user_id=user_id, title=topic_title)
            hop_write = self.append_conversation_hop(
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
            return RepositoryTransactionResult(
                committed=True,
                results=(),
                audit_hop_id=hop_write.hop_id,
                indexing_outbox_ids=(hop_write.outbox_job_id,),
            )

    def _apply_knowledge_action(
        self, cursor: Connection, *, user_id: str, action: ValidatedKnowledgeAction
    ) -> RepositoryActionResult:
        action_type = action.action
        if action_type is KnowledgeAction.ADD:
            _, chunk_id, outbox_job_id = self.add_knowledge_chunk(
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
                indexing_outbox_ids=(outbox_job_id,) if outbox_job_id else (),
                user_safe_summary="Added new knowledge." if outbox_job_id else "Knowledge was already known.",
            )
        if action_type in {KnowledgeAction.DELETE, KnowledgeAction.MODIFY}:
            chunk_id = action.target_chunk_ids[0]
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
                )
                cursor.execute(
                    update(knowledge_chunks)
                    .where(and_(knowledge_chunks.c.user_id == user_id, knowledge_chunks.c.chunk_id == chunk_id))
                    .values(replaced_by_chunk_id=new_chunk_id)
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
                    parent_hop_id=parent_hop_id,
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
                results=(),
                audit_hop_id=None,
                indexing_outbox_ids=(),
                error_type="optimistic_concurrency_error",
                reason_summary=safe_repository_reason(exc),
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
                hop_write = self.append_conversation_hop(
                    cursor,
                    topic_id=topic_id,
                    user_id=user_id,
                    intent=Intent.REMINDER.value,
                    raw_user_query=raw_user_query,
                    rewritten_user_query=rewritten_user_query,
                    raw_response=response_text,
                    response_type=ResponseType.REMINDER_ACTION.value,
                    entities={"reminders": reminder_entities},
                    hop_id=audit_hop_id,
                    parent_hop_id=parent_hop_id,
                )
                for action in actions:
                    result = self._apply_reminder_action(
                        cursor,
                        user_id=user_id,
                        source_topic_id=topic_id,
                        action=action,
                        audit_hop_id=audit_hop_id,
                    )
                    results.append(result)
                    if result.domain_entity_id:
                        reminder_entities.append(
                            {
                                "entity_type": "reminder",
                                "reminder_id": result.domain_entity_id,
                                "subject": str(action.subject or action.replacement_subject or action.reminder_summary or ""),
                            }
                        )

                cursor.execute(
                    update(conversation_hops)
                    .where(
                        and_(
                            conversation_hops.c.hop_id == hop_write.hop_id,
                            conversation_hops.c.user_id == user_id,
                        )
                    )
                    .values(entities_json=json.dumps({"reminders": reminder_entities}))
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
                            audit_hop_id=hop_write.hop_id,
                            indexing_outbox_ids=r.indexing_outbox_ids,
                            user_safe_summary=r.user_safe_summary,
                            reason_summary=r.reason_summary,
                        )
                    )
                    
                return RepositoryTransactionResult(
                    committed=True,
                    results=tuple(updated_results),
                    audit_hop_id=hop_write.hop_id,
                    indexing_outbox_ids=(hop_write.outbox_job_id,),
                )
        except RepositoryConflictError as exc:
            from .errors import safe_repository_reason
            return RepositoryTransactionResult(
                committed=False,
                results=(),
                audit_hop_id=None,
                indexing_outbox_ids=(),
                error_type=type(exc).__name__,
                reason_summary=safe_repository_reason(exc),
            )
        except Exception:
            return RepositoryTransactionResult(
                committed=False,
                results=(),
                audit_hop_id=None,
                indexing_outbox_ids=(),
                error_type="RepositoryTransactionError",
                reason_summary="The database transaction failed and was rolled back.",
            )

    def _apply_reminder_action(
        self,
        cursor: Connection,
        *,
        user_id: str,
        source_topic_id: str,
        action: ValidatedReminderAction,
        audit_hop_id: str,
    ) -> RepositoryActionResult:
        action_type = action.action
        if action_type is ReminderAction.ADD:
            if not action.reminder_time:
                raise RepositoryValidationError("Reminder time is required")
            duplicate = self.find_active_reminder_duplicates(
                user_id=user_id,
                subject=action.subject or action.reminder_summary or action.raw_reminder or "",
                reminder_time=action.reminder_time,
                limit=20,
            )
            if duplicate.get("type") == "exact":
                return RepositoryActionResult(
                    action_id=new_id(),
                    action_type=action_type.value,
                    status="skipped",
                    domain_entity_type="reminder",
                    user_safe_summary="That reminder already exists.",
                    reason_summary="Exact duplicate reminder detected inside transaction.",
                )
            reminder_id = self.add_reminder(
                cursor,
                user_id=user_id,
                source_topic_id=source_topic_id,
                source_hop_id=audit_hop_id,
                reminder_time=action.reminder_time.isoformat(),
                event_time=(action.event_time or action.reminder_time).isoformat(),
                raw_reminder=action.raw_reminder or "",
                reminder_summary=action.reminder_summary or "",
                subject=action.subject or "",
                user_timezone=action.user_timezone or "UTC",
                original_time_text=action.original_time_text,
                recurrence_rule=action.recurrence_rule,
                recurrence_timezone=action.recurrence_timezone or action.user_timezone,
                next_fire_time=action.next_fire_time.isoformat() if action.next_fire_time else (action.reminder_time.isoformat() if action.recurrence_rule and action.reminder_time else None),
                parent_recurring_reminder_id=action.parent_recurring_reminder_id,
            )
            return RepositoryActionResult(
                action_id=new_id(),
                action_type=action_type.value,
                status="committed",
                domain_entity_type="reminder",
                domain_entity_id=reminder_id,
                indexing_outbox_ids=(),
                user_safe_summary=(
                    f"Added reminder '{action.subject}'. Timing will be finalized by autoscan before it can notify you."
                ),
            )

        if not action.target_reminder_ids:
            raise RepositoryValidationError("Reminder target id is required")

        reminder_id = action.target_reminder_ids[0]
        if action_type is ReminderAction.MODIFY:
            self.update_reminder_status(
                cursor,
                user_id=user_id,
                reminder_id=reminder_id,
                status=ReminderStatus.CANCELLED.value,
                expected_version=action.observed_version,
                expected_status=action.observed_status,
            )
            replacement_time = action.replacement_time or action.reminder_time or action.observed_reminder_time
            if not replacement_time:
                raise RepositoryValidationError("Replacement reminder time is required")
            new_reminder_id = self.add_reminder(
                cursor,
                user_id=user_id,
                source_topic_id=source_topic_id,
                source_hop_id=audit_hop_id,
                reminder_time=replacement_time.isoformat(),
                event_time=(action.event_time or replacement_time).isoformat(),
                raw_reminder=action.raw_reminder or "",
                reminder_summary=action.replacement_summary or action.reminder_summary or "",
                subject=action.replacement_subject or action.subject or "",
                user_timezone=action.user_timezone or "UTC",
                original_time_text=action.original_time_text,
                recurrence_rule=action.replacement_recurrence_rule or action.recurrence_rule,
                recurrence_timezone=action.replacement_recurrence_timezone or action.recurrence_timezone or action.user_timezone,
                next_fire_time=action.next_fire_time.isoformat() if action.next_fire_time else (replacement_time.isoformat() if (action.replacement_recurrence_rule or action.recurrence_rule) else None),
                parent_recurring_reminder_id=reminder_id,
            )
            return RepositoryActionResult(
                action_id=new_id(),
                action_type=action_type.value,
                status="committed",
                domain_entity_type="reminder",
                domain_entity_id=new_reminder_id,
                indexing_outbox_ids=(),
                user_safe_summary=(
                    f"Modified reminder '{action.replacement_subject or action.subject}'. Timing will be recalculated by autoscan."
                ),
            )

        status_by_action = {
            ReminderAction.TURN_ON: ReminderStatus.SCHEDULED.value,
            ReminderAction.TURN_OFF: ReminderStatus.CANCELLED.value,
            ReminderAction.DELETE: ReminderStatus.DISMISSED.value,
        }
        if action_type not in status_by_action:
            raise ValueError("Unsupported reminder action")
        new_status = status_by_action[action_type]
        if action_type is ReminderAction.TURN_ON:
            result = cursor.execute(
                update(reminders)
                .where(and_(
                    reminders.c.user_id == user_id,
                    reminders.c.reminder_id == reminder_id,
                    reminders.c.version == action.observed_version,
                    reminders.c.status == action.observed_status,
                ))
                .values(
                    status=ReminderStatus.SCHEDULED.value,
                    reminder_time=func.coalesce(reminders.c.event_time, reminders.c.reminder_time),
                    next_fire_time=case(
                        (reminders.c.recurrence_rule.is_not(None), func.coalesce(reminders.c.event_time, reminders.c.reminder_time)),
                        else_=None,
                    ),
                    timing_plan_status="pending", timing_planned_at=None,
                    timing_plan_reason=None, updated_at=now_iso(),
                    version=reminders.c.version + 1,
                )
            )
            if result.rowcount != 1:
                raise ReminderConflictError("Reminder update failed ownership, version, or status validation")
        else:
            self.update_reminder_status(
                cursor,
                user_id=user_id,
                reminder_id=reminder_id,
                status=new_status,
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
                else f"Reminder '{action.subject or reminder_id}' is now {new_status}."
            ),
        )

    def list_reminder_candidates(
        self, user_id: str, statuses: tuple[str, ...], time_window: tuple[datetime, datetime] | None, limit: int
    ) -> list[ReminderCandidateSummary]:
        stmt = select(
            reminders.c.reminder_id,
            reminders.c.subject,
            reminders.c.reminder_summary,
            reminders.c.raw_reminder,
            reminders.c.reminder_time,
            reminders.c.status,
            reminders.c.created_at,
            reminders.c.updated_at,
            reminders.c.version
        ).where(
            and_(
                reminders.c.user_id == user_id,
                reminders.c.status.in_(statuses)
            )
        )
        if time_window:
            stmt = stmt.where(
                and_(
                    reminders.c.reminder_time >= time_window[0].isoformat(),
                    reminders.c.reminder_time <= time_window[1].isoformat()
                )
            )
        stmt = stmt.order_by(reminders.c.reminder_time).limit(limit)
        
        with self.engine.connect() as conn:
            rows = conn.execute(stmt).fetchall()
            return [
                ReminderCandidateSummary(
                    reminder_id=row[0],
                    subject=row[1],
                    reminder_summary=row[2],
                    raw_reminder=row[3],
                    reminder_time=datetime.fromisoformat(row[4]) if row[4] else None,
                    status=row[5],
                    created_at=datetime.fromisoformat(row[6]) if row[6] else datetime.now(timezone.utc),
                    updated_at=datetime.fromisoformat(row[7]) if row[7] else None,
                    version=row[8],
                    is_deleted=False,
                ) for row in rows
            ]

    def list_notifications(
        self, *, user_id: str, include_deleted: bool = False
    ) -> list[dict[str, Any]]:
        stmt = select(
            reminder_notifications.c.notification_id,
            reminder_notifications.c.ui_status,
            reminder_notifications.c.delivery_status,
            reminder_notifications.c.delivery_attempts,
            reminder_notifications.c.last_delivery_error,
            reminder_notifications.c.sent_at,
            reminder_notifications.c.fire_time,
            reminder_notifications.c.created_at,
            reminder_notifications.c.read_at,
            reminder_notifications.c.deleted_at,
            reminders.c.reminder_id,
            reminders.c.subject,
            reminders.c.reminder_time,
            reminders.c.status,
            reminders.c.reminder_summary,
        ).select_from(
            reminder_notifications.join(
                reminders, reminder_notifications.c.reminder_id == reminders.c.reminder_id
            )
        ).where(reminder_notifications.c.user_id == user_id)
        
        if not include_deleted:
            stmt = stmt.where(reminder_notifications.c.ui_status != 'deleted')
            
        stmt = stmt.order_by(
            func.coalesce(
                reminder_notifications.c.fire_time,
                reminders.c.reminder_time,
                reminder_notifications.c.created_at,
            ).desc(),
            reminder_notifications.c.created_at.desc(),
        )
        
        with self.engine.connect() as conn:
            rows = conn.execute(stmt).fetchall()
            return [
                {
                    "notification_id": row[0],
                    "ui_status": row[1],
                    "delivery_status": row[2],
                    "delivery_attempts": row[3],
                    "last_delivery_error": row[4],
                    "sent_at": row[5],
                    "fire_time": row[6],
                    "created_at": row[7],
                    "read_at": row[8],
                    "deleted_at": row[9],
                    "reminder_id": row[10],
                    "subject": row[11],
                    "reminder_time": row[12],
                    "reminder_status": row[13],
                    "reminder_summary": row[14],
                } for row in rows
            ]

    def update_notification_ui_status(
        self, *, user_id: str, notification_id: str, ui_status: str
    ) -> dict[str, Any]:
        with self.transaction() as cursor:
            stmt = update(reminder_notifications).where(
                and_(
                    reminder_notifications.c.user_id == user_id,
                    reminder_notifications.c.notification_id == notification_id
                )
            ).values(
                ui_status=ui_status,
                read_at=now_iso() if ui_status == 'read' else reminder_notifications.c.read_at,
                deleted_at=now_iso() if ui_status == 'deleted' else reminder_notifications.c.deleted_at
            ).returning(
                reminder_notifications.c.notification_id,
                reminder_notifications.c.reminder_id,
                reminder_notifications.c.user_id,
                reminder_notifications.c.ui_status,
                reminder_notifications.c.delivery_status,
                reminder_notifications.c.delivery_attempts,
                reminder_notifications.c.last_delivery_error,
                reminder_notifications.c.sent_at,
                reminder_notifications.c.fire_time,
                reminder_notifications.c.created_at,
                reminder_notifications.c.read_at,
                reminder_notifications.c.deleted_at,
            )
            
            row = cursor.execute(stmt).fetchone()
            if not row:
                raise ValueError("Notification not found")
            return {
                "notification_id": row[0],
                "reminder_id": row[1],
                "user_id": row[2],
                "ui_status": row[3],
                "delivery_status": row[4],
                "delivery_attempts": row[5],
                "last_delivery_error": row[6],
                "sent_at": row[7],
                "created_at": row[8],
                "read_at": row[9],
                "deleted_at": row[10],
            }

    def mark_notification_delivery_sent(
        self, *, user_id: str, notification_id: str
    ) -> dict[str, Any]:
        with self.transaction() as cursor:
            stmt = update(reminder_notifications).where(
                and_(
                    reminder_notifications.c.user_id == user_id,
                    reminder_notifications.c.notification_id == notification_id,
                )
            ).values(
                delivery_status=NotificationDeliveryStatus.SENT.value,
                sent_at=now_iso(),
                last_delivery_error=None,
            ).returning(reminder_notifications)
            row = cursor.execute(stmt).fetchone()
            if not row:
                raise ValueError("Notification not found")
            return dict(row._mapping)

    def mark_notification_delivery_failed(
        self, *, user_id: str, notification_id: str, error_message: str, retrying: bool = False
    ) -> dict[str, Any]:
        status = NotificationDeliveryStatus.RETRYING.value if retrying else NotificationDeliveryStatus.FAILED.value
        with self.transaction() as cursor:
            stmt = update(reminder_notifications).where(
                and_(
                    reminder_notifications.c.user_id == user_id,
                    reminder_notifications.c.notification_id == notification_id,
                )
            ).values(
                delivery_status=status,
                delivery_attempts=reminder_notifications.c.delivery_attempts + 1,
                last_delivery_error=error_message,
            ).returning(reminder_notifications)
            row = cursor.execute(stmt).fetchone()
            if not row:
                raise ValueError("Notification not found")
            return dict(row._mapping)

    def claim_idempotency_key(
        self, *, user_id: str, idempotency_key: str, payload_hash: str
    ) -> IdempotencyClaimResult:
        with self.transaction() as cursor:
            row = cursor.execute(
                select(
                    mutation_requests.c.request_id,
                    mutation_requests.c.payload_hash,
                    mutation_requests.c.status,
                    mutation_requests.c.stored_response_json,
                ).where(
                    and_(
                        mutation_requests.c.user_id == user_id,
                        mutation_requests.c.idempotency_key == idempotency_key,
                    )
                )
            ).fetchone()
            if row:
                if row[1] != payload_hash:
                    return IdempotencyClaimResult(status="conflict", request_id=row[0], reason="idempotency_conflict")
                if row[2] == MutationRequestStatus.COMPLETED.value:
                    return IdempotencyClaimResult(status="replay", request_id=row[0], stored_response_json=row[3])
                if row[2] == MutationRequestStatus.IN_PROGRESS.value:
                    return IdempotencyClaimResult(status="in_progress", request_id=row[0], reason="request_in_progress")
                return IdempotencyClaimResult(status="failed_retry", request_id=row[0])
            request_id = new_id()
            timestamp = now_iso()
            cursor.execute(
                insert(mutation_requests).values(
                    request_id=request_id,
                    user_id=user_id,
                    idempotency_key=idempotency_key,
                    payload_hash=payload_hash,
                    status=MutationRequestStatus.IN_PROGRESS.value,
                    created_at=timestamp,
                    updated_at=timestamp,
                )
            )
            return IdempotencyClaimResult(status="started", request_id=request_id)

    def complete_idempotency_request(self, *, request_id: str, stored_response_json: str) -> None:
        with self.transaction() as cursor:
            cursor.execute(
                update(mutation_requests).where(mutation_requests.c.request_id == request_id).values(
                    status=MutationRequestStatus.COMPLETED.value,
                    stored_response_json=stored_response_json,
                    updated_at=now_iso(),
                )
            )

    def fail_idempotency_request(self, *, request_id: str, error_message: str | None = None) -> None:
        with self.transaction() as cursor:
            cursor.execute(
                update(mutation_requests).where(mutation_requests.c.request_id == request_id).values(
                    status=MutationRequestStatus.FAILED.value,
                    updated_at=now_iso(),
                )
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
                insert(pending_action_confirmations).values(
                    confirmation_token=token,
                    user_id=user_id,
                    action_type=action_type,
                    target_entity_type=target_entity_type,
                    target_entity_id=target_entity_id,
                    proposed_action_json=json.dumps(proposed_action, default=str),
                    target_snapshot_json=json.dumps(target_snapshot, default=str),
                    expires_at=expires_at,
                    status=ConfirmationStatus.PENDING.value,
                    created_at=timestamp,
                )
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
        with self.engine.connect() as conn:
            row = conn.execute(
                select(pending_action_confirmations).where(
                    and_(
                        pending_action_confirmations.c.user_id == user_id,
                        pending_action_confirmations.c.confirmation_token == confirmation_token,
                    )
                )
            ).fetchone()
        if not row:
            raise ValueError("Confirmation not found for user")
        payload = dict(row._mapping)
        if payload["status"] != ConfirmationStatus.PENDING.value:
            raise ValueError("Confirmation is not pending")
        if payload["expires_at"] <= now_value:
            with self.transaction() as cursor:
                cursor.execute(
                    update(pending_action_confirmations)
                    .where(pending_action_confirmations.c.confirmation_token == confirmation_token)
                    .values(status=ConfirmationStatus.EXPIRED.value)
                )
            raise ValueError("Confirmation has expired")
        payload["proposed_action"] = json.loads(payload["proposed_action_json"])
        payload["target_snapshot"] = json.loads(payload["target_snapshot_json"])
        return payload

    def mark_confirmation_confirmed(self, *, user_id: str, confirmation_token: str) -> None:
        with self.transaction() as cursor:
            result = cursor.execute(
                update(pending_action_confirmations).where(
                    and_(
                        pending_action_confirmations.c.user_id == user_id,
                        pending_action_confirmations.c.confirmation_token == confirmation_token,
                        pending_action_confirmations.c.status == ConfirmationStatus.PENDING.value,
                    )
                ).values(
                    status=ConfirmationStatus.CONFIRMED.value,
                    confirmed_at=now_iso(),
                )
            )
            if result.rowcount != 1:
                raise ValueError("Confirmation could not be confirmed")

    def list_reminders(
        self, *, user_id: str, status: str | None = None, from_time: str | None = None, to_time: str | None = None
    ) -> list[dict[str, Any]]:
        stmt = select(reminders).where(reminders.c.user_id == user_id)
        if status:
            stmt = stmt.where(reminders.c.status == status)
        if from_time:
            stmt = stmt.where(reminders.c.reminder_time >= from_time)
        if to_time:
            stmt = stmt.where(reminders.c.reminder_time <= to_time)
            
        stmt = stmt.order_by(reminders.c.reminder_time)
        
        with self.engine.connect() as conn:
            rows = conn.execute(stmt).fetchall()
            return [dict(r._mapping) for r in rows]

    def append_reminder_reply(
        self, *, user_id: str, reminder_id: str, notification_id: str, reply_text: str, response_text: str
    ) -> HopWrite:
        with self.transaction() as cursor:
            ctx = self.load_reminder_reply_context(
                user_id=user_id, reminder_id=reminder_id, notification_id=notification_id
            )
            topic_id = ctx["topic_id"]
            if not topic_id:
                topic_id = self.ensure_topic(
                    cursor, user_id=user_id, title=f"Reminder: {ctx['subject']}"
                )
            return self.append_conversation_hop(
                cursor,
                topic_id=topic_id,
                user_id=user_id,
                intent=Intent.REMINDER.value,
                raw_user_query=reply_text,
                rewritten_user_query=reply_text,
                raw_response=response_text,
                response_type=ResponseType.REMINDER_REPLY.value,
                parent_hop_id=ctx["source_hop_id"]
            )

    def claim_outbox_jobs(self, *, max_attempts: int, batch_size: int, retry_cutoff: str) -> list[dict[str, Any]]:
        with self.transaction() as cursor:
            stmt = update(indexing_outbox).where(
                indexing_outbox.c.job_id.in_(
                    select(indexing_outbox.c.job_id).where(
                        and_(
                            indexing_outbox.c.retry_count < max_attempts,
                            or_(
                                indexing_outbox.c.status == 'pending',
                                and_(
                                    indexing_outbox.c.status == 'failed',
                                    indexing_outbox.c.updated_at <= retry_cutoff
                                )
                            )
                        )
                    ).order_by(indexing_outbox.c.created_at).limit(batch_size)
                )
            ).values(
                status='processing',
                updated_at=now_iso()
            ).returning(
                indexing_outbox.c.job_id,
                indexing_outbox.c.entity_type,
                indexing_outbox.c.entity_id,
                indexing_outbox.c.operation
            )
            
            rows = cursor.execute(stmt).fetchall()
            return [
                {
                    "job_id": r[0],
                    "entity_type": r[1],
                    "entity_id": r[2],
                    "operation": r[3]
                } for r in rows
            ]

    def release_stale_processing_jobs(self, *, max_attempts: int, timeout_cutoff: str) -> None:
        with self.transaction() as cursor:
            stmt = update(indexing_outbox).where(
                and_(
                    indexing_outbox.c.status == 'processing',
                    indexing_outbox.c.updated_at < timeout_cutoff,
                    indexing_outbox.c.retry_count < max_attempts
                )
            ).values(
                status='failed',
                retry_count=indexing_outbox.c.retry_count + 1,
                error_message='Indexing job timed out while processing',
                updated_at=now_iso()
            )
            cursor.execute(stmt)

    def mark_outbox_job_completed(self, *, job_id: str) -> None:
        with self.transaction() as cursor:
            stmt = update(indexing_outbox).where(
                indexing_outbox.c.job_id == job_id
            ).values(
                status='completed',
                error_message=None,
                updated_at=now_iso()
            )
            cursor.execute(stmt)

    def mark_outbox_job_failed(self, *, job_id: str, error_message: str) -> None:
        with self.transaction() as cursor:
            stmt = update(indexing_outbox).where(
                indexing_outbox.c.job_id == job_id
            ).values(
                status='failed',
                retry_count=indexing_outbox.c.retry_count + 1,
                error_message=error_message,
                updated_at=now_iso()
            )
            cursor.execute(stmt)

    def load_outbox_entity(self, *, entity_type: str, entity_id: str) -> OutboxIndexPayload:
        with self.engine.connect() as conn:
            if entity_type == "conversation_hop":
                stmt = select(
                    conversation_hops.c.user_id,
                    conversation_hops.c.topic_id,
                    conversation_hops.c.hop_id,
                    conversation_hops.c.parent_hop_id,
                    conversation_hops.c.root_hop_id,
                    conversation_hops.c.branch_id,
                    conversation_hops.c.intent,
                    conversation_hops.c.response_type,
                    conversation_hops.c.created_at,
                    conversation_hops.c.raw_user_query,
                    conversation_hops.c.raw_response,
                    conversation_topics.c.status.label("topic_status"),
                ).select_from(
                    conversation_hops.join(
                        conversation_topics,
                        and_(
                            conversation_topics.c.topic_id == conversation_hops.c.topic_id,
                            conversation_topics.c.user_id == conversation_hops.c.user_id,
                        ),
                    )
                ).where(conversation_hops.c.hop_id == entity_id)
                row = conn.execute(stmt).fetchone()
                if not row:
                    raise ValueError("Conversation hop not found")
                if not is_indexable_conversation_hop({"topic_status": row[11]}):
                    raise ValueError("Conversation hop is not indexable")
                text = f"User: {row[9]}\nAssistant: {row[10]}"
                metadata = {
                    "user_id": row[0],
                    "topic_id": row[1],
                    "hop_id": row[2],
                    "parent_hop_id": row[3] or "",
                    "root_hop_id": row[4] or "",
                    "branch_id": row[5] or "",
                    "intent": row[6],
                    "response_type": row[7],
                    "created_at": row[8],
                }
                return OutboxIndexPayload(
                    user_id=row[0],
                    entity_type=entity_type,
                    entity_id=entity_id,
                    text=text,
                    metadata=metadata,
                )
            elif entity_type == "knowledge_chunk":
                stmt = select(
                    knowledge_chunks.c.user_id,
                    knowledge_chunks.c.knowledge_topic_id,
                    knowledge_chunks.c.chunk_id,
                    knowledge_chunks.c.source_id,
                    knowledge_chunks.c.chunk_index,
                    knowledge_chunks.c.normalized_text,
                    knowledge_chunks.c.content_hash,
                    knowledge_chunks.c.is_deleted,
                    knowledge_chunks.c.version,
                    knowledge_chunks.c.created_at,
                    knowledge_chunks.c.updated_at,
                ).where(
                    and_(
                        knowledge_chunks.c.chunk_id == entity_id,
                        knowledge_chunks.c.is_deleted == 0
                    )
                )
                row = conn.execute(stmt).fetchone()
                if not row:
                    raise ValueError("Active knowledge chunk not found")
                if not is_indexable_knowledge_chunk({"is_deleted": row[7]}):
                    raise ValueError("Knowledge chunk is not indexable")
                metadata = {
                    "user_id": row[0],
                    "knowledge_topic_id": row[1],
                    "chunk_id": row[2],
                    "source_id": row[3] or "",
                    "chunk_index": int(row[4]),
                    "content_hash": row[6],
                    "is_deleted": bool(row[7]),
                    "version": int(row[8]),
                    "created_at": row[9],
                    "updated_at": row[10],
                }
                return OutboxIndexPayload(
                    user_id=row[0],
                    entity_type=entity_type,
                    entity_id=entity_id,
                    text=row[5],
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
        with self.engine.connect() as conn:
            rows = conn.execute(
                select(
                    conversation_hops.c.hop_id,
                    conversation_hops.c.user_id,
                    conversation_hops.c.topic_id,
                    conversation_hops.c.parent_hop_id,
                    conversation_hops.c.root_hop_id,
                    conversation_hops.c.branch_id,
                    conversation_hops.c.intent,
                    conversation_hops.c.response_type,
                    conversation_hops.c.supporting_questions_json,
                    conversation_hops.c.raw_user_query,
                    conversation_hops.c.raw_response,
                    conversation_hops.c.created_at,
                ).where(
                    and_(
                        conversation_hops.c.user_id == user_id,
                        conversation_hops.c.hop_id.in_(ids),
                    )
                )
            ).fetchall()
        hop_map = {str(row[0]): row for row in rows}
        hydrated: list[RetrievalResult] = []
        for result in results:
            hop = hop_map.get(result.entity_id)
            if not hop:
                continue
            payload = dict(result.payload)
            payload.update(
                {
                    "hop_id": hop[0],
                    "user_id": hop[1],
                    "topic_id": hop[2],
                    "parent_hop_id": hop[3],
                    "root_hop_id": hop[4],
                    "branch_id": hop[5],
                    "intent": hop[6],
                    "response_type": hop[7],
                    "supporting_questions_json": hop[8],
                    "raw_user_query": hop[9],
                    "raw_response": hop[10],
                    "created_at": hop[11],
                    "text": f"User: {hop[9]}\nAssistant: {hop[10]}",
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

    def list_all_outbox_entities(self) -> list[tuple[str, str]]:
        results = []
        with self.engine.connect() as conn:
            hops = conn.execute(
                select(conversation_hops.c.hop_id)
                .select_from(
                    conversation_hops.join(
                        conversation_topics,
                        and_(
                            conversation_topics.c.topic_id == conversation_hops.c.topic_id,
                            conversation_topics.c.user_id == conversation_hops.c.user_id,
                        ),
                    )
                )
                .where(conversation_topics.c.status == "active")
                .order_by(conversation_hops.c.created_at)
            ).fetchall()
            for r in hops:
                results.append(("conversation_hop", r[0]))
                
            chunks = conn.execute(
                select(knowledge_chunks.c.chunk_id)
                .where(knowledge_chunks.c.is_deleted == 0)
                .order_by(knowledge_chunks.c.created_at)
            ).fetchall()
            for r in chunks:
                results.append(("knowledge_chunk", r[0]))
        return results
