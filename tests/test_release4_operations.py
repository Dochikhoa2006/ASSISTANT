from __future__ import annotations

from datetime import datetime, timedelta, timezone
import sqlite3

import pytest

from assistant_rag.contracts import OutboxEntityType, OutboxOperation
from assistant_rag.database import SQLiteRepository
from assistant_rag.evaluation import EvaluationRunner
from assistant_rag.health import build_health_report
from assistant_rag.index_ops import IndexOperations
from assistant_rag.lifecycle import confirmation_can_execute, is_artifact_downloadable
from assistant_rag.metrics import GLOBAL_METRICS
from assistant_rag.observability import StageTimer, redact_payload, start_trace
from assistant_rag.recurrence import calculate_next_fire_time
from assistant_rag.settings import OperationsSettings, ProductionSettings


def make_repo() -> SQLiteRepository:
    repo = SQLiteRepository.in_memory()
    repo.initialize_schema()
    return repo


class FakeIndex:
    def __init__(self) -> None:
        self.docs: dict[str, dict[str, object]] = {}

    def search(self, *, user_id: str, query: str, limit: int):
        return []

    def upsert(self, *, user_id: str, entity_type: str, entity_id: str, text: str, metadata=None) -> None:
        self.docs[entity_id] = {
            "user_id": user_id,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "text": text,
            **(metadata or {}),
        }

    def delete(self, *, entity_id: str) -> None:
        self.docs.pop(entity_id, None)

    def clear(self) -> None:
        self.docs.clear()

    def documents(self):
        return list(self.docs.values())

    def get_document(self, *, entity_id: str):
        return self.docs.get(entity_id)

    def count_by_entity_type(self, *, user_id=None):
        counts: dict[str, int] = {}
        for doc in self.docs.values():
            if user_id and doc["user_id"] != user_id:
                continue
            entity_type = str(doc["entity_type"])
            counts[entity_type] = counts.get(entity_type, 0) + 1
        return counts


def test_observability_redacts_and_records_stage_latency() -> None:
    trace = start_trace("req-1")
    with StageTimer("rewrite", {"raw_query": "secret question", "authorization": "Bearer token"}):
        pass

    summary = trace.summary()
    assert summary.request_id == "req-1"
    assert summary.stages[0].stage == "rewrite"
    assert summary.stages[0].metadata["raw_query"] == "[REDACTED]"
    assert summary.stages[0].metadata["authorization"] == "[REDACTED]"
    redacted = redact_payload({"storage_path": "/tmp/private/file.pdf", "message": "hello"})
    assert redacted["storage_path"] == "[REDACTED]"
    assert redacted["message"] == "[REDACTED]"


def test_metrics_snapshot_has_no_raw_content() -> None:
    GLOBAL_METRICS.reset()
    GLOBAL_METRICS.increment("chat_requests_total")
    GLOBAL_METRICS.observe_latency("chat_latency_ms", 12.5)
    snapshot = GLOBAL_METRICS.snapshot_dict()
    assert snapshot["counters"]["chat_requests_total"] == 1
    assert "chat_latency_ms" in snapshot["latency_ms"]
    assert "raw_query" not in str(snapshot)


def test_health_sql_and_optional_dependency_statuses() -> None:
    repo = make_repo()
    healthy = build_health_report(repository=repo, settings=ProductionSettings())
    assert healthy["status"] == "ok"
    assert healthy["outbox_failed_count"] == 0

    degraded = build_health_report(
        repository=repo,
        settings=ProductionSettings(),
        checks={"opensearch": lambda: {"ok": False, "detail": "down"}},
    )
    assert degraded["status"] == "degraded"

    strict = build_health_report(
        repository=repo,
        settings=ProductionSettings(operations=OperationsSettings(health_strict_opensearch=True)),
        checks={"opensearch": lambda: {"ok": False, "detail": "down"}},
    )
    assert strict["status"] == "unhealthy"


def test_recurring_reminder_scan_advances_and_dedupes_fire_notifications() -> None:
    repo = make_repo()
    fire = datetime(2026, 7, 10, 9, 0, tzinfo=timezone.utc)
    with repo.transaction() as cursor:
        reminder_id = repo.add_reminder(
            cursor,
            user_id="u1",
            source_topic_id=None,
            source_hop_id=None,
            reminder_time=fire.isoformat(),
            raw_reminder="daily standup",
            reminder_summary="daily standup",
            subject="Daily standup",
            recurrence_rule="daily",
            recurrence_timezone="UTC",
            next_fire_time=fire.isoformat(),
        )

    assert repo.scan_due_reminders(now_value=fire.isoformat()) == [reminder_id]
    row = repo.list_reminders(user_id="u1")[0]
    assert row["status"] == "scheduled"
    assert row["last_fire_time"] == fire.isoformat()
    assert row["next_fire_time"] == (fire + timedelta(days=1)).isoformat()
    assert len(repo.list_notifications(user_id="u1")) == 1

    repo.scan_due_reminders(now_value=fire.isoformat())
    assert len(repo.list_notifications(user_id="u1")) == 1


def test_sqlite_release4_upgrade_removes_legacy_notification_unique_constraint(tmp_path) -> None:
    db_path = tmp_path / "legacy.sqlite3"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE reminders (
            reminder_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            source_topic_id TEXT NULL,
            source_hop_id TEXT NULL,
            reminder_time TEXT NOT NULL,
            status TEXT NOT NULL,
            raw_reminder TEXT NOT NULL,
            reminder_summary TEXT NOT NULL,
            subject TEXT NOT NULL,
            supporting_question TEXT,
            supporting_response TEXT,
            user_timezone TEXT NOT NULL DEFAULT 'UTC',
            original_time_text TEXT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            version INTEGER NOT NULL
        );
        CREATE TABLE reminder_notifications (
            notification_id TEXT PRIMARY KEY,
            reminder_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            ui_status TEXT NOT NULL,
            delivery_status TEXT NOT NULL DEFAULT 'pending',
            delivery_attempts INTEGER NOT NULL DEFAULT 0,
            last_delivery_error TEXT NULL,
            created_at TEXT NOT NULL,
            read_at TEXT NULL,
            deleted_at TEXT NULL,
            sent_at TEXT NULL,
            UNIQUE (reminder_id, user_id)
        );
        """
    )
    conn.commit()
    conn.close()

    repo = SQLiteRepository.persistent(str(db_path), enable_wal=False, busy_timeout_ms=1000)
    repo.initialize_schema()
    with repo.transaction() as cursor:
        reminder_id = repo.add_reminder(
            cursor,
            user_id="u1",
            source_topic_id=None,
            source_hop_id=None,
            reminder_time="2026-07-10T09:00:00+00:00",
            raw_reminder="daily",
            reminder_summary="daily",
            subject="Daily",
        )
        first = repo.create_notification_if_absent(
            cursor,
            user_id="u1",
            reminder_id=reminder_id,
            fire_time="2026-07-10T09:00:00+00:00",
        )
        second = repo.create_notification_if_absent(
            cursor,
            user_id="u1",
            reminder_id=reminder_id,
            fire_time="2026-07-11T09:00:00+00:00",
        )
    assert first != second


def test_recurrence_daily_weekly_monthly_and_timezone() -> None:
    assert calculate_next_fire_time(
        previous_fire_time="2026-07-10T02:00:00+00:00",
        recurrence_rule="daily",
        recurrence_timezone="Asia/Ho_Chi_Minh",
    ) == "2026-07-11T02:00:00+00:00"
    assert calculate_next_fire_time(
        previous_fire_time="2026-07-10T09:00:00+00:00",
        recurrence_rule="weekly",
        recurrence_timezone="UTC",
    ) == "2026-07-17T09:00:00+00:00"
    assert calculate_next_fire_time(
        previous_fire_time="2026-01-31T09:00:00+00:00",
        recurrence_rule="monthly",
        recurrence_timezone="UTC",
    ) == "2026-02-28T09:00:00+00:00"


def test_lifecycle_excludes_archived_conversations_deleted_chunks_and_expired_artifacts() -> None:
    repo = make_repo()
    with repo.transaction() as cursor:
        topic_id = repo.ensure_topic(cursor, user_id="u1", title="Topic")
        hop = repo.append_conversation_hop(
            cursor,
            topic_id=topic_id,
            user_id="u1",
            intent="general_response",
            raw_user_query="hello",
            rewritten_user_query="hello",
            raw_response="hi",
            response_type="normal",
        )
        _, chunk_id, _ = repo.add_knowledge_chunk(cursor, user_id="u1", title="Facts", text="A fact")
        repo.soft_delete_knowledge_chunk(cursor, user_id="u1", chunk_id=chunk_id)
        cursor.execute("UPDATE conversation_topics SET status = 'archived' WHERE topic_id = ?", (topic_id,))

    with pytest.raises(ValueError):
        repo.load_outbox_entity(entity_type="conversation_hop", entity_id=hop.hop_id)
    with pytest.raises(ValueError):
        repo.load_outbox_entity(entity_type="knowledge_chunk", entity_id=chunk_id)

    expired = {"status": "created", "expires_at": "2026-07-01T00:00:00+00:00"}
    assert not is_artifact_downloadable(expired, now_value="2026-07-09T00:00:00+00:00")
    confirmation = {"status": "pending", "expires_at": "2026-07-01T00:00:00+00:00"}
    assert not confirmation_can_execute(confirmation, now_value="2026-07-09T00:00:00+00:00")


def test_index_rebuild_and_drift_detects_missing_deleted_and_reminder_docs() -> None:
    repo = make_repo()
    with repo.transaction() as cursor:
        topic_id = repo.ensure_topic(cursor, user_id="u1", title="Topic")
        repo.append_conversation_hop(
            cursor,
            topic_id=topic_id,
            user_id="u1",
            intent="general_response",
            raw_user_query="hello",
            rewritten_user_query="hello",
            raw_response="hi",
            response_type="normal",
        )
        _, chunk_id, _ = repo.add_knowledge_chunk(cursor, user_id="u1", title="Facts", text="Important fact")
        reminder_id = repo.add_reminder(
            cursor,
            user_id="u1",
            source_topic_id=None,
            source_hop_id=None,
            reminder_time="2026-07-10T09:00:00+00:00",
            raw_reminder="call mom",
            reminder_summary="call mom",
            subject="Call mom",
        )
        repo.insert_outbox_job(cursor, entity_type=OutboxEntityType.KNOWLEDGE_CHUNK, entity_id=chunk_id, operation=OutboxOperation.UPSERT)

    bm25 = FakeIndex()
    chroma = FakeIndex()
    ops = IndexOperations(repository=repo, bm25=bm25, chroma=chroma)
    assert ops.rebuild_all_indexes() == 2
    assert all(doc["entity_type"] != "reminder" for doc in bm25.documents())

    bm25.delete(entity_id=chunk_id)
    chroma.docs[reminder_id] = {"entity_id": reminder_id, "entity_type": "reminder", "user_id": "u1"}
    report = ops.check_index_drift(user_id="u1")
    kinds = {issue["kind"] for issue in report["issues"]}
    assert "missing_doc" in kinds or "count_mismatch" in kinds
    assert "forbidden_doc" in kinds


def test_evaluation_report_and_strict_failure() -> None:
    runner = EvaluationRunner(settings=OperationsSettings(eval_top_1_threshold=1.0))
    report = runner.run_cases(
        [
            {
                "case_id": "mixed_vi_en",
                "query": "",
                "expected_entity_id": "missing",
                "clarification_required": True,
            }
        ],
        strict=True,
    )
    assert report["case_count"] == 1
    assert report["passed"] is False
    assert report["strict_failed"] is True
