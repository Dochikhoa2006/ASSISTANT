from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from assistant_rag.config import OutboxConfig
from assistant_rag.contracts import (
    ActionValidationResult,
    Intent,
    OutboxOperation,
    ReminderAction,
    ResponseType,
    RetrievalResult,
    ValidatedReminderAction,
)
from assistant_rag.database import SQLiteRepository
from assistant_rag.indexing import BackgroundIndexer


def make_repo() -> SQLiteRepository:
    repo = SQLiteRepository.in_memory()
    repo.initialize_schema()
    return repo


def test_sqlite_conversation_tree_metadata_matches_release_contract() -> None:
    repo = make_repo()
    with repo.transaction() as cursor:
        topic_id = repo.ensure_topic(cursor, user_id="u1", title="General")
        root = repo.append_conversation_hop(
            cursor,
            topic_id=topic_id,
            user_id="u1",
            intent=Intent.GENERAL_RESPONSE.value,
            raw_user_query="hello",
            rewritten_user_query="hello",
            raw_response="hi",
            response_type=ResponseType.NORMAL.value,
        )
        child = repo.append_conversation_hop(
            cursor,
            topic_id=topic_id,
            user_id="u1",
            intent=Intent.REMINDER.value,
            raw_user_query="remind me",
            rewritten_user_query="remind me",
            raw_response="done",
            response_type=ResponseType.REMINDER_ACTION.value,
            parent_hop_id=root.hop_id,
        )

    rows = repo.connection.execute(
        """
        SELECT hop_id, previous_hop_id, parent_hop_id, root_hop_id, branch_id, depth_from_root
        FROM conversation_hops
        ORDER BY created_at
        """
    ).fetchall()
    assert rows[0]["root_hop_id"] is None
    assert rows[0]["branch_id"] == root.hop_id
    assert rows[0]["depth_from_root"] == 0
    assert rows[1]["previous_hop_id"] == root.hop_id
    assert rows[1]["parent_hop_id"] == root.hop_id
    assert rows[1]["root_hop_id"] == root.hop_id
    assert rows[1]["branch_id"] == root.hop_id
    assert rows[1]["depth_from_root"] == 1


def test_sqlite_reminder_actions_cover_add_modify_turn_on_turn_off_delete() -> None:
    repo = make_repo()
    reminder_time = datetime(2026, 7, 10, 9, 0, tzinfo=timezone.utc)

    add_result = repo.transactional_reminder_actions(
        user_id="u1",
        topic_title="Reminders",
        raw_user_query="remind me to pay rent",
        rewritten_user_query="remind me to pay rent",
        response_text="Added.",
        actions=[
            ValidatedReminderAction(
                action=ReminderAction.ADD,
                validation_result=ActionValidationResult.EXECUTE,
                subject="Pay rent",
                reminder_time=reminder_time,
                reminder_summary="Pay rent",
            )
        ],
    )
    assert add_result.committed
    original_id = add_result.results[0].domain_entity_id
    original = repo.list_reminders(user_id="u1")[0]
    assert original["status"] == "scheduled"

    modify_result = repo.transactional_reminder_actions(
        user_id="u1",
        topic_title="Reminders",
        raw_user_query="change rent reminder",
        rewritten_user_query="change rent reminder",
        response_text="Modified.",
        actions=[
            ValidatedReminderAction(
                action=ReminderAction.MODIFY,
                validation_result=ActionValidationResult.EXECUTE,
                target_reminder_ids=(str(original_id),),
                observed_status="scheduled",
                observed_version=original["version"],
                subject="Pay rent",
                replacement_subject="Pay rent later",
                replacement_time=datetime(2026, 7, 11, 9, 0, tzinfo=timezone.utc),
            )
        ],
    )
    assert modify_result.committed
    assert repo.list_reminders(user_id="u1", status="cancelled")[0]["reminder_id"] == original_id
    replacement_id = modify_result.results[0].domain_entity_id
    replacement = next(r for r in repo.list_reminders(user_id="u1") if r["reminder_id"] == replacement_id)
    assert replacement["status"] == "scheduled"

    turn_off = repo.transactional_reminder_actions(
        user_id="u1",
        topic_title="Reminders",
        raw_user_query="turn off rent",
        rewritten_user_query="turn off rent",
        response_text="Off.",
        actions=[
            ValidatedReminderAction(
                action=ReminderAction.TURN_OFF,
                validation_result=ActionValidationResult.EXECUTE,
                target_reminder_ids=(str(replacement_id),),
                observed_status="scheduled",
                observed_version=replacement["version"],
                subject="Pay rent later",
            )
        ],
    )
    assert turn_off.committed
    off_row = next(r for r in repo.list_reminders(user_id="u1") if r["reminder_id"] == replacement_id)
    assert off_row["status"] == "cancelled"

    turn_on = repo.transactional_reminder_actions(
        user_id="u1",
        topic_title="Reminders",
        raw_user_query="turn on rent",
        rewritten_user_query="turn on rent",
        response_text="On.",
        actions=[
            ValidatedReminderAction(
                action=ReminderAction.TURN_ON,
                validation_result=ActionValidationResult.EXECUTE,
                target_reminder_ids=(str(replacement_id),),
                observed_status="cancelled",
                observed_version=off_row["version"],
                subject="Pay rent later",
            )
        ],
    )
    assert turn_on.committed
    on_row = next(r for r in repo.list_reminders(user_id="u1") if r["reminder_id"] == replacement_id)
    assert on_row["status"] == "scheduled"

    delete = repo.transactional_reminder_actions(
        user_id="u1",
        topic_title="Reminders",
        raw_user_query="delete rent",
        rewritten_user_query="delete rent",
        response_text="Deleted.",
        actions=[
            ValidatedReminderAction(
                action=ReminderAction.DELETE,
                validation_result=ActionValidationResult.EXECUTE,
                target_reminder_ids=(str(replacement_id),),
                observed_status="scheduled",
                observed_version=on_row["version"],
                subject="Pay rent later",
            )
        ],
    )
    assert delete.committed
    deleted_row = next(r for r in repo.list_reminders(user_id="u1") if r["reminder_id"] == replacement_id)
    assert deleted_row["status"] == "dismissed"


def test_outbox_payload_contains_metadata_for_conversation_and_knowledge() -> None:
    repo = make_repo()
    with repo.transaction() as cursor:
        topic_id = repo.ensure_topic(cursor, user_id="u1", title="General")
        hop = repo.append_conversation_hop(
            cursor,
            topic_id=topic_id,
            user_id="u1",
            intent=Intent.GENERAL_RESPONSE.value,
            raw_user_query="what did we discuss",
            rewritten_user_query="what did we discuss",
            raw_response="Project Atlas",
            response_type=ResponseType.NORMAL.value,
        )
        _, chunk_id, _ = repo.add_knowledge_chunk(
            cursor,
            user_id="u1",
            title="Project Atlas",
            text="Project Atlas retention is 30 days.",
        )

    hop_payload = repo.load_outbox_entity(entity_type="conversation_hop", entity_id=hop.hop_id)
    assert hop_payload.metadata["topic_id"] == topic_id
    assert hop_payload.metadata["hop_id"] == hop.hop_id
    assert hop_payload.metadata["intent"] == Intent.GENERAL_RESPONSE.value
    assert hop_payload.metadata["response_type"] == ResponseType.NORMAL.value

    chunk_payload = repo.load_outbox_entity(entity_type="knowledge_chunk", entity_id=chunk_id)
    assert chunk_payload.metadata["chunk_id"] == chunk_id
    assert chunk_payload.metadata["knowledge_topic_id"]
    assert chunk_payload.metadata["is_deleted"] is False
    assert chunk_payload.metadata["version"] == 1


@dataclass
class FakeIndex:
    upserts: list[dict[str, Any]]
    deletes: list[str]

    def search(self, *, user_id: str, query: str, limit: int) -> list[RetrievalResult]:
        return []

    def upsert(
        self,
        *,
        user_id: str,
        entity_type: str,
        entity_id: str,
        text: str,
        metadata: dict[str, str | int | float | bool] | None = None,
    ) -> None:
        self.upserts.append(
            {
                "user_id": user_id,
                "entity_type": entity_type,
                "entity_id": entity_id,
                "text": text,
                "metadata": metadata or {},
            }
        )

    def delete(self, *, entity_id: str) -> None:
        self.deletes.append(entity_id)

    def clear(self) -> None:
        self.upserts.clear()
        self.deletes.clear()


def test_background_indexer_forwards_sql_metadata() -> None:
    repo = make_repo()
    with repo.transaction() as cursor:
        repo.add_knowledge_chunk(
            cursor,
            user_id="u1",
            title="Project Atlas",
            text="Project Atlas retention is 30 days.",
        )
    bm25 = FakeIndex(upserts=[], deletes=[])
    chroma = FakeIndex(upserts=[], deletes=[])

    processed = BackgroundIndexer(
        repository=repo,
        bm25=bm25,
        chroma=chroma,
        config=OutboxConfig(max_attempts=3, batch_size=10),
    ).process_pending()

    assert processed == 1
    assert bm25.upserts[0]["metadata"]["knowledge_topic_id"]
    assert chroma.upserts[0]["metadata"]["content_hash"]


def test_sql_hydration_drops_wrong_user_and_deleted_results() -> None:
    repo = make_repo()
    with repo.transaction() as cursor:
        _, active_id, _ = repo.add_knowledge_chunk(
            cursor,
            user_id="u1",
            title="Project Atlas",
            text="Active fact",
        )
        _, deleted_id, _ = repo.add_knowledge_chunk(
            cursor,
            user_id="u1",
            title="Project Atlas",
            text="Deleted fact",
        )
        _, other_user_id, _ = repo.add_knowledge_chunk(
            cursor,
            user_id="u2",
            title="Project Atlas",
            text="Other user fact",
        )
        repo.soft_delete_knowledge_chunk(cursor, user_id="u1", chunk_id=deleted_id)

    results = [
        RetrievalResult("knowledge_chunk", active_id, {}, 1.0, 1.0, "candidate", {}),
        RetrievalResult("knowledge_chunk", deleted_id, {}, 1.0, 1.0, "candidate", {}),
        RetrievalResult("knowledge_chunk", other_user_id, {}, 1.0, 1.0, "candidate", {}),
        RetrievalResult("knowledge_chunk", "missing", {}, 1.0, 1.0, "candidate", {}),
    ]

    hydrated = repo.hydrate_knowledge_retrieval_results(user_id="u1", results=results)

    assert [r.entity_id for r in hydrated] == [active_id]
    assert hydrated[0].validation_status == "sql_validated"
    assert hydrated[0].payload["user_id"] == "u1"
    assert hydrated[0].payload["is_deleted"] is False
