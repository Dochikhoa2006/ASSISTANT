from __future__ import annotations

import json
from typing import Any, Iterator

import pytest

from assistant_rag.branches import BranchRouter
from assistant_rag.chat_history import last_qa_chat_history, supporting_question_context
from assistant_rag.contracts import (
    BranchResult,
    ChatRequest,
    Intent,
    LastQAState,
    OutboundMessageState,
    PipelineContext,
    ResponseType,
)
from assistant_rag.database import SQLiteRepository
from assistant_rag.last_qa import DiskCacheLastQAStore
from assistant_rag.postgres_repository import PostgresRepository
from assistant_rag.reminder_reply import (
    build_reminder_reply_last_qa,
    build_reminder_state,
    mark_reminder_state_replied,
    reminder_notification_key,
    reminder_state_hash,
    verified_reminder_state,
)


USER_ID = "reminder-reply-state-user"
FIRE_TIME = "2026-07-18T08:45:00+00:00"


@pytest.fixture(params=("sqlite", "sqlalchemy"))
def repository(request: pytest.FixtureRequest) -> Iterator[Any]:
    if request.param == "sqlite":
        value = SQLiteRepository.in_memory()
    else:
        value = PostgresRepository.create("sqlite+pysqlite:///:memory:")
    value.initialize_schema()
    try:
        yield value
    finally:
        if hasattr(value, "close"):
            value.close()
        else:
            value.connection.close()


def _seed_notification(repository: Any, *, supporting_question: str | None) -> tuple[str, str]:
    with repository.transaction() as cursor:
        topic_id = repository.ensure_topic(
            cursor,
            user_id=USER_ID,
            title="Reminder reply source",
        )
        source_hop = repository.append_conversation_hop(
            cursor,
            topic_id=topic_id,
            user_id=USER_ID,
            intent=Intent.REMINDER.value,
            raw_user_query="Remind me about the client meeting tomorrow.",
            rewritten_user_query="Create a client meeting reminder for tomorrow.",
            raw_response="The reminder was created.",
            response_type=ResponseType.REMINDER_ACTION.value,
        )
        reminder_id = repository.add_reminder(
            cursor,
            user_id=USER_ID,
            source_topic_id=topic_id,
            source_hop_id=source_hop.hop_id,
            reminder_time="2026-07-18T09:00:00+00:00",
            event_time="2026-07-18T09:00:00+00:00",
            raw_reminder="Client meeting tomorrow",
            reminder_summary="Prepare for the client meeting.",
            subject="Client meeting",
            supporting_question=supporting_question,
            user_timezone="UTC",
            original_time_text="tomorrow at 9 AM",
        )
        notification_id = repository.create_notification_if_absent(
            cursor,
            user_id=USER_ID,
            reminder_id=reminder_id,
            fire_time=FIRE_TIME,
        )
    return reminder_id, notification_id


def test_notification_lookup_key_is_a_stable_sha256_digest() -> None:
    key = reminder_notification_key("reminder-1", "notification-1")

    assert len(key) == 64
    assert key == reminder_notification_key("reminder-1", "notification-1")
    assert key != reminder_notification_key("reminder-1", "notification-2")
    int(key, 16)


@pytest.mark.parametrize(
    ("supporting_question", "expected_reply_kind"),
    (
        ("Would you like a meeting checklist?", "supporting_question_reply"),
        (None, "notification_purpose_reply"),
    ),
)
def test_sql_context_and_direct_reply_hop_preserve_complete_reminder_state(
    repository: Any,
    supporting_question: str | None,
    expected_reply_kind: str,
) -> None:
    reminder_id, notification_id = _seed_notification(
        repository,
        supporting_question=supporting_question,
    )
    context = repository.load_reminder_reply_context(
        user_id=USER_ID,
        reminder_id=reminder_id,
        notification_id=notification_id,
    )

    assert context["subject"] == "Client meeting"
    assert context["raw_reminder"] == "Client meeting tomorrow"
    assert context["notification_fire_time"] == FIRE_TIME
    assert context["source_hop_id"]

    last_qa = build_reminder_reply_last_qa(
        context,
        reminder_id=reminder_id,
        notification_id=notification_id,
    )
    assert last_qa.reminder_state is not None
    assert last_qa.reminder_state["title"] == "Client meeting"
    assert last_qa.reminder_state["supporting_question"] == supporting_question
    assert last_qa.reminder_state["has_been_notified"] is True
    assert last_qa.reminder_state_hash == reminder_state_hash(last_qa.reminder_state)

    reply_hop = repository.append_reminder_reply(
        user_id=USER_ID,
        reminder_id=reminder_id,
        notification_id=notification_id,
        reply_text="Please make a preparation plan.",
        response_text="I will help prepare the meeting.",
    )
    if isinstance(repository, SQLiteRepository):
        row = repository.connection.execute(
            "SELECT entities_json FROM conversation_hops WHERE hop_id = ?",
            (reply_hop.hop_id,),
        ).fetchone()
        payload = json.loads(row["entities_json"])
    else:
        from assistant_rag.postgres_schema import conversation_hops
        from sqlalchemy import select

        with repository.engine.connect() as connection:
            row = connection.execute(
                select(conversation_hops.c.entities_json).where(
                    conversation_hops.c.hop_id == reply_hop.hop_id
                )
            ).fetchone()
        payload = json.loads(row[0])

    stored = payload["reminder_reply"]
    assert stored["state"]["reply_received"] is True
    assert stored["state"]["reply_kind"] == expected_reply_kind
    assert stored["state"]["reply_conversation_hop_id"] == reply_hop.hop_id
    assert verified_reminder_state(stored["state"], stored["state_hash"]) is not None


def test_last_qa_disk_and_chat_history_round_trip_reminder_state(tmp_path: Any) -> None:
    base = build_reminder_state(
        {
            "reminder_id": "reminder-1",
            "notification_id": "notification-1",
            "subject": "Submit report",
            "reminder_summary": "Submit the monthly report.",
            "notification_fire_time": FIRE_TIME,
            "source_topic_id": "topic-1",
            "source_hop_id": "hop-1",
        }
    )
    replied = mark_reminder_state_replied(base, reply_hop_id="reply-hop-1")
    state = LastQAState(
        last_user_query="Create the spreadsheet first.",
        last_response="I can create it.",
        response_type=ResponseType.NORMAL,
        linked_topic_id="topic-1",
        linked_hop_id="reply-hop-1",
        reminder_state=replied,
        reminder_state_hash=reminder_state_hash(replied),
        outbound_state=OutboundMessageState(
            channel="gmail",
            status="draft_ready",
            recipients=("alice@example.com", "bob@example.com"),
            subject="Monthly report",
            body="The monthly report is attached.",
            artifact_ids=("artifact-1",),
            attachment_filenames=("monthly-report.pdf",),
            source_topic_id="topic-1",
            source_hop_id="reply-hop-1",
        ),
    )
    store = DiskCacheLastQAStore(str(tmp_path / "last-qa.sqlite3"), ttl_seconds=60)

    store.save(USER_ID, state)
    restored = store.get(USER_ID)

    assert restored is not None
    assert restored.reminder_state == replied
    assert restored.reminder_state_hash == reminder_state_hash(replied)
    assert restored.outbound_state == state.outbound_state
    history = last_qa_chat_history(restored)
    assert history[0]["reminder_state"]["title"] == "Submit report"
    projected = supporting_question_context(history)
    assert projected[0]["reminder_state"]["reply_kind"] == "notification_purpose_reply"


def test_generated_artifact_is_bound_once_to_its_user_owned_hop(
    repository: Any,
    tmp_path: Any,
) -> None:
    with repository.transaction() as cursor:
        topic_id = repository.ensure_topic(
            cursor,
            user_id=USER_ID,
            title="Artifact binding",
        )
        first_hop = repository.append_conversation_hop(
            cursor,
            topic_id=topic_id,
            user_id=USER_ID,
            intent=Intent.GENERAL_RESPONSE.value,
            raw_user_query="Create the report.",
            rewritten_user_query="Create the report.",
            raw_response="Created.",
            response_type=ResponseType.NORMAL.value,
            entities={},
        )
        second_hop = repository.append_conversation_hop(
            cursor,
            topic_id=topic_id,
            user_id=USER_ID,
            intent=Intent.GENERAL_RESPONSE.value,
            raw_user_query="Another turn.",
            rewritten_user_query="Another turn.",
            raw_response="Done.",
            response_type=ResponseType.NORMAL.value,
            entities={},
        )
    path = tmp_path / "medical-details.pdf"
    path.write_bytes(b"%PDF-1.4 binding")
    artifact = repository.create_generated_artifact(
        user_id=USER_ID,
        conversation_hop_id=None,
        file_type="pdf",
        filename=path.name,
        storage_path=str(path),
        storage_url=f"/artifacts/{path.name}",
    )

    bound = repository.bind_generated_artifacts(
        user_id=USER_ID,
        artifact_ids=[artifact["artifact_id"], artifact["artifact_id"]],
        conversation_hop_id=first_hop.hop_id,
    )
    refused_rebind = repository.bind_generated_artifacts(
        user_id=USER_ID,
        artifact_ids=[artifact["artifact_id"]],
        conversation_hop_id=second_hop.hop_id,
    )

    restored = repository.get_generated_artifact(
        user_id=USER_ID,
        artifact_id=artifact["artifact_id"],
    )
    assert bound == [artifact["artifact_id"]]
    assert refused_rebind == []
    assert restored["conversation_hop_id"] == first_hop.hop_id


class _WritingBranch:
    def execute(self, context: PipelineContext, repository: Any) -> BranchResult:
        with repository.transaction() as cursor:
            topic_id = repository.ensure_topic(
                cursor,
                user_id=context.request.user_id,
                title="Normal reply branch",
            )
            hop = repository.append_conversation_hop(
                cursor,
                topic_id=topic_id,
                user_id=context.request.user_id,
                intent=Intent.GENERAL_RESPONSE.value,
                raw_user_query=context.request.raw_query,
                rewritten_user_query=context.rewritten_query,
                raw_response="I can help with that.",
                response_type=ResponseType.NORMAL.value,
                entities={"existing": {"preserved": True}},
            )
        return BranchResult(
            response_type=ResponseType.NORMAL,
            normal_response_text="I can help with that.",
            linked_topic_id=topic_id,
            linked_hop_id=hop.hop_id,
            indexing_job_result={"outbox_job_ids": [hop.outbox_job_id]},
        )


def test_common_branch_persistence_merges_verified_reminder_state_into_reply_hop(
    repository: Any,
) -> None:
    base = build_reminder_state(
        {
            "reminder_id": "reminder-1",
            "notification_id": "notification-1",
            "subject": "Client meeting",
            "supporting_question": "Would you like a checklist?",
            "source_topic_id": "topic-source",
            "source_hop_id": "hop-source",
        }
    )
    state = LastQAState(
        last_user_query="Create the reminder.",
        last_response="Reminder created.",
        response_type=ResponseType.REMINDER_ACTION,
        linked_topic_id="topic-source",
        linked_hop_id="hop-source",
        reminder_state=base,
        reminder_state_hash=reminder_state_hash(base),
    )
    context = PipelineContext(
        request=ChatRequest(user_id=USER_ID, raw_query="Yes, a checklist please."),
        rewritten_query="Yes, a checklist please.",
        last_qa_state=state,
        conversation_results=[],
        intent=Intent.GENERAL_RESPONSE,
        last_qa_trace={"interaction_type": "reminder_notification_reply"},
    )

    result = BranchRouter(
        {Intent.GENERAL_RESPONSE: _WritingBranch()}
    ).route(context, repository)

    if isinstance(repository, SQLiteRepository):
        row = repository.connection.execute(
            "SELECT entities_json FROM conversation_hops WHERE hop_id = ?",
            (result.linked_hop_id,),
        ).fetchone()
        entities = json.loads(row["entities_json"])
    else:
        from assistant_rag.postgres_schema import conversation_hops
        from sqlalchemy import select

        with repository.engine.connect() as connection:
            row = connection.execute(
                select(conversation_hops.c.entities_json).where(
                    conversation_hops.c.hop_id == result.linked_hop_id
                )
            ).fetchone()
        entities = json.loads(row[0])
    assert entities["existing"] == {"preserved": True}
    stored = entities["reminder_reply"]
    assert stored["state"]["reply_kind"] == "supporting_question_reply"
    assert stored["state"]["reply_conversation_hop_id"] == result.linked_hop_id
    assert verified_reminder_state(stored["state"], stored["state_hash"]) is not None
