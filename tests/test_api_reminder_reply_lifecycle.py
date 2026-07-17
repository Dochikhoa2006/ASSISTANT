from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from fastapi.testclient import TestClient

from assistant_rag.api import build_api_app
from assistant_rag.auth import get_auth_context
from assistant_rag.contracts import (
    AuthContext,
    BundledResponse,
    ChatRequest,
    IdempotencyClaimResult,
    LastQAState,
    RateLimitResult,
    ResponseType,
)
from assistant_rag.reminder_reply import reminder_state_hash


class RecordingLastQAStore:
    def __init__(self) -> None:
        self.saved: list[tuple[str, LastQAState]] = []

    def save(self, user_id: str, state: LastQAState) -> None:
        self.saved.append((user_id, state))


class RecordingPipeline:
    def __init__(self, responses: list[BundledResponse]) -> None:
        self.responses = list(responses)
        self.requests: list[ChatRequest] = []
        self.last_qa_store = RecordingLastQAStore()

    def handle(self, request: ChatRequest, _repository: object) -> BundledResponse:
        self.requests.append(request)
        if not self.responses:
            raise AssertionError("Pipeline was called more times than expected")
        return self.responses.pop(0)


@dataclass
class _IdempotencyEntry:
    request_id: str
    payload_hash: str
    status: str = "in_progress"
    stored_response_json: str | None = None


class RecordingRepository:
    def __init__(self) -> None:
        self.notification_status = "unread"
        self.notification_updates: list[tuple[str, str, str]] = []
        self.context_loads: list[tuple[str, str, str]] = []
        self.entries: dict[tuple[str, str], _IdempotencyEntry] = {}
        self.claim_statuses: list[str] = []

    def load_reminder_reply_context(
        self, *, user_id: str, reminder_id: str, notification_id: str
    ) -> dict[str, Any]:
        self.context_loads.append((user_id, reminder_id, notification_id))
        return {
            "reminder_id": reminder_id,
            "notification_id": notification_id,
            "subject": "Project review",
            "reminder_summary": "Review the project plan",
            "raw_reminder": "Remind me to review the project plan",
            "reminder_status": "notified",
            "reminder_time": "2026-07-18T15:00:00+00:00",
            "notification_ui_status": "unread",
            "notification_delivery_status": "sent",
            "notification_fire_time": "2026-07-18T14:45:00+00:00",
            "source_topic_id": "topic-source",
            "source_hop_id": "hop-source",
            "source_raw_user_query": "Set a project review reminder.",
            "source_rewritten_user_query": "Set a project review reminder.",
            "source_raw_response": "When should I remind you?",
            "source_response_type": ResponseType.REMINDER_ACTION.value,
            "supporting_question": "When should I remind you?",
            "supporting_questions_json": "[]",
        }

    def update_notification_ui_status(
        self, *, user_id: str, notification_id: str, ui_status: str
    ) -> dict[str, Any]:
        self.notification_updates.append((user_id, notification_id, ui_status))
        self.notification_status = ui_status
        return {"notification_id": notification_id, "ui_status": ui_status}

    def claim_idempotency_key(
        self, *, user_id: str, idempotency_key: str, payload_hash: str
    ) -> IdempotencyClaimResult:
        key = (user_id, idempotency_key)
        entry = self.entries.get(key)
        if entry is None:
            entry = _IdempotencyEntry(
                request_id=f"request-{len(self.entries) + 1}",
                payload_hash=payload_hash,
            )
            self.entries[key] = entry
            self.claim_statuses.append("started")
            return IdempotencyClaimResult(
                status="started", request_id=entry.request_id
            )
        if entry.payload_hash != payload_hash:
            self.claim_statuses.append("conflict")
            return IdempotencyClaimResult(
                status="conflict",
                request_id=entry.request_id,
                reason="idempotency_conflict",
            )
        if entry.status == "completed":
            self.claim_statuses.append("replay")
            return IdempotencyClaimResult(
                status="replay",
                request_id=entry.request_id,
                stored_response_json=entry.stored_response_json,
            )
        if entry.status == "failed":
            entry.status = "in_progress"
            self.claim_statuses.append("failed_retry")
            return IdempotencyClaimResult(
                status="failed_retry", request_id=entry.request_id
            )
        self.claim_statuses.append("in_progress")
        return IdempotencyClaimResult(
            status="in_progress",
            request_id=entry.request_id,
            reason="request_in_progress",
        )

    def complete_idempotency_request(
        self, *, request_id: str, stored_response_json: str
    ) -> None:
        entry = self._entry_for_request(request_id)
        entry.status = "completed"
        entry.stored_response_json = stored_response_json

    def fail_idempotency_request(
        self, *, request_id: str, error_message: str | None = None
    ) -> None:
        del error_message
        self._entry_for_request(request_id).status = "failed"

    def _entry_for_request(self, request_id: str) -> _IdempotencyEntry:
        return next(
            entry for entry in self.entries.values() if entry.request_id == request_id
        )


class AlwaysAllowRateLimiter:
    def allow(self, _key: str, _limit: int, _window_seconds: int) -> RateLimitResult:
        return RateLimitResult(allowed=True)


def _response(response_type: ResponseType, text: str) -> BundledResponse:
    return BundledResponse(
        final_chat_text=text,
        response_type=response_type,
        last_qa_state=LastQAState(
            last_user_query="At 3 PM.",
            last_response=text,
            response_type=response_type,
        ),
        conversation_topic_id="topic-source",
        conversation_hop_id="hop-reply",
    )


def _client(
    responses: list[BundledResponse],
) -> tuple[TestClient, RecordingRepository, RecordingPipeline]:
    repository = RecordingRepository()
    pipeline = RecordingPipeline(responses)
    app = build_api_app(
        pipeline=pipeline,  # type: ignore[arg-type]
        repository_factory=lambda: repository,  # type: ignore[arg-type]
        rate_limiter=AlwaysAllowRateLimiter(),  # type: ignore[arg-type]
    )
    app.dependency_overrides[get_auth_context] = lambda: AuthContext(
        user_id="reply-user"
    )
    return TestClient(app), repository, pipeline


def _payload(**updates: str) -> dict[str, str]:
    payload = {
        "user_id": "reply-user",
        "notification_id": "notification-1",
        "reply_text": "At 3 PM.",
    }
    payload.update(updates)
    return payload


def test_reminder_reply_success_uses_lifecycle_and_marks_notification_read() -> None:
    client, repository, pipeline = _client(
        [_response(ResponseType.REMINDER_REPLY, "The reminder is set for 3 PM.")]
    )

    result = client.post("/reminders/reminder-1/reply", json=_payload())

    assert result.status_code == 200
    body = result.json()
    assert body["final_chat_text"] == "The reminder is set for 3 PM."
    assert body["response_type"] == ResponseType.REMINDER_REPLY.value
    assert "trace_summary" in body
    assert repository.notification_status == "read"
    assert repository.notification_updates == [
        ("reply-user", "notification-1", "read")
    ]
    assert len(pipeline.requests) == 1
    request = pipeline.requests[0]
    assert request.reminder_id == "reminder-1"
    assert request.notification_id == "notification-1"
    assert request.reply_text == "At 3 PM."
    assert request.parent_hop_id == "hop-source"
    assert request.metadata["reminder_reply_context"] is True
    metadata_state = request.metadata["reminder_state"]
    assert metadata_state["title"] == "Project review"
    assert metadata_state["supporting_question"] == "When should I remind you?"
    assert metadata_state["notification_created"] is True
    assert metadata_state["has_been_notified"] is True
    assert request.metadata["reminder_state_hash"] == reminder_state_hash(
        metadata_state
    )
    saved_state = pipeline.last_qa_store.saved[0][1]
    assert saved_state.linked_hop_id == "hop-source"
    assert saved_state.reminder_state == metadata_state
    assert saved_state.reminder_state_hash == request.metadata["reminder_state_hash"]


def test_reminder_reply_error_stays_unread_and_same_key_can_retry() -> None:
    client, repository, pipeline = _client(
        [
            _response(ResponseType.ERROR, "Temporary failure."),
            _response(ResponseType.REMINDER_REPLY, "Retry succeeded."),
        ]
    )
    payload = _payload(idempotency_key="reply-retry-key")

    failed = client.post("/reminders/reminder-1/reply", json=payload)

    assert failed.status_code == 200
    assert failed.json()["response_type"] == ResponseType.ERROR.value
    assert repository.notification_status == "unread"
    assert repository.notification_updates == []
    assert repository.entries[("reply-user", "reply-retry-key")].status == "failed"

    retried = client.post("/reminders/reminder-1/reply", json=payload)

    assert retried.status_code == 200
    assert retried.json()["final_chat_text"] == "Retry succeeded."
    assert repository.notification_status == "read"
    assert repository.notification_updates == [
        ("reply-user", "notification-1", "read")
    ]
    assert repository.claim_statuses == ["started", "failed_retry"]
    assert len(pipeline.requests) == 2


def test_keyed_plain_reminder_reply_replays_without_pipeline_or_ui_side_effects() -> None:
    client, repository, pipeline = _client(
        [_response(ResponseType.REMINDER_REPLY, "Recorded for 3 PM.")]
    )
    payload = _payload(idempotency_key="plain-reply-key")

    first = client.post("/reminders/reminder-1/reply", json=payload)
    replay = client.post("/reminders/reminder-1/reply", json=payload)

    assert first.status_code == 200
    assert replay.status_code == 200
    assert replay.json() == first.json()
    assert repository.claim_statuses == ["started", "replay"]
    assert len(pipeline.requests) == 1
    assert len(pipeline.last_qa_store.saved) == 1
    assert repository.notification_updates == [
        ("reply-user", "notification-1", "read")
    ]
    stored = repository.entries[("reply-user", "plain-reply-key")]
    assert stored.status == "completed"
    assert json.loads(stored.stored_response_json or "{}") == first.json()


def test_reminder_reply_idempotency_conflict_does_not_rehydrate_or_execute() -> None:
    client, repository, pipeline = _client(
        [_response(ResponseType.REMINDER_REPLY, "Recorded for 3 PM.")]
    )
    initial = _payload(idempotency_key="reply-conflict-key")
    changed = _payload(
        idempotency_key="reply-conflict-key",
        reply_text="At 4 PM instead.",
    )

    first = client.post("/reminders/reminder-1/reply", json=initial)
    conflict = client.post("/reminders/reminder-1/reply", json=changed)

    assert first.status_code == 200
    assert conflict.status_code == 409
    assert conflict.json()["detail"] == "idempotency_conflict"
    assert repository.claim_statuses == ["started", "conflict"]
    assert len(pipeline.requests) == 1
    assert len(pipeline.last_qa_store.saved) == 1
    assert repository.notification_updates == [
        ("reply-user", "notification-1", "read")
    ]


def test_strict_idempotency_policy_rejects_unkeyed_reminder_reply_without_side_effects(
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("ASSISTANT_ALLOW_MISSING_IDEMPOTENCY_KEY", "false")
    client, repository, pipeline = _client(
        [_response(ResponseType.REMINDER_REPLY, "Must not execute.")]
    )

    result = client.post("/reminders/reminder-1/reply", json=_payload())

    assert result.status_code == 400
    assert result.json()["detail"] == (
        "idempotency_key is required for mutation requests"
    )
    assert pipeline.requests == []
    assert pipeline.last_qa_store.saved == []
    assert repository.notification_status == "unread"
    assert repository.notification_updates == []
