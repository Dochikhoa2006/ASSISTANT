from __future__ import annotations

import base64
import hashlib
import hmac
import json
from datetime import datetime, timedelta, timezone

import pytest

from assistant_rag.auth import AuthSettings, authenticate_token
from assistant_rag.api import _response_payload
from assistant_rag.contracts import (
    ActionValidationResult,
    BundledResponse,
    ConfirmationStatus,
    LastQAState,
    ReminderAction,
    ResponseType,
    ValidatedReminderAction,
)
from assistant_rag.database import SQLiteRepository
from assistant_rag.rate_limit import InMemoryRateLimiter
from assistant_rag.reminder_safety import ReminderTimeNormalizer, ReminderTimeNormalizationError


def make_repo() -> SQLiteRepository:
    repo = SQLiteRepository.in_memory()
    repo.initialize_schema()
    return repo


def _add_reminder(repo: SQLiteRepository, *, subject: str, reminder_time: datetime) -> str:
    result = repo.transactional_reminder_actions(
        user_id="u1",
        topic_title="Reminders",
        raw_user_query=f"remind me to {subject}",
        rewritten_user_query=f"remind me to {subject}",
        response_text="Added.",
        actions=[
            ValidatedReminderAction(
                action=ReminderAction.ADD,
                validation_result=ActionValidationResult.EXECUTE,
                subject=subject,
                reminder_time=reminder_time,
                reminder_summary=subject,
                user_timezone="UTC",
                original_time_text=reminder_time.isoformat(),
            )
        ],
    )
    assert result.committed
    assert result.results[0].domain_entity_id
    return str(result.results[0].domain_entity_id)


def test_duplicate_reminder_lookup_exact_similar_and_distinct() -> None:
    repo = make_repo()
    target_time = datetime(2026, 7, 10, 9, 0, tzinfo=timezone.utc)
    _add_reminder(repo, subject="Pay rent", reminder_time=target_time)

    exact = repo.find_active_reminder_duplicates(
        user_id="u1",
        subject="pay rent",
        reminder_time=target_time.replace(second=30),
    )
    assert exact["type"] == "exact"

    similar = repo.find_active_reminder_duplicates(
        user_id="u1",
        subject="Pay apartment rent",
        reminder_time=target_time + timedelta(minutes=20),
    )
    assert similar["type"] == "similar"

    distinct = repo.find_active_reminder_duplicates(
        user_id="u1",
        subject="Call dentist",
        reminder_time=target_time + timedelta(hours=2),
    )
    assert distinct["type"] == "none"


def test_timezone_normalizer_interprets_naive_times_in_user_timezone_and_stores_utc() -> None:
    normalizer = ReminderTimeNormalizer(default_timezone="UTC")
    naive = datetime(2026, 7, 10, 9, 0)
    normalized = normalizer.normalize(
        naive,
        platform_timezone="Asia/Ho_Chi_Minh",
        original_time_text="tomorrow at 9",
    )
    assert normalized.user_timezone == "Asia/Ho_Chi_Minh"
    assert normalized.reminder_time_utc.isoformat() == "2026-07-10T02:00:00+00:00"
    assert normalized.original_time_text == "tomorrow at 9"

    aware = datetime(2026, 7, 10, 9, 0, tzinfo=timezone.utc)
    assert normalizer.normalize(
        aware,
        platform_timezone="America/Toronto",
        original_time_text=None,
    ).reminder_time_utc == aware

    with pytest.raises(ReminderTimeNormalizationError):
        normalizer.normalize(naive, platform_timezone="Mars/Base", original_time_text=None)


def test_notification_delivery_state_is_separate_from_ui_state() -> None:
    repo = make_repo()
    reminder_id = _add_reminder(
        repo,
        subject="Pay rent",
        reminder_time=datetime(2026, 7, 10, 9, 0, tzinfo=timezone.utc),
    )
    with repo.transaction() as cursor:
        notification_id = repo.create_notification_if_absent(
            cursor,
            user_id="u1",
            reminder_id=reminder_id,
        )

    listed = repo.list_notifications(user_id="u1")[0]
    assert listed["ui_status"] == "unread"
    assert listed["delivery_status"] == "pending"
    assert listed["delivery_attempts"] == 0

    failed = repo.mark_notification_delivery_failed(
        user_id="u1",
        notification_id=notification_id,
        error_message="socket closed",
        retrying=True,
    )
    assert failed["delivery_status"] == "retrying"
    assert failed["delivery_attempts"] == 1

    read = repo.update_notification_ui_status(
        user_id="u1",
        notification_id=notification_id,
        ui_status="read",
    )
    assert read["ui_status"] == "read"
    assert read["delivery_status"] == "retrying"

    sent = repo.mark_notification_delivery_sent(user_id="u1", notification_id=notification_id)
    assert sent["delivery_status"] == "sent"
    assert sent["sent_at"]


def test_idempotency_claim_replay_in_progress_and_conflict() -> None:
    repo = make_repo()
    claim = repo.claim_idempotency_key(user_id="u1", idempotency_key="k1", payload_hash="hash-a")
    assert claim.status == "started"
    assert repo.claim_idempotency_key(user_id="u1", idempotency_key="k1", payload_hash="hash-a").status == "in_progress"
    assert repo.claim_idempotency_key(user_id="u1", idempotency_key="k1", payload_hash="hash-b").status == "conflict"

    stored = {"ok": True}
    repo.complete_idempotency_request(request_id=claim.request_id, stored_response_json=json.dumps(stored))
    replay = repo.claim_idempotency_key(user_id="u1", idempotency_key="k1", payload_hash="hash-a")
    assert replay.status == "replay"
    assert json.loads(replay.stored_response_json or "{}") == stored


def test_confirmation_lifecycle_pending_expired_and_confirmed() -> None:
    repo = make_repo()
    expires_at = (datetime.now(timezone.utc) + timedelta(minutes=15)).isoformat()
    confirmation = repo.create_pending_confirmation(
        user_id="u1",
        action_type="reminder_mutation",
        target_entity_type="reminder",
        target_entity_id="r1",
        proposed_action={"domain": "reminder", "actions": []},
        target_snapshot={"version": 1},
        expires_at=expires_at,
    )
    loaded = repo.load_pending_confirmation(
        user_id="u1",
        confirmation_token=confirmation["confirmation_token"],
        now_value=datetime.now(timezone.utc).isoformat(),
    )
    assert loaded["status"] == ConfirmationStatus.PENDING.value
    repo.mark_confirmation_confirmed(
        user_id="u1",
        confirmation_token=confirmation["confirmation_token"],
    )
    with pytest.raises(ValueError):
        repo.load_pending_confirmation(
            user_id="u1",
            confirmation_token=confirmation["confirmation_token"],
            now_value=datetime.now(timezone.utc).isoformat(),
        )

    expired = repo.create_pending_confirmation(
        user_id="u1",
        action_type="reminder_mutation",
        target_entity_type="reminder",
        target_entity_id="r2",
        proposed_action={"domain": "reminder", "actions": []},
        target_snapshot={"version": 1},
        expires_at=(datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(),
    )
    with pytest.raises(ValueError):
        repo.load_pending_confirmation(
            user_id="u1",
            confirmation_token=expired["confirmation_token"],
            now_value=datetime.now(timezone.utc).isoformat(),
        )


def _jwt(secret: str, claims: dict[str, object]) -> str:
    header = {"alg": "HS256", "typ": "JWT"}

    def enc(payload: dict[str, object]) -> str:
        raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")

    signing_input = f"{enc(header)}.{enc(claims)}"
    signature = hmac.new(secret.encode("utf-8"), signing_input.encode("ascii"), hashlib.sha256).digest()
    return f"{signing_input}.{base64.urlsafe_b64encode(signature).rstrip(b'=').decode('ascii')}"


def test_jwt_auth_valid_expired_wrong_issuer_and_scope() -> None:
    settings = AuthSettings(
        issuer="issuer-a",
        audience="assistant",
        public_key="secret",
        required_scopes=("chat:write",),
    )
    valid = _jwt(
        "secret",
        {
            "sub": "u1",
            "iss": "issuer-a",
            "aud": "assistant",
            "exp": int(datetime.now(timezone.utc).timestamp()) + 60,
            "scope": "chat:write",
        },
    )
    assert authenticate_token(valid, settings).user_id == "u1"

    expired = _jwt(
        "secret",
        {
            "sub": "u1",
            "iss": "issuer-a",
            "aud": "assistant",
            "exp": int(datetime.now(timezone.utc).timestamp()) - 60,
            "scope": "chat:write",
        },
    )
    with pytest.raises(Exception):
        authenticate_token(expired, settings)

    wrong_issuer = _jwt(
        "secret",
        {
            "sub": "u1",
            "iss": "issuer-b",
            "aud": "assistant",
            "exp": int(datetime.now(timezone.utc).timestamp()) + 60,
            "scope": "chat:write",
        },
    )
    with pytest.raises(Exception):
        authenticate_token(wrong_issuer, settings)

    missing_scope = _jwt(
        "secret",
        {
            "sub": "u1",
            "iss": "issuer-a",
            "aud": "assistant",
            "exp": int(datetime.now(timezone.utc).timestamp()) + 60,
        },
    )
    with pytest.raises(Exception):
        authenticate_token(missing_scope, settings)


def test_in_memory_rate_limiter_is_per_key_and_reports_retry_after() -> None:
    limiter = InMemoryRateLimiter()
    assert limiter.allow("chat:u1", 2, 60).allowed
    assert limiter.allow("chat:u1", 2, 60).allowed
    blocked = limiter.allow("chat:u1", 2, 60)
    assert not blocked.allowed
    assert blocked.retry_after_seconds > 0
    assert limiter.allow("chat:u2", 2, 60).allowed


def test_chat_response_payload_has_release2_shape() -> None:
    response = BundledResponse(
        final_chat_text="Done.",
        response_type=ResponseType.NORMAL,
        last_qa_state=LastQAState(
            last_user_query="hello",
            last_response="Done.",
            response_type=ResponseType.NORMAL,
        ),
        conversation_topic_id="topic-1",
        conversation_hop_id="hop-1",
        actions_committed=[],
        actions_pending_confirmation=[],
        warnings=["compatibility warning"],
        persistence_instructions={"audit_hop_id": "hop-1"},
        platform_payload={"kind": "text"},
    )
    payload = _response_payload(response, request_id="req-1", latency_ms=12)
    assert set(payload) == {
        "request_id",
        "final_chat_text",
        "response_type",
        "conversation_topic_id",
        "conversation_hop_id",
        "actions_committed",
        "actions_pending_confirmation",
        "warnings",
        "latency_ms",
        "persistence_instructions",
        "platform_payload",
    }
    assert payload["request_id"] == "req-1"
    assert payload["latency_ms"] == 12
