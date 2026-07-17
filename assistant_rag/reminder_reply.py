"""Exact reminder-notification reply context and Last-QA restoration."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from typing import Any

from .contracts import (
    ExpectedResponseType,
    GeneratedQuestion,
    LastQAState,
    QuestionSource,
    ResponseType,
)


def reminder_notification_key(reminder_id: str, notification_id: str) -> str:
    """Hash one reminder/notification pair for collision-safe UI lookup."""
    encoded = json.dumps(
        [str(reminder_id), str(notification_id)],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def build_reminder_reply_context_index(
    *, repository: Any, user_id: str, notifications: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Hash every visible notification to the authoritative source conversation hop."""
    index: dict[str, dict[str, Any]] = {}
    for notification in notifications:
        reminder_id = str(notification.get("reminder_id") or "")
        notification_id = str(notification.get("notification_id") or "")
        if not reminder_id or not notification_id:
            continue
        context = repository.load_reminder_reply_context(
            user_id=user_id,
            reminder_id=reminder_id,
            notification_id=notification_id,
        )
        if context:
            index[reminder_notification_key(reminder_id, notification_id)] = dict(context)
    return index


def build_reminder_state(
    context: dict[str, Any],
    *,
    reminder_id: str | None = None,
    notification_id: str | None = None,
) -> dict[str, Any]:
    """Project SQL reminder/notification facts into bounded semantic state."""
    resolved_reminder_id = str(
        reminder_id or context.get("reminder_id") or ""
    ).strip()
    resolved_notification_id = str(
        notification_id or context.get("notification_id") or ""
    ).strip()
    supporting_question = _question_text(context)
    state = {
        "reminder_id": resolved_reminder_id,
        "notification_id": resolved_notification_id,
        "title": _optional_text(context.get("subject")),
        "subject": _optional_text(context.get("subject")),
        "reminder_summary": _optional_text(context.get("reminder_summary")),
        "raw_reminder": _optional_text(context.get("raw_reminder")),
        "reminder_status": _optional_text(context.get("reminder_status")),
        "notification_created": bool(resolved_notification_id),
        # Recurring reminders remain ``scheduled`` after firing, so the
        # durable notification row—not reminder status—is the authoritative
        # evidence that this occurrence has already notified the user.
        "has_been_notified": bool(resolved_notification_id),
        "notification_ui_status": _optional_text(
            context.get("notification_ui_status") or context.get("ui_status")
        ),
        "notification_delivery_status": _optional_text(
            context.get("notification_delivery_status")
            or context.get("delivery_status")
        ),
        "notification_fire_time": _optional_text(
            context.get("notification_fire_time") or context.get("fire_time")
        ),
        "notification_created_at": _optional_text(
            context.get("notification_created_at")
        ),
        "reminder_time": _optional_text(context.get("reminder_time")),
        "event_time": _optional_text(context.get("event_time")),
        "last_fire_time": _optional_text(context.get("last_fire_time")),
        "next_fire_time": _optional_text(context.get("next_fire_time")),
        "user_timezone": _optional_text(context.get("user_timezone")),
        "original_time_text": _optional_text(context.get("original_time_text")),
        "recurrence_rule": _optional_text(context.get("recurrence_rule")),
        "recurrence_timezone": _optional_text(
            context.get("recurrence_timezone")
        ),
        "parent_recurring_reminder_id": _optional_text(
            context.get("parent_recurring_reminder_id")
        ),
        "timing_plan_status": _optional_text(context.get("timing_plan_status")),
        "supporting_question_plan_status": _optional_text(
            context.get("supporting_question_plan_status")
        ),
        "supporting_question_confidence": _optional_number(
            context.get("supporting_question_confidence")
        ),
        "supporting_question": supporting_question,
        "supporting_response": _optional_text(context.get("supporting_response")),
        "has_supporting_question": bool(supporting_question),
        "reply_received": False,
        "reply_kind": "awaiting_reply",
        "source_topic_id": _optional_text(
            context.get("source_topic_id") or context.get("topic_id")
        ),
        "source_hop_id": _optional_text(context.get("source_hop_id")),
    }
    return state


def reminder_state_hash(state: dict[str, Any]) -> str:
    encoded = json.dumps(
        state,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def verified_reminder_state(
    state: Any,
    state_hash: Any,
) -> dict[str, Any] | None:
    """Return a defensive copy only when the reminder state is self-consistent."""
    if not isinstance(state, dict) or not isinstance(state_hash, str):
        return None
    if reminder_state_hash(state) != state_hash:
        return None
    required_text = (
        "reminder_id",
        "notification_id",
        "source_topic_id",
        "source_hop_id",
    )
    if any(not _optional_text(state.get(field)) for field in required_text):
        return None
    if (
        state.get("notification_created") is not True
        or state.get("has_been_notified") is not True
    ):
        return None
    return deepcopy(state)


def mark_reminder_state_replied(
    state: dict[str, Any],
    *,
    reply_hop_id: str | None = None,
) -> dict[str, Any]:
    replied = deepcopy(state)
    replied["reply_received"] = True
    replied["reply_kind"] = (
        "supporting_question_reply"
        if replied.get("has_supporting_question")
        else "notification_purpose_reply"
    )
    replied["reply_conversation_hop_id"] = _optional_text(reply_hop_id)
    return replied


def reminder_reply_hop_entities(
    state: dict[str, Any],
    *,
    reply_hop_id: str | None = None,
) -> dict[str, Any]:
    replied = mark_reminder_state_replied(state, reply_hop_id=reply_hop_id)
    return {
        "reminder_reply": {
            "state": replied,
            "state_hash": reminder_state_hash(replied),
        }
    }


def build_reminder_reply_last_qa(
    context: dict[str, Any],
    *,
    reminder_id: str | None = None,
    notification_id: str | None = None,
) -> LastQAState:
    """Rebuild Last-QA from the reminder's source hop, never from UI text."""
    supporting_questions = _questions_from_json(context.get("supporting_questions_json"))
    reminder_question = _question_from_value(context.get("supporting_question"), QuestionSource.REMINDER_SUPPORTING_QUESTION)
    response_type = _response_type(context.get("source_response_type"))
    reminder_state = build_reminder_state(
        context,
        reminder_id=reminder_id,
        notification_id=notification_id,
    )
    return LastQAState(
        last_user_query=str(context.get("source_rewritten_user_query") or ""),
        last_response=str(context.get("source_raw_response") or context.get("supporting_response") or ""),
        response_type=response_type,
        supporting_questions=supporting_questions,
        reminder_supporting_question=reminder_question,
        linked_topic_id=_optional_text(context.get("source_topic_id") or context.get("topic_id")),
        linked_hop_id=_optional_text(context.get("source_hop_id")),
        reminder_state=reminder_state,
        reminder_state_hash=reminder_state_hash(reminder_state),
    )


def build_reminder_reply_metadata(
    *, reminder_id: str, notification_id: str, context: dict[str, Any],
) -> dict[str, Any]:
    """Pass immutable source identifiers to the normal pipeline."""
    reminder_state = build_reminder_state(
        context,
        reminder_id=reminder_id,
        notification_id=notification_id,
    )
    return {
        "reminder_reply_context": True,
        "reminder_id": str(reminder_id),
        "notification_id": str(notification_id),
        "source_topic_id": _optional_text(context.get("source_topic_id") or context.get("topic_id")),
        "source_hop_id": _optional_text(context.get("source_hop_id")),
        "supporting_question": _question_text(context),
        "reminder_state": reminder_state,
        "reminder_state_hash": reminder_state_hash(reminder_state),
    }


def _question_text(context: dict[str, Any]) -> str | None:
    question = _question_from_value(context.get("supporting_question"), QuestionSource.REMINDER_SUPPORTING_QUESTION)
    if question:
        return question.text
    questions = _questions_from_json(context.get("supporting_questions_json"))
    return questions[0].text if questions else None


def _questions_from_json(value: Any) -> list[GeneratedQuestion]:
    if not value:
        return []
    try:
        payload = json.loads(value) if isinstance(value, str) else value
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    if not isinstance(payload, list):
        return []
    result: list[GeneratedQuestion] = []
    for item in payload:
        question = _question_from_value(item, QuestionSource.HUMAN_SUPPORTING_QUESTION)
        if question:
            result.append(question)
    return result


def _question_from_value(value: Any, default_source: QuestionSource) -> GeneratedQuestion | None:
    if not value:
        return None
    payload: dict[str, Any]
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            decoded = None
        payload = decoded if isinstance(decoded, dict) else {"text": value}
    elif isinstance(value, dict):
        payload = value
    else:
        return None
    text = str(payload.get("text") or payload.get("question_text") or "").strip()
    if not text:
        return None
    try:
        source = QuestionSource(payload.get("source") or payload.get("question_source") or default_source.value)
    except ValueError:
        source = default_source
    try:
        expected = ExpectedResponseType(payload.get("expected_response_type") or ExpectedResponseType.UNKNOWN.value)
    except ValueError:
        expected = ExpectedResponseType.UNKNOWN
    return GeneratedQuestion(
        text=text,
        source=source,
        purpose=str(payload.get("purpose") or "reminder_follow_up"),
        confidence=float(payload.get("confidence") or 1.0),
        should_ask=bool(payload.get("should_ask", True)),
        expected_response_type=expected,
    )


def _response_type(value: Any) -> ResponseType:
    try:
        return ResponseType(str(value))
    except ValueError:
        return ResponseType.REMINDER_ACTION


def _optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _optional_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
