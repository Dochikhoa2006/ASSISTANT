"""Exact reminder-notification reply context and Last-QA restoration."""

from __future__ import annotations

import json
from typing import Any

from .contracts import (
    ExpectedResponseType,
    GeneratedQuestion,
    LastQAState,
    QuestionSource,
    ResponseType,
)


def reminder_notification_key(reminder_id: str, notification_id: str) -> tuple[str, str]:
    """Collision-free dictionary key for recurring reminder notifications."""
    return (str(reminder_id), str(notification_id))


def build_reminder_reply_context_index(
    *, repository: Any, user_id: str, notifications: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Hash every visible notification to the authoritative source conversation hop."""
    index: dict[tuple[str, str], dict[str, Any]] = {}
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


def build_reminder_reply_last_qa(context: dict[str, Any]) -> LastQAState:
    """Rebuild Last-QA from the reminder's source hop, never from UI text."""
    supporting_questions = _questions_from_json(context.get("supporting_questions_json"))
    reminder_question = _question_from_value(context.get("supporting_question"), QuestionSource.REMINDER_SUPPORTING_QUESTION)
    response_type = _response_type(context.get("source_response_type"))
    return LastQAState(
        last_user_query=str(context.get("source_rewritten_user_query") or context.get("source_raw_user_query") or ""),
        last_response=str(context.get("source_raw_response") or context.get("supporting_response") or ""),
        response_type=response_type,
        supporting_questions=supporting_questions,
        reminder_supporting_question=reminder_question,
        linked_topic_id=_optional_text(context.get("source_topic_id") or context.get("topic_id")),
        linked_hop_id=_optional_text(context.get("source_hop_id")),
    )


def build_reminder_reply_metadata(
    *, reminder_id: str, notification_id: str, context: dict[str, Any],
) -> dict[str, Any]:
    """Pass immutable source identifiers to the normal pipeline."""
    return {
        "reminder_reply_context": True,
        "reminder_id": str(reminder_id),
        "notification_id": str(notification_id),
        "source_topic_id": _optional_text(context.get("source_topic_id") or context.get("topic_id")),
        "source_hop_id": _optional_text(context.get("source_hop_id")),
        "supporting_question": _question_text(context),
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
