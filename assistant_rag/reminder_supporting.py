"""Autoscan planning for one useful reminder supporting question.

The online reminder mutation pipeline persists reminder facts only.  This
module lets autoscan decide, from those facts alone, whether one optional
next-step question would materially help the user prepare for the reminder.
It never changes reminder timing and never receives a raw query or chat
history.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from math import isfinite

from .llm import LLMClient, LLMTask, is_structured_fallback


@dataclass(frozen=True)
class ReminderSupportingQuestionDecision:
    question: str | None
    confidence: float
    reason: str
    needs_review: bool = False


@dataclass
class ReminderSupportingQuestionPlanner:
    """Choose at most one concrete, high-confidence preparation question."""

    llm: LLMClient | None
    min_confidence: float = 0.85
    max_question_length: int = 220

    def plan(
        self,
        *,
        subject: str,
        reminder_summary: str,
        reminder_context: str,
        notification_time: datetime | None,
        event_time: datetime | None,
        user_timezone: str,
        recurrence_rule: str | None,
        now: datetime | None = None,
    ) -> ReminderSupportingQuestionDecision:
        if self.llm is None:
            return ReminderSupportingQuestionDecision(
                None,
                0.0,
                "Reminder supporting-question LLM is not configured.",
                True,
            )

        now_utc = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        schema = {
            "type": "object",
            "required": [
                "should_ask",
                "question_text",
                "confidence",
            ],
            "additionalProperties": False,
            "properties": {
                "should_ask": {"type": "boolean"},
                "question_text": {"type": "string"},
                "confidence": {"type": "number"},
            },
        }
        reminder_facts = {
            "subject": _bounded_text(subject),
            "reminder_summary": _bounded_text(reminder_summary),
            "reminder_context": _bounded_text(reminder_context),
            "notification_time_utc": _iso_or_none(notification_time),
            "event_time_utc": _iso_or_none(event_time),
            "minutes_until_notification": _minutes_until(
                notification_time,
                now_utc,
            ),
            "minutes_until_event": _minutes_until(event_time, now_utc),
            "user_timezone": _bounded_text(user_timezone, max_length=100) or "UTC",
            "recurrence_rule": _bounded_text(recurrence_rule, max_length=300) or None,
            "current_time_utc": now_utc.isoformat(),
        }
        try:
            payload = self.llm.generate_json(
                task=LLMTask.ACTION_PLANNING,
                system_prompt=(
                    "You decide whether one optional supporting question would materially help "
                    "a user prepare for a persisted reminder. Use only the structured reminder "
                    "facts supplied; no user query or chat history is available or authoritative. "
                    "Predict the single most likely useful next assistance from concrete reminder "
                    "content. For example, a sufficiently specific meeting may benefit from an "
                    "agenda, briefing PDF, presentation, spreadsheet tracker, or preparation plan; "
                    "choose only options genuinely supported by that reminder. Other reminders may "
                    "suggest a checklist, comparison, draft, itinerary, or no assistance. Ask one "
                    "concise question addressed to the user and offer no more than three closely "
                    "related choices. Do not ask for missing reminder fields, scheduling details, "
                    "lead time, recurrence, confirmation, or sensitive personal information. Do "
                    "not execute work or claim that an artifact already exists. Set should_ask=false "
                    "when the likely need is generic, weak, intrusive, unsafe, or not grounded in "
                    "the reminder. A blank question is required when should_ask=false."
                ),
                user_prompt=json.dumps(
                    {"reminder_facts": reminder_facts},
                    ensure_ascii=False,
                ),
                schema=schema,
            )
        except Exception as exc:
            return ReminderSupportingQuestionDecision(
                None,
                0.0,
                f"Supporting-question planner unavailable: {type(exc).__name__}.",
                True,
            )

        if is_structured_fallback(payload):
            return ReminderSupportingQuestionDecision(
                None,
                0.0,
                payload.reason,
                True,
            )

        should_ask = payload.get("should_ask")
        raw_confidence = payload.get("confidence")
        question = " ".join(str(payload.get("question_text") or "").split())
        if not isinstance(should_ask, bool):
            return ReminderSupportingQuestionDecision(
                None, 0.0, "The supporting-question decision was not boolean.", True
            )
        if isinstance(raw_confidence, bool):
            return ReminderSupportingQuestionDecision(
                None, 0.0, "The supporting-question confidence was invalid.", True
            )
        try:
            confidence = float(raw_confidence)
        except (TypeError, ValueError):
            return ReminderSupportingQuestionDecision(
                None, 0.0, "The supporting-question confidence was invalid.", True
            )
        if not isfinite(confidence):
            return ReminderSupportingQuestionDecision(
                None, 0.0, "The supporting-question confidence was not finite.", True
            )
        confidence = max(0.0, min(1.0, confidence))

        if not should_ask:
            if question:
                return ReminderSupportingQuestionDecision(
                    None,
                    confidence,
                    "The no-question decision unexpectedly contained question text.",
                    True,
                )
            # The structured-output safe fallback has this exact minimal
            # shape.  Treat it as an unavailable planner, not as an intentional
            # high-confidence decision to ask nothing.
            if confidence == 0.0:
                return ReminderSupportingQuestionDecision(
                    None,
                    confidence,
                    "The supporting-question planner returned no usable decision.",
                    True,
                )
            return ReminderSupportingQuestionDecision(
                None,
                confidence,
                "No sufficiently specific supporting need was inferred.",
            )
        if confidence < self.min_confidence:
            return ReminderSupportingQuestionDecision(
                None,
                confidence,
                "Supporting-question confidence was below the required threshold.",
            )
        if (
            not question
            or len(question) > self.max_question_length
            or not question.endswith("?")
            or question.count("?") != 1
        ):
            return ReminderSupportingQuestionDecision(
                None,
                confidence,
                "The proposed supporting question did not satisfy the one-question contract.",
                True,
            )
        return ReminderSupportingQuestionDecision(
            question,
            confidence,
            "One grounded preparation need was identified.",
        )


def _bounded_text(value: str | None, *, max_length: int = 4_000) -> str:
    text = str(value or "").strip()
    return text if len(text) <= max_length else text[:max_length].rstrip()


def _iso_or_none(value: datetime | None) -> str | None:
    if value is None:
        return None
    aware = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return aware.astimezone(timezone.utc).isoformat()


def _minutes_until(value: datetime | None, now_utc: datetime) -> int | None:
    if value is None:
        return None
    aware = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return int(
        (aware.astimezone(timezone.utc) - now_utc).total_seconds() // 60
    )
