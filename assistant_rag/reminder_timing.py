"""Conservative LLM planning for event reminders.

The event time and the time at which a notification is delivered are different
facts.  This module lets an LLM select a lead time for an event only when the
user did not explicitly specify the notification time.  The application, not
the model, calculates the final timestamp from that lead time.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json

from .llm import LLMClient, LLMTask


@dataclass(frozen=True)
class ReminderTimingDecision:
    notification_time: datetime | None
    confidence: float
    reason: str
    needs_clarification: bool = False


@dataclass
class ReminderTimingPlanner:
    """Choose a notification lead time without allowing timestamp invention."""

    llm: LLMClient | None
    min_confidence: float = 0.90
    max_lead_minutes: int = 7 * 24 * 60

    def plan(
        self,
        *,
        event_time: datetime,
        subject: str,
        raw_query: str,
        user_timezone: str,
        now: datetime | None = None,
    ) -> ReminderTimingDecision:
        if self.llm is None:
            return ReminderTimingDecision(None, 0.0, "Reminder timing LLM is not configured.", True)

        now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        event_time = event_time.astimezone(timezone.utc)
        minutes_until_event = int((event_time - now).total_seconds() // 60)
        if minutes_until_event <= 0:
            return ReminderTimingDecision(
                None, 0.0, "The event time is no longer in the future.", True
            )

        schema = {
            "type": "object",
            "required": ["lead_minutes", "confidence", "needs_clarification", "reason"],
            "properties": {
                "lead_minutes": {"type": "integer", "minimum": 0, "maximum": self.max_lead_minutes},
                "confidence": {"type": "number"},
                "needs_clarification": {"type": "boolean"},
                "reason": {"type": "string"},
            },
        }
        try:
            payload = self.llm.generate_json(
                task=LLMTask.ACTION_PLANNING,
                system_prompt=(
                    "You plan the lead time for a reminder about an event. The user did not state a "
                    "notification time, only an event time. Choose a conservative lead time in whole "
                    "minutes based only on the stated context. Do not choose a date or clock time; the "
                    "application calculates it from your lead_minutes. For meetings, choose a practical "
                    "lead such as 15, 30, or 60 minutes when context supports it. For all-day or distant "
                    "events, a day-before reminder can be appropriate. If the stakes, timezone, event "
                    "meaning, or timing preference are unclear, set needs_clarification=true. Never change "
                    "an explicit user-selected notification time; such requests do not reach this planner."
                ),
                user_prompt=json.dumps(
                    {
                        "user_query": raw_query,
                        "subject": subject,
                        "event_time_utc": event_time.isoformat(),
                        "user_timezone": user_timezone,
                        "current_time_utc": now.isoformat(),
                        "minutes_until_event": minutes_until_event,
                    },
                    ensure_ascii=False,
                ),
                schema=schema,
            )
            confidence = max(0.0, min(1.0, float(payload.get("confidence", 0.0))))
            lead_minutes = int(payload.get("lead_minutes", -1))
            needs_clarification = bool(payload.get("needs_clarification", False))
            reason = str(payload.get("reason") or "").strip()
        except Exception as exc:
            return ReminderTimingDecision(None, 0.0, f"Timing planner unavailable: {type(exc).__name__}.", True)

        if needs_clarification or confidence < self.min_confidence:
            return ReminderTimingDecision(None, confidence, reason or "Timing confidence is too low.", True)
        if lead_minutes < 0 or lead_minutes > self.max_lead_minutes:
            return ReminderTimingDecision(None, confidence, "The timing lead was outside the allowed range.", True)

        # Do not schedule in the past. A near event is notified immediately;
        # the model still cannot alter its actual event timestamp.
        lead_minutes = min(lead_minutes, minutes_until_event)
        return ReminderTimingDecision(
            notification_time=event_time - timedelta(minutes=lead_minutes),
            confidence=confidence,
            reason=reason or f"Notify {lead_minutes} minutes before the event.",
        )
