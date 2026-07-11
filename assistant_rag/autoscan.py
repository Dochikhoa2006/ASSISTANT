"""SQL-only reminder autoscan and reply context restoration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging

from .database import AssistantRepository
from .reminder_timing import ReminderTimingPlanner


logger = logging.getLogger(__name__)


@dataclass
class ReminderAutoscan:
    repository: AssistantRepository
    timing_planner: ReminderTimingPlanner | None = None

    def plan_pending_timing(
        self, *, limit: int = 100, now: datetime | None = None
    ) -> dict[str, int]:
        """Use the timing LLM exactly once for each newly pending reminder."""
        now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        planned = 0
        needs_review = 0
        for reminder in self.repository.list_reminders_requiring_timing(limit=limit):
            if reminder.source_time == datetime.min.replace(tzinfo=timezone.utc):
                if self.repository.fail_reminder_timing_plan(
                    reminder_id=reminder.reminder_id, user_id=reminder.user_id,
                    expected_version=reminder.version,
                    reason="The reminder timestamp is invalid and cannot be planned safely.",
                ):
                    needs_review += 1
                continue
            decision = self.timing_planner.plan(
                event_time=reminder.source_time,
                subject=reminder.subject or "event",
                raw_query=reminder.raw_reminder or reminder.subject,
                user_timezone=reminder.user_timezone,
                now=now,
            ) if self.timing_planner else None
            if decision is None or decision.needs_clarification or decision.notification_time is None:
                reason = decision.reason if decision else "Reminder timing LLM is not configured."
                # A restarted service can discover a reminder whose source
                # time is already past. The LLM was still consulted, but a
                # clarification can no longer arrive in time. Persist an
                # immediate fire time so this important reminder is late, not
                # silently missed. Future ambiguous reminders remain review.
                if reminder.source_time <= now and self.timing_planner is not None:
                    if self.repository.complete_reminder_timing_plan(
                        reminder_id=reminder.reminder_id, user_id=reminder.user_id,
                        expected_version=reminder.version, notification_time=now,
                        reason=f"Late reminder fallback: {reason}",
                    ):
                        planned += 1
                    continue
                if self.repository.fail_reminder_timing_plan(
                    reminder_id=reminder.reminder_id, user_id=reminder.user_id,
                    expected_version=reminder.version, reason=reason,
                ):
                    needs_review += 1
                continue
            if self.repository.complete_reminder_timing_plan(
                reminder_id=reminder.reminder_id, user_id=reminder.user_id,
                expected_version=reminder.version,
                notification_time=decision.notification_time, reason=decision.reason,
            ):
                planned += 1
        if planned or needs_review:
            logger.info("Reminder timing autoscan: planned=%s needs_review=%s", planned, needs_review)
        return {"planned": planned, "needs_review": needs_review}

    def scan_due(self, *, now_value: str, limit: int = 100) -> list[str]:
        scan_now = _parse_scan_time(now_value)
        self.plan_pending_timing(limit=limit, now=scan_now)
        return self.repository.scan_due_reminders(now_value=now_value, limit=limit)

    def catch_up_due(self, *, now_value: str, batch_size: int = 100) -> dict[str, int]:
        """Drain every pending plan and every due reminder in timestamp order.

        This is appropriate when an application process comes back online. Each
        repository scan is ordered oldest-first, so overdue reminders are
        created durably from farthest overdue to nearest overdue. The database
        uniqueness rule makes the operation safe to repeat after a restart.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        totals = {"planned": 0, "needs_review": 0, "notified": 0, "batches": 0}
        scan_now = _parse_scan_time(now_value)
        while True:
            timing = self.plan_pending_timing(limit=batch_size, now=scan_now)
            notified = self.repository.scan_due_reminders(now_value=now_value, limit=batch_size)
            totals["planned"] += timing["planned"]
            totals["needs_review"] += timing["needs_review"]
            totals["notified"] += len(notified)
            totals["batches"] += 1
            # Each successful plan/review transition removes a row from the
            # pending set; each notification removes a due one-time row or
            # advances a recurring row. If none changed, another process owns
            # the remaining work or the backlog is clear.
            if not timing["planned"] and not timing["needs_review"] and not notified:
                break
        if totals["planned"] or totals["needs_review"] or totals["notified"]:
            logger.info("Reminder autoscan catch-up: %s", totals)
        return totals


def _parse_scan_time(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return (parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)
    except (AttributeError, ValueError):
        return datetime.now(timezone.utc)


@dataclass
class ReminderReplyLoader:
    repository: AssistantRepository

    def load_context(
        self, *, user_id: str, reminder_id: str, notification_id: str
    ) -> dict[str, str | None]:
        return self.repository.load_reminder_reply_context(
            user_id=user_id,
            reminder_id=reminder_id,
            notification_id=notification_id,
        )
