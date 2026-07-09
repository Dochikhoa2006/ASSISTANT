"""SQL-only reminder autoscan and reply context restoration."""

from __future__ import annotations

from dataclasses import dataclass

from .database import AssistantRepository, now_iso


@dataclass
class ReminderAutoscan:
    repository: AssistantRepository

    def scan_due(self, *, now_value: str, limit: int = 100) -> list[str]:
        return self.repository.scan_due_reminders(now_value=now_value, limit=limit)


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
