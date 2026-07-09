"""SQL-only reminder autoscan and reply context restoration."""

from __future__ import annotations

from dataclasses import dataclass
import sqlite3

from .database import SQLRepository, now_iso


@dataclass
class ReminderAutoscan:
    repository: SQLRepository

    def scan_due(self, *, now_value: str) -> list[str]:
        notified: list[str] = []
        with self.repository.transaction() as cursor:
            rows = cursor.execute(
                """
                SELECT reminder_id, user_id
                FROM reminders
                WHERE status = 'scheduled' AND reminder_time <= ?
                ORDER BY reminder_time
                """,
                (now_value,),
            ).fetchall()
            for row in rows:
                self.repository.create_notification_if_absent(
                    cursor,
                    user_id=row["user_id"],
                    reminder_id=row["reminder_id"],
                )
                self.repository.update_reminder_status(
                    cursor,
                    user_id=row["user_id"],
                    reminder_id=row["reminder_id"],
                    status="notified",
                )
                notified.append(str(row["reminder_id"]))
        return notified


@dataclass
class ReminderReplyLoader:
    connection: sqlite3.Connection

    def load_context(
        self, *, user_id: str, reminder_id: str, notification_id: str
    ) -> dict[str, str | None]:
        self.connection.row_factory = sqlite3.Row
        row = self.connection.execute(
            """
            SELECT r.source_topic_id, r.source_hop_id
            FROM reminders r
            JOIN reminder_notifications n ON n.reminder_id = r.reminder_id
            WHERE r.user_id = ? AND r.reminder_id = ? AND n.notification_id = ?
            """,
            (user_id, reminder_id, notification_id),
        ).fetchone()
        if not row:
            raise ValueError("Reminder reply context not found for user")
        return {
            "source_topic_id": row["source_topic_id"],
            "source_hop_id": row["source_hop_id"],
            "loaded_at": now_iso(),
        }

