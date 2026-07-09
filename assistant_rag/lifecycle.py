"""Central lifecycle rules for SQL reads, retrieval, and indexability."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


ACTIVE_REMINDER_CANDIDATE_STATUSES = ("scheduled", "notified")
TERMINAL_REMINDER_STATUSES = ("dismissed", "completed", "cancelled")


def is_active_knowledge_chunk(row: dict[str, Any]) -> bool:
    return not bool(row.get("is_deleted"))


def is_indexable_knowledge_chunk(row: dict[str, Any]) -> bool:
    return is_active_knowledge_chunk(row)


def is_retrievable_conversation_topic(row: dict[str, Any]) -> bool:
    return str(row.get("status", "active")) == "active"


def is_indexable_conversation_hop(row: dict[str, Any]) -> bool:
    return str(row.get("topic_status", row.get("status", "active"))) == "active"


def is_artifact_downloadable(row: dict[str, Any], *, now_value: str | None = None) -> bool:
    if str(row.get("status")) != "created":
        return False
    expires_at = row.get("expires_at")
    if not expires_at:
        return True
    now = datetime.fromisoformat(now_value) if now_value else datetime.now(timezone.utc)
    expires = datetime.fromisoformat(str(expires_at))
    if expires.tzinfo is None:
        expires = expires.replace(tzinfo=timezone.utc)
    return expires > now


def confirmation_can_execute(row: dict[str, Any], *, now_value: str) -> bool:
    if str(row.get("status")) != "pending":
        return False
    expires = datetime.fromisoformat(str(row["expires_at"]))
    now = datetime.fromisoformat(now_value)
    if expires.tzinfo is None:
        expires = expires.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return expires > now


def reminder_is_active_candidate(row: dict[str, Any], *, allowed_statuses: tuple[str, ...] = ACTIVE_REMINDER_CANDIDATE_STATUSES) -> bool:
    return str(row.get("status")) in allowed_statuses
