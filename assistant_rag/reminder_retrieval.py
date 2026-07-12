"""Canonical SQL reminder-candidate retrieval shared by all branches."""

from __future__ import annotations

from typing import Any

from .contracts import ReminderCandidateSummary
from .database import AssistantRepository


def retrieve_reminder_candidates(
    *,
    repository: AssistantRepository,
    user_id: str,
    statuses: tuple[str, ...],
    candidate_limit: int,
) -> list[ReminderCandidateSummary]:
    """Load the same bounded, user-owned reminder candidate set everywhere."""
    return repository.list_reminder_candidates(
        user_id=user_id,
        statuses=statuses,
        time_window=None,
        limit=candidate_limit,
    )


def reminder_candidate_to_context(
    *, user_id: str, candidate: ReminderCandidateSummary
) -> dict[str, Any]:
    """Expose a canonical candidate as validated general-answer context."""
    return {
        "reminder_id": candidate.reminder_id,
        "user_id": user_id,
        "subject": candidate.subject,
        "reminder_summary": candidate.reminder_summary,
        "raw_reminder": candidate.raw_reminder,
        "reminder_time": candidate.reminder_time.isoformat() if candidate.reminder_time else None,
        "status": candidate.status,
        "created_at": candidate.created_at.isoformat() if candidate.created_at else None,
        "updated_at": candidate.updated_at.isoformat() if candidate.updated_at else None,
        "version": candidate.version,
        "is_deleted": candidate.is_deleted,
        "confidence": 1.0,
    }
