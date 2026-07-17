"""Deterministic recurrence helpers for reminders."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import calendar
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .contracts import RecurrenceFrequency


@dataclass(frozen=True)
class RecurrenceRule:
    frequency: RecurrenceFrequency
    count: int | None = None
    until: datetime | None = None


def normalize_recurrence_rule(raw_rule: str | None) -> RecurrenceRule | None:
    if not raw_rule:
        return None
    text = raw_rule.strip()
    lowered = text.casefold()
    if lowered in {"daily", "every day"}:
        return RecurrenceRule(RecurrenceFrequency.DAILY)
    if lowered in {"weekly", "every week"}:
        return RecurrenceRule(RecurrenceFrequency.WEEKLY)
    if lowered in {"monthly", "every month"}:
        return RecurrenceRule(RecurrenceFrequency.MONTHLY)
    if lowered.startswith("freq="):
        parts = dict(
            part.split("=", 1)
            for part in lowered.replace("rrule:", "").split(";")
            if "=" in part
        )
        freq = parts.get("freq", "")
        if freq in {"daily", "weekly", "monthly"}:
            count = int(parts["count"]) if parts.get("count", "").isdigit() else None
            if count is not None and count < 1:
                raise ValueError("Recurrence COUNT must be a positive integer.")
            until = _parse_until(parts.get("until"))
            return RecurrenceRule(RecurrenceFrequency(freq), count=count, until=until)
    if lowered.startswith("rrule:"):
        return normalize_recurrence_rule(lowered.removeprefix("rrule:"))
    raise ValueError("Unsupported recurrence rule. Use daily, weekly, monthly, or FREQ=DAILY/WEEKLY/MONTHLY.")


def calculate_next_fire_time(
    *,
    previous_fire_time: str | datetime,
    recurrence_rule: str,
    recurrence_timezone: str,
    completed_occurrences: int = 1,
) -> str | None:
    rule = normalize_recurrence_rule(recurrence_rule)
    if rule is None:
        return None
    if completed_occurrences < 1:
        raise ValueError("completed_occurrences must include the current fire.")
    # RFC-style COUNT includes the initial occurrence. This helper runs after
    # the current occurrence has fired, so reaching COUNT means the series is
    # exhausted and must not receive another next_fire_time.
    if rule.count is not None and completed_occurrences >= rule.count:
        return None
    zone = _zone(recurrence_timezone)
    previous_utc = _coerce_aware(previous_fire_time)
    local_previous = previous_utc.astimezone(zone)
    if rule.frequency is RecurrenceFrequency.DAILY:
        local_next = local_previous.replace(day=local_previous.day) + _days(1)
    elif rule.frequency is RecurrenceFrequency.WEEKLY:
        local_next = local_previous + _days(7)
    elif rule.frequency is RecurrenceFrequency.MONTHLY:
        local_next = _add_month(local_previous)
    else:
        raise ValueError(f"Unsupported recurrence frequency: {rule.frequency.value}")
    if rule.until and local_next.astimezone(timezone.utc) > rule.until.astimezone(timezone.utc):
        return None
    return local_next.astimezone(timezone.utc).isoformat()


def first_fire_time(
    *,
    reminder_time: str | datetime,
    recurrence_timezone: str,
) -> str:
    _zone(recurrence_timezone)
    return _coerce_aware(reminder_time).astimezone(timezone.utc).isoformat()


def _coerce_aware(value: str | datetime) -> datetime:
    dt = datetime.fromisoformat(value) if isinstance(value, str) else value
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _zone(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"Invalid recurrence timezone: {name}") from exc


def _add_month(value: datetime) -> datetime:
    year = value.year + (value.month // 12)
    month = 1 if value.month == 12 else value.month + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return value.replace(year=year, month=month, day=day)


def _days(amount: int):
    from datetime import timedelta

    return timedelta(days=amount)


def _parse_until(raw: str | None) -> datetime | None:
    if not raw:
        return None
    value = raw.strip()
    if value.endswith("z"):
        value = value[:-1] + "+00:00"
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed
