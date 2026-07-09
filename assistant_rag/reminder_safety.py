from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
import re
import unicodedata


def normalize_subject(text: str | None) -> str:
    if not text:
        return ""
    value = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
    value = re.sub(r"[^\w\s]", " ", value.casefold())
    return " ".join(value.split())


def token_similarity(left: str | None, right: str | None) -> float:
    left_tokens = set(normalize_subject(left).split())
    right_tokens = set(normalize_subject(right).split())
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def utc_minute(value: datetime) -> datetime:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    value = value.astimezone(timezone.utc)
    return value.replace(second=0, microsecond=0)


@dataclass(frozen=True)
class NormalizedReminderTime:
    reminder_time_utc: datetime
    user_timezone: str
    original_time_text: str | None
    warning: str | None = None


class ReminderTimeNormalizationError(ValueError):
    pass


@dataclass(frozen=True)
class ReminderTimeNormalizer:
    default_timezone: str = "UTC"

    def normalize(
        self,
        value: datetime,
        *,
        platform_timezone: str | None,
        original_time_text: str | None,
    ) -> NormalizedReminderTime:
        timezone_name = platform_timezone or self.default_timezone
        try:
            user_tz = ZoneInfo(timezone_name)
        except ZoneInfoNotFoundError as exc:
            raise ReminderTimeNormalizationError(f"Invalid timezone: {timezone_name}") from exc

        warning = None
        if not platform_timezone:
            warning = f"Timezone was missing; used default timezone {self.default_timezone}."

        if value.tzinfo is None:
            local_value = value.replace(tzinfo=user_tz)
        else:
            local_value = value.astimezone(user_tz)

        return NormalizedReminderTime(
            reminder_time_utc=local_value.astimezone(timezone.utc),
            user_timezone=timezone_name,
            original_time_text=original_time_text,
            warning=warning,
        )


def within_minutes(left: datetime, right: datetime, minutes: int) -> bool:
    return abs(utc_minute(left) - utc_minute(right)) <= timedelta(minutes=minutes)
