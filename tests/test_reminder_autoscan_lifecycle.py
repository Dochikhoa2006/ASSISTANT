from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Iterator

import pytest

from assistant_rag.autoscan import ReminderAutoscan
from assistant_rag.contracts import Intent, ResponseType
from assistant_rag.database import SQLiteRepository
from assistant_rag.postgres_repository import PostgresRepository
from assistant_rag.recurrence import calculate_next_fire_time


USER_ID = "autoscan-lifecycle-user"
FIRST_FIRE = datetime(2035, 1, 1, 9, 0, tzinfo=timezone.utc)


@pytest.fixture(params=("sqlite", "sqlalchemy"))
def repository(request: pytest.FixtureRequest) -> Iterator[object]:
    if request.param == "sqlite":
        value = SQLiteRepository.in_memory()
    else:
        # Exercise the SQLAlchemy/PostgreSQL repository implementation without
        # requiring an external server in the unit-test environment.
        value = PostgresRepository.create("sqlite+pysqlite:///:memory:")
    value.initialize_schema()
    try:
        yield value
    finally:
        if hasattr(value, "close"):
            value.close()
        else:
            value.connection.close()


def _seed_planned_reminder(
    repository: object,
    *,
    fire_time: datetime = FIRST_FIRE,
    recurrence_rule: str | None = None,
) -> str:
    with repository.transaction() as cursor:
        topic_id = repository.ensure_topic(
            cursor,
            user_id=USER_ID,
            title="Autoscan lifecycle",
        )
        hop = repository.append_conversation_hop(
            cursor,
            topic_id=topic_id,
            user_id=USER_ID,
            intent=Intent.REMINDER.value,
            raw_user_query="Create the autoscan reminder.",
            rewritten_user_query="Create the autoscan reminder.",
            raw_response="The autoscan reminder was created.",
            response_type=ResponseType.REMINDER_ACTION.value,
        )
        reminder_id = repository.add_reminder(
            cursor,
            user_id=USER_ID,
            source_topic_id=topic_id,
            source_hop_id=hop.hop_id,
            reminder_time=fire_time.isoformat(),
            event_time=fire_time.isoformat(),
            raw_reminder="Autoscan lifecycle reminder",
            reminder_summary="Autoscan lifecycle reminder",
            subject="Autoscan lifecycle reminder",
            user_timezone="UTC",
            original_time_text=fire_time.isoformat(),
            recurrence_rule=recurrence_rule,
            recurrence_timezone="UTC" if recurrence_rule else None,
        )
    assert repository.complete_reminder_timing_plan(
        reminder_id=reminder_id,
        user_id=USER_ID,
        expected_version=1,
        notification_time=fire_time,
        reason="Explicit autoscan lifecycle test plan.",
    )
    return reminder_id


def _row(repository: object, reminder_id: str) -> dict:
    return next(
        row
        for row in repository.list_reminders(user_id=USER_ID)
        if row["reminder_id"] == reminder_id
    )


def test_one_time_reminder_becomes_permanently_notified(repository: object) -> None:
    reminder_id = _seed_planned_reminder(repository)
    autoscan = ReminderAutoscan(repository=repository)

    assert autoscan.scan_due(now_value=FIRST_FIRE.isoformat()) == [reminder_id]

    fired = _row(repository, reminder_id)
    assert fired["status"] == "notified"
    assert fired["last_fire_time"] == FIRST_FIRE.isoformat()
    assert fired["next_fire_time"] is None
    assert len(repository.list_notifications(user_id=USER_ID)) == 1

    assert autoscan.scan_due(
        now_value=(FIRST_FIRE + timedelta(days=365)).isoformat()
    ) == []
    unchanged = _row(repository, reminder_id)
    assert unchanged["status"] == "notified"
    assert unchanged["last_fire_time"] == FIRST_FIRE.isoformat()
    assert unchanged["next_fire_time"] is None
    assert len(repository.list_notifications(user_id=USER_ID)) == 1


def test_unbounded_recurring_reminder_stays_scheduled_and_advances(
    repository: object,
) -> None:
    reminder_id = _seed_planned_reminder(
        repository,
        recurrence_rule="FREQ=WEEKLY",
    )
    autoscan = ReminderAutoscan(repository=repository)

    assert autoscan.scan_due(now_value=FIRST_FIRE.isoformat()) == [reminder_id]
    first = _row(repository, reminder_id)
    second_fire = calculate_next_fire_time(
        previous_fire_time=FIRST_FIRE,
        recurrence_rule="FREQ=WEEKLY",
        recurrence_timezone="UTC",
        completed_occurrences=1,
    )
    assert second_fire is not None
    assert first["status"] == "scheduled"
    assert first["last_fire_time"] == FIRST_FIRE.isoformat()
    assert first["next_fire_time"] == second_fire

    assert autoscan.scan_due(now_value=second_fire) == [reminder_id]
    second = _row(repository, reminder_id)
    third_fire = calculate_next_fire_time(
        previous_fire_time=second_fire,
        recurrence_rule="FREQ=WEEKLY",
        recurrence_timezone="UTC",
        completed_occurrences=2,
    )
    assert second["status"] == "scheduled"
    assert second["last_fire_time"] == second_fire
    assert second["next_fire_time"] == third_fire
    assert len(repository.list_notifications(user_id=USER_ID)) == 2


@pytest.mark.parametrize(
    "recurrence_rule",
    (
        "FREQ=DAILY;COUNT=5",
        "FREQ=DAILY;UNTIL=2035-01-05T09:00:00+00:00",
    ),
)
def test_exhausted_recurring_reminder_completes_and_clears_next_fire(
    repository: object,
    recurrence_rule: str,
) -> None:
    reminder_id = _seed_planned_reminder(
        repository,
        recurrence_rule=recurrence_rule,
    )
    autoscan = ReminderAutoscan(repository=repository)

    totals = autoscan.catch_up_due(
        now_value=(FIRST_FIRE + timedelta(days=9)).isoformat(),
        batch_size=2,
    )

    assert totals["notified"] == 5
    exhausted = _row(repository, reminder_id)
    assert exhausted["status"] == "completed"
    assert exhausted["last_fire_time"] == (
        FIRST_FIRE + timedelta(days=4)
    ).isoformat()
    assert exhausted["next_fire_time"] is None
    assert len(repository.list_notifications(user_id=USER_ID)) == 5

    repeated = autoscan.catch_up_due(
        now_value=(FIRST_FIRE + timedelta(days=30)).isoformat(),
        batch_size=2,
    )
    assert repeated["notified"] == 0
    assert len(repository.list_notifications(user_id=USER_ID)) == 5
