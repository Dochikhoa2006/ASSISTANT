from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from typing import Any, Iterator

import pytest

from assistant_rag.autoscan import ReminderAutoscan
from assistant_rag.contracts import Intent, ResponseType
from assistant_rag.database import SQLiteRepository
from assistant_rag.postgres_repository import PostgresRepository
from assistant_rag.reminder_supporting import ReminderSupportingQuestionPlanner


USER_ID = "autoscan-supporting-question-user"
FIRE_TIME = datetime(2035, 6, 10, 9, 0, tzinfo=timezone.utc)
MEETING_QUESTION = (
    "Would you like a meeting agenda, a briefing PDF, or an action tracker?"
)


class ScriptedLLM:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def generate_json(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(dict(kwargs))
        if not self.responses:
            raise AssertionError("unexpected extra supporting-question LLM call")
        return self.responses.pop(0)

    def chat(self, **_kwargs: Any) -> str:
        raise AssertionError("supporting-question planning must use strict JSON")


@pytest.fixture(params=("sqlite", "sqlalchemy"))
def repository(request: pytest.FixtureRequest) -> Iterator[Any]:
    if request.param == "sqlite":
        value = SQLiteRepository.in_memory()
    else:
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
    repository: Any,
    *,
    subject: str = "Quarterly planning meeting",
    summary: str = "Review milestones, risks, and owners for next quarter.",
    raw_reminder: str = "Prepare for the quarterly planning meeting.",
    fire_time: datetime = FIRE_TIME,
    existing_question: str | None = None,
) -> str:
    with repository.transaction() as cursor:
        topic_id = repository.ensure_topic(
            cursor,
            user_id=USER_ID,
            title=f"Supporting question source {subject} {fire_time.isoformat()}",
        )
        hop = repository.append_conversation_hop(
            cursor,
            topic_id=topic_id,
            user_id=USER_ID,
            intent=Intent.REMINDER.value,
            raw_user_query="RAW_QUERY_MUST_NOT_REACH_AUTOSCAN",
            rewritten_user_query="REWRITTEN_QUERY_MUST_NOT_REACH_AUTOSCAN",
            raw_response="The reminder was created.",
            response_type=ResponseType.REMINDER_ACTION.value,
        )
        reminder_id = repository.add_reminder(
            cursor,
            user_id=USER_ID,
            source_topic_id=topic_id,
            source_hop_id=hop.hop_id,
            reminder_time=fire_time.isoformat(),
            event_time=fire_time.isoformat(),
            raw_reminder=raw_reminder,
            reminder_summary=summary,
            subject=subject,
            supporting_question=existing_question,
            user_timezone="UTC",
            original_time_text=fire_time.isoformat(),
        )
    assert repository.complete_reminder_timing_plan(
        reminder_id=reminder_id,
        user_id=USER_ID,
        expected_version=1,
        notification_time=fire_time,
        reason="Explicit notification time for supporting-question test.",
    )
    return reminder_id


def _row(repository: Any, reminder_id: str) -> dict[str, Any]:
    return next(
        row
        for row in repository.list_reminders(user_id=USER_ID)
        if row["reminder_id"] == reminder_id
    )


def _planner(response: dict[str, Any]) -> tuple[ReminderSupportingQuestionPlanner, ScriptedLLM]:
    llm = ScriptedLLM([response])
    return ReminderSupportingQuestionPlanner(llm=llm), llm


def test_planner_receives_only_bounded_persisted_reminder_facts() -> None:
    planner, llm = _planner(
        {
            "should_ask": True,
            "question_text": MEETING_QUESTION,
            "confidence": 0.97,
            "reason_summary": "The meeting has concrete preparation deliverables.",
        }
    )

    decision = planner.plan(
        subject="Quarterly planning meeting",
        reminder_summary="Review milestones and owners.",
        reminder_context="Prepare for the quarterly planning meeting.",
        notification_time=FIRE_TIME - timedelta(hours=1),
        event_time=FIRE_TIME,
        user_timezone="UTC",
        recurrence_rule=None,
        now=FIRE_TIME - timedelta(days=1),
    )

    assert decision.question == MEETING_QUESTION
    prompt = json.loads(llm.calls[0]["user_prompt"])
    assert set(prompt) == {"reminder_facts"}
    assert prompt["reminder_facts"]["subject"] == "Quarterly planning meeting"
    assert prompt["reminder_facts"]["minutes_until_notification"] == 1_380
    assert prompt["reminder_facts"]["minutes_until_event"] == 1_440
    serialized = json.dumps(prompt).casefold()
    assert "raw_query" not in serialized
    assert "rewritten_query" not in serialized
    assert "chat_history" not in serialized
    assert "conversation_history" not in serialized


def test_structured_llm_fallback_requires_review_instead_of_silent_completion() -> None:
    planner, _ = _planner(
        {
            "should_ask": False,
            "question_text": "",
            "confidence": 0.0,
            "reason_summary": (
                "structured fallback after invalid LLM JSON: malformed output"
            ),
        }
    )

    decision = planner.plan(
        subject="Quarterly planning meeting",
        reminder_summary="Review milestones and owners.",
        reminder_context="Prepare for the quarterly planning meeting.",
        notification_time=FIRE_TIME - timedelta(hours=1),
        event_time=FIRE_TIME,
        user_timezone="UTC",
        recurrence_rule=None,
        now=FIRE_TIME - timedelta(days=1),
    )

    assert decision.needs_review
    assert decision.question is None
    assert decision.confidence == 0.0


def test_autoscan_plans_one_question_once_and_marks_state(repository: Any) -> None:
    reminder_id = _seed_planned_reminder(repository)
    planner, llm = _planner(
        {
            "should_ask": True,
            "question_text": MEETING_QUESTION,
            "confidence": 0.97,
            "reason_summary": "A planning deliverable is likely useful.",
        }
    )
    autoscan = ReminderAutoscan(
        repository=repository,
        supporting_question_planner=planner,
    )

    assert autoscan.plan_pending_supporting_questions(
        now=FIRE_TIME - timedelta(days=1)
    ) == {"planned": 1, "needs_review": 0, "questions_created": 1}
    planned = _row(repository, reminder_id)
    assert planned["supporting_question_plan_status"] == "planned"
    assert planned["supporting_question"] == MEETING_QUESTION
    assert planned["supporting_question_confidence"] == pytest.approx(0.97)
    assert planned["supporting_question_planned_at"]
    assert planned["supporting_question_plan_reason"]
    assert repository.list_reminders_requiring_supporting_question() == []

    assert autoscan.plan_pending_supporting_questions(
        now=FIRE_TIME - timedelta(hours=12)
    ) == {"planned": 0, "needs_review": 0, "questions_created": 0}
    assert len(llm.calls) == 1


def test_low_confidence_prediction_is_planned_without_a_question(
    repository: Any,
) -> None:
    reminder_id = _seed_planned_reminder(
        repository,
        subject="Routine reminder",
        summary="Check the item.",
        raw_reminder="Remember the item.",
    )
    planner, _ = _planner(
        {
            "should_ask": True,
            "question_text": "Would you like me to prepare something for this?",
            "confidence": 0.41,
            "reason_summary": "The likely demand is too generic.",
        }
    )

    result = ReminderAutoscan(
        repository=repository,
        supporting_question_planner=planner,
    ).plan_pending_supporting_questions(now=FIRE_TIME - timedelta(days=1))

    assert result == {"planned": 1, "needs_review": 0, "questions_created": 0}
    row = _row(repository, reminder_id)
    assert row["supporting_question_plan_status"] == "planned"
    assert row["supporting_question"] is None
    assert row["supporting_question_confidence"] == pytest.approx(0.41)


def test_invalid_question_contract_is_marked_needs_review(repository: Any) -> None:
    reminder_id = _seed_planned_reminder(repository)
    planner, llm = _planner(
        {
            "should_ask": False,
            "question_text": "This contradictory question must not be persisted?",
            "confidence": 0.99,
            "reason_summary": "Contradictory output.",
        }
    )
    autoscan = ReminderAutoscan(
        repository=repository,
        supporting_question_planner=planner,
    )

    assert autoscan.plan_pending_supporting_questions() == {
        "planned": 0,
        "needs_review": 1,
        "questions_created": 0,
    }
    row = _row(repository, reminder_id)
    assert row["supporting_question_plan_status"] == "needs_review"
    assert row["supporting_question"] is None
    assert autoscan.plan_pending_supporting_questions() == {
        "planned": 0,
        "needs_review": 0,
        "questions_created": 0,
    }
    assert len(llm.calls) == 1


def test_limit_keeps_unprocessed_reminders_pending(repository: Any) -> None:
    first_id = _seed_planned_reminder(
        repository,
        subject="First meeting",
        fire_time=FIRE_TIME,
    )
    second_id = _seed_planned_reminder(
        repository,
        subject="Second meeting",
        fire_time=FIRE_TIME + timedelta(hours=1),
    )
    llm = ScriptedLLM(
        [
            {
                "should_ask": False,
                "question_text": "",
                "confidence": 0.95,
                "reason_summary": "No question needed.",
            }
        ]
    )

    result = ReminderAutoscan(
        repository=repository,
        supporting_question_planner=ReminderSupportingQuestionPlanner(llm=llm),
    ).plan_pending_supporting_questions(limit=1)

    assert result == {"planned": 1, "needs_review": 0, "questions_created": 0}
    assert _row(repository, first_id)["supporting_question_plan_status"] == "planned"
    assert _row(repository, second_id)["supporting_question_plan_status"] == "pending"


def test_scan_due_persists_question_before_notification(repository: Any) -> None:
    reminder_id = _seed_planned_reminder(repository)
    planner, _ = _planner(
        {
            "should_ask": True,
            "question_text": MEETING_QUESTION,
            "confidence": 0.96,
            "reason_summary": "Meeting preparation is useful.",
        }
    )
    autoscan = ReminderAutoscan(
        repository=repository,
        supporting_question_planner=planner,
    )

    assert autoscan.scan_due(now_value=FIRE_TIME.isoformat()) == [reminder_id]
    row = _row(repository, reminder_id)
    assert row["status"] == "notified"
    assert row["supporting_question_plan_status"] == "planned"
    notification = repository.list_notifications(user_id=USER_ID)[0]
    context = repository.load_reminder_reply_context(
        user_id=USER_ID,
        reminder_id=reminder_id,
        notification_id=notification["notification_id"],
    )
    assert context["supporting_question"] == MEETING_QUESTION
    assert context["reminder_id"] == reminder_id
    assert context["notification_id"] == notification["notification_id"]
    assert context["subject"] == "Quarterly planning meeting"
    assert context["reminder_status"] == "notified"
    assert context["notification_ui_status"] == "unread"
    assert context["notification_fire_time"] == FIRE_TIME.isoformat()


def test_existing_supporting_context_is_never_overwritten(repository: Any) -> None:
    existing = "Would you like the existing checklist?"
    reminder_id = _seed_planned_reminder(
        repository,
        existing_question=existing,
    )
    llm = ScriptedLLM([])
    autoscan = ReminderAutoscan(
        repository=repository,
        supporting_question_planner=ReminderSupportingQuestionPlanner(llm=llm),
    )

    assert _row(repository, reminder_id)["supporting_question_plan_status"] == "planned"
    assert autoscan.plan_pending_supporting_questions() == {
        "planned": 0,
        "needs_review": 0,
        "questions_created": 0,
    }
    assert _row(repository, reminder_id)["supporting_question"] == existing
    assert llm.calls == []
