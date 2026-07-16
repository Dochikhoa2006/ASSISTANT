from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from assistant_rag.autoscan import ReminderAutoscan
from assistant_rag.branches import ReminderBranch
from assistant_rag.chat_history import canonical_chat_history_scope
from assistant_rag.contracts import (
    ChatRequest,
    ExpectedResponseType,
    GeneratedQuestion,
    Intent,
    PipelineContext,
    QuestionSource,
    ReminderAction,
    ResponseType,
)
from assistant_rag.database import SQLiteRepository
from assistant_rag.llm import LLMTask
from assistant_rag.production_factory import build_assistant_config
from assistant_rag.prompts import DEFAULT_PROMPT_REGISTRY
from assistant_rag.reminder_mutation import (
    REMINDER_EDITABLE_FIELDS,
    LLMReminderActionDetector,
    ReminderActionValidationStrategy,
    ReminderContentFinalizationStrategy,
    ReminderMutationPipeline,
)
from assistant_rag.reminder_timing import ReminderTimingPlanner
from assistant_rag.request_lifecycle import ChatRequestLifecycleExecutor
from assistant_rag.settings import ProductionSettings


USER_ID = "reminder-three-llm-user"
BASE_TIME = "2035-04-05T09:30:00+00:00"


class ScriptedLLM:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def generate_json(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(dict(kwargs))
        if not self.responses:
            raise AssertionError("unexpected extra reminder LLM call")
        return self.responses.pop(0)

    def chat(self, **_kwargs: Any) -> str:
        raise AssertionError("reminder mutation stages must use strict JSON")


class RejectingTimingPlanner:
    """Prove that autoscan never replans an explicit notification time."""

    def __init__(self) -> None:
        self.calls = 0

    def plan(self, **_kwargs: Any) -> Any:
        self.calls += 1
        raise AssertionError("autoscan must not replan an explicit notification time")


def test_timing_planner_prompt_uses_derived_reminder_context_not_raw_query() -> None:
    llm = ScriptedLLM(
        [
            {
                "lead_minutes": 30,
                "confidence": 0.99,
                "needs_clarification": False,
                "reason": "Notify shortly before the event.",
            }
        ]
    )
    planner = ReminderTimingPlanner(llm=llm)

    decision = planner.plan(
        event_time=datetime(2035, 4, 5, 10, 0, tzinfo=timezone.utc),
        subject="Project kickoff",
        reminder_context="DERIVED_REWRITTEN_REMINDER_CONTEXT",
        user_timezone="UTC",
        now=datetime(2035, 4, 5, 9, 0, tzinfo=timezone.utc),
    )

    assert decision.notification_time == datetime(
        2035, 4, 5, 9, 30, tzinfo=timezone.utc
    )
    prompt = json.loads(llm.calls[0]["user_prompt"])
    assert prompt["reminder_context"] == "DERIVED_REWRITTEN_REMINDER_CONTEXT"
    assert "raw_query" not in prompt
    assert "user_query" not in prompt


def _repository() -> SQLiteRepository:
    repository = SQLiteRepository.in_memory()
    repository.initialize_schema()
    return repository


def _seed_source(
    repository: SQLiteRepository,
    *,
    user_id: str = USER_ID,
) -> tuple[str, str]:
    with repository.transaction() as cursor:
        topic_id = repository.ensure_topic(
            cursor,
            user_id=user_id,
            title="Reminder mutation source",
        )
        hop = repository.append_conversation_hop(
            cursor,
            topic_id=topic_id,
            user_id=user_id,
            intent=Intent.REMINDER.value,
            raw_user_query="Create the source reminder.",
            rewritten_user_query="Create the source reminder.",
            raw_response="The source reminder was created.",
            response_type=ResponseType.REMINDER_ACTION.value,
        )
    return topic_id, hop.hop_id


def _seed_reminder(
    repository: SQLiteRepository,
    *,
    subject: str = "Payroll",
    reminder_summary: str = "Prepare payroll",
    raw_reminder: str = "Remind me to prepare payroll.",
    reminder_time: str = BASE_TIME,
    event_time: str = BASE_TIME,
    status: str = "scheduled",
    supporting_question: str | None = "Should this repeat?",
    supporting_response: str | None = "Monthly",
    user_timezone: str = "UTC",
    original_time_text: str | None = BASE_TIME,
    recurrence_rule: str | None = "FREQ=MONTHLY",
    recurrence_timezone: str | None = "UTC",
    next_fire_time: str | None = None,
    parent_recurring_reminder_id: str | None = None,
    user_id: str = USER_ID,
) -> str:
    topic_id, hop_id = _seed_source(repository, user_id=user_id)
    with repository.transaction() as cursor:
        reminder_id = repository.add_reminder(
            cursor,
            user_id=user_id,
            source_topic_id=topic_id,
            source_hop_id=hop_id,
            reminder_time=reminder_time,
            event_time=event_time,
            raw_reminder=raw_reminder,
            reminder_summary=reminder_summary,
            subject=subject,
            supporting_question=supporting_question,
            supporting_response=supporting_response,
            user_timezone=user_timezone,
            original_time_text=original_time_text,
            recurrence_rule=recurrence_rule,
            recurrence_timezone=recurrence_timezone,
            next_fire_time=(
                next_fire_time
                if next_fire_time is not None
                else reminder_time
                if recurrence_rule
                else None
            ),
            parent_recurring_reminder_id=parent_recurring_reminder_id,
        )
        if status != "scheduled":
            repository.update_reminder_status(
                cursor,
                user_id=user_id,
                reminder_id=reminder_id,
                status=status,
            )
    return reminder_id


def _reminder_row(repository: SQLiteRepository, reminder_id: str) -> dict[str, Any]:
    row = repository.connection.execute(
        "SELECT * FROM reminders WHERE reminder_id = ?",
        (reminder_id,),
    ).fetchone()
    assert row is not None
    return dict(row)


def _record_from_row(row: dict[str, Any]) -> dict[str, str]:
    key_by_field = {
        "notification_time": "reminder_time",
    }
    return {
        field: str(row.get(key_by_field.get(field, field)) or "").strip()
        for field in REMINDER_EDITABLE_FIELDS
    }


def _empty_record() -> dict[str, str]:
    return {field: "" for field in REMINDER_EDITABLE_FIELDS}


def _context(raw_query: str, history: list[dict[str, Any]]) -> PipelineContext:
    return PipelineContext(
        request=ChatRequest(
            user_id=USER_ID,
            raw_query=raw_query,
            platform_context={"timezone": "UTC"},
        ),
        rewritten_query=raw_query,
        last_qa_state=None,
        conversation_results=[],
        intent=Intent.REMINDER,
        chat_history=history,
    )


def _branch(llm: ScriptedLLM) -> ReminderBranch:
    config = build_assistant_config(ProductionSettings())
    validator = ReminderActionValidationStrategy(
        config=config,
        llm=llm,
        prompts=DEFAULT_PROMPT_REGISTRY,
    )
    finalizer = ReminderContentFinalizationStrategy(
        llm=llm,
        prompts=DEFAULT_PROMPT_REGISTRY,
        min_confidence=(
            config.retrieval_validation.reminder_llm_validation_min_confidence
        ),
    )
    return ReminderBranch(
        config=config,
        action_detector=LLMReminderActionDetector(
            llm=llm,
            prompts=DEFAULT_PROMPT_REGISTRY,
            min_confidence=0.76,
            default_timezone=config.default_timezone,
        ),
        prompt_registry=DEFAULT_PROMPT_REGISTRY,
        reminder_mutation_pipeline=ReminderMutationPipeline(
            config=config,
            validator=validator,
            finalizer=finalizer,
        ),
    )


def _prompt_payload(call: dict[str, Any]) -> dict[str, Any]:
    prefix = "Runtime context:\n"
    prompt = str(call["user_prompt"])
    assert prompt.startswith(prefix)
    return json.loads(prompt[len(prefix) :])


def _extraction(
    *,
    action: str,
    retrieval_text: str,
    toggle_direction: str = "",
    changed_fields: list[str] | None = None,
    time_semantics: str = "unchanged",
    confidence: float = 0.99,
    missing_fields: list[str] | None = None,
    **values: str,
) -> dict[str, Any]:
    record = _empty_record()
    record.update(values)
    return {
        "action": action,
        "toggle_direction": toggle_direction,
        "retrieval_text": retrieval_text,
        "changed_fields": list(changed_fields or []),
        **record,
        "time_semantics": time_semantics,
        "confidence": confidence,
        "missing_fields": list(missing_fields or []),
        "reason_summary": "One grounded reminder action was extracted.",
    }


def _add_extraction(
    *,
    subject: str = "Submit payroll",
    reminder_time: str = BASE_TIME,
) -> dict[str, Any]:
    values = {
        "subject": subject,
        "reminder_summary": subject,
        "raw_reminder": subject,
        "notification_time": reminder_time,
        "event_time": reminder_time,
        "user_timezone": "UTC",
        "original_time_text": reminder_time,
    }
    return _extraction(
        action="add",
        retrieval_text=subject,
        # ADD owns one complete new reminder; changed_fields is reserved for
        # an existing reminder's MODIFY patch.
        changed_fields=[],
        time_semantics="both",
        **values,
    )


def _toggle_extraction(*, direction: str, retrieval_text: str) -> dict[str, Any]:
    return _extraction(
        action="toggle",
        toggle_direction=direction,
        retrieval_text=retrieval_text,
    )


def _validation(
    *,
    operation: str,
    result: str,
    selected: list[str],
    assessments: list[dict[str, Any]],
    should_execute: bool,
    confidence: float = 0.99,
    requires_hitl: bool = False,
    factuality_concern: bool = False,
    ambiguous: bool = False,
    hitl_reason: str = "",
    reason: str = "The reminder action was validated.",
) -> dict[str, Any]:
    return {
        "operation": operation,
        "validation_result": result,
        "selected_candidate_keys": selected,
        "confidence": confidence,
        "ambiguous": ambiguous,
        "should_execute": should_execute,
        "requires_hitl": requires_hitl,
        "factuality_concern": factuality_concern,
        "hitl_reason": hitl_reason,
        "reason_summary": reason,
        "candidate_assessments": assessments,
    }


def _assessment(
    reminder_id: str,
    *,
    matches: bool,
    compatible: bool = True,
    matched_fields: list[str] | None = None,
    matched_text: str = "",
    confidence: float = 0.99,
) -> dict[str, Any]:
    return {
        "candidate_key": reminder_id,
        "matches_target": matches,
        "action_compatible": compatible,
        "confidence": confidence,
        "matched_fields": list(matched_fields or []),
        "matched_text": matched_text,
        "reason_summary": "The complete SQL reminder was assessed.",
    }


def _finalization(
    *,
    operation: str,
    selected_candidate_key: str,
    changed_fields: list[str] | None = None,
    confidence: float = 0.99,
) -> dict[str, Any]:
    changed = set(changed_fields or [])
    return {
        "operation": operation,
        "selected_candidate_key": selected_candidate_key,
        "field_bindings": [
            {
                "field": field,
                "source": (
                    "extracted_action"
                    if operation == "add"
                    or (operation == "modify" and field in changed)
                    else "selected_candidate"
                ),
            }
            for field in REMINDER_EDITABLE_FIELDS
        ],
        "confidence": confidence,
        "reason_summary": "Every field was bound to its exact trusted source.",
    }


def test_add_runs_exactly_three_llm_stages_with_identical_history_and_query() -> None:
    query = (
        "Add a reminder: Submit payroll at "
        f"{BASE_TIME} in UTC."
    )
    history = [
        {"hop_id": f"hop-{index}", "text": f"Payroll context {index}"}
        for index in range(8)
    ]
    extraction = _add_extraction()
    llm = ScriptedLLM(
        [
            extraction,
            _validation(
                operation="add",
                result="EXECUTE",
                selected=[],
                assessments=[],
                should_execute=True,
            ),
            _finalization(
                operation="add",
                selected_candidate_key="",
            ),
        ]
    )
    repository = _repository()
    forbidden_raw_query = "FORBIDDEN_RAW_REMINDER_QUERY"
    context = replace(
        _context(forbidden_raw_query, history),
        rewritten_query=query,
    )

    with canonical_chat_history_scope(history):
        result = _branch(llm).execute(context, repository)

    assert result.response_type is ResponseType.REMINDER_ACTION
    assert [call["task"] for call in llm.calls] == [
        LLMTask.REMINDER_ACTION_EXTRACTION,
        LLMTask.REMINDER_ACTION_VALIDATION,
        LLMTask.REMINDER_CONTENT_FINALIZATION,
    ]
    for call in llm.calls:
        payload = _prompt_payload(call)
        assert "raw_query" not in payload
        assert payload["rewritten_query"] == query
        assert payload["chat_history"] == history
        assert forbidden_raw_query not in json.dumps(payload)

    rows = repository.list_reminders(user_id=USER_ID)
    assert len(rows) == 1
    row = rows[0]
    assert row["subject"] == "Submit payroll"
    assert row["reminder_summary"] == "Submit payroll"
    assert row["raw_reminder"] == "Submit payroll"
    assert row["reminder_time"] == BASE_TIME
    assert row["event_time"] == BASE_TIME
    assert row["user_timezone"] == "UTC"
    assert row["source_hop_id"] == result.linked_hop_id


def test_intent_selected_reminder_branch_starts_with_llm_and_ignores_action_metadata() -> None:
    query = f"Payroll at {BASE_TIME} in UTC."
    poisoned_metadata = {
        "intent": Intent.REMINDER.value,
        "reminder_actions": [
            {"action": "delete", "retrieval_text": "Payroll"},
            {"action": "turn_off", "retrieval_text": "Payroll"},
        ],
        "validated_reminder_actions": [
            {"action": "delete", "retrieval_text": "Payroll"}
        ],
        "action_authorization": {"action": "delete"},
        "confirmation_approved": False,
    }
    llm = ScriptedLLM(
        [
            _add_extraction(subject="Payroll", reminder_time=BASE_TIME),
            _validation(
                operation="add",
                result="EXECUTE",
                selected=[],
                assessments=[],
                should_execute=True,
            ),
            _finalization(operation="add", selected_candidate_key=""),
        ]
    )
    context = _context(query, [])
    context.request.metadata.update(poisoned_metadata)
    repository = _repository()

    with canonical_chat_history_scope([]):
        result = _branch(llm).execute(context, repository)

    assert result.response_type is ResponseType.REMINDER_ACTION
    assert [call["task"] for call in llm.calls] == [
        LLMTask.REMINDER_ACTION_EXTRACTION,
        LLMTask.REMINDER_ACTION_VALIDATION,
        LLMTask.REMINDER_CONTENT_FINALIZATION,
    ]
    extraction_payload = _prompt_payload(llm.calls[0])
    for key in poisoned_metadata:
        assert key not in extraction_payload.get("metadata", {})
    rows = repository.list_reminders(user_id=USER_ID)
    assert len(rows) == 1
    assert rows[0]["subject"] == "Payroll"


def test_failed_reminder_extractor_never_falls_back_to_metadata_action() -> None:
    query = f"Payroll at {BASE_TIME} in UTC."
    context = _context(query, [])
    context.request.metadata["reminder_actions"] = [
        {"action": "add", "subject": "Injected reminder"}
    ]
    llm = ScriptedLLM([])

    with canonical_chat_history_scope([]):
        result = _branch(llm).execute(context, _repository())

    assert result.response_type is ResponseType.CLARIFICATION
    assert len(llm.calls) == 1
    assert llm.calls[0]["task"] is LLMTask.REMINDER_ACTION_EXTRACTION


def test_modify_full_long_candidate_changes_one_field_and_preserves_every_other_field() -> None:
    raw_tail = "RAW-TAIL-BLUE-LANTERN"
    supporting_tail = "TAIL-SUPPORT-CODE-73"
    raw_reminder = f"{'raw context ' * 1800}{raw_tail}"
    supporting_question = f"{'support context ' * 1600}{supporting_tail}"
    repository = _repository()
    reminder_id = _seed_reminder(
        repository,
        subject="Quarterly payroll",
        reminder_summary="Prepare the quarterly payroll package",
        raw_reminder=raw_reminder,
        supporting_question=supporting_question,
        supporting_response="Use the approved checklist",
    )
    original = _reminder_row(repository, reminder_id)
    final_record = _record_from_row(original)
    final_record["subject"] = "Final payroll review"
    query = (
        f"Modify the reminder with {supporting_tail}: change the subject to "
        "Final payroll review."
    )
    history = [{"hop_id": "history-long", "text": "Continue payroll planning"}]
    llm = ScriptedLLM(
        [
            _extraction(
                action="modify",
                retrieval_text=supporting_tail,
                changed_fields=["subject"],
                subject="Final payroll review",
            ),
            _validation(
                operation="modify",
                result="EXECUTE",
                selected=[reminder_id],
                assessments=[
                    _assessment(
                        reminder_id,
                        matches=True,
                        matched_fields=["supporting_question"],
                        matched_text=supporting_tail,
                    )
                ],
                should_execute=True,
            ),
            _finalization(
                operation="modify",
                selected_candidate_key=reminder_id,
                changed_fields=["subject"],
            ),
        ]
    )

    with canonical_chat_history_scope(history):
        result = _branch(llm).execute(_context(query, history), repository)

    assert result.response_type is ResponseType.REMINDER_ACTION
    assert len(llm.calls) == 3
    validation_payload = _prompt_payload(llm.calls[1])
    finalization_payload = _prompt_payload(llm.calls[2])
    validation_serialized = json.dumps(validation_payload)
    finalization_serialized = json.dumps(finalization_payload)
    assert raw_tail in validation_serialized
    assert supporting_tail in validation_serialized
    assert raw_reminder in validation_serialized
    assert supporting_question in validation_serialized
    assert raw_reminder not in finalization_serialized
    assert supporting_question not in finalization_serialized
    assert len(finalization_serialized) < 12_000
    finalization_extra = finalization_payload["extra"]
    assert "candidate_reminders" not in finalization_extra
    assert "extracted_action" not in finalization_extra
    selected_manifest = finalization_extra["selected_candidate_manifest"]
    raw_manifest = next(
        item
        for item in selected_manifest["field_manifest"]
        if item["field"] == "raw_reminder"
    )
    supporting_manifest = next(
        item
        for item in selected_manifest["field_manifest"]
        if item["field"] == "supporting_question"
    )
    assert raw_manifest["character_count"] == len(raw_reminder)
    assert raw_manifest["bounded_preview"].endswith(raw_tail)
    assert supporting_manifest["character_count"] == len(supporting_question)
    assert supporting_manifest["bounded_preview"].endswith(supporting_tail)

    old_row = _reminder_row(repository, reminder_id)
    assert old_row["status"] == "cancelled"
    successors = [
        row
        for row in repository.list_reminders(user_id=USER_ID)
        if row["reminder_id"] != reminder_id
    ]
    assert len(successors) == 1
    successor = successors[0]
    assert successor["status"] == "scheduled"
    assert successor["parent_recurring_reminder_id"] == reminder_id
    assert successor["source_topic_id"] == original["source_topic_id"]
    assert successor["source_hop_id"] == original["source_hop_id"]
    assert _record_from_row(successor) == final_record


@pytest.mark.parametrize(
    "failure",
    (
        "missing_field",
        "duplicate_field",
        "unknown_field",
        "wrong_source",
        "extra_root_field",
    ),
)
def test_finalization_requires_one_exact_binding_for_every_editable_field(
    failure: str,
) -> None:
    repository = _repository()
    reminder_id = _seed_reminder(
        repository,
        subject="Binding contract target",
        reminder_summary="Binding contract target",
        raw_reminder="Binding contract target",
    )
    before = _reminder_row(repository, reminder_id)
    finalization = _finalization(
        operation="delete",
        selected_candidate_key=reminder_id,
    )
    bindings = finalization["field_bindings"]
    if failure == "missing_field":
        bindings.pop()
    elif failure == "duplicate_field":
        bindings[-1] = dict(bindings[0])
    elif failure == "unknown_field":
        bindings[-1] = {
            "field": "unknown_field",
            "source": "selected_candidate",
        }
    elif failure == "wrong_source":
        bindings[0]["source"] = "extracted_action"
    else:
        finalization["final_reminder"] = {
            field: "must not be accepted" for field in REMINDER_EDITABLE_FIELDS
        }

    llm = ScriptedLLM(
        [
            _extraction(action="delete", retrieval_text="Binding contract target"),
            _validation(
                operation="delete",
                result="EXECUTE",
                selected=[reminder_id],
                assessments=[
                    _assessment(
                        reminder_id,
                        matches=True,
                        matched_fields=["subject"],
                        matched_text="Binding contract target",
                    )
                ],
                should_execute=True,
            ),
            finalization,
        ]
    )

    with canonical_chat_history_scope([]):
        result = _branch(llm).execute(
            _context("Delete the Binding contract target reminder.", []),
            repository,
        )

    assert result.response_type is ResponseType.CLARIFICATION
    assert result.clarification_question is not None
    assert len(llm.calls) == 3
    after = _reminder_row(repository, reminder_id)
    assert after["status"] == before["status"]
    assert after["version"] == before["version"]


def test_delete_confirmation_replay_starts_with_extraction_only() -> None:
    repository = _repository()
    reminder_id = _seed_reminder(
        repository,
        subject="Legacy review",
        reminder_summary="Legacy review",
        raw_reminder="Review the legacy process.",
        recurrence_rule=None,
        recurrence_timezone=None,
    )
    original = _reminder_row(repository, reminder_id)
    query = "Delete the Legacy review reminder."
    history = [{"hop_id": "delete-history", "text": "Legacy review context"}]
    llm = ScriptedLLM(
        [
            _extraction(action="delete", retrieval_text="Legacy review"),
            _validation(
                operation="delete",
                result="EXECUTE",
                selected=[reminder_id],
                assessments=[
                    _assessment(
                        reminder_id,
                        matches=True,
                        matched_fields=["subject"],
                        matched_text="Legacy review",
                    )
                ],
                should_execute=True,
            ),
            _finalization(
                operation="delete",
                selected_candidate_key=reminder_id,
            ),
        ]
    )
    branch = _branch(llm)

    with canonical_chat_history_scope(history):
        pending = branch.execute(_context(query, history), repository)

    assert pending.response_type is ResponseType.REMINDER_ACTION
    assert pending.actions_pending_confirmation
    assert _reminder_row(repository, reminder_id)["status"] == "scheduled"
    assert len(llm.calls) == 3

    token = pending.actions_pending_confirmation[0]["confirmation_token"]
    lifecycle = ChatRequestLifecycleExecutor(
        pipeline=SimpleNamespace(),  # type: ignore[arg-type]
        repository=repository,
    )
    prepared, confirmation_to_mark = lifecycle._hydrate_confirmation(
        ChatRequest(
            user_id=USER_ID,
            raw_query="Confirm",
            confirmation_token=token,
        )
    )
    replay_context = PipelineContext(
        request=prepared,
        rewritten_query="Confirm",
        last_qa_state=None,
        conversation_results=[],
        intent=Intent.REMINDER,
        chat_history=history,
    )
    llm.responses.append(
        _extraction(action="delete", retrieval_text="Legacy review")
    )
    with canonical_chat_history_scope(history):
        committed = branch.execute(replay_context, repository)

    assert confirmation_to_mark == token
    assert committed.response_type is ResponseType.REMINDER_ACTION
    assert len(llm.calls) == 4
    assert llm.calls[-1]["task"] is LLMTask.REMINDER_ACTION_EXTRACTION
    confirmation_payload = _prompt_payload(llm.calls[-1])
    assert confirmation_payload["extra"]["confirmation_replay"] is True
    assert len(
        confirmation_payload["extra"]["trusted_confirmation_action_context"]
    ) == 1
    assert _reminder_row(repository, reminder_id)["status"] == "dismissed"


@pytest.mark.parametrize(
    ("action", "query", "initial_status", "expected_status"),
    (
        ("turn_on", "Turn on the Payroll reminder.", "cancelled", "scheduled"),
        ("turn_off", "Turn off the Payroll reminder.", "scheduled", "cancelled"),
    ),
)
def test_toggle_directions_run_three_stages_and_only_change_status(
    action: str,
    query: str,
    initial_status: str,
    expected_status: str,
) -> None:
    repository = _repository()
    reminder_id = _seed_reminder(repository, status=initial_status)
    original = _reminder_row(repository, reminder_id)
    original_record = _record_from_row(original)
    llm = ScriptedLLM(
        [
            _toggle_extraction(direction=action, retrieval_text="Payroll"),
            _validation(
                operation=action,
                result="EXECUTE",
                selected=[reminder_id],
                assessments=[
                    _assessment(
                        reminder_id,
                        matches=True,
                        matched_fields=["subject"],
                        matched_text="Payroll",
                    )
                ],
                should_execute=True,
            ),
            _finalization(
                operation=action,
                selected_candidate_key=reminder_id,
            ),
        ]
    )

    with canonical_chat_history_scope([]):
        result = _branch(llm).execute(_context(query, []), repository)

    assert result.response_type is ResponseType.REMINDER_ACTION
    assert [call["task"] for call in llm.calls] == [
        LLMTask.REMINDER_ACTION_EXTRACTION,
        LLMTask.REMINDER_ACTION_VALIDATION,
        LLMTask.REMINDER_CONTENT_FINALIZATION,
    ]
    updated = _reminder_row(repository, reminder_id)
    assert updated["status"] == expected_status
    assert _record_from_row(updated) == original_record


@pytest.mark.parametrize("direction", ("turn_on", "turn_off"))
def test_first_llm_uses_one_toggle_action_then_maps_direction_downstream(
    direction: str,
) -> None:
    query = f"{direction.replace('_', ' ').title()} the Payroll reminder."
    llm = ScriptedLLM(
        [_toggle_extraction(direction=direction, retrieval_text="Payroll")]
    )
    detector = LLMReminderActionDetector(
        llm=llm,
        prompts=DEFAULT_PROMPT_REGISTRY,
        min_confidence=0.76,
    )

    with canonical_chat_history_scope([]):
        result = detector.detect(
            ChatRequest(user_id=USER_ID, raw_query=query),
            query,
            Intent.REMINDER,
        )

    assert not result.requires_clarification
    assert len(result.metadata["reminder_actions"]) == 1
    action = result.metadata["reminder_actions"][0]
    assert action["extracted_action"] == "toggle"
    assert action["toggle_direction"] == direction
    assert action["action"] == direction
    prompt = _prompt_payload(llm.calls[0])
    assert prompt["extra"]["allowed_actions"] == [
        "add",
        "delete",
        "modify",
        "toggle",
    ]
    assert prompt["extra"]["toggle_directions"] == ["turn_on", "turn_off"]


@pytest.mark.parametrize(
    ("payload", "expected_risk"),
    (
        (
            _extraction(action="turn_on", retrieval_text="Payroll"),
            "low_confidence_or_incomplete_reminder_extraction",
        ),
        (
            _extraction(action="toggle", retrieval_text="Payroll"),
            "toggle_reminder_requires_exact_direction",
        ),
        (
            _extraction(
                action="add",
                toggle_direction="turn_on",
                retrieval_text="Payroll",
            ),
            "non_toggle_reminder_forbids_toggle_direction",
        ),
    ),
)
def test_first_llm_external_action_and_toggle_shape_fail_closed(
    payload: dict[str, Any],
    expected_risk: str,
) -> None:
    llm = ScriptedLLM([payload])
    detector = LLMReminderActionDetector(
        llm=llm,
        prompts=DEFAULT_PROMPT_REGISTRY,
        min_confidence=0.76,
    )

    with canonical_chat_history_scope([]):
        result = detector.detect(
            ChatRequest(
                user_id=USER_ID,
                raw_query="Change the Payroll reminder state.",
            ),
            "Change the Payroll reminder state.",
            Intent.REMINDER,
        )

    assert result.requires_clarification
    assert result.metadata == {}
    assert expected_risk in result.risk_flags
    assert len(llm.calls) == 1


def test_first_llm_modify_carries_every_reminder_section_in_one_action() -> None:
    notification_time = "2035-04-05T08:30:00+00:00"
    event_time = "2035-04-05T10:00:00+00:00"
    original_time_text = (
        f"notify at {notification_time} and event at {event_time}"
    )
    query = (
        "Modify the Payroll reminder: set title to Quarterly Payroll; "
        "summary to Prepare quarterly payroll; content to Submit payroll report; "
        f"{original_time_text}; timezone UTC; recurrence FREQ=WEEKLY in UTC; "
        "supporting question Should finance approve?; "
        "supporting response Finance approved."
    )
    values = {
        "subject": "Quarterly Payroll",
        "reminder_summary": "Prepare quarterly payroll",
        "raw_reminder": "Submit payroll report",
        "notification_time": notification_time,
        "event_time": event_time,
        "user_timezone": "UTC",
        "original_time_text": original_time_text,
        "recurrence_rule": "FREQ=WEEKLY",
        "recurrence_timezone": "UTC",
        "supporting_question": "Should finance approve?",
        "supporting_response": "Finance approved",
    }
    llm = ScriptedLLM(
        [
            _extraction(
                action="modify",
                retrieval_text="Payroll",
                changed_fields=list(REMINDER_EDITABLE_FIELDS),
                time_semantics="both",
                **values,
            )
        ]
    )
    detector = LLMReminderActionDetector(
        llm=llm,
        prompts=DEFAULT_PROMPT_REGISTRY,
        min_confidence=0.76,
    )

    with canonical_chat_history_scope([]):
        result = detector.detect(
            ChatRequest(
                user_id=USER_ID,
                raw_query=query,
                platform_context={"timezone": "UTC"},
            ),
            query,
            Intent.REMINDER,
        )

    assert not result.requires_clarification
    assert len(result.metadata["reminder_actions"]) == 1
    action = result.metadata["reminder_actions"][0]
    assert action["action"] == "modify"
    assert action["extracted_action"] == "modify"
    assert action["toggle_direction"] == ""
    assert action["changed_fields"] == list(REMINDER_EDITABLE_FIELDS)
    assert {field: action[field] for field in REMINDER_EDITABLE_FIELDS} == values
    assert len(llm.calls) == 1


@pytest.mark.parametrize(
    "malformed_payload",
    (
        {
            **_extraction(action="delete", retrieval_text="Payroll"),
            "action": ["delete", "modify"],
        },
        {
            **_extraction(action="delete", retrieval_text="Payroll"),
            "secondary_action": "modify",
        },
        {
            key: value
            for key, value in _extraction(
                action="delete",
                retrieval_text="Payroll",
            ).items()
            if key != "toggle_direction"
        },
        {
            **_extraction(
                action="modify",
                retrieval_text="Payroll",
                changed_fields=["subject"],
                subject="Quarterly Payroll",
            ),
            "changed_fields": "subject",
        },
    ),
)
def test_reminder_extractor_rejects_non_single_schema_complete_payloads(
    malformed_payload: dict[str, Any],
) -> None:
    query = "Delete the Payroll reminder."
    llm = ScriptedLLM([malformed_payload])
    detector = LLMReminderActionDetector(
        llm=llm,
        prompts=DEFAULT_PROMPT_REGISTRY,
        min_confidence=0.76,
    )

    with canonical_chat_history_scope([]):
        detection = detector.detect(
            ChatRequest(user_id=USER_ID, raw_query=query),
            query,
            Intent.REMINDER,
        )

    assert detection.requires_clarification
    assert detection.missing_fields == ["action"]
    assert detection.metadata.get("reminder_actions") is None
    assert len(llm.calls) == 1


def test_reminder_first_llm_receives_complete_context_and_field_contract() -> None:
    query = "Change that reminder title to Quarterly Payroll."
    rewritten_query = "Change the Payroll reminder title to Quarterly Payroll."
    history = [
        {
            "hop_id": "hop-payroll",
            "raw_user_query": "FORBIDDEN_RAW_HISTORY_QUERY",
            "rewritten_user_query": "Set the Payroll reminder.",
            "raw_response": "The Payroll reminder is scheduled.",
        }
    ]
    request = ChatRequest(
        user_id=USER_ID,
        raw_query=query,
        metadata={
            "locale": "en-US",
            "reminder_actions": [{"action": "delete"}],
            "validated_reminder_actions": [{"action": "turn_off"}],
            "action_authorization": {"action": "delete"},
        },
        platform_context={"platform": "streamlit", "timezone": "UTC"},
    )
    llm = ScriptedLLM(
        [
            _extraction(
                action="modify",
                retrieval_text="Payroll",
                changed_fields=["subject"],
                subject="Quarterly Payroll",
            )
        ]
    )
    detector = LLMReminderActionDetector(
        llm=llm,
        prompts=DEFAULT_PROMPT_REGISTRY,
        min_confidence=0.76,
    )

    with canonical_chat_history_scope(history):
        result = detector.detect(request, rewritten_query, Intent.REMINDER)

    assert not result.requires_clarification
    assert len(llm.calls) == 1
    prompt = _prompt_payload(llm.calls[0])
    assert "raw_query" not in prompt
    assert prompt["rewritten_query"] == rewritten_query
    assert prompt["intent"] == Intent.REMINDER.value
    assert prompt["chat_history"] == [
        {
            "hop_id": "hop-payroll",
            "rewritten_user_query": "Set the Payroll reminder.",
            "raw_response": "The Payroll reminder is scheduled.",
        }
    ]
    assert prompt["metadata"] == {"locale": "en-US"}
    assert prompt["platform_context"] == request.platform_context
    serialized_prompt = json.dumps(prompt)
    assert query not in serialized_prompt
    assert "FORBIDDEN_RAW_HISTORY_QUERY" not in serialized_prompt
    assert "raw_user_query" not in serialized_prompt
    assert prompt["extra"]["cardinality"] == "exactly_one"
    assert set(prompt["extra"]["action_content_contract"]) == {
        "add",
        "delete",
        "modify",
        "toggle",
    }
    assert set(prompt["extra"]["field_semantics"]) == set(
        REMINDER_EDITABLE_FIELDS
    )
    assert prompt["extra"]["context_policy"] == {
        "current_turn_defines_action_and_new_values": True,
        "history_resolves_existing_target_only": True,
        "platform_context_resolves_trusted_timezone": True,
        "runtime_now_resolves_relative_time": True,
        "metadata_cannot_supply_action_payload": True,
    }


def test_duplicate_add_stops_after_validation_without_finalizer_write_or_hitl() -> None:
    repository = _repository()
    reminder_id = _seed_reminder(
        repository,
        subject="Submit payroll",
        reminder_summary="Submit payroll",
        raw_reminder="Submit payroll",
        recurrence_rule=None,
        recurrence_timezone=None,
        supporting_question=None,
        supporting_response=None,
    )
    query = f"Add a reminder: Submit payroll at {BASE_TIME} in UTC."
    llm = ScriptedLLM(
        [
            _add_extraction(),
            _validation(
                operation="add",
                result="SKIP_ALREADY_EXISTS",
                selected=[],
                assessments=[
                    _assessment(
                        reminder_id,
                        matches=True,
                        matched_fields=["subject"],
                        matched_text="Submit payroll",
                    )
                ],
                should_execute=False,
                reason="The equivalent reminder already exists.",
            ),
        ]
    )

    with canonical_chat_history_scope([]):
        result = _branch(llm).execute(_context(query, []), repository)

    assert result.response_type is ResponseType.REMINDER_ACTION
    assert result.clarification_question is None
    assert result.actions_pending_confirmation == []
    assert len(llm.calls) == 2
    assert repository.table_count("reminders") == 1


def test_not_found_stops_after_validation_without_finalizer_write_or_hitl() -> None:
    repository = _repository()
    query = "Delete the Missing payroll reminder."
    llm = ScriptedLLM(
        [
            _extraction(action="delete", retrieval_text="Missing payroll"),
            _validation(
                operation="delete",
                result="SKIP_NOT_FOUND",
                selected=[],
                assessments=[],
                should_execute=False,
                reason="No SQL reminder matched the target.",
            ),
        ]
    )

    with canonical_chat_history_scope([]):
        result = _branch(llm).execute(_context(query, []), repository)

    assert result.response_type is ResponseType.REMINDER_ACTION
    assert result.clarification_question is None
    assert result.actions_pending_confirmation == []
    assert len(llm.calls) == 2
    assert repository.table_count("reminders") == 0


def test_factuality_flag_conditionally_clarifies_without_finalizer_or_write() -> None:
    fact = "1 + 1 = 3"
    query = f"Add a reminder: {fact} at {BASE_TIME} in UTC."
    extraction = _add_extraction(subject=fact)
    llm = ScriptedLLM(
        [
            extraction,
            _validation(
                operation="add",
                result="CLARIFY_MISSING_FIELDS",
                selected=[],
                assessments=[],
                should_execute=False,
                requires_hitl=True,
                factuality_concern=True,
                hitl_reason="factuality_concern",
                reason="The asserted arithmetic is false.",
            ),
        ]
    )
    repository = _repository()

    with canonical_chat_history_scope([]):
        result = _branch(llm).execute(_context(query, []), repository)

    assert result.response_type is ResponseType.CLARIFICATION
    assert result.clarification_question is not None
    assert len(llm.calls) == 2
    assert repository.table_count("reminders") == 0


@pytest.mark.parametrize("failure", ("missing_assessment", "invented_id", "low_confidence"))
def test_invalid_validation_contracts_fail_closed_before_finalization(
    failure: str,
) -> None:
    repository = _repository()
    reminder_id = _seed_reminder(repository, subject="Payroll")
    query = "Delete the Payroll reminder."
    assessments = [
        _assessment(
            reminder_id,
            matches=True,
            matched_fields=["subject"],
            matched_text="Payroll",
        )
    ]
    selected = [reminder_id]
    confidence = 0.99
    if failure == "missing_assessment":
        assessments = []
    elif failure == "invented_id":
        selected = ["invented-reminder-id"]
    else:
        confidence = 0.20
    llm = ScriptedLLM(
        [
            _extraction(action="delete", retrieval_text="Payroll"),
            _validation(
                operation="delete",
                result="EXECUTE",
                selected=selected,
                assessments=assessments,
                should_execute=True,
                confidence=confidence,
            ),
        ]
    )

    with canonical_chat_history_scope([]):
        result = _branch(llm).execute(_context(query, []), repository)

    assert result.response_type is ResponseType.CLARIFICATION
    assert result.clarification_question is not None
    assert len(llm.calls) == 2
    assert _reminder_row(repository, reminder_id)["status"] == "scheduled"


@pytest.mark.parametrize(
    ("query", "history", "payload", "expected_risk"),
    (
        (
            "Delete the Payroll reminder.",
            [],
            _extraction(
                action="delete",
                retrieval_text="Payroll",
                changed_fields=["subject"],
                subject="Payroll",
            ),
            "invalid_target_only_reminder_content_contract",
        ),
        (
            "Modify the Payroll reminder.",
            [{"text": "The new title should be Quarterly payroll"}],
            _extraction(
                action="modify",
                retrieval_text="Payroll",
                changed_fields=["subject"],
                subject="Quarterly payroll",
            ),
            "reminder_content_not_grounded_in_current_turn",
        ),
    ),
)
def test_extraction_shape_and_grounding_fail_closed(
    query: str,
    history: list[dict[str, Any]],
    payload: dict[str, Any],
    expected_risk: str,
) -> None:
    llm = ScriptedLLM([payload])
    detector = LLMReminderActionDetector(
        llm=llm,
        prompts=DEFAULT_PROMPT_REGISTRY,
        min_confidence=0.76,
    )

    with canonical_chat_history_scope(history):
        result = detector.detect(
            ChatRequest(user_id=USER_ID, raw_query=query),
            query,
            Intent.REMINDER,
        )

    assert result.requires_clarification
    assert result.metadata == {}
    assert expected_risk in result.risk_flags
    assert len(llm.calls) == 1


def test_single_extracted_reminder_action_is_not_rechecked_by_raw_keywords() -> None:
    query = "Delete the Payroll reminder and add a new reminder."
    llm = ScriptedLLM(
        [_extraction(action="delete", retrieval_text="Payroll")]
    )
    detector = LLMReminderActionDetector(
        llm=llm,
        prompts=DEFAULT_PROMPT_REGISTRY,
        min_confidence=0.76,
    )

    with canonical_chat_history_scope([]):
        result = detector.detect(
            ChatRequest(user_id=USER_ID, raw_query=query),
            query,
            Intent.REMINDER,
        )

    assert not result.requires_clarification
    assert result.metadata["reminder_actions"][0]["action"] == "delete"
    assert result.metadata["action_authorization"]["source"] == (
        "llm_action_extraction"
    )
    assert len(llm.calls) == 1


def test_modify_target_may_use_history_but_replacement_must_use_current_turn() -> None:
    query = "Modify that reminder to Final payroll review."
    history = [
        {
            "hop_id": "prior-reminder",
            "text": "The referenced reminder is Quarterly payroll.",
        }
    ]
    llm = ScriptedLLM(
        [
            _extraction(
                action="modify",
                retrieval_text="Quarterly payroll",
                changed_fields=["subject"],
                subject="Final payroll review",
            )
        ]
    )
    detector = LLMReminderActionDetector(
        llm=llm,
        prompts=DEFAULT_PROMPT_REGISTRY,
        min_confidence=0.76,
    )

    with canonical_chat_history_scope(history):
        result = detector.detect(
            ChatRequest(user_id=USER_ID, raw_query=query),
            query,
            Intent.REMINDER,
        )

    assert not result.requires_clarification
    actions = result.metadata["reminder_actions"]
    assert len(actions) == 1
    assert actions[0]["retrieval_text"] == "Quarterly payroll"
    assert actions[0]["subject"] == "Final payroll review"


def test_modify_replacement_sourced_only_from_history_is_rejected() -> None:
    query = "Modify the Quarterly payroll reminder."
    history = [
        {
            "hop_id": "prior-replacement",
            "text": "A discarded replacement title was Final payroll review.",
        }
    ]
    llm = ScriptedLLM(
        [
            _extraction(
                action="modify",
                retrieval_text="Quarterly payroll",
                changed_fields=["subject"],
                subject="Final payroll review",
            )
        ]
    )
    detector = LLMReminderActionDetector(
        llm=llm,
        prompts=DEFAULT_PROMPT_REGISTRY,
        min_confidence=0.76,
    )

    with canonical_chat_history_scope(history):
        result = detector.detect(
            ChatRequest(user_id=USER_ID, raw_query=query),
            query,
            Intent.REMINDER,
        )

    assert result.requires_clarification
    assert result.metadata == {}
    assert "reminder_content_not_grounded_in_current_turn" in result.risk_flags


def test_add_persists_distinct_notification_and_event_times_independently() -> None:
    notification_time = "2035-04-05T08:30:00+00:00"
    event_time = "2035-04-05T10:00:00+00:00"
    original_time_text = (
        f"notify at {notification_time} for event at {event_time}"
    )
    query = f"Add a Project review reminder: {original_time_text} in UTC."
    extraction = _extraction(
        action="add",
        retrieval_text="Project review",
        subject="Project review",
        reminder_summary="Project review",
        raw_reminder="Project review",
        notification_time=notification_time,
        event_time=event_time,
        user_timezone="UTC",
        original_time_text=original_time_text,
        time_semantics="both",
    )
    llm = ScriptedLLM(
        [
            extraction,
            _validation(
                operation="add",
                result="EXECUTE",
                selected=[],
                assessments=[],
                should_execute=True,
            ),
            _finalization(
                operation="add",
                selected_candidate_key="",
            ),
        ]
    )
    repository = _repository()

    with canonical_chat_history_scope([]):
        result = _branch(llm).execute(_context(query, []), repository)

    assert result.response_type is ResponseType.REMINDER_ACTION
    assert len(llm.calls) == 3
    rows = repository.list_reminders(user_id=USER_ID)
    assert len(rows) == 1
    assert rows[0]["reminder_time"] == notification_time
    assert rows[0]["event_time"] == event_time
    assert rows[0]["reminder_time"] != rows[0]["event_time"]
    assert rows[0]["timing_plan_status"] == "planned"

    planner = RejectingTimingPlanner()
    timing_result = ReminderAutoscan(
        repository=repository,
        timing_planner=planner,
    ).plan_pending_timing(now=datetime(2035, 4, 1, tzinfo=timezone.utc))

    assert timing_result == {"planned": 0, "needs_review": 0}
    assert planner.calls == 0
    unchanged = repository.list_reminders(user_id=USER_ID)[0]
    assert unchanged["reminder_time"] == notification_time
    assert unchanged["event_time"] == event_time


def test_notification_only_add_is_planned_and_never_replanned_by_autoscan() -> None:
    notification_time = "2035-04-05T08:30:00+00:00"
    original_time_text = f"notify at {notification_time}"
    query = f"Add a Project briefing reminder: {original_time_text} in UTC."
    extraction = _extraction(
        action="add",
        retrieval_text="Project briefing",
        subject="Project briefing",
        reminder_summary="Project briefing",
        raw_reminder="Project briefing",
        notification_time=notification_time,
        user_timezone="UTC",
        original_time_text=original_time_text,
        time_semantics="notification_time",
    )
    llm = ScriptedLLM(
        [
            extraction,
            _validation(
                operation="add",
                result="EXECUTE",
                selected=[],
                assessments=[],
                should_execute=True,
            ),
            _finalization(
                operation="add",
                selected_candidate_key="",
            ),
        ]
    )
    repository = _repository()

    with canonical_chat_history_scope([]):
        result = _branch(llm).execute(_context(query, []), repository)

    assert result.response_type is ResponseType.REMINDER_ACTION
    assert len(llm.calls) == 3
    rows = repository.list_reminders(user_id=USER_ID)
    assert len(rows) == 1
    assert rows[0]["reminder_time"] == notification_time
    assert rows[0]["timing_plan_status"] == "planned"
    stored_event_time = rows[0]["event_time"]

    planner = RejectingTimingPlanner()
    timing_result = ReminderAutoscan(
        repository=repository,
        timing_planner=planner,
    ).plan_pending_timing(now=datetime(2035, 4, 1, tzinfo=timezone.utc))

    assert timing_result == {"planned": 0, "needs_review": 0}
    assert planner.calls == 0
    unchanged = repository.list_reminders(user_id=USER_ID)[0]
    assert unchanged["reminder_time"] == notification_time
    assert unchanged["event_time"] == stored_event_time


def test_event_only_add_uses_pending_planner_storage_fallback() -> None:
    event_time = "2035-04-05T10:00:00+00:00"
    original_time_text = f"event at {event_time}"
    query = f"Add a Project kickoff reminder for the {original_time_text} in UTC."
    extraction = _extraction(
        action="add",
        retrieval_text="Project kickoff",
        subject="Project kickoff",
        reminder_summary="Project kickoff",
        raw_reminder="Project kickoff",
        event_time=event_time,
        user_timezone="UTC",
        original_time_text=original_time_text,
        time_semantics="event_time",
    )
    llm = ScriptedLLM(
        [
            extraction,
            _validation(
                operation="add",
                result="EXECUTE",
                selected=[],
                assessments=[],
                should_execute=True,
            ),
            _finalization(
                operation="add",
                selected_candidate_key="",
            ),
        ]
    )
    repository = _repository()

    with canonical_chat_history_scope([]):
        result = _branch(llm).execute(_context(query, []), repository)

    assert result.response_type is ResponseType.REMINDER_ACTION
    assert len(llm.calls) == 3
    rows = repository.list_reminders(user_id=USER_ID)
    assert len(rows) == 1
    # reminder_time is the required pending-planner source placeholder until
    # autoscan derives a notification time; event_time remains independently
    # preserved as the user's explicit event timestamp.
    assert rows[0]["reminder_time"] == event_time
    assert rows[0]["event_time"] == event_time
    assert rows[0]["timing_plan_status"] == "pending"
    pending = repository.list_reminders_requiring_timing()
    assert [candidate.reminder_id for candidate in pending] == [
        rows[0]["reminder_id"]
    ]


@pytest.mark.parametrize(
    ("changed_time_field", "new_time", "time_semantics"),
    (
        ("event_time", "2035-04-06T11:00:00+00:00", "event_time"),
        (
            "notification_time",
            "2035-04-06T07:15:00+00:00",
            "notification_time",
        ),
    ),
)
def test_modify_one_time_keeps_notification_and_event_times_independent(
    changed_time_field: str,
    new_time: str,
    time_semantics: str,
) -> None:
    original_notification = "2035-04-05T08:30:00+00:00"
    original_event = "2035-04-05T10:00:00+00:00"
    repository = _repository()
    reminder_id = _seed_reminder(
        repository,
        subject="Project review",
        reminder_summary="Project review",
        raw_reminder="Project review",
        reminder_time=original_notification,
        event_time=original_event,
        recurrence_rule=None,
        recurrence_timezone=None,
    )
    original = _reminder_row(repository, reminder_id)
    original_time_text = f"{changed_time_field} to {new_time}"
    query = (
        "Modify the Project review reminder: change only "
        f"{original_time_text}."
    )
    extraction = _extraction(
        action="modify",
        retrieval_text="Project review",
        changed_fields=[changed_time_field, "original_time_text"],
        time_semantics=time_semantics,
        original_time_text=original_time_text,
        **{changed_time_field: new_time},
    )
    final_record = _record_from_row(original)
    final_record[changed_time_field] = new_time
    final_record["original_time_text"] = original_time_text
    llm = ScriptedLLM(
        [
            extraction,
            _validation(
                operation="modify",
                result="EXECUTE",
                selected=[reminder_id],
                assessments=[
                    _assessment(
                        reminder_id,
                        matches=True,
                        matched_fields=["subject"],
                        matched_text="Project review",
                    )
                ],
                should_execute=True,
            ),
            _finalization(
                operation="modify",
                selected_candidate_key=reminder_id,
                changed_fields=[changed_time_field, "original_time_text"],
            ),
        ]
    )

    with canonical_chat_history_scope([]):
        result = _branch(llm).execute(_context(query, []), repository)

    assert result.response_type is ResponseType.REMINDER_ACTION
    successors = [
        row
        for row in repository.list_reminders(user_id=USER_ID)
        if row["reminder_id"] != reminder_id
    ]
    assert len(successors) == 1
    successor = successors[0]
    expected_notification = (
        new_time
        if changed_time_field == "notification_time"
        else original_notification
    )
    expected_event = (
        new_time if changed_time_field == "event_time" else original_event
    )
    assert successor["reminder_time"] == expected_notification
    assert successor["event_time"] == expected_event
    assert successor["reminder_time"] != successor["event_time"]


def test_relevance_ranking_considers_exact_older_target_before_candidate_slicing() -> None:
    repository = _repository()
    target_id = _seed_reminder(
        repository,
        subject="Ancient exact target",
        reminder_summary="Ancient exact target",
        raw_reminder="Ancient exact target",
        reminder_time="2030-01-01T09:00:00+00:00",
        event_time="2030-01-01T09:00:00+00:00",
        recurrence_rule=None,
        recurrence_timezone=None,
    )
    for index in range(15):
        _seed_reminder(
            repository,
            subject=f"Unrelated future schedule {index}",
            reminder_summary=f"Unrelated future schedule {index}",
            raw_reminder=f"Unrelated future schedule {index}",
            reminder_time=f"{2040 + index:04d}-01-01T09:00:00+00:00",
            event_time=f"{2040 + index:04d}-01-01T09:00:00+00:00",
            recurrence_rule=None,
            recurrence_timezone=None,
        )

    config = build_assistant_config(ProductionSettings())
    assert repository.table_count("reminders") > (
        config.reminder_resolver.reminder_target_candidate_limit
    )
    target_text = "Ancient exact target"
    ranked = [
        ReminderMutationPipeline._candidate_from_row(
            str(row["reminder_id"]),
            row,
            target_text,
        )
        for row in repository.list_reminders(user_id=USER_ID)
        if row["status"] in config.reminder_resolver.allowed_reminder_delete_statuses
    ]
    ranked.sort(
        key=lambda item: (item.deterministic_score, item.candidate_key),
        reverse=True,
    )
    intended_candidates = ranked[
        : config.retrieval_validation.reminder_llm_validation_max_candidates
    ]
    assert target_id in {candidate.candidate_key for candidate in intended_candidates}
    assessments = [
        _assessment(
            candidate.candidate_key,
            matches=candidate.candidate_key == target_id,
            matched_fields=["subject"] if candidate.candidate_key == target_id else [],
            matched_text=target_text if candidate.candidate_key == target_id else "",
        )
        for candidate in intended_candidates
    ]
    query = "Delete the Ancient exact target reminder."
    llm = ScriptedLLM(
        [
            _extraction(action="delete", retrieval_text=target_text),
            _validation(
                operation="delete",
                result="EXECUTE",
                selected=[target_id],
                assessments=assessments,
                should_execute=True,
            ),
            _finalization(
                operation="delete",
                selected_candidate_key=target_id,
            ),
        ]
    )

    with canonical_chat_history_scope([]):
        result = _branch(llm).execute(_context(query, []), repository)

    assert result.response_type is ResponseType.REMINDER_ACTION
    assert result.actions_pending_confirmation
    assert len(llm.calls) == 3
    validation_candidates = _prompt_payload(llm.calls[1])["extra"][
        "candidate_reminders"
    ]
    assert target_id in {
        candidate["candidate_key"] for candidate in validation_candidates
    }


@pytest.mark.parametrize(
    ("action", "query", "already_status"),
    (
        ("turn_on", "Turn on the Payroll reminder.", "scheduled"),
        ("turn_off", "Turn off the Payroll reminder.", "cancelled"),
    ),
)
def test_toggle_already_in_requested_state_is_stage_two_safe_noop(
    action: str,
    query: str,
    already_status: str,
) -> None:
    repository = _repository()
    reminder_id = _seed_reminder(repository, status=already_status)
    original = _reminder_row(repository, reminder_id)
    llm = ScriptedLLM(
        [
            _toggle_extraction(direction=action, retrieval_text="Payroll"),
            _validation(
                operation=action,
                result="SKIP_ALREADY_EXISTS",
                selected=[reminder_id],
                assessments=[
                    _assessment(
                        reminder_id,
                        matches=True,
                        compatible=False,
                        matched_fields=["subject"],
                        matched_text="Payroll",
                    )
                ],
                should_execute=False,
                reason="The reminder is already in the requested state.",
            ),
        ]
    )

    with canonical_chat_history_scope([]):
        result = _branch(llm).execute(_context(query, []), repository)

    assert result.response_type is ResponseType.REMINDER_ACTION
    assert result.clarification_question is None
    assert result.actions_pending_confirmation == []
    assert len(llm.calls) == 2
    updated = _reminder_row(repository, reminder_id)
    assert updated["status"] == original["status"]
    assert updated["version"] == original["version"]


def test_modify_setting_existing_value_is_stage_two_safe_noop() -> None:
    repository = _repository()
    reminder_id = _seed_reminder(repository, subject="Payroll")
    original = _reminder_row(repository, reminder_id)
    query = "Modify the Payroll reminder subject to Payroll."
    llm = ScriptedLLM(
        [
            _extraction(
                action="modify",
                retrieval_text="Payroll",
                changed_fields=["subject"],
                subject="Payroll",
            ),
            _validation(
                operation="modify",
                result="SKIP_ALREADY_EXISTS",
                selected=[reminder_id],
                assessments=[
                    _assessment(
                        reminder_id,
                        matches=True,
                        matched_fields=["subject"],
                        matched_text="Payroll",
                    )
                ],
                should_execute=False,
                reason="The requested subject already has that value.",
            ),
        ]
    )

    with canonical_chat_history_scope([]):
        result = _branch(llm).execute(_context(query, []), repository)

    assert result.response_type is ResponseType.REMINDER_ACTION
    assert result.clarification_question is None
    assert result.actions_pending_confirmation == []
    assert len(llm.calls) == 2
    updated = _reminder_row(repository, reminder_id)
    assert updated["status"] == original["status"]
    assert updated["version"] == original["version"]
    assert repository.table_count("reminders") == 1


def test_compatible_and_incompatible_same_target_requires_ambiguity_hitl() -> None:
    repository = _repository()
    compatible_id = _seed_reminder(
        repository,
        subject="Payroll",
        reminder_summary="Payroll",
        raw_reminder="Payroll",
        status="cancelled",
    )
    incompatible_id = _seed_reminder(
        repository,
        subject="Payroll",
        reminder_summary="Payroll",
        raw_reminder="Payroll",
        status="scheduled",
    )
    compatible_before = _reminder_row(repository, compatible_id)
    incompatible_before = _reminder_row(repository, incompatible_id)
    query = "Turn on the Payroll reminder."
    llm = ScriptedLLM(
        [
            _toggle_extraction(direction="turn_on", retrieval_text="Payroll"),
            _validation(
                operation="turn_on",
                result="CLARIFY_AMBIGUOUS_TARGET",
                selected=[],
                assessments=[
                    _assessment(
                        compatible_id,
                        matches=True,
                        compatible=True,
                        matched_fields=["subject"],
                        matched_text="Payroll",
                    ),
                    _assessment(
                        incompatible_id,
                        matches=True,
                        compatible=False,
                        matched_fields=["subject"],
                        matched_text="Payroll",
                    ),
                ],
                should_execute=False,
                requires_hitl=True,
                ambiguous=True,
                hitl_reason="ambiguous_target",
                reason="Two reminder rows share the requested target.",
            ),
        ]
    )

    with canonical_chat_history_scope([]):
        result = _branch(llm).execute(_context(query, []), repository)

    assert result.response_type is ResponseType.CLARIFICATION
    assert result.clarification_question is not None
    assert len(llm.calls) == 2
    compatible_after = _reminder_row(repository, compatible_id)
    incompatible_after = _reminder_row(repository, incompatible_id)
    assert compatible_after["status"] == compatible_before["status"]
    assert compatible_after["version"] == compatible_before["version"]
    assert incompatible_after["status"] == incompatible_before["status"]
    assert incompatible_after["version"] == incompatible_before["version"]


def test_validation_and_finalization_candidates_include_recurrence_linkage() -> None:
    repository = _repository()
    parent_id = _seed_reminder(
        repository,
        subject="Payroll series parent",
        reminder_summary="Payroll series parent",
        raw_reminder="Payroll series parent",
    )
    next_fire_time = "2035-05-05T09:30:00+00:00"
    child_id = _seed_reminder(
        repository,
        subject="Payroll series child",
        reminder_summary="Payroll series child",
        raw_reminder="Payroll series child",
        next_fire_time=next_fire_time,
        parent_recurring_reminder_id=parent_id,
    )
    query = "Turn off the Payroll series child reminder."
    llm = ScriptedLLM(
        [
            _toggle_extraction(
                direction="turn_off",
                retrieval_text="Payroll series child",
            ),
            _validation(
                operation="turn_off",
                result="EXECUTE",
                selected=[child_id],
                assessments=[
                    _assessment(
                        child_id,
                        matches=True,
                        matched_fields=["subject"],
                        matched_text="Payroll series child",
                    ),
                    _assessment(parent_id, matches=False),
                ],
                should_execute=True,
            ),
            _finalization(
                operation="turn_off",
                selected_candidate_key=child_id,
            ),
        ]
    )

    with canonical_chat_history_scope([]):
        result = _branch(llm).execute(_context(query, []), repository)

    assert result.response_type is ResponseType.REMINDER_ACTION
    assert len(llm.calls) == 3
    validation_candidates = _prompt_payload(llm.calls[1])["extra"][
        "candidate_reminders"
    ]
    child = next(
        candidate
        for candidate in validation_candidates
        if candidate["candidate_key"] == child_id
    )
    assert child["next_fire_time"] == next_fire_time
    assert child["parent_recurring_reminder_id"] == parent_id

    finalization_candidate = _prompt_payload(llm.calls[2])["extra"][
        "selected_candidate_manifest"
    ]
    assert finalization_candidate["candidate_key"] == child_id
    assert finalization_candidate["next_fire_time_present"] is True
    assert finalization_candidate["parent_recurring_reminder_present"] is True


@pytest.mark.parametrize(
    ("action", "query", "initial_status", "expected_status", "pending"),
    (
        (
            "delete",
            "Delete the Legacy timezone reminder.",
            "scheduled",
            "scheduled",
            True,
        ),
        (
            "turn_off",
            "Turn off the Legacy timezone reminder.",
            "scheduled",
            "cancelled",
            False,
        ),
        (
            "turn_on",
            "Turn on the Legacy timezone reminder.",
            "cancelled",
            "scheduled",
            False,
        ),
    ),
)
def test_target_only_actions_do_not_normalize_legacy_invalid_timezone(
    action: str,
    query: str,
    initial_status: str,
    expected_status: str,
    pending: bool,
) -> None:
    repository = _repository()
    reminder_id = _seed_reminder(
        repository,
        subject="Legacy timezone",
        reminder_summary="Legacy timezone",
        raw_reminder="Legacy timezone",
        status=initial_status,
        user_timezone="Legacy/Invalid",
        recurrence_timezone="Legacy/Invalid",
    )
    original = _reminder_row(repository, reminder_id)
    llm = ScriptedLLM(
        [
            (
                _toggle_extraction(
                    direction=action,
                    retrieval_text="Legacy timezone",
                )
                if action in {"turn_on", "turn_off"}
                else _extraction(
                    action=action,
                    retrieval_text="Legacy timezone",
                )
            ),
            _validation(
                operation=action,
                result="EXECUTE",
                selected=[reminder_id],
                assessments=[
                    _assessment(
                        reminder_id,
                        matches=True,
                        matched_fields=["subject"],
                        matched_text="Legacy timezone",
                    )
                ],
                should_execute=True,
            ),
            _finalization(
                operation=action,
                selected_candidate_key=reminder_id,
            ),
        ]
    )

    with canonical_chat_history_scope([]):
        result = _branch(llm).execute(_context(query, []), repository)

    assert result.response_type is ResponseType.REMINDER_ACTION
    assert result.clarification_question is None
    assert bool(result.actions_pending_confirmation) is pending
    assert len(llm.calls) == 3
    assert _reminder_row(repository, reminder_id)["status"] == expected_status


def test_generated_supporting_question_does_not_overwrite_finalized_fields() -> None:
    finalized_question = "Which checklist should I use?"
    finalized_response = "Use checklist A"
    generated_question = "Should this reminder repeat next quarter?"
    query = (
        "Add a Submit payroll reminder at "
        f"{BASE_TIME} in UTC. Supporting question: {finalized_question} "
        f"Supporting response: {finalized_response}."
    )
    extraction = _extraction(
        action="add",
        retrieval_text="Submit payroll",
        subject="Submit payroll",
        reminder_summary="Submit payroll",
        raw_reminder="Submit payroll",
        notification_time=BASE_TIME,
        event_time=BASE_TIME,
        user_timezone="UTC",
        original_time_text=BASE_TIME,
        supporting_question=finalized_question,
        supporting_response=finalized_response,
        time_semantics="both",
    )
    llm = ScriptedLLM(
        [
            extraction,
            _validation(
                operation="add",
                result="EXECUTE",
                selected=[],
                assessments=[],
                should_execute=True,
            ),
            _finalization(
                operation="add",
                selected_candidate_key="",
            ),
        ]
    )

    class SupportingStrategy:
        def generate(self, *_args: Any, **_kwargs: Any) -> GeneratedQuestion:
            return GeneratedQuestion(
                text=generated_question,
                source=QuestionSource.REMINDER_SUPPORTING_QUESTION,
                purpose="optional_context",
                confidence=1.0,
                expected_response_type=(
                    ExpectedResponseType.REMINDER_FOLLOWUP_ANSWER
                ),
            )

    repository = _repository()
    branch = _branch(llm)
    branch.reminder_supporting_strategy = SupportingStrategy()

    with canonical_chat_history_scope([]):
        result = branch.execute(_context(query, []), repository)

    assert result.response_type is ResponseType.REMINDER_ACTION
    assert result.reminder_supporting_question is not None
    assert result.reminder_supporting_question.text == generated_question
    row = repository.list_reminders(user_id=USER_ID)[0]
    assert row["supporting_question"] == finalized_question
    assert row["supporting_response"] == finalized_response
    assert row["supporting_question"] != generated_question


def test_foreign_user_reminder_never_reaches_validation_prompt() -> None:
    repository = _repository()
    _seed_reminder(
        repository,
        user_id="another-user",
        subject="Foreign payroll",
        reminder_summary="Foreign payroll",
        raw_reminder="Foreign payroll",
    )
    query = "Delete the Missing payroll reminder."
    llm = ScriptedLLM(
        [
            _extraction(action="delete", retrieval_text="Missing payroll"),
            _validation(
                operation="delete",
                result="SKIP_NOT_FOUND",
                selected=[],
                assessments=[],
                should_execute=False,
            ),
        ]
    )

    with canonical_chat_history_scope([]):
        result = _branch(llm).execute(_context(query, []), repository)

    assert result.response_type is ResponseType.REMINDER_ACTION
    validation_payload = _prompt_payload(llm.calls[1])
    assert validation_payload["extra"]["candidate_reminders"] == []
    assert repository.table_count("reminders") == 1
