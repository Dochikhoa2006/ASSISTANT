from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

import pytest

from assistant_rag.autoscan import ReminderAutoscan
from assistant_rag.branches import ReminderBranch
from assistant_rag.chat_history import canonical_chat_history_scope
from assistant_rag.contracts import (
    ChatRequest,
    Intent,
    PipelineContext,
    ReminderAction,
    ResponseType,
)
from assistant_rag.database import SQLiteRepository
from assistant_rag.llm import (
    LLMTask,
    structured_fallback_payload,
    validate_json_schema,
)
from assistant_rag.production_factory import build_assistant_config
from assistant_rag.prompts import (
    DEFAULT_PROMPT_REGISTRY,
    REMINDER_ACTION_EXTRACTION_SCHEMA,
    REMINDER_ACTION_VALIDATION_SCHEMA,
    REMINDER_CONTENT_FINALIZATION_SCHEMA,
)
from assistant_rag.reminder_mutation import (
    REMINDER_EDITABLE_FIELDS,
    LLMReminderActionDetector,
    ReminderActionValidationStrategy,
    ReminderContentFinalizationStrategy,
    ReminderMutationCandidate,
    ReminderMutationPipeline,
)
from assistant_rag.reminder_timing import ReminderTimingPlanner
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


def test_all_reminder_llm_contracts_are_minimal_and_have_no_clarification_component() -> None:
    assert REMINDER_ACTION_EXTRACTION_SCHEMA["required"] == [
        "action",
        "retrieval_text",
        "field_values",
        "confidence",
    ]
    extraction_fields = set(
        REMINDER_ACTION_EXTRACTION_SCHEMA["properties"]["field_values"]
        ["items"]["properties"]["field"]["enum"]
    )
    assert {
        "notification_time",
        "event_time",
        "user_timezone",
        "original_time_text",
    } <= extraction_fields
    assert {"supporting_question", "supporting_response"}.isdisjoint(
        extraction_fields
    )
    properties = REMINDER_ACTION_VALIDATION_SCHEMA["properties"]
    decisions = set(properties["validation_result"]["enum"])

    assert REMINDER_ACTION_VALIDATION_SCHEMA["required"] == [
        "validation_result",
        "selected_candidate_keys",
        "confidence",
        "clarification_question",
        "candidate_assessments",
    ]
    assessment_properties = properties["candidate_assessments"]["items"][
        "properties"
    ]
    assert set(assessment_properties) == {
        "candidate_key",
        "match_kind",
        "confidence",
        "evidence_field",
        "matched_text",
    }
    assert not decisions & {
        "EXECUTE",
        "SKIP_NOT_FOUND",
        "SKIP_ALREADY_EXISTS",
        "REJECT_UNSAFE_TRANSITION",
        "CLARIFY_AMBIGUOUS_TARGET",
        "CLARIFY_MISSING_FIELDS",
    }
    assert decisions == {"PASS", "FAIL"}
    assert not {
        "ambiguous",
        "requires_hitl",
        "factuality_concern",
        "hitl_reason",
        "operation",
        "should_execute",
        "reason_summary",
    } & set(properties)
    assert REMINDER_CONTENT_FINALIZATION_SCHEMA["required"] == [
        "approved",
        "confidence",
    ]
    assert "clarification_strategy" not in ReminderBranch.__dataclass_fields__
    assert "validated_action_builder" not in ReminderBranch.__dataclass_fields__
    assert "reminder_supporting_strategy" not in ReminderBranch.__dataclass_fields__


@pytest.mark.parametrize(
    ("task", "schema", "expected_keys"),
    (
        (
            LLMTask.REMINDER_ACTION_EXTRACTION,
            REMINDER_ACTION_EXTRACTION_SCHEMA,
            {"action", "retrieval_text", "field_values", "confidence"},
        ),
        (
            LLMTask.REMINDER_ACTION_VALIDATION,
            REMINDER_ACTION_VALIDATION_SCHEMA,
            {
                "validation_result",
                "selected_candidate_keys",
                "confidence",
                "clarification_question",
                "candidate_assessments",
            },
        ),
        (
            LLMTask.REMINDER_CONTENT_FINALIZATION,
            REMINDER_CONTENT_FINALIZATION_SCHEMA,
            {"approved", "confidence"},
        ),
    ),
)
def test_reminder_structured_fallbacks_match_minimal_schemas(
    task: LLMTask,
    schema: dict[str, Any],
    expected_keys: set[str],
) -> None:
    fallback = structured_fallback_payload(
        task=task,
        schema=schema,
        user_prompt="Runtime context:\n{}",
        error=ValueError("invalid output"),
    )

    validate_json_schema(fallback, schema)
    assert set(fallback) == expected_keys


def test_reminder_repository_has_no_standalone_duplicate_detector() -> None:
    assert not hasattr(_repository(), "find_active_reminder_duplicates")


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
    returned_action = (
        toggle_direction if action == "toggle" and toggle_direction else action
    )
    supplied_fields = (
        [
            field
            for field in REMINDER_EDITABLE_FIELDS
            if action == "add" and record[field]
        ]
        if action == "add"
        else list(changed_fields)
        if changed_fields is not None
        else []
    )
    response = {
        "action": returned_action,
        "retrieval_text": retrieval_text,
        "field_values": [
            {"field": field, "value": record[field]}
            for field in supplied_fields
        ],
        "confidence": confidence,
    }
    if action != "toggle" and toggle_direction:
        response["toggle_direction"] = toggle_direction
    return response


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
    reason: str = "The reminder action was validated.",
    clarification_question: str | None = None,
) -> dict[str, Any]:
    decision = (
        result
        if result in {"PASS", "FAIL"}
        else "PASS"
        if result == "EXECUTE"
        else "FAIL"
    )
    return {
        "validation_result": decision,
        "selected_candidate_keys": selected,
        "confidence": confidence,
        "clarification_question": (
            ""
            if decision == "PASS"
            else clarification_question
            or "Could you clarify the reminder change you want?"
        ),
        "candidate_assessments": [
            {
                "candidate_key": item["candidate_key"],
                "match_kind": (
                    "EQUIVALENT"
                    if operation == "add" and item["matches_target"]
                    else "TARGET"
                    if item["matches_target"]
                    else "NONE"
                ),
                "confidence": item["confidence"],
                "evidence_field": (
                    item["matched_fields"][0]
                    if item["matched_fields"]
                    else ""
                ),
                "matched_text": item["matched_text"],
            }
            for item in assessments
        ],
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
    return {
        "approved": True,
        "confidence": confidence,
    }


def test_add_runs_two_stages_with_evidence_isolated_validation_prompt() -> None:
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
                result="PASS",
                selected=[],
                assessments=[],
                should_execute=True,
            ),
        ]
    )
    repository = _repository()
    forbidden_raw_query = "FORBIDDEN_RAW_REMINDER_QUERY"
    context = replace(
        _context(forbidden_raw_query, history),
        rewritten_query=query,
    )
    context.request.metadata["leak_probe"] = "FORBIDDEN_REMINDER_METADATA"
    context.request.platform_context["leak_probe"] = (
        "FORBIDDEN_REMINDER_PLATFORM_CONTEXT"
    )

    with canonical_chat_history_scope(history):
        result = _branch(llm).execute(context, repository)

    assert result.response_type is ResponseType.REMINDER_ACTION
    assert [call["task"] for call in llm.calls] == [
        LLMTask.REMINDER_ACTION_EXTRACTION,
        LLMTask.REMINDER_ACTION_VALIDATION,
    ]
    extraction_prompt = _prompt_payload(llm.calls[0])
    assert "raw_query" not in extraction_prompt
    assert extraction_prompt["rewritten_query"] == query
    assert extraction_prompt["chat_history"] == history
    assert forbidden_raw_query not in json.dumps(extraction_prompt)
    validation_payload = _prompt_payload(llm.calls[1])
    assert validation_payload == {
        "first_model_response": extraction,
        "reminder_retrieval": [],
    }
    serialized_validation = json.dumps(validation_payload)
    assert query not in serialized_validation
    assert forbidden_raw_query not in serialized_validation
    assert "FORBIDDEN_REMINDER_METADATA" not in serialized_validation
    assert "FORBIDDEN_REMINDER_PLATFORM_CONTEXT" not in serialized_validation
    assert "chat_history" not in validation_payload
    assert "user_id" not in validation_payload
    assert "metadata" not in validation_payload
    assert "platform_context" not in validation_payload
    assert "extra" not in validation_payload
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
        "reminder_action_extraction_response": {
            "action": "delete",
            "retrieval_text": "poisoned extraction state",
        },
        "action_authorization": {"action": "delete"},
        "confirmation_approved": False,
    }
    llm = ScriptedLLM(
        [
            _add_extraction(subject="Payroll", reminder_time=BASE_TIME),
            _validation(
                operation="add",
                result="PASS",
                selected=[],
                assessments=[],
                should_execute=True,
            ),
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

    assert result.response_type is ResponseType.SAFE_NOOP
    assert result.clarification_question is None
    assert len(llm.calls) == 1
    assert llm.calls[0]["task"] is LLMTask.REMINDER_ACTION_EXTRACTION


def test_modify_model_three_receives_only_model_one_state_and_preserves_sql_fields() -> None:
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
    extraction = _extraction(
        action="modify",
        retrieval_text=supporting_tail,
        changed_fields=["subject"],
        subject="Final payroll review",
    )
    llm = ScriptedLLM(
        [
            extraction,
            _validation(
                operation="modify",
                result="PASS",
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
    assert finalization_payload == {"first_model_response": extraction}
    assert set(finalization_payload) == {"first_model_response"}
    assert raw_reminder not in finalization_serialized
    assert supporting_question not in finalization_serialized
    assert query not in finalization_serialized
    assert "history-long" not in finalization_serialized
    assert "chat_history" not in finalization_serialized
    assert "reminder_retrieval" not in finalization_serialized
    assert "selected_candidate" not in finalization_serialized
    assert "validation" not in finalization_serialized
    assert "metadata" not in finalization_serialized
    assert "platform_context" not in finalization_serialized

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
        "rejected",
        "low_confidence",
        "missing_approved",
        "wrong_type",
        "extra_root_field",
    ),
)
def test_finalization_requires_one_exact_high_confidence_approval(
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
        operation="modify",
        selected_candidate_key=reminder_id,
    )
    if failure == "rejected":
        finalization["approved"] = False
    elif failure == "low_confidence":
        finalization["confidence"] = 0.1
    elif failure == "missing_approved":
        finalization.pop("approved")
    elif failure == "wrong_type":
        finalization["approved"] = "yes"
    else:
        finalization["field_bindings"] = []

    llm = ScriptedLLM(
        [
            _extraction(
                action="modify",
                retrieval_text="Binding contract target",
                changed_fields=["subject"],
                subject="Updated binding contract target",
            ),
            _validation(
                operation="modify",
                result="PASS",
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
            _context(
                "Modify the Binding contract target reminder subject to "
                "Updated binding contract target.",
                [],
            ),
            repository,
        )

    assert result.response_type is ResponseType.ERROR
    assert result.clarification_question is None
    assert len(llm.calls) == 3
    after = _reminder_row(repository, reminder_id)
    assert after["status"] == before["status"]
    assert after["version"] == before["version"]


def test_delete_executes_immediately_without_pending_confirmation() -> None:
    repository = _repository()
    reminder_id = _seed_reminder(
        repository,
        subject="Legacy review",
        reminder_summary="Legacy review",
        raw_reminder="Review the legacy process.",
        recurrence_rule=None,
        recurrence_timezone=None,
    )
    query = "Delete the Legacy review reminder."
    history = [{"hop_id": "delete-history", "text": "Legacy review context"}]
    llm = ScriptedLLM(
        [
            _extraction(action="delete", retrieval_text="Legacy review"),
            _validation(
                operation="delete",
                result="PASS",
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
        ]
    )
    with canonical_chat_history_scope(history):
        result = _branch(llm).execute(_context(query, history), repository)

    assert result.response_type is ResponseType.REMINDER_ACTION
    assert result.actions_pending_confirmation == []
    assert [call["task"] for call in llm.calls] == [
        LLMTask.REMINDER_ACTION_EXTRACTION,
        LLMTask.REMINDER_ACTION_VALIDATION,
    ]
    extraction_payload = _prompt_payload(llm.calls[0])
    assert "confirmation_replay" not in extraction_payload["extra"]
    assert "trusted_confirmation_action_context" not in extraction_payload["extra"]
    validation_payload = _prompt_payload(llm.calls[1])
    assert set(validation_payload) == {
        "first_model_response",
        "reminder_retrieval",
    }
    assert "chat_history" not in validation_payload
    assert "delete-history" not in json.dumps(validation_payload)
    assert _reminder_row(repository, reminder_id)["status"] == "dismissed"
    assert repository.connection.execute(
        "SELECT * FROM pending_action_confirmations"
    ).fetchall() == []


@pytest.mark.parametrize(
    ("action", "query", "initial_status", "expected_status"),
    (
        ("turn_on", "Turn on the Payroll reminder.", "cancelled", "scheduled"),
        ("turn_off", "Turn off the Payroll reminder.", "scheduled", "cancelled"),
    ),
)
def test_toggle_directions_run_two_stages_and_only_change_status(
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
                result="PASS",
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
        ]
    )

    with canonical_chat_history_scope([]):
        result = _branch(llm).execute(_context(query, []), repository)

    assert result.response_type is ResponseType.REMINDER_ACTION
    assert [call["task"] for call in llm.calls] == [
        LLMTask.REMINDER_ACTION_EXTRACTION,
        LLMTask.REMINDER_ACTION_VALIDATION,
    ]
    updated = _reminder_row(repository, reminder_id)
    assert updated["status"] == expected_status
    assert _record_from_row(updated) == original_record


@pytest.mark.parametrize("direction", ("turn_on", "turn_off"))
def test_first_llm_emits_one_direct_lifecycle_action(
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
    assert action["extracted_action"] == direction
    assert action["toggle_direction"] == direction
    assert action["action"] == direction
    prompt = _prompt_payload(llm.calls[0])
    assert prompt["extra"]["allowed_actions"] == [
        "add",
        "delete",
        "modify",
        "turn_on",
        "turn_off",
    ]
    assert "toggle_directions" not in prompt["extra"]


@pytest.mark.parametrize(
    ("payload", "expected_risk"),
    (
        (
            {**_extraction(action="turn_on", retrieval_text="Payroll"), "action": "toggle"},
            "invalid_reminder_action_extraction",
        ),
        (
            {**_extraction(action="turn_on", retrieval_text="Payroll"), "action": "enable"},
            "invalid_reminder_action_extraction",
        ),
        (
            _extraction(
                action="add",
                toggle_direction="turn_on",
                retrieval_text="Payroll",
            ),
            "invalid_reminder_action_extraction",
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


def test_first_llm_modify_carries_every_mutable_reminder_section_in_one_action() -> None:
    notification_time = "2035-04-05T08:30:00+00:00"
    event_time = "2035-04-05T10:00:00+00:00"
    original_time_text = (
        f"notify at {notification_time} and event at {event_time}"
    )
    query = (
        "Modify the Payroll reminder: set title to Quarterly Payroll; "
        "summary to Prepare quarterly payroll; content to Submit payroll report; "
        f"{original_time_text}; timezone UTC; recurrence FREQ=WEEKLY in UTC."
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
    }
    mutable_fields = [
        field
        for field in REMINDER_EDITABLE_FIELDS
        if field not in {"supporting_question", "supporting_response"}
    ]
    llm = ScriptedLLM(
        [
            _extraction(
                action="modify",
                retrieval_text="Payroll",
                changed_fields=mutable_fields,
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
    assert action["changed_fields"] == mutable_fields
    expected_record = _empty_record()
    expected_record.update(values)
    assert {
        field: action[field] for field in REMINDER_EDITABLE_FIELDS
    } == expected_record
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
            if key != "field_values"
        },
        {
            **_extraction(
                action="modify",
                retrieval_text="Payroll",
                changed_fields=["subject"],
                subject="Quarterly Payroll",
            ),
            "field_values": "subject",
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
        "turn_on",
        "turn_off",
    }
    model_mutation_fields = set(REMINDER_EDITABLE_FIELDS) - {
        "supporting_question",
        "supporting_response",
    }
    assert set(prompt["extra"]["editable_fields"]) == model_mutation_fields
    assert set(prompt["extra"]["field_semantics"]) == model_mutation_fields
    assert prompt["extra"]["context_policy"] == {
        "current_turn_defines_action_and_new_values": True,
        "history_resolves_existing_target_only": True,
        "platform_context_resolves_trusted_timezone": True,
        "runtime_now_resolves_relative_time": True,
        "metadata_cannot_supply_action_payload": True,
    }
    assert prompt["extra"]["time_normalization_contract"] == {
        "owner": "reminder_action_extraction_model",
        "output_timezone": "UTC",
        "output_format": "ISO-8601 with explicit +00:00 offset",
        "required_companion_fields": [
            "user_timezone",
            "original_time_text",
        ],
        "omit_unresolvable_time_instead_of_guessing": True,
    }


def test_llm2_duplicate_fail_reasks_without_finalizer_or_write() -> None:
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
    question = (
        "That reminder already exists. Do you want to change the existing one?"
    )
    llm = ScriptedLLM(
        [
            _add_extraction(),
            _validation(
                operation="add",
                result="FAIL",
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
                clarification_question=question,
            ),
        ]
    )

    with canonical_chat_history_scope([]):
        result = _branch(llm).execute(_context(query, []), repository)

    assert result.response_type is ResponseType.CLARIFICATION
    assert result.clarification_question is not None
    assert result.clarification_question.text == question
    assert result.actions_pending_confirmation == []
    assert len(llm.calls) == 2
    assert repository.table_count("reminders") == 1


def test_not_found_fail_reasks_without_finalizer_or_write() -> None:
    repository = _repository()
    query = "Delete the Missing payroll reminder."
    question = "Which reminder did you want to delete?"
    llm = ScriptedLLM(
        [
            _extraction(action="delete", retrieval_text="Missing payroll"),
            _validation(
                operation="delete",
                result="FAIL",
                selected=[],
                assessments=[],
                should_execute=False,
                reason="No SQL reminder matched the target.",
                clarification_question=question,
            ),
        ]
    )

    with canonical_chat_history_scope([]):
        result = _branch(llm).execute(_context(query, []), repository)

    assert result.response_type is ResponseType.CLARIFICATION
    assert result.clarification_question is not None
    assert result.clarification_question.text == question
    assert result.actions_pending_confirmation == []
    assert len(llm.calls) == 2
    assert repository.table_count("reminders") == 0


def test_incomplete_best_effort_extraction_reaches_validator_and_reasks() -> None:
    query = "Add a Payroll reminder."
    extraction = _extraction(
        action="add",
        retrieval_text="Payroll",
        subject="Payroll",
        confidence=0.91,
        missing_fields=["raw_reminder", "notification_time"],
    )
    question = "What should the Payroll reminder say, and when should it notify you?"
    llm = ScriptedLLM(
        [
            extraction,
            _validation(
                operation="add",
                result="FAIL",
                selected=[],
                assessments=[],
                should_execute=False,
                reason="The best-effort extraction is incomplete.",
                clarification_question=question,
            ),
        ]
    )
    repository = _repository()

    with canonical_chat_history_scope([]):
        result = _branch(llm).execute(_context(query, []), repository)

    assert result.response_type is ResponseType.CLARIFICATION
    assert result.clarification_question is not None
    assert result.clarification_question.text == question
    assert [call["task"] for call in llm.calls] == [
        LLMTask.REMINDER_ACTION_EXTRACTION,
        LLMTask.REMINDER_ACTION_VALIDATION,
    ]
    assert _prompt_payload(llm.calls[1]) == {
        "first_model_response": extraction,
        "reminder_retrieval": [],
    }
    assert repository.table_count("reminders") == 0


def test_unsafe_factuality_fail_reasks_without_finalizer_or_write() -> None:
    fact = "1 + 1 = 3"
    query = f"Add a reminder: {fact} at {BASE_TIME} in UTC."
    extraction = _add_extraction(subject=fact)
    question = "Did you mean to use the statement 1 + 1 = 3 in this reminder?"
    llm = ScriptedLLM(
        [
            extraction,
            _validation(
                operation="add",
                result="FAIL",
                selected=[],
                assessments=[],
                should_execute=False,
                reason="The asserted arithmetic is false.",
                clarification_question=question,
            ),
        ]
    )
    repository = _repository()

    with canonical_chat_history_scope([]):
        result = _branch(llm).execute(_context(query, []), repository)

    assert result.response_type is ResponseType.CLARIFICATION
    assert result.clarification_question is not None
    assert result.clarification_question.text == question
    assert len(llm.calls) == 2
    assert repository.table_count("reminders") == 0


@pytest.mark.parametrize("failure", ("missing_assessment", "invented_id", "low_confidence"))
def test_invalid_validation_contracts_error_before_finalization(
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
                result="PASS",
                selected=selected,
                assessments=assessments,
                should_execute=True,
                confidence=confidence,
            ),
        ]
    )

    with canonical_chat_history_scope([]):
        result = _branch(llm).execute(_context(query, []), repository)

    assert result.response_type is ResponseType.ERROR
    assert result.clarification_question is None
    assert len(llm.calls) == 2
    assert _reminder_row(repository, reminder_id)["status"] == "scheduled"


def test_first_guard_rejects_ungrounded_replacement_before_isolated_validator() -> None:
    query = "Modify the Payroll reminder."
    history = [{"text": "The new title should be Quarterly payroll"}]
    payload = _extraction(
        action="modify",
        retrieval_text="Payroll",
        changed_fields=["subject"],
        subject="Quarterly payroll",
    )
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
    assert "reminder_content_not_grounded_in_current_turn" in result.risk_flags
    assert len(llm.calls) == 1


def test_pipeline_guard_one_rejects_divergent_reminder_action_state_before_validation() -> None:
    first_response = _add_extraction()
    values = _empty_record()
    values.update(
        {
            item["field"]: item["value"]
            for item in first_response["field_values"]
        }
    )
    action_payload = {
        "action": "add",
        "extracted_action": "add",
        "toggle_direction": "",
        "retrieval_text": "A divergent target model 1 did not return",
        "changed_fields": [],
        "time_semantics": "both",
        "runtime_now_utc": "2035-04-01T00:00:00+00:00",
        "confidence": 0.99,
        **values,
    }
    context = _context("Add the Submit payroll reminder.", [])
    context.request.metadata["reminder_action_extraction_response"] = (
        first_response
    )
    llm = ScriptedLLM([])
    branch = _branch(llm)

    validated = branch.reminder_mutation_pipeline.build_action(
        context=context,
        action_payload=action_payload,
        repository=_repository(),
    )

    assert validated.hitl_reason == "internal_pipeline_failure"
    assert "guard 1" in validated.reason_summary.casefold()
    assert llm.calls == []


def test_first_guard_forwards_grounded_action_field_mismatch_to_model_two() -> None:
    query = "Delete the Payroll reminder."
    extraction = _extraction(
        action="delete",
        retrieval_text="Payroll",
        changed_fields=["subject"],
        subject="Payroll",
    )
    question = "Do you want to delete the reminder or change its subject?"
    llm = ScriptedLLM(
        [
            extraction,
            _validation(
                operation="delete",
                result="FAIL",
                selected=[],
                assessments=[],
                should_execute=False,
                confidence=0.20,
                clarification_question=question,
            ),
        ]
    )
    repository = _repository()
    reminder_id = _seed_reminder(repository, subject="Payroll")

    with canonical_chat_history_scope([]):
        result = _branch(llm).execute(_context(query, []), repository)

    assert result.response_type is ResponseType.CLARIFICATION
    assert result.clarification_question is not None
    assert result.clarification_question.text == question
    assert [call["task"] for call in llm.calls] == [
        LLMTask.REMINDER_ACTION_EXTRACTION,
        LLMTask.REMINDER_ACTION_VALIDATION,
    ]
    assert _prompt_payload(llm.calls[1])["first_model_response"] == extraction
    assert _reminder_row(repository, reminder_id)["status"] == "scheduled"


def test_second_guard_accepts_low_confidence_partial_fail_with_question() -> None:
    config = build_assistant_config(ProductionSettings())
    llm = ScriptedLLM(
        [
            _validation(
                operation="delete",
                result="FAIL",
                selected=[],
                assessments=[],
                should_execute=False,
                confidence=0.20,
            )
        ]
    )
    validator = ReminderActionValidationStrategy(
        config=config,
        llm=llm,
        prompts=DEFAULT_PROMPT_REGISTRY,
    )
    fields = _empty_record()
    fields.update(
        subject="Payroll",
        reminder_summary="Prepare payroll",
        raw_reminder="Remind me to prepare payroll.",
    )
    candidate = ReminderMutationCandidate(
        candidate_key="reminder-1",
        status="scheduled",
        version=1,
        fields=fields,
        deterministic_score=1.0,
    )
    first_response = _extraction(action="delete", retrieval_text="Payroll")
    action_payload = {
        "action": "delete",
        "changed_fields": [],
        "time_semantics": "unchanged",
        **_empty_record(),
    }

    result = validator.validate(
        context=_context("FORBIDDEN_QUERY", []),
        action_payload=action_payload,
        candidates=[candidate],
        first_model_response=first_response,
    )

    assert result.validation_result.value == "clarify_missing_fields"
    assert result.confidence == 0.20
    assert result.hitl_reason == "reminder_validation_fail"
    assert result.clarification_question == (
        "Could you clarify the reminder change you want?"
    )
    assert result.candidate_assessments == ()


@pytest.mark.parametrize(
    ("decision", "question"),
    (
        ("FAIL", ""),
        ("PASS", "Why should this pass?"),
    ),
)
def test_second_guard_errors_on_invalid_decision_question_pair(
    decision: str,
    question: str,
) -> None:
    extraction = _add_extraction()
    validation = _validation(
        operation="add",
        result=decision,
        selected=[],
        assessments=[],
        should_execute=decision == "PASS",
    )
    validation["clarification_question"] = question
    llm = ScriptedLLM([extraction, validation])
    repository = _repository()
    query = f"Add a reminder: Submit payroll at {BASE_TIME} in UTC."

    with canonical_chat_history_scope([]):
        result = _branch(llm).execute(_context(query, []), repository)

    assert result.response_type is ResponseType.ERROR
    assert result.clarification_question is None
    assert len(llm.calls) == 2
    assert repository.table_count("reminders") == 0


def test_model_two_can_authorize_complete_low_confidence_model_one_state() -> None:
    repository = _repository()
    reminder_id = _seed_reminder(repository, subject="Payroll")
    query = "Delete the Payroll reminder."
    llm = ScriptedLLM(
        [
            _extraction(
                action="delete",
                retrieval_text="Payroll",
                confidence=0.20,
            ),
            _validation(
                operation="delete",
                result="PASS",
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
        ]
    )

    with canonical_chat_history_scope([]):
        result = _branch(llm).execute(_context(query, []), repository)

    assert result.response_type is ResponseType.REMINDER_ACTION
    assert len(llm.calls) == 2
    assert _reminder_row(repository, reminder_id)["status"] == "dismissed"


@pytest.mark.parametrize(
    "invalid_time",
    (
        "not-a-calendar-time",
        "2035-04-05T09:30:00",
        "2035-04-05T09:30:00+07:00",
    ),
)
def test_second_guard_errors_on_pass_for_non_normalized_model_one_time(
    invalid_time: str,
) -> None:
    query = f"Add Payroll at {invalid_time} in UTC."
    extraction = _extraction(
        action="add",
        retrieval_text="Payroll",
        subject="Payroll",
        raw_reminder="Payroll",
        notification_time=invalid_time,
        user_timezone="UTC",
        original_time_text=invalid_time,
    )
    llm = ScriptedLLM(
        [
            extraction,
            _validation(
                operation="add",
                result="PASS",
                selected=[],
                assessments=[],
                should_execute=True,
            ),
        ]
    )
    repository = _repository()

    with canonical_chat_history_scope([]):
        result = _branch(llm).execute(_context(query, []), repository)

    assert result.response_type is ResponseType.ERROR
    assert len(llm.calls) == 2
    assert _prompt_payload(llm.calls[1])["first_model_response"] == extraction
    assert repository.table_count("reminders") == 0


def test_second_guard_errors_on_pass_for_invalid_model_one_timezone() -> None:
    invalid_timezone = "Invalid/Reminder_Zone"
    query = f"Add Payroll at {BASE_TIME} in {invalid_timezone}."
    extraction = _extraction(
        action="add",
        retrieval_text="Payroll",
        subject="Payroll",
        raw_reminder="Payroll",
        notification_time=BASE_TIME,
        user_timezone=invalid_timezone,
        original_time_text=BASE_TIME,
    )
    llm = ScriptedLLM(
        [
            extraction,
            _validation(
                operation="add",
                result="PASS",
                selected=[],
                assessments=[],
                should_execute=True,
            ),
        ]
    )
    repository = _repository()

    with canonical_chat_history_scope([]):
        result = _branch(llm).execute(_context(query, []), repository)

    assert result.response_type is ResponseType.ERROR
    assert len(llm.calls) == 2
    assert repository.table_count("reminders") == 0


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
                result="PASS",
                selected=[],
                assessments=[],
                should_execute=True,
            ),
        ]
    )
    repository = _repository()

    with canonical_chat_history_scope([]):
        result = _branch(llm).execute(_context(query, []), repository)

    assert result.response_type is ResponseType.REMINDER_ACTION
    assert len(llm.calls) == 2
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
                result="PASS",
                selected=[],
                assessments=[],
                should_execute=True,
            ),
        ]
    )
    repository = _repository()

    with canonical_chat_history_scope([]):
        result = _branch(llm).execute(_context(query, []), repository)

    assert result.response_type is ResponseType.REMINDER_ACTION
    assert len(llm.calls) == 2
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
                result="PASS",
                selected=[],
                assessments=[],
                should_execute=True,
            ),
        ]
    )
    repository = _repository()

    with canonical_chat_history_scope([]):
        result = _branch(llm).execute(_context(query, []), repository)

    assert result.response_type is ResponseType.REMINDER_ACTION
    assert len(llm.calls) == 2
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
        changed_fields=[
            changed_time_field,
            "user_timezone",
            "original_time_text",
        ],
        time_semantics=time_semantics,
        user_timezone="UTC",
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
                result="PASS",
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
                changed_fields=[
                    changed_time_field,
                    "user_timezone",
                    "original_time_text",
                ],
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
                result="PASS",
                selected=[target_id],
                assessments=assessments,
                should_execute=True,
            ),
        ]
    )

    with canonical_chat_history_scope([]):
        result = _branch(llm).execute(_context(query, []), repository)

    assert result.response_type is ResponseType.REMINDER_ACTION
    assert result.actions_pending_confirmation == []
    assert _reminder_row(repository, target_id)["status"] == "dismissed"
    assert len(llm.calls) == 2
    validation_candidates = _prompt_payload(llm.calls[1])[
        "reminder_retrieval"
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
def test_toggle_already_in_requested_state_is_stage_two_fail_with_question(
    action: str,
    query: str,
    already_status: str,
) -> None:
    repository = _repository()
    reminder_id = _seed_reminder(repository, status=already_status)
    original = _reminder_row(repository, reminder_id)
    question = "That reminder is already in this state. What would you like to change?"
    llm = ScriptedLLM(
        [
            _toggle_extraction(direction=action, retrieval_text="Payroll"),
            _validation(
                operation=action,
                result="FAIL",
                selected=[],
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
                clarification_question=question,
            ),
        ]
    )

    with canonical_chat_history_scope([]):
        result = _branch(llm).execute(_context(query, []), repository)

    assert result.response_type is ResponseType.CLARIFICATION
    assert result.clarification_question is not None
    assert result.clarification_question.text == question
    assert result.actions_pending_confirmation == []
    assert len(llm.calls) == 2
    updated = _reminder_row(repository, reminder_id)
    assert updated["status"] == original["status"]
    assert updated["version"] == original["version"]


def test_modify_setting_existing_value_is_stage_two_fail_with_question() -> None:
    repository = _repository()
    reminder_id = _seed_reminder(repository, subject="Payroll")
    original = _reminder_row(repository, reminder_id)
    query = "Modify the Payroll reminder subject to Payroll."
    question = "The subject is already Payroll. What different subject do you want?"
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
                result="FAIL",
                selected=[],
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
                clarification_question=question,
            ),
        ]
    )

    with canonical_chat_history_scope([]):
        result = _branch(llm).execute(_context(query, []), repository)

    assert result.response_type is ResponseType.CLARIFICATION
    assert result.clarification_question is not None
    assert result.clarification_question.text == question
    assert result.actions_pending_confirmation == []
    assert len(llm.calls) == 2
    updated = _reminder_row(repository, reminder_id)
    assert updated["status"] == original["status"]
    assert updated["version"] == original["version"]
    assert repository.table_count("reminders") == 1


def test_compatible_and_incompatible_same_target_fail_reasks() -> None:
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
    question = "Which Payroll reminder do you want to turn on?"
    llm = ScriptedLLM(
        [
            _toggle_extraction(direction="turn_on", retrieval_text="Payroll"),
            _validation(
                operation="turn_on",
                result="FAIL",
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
                reason="Two reminder rows share the requested target.",
                clarification_question=question,
            ),
        ]
    )

    with canonical_chat_history_scope([]):
        result = _branch(llm).execute(_context(query, []), repository)

    assert result.response_type is ResponseType.CLARIFICATION
    assert result.clarification_question is not None
    assert result.clarification_question.text == question
    assert len(llm.calls) == 2
    compatible_after = _reminder_row(repository, compatible_id)
    incompatible_after = _reminder_row(repository, incompatible_id)
    assert compatible_after["status"] == compatible_before["status"]
    assert compatible_after["version"] == compatible_before["version"]
    assert incompatible_after["status"] == incompatible_before["status"]
    assert incompatible_after["version"] == incompatible_before["version"]


def test_validation_candidates_include_recurrence_linkage_without_model_three() -> None:
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
                result="PASS",
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
        ]
    )

    with canonical_chat_history_scope([]):
        result = _branch(llm).execute(_context(query, []), repository)

    assert result.response_type is ResponseType.REMINDER_ACTION
    assert len(llm.calls) == 2
    validation_candidates = _prompt_payload(llm.calls[1])[
        "reminder_retrieval"
    ]
    child = next(
        candidate
        for candidate in validation_candidates
        if candidate["candidate_key"] == child_id
    )
    assert child["next_fire_time"] == next_fire_time
    assert child["parent_recurring_reminder_id"] == parent_id

@pytest.mark.parametrize(
    ("action", "query", "initial_status", "expected_status"),
    (
        (
            "delete",
            "Delete the Legacy timezone reminder.",
            "scheduled",
            "dismissed",
        ),
        (
            "turn_off",
            "Turn off the Legacy timezone reminder.",
            "scheduled",
            "cancelled",
        ),
        (
            "turn_on",
            "Turn on the Legacy timezone reminder.",
            "cancelled",
            "scheduled",
        ),
    ),
)
def test_target_only_actions_do_not_normalize_legacy_invalid_timezone(
    action: str,
    query: str,
    initial_status: str,
    expected_status: str,
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
                result="PASS",
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
        ]
    )

    with canonical_chat_history_scope([]):
        result = _branch(llm).execute(_context(query, []), repository)

    assert result.response_type is ResponseType.REMINDER_ACTION
    assert result.clarification_question is None
    assert result.actions_pending_confirmation == []
    assert len(llm.calls) == 2
    assert _reminder_row(repository, reminder_id)["status"] == expected_status


def test_reminder_branch_does_not_create_supporting_fields_before_autoscan() -> None:
    finalized_question = "Which checklist should I use?"
    finalized_response = "Use checklist A"
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
        time_semantics="both",
    )
    llm = ScriptedLLM(
        [
            extraction,
            _validation(
                operation="add",
                result="PASS",
                selected=[],
                assessments=[],
                should_execute=True,
            ),
        ]
    )

    repository = _repository()
    branch = _branch(llm)

    with canonical_chat_history_scope([]):
        result = branch.execute(_context(query, []), repository)

    assert result.response_type is ResponseType.REMINDER_ACTION
    assert result.reminder_supporting_question is None
    assert [call["task"] for call in llm.calls] == [
        LLMTask.REMINDER_ACTION_EXTRACTION,
        LLMTask.REMINDER_ACTION_VALIDATION,
    ]
    row = repository.list_reminders(user_id=USER_ID)[0]
    assert not row["supporting_question"]
    assert not row["supporting_response"]


def test_first_model_guard_rejects_removed_supporting_question_fields() -> None:
    query = "Add a Payroll reminder. Supporting question: Should this repeat?"
    payload = _extraction(
        action="add",
        retrieval_text="Payroll",
        subject="Payroll",
        raw_reminder="Payroll",
        notification_time=BASE_TIME,
        user_timezone="UTC",
        original_time_text=BASE_TIME,
        supporting_question="Should this repeat?",
    )
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
                raw_query=query,
                platform_context={"timezone": "UTC"},
            ),
            query,
            Intent.REMINDER,
        )

    assert result.metadata == {}
    assert result.requires_clarification
    assert "invalid_reminder_action_extraction" in result.risk_flags
    assert len(llm.calls) == 1


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
                result="FAIL",
                selected=[],
                assessments=[],
                should_execute=False,
            ),
        ]
    )

    with canonical_chat_history_scope([]):
        result = _branch(llm).execute(_context(query, []), repository)

    assert result.response_type is ResponseType.CLARIFICATION
    assert result.clarification_question is not None
    validation_payload = _prompt_payload(llm.calls[1])
    assert validation_payload["reminder_retrieval"] == []
    assert repository.table_count("reminders") == 1
