from __future__ import annotations

from dataclasses import fields, replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from assistant_rag.branch_orchestration import (
    ReminderTargetResolver,
    ValidatedActionBuilder,
)
from assistant_rag.branches import (
    KnowledgeFactsBranch,
    ReminderBranch,
)
from assistant_rag.config import ReminderTargetResolverConfig
from assistant_rag.contracts import (
    ActionValidationResult,
    ChatRequest,
    Intent,
    KnowledgeAction,
    ReminderAction,
    ReminderStatus,
    ReminderTargetResolution,
    ResponseType,
    ValidatedKnowledgeAction,
    ValidatedReminderAction,
)
from assistant_rag.database import SQLiteRepository
from assistant_rag.production_factory import build_assistant_config
from assistant_rag.settings import ProductionSettings, TargetNotFoundPolicy


USER_ID = "mutation-matrix-user"
BASE_TIME = datetime(2035, 4, 5, 9, 30, tzinfo=timezone.utc)


@pytest.fixture
def repository() -> SQLiteRepository:
    value = SQLiteRepository.in_memory()
    value.initialize_schema()
    return value


@pytest.fixture(scope="module")
def config():
    return build_assistant_config(ProductionSettings())


def _seed_source(repository: SQLiteRepository) -> tuple[str, str]:
    with repository.transaction() as cursor:
        topic_id = repository.ensure_topic(
            cursor,
            user_id=USER_ID,
            title="Mutation matrix source",
        )
        hop = repository.append_conversation_hop(
            cursor,
            topic_id=topic_id,
            user_id=USER_ID,
            intent=Intent.REMINDER.value,
            raw_user_query="Create the source reminder.",
            rewritten_user_query="Create the source reminder.",
            raw_response="Source reminder created.",
            response_type=ResponseType.REMINDER_ACTION.value,
        )
    return topic_id, hop.hop_id


def _seed_reminder(
    repository: SQLiteRepository,
    *,
    status: ReminderStatus = ReminderStatus.SCHEDULED,
    subject: str = "Payroll",
    reminder_time: datetime = BASE_TIME,
) -> dict:
    topic_id, hop_id = _seed_source(repository)
    with repository.transaction() as cursor:
        reminder_id = repository.add_reminder(
            cursor,
            user_id=USER_ID,
            source_topic_id=topic_id,
            source_hop_id=hop_id,
            reminder_time=reminder_time.isoformat(),
            event_time=reminder_time.isoformat(),
            raw_reminder=f"Remind me about {subject}.",
            reminder_summary=subject,
            subject=subject,
            user_timezone="UTC",
        )
        cursor.execute(
            """
            UPDATE reminders
            SET timing_plan_status = 'planned', timing_planned_at = ?,
                timing_plan_reason = 'test fixture'
            WHERE reminder_id = ?
            """,
            (BASE_TIME.isoformat(), reminder_id),
        )
        if status is not ReminderStatus.SCHEDULED:
            repository.update_reminder_status(
                cursor,
                user_id=USER_ID,
                reminder_id=reminder_id,
                status=status.value,
            )
    return _reminder_row(repository, reminder_id)


def _seed_knowledge(
    repository: SQLiteRepository,
    *,
    text: str = "Atlas retention is 30 days.",
) -> dict:
    with repository.transaction() as cursor:
        _, chunk_id, _ = repository.add_knowledge_chunk(
            cursor,
            user_id=USER_ID,
            title="Knowledge",
            text=text,
        )
    return _knowledge_row(repository, chunk_id)


def _reminder_row(repository: SQLiteRepository, reminder_id: str) -> dict:
    row = repository.connection.execute(
        "SELECT * FROM reminders WHERE reminder_id = ?",
        (reminder_id,),
    ).fetchone()
    assert row is not None
    return dict(row)


def _knowledge_row(repository: SQLiteRepository, chunk_id: str) -> dict:
    row = repository.connection.execute(
        "SELECT * FROM knowledge_chunks WHERE chunk_id = ?",
        (chunk_id,),
    ).fetchone()
    assert row is not None
    return dict(row)


def _knowledge_transaction(
    repository: SQLiteRepository,
    actions: list[ValidatedKnowledgeAction],
):
    return repository.transactional_knowledge_actions(
        user_id=USER_ID,
        topic_title="Knowledge",
        raw_user_query="Knowledge mutation.",
        rewritten_user_query="Knowledge mutation.",
        response_text="Knowledge mutation complete.",
        actions=actions,
    )


def _reminder_transaction(
    repository: SQLiteRepository,
    actions: list[ValidatedReminderAction],
):
    return repository.transactional_reminder_actions(
        user_id=USER_ID,
        topic_title="Reminders",
        raw_user_query="Reminder mutation.",
        rewritten_user_query="Reminder mutation.",
        response_text="Reminder mutation complete.",
        actions=actions,
    )


def _targeted_reminder_action(
    action: ReminderAction,
    row: dict,
    **changes,
) -> ValidatedReminderAction:
    values = {
        "action": action,
        "validation_result": ActionValidationResult.EXECUTE,
        "target_reminder_ids": (str(row["reminder_id"]),),
        "observed_status": str(row["status"]),
        "observed_version": int(row["version"]),
        "observed_reminder_time": datetime.fromisoformat(str(row["reminder_time"])),
        "subject": str(row["subject"]),
        "confidence": 1.0,
    }
    values.update(changes)
    return ValidatedReminderAction(**values)


REMINDER_ELIGIBLE_STATUSES = {
    ReminderAction.MODIFY: {
        ReminderStatus.SCHEDULED,
        ReminderStatus.NOTIFIED,
    },
    ReminderAction.DELETE: set(ReminderStatus),
    ReminderAction.TURN_ON: {
        ReminderStatus.CANCELLED,
        ReminderStatus.DISMISSED,
        ReminderStatus.COMPLETED,
    },
    ReminderAction.TURN_OFF: {
        ReminderStatus.SCHEDULED,
        ReminderStatus.NOTIFIED,
    },
}


@pytest.mark.parametrize("action", tuple(REMINDER_ELIGIBLE_STATUSES))
@pytest.mark.parametrize("status", tuple(ReminderStatus))
def test_reminder_target_resolution_status_eligibility_matrix(
    repository: SQLiteRepository,
    config,
    action: ReminderAction,
    status: ReminderStatus,
) -> None:
    seeded = _seed_reminder(repository, status=status)
    resolver = ReminderTargetResolver(config)

    result = resolver.resolve(
        user_id=USER_ID,
        action=action,
        target_description="Payroll",
        user_query=f"{action.value} Payroll",
        rewritten_query=f"{action.value} Payroll",
        repository=repository,
    )

    if status in REMINDER_ELIGIBLE_STATUSES[action]:
        assert result.validation_result is ActionValidationResult.EXECUTE
        assert result.target_reminder_ids == (seeded["reminder_id"],)
        assert result.chosen_candidate is not None
        assert result.chosen_candidate.status == status.value
        assert result.chosen_candidate.version == seeded["version"]
    else:
        assert result.validation_result is ActionValidationResult.SKIP_NOT_FOUND
        assert result.target_reminder_ids == ()
        assert result.chosen_candidate is None


def test_add_has_no_existing_reminder_target_statuses(config) -> None:
    resolver = ReminderTargetResolver(config)

    assert resolver.candidate_statuses_for_action(ReminderAction.ADD) == ()


def test_reminder_resolver_config_accepts_every_canonical_status() -> None:
    settings = ProductionSettings().reminder_resolver
    values = {
        field.name: getattr(settings, field.name)
        for field in fields(ReminderTargetResolverConfig)
    }

    value = ReminderTargetResolverConfig(**values)

    assert ReminderStatus.COMPLETED.value in value.allowed_reminder_turn_on_statuses
    assert ReminderStatus.COMPLETED.value in value.allowed_reminder_delete_statuses


def test_reminder_target_missing_description_requires_clarification(
    repository: SQLiteRepository,
    config,
) -> None:
    result = ReminderTargetResolver(config).resolve(
        user_id=USER_ID,
        action=ReminderAction.DELETE,
        target_description="",
        user_query="Delete a reminder.",
        rewritten_query="Delete a reminder.",
        repository=repository,
    )

    assert result.validation_result is ActionValidationResult.CLARIFY_MISSING_FIELDS


def test_reminder_target_not_found_can_be_configured_to_clarify(
    repository: SQLiteRepository,
) -> None:
    settings = ProductionSettings()
    settings = replace(
        settings,
        reminder_resolver=replace(
            settings.reminder_resolver,
            reminder_target_not_found_policy=TargetNotFoundPolicy.CLARIFY_ON_NOT_FOUND,
        ),
    )
    config = build_assistant_config(settings)

    result = ReminderTargetResolver(config).resolve(
        user_id=USER_ID,
        action=ReminderAction.TURN_OFF,
        target_description="Missing",
        user_query="Turn off Missing.",
        rewritten_query="Turn off Missing.",
        repository=repository,
    )

    assert result.validation_result is ActionValidationResult.CLARIFY_AMBIGUOUS_TARGET


def test_equally_matching_reminder_targets_are_ambiguous(
    repository: SQLiteRepository,
    config,
) -> None:
    _seed_reminder(repository, subject="Payroll", reminder_time=BASE_TIME)
    _seed_reminder(
        repository,
        subject="Payroll",
        reminder_time=BASE_TIME + timedelta(hours=1),
    )

    result = ReminderTargetResolver(config).resolve(
        user_id=USER_ID,
        action=ReminderAction.DELETE,
        target_description="Payroll",
        user_query="Delete Payroll.",
        rewritten_query="Delete Payroll.",
        repository=repository,
    )

    assert result.validation_result is ActionValidationResult.CLARIFY_AMBIGUOUS_TARGET
    assert len(result.clarification_candidates) == 2
    assert result.target_reminder_ids == ()


@pytest.mark.parametrize(
    ("payload", "expected"),
    (
        (
            {
                "action": "add",
                "subject": "Payroll",
                "reminder_time": BASE_TIME.isoformat(),
            },
            ActionValidationResult.EXECUTE,
        ),
        (
            {"action": "add", "subject": "Payroll"},
            ActionValidationResult.CLARIFY_MISSING_FIELDS,
        ),
        (
            {"action": "add", "reminder_time": BASE_TIME.isoformat()},
            ActionValidationResult.CLARIFY_MISSING_FIELDS,
        ),
        (
            {
                "action": "add",
                "subject": "Payroll",
                "reminder_time": "not-a-timestamp",
            },
            ActionValidationResult.CLARIFY_MISSING_FIELDS,
        ),
    ),
)
def test_reminder_add_field_validation_matrix(
    config,
    payload: dict,
    expected: ActionValidationResult,
) -> None:
    builder = ValidatedActionBuilder(
        config=config,
        knowledge_resolver=SimpleNamespace(),
        reminder_resolver=SimpleNamespace(),
    )

    actions = builder.build_reminder_actions(
        USER_ID,
        [payload],
        "Create a reminder.",
        "Create a reminder.",
        SimpleNamespace(),
    )

    assert len(actions) == 1
    assert actions[0].action is ReminderAction.ADD
    assert actions[0].validation_result is expected


@pytest.mark.parametrize("invalid_time", ("not-a-timestamp", "2035-99-99T25:61:00"))
def test_reminder_modify_invalid_replacement_time_clarifies_without_crashing(
    config,
    invalid_time: str,
) -> None:
    resolution = ReminderTargetResolution(
        validation_result=ActionValidationResult.EXECUTE,
        target_reminder_ids=("reminder-1",),
        chosen_candidate=SimpleNamespace(
            status=ReminderStatus.SCHEDULED.value,
            version=3,
            reminder_time=BASE_TIME,
        ),
    )
    builder = ValidatedActionBuilder(
        config=config,
        knowledge_resolver=SimpleNamespace(),
        reminder_resolver=SimpleNamespace(resolve=lambda **_kwargs: resolution),
    )

    actions = builder.build_reminder_actions(
        USER_ID,
        [
            {
                "action": "modify",
                "target_description": "Payroll",
                "new_reminder_time": invalid_time,
            }
        ],
        "Move Payroll.",
        "Move Payroll.",
        SimpleNamespace(),
    )

    assert len(actions) == 1
    assert actions[0].validation_result is ActionValidationResult.CLARIFY_MISSING_FIELDS
    assert actions[0].replacement_time is None
    assert actions[0].observed_status is None
    assert actions[0].observed_version is None


@pytest.mark.parametrize(
    ("action", "resolver_result", "replacement_text", "expected"),
    (
        (
            "delete",
            ActionValidationResult.EXECUTE,
            None,
            ActionValidationResult.EXECUTE,
        ),
        (
            "delete",
            ActionValidationResult.SKIP_NOT_FOUND,
            None,
            ActionValidationResult.SKIP_NOT_FOUND,
        ),
        (
            "delete",
            ActionValidationResult.CLARIFY_AMBIGUOUS_TARGET,
            None,
            ActionValidationResult.CLARIFY_AMBIGUOUS_TARGET,
        ),
        (
            "modify",
            ActionValidationResult.EXECUTE,
            "Atlas retention is 45 days.",
            ActionValidationResult.EXECUTE,
        ),
        (
            "modify",
            ActionValidationResult.EXECUTE,
            None,
            ActionValidationResult.CLARIFY_MISSING_FIELDS,
        ),
    ),
)
def test_knowledge_target_validation_matrix(
    config,
    action: str,
    resolver_result: ActionValidationResult,
    replacement_text: str | None,
    expected: ActionValidationResult,
) -> None:
    resolver = SimpleNamespace(
        resolve=lambda **_kwargs: (
            ("chunk-1",) if resolver_result is ActionValidationResult.EXECUTE else (),
            resolver_result,
        )
    )
    builder = ValidatedActionBuilder(
        config=config,
        knowledge_resolver=resolver,
        reminder_resolver=SimpleNamespace(),
    )

    actions = builder.build_knowledge_actions(
        USER_ID,
        [
            {
                "action": action,
                "target_description": "Atlas retention",
                "replacement_text": replacement_text,
            }
        ],
        f"{action} Atlas retention.",
        f"{action} Atlas retention.",
        SimpleNamespace(),
    )

    assert len(actions) == 1
    assert actions[0].validation_result is expected


@pytest.mark.parametrize(
    ("payload", "expected"),
    (
        (
            {"action": "add", "text": "Atlas retention is 30 days."},
            ActionValidationResult.EXECUTE,
        ),
        ({"action": "add", "text": ""}, ActionValidationResult.CLARIFY_MISSING_FIELDS),
        ({"action": "archive"}, ActionValidationResult.REJECT_UNSUPPORTED_OPERATION),
        (
            {"action": "delete", "risk_approved": False},
            ActionValidationResult.REJECT_UNSUPPORTED_OPERATION,
        ),
    ),
)
def test_knowledge_add_and_unsupported_validation_matrix(
    config,
    payload: dict,
    expected: ActionValidationResult,
) -> None:
    builder = ValidatedActionBuilder(
        config=config,
        knowledge_resolver=SimpleNamespace(),
        reminder_resolver=SimpleNamespace(),
    )

    actions = builder.build_knowledge_actions(
        USER_ID,
        [payload],
        "Knowledge action.",
        "Knowledge action.",
        SimpleNamespace(),
    )

    assert len(actions) == 1
    assert actions[0].validation_result is expected


def test_knowledge_add_and_exact_duplicate_are_idempotent(
    repository: SQLiteRepository,
) -> None:
    action = ValidatedKnowledgeAction(
        action=KnowledgeAction.ADD,
        validation_result=ActionValidationResult.EXECUTE,
        knowledge_text="Atlas retention is 30 days.",
        topic_title="Knowledge",
    )

    first = _knowledge_transaction(repository, [action])
    failed_job_id = first.results[0].indexing_outbox_ids[0]
    repository.mark_outbox_job_failed(
        job_id=failed_job_id,
        error_message="derived index unavailable",
    )
    second = _knowledge_transaction(repository, [action])

    assert first.committed and second.committed
    assert first.results[0].status == "committed"
    assert second.results[0].status == "committed"
    assert second.results[0].user_safe_summary == "Knowledge was already known."
    assert first.results[0].domain_entity_id == second.results[0].domain_entity_id
    assert second.results[0].indexing_outbox_ids
    assert second.results[0].indexing_outbox_ids[0] != failed_job_id
    repair_status = repository.connection.execute(
        "SELECT status FROM indexing_outbox WHERE job_id = ?",
        (second.results[0].indexing_outbox_ids[0],),
    ).fetchone()
    assert repair_status is not None
    assert repair_status["status"] == "pending"
    assert repository.table_count("knowledge_chunks") == 1


def test_knowledge_modify_replaces_old_chunk_atomically(
    repository: SQLiteRepository,
) -> None:
    original = _seed_knowledge(repository)
    action = ValidatedKnowledgeAction(
        action=KnowledgeAction.MODIFY,
        validation_result=ActionValidationResult.EXECUTE,
        target_chunk_ids=(original["chunk_id"],),
        observed_versions={original["chunk_id"]: original["version"]},
        replacement_text="Atlas retention is 45 days.",
        topic_title="Knowledge",
        target_description="Atlas retention",
    )

    result = _knowledge_transaction(repository, [action])

    assert result.committed
    assert result.results[0].status == "committed"
    replacement_id = result.results[0].domain_entity_id
    assert replacement_id and replacement_id != original["chunk_id"]
    old_row = _knowledge_row(repository, original["chunk_id"])
    new_row = _knowledge_row(repository, replacement_id)
    assert old_row["is_deleted"] == 1
    assert old_row["replaced_by_chunk_id"] == replacement_id
    assert new_row["is_deleted"] == 0
    assert new_row["replaces_chunk_id"] == original["chunk_id"]
    assert new_row["raw_text"] == "Atlas retention is 45 days."


def test_knowledge_delete_soft_deletes_exact_version(
    repository: SQLiteRepository,
) -> None:
    original = _seed_knowledge(repository)
    action = ValidatedKnowledgeAction(
        action=KnowledgeAction.DELETE,
        validation_result=ActionValidationResult.EXECUTE,
        target_chunk_ids=(original["chunk_id"],),
        observed_versions={original["chunk_id"]: original["version"]},
        target_description="Atlas retention",
    )

    result = _knowledge_transaction(repository, [action])

    assert result.committed
    assert result.results[0].domain_entity_id == original["chunk_id"]
    deleted = _knowledge_row(repository, original["chunk_id"])
    assert deleted["is_deleted"] == 1
    assert deleted["version"] == original["version"] + 1


@pytest.mark.parametrize("failure", ("missing", "stale_version"))
def test_knowledge_conflict_rolls_back_audit_and_mutation(
    repository: SQLiteRepository,
    failure: str,
) -> None:
    original = _seed_knowledge(repository)
    target_id = "missing-chunk" if failure == "missing" else original["chunk_id"]
    observed_version = original["version"] + (1 if failure == "stale_version" else 0)
    hops_before = repository.table_count("conversation_hops")
    action = ValidatedKnowledgeAction(
        action=KnowledgeAction.DELETE,
        validation_result=ActionValidationResult.EXECUTE,
        target_chunk_ids=(target_id,),
        observed_versions={target_id: observed_version},
        target_description="Atlas retention",
    )

    result = _knowledge_transaction(repository, [action])

    assert not result.committed
    assert result.error_type == "KnowledgeConflictError"
    assert repository.table_count("conversation_hops") == hops_before
    unchanged = _knowledge_row(repository, original["chunk_id"])
    assert unchanged["is_deleted"] == 0
    assert unchanged["version"] == original["version"]


def test_knowledge_multi_step_failure_rolls_back_prior_write(
    repository: SQLiteRepository,
) -> None:
    original = _seed_knowledge(repository)
    chunks_before = repository.table_count("knowledge_chunks")
    actions = [
        ValidatedKnowledgeAction(
            action=KnowledgeAction.ADD,
            validation_result=ActionValidationResult.EXECUTE,
            knowledge_text="This write must roll back.",
            topic_title="Knowledge",
        ),
        ValidatedKnowledgeAction(
            action=KnowledgeAction.DELETE,
            validation_result=ActionValidationResult.EXECUTE,
            target_chunk_ids=(original["chunk_id"],),
            observed_versions={original["chunk_id"]: original["version"] + 1},
        ),
    ]

    result = _knowledge_transaction(repository, actions)

    assert not result.committed
    assert result.error_type == "KnowledgeConflictError"
    assert repository.table_count("knowledge_chunks") == chunks_before
    assert _knowledge_row(repository, original["chunk_id"])["is_deleted"] == 0


REMINDER_TRANSITIONS = (
    *((ReminderAction.TURN_ON, status, ReminderStatus.SCHEDULED) for status in REMINDER_ELIGIBLE_STATUSES[ReminderAction.TURN_ON]),
    *((ReminderAction.TURN_OFF, status, ReminderStatus.CANCELLED) for status in REMINDER_ELIGIBLE_STATUSES[ReminderAction.TURN_OFF]),
    *((ReminderAction.DELETE, status, ReminderStatus.DISMISSED) for status in REMINDER_ELIGIBLE_STATUSES[ReminderAction.DELETE]),
)


@pytest.mark.parametrize(("action", "initial_status", "expected_status"), REMINDER_TRANSITIONS)
def test_reminder_repository_status_transition_matrix(
    repository: SQLiteRepository,
    action: ReminderAction,
    initial_status: ReminderStatus,
    expected_status: ReminderStatus,
) -> None:
    original = _seed_reminder(repository, status=initial_status)

    result = _reminder_transaction(
        repository,
        [_targeted_reminder_action(action, original)],
    )

    assert result.committed
    assert result.results[0].status == "committed"
    updated = _reminder_row(repository, original["reminder_id"])
    assert updated["status"] == expected_status.value
    assert updated["version"] == original["version"] + 1
    if action is ReminderAction.TURN_ON:
        assert updated["timing_plan_status"] == "pending"
        assert updated["timing_planned_at"] is None
        assert updated["timing_plan_reason"] is None


@pytest.mark.parametrize(
    "initial_status",
    (ReminderStatus.SCHEDULED, ReminderStatus.NOTIFIED),
)
def test_reminder_modify_creates_successor_and_preserves_source_identity(
    repository: SQLiteRepository,
    initial_status: ReminderStatus,
) -> None:
    original = _seed_reminder(repository, status=initial_status)
    replacement_time = BASE_TIME + timedelta(days=2)
    action = _targeted_reminder_action(
        ReminderAction.MODIFY,
        original,
        replacement_subject="Quarterly payroll",
        replacement_time=replacement_time,
        replacement_summary="Run quarterly payroll",
    )

    result = _reminder_transaction(repository, [action])

    assert result.committed
    replacement_id = result.results[0].domain_entity_id
    assert replacement_id and replacement_id != original["reminder_id"]
    old_row = _reminder_row(repository, original["reminder_id"])
    new_row = _reminder_row(repository, replacement_id)
    assert old_row["status"] == ReminderStatus.CANCELLED.value
    assert new_row["status"] == ReminderStatus.SCHEDULED.value
    assert new_row["subject"] == "Quarterly payroll"
    assert new_row["reminder_time"] == replacement_time.isoformat()
    assert new_row["parent_recurring_reminder_id"] == original["reminder_id"]
    assert new_row["source_topic_id"] == original["source_topic_id"]
    assert new_row["source_hop_id"] == original["source_hop_id"]
    assert new_row["timing_plan_status"] == "pending"


def test_reminder_add_binds_new_reminder_to_audit_hop(
    repository: SQLiteRepository,
) -> None:
    action = ValidatedReminderAction(
        action=ReminderAction.ADD,
        validation_result=ActionValidationResult.EXECUTE,
        subject="Payroll",
        reminder_summary="Run payroll",
        raw_reminder="Remind me to run payroll.",
        event_time=BASE_TIME,
        reminder_time=BASE_TIME,
        user_timezone="UTC",
    )

    result = _reminder_transaction(repository, [action])

    assert result.committed
    assert result.results[0].status == "committed"
    reminder_id = result.results[0].domain_entity_id
    assert reminder_id is not None
    row = _reminder_row(repository, reminder_id)
    assert row["source_hop_id"] == result.audit_hop_id
    assert row["status"] == ReminderStatus.SCHEDULED.value
    assert row["timing_plan_status"] == "pending"


def test_reminder_add_persists_supporting_question_and_response(
    repository: SQLiteRepository,
) -> None:
    action = ValidatedReminderAction(
        action=ReminderAction.ADD,
        validation_result=ActionValidationResult.EXECUTE,
        subject="Payroll",
        reminder_summary="Run payroll",
        raw_reminder="Remind me to run payroll.",
        supporting_question="Should this include contractors?",
        supporting_response="Yes, include contractors.",
        event_time=BASE_TIME,
        reminder_time=BASE_TIME,
        user_timezone="UTC",
    )

    result = _reminder_transaction(repository, [action])

    assert result.committed
    reminder_id = result.results[0].domain_entity_id
    assert reminder_id is not None
    row = _reminder_row(repository, reminder_id)
    assert row["supporting_question"] == "Should this include contractors?"
    assert row["supporting_response"] == "Yes, include contractors."


@pytest.mark.parametrize(
    (
        "replacement_summary",
        "replacement_recurrence_rule",
        "replacement_recurrence_timezone",
        "expected_summary",
        "expected_recurrence_rule",
        "expected_recurrence_timezone",
    ),
    (
        (None, None, None, "Run payroll", "weekly", "UTC"),
        ("", "", "", "", "", ""),
    ),
)
def test_reminder_modify_persists_full_optional_fields_with_none_aware_replacements(
    repository: SQLiteRepository,
    replacement_summary: str | None,
    replacement_recurrence_rule: str | None,
    replacement_recurrence_timezone: str | None,
    expected_summary: str,
    expected_recurrence_rule: str,
    expected_recurrence_timezone: str,
) -> None:
    original = _seed_reminder(repository)
    replacement_time = BASE_TIME + timedelta(days=1)
    action = _targeted_reminder_action(
        ReminderAction.MODIFY,
        original,
        replacement_time=replacement_time,
        reminder_summary="Run payroll",
        raw_reminder="Remind me to run payroll.",
        supporting_question="Should this include contractors?",
        supporting_response="Yes, include contractors.",
        user_timezone="UTC",
        original_time_text="tomorrow at 9:30",
        recurrence_rule="weekly",
        recurrence_timezone="UTC",
        replacement_summary=replacement_summary,
        replacement_recurrence_rule=replacement_recurrence_rule,
        replacement_recurrence_timezone=replacement_recurrence_timezone,
    )

    result = _reminder_transaction(repository, [action])

    assert result.committed
    replacement_id = result.results[0].domain_entity_id
    assert replacement_id is not None
    row = _reminder_row(repository, replacement_id)
    assert row["reminder_summary"] == expected_summary
    assert row["recurrence_rule"] == expected_recurrence_rule
    assert row["recurrence_timezone"] == expected_recurrence_timezone
    assert row["supporting_question"] == "Should this include contractors?"
    assert row["supporting_response"] == "Yes, include contractors."
    assert row["source_topic_id"] == original["source_topic_id"]
    assert row["source_hop_id"] == original["source_hop_id"]


def test_reminder_repository_does_not_override_llm2_add_authorization_with_duplicate_detection(
    repository: SQLiteRepository,
) -> None:
    existing = _seed_reminder(repository)
    reminders_before = repository.table_count("reminders")
    action = ValidatedReminderAction(
        action=ReminderAction.ADD,
        validation_result=ActionValidationResult.EXECUTE,
        subject=existing["subject"],
        reminder_summary=existing["reminder_summary"],
        reminder_time=datetime.fromisoformat(existing["reminder_time"]),
        event_time=datetime.fromisoformat(existing["reminder_time"]),
    )

    result = _reminder_transaction(repository, [action])

    assert result.committed
    assert result.results[0].status == "committed"
    assert result.results[0].domain_entity_id is not None
    assert repository.table_count("reminders") == reminders_before + 1


@pytest.mark.parametrize("failure", ("missing", "stale_version", "wrong_status"))
def test_reminder_conflict_rolls_back_audit_and_state(
    repository: SQLiteRepository,
    failure: str,
) -> None:
    original = _seed_reminder(repository)
    action = _targeted_reminder_action(ReminderAction.TURN_OFF, original)
    if failure == "missing":
        action = replace(action, target_reminder_ids=("missing-reminder",))
    elif failure == "stale_version":
        action = replace(action, observed_version=original["version"] + 1)
    else:
        action = replace(action, observed_status=ReminderStatus.NOTIFIED.value)
    hops_before = repository.table_count("conversation_hops")

    result = _reminder_transaction(repository, [action])

    assert not result.committed
    assert result.error_type == "ReminderConflictError"
    assert repository.table_count("conversation_hops") == hops_before
    unchanged = _reminder_row(repository, original["reminder_id"])
    assert unchanged["status"] == original["status"]
    assert unchanged["version"] == original["version"]


def test_reminder_modify_missing_replacement_time_rolls_back_as_error(
    repository: SQLiteRepository,
) -> None:
    original = _seed_reminder(repository)
    action = _targeted_reminder_action(
        ReminderAction.MODIFY,
        original,
        observed_reminder_time=None,
        reminder_time=None,
        replacement_time=None,
    )
    hops_before = repository.table_count("conversation_hops")

    result = _reminder_transaction(repository, [action])

    assert not result.committed
    assert result.error_type == "RepositoryTransactionError"
    assert repository.table_count("conversation_hops") == hops_before
    unchanged = _reminder_row(repository, original["reminder_id"])
    assert unchanged["status"] == ReminderStatus.SCHEDULED.value


def test_reminder_multi_step_failure_rolls_back_prior_add(
    repository: SQLiteRepository,
) -> None:
    original = _seed_reminder(repository)
    reminders_before = repository.table_count("reminders")
    hops_before = repository.table_count("conversation_hops")
    actions = [
        ValidatedReminderAction(
            action=ReminderAction.ADD,
            validation_result=ActionValidationResult.EXECUTE,
            subject="Must roll back",
            reminder_summary="Must roll back",
            reminder_time=BASE_TIME + timedelta(days=10),
            event_time=BASE_TIME + timedelta(days=10),
        ),
        replace(
            _targeted_reminder_action(ReminderAction.TURN_OFF, original),
            observed_version=original["version"] + 1,
        ),
    ]

    result = _reminder_transaction(repository, actions)

    assert not result.committed
    assert result.error_type == "ReminderConflictError"
    assert repository.table_count("reminders") == reminders_before
    assert repository.table_count("conversation_hops") == hops_before
    assert _reminder_row(repository, original["reminder_id"])["status"] == original["status"]


class _StaticKnowledgeBuilder:
    def __init__(self, action: ValidatedKnowledgeAction) -> None:
        self.action = action

    def build_knowledge_actions(self, *_args, **_kwargs):
        return [self.action]


class _StaticKnowledgePipeline:
    def __init__(self, action: ValidatedKnowledgeAction) -> None:
        self.action = action

    def build_action(self, **_kwargs):
        return self.action


class _StaticReminderPipeline:
    def __init__(self, action: ValidatedReminderAction) -> None:
        self.action = action

    def build_action(self, **_kwargs):
        return self.action


class _StaticExtractionDetector:
    def __init__(self, metadata_key: str, payload: dict) -> None:
        self.metadata_key = metadata_key
        self.payload = dict(payload)
        self.calls = 0

    def detect(self, *_args, **_kwargs):
        self.calls += 1
        return SimpleNamespace(
            requires_clarification=False,
            missing_fields=[],
            metadata={self.metadata_key: [dict(self.payload)]},
        )


def _context(query: str, metadata: dict) -> SimpleNamespace:
    return SimpleNamespace(
        request=ChatRequest(user_id=USER_ID, raw_query=query, metadata=metadata),
        rewritten_query=query,
        approved_conversation_context=None,
    )


@pytest.mark.parametrize("action_type", (KnowledgeAction.MODIFY, KnowledgeAction.DELETE))
def test_knowledge_pass_executes_directly_without_pending_confirmation(
    repository: SQLiteRepository,
    config,
    action_type: KnowledgeAction,
) -> None:
    original = _seed_knowledge(repository)
    action = ValidatedKnowledgeAction(
        action=action_type,
        validation_result=ActionValidationResult.EXECUTE,
        target_chunk_ids=(original["chunk_id"],),
        observed_versions={original["chunk_id"]: original["version"]},
        replacement_text=(
            "Atlas retention is 45 days."
            if action_type is KnowledgeAction.MODIFY
            else None
        ),
        new_text=(
            "Atlas retention is 45 days."
            if action_type is KnowledgeAction.MODIFY
            else None
        ),
        target_description="Atlas retention",
        confidence=1.0,
    )
    verb = "Modify" if action_type is KnowledgeAction.MODIFY else "Delete"
    payload = {
        "action": action_type.value,
        "target_description": "Atlas retention",
    }
    if action_type is KnowledgeAction.MODIFY:
        payload["replacement_text"] = "Atlas retention is 45 days."
    detector = _StaticExtractionDetector("knowledge_actions", payload)
    branch = KnowledgeFactsBranch(
        config=config,
        action_detector=detector,
        knowledge_mutation_pipeline=_StaticKnowledgePipeline(action),
    )

    result = branch.execute(
        _context(f"{verb} the Atlas retention fact.", {"knowledge_actions": [payload]}),
        repository,
    )

    assert result.response_type is ResponseType.KNOWLEDGE_ACTION
    assert result.actions_pending_confirmation == []
    assert _knowledge_row(repository, original["chunk_id"])["is_deleted"] == 1
    pending = repository.connection.execute(
        "SELECT * FROM pending_action_confirmations"
    ).fetchall()
    assert pending == []
    assert result.knowledge_operation_results
    assert all(
        operation.status == "committed"
        for operation in result.knowledge_operation_results
    )
    active_rows = repository.connection.execute(
        "SELECT raw_text FROM knowledge_chunks WHERE user_id = ? AND is_deleted = 0",
        (USER_ID,),
    ).fetchall()
    if action_type is KnowledgeAction.MODIFY:
        assert [row["raw_text"] for row in active_rows] == [
            "Atlas retention is 45 days."
        ]
    else:
        assert active_rows == []
    assert detector.calls == 1


@pytest.mark.parametrize(
    ("action_type", "confidence"),
    (
        (ReminderAction.DELETE, 1.0),
        (ReminderAction.MODIFY, 0.50),
        (ReminderAction.MODIFY, 1.0),
        (ReminderAction.TURN_ON, 1.0),
        (ReminderAction.TURN_OFF, 1.0),
    ),
)
def test_reminder_validated_actions_execute_without_danger_or_pending_confirmation(
    repository: SQLiteRepository,
    config,
    action_type: ReminderAction,
    confidence: float,
) -> None:
    initial_status = (
        ReminderStatus.CANCELLED
        if action_type is ReminderAction.TURN_ON
        else ReminderStatus.SCHEDULED
    )
    original = _seed_reminder(repository, status=initial_status)
    changes = {"confidence": confidence}
    if action_type is ReminderAction.MODIFY:
        changes.update(
            replacement_subject="Moved payroll",
            replacement_time=BASE_TIME + timedelta(days=1),
        )
    action = _targeted_reminder_action(action_type, original, **changes)
    verb = {
        ReminderAction.DELETE: "Delete",
        ReminderAction.MODIFY: "Modify",
        ReminderAction.TURN_ON: "Turn on",
        ReminderAction.TURN_OFF: "Turn off",
    }[action_type]
    payload = {
        "action": action_type.value,
        "target_description": "Payroll",
    }
    if action_type is ReminderAction.MODIFY:
        payload["new_subject"] = "Moved payroll"
    detector = _StaticExtractionDetector("reminder_actions", payload)
    branch = ReminderBranch(
        config=config,
        action_detector=detector,
        reminder_mutation_pipeline=_StaticReminderPipeline(action),
    )

    result = branch.execute(
        _context(f"{verb} the Payroll reminder.", {"reminder_actions": [payload]}),
        repository,
    )

    assert result.response_type is ResponseType.REMINDER_ACTION
    assert result.actions_pending_confirmation == []
    assert len(result.reminder_operation_results) == 1
    assert result.reminder_operation_results[0].status == "committed"
    assert repository.connection.execute(
        "SELECT * FROM pending_action_confirmations"
    ).fetchall() == []
    assert detector.calls == 1


def test_reminder_branch_does_not_recheck_duplicate_after_llm2_execute(
    repository: SQLiteRepository,
    config,
) -> None:
    existing = _seed_reminder(repository)
    action = ValidatedReminderAction(
        action=ReminderAction.ADD,
        validation_result=ActionValidationResult.EXECUTE,
        subject=existing["subject"],
        reminder_summary=existing["reminder_summary"],
        raw_reminder="Create the Payroll reminder.",
        reminder_time=datetime.fromisoformat(existing["reminder_time"]),
        event_time=datetime.fromisoformat(existing["reminder_time"]),
        confidence=1.0,
    )
    payload = {
        "action": "add",
        "subject": "Payroll",
        "reminder_time": existing["reminder_time"],
    }
    detector = _StaticExtractionDetector("reminder_actions", payload)
    branch = ReminderBranch(
        config=config,
        action_detector=detector,
        reminder_mutation_pipeline=_StaticReminderPipeline(action),
    )

    result = branch.execute(
        _context(
            "Create the Payroll reminder.",
            {
                "reminder_actions": [payload]
            },
        ),
        repository,
    )

    assert result.response_type is ResponseType.REMINDER_ACTION
    assert result.actions_pending_confirmation == []
    assert result.reminder_operation_results[0].status == "committed"
    assert repository.table_count("reminders") == 2
    assert detector.calls == 1
