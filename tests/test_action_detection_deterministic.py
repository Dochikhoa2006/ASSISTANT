from __future__ import annotations

from types import SimpleNamespace

import pytest

from assistant_rag.action_keywords import (
    ADD_ACTION_KEYWORDS,
    DELETE_ACTION_KEYWORDS,
    MODIFY_ACTION_KEYWORDS,
    TURN_OFF_ACTION_KEYWORDS,
    TURN_ON_ACTION_KEYWORDS,
)
from assistant_rag.action_detection import (
    DeterministicActionDetector,
    action_payload_is_authorized,
    classify_action_request,
)
from assistant_rag.branch_orchestration import ValidatedActionBuilder
from assistant_rag.branches import KnowledgeFactsBranch, ReminderBranch
from assistant_rag.contracts import ChatRequest, Intent, ResponseType
from assistant_rag.settings import MutationPartialExecutionPolicy


ALL_ACTION_CASES = (
    *((Intent.KNOWLEDGE_FACTS, keyword, "delete") for keyword in DELETE_ACTION_KEYWORDS),
    *((Intent.KNOWLEDGE_FACTS, keyword, "modify") for keyword in MODIFY_ACTION_KEYWORDS),
    *((Intent.KNOWLEDGE_FACTS, keyword, "add") for keyword in ADD_ACTION_KEYWORDS),
    *((Intent.REMINDER, keyword, "delete") for keyword in DELETE_ACTION_KEYWORDS),
    *((Intent.REMINDER, keyword, "modify") for keyword in MODIFY_ACTION_KEYWORDS),
    *((Intent.REMINDER, keyword, "add") for keyword in ADD_ACTION_KEYWORDS),
    *((Intent.REMINDER, keyword, "turn_on") for keyword in TURN_ON_ACTION_KEYWORDS),
    *((Intent.REMINDER, keyword, "turn_off") for keyword in TURN_OFF_ACTION_KEYWORDS),
)


@pytest.mark.parametrize(("intent", "keyword", "expected"), ALL_ACTION_CASES)
def test_every_configured_keyword_maps_to_its_single_action(
    intent: Intent,
    keyword: str,
    expected: str,
) -> None:
    decision = classify_action_request(f"Please {keyword} the requested item.", intent)

    assert decision.selected_action == expected, (keyword, decision)


@pytest.mark.parametrize(
    ("intent", "query", "expected"),
    (
        (Intent.KNOWLEDGE_FACTS, "Forget the Atlas retention fact.", "delete"),
        (Intent.KNOWLEDGE_FACTS, "Revise Atlas retention to 45 days.", "modify"),
        (Intent.KNOWLEDGE_FACTS, "Remember that Atlas retention is 30 days.", "add"),
        (Intent.REMINDER, "Delete the payroll reminder.", "delete"),
        (Intent.REMINDER, "Move the payroll reminder to Friday.", "modify"),
        (Intent.REMINDER, "Set a reminder for payroll.", "add"),
        (Intent.REMINDER, "Turn back on the payroll reminder.", "turn_on"),
        (Intent.REMINDER, "Temporarily turn off the payroll reminder.", "turn_off"),
    ),
)
def test_explicit_keyword_selects_one_supported_branch_action(
    intent: Intent,
    query: str,
    expected: str,
) -> None:
    decision = classify_action_request(query, intent)

    assert decision.selected_action == expected
    assert decision.matched_actions == (expected,)


@pytest.mark.parametrize(
    ("query", "expected"),
    (
        ("Cancel and delete the legacy reminder.", "delete"),
        ("Set to inactive the payroll reminder.", "turn_off"),
        ("Set to active the payroll reminder.", "turn_on"),
    ),
)
def test_longer_explicit_phrase_owns_contained_cross_category_keyword(
    query: str,
    expected: str,
) -> None:
    decision = classify_action_request(query, Intent.REMINDER)

    assert decision.selected_action == expected
    assert decision.matched_actions == (expected,)


def test_independent_multi_action_request_fails_closed() -> None:
    result = DeterministicActionDetector().detect(
        ChatRequest(
            user_id="user",
            raw_query="Delete the old reminder and create a new reminder.",
        ),
        "Delete the old reminder and create a new reminder.",
        Intent.REMINDER,
    )

    assert result.requires_clarification
    assert result.metadata == {}
    assert result.missing_fields == ["single_action"]
    assert result.risk_flags == ["ambiguous_action_keywords"]


def test_multiple_payload_actions_are_rejected_even_for_one_keyword_category() -> None:
    result = DeterministicActionDetector().detect(
        ChatRequest(
            user_id="user",
            raw_query="Add these facts.",
            metadata={
                "knowledge_actions": [
                    {"action": "add", "text": "one"},
                    {"action": "add", "text": "two"},
                ]
            },
        ),
        "Add these facts.",
        Intent.KNOWLEDGE_FACTS,
    )

    assert result.requires_clarification
    assert not result.metadata
    assert result.risk_flags == ["multiple_action_payloads_rejected"]


def test_metadata_action_must_match_the_raw_query_keyword() -> None:
    result = DeterministicActionDetector().detect(
        ChatRequest(
            user_id="user",
            raw_query="Delete the Atlas fact.",
            metadata={"knowledge_actions": [{"action": "add", "text": "Atlas"}]},
        ),
        "Delete the Atlas fact.",
        Intent.KNOWLEDGE_FACTS,
    )

    assert result.requires_clarification
    assert result.risk_flags == ["action_payload_keyword_mismatch"]


def test_malformed_action_metadata_fails_closed() -> None:
    result = DeterministicActionDetector().detect(
        ChatRequest(
            user_id="user",
            raw_query="Delete the Atlas fact.",
            metadata={"knowledge_actions": ["delete"]},
        ),
        "Delete the Atlas fact.",
        Intent.KNOWLEDGE_FACTS,
    )

    assert result.requires_clarification
    assert result.risk_flags == ["invalid_action_payload_rejected"]


def test_rewritten_query_cannot_inject_an_action() -> None:
    result = DeterministicActionDetector().detect(
        ChatRequest(user_id="user", raw_query="What is the Atlas retention policy?"),
        "Delete the Atlas retention policy.",
        Intent.KNOWLEDGE_FACTS,
    )

    assert not result.requires_clarification
    assert result.metadata == {"knowledge_lookup": True}


def test_hyphenated_noun_does_not_create_a_second_action() -> None:
    decision = classify_action_request(
        "Update the primary data-store policy to the approved statement.",
        Intent.KNOWLEDGE_FACTS,
    )

    assert decision.selected_action == "modify"
    assert decision.matched_actions == ("modify",)


def test_branch_payload_authorization_requires_exactly_one_matching_action() -> None:
    request = ChatRequest(user_id="user", raw_query="Turn off the payroll reminder.")

    assert action_payload_is_authorized(
        request,
        Intent.REMINDER,
        [{"action": "turn_off", "target_description": "payroll"}],
    )[0]
    assert not action_payload_is_authorized(
        request,
        Intent.REMINDER,
        [{"action": "delete", "target_description": "payroll"}],
    )[0]
    assert not action_payload_is_authorized(
        request,
        Intent.REMINDER,
        [
            {"action": "turn_off", "target_description": "payroll"},
            {"action": "delete", "target_description": "payroll"},
        ],
    )[0]


def test_confirmation_replay_requires_matching_stored_authorization() -> None:
    action = {"action": "delete", "target_description": "Atlas"}
    base_metadata = {
        "confirmation_approved": True,
        "validated_knowledge_actions": [action],
        "knowledge_actions": [action],
    }
    valid_request = ChatRequest(
        user_id="user",
        raw_query="yes",
        confirmation_token="verified-token",
        metadata={
            **base_metadata,
            "action_authorization": {
                "intent": Intent.KNOWLEDGE_FACTS.value,
                "action": "delete",
            },
        },
    )
    invalid_request = ChatRequest(
        user_id="user",
        raw_query="yes",
        confirmation_token="verified-token",
        metadata={
            **base_metadata,
            "action_authorization": {
                "intent": Intent.KNOWLEDGE_FACTS.value,
                "action": "modify",
            },
        },
    )

    assert action_payload_is_authorized(valid_request, Intent.KNOWLEDGE_FACTS, [action])[0]
    assert not action_payload_is_authorized(invalid_request, Intent.KNOWLEDGE_FACTS, [action])[0]


def test_validated_action_builder_refuses_multi_action_lists() -> None:
    builder = ValidatedActionBuilder(
        config=SimpleNamespace(),
        knowledge_resolver=SimpleNamespace(),
        reminder_resolver=SimpleNamespace(),
    )
    actions = [{"action": "add"}, {"action": "delete"}]

    assert builder.build_knowledge_actions("u", actions, "q", "q", SimpleNamespace()) == []
    assert builder.build_reminder_actions("u", actions, "q", "q", SimpleNamespace()) == []


@pytest.mark.parametrize(
    ("branch", "intent", "query", "metadata"),
    (
        (
            KnowledgeFactsBranch(config=SimpleNamespace()),
            Intent.KNOWLEDGE_FACTS,
            "Add these facts.",
            {
                "knowledge_actions": [
                    {"action": "add", "text": "one"},
                    {"action": "add", "text": "two"},
                ]
            },
        ),
        (
            ReminderBranch(config=SimpleNamespace()),
            Intent.REMINDER,
            "Turn off these reminders.",
            {
                "reminder_actions": [
                    {"action": "turn_off", "target_description": "one"},
                    {"action": "turn_off", "target_description": "two"},
                ]
            },
        ),
    ),
)
def test_branch_rejects_multi_action_payload_before_repository_execution(
    branch: object,
    intent: Intent,
    query: str,
    metadata: dict,
) -> None:
    class FailingRepository:
        def __getattr__(self, name: str):
            raise AssertionError(f"multi-action request reached repository method {name}")

    context = SimpleNamespace(
        request=ChatRequest(user_id="user", raw_query=query, metadata=metadata),
        rewritten_query=query,
        intent=intent,
        approved_conversation_context=None,
    )

    result = branch.execute(context, FailingRepository())

    assert result.response_type is ResponseType.CLARIFICATION


def test_knowledge_branch_executes_one_matching_prevalidated_confirmation() -> None:
    action = {
        "action": "delete",
        "validation_result": "execute",
        "target_chunk_ids": ["chunk-1"],
        "confidence": 1.0,
    }
    request = ChatRequest(
        user_id="user",
        raw_query="yes",
        confirmation_token="verified-token",
        metadata={
            "confirmation_approved": True,
            "knowledge_actions": [action],
            "validated_knowledge_actions": [action],
            "action_authorization": {
                "intent": Intent.KNOWLEDGE_FACTS.value,
                "action": "delete",
            },
        },
    )
    context = SimpleNamespace(
        request=request,
        rewritten_query="yes",
        approved_conversation_context=None,
    )

    class RecordingRepository:
        calls = 0

        def transactional_knowledge_actions(self, **kwargs):
            self.calls += 1
            assert len(kwargs["actions"]) == 1
            assert kwargs["actions"][0].action.value == "delete"
            return SimpleNamespace(committed=True, results=(), audit_hop_id="hop")

    repository = RecordingRepository()
    branch = KnowledgeFactsBranch(
        config=SimpleNamespace(
            mutation_policy=SimpleNamespace(
                partial_execution_policy=MutationPartialExecutionPolicy.ALL_OR_NOTHING
            )
        ),
        validated_action_builder=SimpleNamespace(),
    )

    result = branch.execute(context, repository)

    assert result.response_type is ResponseType.KNOWLEDGE_ACTION
    assert repository.calls == 1


def test_reminder_branch_executes_one_matching_prevalidated_confirmation() -> None:
    action = {
        "action": "turn_off",
        "validation_result": "execute",
        "target_reminder_ids": ["reminder-1"],
        "confidence": 1.0,
    }
    request = ChatRequest(
        user_id="user",
        raw_query="yes",
        confirmation_token="verified-token",
        metadata={
            "confirmation_approved": True,
            "reminder_actions": [action],
            "validated_reminder_actions": [action],
            "action_authorization": {
                "intent": Intent.REMINDER.value,
                "action": "turn_off",
            },
        },
    )
    context = SimpleNamespace(
        request=request,
        rewritten_query="yes",
        approved_conversation_context=None,
    )

    class RecordingRepository:
        calls = 0

        def transactional_reminder_actions(self, **kwargs):
            self.calls += 1
            assert len(kwargs["actions"]) == 1
            assert kwargs["actions"][0].action.value == "turn_off"
            return SimpleNamespace(committed=True, results=(), audit_hop_id="hop")

    repository = RecordingRepository()
    branch = ReminderBranch(
        config=SimpleNamespace(
            mutation_policy=SimpleNamespace(
                partial_execution_policy=MutationPartialExecutionPolicy.ALL_OR_NOTHING
            ),
            default_timezone="UTC",
            confirmation_high_confidence_threshold=0.92,
        ),
        validated_action_builder=SimpleNamespace(),
    )

    result = branch.execute(context, repository)

    assert result.response_type is ResponseType.REMINDER_ACTION
    assert repository.calls == 1
