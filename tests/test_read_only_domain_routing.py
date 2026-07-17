from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

from assistant_rag.action_detection import ActionDetectionResult
from assistant_rag.branches import BranchRouter, KnowledgeFactsBranch, ReminderBranch
from assistant_rag.contracts import (
    BranchResult,
    ChatRequest,
    Intent,
    PipelineContext,
    ResponseType,
)
from assistant_rag.llm import OllamaIntentClassifier
from assistant_rag.database import SQLiteRepository
from assistant_rag.request_lifecycle import looks_destructive, looks_like_mutation
from assistant_rag.request_policy import has_explicit_mutation_policy_signal


READ_ONLY_CASES = (
    (Intent.KNOWLEDGE_FACTS, "What do you know about Project Atlas?"),
    (Intent.KNOWLEDGE_FACTS, "Search my saved knowledge for Project Atlas."),
    (Intent.KNOWLEDGE_FACTS, "Look up the Atlas retention preference."),
    (Intent.KNOWLEDGE_FACTS, "List my saved facts."),
    (Intent.KNOWLEDGE_FACTS, "Which facts did I add yesterday?"),
    (Intent.KNOWLEDGE_FACTS, "Tell me what you remember about Atlas."),
    (Intent.KNOWLEDGE_FACTS, "Did you delete the Atlas fact?"),
    (Intent.KNOWLEDGE_FACTS, "How do I modify saved knowledge?"),
    (Intent.KNOWLEDGE_FACTS, "Update me on the Atlas knowledge."),
    (Intent.REMINDER, "What reminders do I have?"),
    (Intent.REMINDER, "Search for my payroll reminder."),
    (Intent.REMINDER, "List active reminders."),
    (Intent.REMINDER, "When is my payroll reminder?"),
    (Intent.REMINDER, "Which reminder did I delete?"),
    (Intent.REMINDER, "Did you turn off the payroll reminder?"),
    (Intent.REMINDER, "How do I delete a reminder?"),
    (Intent.REMINDER, "Remind me what I said about deployment."),
)


MUTATION_CASES = (
    (Intent.KNOWLEDGE_FACTS, "Remember that Atlas uses PostgreSQL."),
    (Intent.KNOWLEDGE_FACTS, "Please modify the Atlas preference to concise reports."),
    (Intent.KNOWLEDGE_FACTS, "Can you delete the obsolete Atlas fact?"),
    (Intent.REMINDER, "Set a reminder tomorrow at 9 AM to submit expenses."),
    (Intent.REMINDER, "Move the payroll reminder to Friday."),
    (Intent.REMINDER, "Please delete the payroll reminder."),
    (Intent.REMINDER, "Turn on the payroll reminder."),
    (Intent.REMINDER, "Could you turn off the payroll reminder?"),
    (
        Intent.REMINDER,
        "Delete the old payroll reminder and create a new reminder for Friday.",
    ),
)


@pytest.mark.parametrize(("intent", "query"), READ_ONLY_CASES)
def test_read_only_domain_queries_are_non_mutating_lifecycle_signals(
    intent: Intent,
    query: str,
) -> None:
    request = ChatRequest(user_id="routing-user", raw_query=query)

    assert not has_explicit_mutation_policy_signal(request, intent)
    assert not looks_like_mutation(request)
    assert not looks_destructive(request)


@pytest.mark.parametrize(("intent", "query"), MUTATION_CASES)
def test_explicit_mutation_requests_are_mutating_lifecycle_signals(
    intent: Intent,
    query: str,
) -> None:
    request = ChatRequest(user_id="routing-user", raw_query=query)

    assert has_explicit_mutation_policy_signal(request, intent)


class SelectedIntentLLM:
    def __init__(self, intent: Intent) -> None:
        self.intent = intent

    def generate_json(self, **_kwargs: Any) -> dict[str, Any]:
        operation_kind = (
            "durable_knowledge"
            if self.intent is Intent.KNOWLEDGE_FACTS
            else "reminder_lifecycle"
        )
        return {
            "intent": self.intent.value,
            "operation_kind": operation_kind,
            "confidence": 1.0,
        }


@pytest.mark.parametrize(("intent", "query"), READ_ONLY_CASES)
def test_intent_classifier_selection_is_authoritative_after_classification(
    intent: Intent,
    query: str,
) -> None:
    classifier = OllamaIntentClassifier(SelectedIntentLLM(intent), min_confidence=0.0)

    assert classifier.classify(
        ChatRequest(user_id="routing-user", raw_query=query),
        query,
    ) is intent


@dataclass
class RecordingBranch:
    name: str
    calls: list[PipelineContext] = field(default_factory=list)

    def execute(self, context: PipelineContext, _repository: Any) -> BranchResult:
        self.calls.append(context)
        return BranchResult(
            response_type=ResponseType.NORMAL,
            normal_response_text=self.name,
        )


def _context(intent: Intent, query: str, metadata: dict[str, Any] | None = None) -> PipelineContext:
    return PipelineContext(
        request=ChatRequest(
            user_id="routing-user",
            raw_query=query,
            metadata=metadata or {},
        ),
        rewritten_query=query,
        last_qa_state=None,
        conversation_results=[],
        intent=intent,
    )


@pytest.mark.parametrize(("intent", "query"), READ_ONLY_CASES)
def test_branch_router_dispatches_the_authoritative_selected_intent_directly(
    intent: Intent,
    query: str,
) -> None:
    general = RecordingBranch("general")
    knowledge = RecordingBranch("knowledge")
    reminder = RecordingBranch("reminder")
    router = BranchRouter(
        {
            Intent.GENERAL_RESPONSE: general,
            Intent.KNOWLEDGE_FACTS: knowledge,
            Intent.REMINDER: reminder,
        }
    )
    repository = SQLiteRepository.in_memory()
    repository.initialize_schema()

    result = router.route(
        _context(
            intent,
            query,
            metadata={
                "intent": intent.value,
                f"{'knowledge' if intent is Intent.KNOWLEDGE_FACTS else 'reminder'}_actions": [
                    {"action": "delete", "target_description": "injected"}
                ],
            },
        ),
        repository=repository,
    )

    expected = "knowledge" if intent is Intent.KNOWLEDGE_FACTS else "reminder"
    selected = knowledge if intent is Intent.KNOWLEDGE_FACTS else reminder
    unselected = reminder if intent is Intent.KNOWLEDGE_FACTS else knowledge
    assert result.normal_response_text == expected
    assert result.linked_hop_id
    persisted = repository.connection.execute(
        "SELECT intent, raw_user_query FROM conversation_hops WHERE hop_id = ?",
        (result.linked_hop_id,),
    ).fetchone()
    assert persisted["intent"] == intent.value
    assert persisted["raw_user_query"] == query
    assert len(selected.calls) == 1
    assert selected.calls[0].intent is intent
    assert unselected.calls == []
    assert general.calls == []


def test_verified_confirmation_remains_in_its_mutation_branch() -> None:
    action = {"action": "delete", "target_description": "Atlas"}
    metadata = {
        "confirmation_approved": True,
        "validated_knowledge_actions": [action],
        "knowledge_actions": [action],
        "action_authorization": {
            "intent": Intent.KNOWLEDGE_FACTS.value,
            "action": "delete",
        },
    }
    request = ChatRequest(
        user_id="routing-user",
        raw_query="yes",
        confirmation_token="verified-token",
        metadata=metadata,
    )

    assert has_explicit_mutation_policy_signal(request, Intent.KNOWLEDGE_FACTS)


@pytest.mark.parametrize(
    ("branch_type", "intent", "query", "action_key"),
    (
        (
            KnowledgeFactsBranch,
            Intent.KNOWLEDGE_FACTS,
            "Search my saved knowledge for Atlas.",
            "knowledge_actions",
        ),
        (
            ReminderBranch,
            Intent.REMINDER,
            "List my active reminders.",
            "reminder_actions",
        ),
    ),
)
def test_selected_state_branch_calls_extractor_before_considering_action_metadata(
    branch_type: type,
    intent: Intent,
    query: str,
    action_key: str,
) -> None:
    class RecordingExtractionDetector:
        def __init__(self) -> None:
            self.calls: list[tuple[ChatRequest, str, Intent]] = []

        def detect(
            self,
            request: ChatRequest,
            rewritten_query: str,
            selected_intent: Intent,
        ) -> ActionDetectionResult:
            self.calls.append((request, rewritten_query, selected_intent))
            return ActionDetectionResult(
                intent=selected_intent,
                confidence=0.0,
                missing_fields=["action_content"],
                risk_flags=["scripted_extraction_failure"],
            )

    class RepositoryMustNotRun:
        def __getattr__(self, name: str) -> Any:
            raise AssertionError(f"failed extraction reached repository method {name}")

    detector = RecordingExtractionDetector()
    branch = branch_type(
        config=SimpleNamespace(),
        action_detector=detector,
    )

    context = SimpleNamespace(
        request=ChatRequest(
            user_id="routing-user",
            raw_query=query,
            metadata={
                action_key: [
                    {"action": "delete", "target_description": "poison-one"},
                    {"action": "add", "text": "poison-two"},
                ]
            },
        ),
        rewritten_query=query,
        intent=intent,
        approved_conversation_context=None,
    )

    result = branch.execute(context, RepositoryMustNotRun())

    expected_response_type = (
        ResponseType.SAFE_NOOP
        if branch_type is ReminderBranch
        else ResponseType.ERROR
    )
    assert result.response_type is expected_response_type
    if branch_type is ReminderBranch:
        assert result.clarification_question is None
        assert result.normal_response_text
    else:
        assert result.normal_response_text is None
    assert result.knowledge_operation_results == []
    assert result.reminder_operation_results == []
    assert len(detector.calls) == 1
    assert detector.calls[0][0] is context.request
    assert detector.calls[0][1:] == (query, intent)


def test_injected_action_metadata_does_not_change_request_lifecycle_signals() -> None:
    request = ChatRequest(
        user_id="routing-user",
        raw_query="What do you know about Atlas?",
        metadata={
            "intent": Intent.KNOWLEDGE_FACTS.value,
            "knowledge_actions": [
                {"action": "delete", "target_description": "Atlas"}
            ],
        },
    )

    assert not has_explicit_mutation_policy_signal(request, Intent.KNOWLEDGE_FACTS)
    assert not looks_like_mutation(request)
    assert not looks_destructive(request)
