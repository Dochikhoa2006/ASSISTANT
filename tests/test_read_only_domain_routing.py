from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

from assistant_rag.action_detection import (
    DeterministicActionDetector,
    enforce_mutation_only_intent,
    request_has_explicit_mutation,
)
from assistant_rag.branches import BranchRouter, KnowledgeFactsBranch, ReminderBranch
from assistant_rag.contracts import (
    BranchResult,
    ChatRequest,
    Intent,
    PipelineContext,
    ResponseType,
)
from assistant_rag.llm import OllamaIntentClassifier
from assistant_rag.request_lifecycle import looks_destructive, looks_like_mutation


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
def test_read_only_domain_queries_are_deterministically_general(
    intent: Intent,
    query: str,
) -> None:
    request = ChatRequest(user_id="routing-user", raw_query=query)

    assert not request_has_explicit_mutation(request, intent)
    assert enforce_mutation_only_intent(request, intent) is Intent.GENERAL_RESPONSE
    assert not looks_like_mutation(request)
    assert not looks_destructive(request)


@pytest.mark.parametrize(("intent", "query"), MUTATION_CASES)
def test_explicit_mutation_requests_retain_the_state_branch(
    intent: Intent,
    query: str,
) -> None:
    request = ChatRequest(user_id="routing-user", raw_query=query)

    assert request_has_explicit_mutation(request, intent)
    assert enforce_mutation_only_intent(request, intent) is intent


class MisroutingLLM:
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
def test_intent_classifier_cannot_force_read_only_queries_into_state_branches(
    intent: Intent,
    query: str,
) -> None:
    classifier = OllamaIntentClassifier(MisroutingLLM(intent), min_confidence=0.0)

    assert classifier.classify(
        ChatRequest(user_id="routing-user", raw_query=query),
        query,
    ) is Intent.GENERAL_RESPONSE


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
def test_branch_router_is_a_second_fail_closed_general_route(
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
        repository=object(),
    )

    assert result.normal_response_text == "general"
    assert len(general.calls) == 1
    assert general.calls[0].intent is Intent.GENERAL_RESPONSE
    assert knowledge.calls == []
    assert reminder.calls == []


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

    assert request_has_explicit_mutation(request, Intent.KNOWLEDGE_FACTS)


@pytest.mark.parametrize(
    ("branch", "intent", "query"),
    (
        (
            KnowledgeFactsBranch(config=SimpleNamespace()),
            Intent.KNOWLEDGE_FACTS,
            "Search my saved knowledge for Atlas.",
        ),
        (
            ReminderBranch(config=SimpleNamespace()),
            Intent.REMINDER,
            "List my active reminders.",
        ),
        (
            KnowledgeFactsBranch(config=SimpleNamespace()),
            Intent.KNOWLEDGE_FACTS,
            "Did you delete the Atlas fact?",
        ),
        (
            ReminderBranch(config=SimpleNamespace()),
            Intent.REMINDER,
            "Which reminder did I turn off?",
        ),
    ),
)
def test_state_branches_have_no_read_or_answer_fallback(
    branch: Any,
    intent: Intent,
    query: str,
) -> None:
    class RepositoryMustNotRun:
        def __getattr__(self, name: str) -> Any:
            raise AssertionError(f"read-only state branch touched repository method {name}")

    context = SimpleNamespace(
        request=ChatRequest(user_id="routing-user", raw_query=query),
        rewritten_query=query,
        intent=intent,
        approved_conversation_context=None,
    )

    result = branch.execute(context, RepositoryMustNotRun())

    assert result.response_type is ResponseType.CLARIFICATION
    assert result.normal_response_text is None
    assert result.knowledge_operation_results == []
    assert result.reminder_operation_results == []


def test_action_detector_never_emits_a_read_operation_marker() -> None:
    result = DeterministicActionDetector().detect(
        ChatRequest(
            user_id="routing-user",
            raw_query="Look up the Atlas retention policy.",
        ),
        "Look up the Atlas retention policy.",
        Intent.KNOWLEDGE_FACTS,
    )

    assert result.requires_clarification
    assert result.metadata == {}
    assert "knowledge_lookup" not in result.metadata


def test_injected_action_metadata_cannot_turn_a_read_query_into_a_mutation() -> None:
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

    assert not request_has_explicit_mutation(request, Intent.KNOWLEDGE_FACTS)
    assert enforce_mutation_only_intent(
        request, Intent.KNOWLEDGE_FACTS
    ) is Intent.GENERAL_RESPONSE
    assert not looks_like_mutation(request)
    assert not looks_destructive(request)
