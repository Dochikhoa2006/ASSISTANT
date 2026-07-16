from __future__ import annotations

from types import SimpleNamespace

import pytest

import assistant_rag.action_detection as action_detection
from assistant_rag.action_detection import ActionDetectionResult
from assistant_rag.branch_orchestration import ValidatedActionBuilder
from assistant_rag.branches import KnowledgeFactsBranch, ReminderBranch
from assistant_rag.contracts import (
    ActionValidationResult,
    ChatRequest,
    Intent,
    KnowledgeAction,
    ReminderAction,
    ResponseType,
    ValidatedKnowledgeAction,
    ValidatedReminderAction,
)
from assistant_rag.settings import MutationPartialExecutionPolicy


class ScriptedExtractionDetector:
    def __init__(
        self,
        *,
        metadata: dict | None = None,
        missing_fields: list[str] | None = None,
    ) -> None:
        self.metadata = dict(metadata or {})
        self.missing_fields = list(missing_fields or [])
        self.calls: list[tuple[ChatRequest, str, Intent]] = []

    def detect(
        self,
        request: ChatRequest,
        rewritten_query: str,
        intent: Intent,
    ) -> ActionDetectionResult:
        self.calls.append((request, rewritten_query, intent))
        return ActionDetectionResult(
            intent=intent,
            confidence=0.0 if self.missing_fields else 1.0,
            metadata=dict(self.metadata),
            missing_fields=list(self.missing_fields),
            risk_flags=(
                ["scripted_extraction_failure"] if self.missing_fields else []
            ),
        )


def test_action_detection_surface_contains_only_neutral_extraction_contracts() -> None:
    prohibited_keyword_detectors = (
        "ActionKeywordDecision",
        "ActionKeywordMatch",
        "DeterministicActionDetector",
        "NoOpActionDetector",
        "action_payload_is_authorized",
        "classify_action_request",
        "enforce_mutation_only_intent",
        "request_has_explicit_mutation",
    )

    assert hasattr(action_detection, "ActionDetectionResult")
    assert hasattr(action_detection, "ActionDetector")
    assert all(
        not hasattr(action_detection, name) for name in prohibited_keyword_detectors
    )


def test_action_detection_result_requires_clarification_only_from_model_contract() -> None:
    complete = ActionDetectionResult(intent=Intent.KNOWLEDGE_FACTS, confidence=1.0)
    incomplete = ActionDetectionResult(
        intent=Intent.REMINDER,
        confidence=1.0,
        missing_fields=["action_content"],
    )

    assert not complete.requires_clarification
    assert incomplete.requires_clarification


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
    ("branch_type", "intent", "query", "action_key"),
    (
        (
            KnowledgeFactsBranch,
            Intent.KNOWLEDGE_FACTS,
            "Atlas has a blue retention label.",
            "knowledge_actions",
        ),
        (
            ReminderBranch,
            Intent.REMINDER,
            "The Payroll reminder.",
            "reminder_actions",
        ),
    ),
)
def test_poisoned_multi_action_metadata_cannot_bypass_failed_extraction(
    branch_type: type,
    intent: Intent,
    query: str,
    action_key: str,
) -> None:
    detector = ScriptedExtractionDetector(missing_fields=["action_content"])
    branch = branch_type(
        config=SimpleNamespace(),
        action_detector=detector,
    )

    class FailingRepository:
        def __getattr__(self, name: str):
            raise AssertionError(f"failed extraction reached repository method {name}")

    context = SimpleNamespace(
        request=ChatRequest(
            user_id="user",
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

    result = branch.execute(context, FailingRepository())

    assert result.response_type is ResponseType.CLARIFICATION
    assert len(detector.calls) == 1
    assert detector.calls[0][0] is context.request


@pytest.mark.parametrize(
    (
        "branch_type",
        "intent",
        "query",
        "action_key",
        "extracted_action",
        "repository_method",
    ),
    (
        (
            KnowledgeFactsBranch,
            Intent.KNOWLEDGE_FACTS,
            "Atlas has a blue retention label.",
            "knowledge_actions",
            {"action": "add", "text": "Atlas has a blue retention label."},
            "transactional_knowledge_actions",
        ),
        (
            ReminderBranch,
            Intent.REMINDER,
            "The Payroll reminder.",
            "reminder_actions",
            {"action": "turn_on", "target_description": "Payroll"},
            "transactional_reminder_actions",
        ),
    ),
)
def test_branch_executes_only_the_extractor_action_not_poisoned_metadata(
    branch_type: type,
    intent: Intent,
    query: str,
    action_key: str,
    extracted_action: dict,
    repository_method: str,
) -> None:
    detector = ScriptedExtractionDetector(
        metadata={action_key: [extracted_action]},
    )
    config = SimpleNamespace(
        default_timezone="UTC",
        confirmation_high_confidence_threshold=0.92,
    )
    branch_kwargs = {"config": config, "action_detector": detector}
    if branch_type is KnowledgeFactsBranch:
        class StaticKnowledgePipeline:
            def build_action(self, **kwargs):
                assert kwargs["action_payload"] == extracted_action
                return ValidatedKnowledgeAction(
                    action=KnowledgeAction.ADD,
                    validation_result=ActionValidationResult.EXECUTE,
                    knowledge_text=str(extracted_action["text"]),
                    new_text=str(extracted_action["text"]),
                    confidence=1.0,
                )

        branch_kwargs["knowledge_mutation_pipeline"] = StaticKnowledgePipeline()
    else:
        class StaticReminderPipeline:
            def build_action(self, **kwargs):
                assert kwargs["action_payload"] == extracted_action
                return ValidatedReminderAction(
                    action=ReminderAction.TURN_ON,
                    validation_result=ActionValidationResult.EXECUTE,
                    target_reminder_ids=("reminder-1",),
                    confidence=1.0,
                )

        branch_kwargs["reminder_mutation_pipeline"] = StaticReminderPipeline()
    branch = branch_type(**branch_kwargs)

    class RecordingRepository:
        def __init__(self) -> None:
            self.calls: list[tuple[str, list[dict]]] = []

        def transactional_knowledge_actions(self, **kwargs):
            self.calls.append(("transactional_knowledge_actions", kwargs["actions"]))
            return SimpleNamespace(committed=True, results=(), audit_hop_id="hop")

        def transactional_reminder_actions(self, **kwargs):
            self.calls.append(("transactional_reminder_actions", kwargs["actions"]))
            return SimpleNamespace(committed=True, results=(), audit_hop_id="hop")

    repository = RecordingRepository()
    context = SimpleNamespace(
        request=ChatRequest(
            user_id="user",
            raw_query=query,
            metadata={
                action_key: [
                    {"action": "delete", "target_description": "poison-one"},
                    {"action": "modify", "target_description": "poison-two"},
                ]
            },
        ),
        rewritten_query=query,
        intent=intent,
        approved_conversation_context=None,
    )

    result = branch.execute(context, repository)

    expected_response = (
        ResponseType.KNOWLEDGE_ACTION
        if intent is Intent.KNOWLEDGE_FACTS
        else ResponseType.REMINDER_ACTION
    )
    assert result.response_type is expected_response
    assert len(detector.calls) == 1
    assert len(repository.calls) == 1
    assert repository.calls[0][0] == repository_method
    committed_action = repository.calls[0][1][0]
    if intent is Intent.KNOWLEDGE_FACTS:
        assert committed_action.action is KnowledgeAction.ADD
        assert committed_action.knowledge_text == extracted_action["text"]
    else:
        assert committed_action.action is ReminderAction.TURN_ON


def test_knowledge_branch_cannot_execute_without_three_stage_pipeline() -> None:
    extracted_action = {"action": "add", "text": "Atlas is blue."}
    detector = ScriptedExtractionDetector(
        metadata={"knowledge_actions": [extracted_action]}
    )
    branch = KnowledgeFactsBranch(
        config=SimpleNamespace(),
        action_detector=detector,
    )

    class MustNotWrite:
        def __getattr__(self, name: str):
            raise AssertionError(f"pipeline bypass reached repository method {name}")

    result = branch.execute(
        SimpleNamespace(
            request=ChatRequest(user_id="user", raw_query="Remember Atlas is blue."),
            rewritten_query="Remember Atlas is blue.",
            intent=Intent.KNOWLEDGE_FACTS,
            approved_conversation_context=None,
        ),
        MustNotWrite(),
    )

    assert result.response_type is ResponseType.ERROR
    assert len(detector.calls) == 1


def test_reminder_branch_cannot_bypass_llm2_with_legacy_builder() -> None:
    extracted_action = {"action": "turn_on", "retrieval_text": "Payroll"}
    detector = ScriptedExtractionDetector(
        metadata={"reminder_actions": [extracted_action]}
    )

    class MustNotBuild:
        def build_reminder_actions(self, *_args, **_kwargs):
            raise AssertionError("legacy reminder validation bypass was called")

    class MustNotWrite:
        def __getattr__(self, name: str):
            raise AssertionError(f"pipeline bypass reached repository method {name}")

    branch = ReminderBranch(
        config=SimpleNamespace(),
        action_detector=detector,
        validated_action_builder=MustNotBuild(),
    )
    result = branch.execute(
        SimpleNamespace(
            request=ChatRequest(user_id="user", raw_query="Turn on Payroll."),
            rewritten_query="Turn on Payroll.",
            intent=Intent.REMINDER,
            approved_conversation_context=None,
        ),
        MustNotWrite(),
    )

    assert result.response_type is ResponseType.ERROR
    assert len(detector.calls) == 1


def test_knowledge_branch_revalidates_matching_legacy_confirmation_before_execution() -> None:
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

    class RevalidatingPipeline:
        calls = 0

        def build_action(self, **kwargs):
            self.calls += 1
            assert kwargs["action_payload"]["action"] == "delete"
            return ValidatedKnowledgeAction(
                action=KnowledgeAction.DELETE,
                validation_result=ActionValidationResult.EXECUTE,
                target_chunk_ids=("chunk-1",),
                confidence=1.0,
            )

    pipeline = RevalidatingPipeline()
    detector = ScriptedExtractionDetector(
        metadata={"knowledge_actions": [action]},
    )
    branch = KnowledgeFactsBranch(
        config=SimpleNamespace(
            mutation_policy=SimpleNamespace(
                partial_execution_policy=MutationPartialExecutionPolicy.ALL_OR_NOTHING
            )
        ),
        action_detector=detector,
        knowledge_mutation_pipeline=pipeline,
    )

    result = branch.execute(context, repository)

    assert result.response_type is ResponseType.KNOWLEDGE_ACTION
    assert len(detector.calls) == 1
    assert detector.calls[0][0] is request
    assert pipeline.calls == 1
    assert repository.calls == 1


def test_reminder_branch_revalidates_one_matching_prevalidated_confirmation() -> None:
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
    detector = ScriptedExtractionDetector(
        metadata={"reminder_actions": [action]},
    )

    class RevalidatingPipeline:
        calls = 0

        def build_action(self, **kwargs):
            self.calls += 1
            assert kwargs["action_payload"] == action
            return ValidatedReminderAction(
                action=ReminderAction.TURN_OFF,
                validation_result=ActionValidationResult.EXECUTE,
                target_reminder_ids=("reminder-1",),
                confidence=1.0,
            )

    pipeline = RevalidatingPipeline()
    branch = ReminderBranch(
        config=SimpleNamespace(
            mutation_policy=SimpleNamespace(
                partial_execution_policy=MutationPartialExecutionPolicy.ALL_OR_NOTHING
            ),
            default_timezone="UTC",
            confirmation_high_confidence_threshold=0.92,
        ),
        action_detector=detector,
        reminder_mutation_pipeline=pipeline,
    )

    result = branch.execute(context, repository)

    assert result.response_type is ResponseType.REMINDER_ACTION
    assert len(detector.calls) == 1
    assert detector.calls[0][0] is request
    assert pipeline.calls == 1
    assert repository.calls == 1
