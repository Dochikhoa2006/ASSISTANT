from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from types import MethodType, SimpleNamespace
from typing import Any

import pytest

from assistant_rag.bundler import ChatOutput, ResponseBundler
from assistant_rag.chat_history import (
    current_chat_history,
    last_qa_chat_history,
    select_chat_history,
)
from assistant_rag.classification import LastQAResolver
from assistant_rag.contracts import (
    ApprovedConversationContext,
    BranchResult,
    ChatRequest,
    ExpectedResponseType,
    GeneratedQuestion,
    Intent,
    LastQAInteractionType,
    LastQAPath,
    LastQAResolution,
    LastQAState,
    QuestionSource,
    ResponseType,
    RetrievalResult,
)
from assistant_rag.llm import OllamaIntentClassifier
from assistant_rag.pipeline import AssistantPipeline


def _last_qa_state() -> LastQAState:
    return LastQAState(
        last_user_query="What is the Atlas rollout plan?",
        last_response="Atlas rolls out after QA signoff.",
        response_type=ResponseType.NORMAL,
        linked_topic_id="topic-atlas",
        linked_hop_id="hop-atlas",
        expected_response_type=ExpectedResponseType.FREE_TEXT_ANSWER,
    )


def _resolution(*, state: LastQAState | None, skip: bool) -> LastQAResolution:
    return LastQAResolution(
        path=(
            LastQAPath.LATEST_CONTEXT_INTERACTION
            if skip
            else LastQAPath.BROAD_RETRIEVAL_REQUIRED
        ),
        rewritten_query="Continue Atlas",
        state=state,
        did_merge_query=False,
        skip_broad_retrieval=skip,
        interaction_type=(LastQAInteractionType.NORMAL_FOLLOW_UP if skip else None),
        question_source=QuestionSource.NONE,
        is_authoritative_state=skip,
    )


def _retrieval_result(entity_id: str = "hop-retrieved") -> RetrievalResult:
    return RetrievalResult(
        entity_type="conversation_hop",
        entity_id=entity_id,
        source_store_evidence={"sql": True},
        rerank_score=0.91,
        confidence=0.93,
        validation_status="candidate",
        payload={"hop_id": entity_id},
    )


def _approved_context(
    history: list[dict[str, Any]],
    *,
    status: str | None = None,
) -> ApprovedConversationContext:
    resolved_status = status or ("approved" if history else "empty")
    return ApprovedConversationContext(
        approved_conversation_history=history,
        human_supporting_questions=[],
        reminder_supporting_questions=[],
        clarification_question_context=None,
        extracted_expected_response_types=[],
        conversation_retrieval_ran=True,
        conversation_context_status=resolved_status,  # type: ignore[arg-type]
        approved_conversation_count=len(history),
    )


@dataclass
class _Store:
    state: LastQAState | None

    def __post_init__(self) -> None:
        self.saved: list[tuple[str, LastQAState]] = []

    def get(self, _user_id: str) -> LastQAState | None:
        return self.state

    def save(self, user_id: str, state: LastQAState) -> None:
        self.saved.append((user_id, state))


class _Resolver:
    def __init__(self, resolution: LastQAResolution) -> None:
        self.resolution = resolution
        self.calls = 0

    def resolve(self, *_args: Any) -> LastQAResolution:
        self.calls += 1
        return self.resolution


class _Retriever:
    def __init__(self, results: list[RetrievalResult]) -> None:
        self.results = results
        self.calls: list[dict[str, Any]] = []

    def retrieve_conversation(self, **kwargs: Any) -> list[RetrievalResult]:
        self.calls.append(kwargs)
        return list(self.results)


class _ContextFilter:
    def __init__(self, approved: ApprovedConversationContext | None) -> None:
        self.approved = approved
        self.calls: list[dict[str, Any]] = []

    def filter_conversation_only(self, **kwargs: Any) -> ApprovedConversationContext:
        self.calls.append(kwargs)
        assert self.approved is not None
        return self.approved


class _Repository:
    def __init__(self, hydrated: list[RetrievalResult]) -> None:
        self.hydrated = hydrated
        self.hydration_calls: list[dict[str, Any]] = []

    def hydrate_conversation_retrieval_results(
        self, **kwargs: Any
    ) -> list[RetrievalResult]:
        self.hydration_calls.append(kwargs)
        return list(self.hydrated)


def _pipeline_probe(
    *,
    resolution: LastQAResolution,
    after_last_qa_enabled: bool,
    before_intent_enabled: bool,
    raw_results: list[RetrievalResult],
    hydrated_results: list[RetrievalResult],
    approved_context: ApprovedConversationContext | None,
) -> tuple[AssistantPipeline, _Retriever, _ContextFilter, _Repository]:
    retriever = _Retriever(raw_results)
    context_filter = _ContextFilter(approved_context)
    repository = _Repository(hydrated_results)
    pipeline = AssistantPipeline(
        config=SimpleNamespace(
            context_filter=SimpleNamespace(
                conversation_retrieval_after_last_qa_enabled=after_last_qa_enabled,
                conversation_retrieval_before_intent_enabled=before_intent_enabled,
            )
        ),
        last_qa_store=_Store(resolution.state),
        query_rewriter=SimpleNamespace(rewrite=lambda query: query),
        last_qa_resolver=_Resolver(resolution),
        retriever=retriever,
        context_filter=context_filter,
        classifier=SimpleNamespace(),
        router=SimpleNamespace(),
        bundler=SimpleNamespace(),
        platform_selector=SimpleNamespace(),
        chat_output=SimpleNamespace(),
    )
    return pipeline, retriever, context_filter, repository


@pytest.mark.parametrize(
    ("skip_broad_retrieval", "after_last_qa_enabled", "before_intent_enabled"),
    list(product((False, True), repeat=3)),
)
def test_pipeline_retrieval_gate_exhaustive_boolean_matrix(
    skip_broad_retrieval: bool,
    after_last_qa_enabled: bool,
    before_intent_enabled: bool,
) -> None:
    state = _last_qa_state()
    candidate = _retrieval_result()
    retrieved_history = [{"hop_id": "hop-retrieved", "raw_user_query": "Earlier"}]
    approved = _approved_context(retrieved_history)
    pipeline, retriever, context_filter, repository = _pipeline_probe(
        resolution=_resolution(state=state, skip=skip_broad_retrieval),
        after_last_qa_enabled=after_last_qa_enabled,
        before_intent_enabled=before_intent_enabled,
        raw_results=[candidate],
        hydrated_results=[candidate],
        approved_context=approved,
    )
    observed: dict[str, Any] = {}

    def stop_after_history_selection(self: AssistantPipeline, **kwargs: Any) -> str:
        observed.update(kwargs)
        observed["scoped_history"] = current_chat_history()
        return "probe-complete"

    pipeline._handle_with_chat_history = MethodType(stop_after_history_selection, pipeline)
    output = pipeline.handle(ChatRequest(user_id="user-1", raw_query="Continue"), repository)

    expected_to_run = (
        not skip_broad_retrieval
        and after_last_qa_enabled
        and before_intent_enabled
    )
    assert output == "probe-complete"
    assert observed["conversation_retrieval"] is expected_to_run
    assert observed["chat_history_source"] == (
        "conversation_retrieval" if expected_to_run else "last_qa"
    )
    assert observed["chat_history"] == (
        retrieved_history if expected_to_run else last_qa_chat_history(state)
    )
    assert observed["scoped_history"] == observed["chat_history"]
    assert observed["resolution"].state is (None if expected_to_run else state)
    assert len(retriever.calls) == int(expected_to_run)
    assert len(repository.hydration_calls) == int(expected_to_run)
    assert len(context_filter.calls) == int(expected_to_run)
    assert current_chat_history() == []


@pytest.mark.parametrize(
    (
        "case",
        "raw_results",
        "hydrated_results",
        "filtered_context",
        "expected_status",
        "expected_history",
        "expected_filter_calls",
    ),
    (
        ("raw_empty", [], [], None, "empty", [], 0),
        ("sql_all_rejected", [_retrieval_result()], [], None, "all_rejected", [], 0),
        (
            "context_all_rejected",
            [_retrieval_result()],
            [_retrieval_result()],
            _approved_context([], status="all_rejected"),
            "all_rejected",
            [],
            1,
        ),
        (
            "approved",
            [_retrieval_result()],
            [_retrieval_result()],
            _approved_context([{"hop_id": "hop-retrieved", "raw_response": "Earlier"}]),
            "approved",
            [{"hop_id": "hop-retrieved", "raw_response": "Earlier"}],
            1,
        ),
    ),
)
def test_pipeline_retrieval_outcome_matrix_is_authoritative(
    case: str,
    raw_results: list[RetrievalResult],
    hydrated_results: list[RetrievalResult],
    filtered_context: ApprovedConversationContext | None,
    expected_status: str,
    expected_history: list[dict[str, Any]],
    expected_filter_calls: int,
) -> None:
    del case
    state = _last_qa_state()
    pipeline, retriever, context_filter, repository = _pipeline_probe(
        resolution=_resolution(state=state, skip=False),
        after_last_qa_enabled=True,
        before_intent_enabled=True,
        raw_results=raw_results,
        hydrated_results=hydrated_results,
        approved_context=filtered_context,
    )
    observed: dict[str, Any] = {}

    def stop_after_history_selection(self: AssistantPipeline, **kwargs: Any) -> str:
        observed.update(kwargs)
        observed["scoped_history"] = current_chat_history()
        return "probe-complete"

    pipeline._handle_with_chat_history = MethodType(stop_after_history_selection, pipeline)
    output = pipeline.handle(ChatRequest(user_id="user-1", raw_query="Find context"), repository)

    approved = observed["approved_conversation_context"]
    assert output == "probe-complete"
    assert len(retriever.calls) == 1
    assert len(repository.hydration_calls) == 1
    assert len(context_filter.calls) == expected_filter_calls
    assert observed["conversation_results"] == hydrated_results
    assert observed["conversation_retrieval"] is True
    assert observed["chat_history_source"] == "conversation_retrieval"
    assert observed["resolution"].state is None
    assert approved is not None
    assert approved.conversation_context_status == expected_status
    assert approved.approved_conversation_count == len(expected_history)
    if expected_status == "all_rejected" and raw_results and not hydrated_results:
        assert approved._rejected_conversation_ids == tuple(
            result.entity_id for result in raw_results
        )
    assert observed["chat_history"] == expected_history
    assert observed["scoped_history"] == expected_history
    assert current_chat_history() == []


@pytest.mark.parametrize("conversation_retrieval", (False, True))
@pytest.mark.parametrize("last_qa_present", (False, True))
@pytest.mark.parametrize("approved_variant", ("none", "empty", "approved"))
def test_chat_history_source_precedence_complete_state_matrix(
    conversation_retrieval: bool,
    last_qa_present: bool,
    approved_variant: str,
) -> None:
    state = _last_qa_state() if last_qa_present else None
    retrieved_history = [{"hop_id": "hop-approved", "raw_user_query": "Earlier"}]
    approved_context = {
        "none": None,
        "empty": _approved_context([]),
        "approved": _approved_context(retrieved_history),
    }[approved_variant]

    selected = select_chat_history(
        conversation_retrieval=conversation_retrieval,
        approved_conversation_context=approved_context,
        last_qa_state=state,
    )

    if conversation_retrieval:
        expected = retrieved_history if approved_variant == "approved" else []
    else:
        expected = last_qa_chat_history(state)
    assert selected == expected


class _Classifier:
    def __init__(self, intent: Intent) -> None:
        self.intent = intent
        self.calls: list[dict[str, Any]] = []

    def classify(
        self,
        request: ChatRequest,
        rewritten_query: str,
        **kwargs: Any,
    ) -> Intent:
        self.calls.append(
            {"request": request, "rewritten_query": rewritten_query, **kwargs}
        )
        return self.intent


class _Router:
    def __init__(self, response_type: ResponseType) -> None:
        self.response_type = response_type
        self.contexts: list[Any] = []

    def route(self, context: Any, _repository: Any) -> BranchResult:
        self.contexts.append(context)
        return BranchResult(
            response_type=self.response_type,
            normal_response_text=f"Handled {context.intent.value}",
        )


class _PlatformSelector:
    def select(self, _bundled: Any, _request: ChatRequest) -> dict[str, Any]:
        return {"delivery": {"channel": "none", "status": "not_requested"}}


@pytest.mark.parametrize(
    ("intent", "response_type"),
    (
        (Intent.CLARIFICATION, ResponseType.CLARIFICATION),
        (Intent.GENERAL_RESPONSE, ResponseType.NORMAL),
        (Intent.KNOWLEDGE_FACTS, ResponseType.KNOWLEDGE_ACTION),
        (Intent.REMINDER, ResponseType.REMINDER_ACTION),
    ),
)
def test_pipeline_routes_each_intent_exactly_once(
    intent: Intent,
    response_type: ResponseType,
) -> None:
    state = _last_qa_state()
    store = _Store(state)
    classifier = _Classifier(intent)
    router = _Router(response_type)
    pipeline = AssistantPipeline(
        config=SimpleNamespace(
            context_filter=SimpleNamespace(
                conversation_retrieval_after_last_qa_enabled=True,
                conversation_retrieval_before_intent_enabled=True,
            )
        ),
        last_qa_store=store,
        query_rewriter=SimpleNamespace(rewrite=lambda query: query),
        last_qa_resolver=_Resolver(_resolution(state=state, skip=True)),
        retriever=_Retriever([]),
        context_filter=_ContextFilter(None),
        classifier=classifier,
        router=router,
        bundler=ResponseBundler(),
        platform_selector=_PlatformSelector(),
        chat_output=ChatOutput(),
    )
    repository = _Repository([])

    response = pipeline.handle(
        ChatRequest(user_id="user-1", raw_query=f"Route {intent.value}"),
        repository,
    )

    assert response.response_type is response_type
    assert response.final_chat_text == f"Handled {intent.value}"
    assert len(classifier.calls) == 1
    assert len(router.contexts) == 1
    routed_context = router.contexts[0]
    assert routed_context.intent is intent
    assert routed_context.conversation_retrieval is False
    assert routed_context.chat_history_source == "last_qa"
    assert routed_context.chat_history == last_qa_chat_history(state)
    assert classifier.calls[0]["last_qa_resolution"].state is state
    assert classifier.calls[0]["approved_conversation_context"] is None
    assert len(store.saved) == 1
    assert current_chat_history() == []


class _LLMStub:
    def __init__(
        self,
        *,
        payload: dict[str, Any] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.payload = payload
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def generate_json(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        assert self.payload is not None
        return dict(self.payload)


def _pending_clarification_resolution() -> LastQAResolution:
    question = GeneratedQuestion(
        text="Which Atlas environment?",
        source=QuestionSource.CLARIFICATION_QUESTION,
        purpose="resolve_environment",
        confidence=1.0,
        expected_response_type=ExpectedResponseType.SELECTION_ANSWER,
    )
    state = LastQAState(
        last_user_query="Deploy Atlas",
        last_response="Which Atlas environment?",
        response_type=ResponseType.CLARIFICATION,
        clarification_question=question,
    )
    return LastQAResolution(
        path=LastQAPath.BROAD_RETRIEVAL_REQUIRED,
        rewritten_query="Production",
        state=state,
        did_merge_query=False,
        skip_broad_retrieval=False,
    )


def test_deterministic_last_qa_resolver_no_state_requires_broad_retrieval() -> None:
    resolution = LastQAResolver().resolve(
        ChatRequest(user_id="user-1", raw_query="New question"),
        "New question",
        None,
    )

    assert resolution.path is LastQAPath.NO_LAST_QA
    assert resolution.state is None
    assert resolution.skip_broad_retrieval is False
    assert resolution.is_authoritative_state is False


def test_deterministic_last_qa_resolver_unresolved_clarification_is_not_reused() -> None:
    question = GeneratedQuestion(
        text="Which environment?",
        source=QuestionSource.CLARIFICATION_QUESTION,
        purpose="resolve_environment",
        confidence=1.0,
    )
    state = LastQAState(
        last_user_query="Deploy Atlas",
        last_response="Which environment?",
        response_type=ResponseType.CLARIFICATION,
        clarification_question=question,
    )

    resolution = LastQAResolver().resolve(
        ChatRequest(user_id="user-1", raw_query="A different request"),
        "A different request",
        state,
    )

    assert resolution.path is LastQAPath.BROAD_RETRIEVAL_REQUIRED
    assert resolution.state is None
    assert resolution.skip_broad_retrieval is False
    assert resolution.is_authoritative_state is False


@pytest.mark.parametrize("linked", (False, True))
def test_deterministic_last_qa_resolver_supporting_answer_requires_linked_hop(
    linked: bool,
) -> None:
    supporting = GeneratedQuestion(
        text="Which Atlas region?",
        source=QuestionSource.HUMAN_SUPPORTING_QUESTION,
        purpose="resolve_region",
        confidence=1.0,
    )
    state = LastQAState(
        last_user_query="Plan Atlas",
        last_response="Atlas is ready.",
        response_type=ResponseType.NORMAL,
        supporting_questions=[supporting],
        linked_topic_id="topic-atlas" if linked else None,
        linked_hop_id="hop-atlas" if linked else None,
    )

    resolution = LastQAResolver().resolve(
        ChatRequest(user_id="user-1", raw_query="Which Atlas region?"),
        "Which Atlas region?",
        state,
    )

    assert resolution.path is (
        LastQAPath.LATEST_CONTEXT_INTERACTION
        if linked
        else LastQAPath.BROAD_RETRIEVAL_REQUIRED
    )
    assert resolution.skip_broad_retrieval is linked
    assert resolution.state is state
    assert resolution.interaction_type is (
        LastQAInteractionType.SUPPORTING_QUESTION_ANSWER if linked else None
    )
    assert resolution.is_authoritative_state is linked


def test_deterministic_last_qa_resolver_unmatched_normal_state_is_rejected() -> None:
    state = _last_qa_state()

    resolution = LastQAResolver().resolve(
        ChatRequest(user_id="user-1", raw_query="Completely new subject"),
        "Completely new subject",
        state,
    )

    assert resolution.path is LastQAPath.BROAD_RETRIEVAL_REQUIRED
    assert resolution.state is None
    assert resolution.skip_broad_retrieval is False
    assert resolution.is_authoritative_state is False


@pytest.mark.parametrize("source_matches", (False, True))
def test_deterministic_last_qa_resolver_reminder_reply_requires_exact_source_hop(
    source_matches: bool,
) -> None:
    state = LastQAState(
        last_user_query="Medication reminder",
        last_response="It is time for your medication.",
        response_type=ResponseType.REMINDER_REPLY,
        linked_topic_id="topic-reminder",
        linked_hop_id="hop-reminder",
    )
    request = ChatRequest(
        user_id="user-1",
        raw_query="Done",
        metadata={
            "reminder_reply_context": True,
            "reminder_id": "reminder-1",
            "notification_id": "notification-1",
            "source_topic_id": "topic-reminder",
            "source_hop_id": "hop-reminder" if source_matches else "wrong-hop",
        },
    )

    resolution = LastQAResolver().resolve(request, "Done", state)

    if source_matches:
        assert resolution.path is LastQAPath.LATEST_CONTEXT_INTERACTION
        assert resolution.state is state
        assert resolution.skip_broad_retrieval is True
        assert (
            resolution.interaction_type
            is LastQAInteractionType.REMINDER_NOTIFICATION_REPLY
        )
        assert resolution.is_authoritative_state is True
    else:
        assert resolution.path is LastQAPath.BROAD_RETRIEVAL_REQUIRED
        assert resolution.state is None
        assert resolution.skip_broad_retrieval is False
        assert resolution.is_authoritative_state is False


@pytest.mark.parametrize(
    ("intent_value", "operation_kind", "expected"),
    (
        ("general_response", "none", Intent.GENERAL_RESPONSE),
        ("knowledge_facts", "durable_knowledge", Intent.KNOWLEDGE_FACTS),
        ("reminder", "reminder_lifecycle", Intent.REMINDER),
        ("clarification", "clarification_reply", Intent.CLARIFICATION),
    ),
)
def test_production_intent_classifier_operation_routes(
    intent_value: str,
    operation_kind: str,
    expected: Intent,
) -> None:
    llm = _LLMStub(
        payload={
            "intent": intent_value,
            "operation_kind": operation_kind,
            "confidence": 1.0,
        }
    )
    classifier = OllamaIntentClassifier(llm, min_confidence=0.7)

    result = classifier.classify(
        ChatRequest(user_id="user-1", raw_query="Route this"),
        "Route this",
        last_qa_resolution=_pending_clarification_resolution(),
        approved_conversation_context=None,
    )

    assert result is expected
    assert len(llm.calls) == 1


@pytest.mark.parametrize(
    ("payload", "error", "resolution"),
    (
        (None, RuntimeError("model unavailable"), None),
        ({"intent": "reminder", "operation_kind": "reminder_lifecycle"}, None, None),
        (
            {
                "intent": "reminder",
                "operation_kind": "reminder_lifecycle",
                "confidence": 0.69,
            },
            None,
            None,
        ),
        (
            {
                "intent": "clarification",
                "operation_kind": "clarification_reply",
                "confidence": 1.0,
            },
            None,
            None,
        ),
    ),
)
def test_production_intent_classifier_failures_are_safe_general_fallbacks(
    payload: dict[str, Any] | None,
    error: Exception | None,
    resolution: LastQAResolution | None,
) -> None:
    llm = _LLMStub(payload=payload, error=error)
    classifier = OllamaIntentClassifier(llm, min_confidence=0.7)

    result = classifier.classify(
        ChatRequest(user_id="user-1", raw_query="Ambiguous request"),
        "Ambiguous request",
        last_qa_resolution=resolution,
        approved_conversation_context=None,
    )

    assert result is Intent.GENERAL_RESPONSE
    assert len(llm.calls) == 1


@pytest.mark.parametrize("intent", tuple(Intent))
def test_production_intent_classifier_valid_metadata_override_bypasses_llm(
    intent: Intent,
) -> None:
    llm = _LLMStub(error=AssertionError("LLM must not run"))
    classifier = OllamaIntentClassifier(llm, min_confidence=0.7)

    result = classifier.classify(
        ChatRequest(
            user_id="user-1",
            raw_query="Explicit route",
            metadata={"intent": intent.value},
        ),
        "Explicit route",
    )

    assert result is intent
    assert llm.calls == []


def test_pipeline_restores_canonical_history_when_classifier_raises() -> None:
    state = _last_qa_state()
    pipeline, _retriever, _context_filter, repository = _pipeline_probe(
        resolution=_resolution(state=state, skip=True),
        after_last_qa_enabled=True,
        before_intent_enabled=True,
        raw_results=[],
        hydrated_results=[],
        approved_context=None,
    )

    class FailingClassifier:
        def classify(self, *_args: Any, **_kwargs: Any) -> Intent:
            assert current_chat_history() == last_qa_chat_history(state)
            raise RuntimeError("classification failed")

    pipeline.classifier = FailingClassifier()

    with pytest.raises(RuntimeError, match="classification failed"):
        pipeline.handle(ChatRequest(user_id="user-1", raw_query="Continue"), repository)

    assert current_chat_history() == []
