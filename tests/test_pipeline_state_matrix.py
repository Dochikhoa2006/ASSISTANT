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
from assistant_rag.classification import LLMLastQAResolver, LastQAResolver
from assistant_rag.config import GeneralPurposeConfig, LastQAConfig
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
    OutboundFollowUpAction,
    OutboundMessageState,
    QuestionSource,
    PipelineContext,
    ResponseType,
    RetrievalResult,
)
from assistant_rag.llm import (
    LLMTask,
    OllamaIntentClassifier,
    StructuredFallbackPayload,
)
from assistant_rag.generation import LLMGeneralHITLStrategy
from assistant_rag.pipeline import AssistantPipeline
from assistant_rag.platform import PlatformSelector
from assistant_rag.prompts import DEFAULT_PROMPT_REGISTRY
from assistant_rag.reminder_reply import build_reminder_state, reminder_state_hash


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
        linked_topic_id=(state.linked_topic_id if skip and state else None),
        linked_hop_id=(state.linked_hop_id if skip and state else None),
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
    def __init__(
        self,
        hydrated: list[RetrievalResult],
        *,
        active_link_valid: bool = True,
    ) -> None:
        self.hydrated = hydrated
        self.active_link_valid = active_link_valid
        self.hydration_calls: list[dict[str, Any]] = []
        self.link_validation_calls: list[dict[str, Any]] = []

    def is_active_conversation_link(self, **kwargs: Any) -> bool:
        self.link_validation_calls.append(kwargs)
        return self.active_link_valid

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
    active_link_valid: bool = True,
) -> tuple[AssistantPipeline, _Retriever, _ContextFilter, _Repository]:
    retriever = _Retriever(raw_results)
    context_filter = _ContextFilter(approved_context)
    repository = _Repository(
        hydrated_results,
        active_link_valid=active_link_valid,
    )
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
    retrieved_history = [
        {
            "hop_id": "hop-retrieved",
            "raw_user_query": "RAW_HISTORY_SENTINEL",
            "rewritten_user_query": "Earlier",
        }
    ]
    semantic_history = [
        {"hop_id": "hop-retrieved", "rewritten_user_query": "Earlier"}
    ]
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
        semantic_history if expected_to_run else last_qa_chat_history(state)
    )
    assert observed["scoped_history"] == observed["chat_history"]
    assert observed["resolution"].state is (None if expected_to_run else state)
    assert len(retriever.calls) == int(expected_to_run)
    assert len(repository.hydration_calls) == int(expected_to_run)
    assert len(context_filter.calls) == int(expected_to_run)
    assert current_chat_history() == []


def test_pipeline_rejects_stale_last_qa_link_before_skipping_broad_retrieval() -> None:
    state = _last_qa_state()
    pipeline, retriever, _context_filter, repository = _pipeline_probe(
        resolution=_resolution(state=state, skip=True),
        after_last_qa_enabled=True,
        before_intent_enabled=True,
        raw_results=[],
        hydrated_results=[],
        approved_context=None,
        active_link_valid=False,
    )
    observed: dict[str, Any] = {}

    def stop_after_history_selection(self: AssistantPipeline, **kwargs: Any) -> str:
        observed.update(kwargs)
        return "probe-complete"

    pipeline._handle_with_chat_history = MethodType(
        stop_after_history_selection,
        pipeline,
    )
    output = pipeline.handle(
        ChatRequest(user_id="user-1", raw_query="Continue"),
        repository,
    )

    assert output == "probe-complete"
    assert repository.link_validation_calls == [
        {
            "user_id": "user-1",
            "topic_id": "topic-atlas",
            "hop_id": "hop-atlas",
        }
    ]
    assert len(retriever.calls) == 1
    assert observed["conversation_retrieval"] is True
    assert observed["resolution"].path is LastQAPath.BROAD_RETRIEVAL_REQUIRED
    assert observed["resolution"].state is None
    assert observed["resolution"].is_authoritative_state is False


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
    retrieved_history = [
        {
            "hop_id": "hop-approved",
            "raw_user_query": "RAW_HISTORY_SENTINEL",
            "rewritten_user_query": "Earlier",
        }
    ]
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
        expected = (
            [{"hop_id": "hop-approved", "rewritten_user_query": "Earlier"}]
            if approved_variant == "approved"
            else []
        )
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
    approved = classifier.calls[0]["approved_conversation_context"]
    assert approved is not None
    assert approved.approved_conversation_history == last_qa_chat_history(state)
    assert approved._internal_selected_topic_candidates == ["topic-atlas"]
    assert approved._internal_selected_hop_candidates == ["hop-atlas"]
    assert len(store.saved) == 1
    assert current_chat_history() == []


def test_active_hitl_question_bypasses_platform_selector_completely() -> None:
    state = _last_qa_state()
    store = _Store(state)
    question = GeneratedQuestion(
        text="Which environment is required?",
        source=QuestionSource.HUMAN_SUPPORTING_QUESTION,
        purpose="resolve_required_context",
        confidence=1.0,
    )

    class QuestionRouter:
        def route(self, _context: Any, _repository: Any) -> BranchResult:
            return BranchResult(
                response_type=ResponseType.NORMAL,
                normal_response_text="I need one required detail before continuing.",
                human_supporting_questions=[question],
            )

    class ForbiddenPlatformSelector:
        calls = 0

        def select(self, _bundled: Any, _request: ChatRequest) -> dict[str, Any]:
            self.calls += 1
            raise AssertionError("PlatformSelector must be bypassed for active HITL")

    selector = ForbiddenPlatformSelector()
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
        classifier=_Classifier(Intent.GENERAL_RESPONSE),
        router=QuestionRouter(),
        bundler=ResponseBundler(),
        platform_selector=selector,
        chat_output=ChatOutput(),
    )

    response = pipeline.handle(
        ChatRequest(user_id="user-1", raw_query="Continue the deployment plan"),
        _Repository([]),
    )

    assert selector.calls == 0
    assert response.platform_payload["platform_selection"]["source"] == (
        "bypassed_active_branch_question"
    )
    assert response.platform_payload["delivery"] == {
        "channel": "none",
        "status": "deferred_by_active_question",
    }
    assert response.last_qa_state.supporting_questions == [question]
    assert "Supporting question: Which environment is required?" in (
        response.final_chat_text
    )


def test_supporting_answer_resumes_deferred_platform_request() -> None:
    question = GeneratedQuestion(
        text="Which recipient email address should receive the update?",
        source=QuestionSource.HUMAN_SUPPORTING_QUESTION,
        purpose="resolve_required_context",
        confidence=1.0,
    )
    state = LastQAState(
        last_user_query="Send an email about release readiness.",
        last_response=(
            "I need one required detail.\nSupporting question: "
            "Which recipient email address should receive the update?"
        ),
        response_type=ResponseType.NORMAL,
        supporting_questions=[question],
        linked_topic_id="topic-email",
        linked_hop_id="hop-email",
    )
    resolution = LastQAResolution(
        path=LastQAPath.LATEST_CONTEXT_INTERACTION,
        rewritten_query="alex@example.com",
        state=state,
        did_merge_query=False,
        skip_broad_retrieval=True,
        confidence=1.0,
        interaction_type=LastQAInteractionType.SUPPORTING_QUESTION_ANSWER,
        question_source=QuestionSource.HUMAN_SUPPORTING_QUESTION,
        linked_topic_id="topic-email",
        linked_hop_id="hop-email",
        matched_question=question.text,
        is_authoritative_state=True,
    )
    pipeline = AssistantPipeline(
        config=SimpleNamespace(
            context_filter=SimpleNamespace(
                conversation_retrieval_after_last_qa_enabled=True,
                conversation_retrieval_before_intent_enabled=True,
            )
        ),
        last_qa_store=_Store(state),
        query_rewriter=SimpleNamespace(rewrite=lambda query: query),
        last_qa_resolver=_Resolver(resolution),
        retriever=_Retriever([]),
        context_filter=_ContextFilter(None),
        classifier=_Classifier(Intent.GENERAL_RESPONSE),
        router=_Router(ResponseType.NORMAL),
        bundler=ResponseBundler(),
        platform_selector=PlatformSelector(llm=None),
        chat_output=ChatOutput(),
    )

    response = pipeline.handle(
        ChatRequest(user_id="user-1", raw_query="alex@example.com"),
        _Repository([]),
    )

    assert response.platform_payload["platform_selection"]["channel"] == "gmail"
    assert response.platform_payload["draft"]["recipients"] == [
        "alex@example.com"
    ]
    assert response.platform_payload["delivery"]["status"] == "needs_input"
    assert "notice" in response.platform_payload["delivery"]
    assert "question" not in response.platform_payload["delivery"]


def test_pipeline_assigns_replied_reminder_state_to_output_last_qa() -> None:
    reminder_state = build_reminder_state(
        {
            "reminder_id": "reminder-1",
            "notification_id": "notification-1",
            "subject": "Project review",
            "source_topic_id": "topic-reminder",
            "source_hop_id": "hop-reminder",
        }
    )
    state = LastQAState(
        last_user_query="Create a project review reminder.",
        last_response="The reminder was created.",
        response_type=ResponseType.REMINDER_ACTION,
        linked_topic_id="topic-reminder",
        linked_hop_id="hop-reminder",
        reminder_state=reminder_state,
        reminder_state_hash=reminder_state_hash(reminder_state),
    )
    resolution = LastQAResolution(
        path=LastQAPath.LATEST_CONTEXT_INTERACTION,
        rewritten_query="Please prepare a PDF agenda.",
        state=state,
        did_merge_query=False,
        skip_broad_retrieval=True,
        interaction_type=LastQAInteractionType.REMINDER_NOTIFICATION_REPLY,
        question_source=QuestionSource.NONE,
        linked_topic_id="topic-reminder",
        linked_hop_id="hop-reminder",
        reminder_id="reminder-1",
        notification_id="notification-1",
        source_topic_id="topic-reminder",
        source_hop_id="hop-reminder",
        is_authoritative_state=True,
    )
    store = _Store(state)
    pipeline = AssistantPipeline(
        config=SimpleNamespace(
            context_filter=SimpleNamespace(
                conversation_retrieval_after_last_qa_enabled=True,
                conversation_retrieval_before_intent_enabled=True,
            )
        ),
        last_qa_store=store,
        query_rewriter=SimpleNamespace(rewrite=lambda query: query),
        last_qa_resolver=_Resolver(resolution),
        retriever=_Retriever([]),
        context_filter=_ContextFilter(None),
        classifier=_Classifier(Intent.GENERAL_RESPONSE),
        router=_Router(ResponseType.NORMAL),
        bundler=ResponseBundler(),
        platform_selector=_PlatformSelector(),
        chat_output=ChatOutput(),
    )

    response = pipeline.handle(
        ChatRequest(user_id="user-1", raw_query="Please prepare a PDF agenda."),
        _Repository([]),
    )

    persisted = response.last_qa_state
    assert persisted.reminder_state is not None
    assert persisted.reminder_state["title"] == "Project review"
    assert persisted.reminder_state["reply_received"] is True
    assert persisted.reminder_state["reply_kind"] == "notification_purpose_reply"
    assert persisted.reminder_state_hash == reminder_state_hash(
        persisted.reminder_state
    )
    assert store.saved[-1][1] == persisted


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
        if isinstance(self.payload, StructuredFallbackPayload):
            return self.payload
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


def test_llm_last_qa_models_latest_relationship_without_active_question() -> None:
    llm = _LLMStub(
        payload={
            "relationship": "unrelated_or_uncertain",
            "confidence": 0.98,
        }
    )
    resolver = LLMLastQAResolver(llm=llm, config=LastQAConfig())

    resolution = resolver.resolve(
        ChatRequest(user_id="user-1", raw_query="Prepare a PDF agenda."),
        "Prepare a PDF agenda.",
        _last_qa_state(),
    )

    assert resolution.path is LastQAPath.BROAD_RETRIEVAL_REQUIRED
    assert resolution.skip_broad_retrieval is False
    assert len(llm.calls) == 1
    assert set(llm.calls[0]["schema"]["properties"]) == {
        "relationship",
        "confidence",
    }
    assert "latest_exchange" in llm.calls[0]["user_prompt"]
    assert "Atlas rolls out after QA signoff." in llm.calls[0]["user_prompt"]


def test_llm_last_qa_prompt_preserves_tail_of_long_latest_response() -> None:
    state = _last_qa_state()
    state.last_response = "A" * 700 + " FINAL_CONCLUSION_DEPLOY_FRIDAY"
    llm = _LLMStub(
        payload={
            "relationship": "unrelated_or_uncertain",
            "confidence": 0.98,
        }
    )

    LLMLastQAResolver(llm=llm, config=LastQAConfig()).resolve(
        ChatRequest(user_id="user-1", raw_query="Why that conclusion?"),
        "Why that conclusion?",
        state,
    )

    prompt = llm.calls[0]["user_prompt"]
    assert "...<truncated-middle>..." in prompt
    assert "FINAL_CONCLUSION_DEPLOY_FRIDAY" in prompt


def test_llm_last_qa_binds_minimal_question_index_to_exact_state_question() -> None:
    question = GeneratedQuestion(
        text="Which deployment environment should Atlas use?",
        source=QuestionSource.HUMAN_SUPPORTING_QUESTION,
        purpose="optional_context",
        confidence=1.0,
        expected_response_type=ExpectedResponseType.SELECTION_ANSWER,
    )
    state = _last_qa_state()
    state.supporting_questions = [question]
    llm = _LLMStub(
        payload={
            "relationship": "supporting_question_answer",
            "matched_question_index": 0,
            "confidence": 1.0,
        }
    )
    resolver = LLMLastQAResolver(llm=llm, config=LastQAConfig())

    resolution = resolver.resolve(
        ChatRequest(user_id="user-1", raw_query="Production"),
        "Production",
        state,
    )

    assert resolution.path is LastQAPath.LATEST_CONTEXT_INTERACTION
    assert resolution.skip_broad_retrieval is True
    assert resolution.interaction_type is LastQAInteractionType.SUPPORTING_QUESTION_ANSWER
    assert resolution.question_source is QuestionSource.HUMAN_SUPPORTING_QUESTION
    assert resolution.matched_question == question.text
    assert set(llm.calls[0]["schema"]["properties"]) == {
        "relationship",
        "matched_question_index",
        "confidence",
    }
    assert "latest_exchange" in llm.calls[0]["user_prompt"]
    assert "active_supporting_questions" in llm.calls[0]["user_prompt"]
    assert question.text in llm.calls[0]["user_prompt"]
    assert question.expected_response_type.value in llm.calls[0]["user_prompt"]


@pytest.mark.parametrize(
    "failure",
    ("structured_fallback", "exception"),
)
def test_llm_last_qa_model_failure_never_uses_lexical_skip_fallback(
    failure: str,
) -> None:
    question = GeneratedQuestion(
        text="Which environment should I use?",
        source=QuestionSource.HUMAN_SUPPORTING_QUESTION,
        purpose="optional_context",
        confidence=1.0,
        expected_response_type=ExpectedResponseType.SELECTION_ANSWER,
    )
    state = _last_qa_state()
    state.supporting_questions = [question]
    if failure == "structured_fallback":
        llm = _LLMStub(
            payload=StructuredFallbackPayload(
                {
                    "relationship": "unrelated_or_uncertain",
                    "matched_question_index": -1,
                    "confidence": 0.0,
                },
                task=LLMTask.LAST_QA,
                reason="synthetic terminal model failure",
            )
        )
    else:
        llm = _LLMStub(error=RuntimeError("synthetic terminal model failure"))

    resolution = LLMLastQAResolver(
        llm=llm,
        config=LastQAConfig(),
    ).resolve(
        ChatRequest(
            user_id="user-1",
            raw_query="Which environment should I avoid?",
        ),
        "Which environment should I avoid?",
        state,
    )

    assert resolution.path is LastQAPath.BROAD_RETRIEVAL_REQUIRED
    assert resolution.skip_broad_retrieval is False
    assert resolution.state is None
    assert resolution.interaction_type is None
    assert resolution.diagnostic_context["model_failure_policy"] == (
        "conservative_broad_retrieval"
    )


@pytest.mark.parametrize(
    (
        "relationship",
        "confidence",
        "question_index",
        "linked",
        "expected_latest",
    ),
    (
        ("normal_follow_up", 0.96, -1, True, True),
        ("normal_follow_up", 0.89, -1, True, False),
        ("normal_follow_up", 0.99, 0, True, False),
        ("normal_follow_up", 0.99, -1, False, False),
        ("normal_follow_up", 1.01, -1, True, False),
        ("normal_follow_up", float("inf"), -1, True, False),
        ("normal_follow_up", float("nan"), -1, True, False),
        ("unrelated_or_uncertain", 0.99, -1, True, False),
    ),
)
def test_llm_last_qa_calibrates_normal_latest_context_without_overmatching(
    relationship: str,
    confidence: float,
    question_index: int,
    linked: bool,
    expected_latest: bool,
) -> None:
    state = _last_qa_state()
    if not linked:
        state.linked_topic_id = None
        state.linked_hop_id = None
    payload: dict[str, Any] = {
        "relationship": relationship,
        "confidence": confidence,
    }
    if question_index != -1:
        state.supporting_questions = [
            GeneratedQuestion(
                text="Which explanation mode should I use?",
                source=QuestionSource.HUMAN_SUPPORTING_QUESTION,
                purpose="optional_context",
                confidence=1.0,
                expected_response_type=ExpectedResponseType.SELECTION_ANSWER,
            )
        ]
        payload["matched_question_index"] = question_index
    llm = _LLMStub(
        payload=payload
    )

    resolution = LLMLastQAResolver(
        llm=llm,
        config=LastQAConfig(),
    ).resolve(
        ChatRequest(
            user_id="user-1",
            raw_query="Could you explain why that signoff is required?",
        ),
        "Could you explain why that signoff is required?",
        state,
    )

    assert resolution.path is (
        LastQAPath.LATEST_CONTEXT_INTERACTION
        if expected_latest
        else LastQAPath.BROAD_RETRIEVAL_REQUIRED
    )
    assert resolution.skip_broad_retrieval is expected_latest
    assert resolution.state is (state if expected_latest else None)
    assert resolution.interaction_type is (
        LastQAInteractionType.NORMAL_FOLLOW_UP if expected_latest else None
    )
    assert resolution.is_authoritative_state is expected_latest


@pytest.mark.parametrize(
    "response_type",
    (
        ResponseType.NORMAL,
        ResponseType.KNOWLEDGE_ACTION,
        ResponseType.REMINDER_ACTION,
        ResponseType.REMINDER_REPLY,
        ResponseType.ERROR,
        ResponseType.SAFE_NOOP,
    ),
)
def test_llm_last_qa_recognizes_latest_relationship_for_every_eligible_state(
    response_type: ResponseType,
) -> None:
    state = _last_qa_state()
    state.response_type = response_type
    llm = _LLMStub(
        payload={
            "relationship": "normal_follow_up",
            "confidence": 0.96,
        }
    )

    resolution = LLMLastQAResolver(
        llm=llm,
        config=LastQAConfig(),
    ).resolve(
        ChatRequest(user_id="user-1", raw_query="Why did that happen?"),
        "Why did that happen?",
        state,
    )

    assert resolution.path is LastQAPath.LATEST_CONTEXT_INTERACTION
    assert resolution.interaction_type is LastQAInteractionType.NORMAL_FOLLOW_UP
    assert resolution.skip_broad_retrieval is True


@pytest.mark.parametrize(
    ("model_action", "expected_action"),
    (
        ("send", OutboundFollowUpAction.SEND),
        ("revise", OutboundFollowUpAction.REVISE),
        ("revise_and_send", OutboundFollowUpAction.REVISE_AND_SEND),
    ),
)
def test_llm_last_qa_binds_semantic_action_to_active_outbound_envelope(
    model_action: str,
    expected_action: OutboundFollowUpAction,
) -> None:
    outbound = OutboundMessageState(
        channel="gmail",
        status="draft_ready",
        recipients=("alice@example.com", "bob@example.com"),
        subject="Schedule notice",
        body="I will be away tomorrow.",
        artifact_ids=("artifact-1",),
        attachment_filenames=("details.pdf",),
        source_topic_id="topic-mail",
        source_hop_id="hop-mail",
    )
    state = LastQAState(
        last_user_query="Prepare a notice.",
        last_response="The draft is ready.",
        response_type=ResponseType.NORMAL,
        linked_topic_id="topic-mail",
        linked_hop_id="hop-mail",
        outbound_state=outbound,
    )
    llm = _LLMStub(
        payload={"outbound_action": model_action, "confidence": 0.99}
    )

    resolution = LLMLastQAResolver(
        llm=llm,
        config=LastQAConfig(),
    ).resolve(
        ChatRequest(user_id="user-1", raw_query="Continue with that message."),
        "Continue with that message.",
        state,
    )

    assert resolution.path is LastQAPath.LATEST_CONTEXT_INTERACTION
    assert resolution.skip_broad_retrieval is True
    assert resolution.state is state
    assert resolution.interaction_type is LastQAInteractionType.OUTBOUND_MESSAGE_ACTION
    assert resolution.outbound_action is expected_action
    assert resolution.linked_hop_id == "hop-mail"
    assert set(llm.calls[0]["schema"]["properties"]) == {
        "outbound_action",
        "confidence",
    }
    assert "active_outbound_state" in llm.calls[0]["user_prompt"]
    assert "details.pdf" in llm.calls[0]["user_prompt"]


def test_llm_last_qa_outbound_none_continues_to_broad_retrieval_without_question() -> None:
    state = LastQAState(
        last_user_query="Prepare a notice.",
        last_response="The draft is ready.",
        response_type=ResponseType.NORMAL,
        outbound_state=OutboundMessageState(
            channel="gmail",
            status="draft_ready",
            recipients=("alice@example.com",),
            subject="Schedule notice",
            body="I will be away tomorrow.",
        ),
    )
    llm = _LLMStub(payload={"outbound_action": "none", "confidence": 0.99})

    resolution = LLMLastQAResolver(
        llm=llm,
        config=LastQAConfig(),
    ).resolve(
        ChatRequest(user_id="user-1", raw_query="Explain database indexes."),
        "Explain database indexes.",
        state,
    )

    assert resolution.path is LastQAPath.BROAD_RETRIEVAL_REQUIRED
    assert resolution.skip_broad_retrieval is False
    assert resolution.state is None
    assert resolution.interaction_type is None


def test_authoritative_outbound_action_skips_optional_supporting_question_llm() -> None:
    llm = _LLMStub(error=AssertionError("optional HITL LLM must not run"))
    context = PipelineContext(
        request=ChatRequest(user_id="user-1", raw_query="Proceed with the draft."),
        rewritten_query="Proceed with the draft.",
        last_qa_state=None,
        conversation_results=[],
        intent=Intent.GENERAL_RESPONSE,
        last_qa_trace={
            "interaction_type": LastQAInteractionType.OUTBOUND_MESSAGE_ACTION.value
        },
    )

    decision = LLMGeneralHITLStrategy(
        llm=llm,
        prompt_registry=DEFAULT_PROMPT_REGISTRY,
        config=GeneralPurposeConfig(hitl_supporting_question_enabled=True),
    ).evaluate(
        context=context,
        response_text="The outbound action is handled by the platform stage.",
        approved_conversation_history=[],
        merged_supporting_detail="",
    )

    assert decision.should_ask is False
    assert decision.confidence == 1.0
    assert llm.calls == []
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
    reminder_state = {
        "reminder_id": "reminder-1",
        "notification_id": "notification-1",
        "source_topic_id": "topic-reminder",
        "source_hop_id": "hop-reminder",
        "notification_created": True,
        "has_been_notified": True,
    }
    state_digest = reminder_state_hash(reminder_state)
    state = LastQAState(
        last_user_query="Medication reminder",
        last_response="It is time for your medication.",
        response_type=ResponseType.REMINDER_REPLY,
        linked_topic_id="topic-reminder",
        linked_hop_id="hop-reminder",
        reminder_state=reminder_state,
        reminder_state_hash=state_digest,
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
            "reminder_state": reminder_state,
            "reminder_state_hash": state_digest,
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


def test_deterministic_reminder_reply_rejects_tampered_state_hash() -> None:
    reminder_state = {
        "reminder_id": "reminder-1",
        "notification_id": "notification-1",
        "source_topic_id": "topic-reminder",
        "source_hop_id": "hop-reminder",
        "notification_created": True,
        "has_been_notified": True,
    }
    state = LastQAState(
        last_user_query="Medication reminder",
        last_response="It is time for your medication.",
        response_type=ResponseType.REMINDER_REPLY,
        linked_topic_id="topic-reminder",
        linked_hop_id="hop-reminder",
        reminder_state={**reminder_state, "title": "tampered"},
        reminder_state_hash=reminder_state_hash(reminder_state),
    )
    request = ChatRequest(
        user_id="user-1",
        raw_query="Done",
        reminder_id="reminder-1",
        notification_id="notification-1",
        metadata={
            "reminder_reply_context": True,
            "reminder_id": "reminder-1",
            "notification_id": "notification-1",
            "source_topic_id": "topic-reminder",
            "source_hop_id": "hop-reminder",
            "reminder_state": reminder_state,
            "reminder_state_hash": reminder_state_hash(reminder_state),
        },
    )

    resolution = LastQAResolver().resolve(request, "Done", state)

    assert resolution.path is LastQAPath.BROAD_RETRIEVAL_REQUIRED
    assert resolution.is_authoritative_state is False


@pytest.mark.parametrize(
    ("intent_value", "query", "expected"),
    (
        ("general_response", "Explain the Atlas policy.", Intent.GENERAL_RESPONSE),
        (
            "knowledge_facts",
            "Remember that Atlas uses PostgreSQL.",
            Intent.KNOWLEDGE_FACTS,
        ),
        (
            "reminder",
            "Remind me tomorrow at 9 AM to submit expenses.",
            Intent.REMINDER,
        ),
        (
            "clarification",
            "The concise-report preference.",
            Intent.CLARIFICATION,
        ),
    ),
)
def test_production_intent_classifier_operation_routes(
    intent_value: str,
    query: str,
    expected: Intent,
) -> None:
    llm = _LLMStub(
        payload={
            "intent": intent_value,
            "confidence": 1.0,
        }
    )
    classifier = OllamaIntentClassifier(llm, min_confidence=0.7)

    result = classifier.classify(
        ChatRequest(user_id="user-1", raw_query=query),
        query,
        last_qa_resolution=_pending_clarification_resolution(),
        approved_conversation_context=None,
    )

    assert result is expected
    assert len(llm.calls) == 1
    schema = llm.calls[0]["schema"]
    assert set(schema["properties"]) == {"intent", "confidence"}
    assert schema["properties"]["intent"]["enum"] == [
        "general_response",
        "knowledge_facts",
        "reminder",
        "clarification",
    ]


@pytest.mark.parametrize(
    ("payload", "error", "resolution"),
    (
        (None, RuntimeError("model unavailable"), None),
        ({"intent": "reminder"}, None, None),
        (
            {
                "intent": "reminder",
                "confidence": 0.69,
            },
            None,
            None,
        ),
        (
            {
                "intent": "clarification",
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


@pytest.mark.parametrize(
    ("intent", "query"),
    (
        (Intent.CLARIFICATION, "The requested target."),
        (Intent.GENERAL_RESPONSE, "Explain the requested target."),
        (Intent.KNOWLEDGE_FACTS, "Remember that Atlas uses PostgreSQL."),
        (Intent.REMINDER, "Set a reminder tomorrow at 9 AM."),
    ),
)
def test_production_intent_classifier_valid_metadata_override_bypasses_llm(
    intent: Intent,
    query: str,
) -> None:
    llm = _LLMStub(error=AssertionError("LLM must not run"))
    classifier = OllamaIntentClassifier(llm, min_confidence=0.7)

    result = classifier.classify(
        ChatRequest(
            user_id="user-1",
            raw_query=query,
            metadata={"intent": intent.value},
        ),
        query,
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
