from __future__ import annotations

import json
from types import MethodType, SimpleNamespace

from assistant_rag.chat_history import (
    CHAT_HISTORY_PROMPT_RULE,
    canonical_chat_history_scope,
    current_chat_history,
    inject_chat_history,
    last_qa_chat_history,
    select_chat_history,
)
from assistant_rag.contracts import (
    ApprovedConversationContext,
    ChatRequest,
    ExpectedResponseType,
    LastQAInteractionType,
    LastQAPath,
    LastQAResolution,
    LastQAState,
    QuestionSource,
    ResponseType,
    RetrievalResult,
)
from assistant_rag.pipeline import AssistantPipeline
from assistant_rag.prompts import DEFAULT_PROMPT_REGISTRY, PromptContext


def _last_qa() -> LastQAState:
    return LastQAState(
        last_user_query="What is the Atlas rollout plan?",
        last_response="Atlas rolls out after QA signoff.",
        response_type=ResponseType.NORMAL,
        linked_topic_id="topic-atlas",
        linked_hop_id="hop-atlas",
        expected_response_type=ExpectedResponseType.FREE_TEXT_ANSWER,
    )


def _approved(history: list[dict]) -> ApprovedConversationContext:
    return ApprovedConversationContext(
        approved_conversation_history=history,
        human_supporting_questions=[],
        reminder_supporting_questions=[],
        clarification_question_context=None,
        extracted_expected_response_types=[],
        conversation_retrieval_ran=True,
        conversation_context_status="approved" if history else "empty",
        approved_conversation_count=len(history),
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


def test_retrieval_history_is_authoritative_and_never_falls_back_to_last_qa() -> None:
    retrieved = [
        {"hop_id": f"hop-{index}", "raw_user_query": f"question-{index}"}
        for index in range(12)
    ]

    assert select_chat_history(
        conversation_retrieval=True,
        approved_conversation_context=_approved(retrieved),
        last_qa_state=_last_qa(),
    ) == retrieved
    assert select_chat_history(
        conversation_retrieval=True,
        approved_conversation_context=_approved([]),
        last_qa_state=_last_qa(),
    ) == []
    assert select_chat_history(
        conversation_retrieval=True,
        approved_conversation_context=None,
        last_qa_state=_last_qa(),
    ) == []


def test_last_qa_is_canonical_history_only_when_retrieval_did_not_run() -> None:
    state = _last_qa()

    history = select_chat_history(
        conversation_retrieval=False,
        approved_conversation_context=_approved([{"hop_id": "must-not-win"}]),
        last_qa_state=state,
    )

    assert history == last_qa_chat_history(state)
    assert history[0]["source"] == "last_qa"
    assert history[0]["raw_user_query"] == state.last_user_query
    assert history[0]["raw_response"] == state.last_response
    assert history[0]["topic_id"] == state.linked_topic_id
    assert history[0]["hop_id"] == state.linked_hop_id


def test_every_registry_prompt_preserves_complete_or_explicitly_empty_history() -> None:
    history = [{"hop_id": f"hop-{index}", "text": "x" * 20} for index in range(12)]
    stages = (
        "intent_classifier",
        "general_sub_branch_detector",
        "risky_action_validation",
        "knowledge_retrieval_validation",
        "question_generation",
        "answer_generation",
        "content_tool_answer_generation",
        "gmail_policy",
    )

    with canonical_chat_history_scope(history):
        for stage in stages:
            payload = PromptContext(stage=stage).stage_payload()
            assert payload["chat_history"] == history
            assert len(payload["chat_history"]) == 12

    with canonical_chat_history_scope([]):
        payload = PromptContext(stage="intent_classifier").stage_payload()
        assert "chat_history" in payload
        assert payload["chat_history"] == []

    assert current_chat_history() == []
    for template_name in DEFAULT_PROMPT_REGISTRY.templates:
        if template_name in {"query_rewrite", "last_qa", "clarification_merge"}:
            continue
        assert CHAT_HISTORY_PROMPT_RULE in DEFAULT_PROMPT_REGISTRY.system(template_name)


def test_raw_json_prompt_helper_uses_the_same_canonical_history() -> None:
    history = [{"hop_id": "hop-1"}, {"hop_id": "hop-2"}]

    with canonical_chat_history_scope(history):
        payload = inject_chat_history({"operation": "modify"})

    assert payload == {"operation": "modify", "chat_history": history}
    assert json.loads(json.dumps(payload))["chat_history"] == history


def _pipeline_for_history_test(*, state: LastQAState, resolution: LastQAResolution, retrieved: list[RetrievalResult], approved: ApprovedConversationContext | None):
    class Store:
        def get(self, _user_id: str) -> LastQAState:
            return state

    class Resolver:
        def resolve(self, *_args):
            return resolution

    class Retriever:
        def retrieve_conversation(self, **_kwargs):
            return retrieved

    class Filter:
        def filter_conversation_only(self, **_kwargs):
            return approved

    pipeline = AssistantPipeline(
        config=SimpleNamespace(
            context_filter=SimpleNamespace(
                conversation_retrieval_after_last_qa_enabled=True,
                conversation_retrieval_before_intent_enabled=True,
            )
        ),
        last_qa_store=Store(),
        query_rewriter=SimpleNamespace(rewrite=lambda query: query),
        last_qa_resolver=Resolver(),
        retriever=Retriever(),
        context_filter=Filter(),
        classifier=SimpleNamespace(),
        router=SimpleNamespace(),
        bundler=SimpleNamespace(),
        platform_selector=SimpleNamespace(),
        chat_output=SimpleNamespace(),
    )
    return pipeline


def test_pipeline_activates_last_qa_history_before_intent_stage_when_retrieval_is_false() -> None:
    state = _last_qa()
    pipeline = _pipeline_for_history_test(
        state=state,
        resolution=_resolution(state=state, skip=True),
        retrieved=[],
        approved=None,
    )
    observed: dict[str, object] = {}

    def stop_after_scope(self, **kwargs):
        observed["active"] = current_chat_history()
        observed.update(kwargs)
        return "done"

    pipeline._handle_with_chat_history = MethodType(stop_after_scope, pipeline)
    result = pipeline.handle(ChatRequest(user_id="u", raw_query="Continue"), SimpleNamespace())

    assert result == "done"
    assert observed["conversation_retrieval"] is False
    assert observed["chat_history_source"] == "last_qa"
    assert observed["active"] == last_qa_chat_history(state)
    assert current_chat_history() == []


def test_pipeline_activates_validated_retrieval_history_even_when_it_is_empty() -> None:
    state = _last_qa()
    result = RetrievalResult(
        entity_type="conversation_hop",
        entity_id="hop-rejected",
        source_store_evidence={},
        rerank_score=1.0,
        confidence=1.0,
        validation_status="candidate",
        payload={"hop_id": "hop-rejected"},
    )
    pipeline = _pipeline_for_history_test(
        state=state,
        resolution=_resolution(state=state, skip=False),
        retrieved=[result],
        approved=_approved([]),
    )
    observed: dict[str, object] = {}

    def stop_after_scope(self, **kwargs):
        observed["active"] = current_chat_history()
        observed.update(kwargs)
        return "done"

    pipeline._handle_with_chat_history = MethodType(stop_after_scope, pipeline)
    repository = SimpleNamespace(
        hydrate_conversation_retrieval_results=lambda **_kwargs: [result]
    )
    output = pipeline.handle(ChatRequest(user_id="u", raw_query="New topic"), repository)

    assert output == "done"
    assert observed["conversation_retrieval"] is True
    assert observed["chat_history_source"] == "conversation_retrieval"
    assert observed["active"] == []
    assert observed["chat_history"] == []
    assert current_chat_history() == []
