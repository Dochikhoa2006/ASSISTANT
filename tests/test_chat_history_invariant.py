from __future__ import annotations

import json
from types import MethodType, SimpleNamespace

import pytest

from assistant_rag.chat_history import (
    CHAT_HISTORY_PROMPT_RULE,
    canonical_chat_history_scope,
    current_chat_history,
    inject_chat_history,
    last_qa_chat_history,
    select_chat_history,
)
from assistant_rag.conversation_embedding import (
    conversation_hop_semantic_payload,
    serialize_conversation_hop,
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
from assistant_rag.database import SQLiteRepository
from assistant_rag.pipeline import AssistantPipeline
from assistant_rag.postgres_repository import PostgresRepository
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
        {
            "hop_id": f"hop-{index}",
            "raw_user_query": f"RAW_QUESTION_{index}",
            "rewritten_user_query": f"REWRITTEN_QUESTION_{index}",
            "nested": {
                "source_raw_user_query": f"NESTED_RAW_QUESTION_{index}",
                "safe": f"safe-{index}",
            },
        }
        for index in range(12)
    ]
    rewritten_only = [
        {
            "hop_id": f"hop-{index}",
            "rewritten_user_query": f"REWRITTEN_QUESTION_{index}",
            "nested": {"safe": f"safe-{index}"},
        }
        for index in range(12)
    ]

    assert select_chat_history(
        conversation_retrieval=True,
        approved_conversation_context=_approved(retrieved),
        last_qa_state=_last_qa(),
    ) == rewritten_only
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
    assert history[0]["rewritten_user_query"] == state.last_user_query
    assert "raw_user_query" not in history[0]
    assert history[0]["raw_response"] == state.last_response
    assert history[0]["topic_id"] == state.linked_topic_id
    assert history[0]["hop_id"] == state.linked_hop_id


def test_every_registry_prompt_preserves_complete_or_explicitly_empty_history() -> None:
    history = [{"hop_id": f"hop-{index}", "text": "x" * 20} for index in range(12)]
    stages = (
        "intent_classifier",
        "general_sub_branch_detector",
        "knowledge_action_extraction",
        "knowledge_content_finalization",
        "reminder_action_extraction",
        "reminder_content_finalization",
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
        if template_name in {
            "query_rewrite",
            "last_qa",
            "clarification_merge",
            "knowledge_action_validation",
            "reminder_action_validation",
        }:
            continue
        assert CHAT_HISTORY_PROMPT_RULE in DEFAULT_PROMPT_REGISTRY.system(template_name)


def test_knowledge_validation_prompt_contains_only_extraction_and_retrieval() -> None:
    history = [{"hop_id": "forbidden-hop", "text": "forbidden history"}]
    first_model_response = {
        "action": "add",
        "text_content": "Atlas retention is 30 days",
        "original_text": "",
        "replacement_text": "",
        "confidence": 0.99,
        "missing_fields": [],
        "reason_summary": "Extracted one action.",
    }
    retrieval = [{"candidate_key": "chunk-1", "text_excerpt": "stored fact"}]

    with canonical_chat_history_scope(history):
        context = PromptContext(
            stage="knowledge_action_validation",
            user_id="forbidden-user",
            rewritten_query="forbidden rewritten query",
            metadata={"forbidden": "metadata"},
            platform_context={"forbidden": "platform"},
            chat_history=history,
            extra={
                "first_model_response": first_model_response,
                "knowledge_retrieval": retrieval,
                "forbidden_extra": "must not survive",
            },
        )
        payload = context.stage_payload()
        safe_payload = context.safe_payload()

    assert payload == {
        "first_model_response": first_model_response,
        "knowledge_retrieval": retrieval,
    }
    assert safe_payload == payload
    system_prompt = DEFAULT_PROMPT_REGISTRY.system(
        "knowledge_action_validation"
    )
    assert CHAT_HISTORY_PROMPT_RULE not in system_prompt
    assert "chat_history" not in system_prompt
    assert "raw query" not in system_prompt.casefold()
    assert "rewritten query" not in system_prompt.casefold()


def test_reminder_validation_prompt_contains_only_extraction_and_retrieval() -> None:
    history = [{"hop_id": "forbidden-hop", "text": "forbidden history"}]
    first_model_response = {
        "action": "delete",
        "toggle_direction": "",
        "retrieval_text": "Payroll",
        "changed_fields": [],
        "subject": "",
        "reminder_summary": "",
        "raw_reminder": "",
        "notification_time": "",
        "event_time": "",
        "user_timezone": "",
        "original_time_text": "",
        "recurrence_rule": "",
        "recurrence_timezone": "",
        "supporting_question": "",
        "supporting_response": "",
        "time_semantics": "unchanged",
        "confidence": 0.99,
        "missing_fields": [],
        "reason_summary": "Extracted one reminder action.",
    }
    retrieval = [{"candidate_key": "reminder-1", "subject": "Payroll"}]

    with canonical_chat_history_scope(history):
        context = PromptContext(
            stage="reminder_action_validation",
            user_id="forbidden-user",
            rewritten_query="forbidden rewritten query",
            metadata={"forbidden": "metadata"},
            platform_context={"forbidden": "platform"},
            chat_history=history,
            extra={
                "first_model_response": first_model_response,
                "reminder_retrieval": retrieval,
                "forbidden_extra": "must not survive",
            },
        )
        payload = context.stage_payload()
        safe_payload = context.safe_payload()

    assert payload == {
        "first_model_response": first_model_response,
        "reminder_retrieval": retrieval,
    }
    assert safe_payload == payload
    system_prompt = DEFAULT_PROMPT_REGISTRY.system(
        "reminder_action_validation"
    )
    assert CHAT_HISTORY_PROMPT_RULE not in system_prompt
    assert "chat_history" not in system_prompt
    assert "raw query" not in system_prompt.casefold()
    assert "rewritten query" not in system_prompt.casefold()


def test_raw_json_prompt_helper_uses_the_same_canonical_history() -> None:
    history = [{"hop_id": "hop-1"}, {"hop_id": "hop-2"}]

    with canonical_chat_history_scope(history):
        payload = inject_chat_history({"operation": "modify"})

    assert payload == {"operation": "modify", "chat_history": history}
    assert json.loads(json.dumps(payload))["chat_history"] == history


def test_prompt_context_allows_raw_query_only_for_query_rewrite() -> None:
    raw_sentinel = "RAW_QUERY_SENTINEL"
    rewritten_sentinel = "REWRITTEN_QUERY_SENTINEL"
    rewrite_context = PromptContext(
        stage="query_rewrite",
        raw_query=raw_sentinel,
        metadata={"raw_user_query": "HISTORICAL_RAW_SENTINEL"},
        chat_history=[
            {
                "raw_user_query": "HISTORICAL_RAW_SENTINEL",
                "rewritten_user_query": "HISTORICAL_REWRITTEN_SENTINEL",
            }
        ],
    )

    for payload in (
        rewrite_context.safe_payload(),
        rewrite_context.stage_payload(),
    ):
        assert payload["raw_query"] == raw_sentinel
        serialized = json.dumps(payload, sort_keys=True)
        assert "HISTORICAL_RAW_SENTINEL" not in serialized
        assert "HISTORICAL_REWRITTEN_SENTINEL" in serialized

    downstream_context = PromptContext(
        stage="intent_classifier",
        raw_query=raw_sentinel,
        rewritten_query=rewritten_sentinel,
    )
    with pytest.raises(ValueError, match="query_rewrite"):
        downstream_context.safe_payload()
    with pytest.raises(ValueError, match="query_rewrite"):
        downstream_context.stage_payload()


def test_prompt_context_recursively_strips_nested_raw_query_fields() -> None:
    raw_sentinel = "RAW_QUERY_SENTINEL"
    summarized_raw_sentinel = "SUMMARIZED_RAW_QUERY_SENTINEL"
    rewritten_sentinel = "REWRITTEN_QUERY_SENTINEL"
    context = PromptContext(
        stage="answer_generation",
        rewritten_query=rewritten_sentinel,
        metadata={
            "raw_query": raw_sentinel,
            "nested": {
                "RAW_USER_QUERY": raw_sentinel,
                "safe": "metadata-safe",
            },
        },
        platform_context={
            "nested": {
                "source_raw_user_query": raw_sentinel,
                "safe": "platform-safe",
            }
        },
        extra={
            "summarized_user_query": summarized_raw_sentinel,
            "nested": {"safe": "extra-safe"},
        },
        chat_history=[
            {
                "raw_user_query": raw_sentinel,
                "rewritten_user_query": rewritten_sentinel,
                "nested": {
                    "source_raw_user_query": raw_sentinel,
                    "safe": "history-safe",
                },
            }
        ],
    )

    for payload in (context.safe_payload(), context.stage_payload()):
        serialized = json.dumps(payload, sort_keys=True)
        assert raw_sentinel not in serialized
        assert summarized_raw_sentinel not in serialized
        assert rewritten_sentinel in serialized
        assert payload["metadata"] == {"nested": {"safe": "metadata-safe"}}
        assert payload["platform_context"] == {
            "nested": {"safe": "platform-safe"}
        }
        assert payload["extra"] == {"nested": {"safe": "extra-safe"}}
        assert payload["chat_history"] == [
            {
                "rewritten_user_query": rewritten_sentinel,
                "nested": {"safe": "history-safe"},
            }
        ]


def test_conversation_embedding_uses_only_rewritten_query_semantics() -> None:
    raw_sentinel = "RAW_QUERY_SENTINEL"
    summarized_raw_sentinel = "SUMMARIZED_RAW_QUERY_SENTINEL"
    rewritten_sentinel = "REWRITTEN_QUERY_SENTINEL"
    row = {
        "hop_id": "hop-1",
        "raw_user_query": raw_sentinel,
        "summarized_user_query": summarized_raw_sentinel,
        "rewritten_user_query": rewritten_sentinel,
        "nested": {
            "raw_query": raw_sentinel,
            "SOURCE_RAW_USER_QUERY": raw_sentinel,
            "safe": "semantic-safe",
        },
    }

    semantic_payload = conversation_hop_semantic_payload(row)
    serialized = serialize_conversation_hop(row)
    serialized_payload = json.loads(serialized)["conversation_hop"]

    assert semantic_payload == {
        "hop_id": "hop-1",
        "rewritten_user_query": rewritten_sentinel,
        "nested": {"safe": "semantic-safe"},
    }
    assert serialized_payload == semantic_payload
    assert rewritten_sentinel in serialized
    assert raw_sentinel not in serialized
    assert summarized_raw_sentinel not in serialized


@pytest.mark.parametrize("repository_kind", ("sqlite", "sqlalchemy"))
def test_sql_hydration_quarantines_raw_query_audit_fields(
    repository_kind: str,
) -> None:
    repository = (
        SQLiteRepository.in_memory()
        if repository_kind == "sqlite"
        else PostgresRepository.create("sqlite+pysqlite:///:memory:")
    )
    repository.initialize_schema()
    raw_sentinel = "RAW_SQL_AUDIT_SENTINEL"
    vector_raw_sentinel = "RAW_VECTOR_SENTINEL"
    rewritten_sentinel = "REWRITTEN_SQL_SENTINEL"
    try:
        with repository.transaction() as cursor:
            topic_id = repository.ensure_topic(
                cursor,
                user_id="query-authority-user",
                title="Query authority",
            )
            hop = repository.append_conversation_hop(
                cursor,
                topic_id=topic_id,
                user_id="query-authority-user",
                intent="general_response",
                raw_user_query=raw_sentinel,
                rewritten_user_query=rewritten_sentinel,
                raw_response="Stored response",
                response_type="normal",
            )

        hydrated = repository.hydrate_conversation_retrieval_results(
            user_id="query-authority-user",
            results=[
                RetrievalResult(
                    entity_type="conversation_hop",
                    entity_id=hop.hop_id,
                    source_store_evidence={"vector": True},
                    rerank_score=0.9,
                    confidence=0.9,
                    validation_status="candidate",
                    payload={
                        "raw_user_query": vector_raw_sentinel,
                        "rewritten_user_query": "UNTRUSTED_VECTOR_REWRITE",
                    },
                )
            ],
        )

        assert len(hydrated) == 1
        payload = hydrated[0].payload
        serialized = json.dumps(payload, sort_keys=True, default=str)
        assert payload["rewritten_user_query"] == rewritten_sentinel
        assert "raw_user_query" not in payload
        assert raw_sentinel not in serialized
        assert vector_raw_sentinel not in serialized
        assert rewritten_sentinel in serialized
    finally:
        close = getattr(repository, "close", None)
        if callable(close):
            close()
        else:
            repository.connection.close()


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
