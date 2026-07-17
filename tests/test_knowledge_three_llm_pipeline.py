from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from assistant_rag.branches import KnowledgeFactsBranch
from assistant_rag.chat_history import canonical_chat_history_scope
from assistant_rag.contracts import (
    ChatRequest,
    Intent,
    PipelineContext,
    ResponseType,
    RetrievalResult,
)
from assistant_rag.database import SQLiteRepository
from assistant_rag.knowledge_mutation import (
    KnowledgeContentFinalizationStrategy,
    KnowledgeMutationPipeline,
    LLMKnowledgeActionDetector,
)
from assistant_rag.llm import LLMTask
from assistant_rag.production_factory import build_assistant_config
from assistant_rag.prompts import DEFAULT_PROMPT_REGISTRY
from assistant_rag.retrieval_validation import KnowledgeRetrievalValidationStrategy
from assistant_rag.settings import ProductionSettings


USER_ID = "knowledge-three-llm-user"


class ScriptedLLM:
    def __init__(self, responses: list[dict[str, Any] | Exception]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def generate_json(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(dict(kwargs))
        if not self.responses:
            raise AssertionError("unexpected extra knowledge LLM call")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def chat(self, **_kwargs: Any) -> str:
        raise AssertionError("knowledge mutation stages must use strict JSON")


class RecordingKnowledgeRetriever:
    def __init__(self, results: list[RetrievalResult]) -> None:
        self.results = list(results)
        self.calls: list[dict[str, Any]] = []

    def retrieve_knowledge(self, **kwargs: Any) -> list[RetrievalResult]:
        self.calls.append(dict(kwargs))
        return list(self.results)


def _repository() -> SQLiteRepository:
    repository = SQLiteRepository.in_memory()
    repository.initialize_schema()
    return repository


def _seed_knowledge(repository: SQLiteRepository, text: str) -> str:
    with repository.transaction() as cursor:
        _, chunk_id, _ = repository.add_knowledge_chunk(
            cursor,
            user_id=USER_ID,
            title="Knowledge",
            text=text,
        )
    return chunk_id


def _retrieval_result(chunk_id: str, text: str, score: float = 0.01) -> RetrievalResult:
    return RetrievalResult(
        entity_type="knowledge_chunk",
        entity_id=chunk_id,
        source_store_evidence={"test": True},
        rerank_score=score,
        confidence=score,
        validation_status="candidate",
        payload={
            "user_id": USER_ID,
            "text": text,
            "is_deleted": False,
        },
    )


def _context(raw_query: str, history: list[dict[str, Any]]) -> PipelineContext:
    return PipelineContext(
        request=ChatRequest(user_id=USER_ID, raw_query=raw_query),
        rewritten_query=raw_query,
        last_qa_state=None,
        conversation_results=[],
        intent=Intent.KNOWLEDGE_FACTS,
        chat_history=history,
    )


def _branch(
    llm: ScriptedLLM,
    retriever: RecordingKnowledgeRetriever,
    *,
    config: Any | None = None,
) -> KnowledgeFactsBranch:
    config = config or build_assistant_config(ProductionSettings())
    validator = KnowledgeRetrievalValidationStrategy(
        config=config.retrieval_validation,
        llm=llm,
        prompts=DEFAULT_PROMPT_REGISTRY,
    )
    finalizer = KnowledgeContentFinalizationStrategy(
        llm=llm,
        prompts=DEFAULT_PROMPT_REGISTRY,
        min_confidence=(
            config.retrieval_validation.knowledge_llm_validation_min_confidence
        ),
    )
    pipeline = KnowledgeMutationPipeline(
        retriever=retriever,  # type: ignore[arg-type]
        config=config,
        validator=validator,
        finalizer=finalizer,
    )
    return KnowledgeFactsBranch(
        config=config,
        action_detector=LLMKnowledgeActionDetector(
            llm=llm,
            prompts=DEFAULT_PROMPT_REGISTRY,
            min_confidence=0.76,
        ),
        prompt_registry=DEFAULT_PROMPT_REGISTRY,
        knowledge_mutation_pipeline=pipeline,
    )


def _detector(llm: ScriptedLLM) -> LLMKnowledgeActionDetector:
    return LLMKnowledgeActionDetector(
        llm=llm,
        prompts=DEFAULT_PROMPT_REGISTRY,
        min_confidence=0.76,
    )


def _prompt_payload(call: dict[str, Any]) -> dict[str, Any]:
    prefix = "Runtime context:\n"
    prompt = str(call["user_prompt"])
    assert prompt.startswith(prefix)
    return json.loads(prompt[len(prefix) :])


def _extraction(
    *,
    action: str,
    text_content: str = "",
    original_text: str = "",
    replacement_text: str = "",
) -> dict[str, Any]:
    return {
        "action": action,
        "text_content": text_content,
        "original_text": original_text,
        "replacement_text": replacement_text,
        "confidence": 0.99,
    }


def _validation(
    *,
    operation: str,
    decision: str,
    selected: list[str],
    assessments: list[dict[str, Any]],
    clarification_question: str = "",
    confidence: float = 0.99,
    reason: str = "Knowledge action was validated.",
) -> dict[str, Any]:
    return {
        "decision": decision,
        "selected_candidate_keys": selected,
        "confidence": confidence,
        "clarification_question": clarification_question,
        "candidate_assessments": assessments,
    }


def _assessment(
    chunk_id: str,
    *,
    matches: bool,
    matched_text: str = "",
    compatible: bool = True,
) -> dict[str, Any]:
    return {
        "candidate_key": chunk_id,
        "confidence": 0.99,
        "matched_text": matched_text if matches else "",
    }


def test_add_runs_extraction_and_pass_validation_only_then_commits() -> None:
    fact = "Atlas retention is 30 days"
    query = f"Remember that {fact}"
    history = [{"hop_id": "hop-1", "text": "Atlas planning context"}]
    llm = ScriptedLLM(
        [
            _extraction(action="add", text_content=fact),
            _validation(
                operation="add",
                decision="PASS",
                selected=[],
                assessments=[],
            ),
        ]
    )
    retriever = RecordingKnowledgeRetriever([])
    repository = _repository()
    branch = _branch(llm, retriever)
    context = _context(query, history)

    # Direct branch execution still binds the PipelineContext history for LLM1;
    # production additionally owns the request-wide canonical scope.
    result = branch.execute(context, repository)

    assert result.response_type is ResponseType.KNOWLEDGE_ACTION
    assert result.actions_pending_confirmation == []
    assert [call["task"] for call in llm.calls] == [
        LLMTask.KNOWLEDGE_ACTION_EXTRACTION,
        LLMTask.KNOWLEDGE_ACTION_VALIDATION,
    ]
    assert [call["enforce_min_score"] for call in retriever.calls] == [False]
    assert retriever.calls[0]["query"] == fact
    extraction_payload = _prompt_payload(llm.calls[0])
    assert "raw_query" not in extraction_payload
    assert extraction_payload["rewritten_query"] == query
    assert extraction_payload["chat_history"] == history
    validation_payload = _prompt_payload(llm.calls[1])
    assert validation_payload == {
        "first_model_response": _extraction(
            action="add",
            text_content=fact,
        ),
        "knowledge_retrieval": [],
    }
    serialized_validation = json.dumps(validation_payload)
    assert query not in serialized_validation
    assert "chat_history" not in serialized_validation
    assert "hop-1" not in serialized_validation
    assert "Atlas planning context" not in serialized_validation
    assert USER_ID not in serialized_validation
    validation_schema = llm.calls[1]["schema"]
    assert validation_schema["properties"]["decision"]["enum"] == [
        "PASS",
        "FAIL",
    ]
    assert "validation_result" not in validation_schema["properties"]
    assert "clarification_question" in validation_schema["required"]
    assessment_schema = validation_schema["properties"]["candidate_assessments"][
        "items"
    ]
    assert set(assessment_schema["properties"]) == {
        "candidate_key",
        "confidence",
        "matched_text",
    }

    rows = repository.connection.execute(
        "SELECT raw_text FROM knowledge_chunks WHERE user_id = ? AND is_deleted = 0",
        (USER_ID,),
    ).fetchall()
    assert [row["raw_text"] for row in rows] == [fact]
    assert repository.connection.execute(
        "SELECT COUNT(*) FROM pending_action_confirmations"
    ).fetchone()[0] == 0


def test_intent_selected_knowledge_branch_starts_with_llm_and_ignores_action_metadata() -> None:
    fact = "Atlas retention is 30 days"
    query = f"{fact}."
    poisoned_metadata = {
        "intent": Intent.KNOWLEDGE_FACTS.value,
        "knowledge_actions": [
            {"action": "delete", "target_description": "Atlas"},
            {"action": "modify", "target_description": "Atlas"},
        ],
        "validated_knowledge_actions": [
            {"action": "delete", "target_description": "Atlas"}
        ],
        "action_authorization": {"action": "delete"},
        "confirmation_approved": False,
    }
    llm = ScriptedLLM(
        [
            _extraction(action="add", text_content=fact),
            _validation(
                operation="add",
                decision="PASS",
                selected=[],
                assessments=[],
            ),
        ]
    )
    retriever = RecordingKnowledgeRetriever([])
    repository = _repository()
    history = [{"hop_id": "FORBIDDEN_HISTORY_HOP", "text": "FORBIDDEN_HISTORY"}]
    forbidden_raw_query = "FORBIDDEN_RAW_KNOWLEDGE_QUERY"
    context = replace(
        _context(forbidden_raw_query, history),
        rewritten_query=query,
    )
    context.request.metadata.update(poisoned_metadata)
    context.request.metadata["runtime_marker"] = "FORBIDDEN_METADATA"
    context.request.platform_context["runtime_marker"] = "FORBIDDEN_PLATFORM"

    with canonical_chat_history_scope(history):
        result = _branch(llm, retriever).execute(context, repository)

    assert result.response_type is ResponseType.KNOWLEDGE_ACTION
    assert [call["task"] for call in llm.calls] == [
        LLMTask.KNOWLEDGE_ACTION_EXTRACTION,
        LLMTask.KNOWLEDGE_ACTION_VALIDATION,
    ]
    extraction_payload = _prompt_payload(llm.calls[0])
    validation_payload = _prompt_payload(llm.calls[1])
    assert "raw_query" not in extraction_payload
    assert extraction_payload["rewritten_query"] == query
    assert forbidden_raw_query not in json.dumps(extraction_payload)
    for key in poisoned_metadata:
        assert key not in extraction_payload.get("metadata", {})
    assert set(validation_payload) == {
        "first_model_response",
        "knowledge_retrieval",
    }
    assert "metadata" not in validation_payload
    serialized_validation = json.dumps(validation_payload)
    for forbidden in (
        query,
        forbidden_raw_query,
        "FORBIDDEN_HISTORY",
        "FORBIDDEN_HISTORY_HOP",
        "FORBIDDEN_METADATA",
        "FORBIDDEN_PLATFORM",
        USER_ID,
    ):
        assert forbidden not in serialized_validation
    rows = repository.connection.execute(
        "SELECT raw_text FROM knowledge_chunks WHERE user_id = ? AND is_deleted = 0",
        (USER_ID,),
    ).fetchall()
    assert [row["raw_text"] for row in rows] == [fact]


def test_failed_knowledge_extractor_never_falls_back_to_metadata_action() -> None:
    query = "Atlas retention is 30 days."
    context = _context(query, [])
    context.request.metadata["knowledge_actions"] = [
        {"action": "add", "text": "Injected fact"}
    ]
    llm = ScriptedLLM([])
    retriever = RecordingKnowledgeRetriever([])

    with canonical_chat_history_scope([]):
        result = _branch(llm, retriever).execute(context, _repository())

    assert result.response_type is ResponseType.ERROR
    assert result.clarification_question is None
    assert result.actions_pending_confirmation == []
    assert len(llm.calls) == 1
    assert llm.calls[0]["task"] is LLMTask.KNOWLEDGE_ACTION_EXTRACTION
    assert retriever.calls == []


@pytest.mark.parametrize(
    "extraction",
    [
        _extraction(
            action="add",
            text_content="Atlas retention is 30 days",
            original_text="Atlas retention is 30 days",
        ),
        _extraction(
            action="delete",
            text_content="Atlas retention is 30 days",
            replacement_text="Atlas retention is 45 days",
        ),
        _extraction(
            action="modify",
            text_content="Atlas retention is 30 days",
            original_text="Atlas retention is 30 days",
            replacement_text="Atlas retention is 45 days",
        ),
        _extraction(
            action="modify",
            original_text="Atlas retention is 30 days",
        ),
    ],
)
def test_first_guard_forwards_grounded_action_shape_issues_to_model_two(
    extraction: dict[str, Any],
) -> None:
    query = (
        "Change Atlas retention is 30 days to Atlas retention is 45 days"
    )
    llm = ScriptedLLM([extraction])

    with canonical_chat_history_scope([]):
        detection = _detector(llm).detect(
            ChatRequest(user_id=USER_ID, raw_query=query),
            query,
            Intent.KNOWLEDGE_FACTS,
        )

    assert not detection.requires_clarification
    assert len(llm.calls) == 1
    assert llm.calls[0]["task"] is LLMTask.KNOWLEDGE_ACTION_EXTRACTION
    assert detection.metadata["knowledge_action_extraction_response"] == extraction
    assert len(detection.metadata["knowledge_actions"]) == 1


def test_first_guard_still_rejects_ungrounded_response_content() -> None:
    query = "Remember that Atlas retention is 30 days"
    llm = ScriptedLLM(
        [
            _extraction(
                action="add",
                text_content="Atlas retention is 30 days",
                original_text="Atlas retention is 14 days",
            )
        ]
    )

    with canonical_chat_history_scope([]):
        detection = _detector(llm).detect(
            ChatRequest(user_id=USER_ID, raw_query=query),
            query,
            Intent.KNOWLEDGE_FACTS,
        )

    assert detection.requires_clarification
    assert detection.metadata == {}
    assert detection.missing_fields == ["target_description"]
    assert detection.risk_flags == [
        "knowledge_content_not_grounded_in_request_context"
    ]


def test_pipeline_guard_one_rejects_divergent_knowledge_action_state_before_validation() -> None:
    canonical_fact = "Atlas retention is 30 days"
    llm = ScriptedLLM([])
    retriever = RecordingKnowledgeRetriever([])
    branch = _branch(llm, retriever)
    context = _context(f"Remember that {canonical_fact}", [])
    context.request.metadata["knowledge_action_extraction_response"] = (
        _extraction(action="add", text_content=canonical_fact)
    )

    validated = branch.knowledge_mutation_pipeline.build_action(
        context=context,
        action_payload={
            "action": "add",
            "text": "A divergent fact that model 1 did not return",
            "confidence": 0.99,
        },
        repository=_repository(),
    )

    assert validated.hitl_reason == "internal_pipeline_failure"
    assert "guard 1" in validated.reason_summary.casefold()
    assert llm.calls == []
    assert retriever.calls == []


def test_best_effort_incomplete_first_response_reaches_model_two_fail() -> None:
    original = "Atlas retention is 30 days"
    query = f"Change {original}"
    extraction = {
        **_extraction(action="modify", original_text=original),
        "confidence": 0.20,
        "missing_fields": ["replacement_text"],
        "reason_summary": "",
    }
    repository = _repository()
    chunk_id = _seed_knowledge(repository, original)
    llm = ScriptedLLM(
        [
            extraction,
            _validation(
                operation="modify",
                decision="FAIL",
                selected=[],
                # A safe FAIL does not need complete candidate coverage in the
                # post-model-2 guard.
                assessments=[],
                clarification_question="What should replace that stored fact?",
                confidence=0.25,
                reason="",
            ),
        ]
    )
    retriever = RecordingKnowledgeRetriever(
        [_retrieval_result(chunk_id, original)]
    )

    with canonical_chat_history_scope([]):
        result = _branch(llm, retriever).execute(
            _context(query, []),
            repository,
        )

    assert result.response_type is ResponseType.CLARIFICATION
    assert result.clarification_question is not None
    assert result.clarification_question.text == (
        "What should replace that stored fact?"
    )
    assert [call["task"] for call in llm.calls] == [
        LLMTask.KNOWLEDGE_ACTION_EXTRACTION,
        LLMTask.KNOWLEDGE_ACTION_VALIDATION,
    ]
    validation_payload = _prompt_payload(llm.calls[1])
    assert validation_payload["first_model_response"] == extraction
    assert len(validation_payload["knowledge_retrieval"]) == 1
    assert repository.table_count("knowledge_chunks") == 1


def test_incomplete_first_response_cannot_be_authorized_by_model_two_pass() -> None:
    extraction = {
        **_extraction(action="add"),
        "missing_fields": ["text_content"],
    }
    llm = ScriptedLLM(
        [
            extraction,
            _validation(
                operation="add",
                decision="PASS",
                selected=[],
                assessments=[],
            ),
        ]
    )
    retriever = RecordingKnowledgeRetriever([])
    repository = _repository()

    with canonical_chat_history_scope([]):
        result = _branch(llm, retriever).execute(
            _context("Remember a fact", []),
            repository,
        )

    assert result.response_type is ResponseType.ERROR
    assert [call["task"] for call in llm.calls] == [
        LLMTask.KNOWLEDGE_ACTION_EXTRACTION,
        LLMTask.KNOWLEDGE_ACTION_VALIDATION,
    ]
    assert retriever.calls == []
    assert repository.table_count("knowledge_chunks") == 0


def test_advisory_missing_fields_do_not_override_complete_model_state() -> None:
    fact = "Atlas retention is 30 days"
    extraction = {
        **_extraction(action="add", text_content=fact),
        "missing_fields": ["text_content"],
    }
    llm = ScriptedLLM(
        [
            extraction,
            _validation(
                operation="add",
                decision="PASS",
                selected=[],
                assessments=[],
                reason="",
            ),
        ]
    )
    repository = _repository()

    with canonical_chat_history_scope([]):
        result = _branch(llm, RecordingKnowledgeRetriever([])).execute(
            _context(f"Remember that {fact}", []),
            repository,
        )

    assert result.response_type is ResponseType.KNOWLEDGE_ACTION
    assert [call["task"] for call in llm.calls] == [
        LLMTask.KNOWLEDGE_ACTION_EXTRACTION,
        LLMTask.KNOWLEDGE_ACTION_VALIDATION,
    ]
    rows = repository.connection.execute(
        "SELECT raw_text FROM knowledge_chunks WHERE user_id = ?",
        (USER_ID,),
    ).fetchall()
    assert [row["raw_text"] for row in rows] == [fact]


def test_model_two_may_authorize_grounded_low_confidence_model_one_state() -> None:
    fact = "Atlas retention is 30 days"
    extraction = {
        **_extraction(action="add", text_content=fact),
        "confidence": 0.20,
    }
    llm = ScriptedLLM(
        [
            extraction,
            _validation(
                operation="add",
                decision="PASS",
                selected=[],
                assessments=[],
                confidence=0.99,
            ),
        ]
    )
    repository = _repository()

    with canonical_chat_history_scope([]):
        result = _branch(llm, RecordingKnowledgeRetriever([])).execute(
            _context(f"Remember that {fact}", []),
            repository,
        )

    assert result.response_type is ResponseType.KNOWLEDGE_ACTION
    assert [call["task"] for call in llm.calls] == [
        LLMTask.KNOWLEDGE_ACTION_EXTRACTION,
        LLMTask.KNOWLEDGE_ACTION_VALIDATION,
    ]
    assert repository.table_count("knowledge_chunks") == 1


@pytest.mark.parametrize(
    "malformed_payload",
    [
        {
            "action": ["add", "delete"],
            "text_content": "Atlas retention is 30 days",
            "original_text": "",
            "replacement_text": "",
            "confidence": 0.99,
            "missing_fields": [],
            "reason_summary": "Invalid multi-action payload.",
        },
        {
            **_extraction(
                action="add",
                text_content="Atlas retention is 30 days",
            ),
            "second_action": "delete",
        },
        {
            key: value
            for key, value in _extraction(
                action="add",
                text_content="Atlas retention is 30 days",
            ).items()
            if key != "replacement_text"
        },
    ],
)
def test_knowledge_extractor_rejects_non_single_schema_complete_payloads(
    malformed_payload: dict[str, Any],
) -> None:
    llm = ScriptedLLM([malformed_payload])
    query = "Remember that Atlas retention is 30 days"

    with canonical_chat_history_scope([]):
        detection = _detector(llm).detect(
            ChatRequest(user_id=USER_ID, raw_query=query),
            query,
            Intent.KNOWLEDGE_FACTS,
        )

    assert detection.requires_clarification
    assert len(llm.calls) == 1
    assert detection.missing_fields == ["action"]
    assert detection.metadata.get("knowledge_actions") is None


def test_knowledge_first_llm_receives_complete_context_and_explicit_shape_contract() -> None:
    query = "Keep that policy"
    rewritten_query = "Store the Atlas retention policy"
    history = [
        {
            "hop_id": "hop-policy",
            "raw_user_query": "FORBIDDEN_RAW_HISTORY_QUERY",
            "rewritten_user_query": "What is the Atlas policy?",
            "raw_response": "Atlas retention is 30 days.",
        }
    ]
    request = ChatRequest(
        user_id=USER_ID,
        raw_query=query,
        metadata={
            "locale": "en-US",
            "knowledge_actions": [{"action": "delete"}],
        },
        platform_context={"platform": "streamlit", "timezone": "Asia/Ho_Chi_Minh"},
    )
    llm = ScriptedLLM(
        [_extraction(action="add", text_content="Atlas retention is 30 days")]
    )
    detector = _detector(llm)

    with canonical_chat_history_scope(history):
        detector.detect(request, rewritten_query, Intent.KNOWLEDGE_FACTS)

    assert len(llm.calls) == 1
    assert llm.calls[0]["task"] is LLMTask.KNOWLEDGE_ACTION_EXTRACTION
    payload = _prompt_payload(llm.calls[0])
    assert "raw_query" not in payload
    assert payload["rewritten_query"] == rewritten_query
    assert payload["intent"] == Intent.KNOWLEDGE_FACTS.value
    assert payload["chat_history"] == [
        {
            "hop_id": "hop-policy",
            "rewritten_user_query": "What is the Atlas policy?",
            "raw_response": "Atlas retention is 30 days.",
        }
    ]
    assert payload["metadata"] == {"locale": "en-US"}
    assert payload["platform_context"] == request.platform_context
    serialized_payload = json.dumps(payload)
    assert query not in serialized_payload
    assert "FORBIDDEN_RAW_HISTORY_QUERY" not in serialized_payload
    assert "raw_user_query" not in serialized_payload
    assert payload["extra"]["cardinality"] == "exactly_one"
    assert payload["extra"]["action_content_contract"] == {
        "add": {
            "required_non_empty": ["text_content"],
            "required_empty": ["original_text", "replacement_text"],
        },
        "delete": {
            "required_non_empty": ["text_content"],
            "required_empty": ["original_text", "replacement_text"],
        },
        "modify": {
            "required_non_empty": ["original_text", "replacement_text"],
            "required_empty": ["text_content"],
        },
    }


def test_duplicate_add_fail_returns_validator_question_without_finalizer_or_write() -> None:
    fact = "Atlas retention is 30 days"
    query = f"Remember that {fact}"
    history = [{"hop_id": "hop-duplicate", "text": "Prior Atlas discussion"}]
    repository = _repository()
    chunk_id = _seed_knowledge(repository, fact)
    llm = ScriptedLLM(
        [
            _extraction(action="add", text_content=fact),
            _validation(
                operation="add",
                decision="FAIL",
                selected=[],
                assessments=[
                    _assessment(chunk_id, matches=True, matched_text=fact)
                ],
                clarification_question=(
                    "That exact fact is already stored. What different fact should I add?"
                ),
                reason="The equivalent fact already exists.",
            ),
        ]
    )
    retriever = RecordingKnowledgeRetriever(
        [_retrieval_result(chunk_id, fact, score=0.01)]
    )
    branch = _branch(llm, retriever)

    with canonical_chat_history_scope(history):
        result = branch.execute(_context(query, history), repository)

    assert result.response_type is ResponseType.CLARIFICATION
    assert result.actions_pending_confirmation == []
    assert result.clarification_question is not None
    assert result.clarification_question.text == (
        "That exact fact is already stored. What different fact should I add?"
    )
    assert len(llm.calls) == 2
    assert retriever.calls[0]["enforce_min_score"] is False
    assert repository.connection.execute(
        "SELECT COUNT(*) FROM knowledge_chunks WHERE user_id = ? AND is_deleted = 0",
        (USER_ID,),
    ).fetchone()[0] == 1
    assert repository.connection.execute(
        "SELECT COUNT(*) FROM pending_action_confirmations"
    ).fetchone()[0] == 0
    validation_payload = _prompt_payload(llm.calls[1])
    candidate = validation_payload["knowledge_retrieval"][0]
    assert candidate["text_excerpt"] == fact
    assert candidate["rerank_score"] == pytest.approx(0.01)


def test_top_five_full_candidates_reach_validation_without_score_gating() -> None:
    fact = "Atlas launch region is Hanoi"
    query = f"Remember that {fact}"
    repository = _repository()
    results: list[RetrievalResult] = []
    assessments: list[dict[str, Any]] = []
    tail_markers: list[str] = []
    for index in range(5):
        marker = f"FULL-CANDIDATE-TAIL-{index}"
        text = f"{'context ' * 1200}{marker}"
        chunk_id = _seed_knowledge(repository, text)
        results.append(_retrieval_result(chunk_id, text, score=-0.10 - index))
        assessments.append(_assessment(chunk_id, matches=False))
        tail_markers.append(marker)
    llm = ScriptedLLM(
        [
            _extraction(action="add", text_content=fact),
            _validation(
                operation="add",
                decision="PASS",
                selected=[],
                assessments=assessments,
            ),
        ]
    )
    retriever = RecordingKnowledgeRetriever(results)
    default_config = build_assistant_config(ProductionSettings())
    # A stale lower validation setting must never hide candidates already
    # returned by the canonical top-five retrieval boundary.
    config = replace(
        default_config,
        retrieval_validation=replace(
            default_config.retrieval_validation,
            knowledge_llm_validation_max_candidates=1,
        ),
    )
    branch = _branch(llm, retriever, config=config)

    with canonical_chat_history_scope([]):
        result = branch.execute(_context(query, []), repository)

    assert result.response_type is ResponseType.KNOWLEDGE_ACTION
    assert len(llm.calls) == 2
    validation_payload = _prompt_payload(llm.calls[1])
    assert len(validation_payload["knowledge_retrieval"]) == 5
    serialized_prompts = json.dumps(validation_payload)
    assert all(marker in serialized_prompts for marker in tail_markers)
    assert retriever.calls[0]["enforce_min_score"] is False


def test_obviously_false_new_fact_conditionally_asks_hitl_without_finalizing_or_writing() -> None:
    fact = "1 + 1 = 3"
    query = f"Remember that {fact}"
    llm = ScriptedLLM(
        [
            _extraction(action="add", text_content=fact),
            _validation(
                operation="add",
                decision="FAIL",
                selected=[],
                assessments=[],
                clarification_question=(
                    "That arithmetic appears incorrect. What corrected fact should I save?"
                ),
                reason="The newly asserted arithmetic is false.",
            ),
        ]
    )
    retriever = RecordingKnowledgeRetriever([])
    repository = _repository()
    branch = _branch(llm, retriever)

    with canonical_chat_history_scope([]):
        result = branch.execute(_context(query, []), repository)

    assert result.response_type is ResponseType.CLARIFICATION
    assert result.actions_pending_confirmation == []
    assert result.clarification_question is not None
    assert result.clarification_question.text == (
        "That arithmetic appears incorrect. What corrected fact should I save?"
    )
    assert len(llm.calls) == 2
    assert repository.table_count("knowledge_chunks") == 0
    assert repository.connection.execute(
        "SELECT COUNT(*) FROM pending_action_confirmations"
    ).fetchone()[0] == 0


def test_low_confidence_pass_is_invalid_and_returns_error_without_writing() -> None:
    fact = "Atlas retention is 30 days"
    query = f"Remember that {fact}"
    llm = ScriptedLLM(
        [
            _extraction(action="add", text_content=fact),
            _validation(
                operation="add",
                decision="PASS",
                selected=[],
                assessments=[],
                confidence=0.25,
                reason="The validator is not confident enough.",
            ),
        ]
    )
    retriever = RecordingKnowledgeRetriever([])
    repository = _repository()

    with canonical_chat_history_scope([]):
        result = _branch(llm, retriever).execute(
            _context(query, []), repository
        )

    assert result.response_type is ResponseType.ERROR
    assert result.clarification_question is None
    assert [call["task"] for call in llm.calls] == [
        LLMTask.KNOWLEDGE_ACTION_EXTRACTION,
        LLMTask.KNOWLEDGE_ACTION_VALIDATION,
    ]
    assert repository.table_count("knowledge_chunks") == 0


def test_conflicting_add_is_owned_by_second_llm_and_never_reaches_finalizer() -> None:
    existing = "Atlas retention is 30 days"
    proposed = "Atlas retention is 45 days"
    query = f"Remember that {proposed}"
    repository = _repository()
    chunk_id = _seed_knowledge(repository, existing)
    llm = ScriptedLLM(
        [
            _extraction(action="add", text_content=proposed),
            _validation(
                operation="add",
                decision="FAIL",
                selected=[],
                assessments=[
                    _assessment(
                        chunk_id,
                        matches=True,
                        matched_text=existing,
                    )
                ],
                clarification_question=(
                    "Atlas retention is stored as 30 days. Should I replace it with 45 days instead?"
                ),
                reason="The proposed fact conflicts with the stored fact.",
            ),
        ]
    )
    retriever = RecordingKnowledgeRetriever(
        [_retrieval_result(chunk_id, existing, score=-0.50)]
    )

    with canonical_chat_history_scope([]):
        result = _branch(llm, retriever).execute(
            _context(query, []), repository
        )

    assert result.response_type is ResponseType.CLARIFICATION
    assert result.clarification_question is not None
    assert result.clarification_question.text == (
        "Atlas retention is stored as 30 days. Should I replace it with 45 days instead?"
    )
    assert [call["task"] for call in llm.calls] == [
        LLMTask.KNOWLEDGE_ACTION_EXTRACTION,
        LLMTask.KNOWLEDGE_ACTION_VALIDATION,
    ]
    assert repository.connection.execute(
        "SELECT COUNT(*) FROM knowledge_chunks WHERE user_id = ? AND is_deleted = 0",
        (USER_ID,),
    ).fetchone()[0] == 1


@pytest.mark.parametrize(
    "invalid_validation",
    [
        _validation(
            operation="add",
            decision="EXECUTE",
            selected=[],
            assessments=[],
        ),
        {
            **_validation(
                operation="add",
                decision="PASS",
                selected=[],
                assessments=[],
            ),
            "should_execute": True,
        },
        _validation(
            operation="add",
            decision="FAIL",
            selected=[],
            assessments=[],
            clarification_question="",
        ),
    ],
)
def test_binary_validator_rejects_old_extra_or_questionless_contracts(
    invalid_validation: dict[str, Any],
) -> None:
    fact = "Atlas retention is 30 days"
    query = f"Remember that {fact}"
    llm = ScriptedLLM(
        [
            _extraction(action="add", text_content=fact),
            invalid_validation,
        ]
    )
    repository = _repository()

    with canonical_chat_history_scope([]):
        result = _branch(llm, RecordingKnowledgeRetriever([])).execute(
            _context(query, []), repository
        )

    assert result.response_type is ResponseType.ERROR
    assert result.clarification_question is None
    assert [call["task"] for call in llm.calls] == [
        LLMTask.KNOWLEDGE_ACTION_EXTRACTION,
        LLMTask.KNOWLEDGE_ACTION_VALIDATION,
    ]
    assert repository.table_count("knowledge_chunks") == 0


def test_validation_model_failure_returns_error_without_hitl_finalizer_or_write() -> None:
    fact = "Atlas retention is 30 days"
    query = f"Remember that {fact}"
    llm = ScriptedLLM(
        [
            _extraction(action="add", text_content=fact),
            RuntimeError("validator unavailable"),
        ]
    )
    repository = _repository()

    with canonical_chat_history_scope([]):
        result = _branch(llm, RecordingKnowledgeRetriever([])).execute(
            _context(query, []), repository
        )

    assert result.response_type is ResponseType.ERROR
    assert result.clarification_question is None
    assert [call["task"] for call in llm.calls] == [
        LLMTask.KNOWLEDGE_ACTION_EXTRACTION,
        LLMTask.KNOWLEDGE_ACTION_VALIDATION,
    ]
    assert repository.table_count("knowledge_chunks") == 0


def test_inconsistent_validation_contract_returns_error_not_user_hitl() -> None:
    fact = "Atlas retention is 30 days"
    query = f"Remember that {fact}"
    llm = ScriptedLLM(
        [
            _extraction(action="add", text_content=fact),
            _validation(
                operation="add",
                decision="PASS",
                selected=[],
                assessments=[],
                clarification_question="This must be empty on PASS.",
                reason="Malformed execute contract.",
            ),
        ]
    )
    repository = _repository()

    with canonical_chat_history_scope([]):
        result = _branch(llm, RecordingKnowledgeRetriever([])).execute(
            _context(query, []), repository
        )

    assert result.response_type is ResponseType.ERROR
    assert result.clarification_question is None
    assert [call["task"] for call in llm.calls] == [
        LLMTask.KNOWLEDGE_ACTION_EXTRACTION,
        LLMTask.KNOWLEDGE_ACTION_VALIDATION,
    ]
    assert repository.table_count("knowledge_chunks") == 0


def test_finalizer_integrity_failure_returns_error_without_hitl_or_write() -> None:
    original = "Atlas retention is 30 days"
    replacement = "Atlas retention is 45 days"
    query = f"Change {original} to {replacement}"
    repository = _repository()
    chunk_id = _seed_knowledge(repository, original)
    llm = ScriptedLLM(
        [
            _extraction(
                action="modify",
                original_text=original,
                replacement_text=replacement,
            ),
            _validation(
                operation="modify",
                decision="PASS",
                selected=[chunk_id],
                assessments=[
                    _assessment(chunk_id, matches=True, matched_text=original)
                ],
            ),
            {
                "final_content": "A different invented fact",
                "confidence": 0.99,
                "reason_summary": "Invalid finalization output.",
            },
        ]
    )
    with canonical_chat_history_scope([]):
        result = _branch(
            llm,
            RecordingKnowledgeRetriever(
                [_retrieval_result(chunk_id, original)]
            ),
        ).execute(
            _context(query, []), repository
        )

    assert result.response_type is ResponseType.ERROR
    assert result.clarification_question is None
    assert [call["task"] for call in llm.calls] == [
        LLMTask.KNOWLEDGE_ACTION_EXTRACTION,
        LLMTask.KNOWLEDGE_ACTION_VALIDATION,
        LLMTask.KNOWLEDGE_CONTENT_FINALIZATION,
    ]
    row = repository.connection.execute(
        "SELECT raw_text FROM knowledge_chunks WHERE chunk_id = ?",
        (chunk_id,),
    ).fetchone()
    assert row["raw_text"] == original


def test_repeated_modify_match_is_deferred_to_finalization_guard() -> None:
    original = "Atlas retention is 30 days"
    replacement = "Atlas retention is 45 days"
    stored = f"{original}. Repeated policy: {original}."
    query = f"Change {original} to {replacement}"
    repository = _repository()
    chunk_id = _seed_knowledge(repository, stored)
    llm = ScriptedLLM(
        [
            _extraction(
                action="modify",
                original_text=original,
                replacement_text=replacement,
            ),
            _validation(
                operation="modify",
                decision="PASS",
                selected=[chunk_id],
                assessments=[
                    _assessment(
                        chunk_id,
                        matches=True,
                        matched_text=original,
                    )
                ],
            ),
        ]
    )

    with canonical_chat_history_scope([]):
        result = _branch(
            llm,
            RecordingKnowledgeRetriever(
                [_retrieval_result(chunk_id, stored)]
            ),
        ).execute(_context(query, []), repository)

    assert result.response_type is ResponseType.ERROR
    # Model 2's response guard accepts the valid candidate decision. The next
    # finalization state owns unique replacement occurrence and stops before
    # calling model 3 when that precondition is ambiguous.
    assert [call["task"] for call in llm.calls] == [
        LLMTask.KNOWLEDGE_ACTION_EXTRACTION,
        LLMTask.KNOWLEDGE_ACTION_VALIDATION,
    ]
    row = repository.connection.execute(
        "SELECT raw_text FROM knowledge_chunks WHERE chunk_id = ?",
        (chunk_id,),
    ).fetchone()
    assert row["raw_text"] == stored


def test_modify_replacement_cannot_be_sourced_only_from_history() -> None:
    history = [
        {
            "hop_id": "hop-old",
            "text": "Atlas retention is 30 days. A discarded idea was 45 days.",
        }
    ]
    llm = ScriptedLLM(
        [
            _extraction(
                action="modify",
                original_text="Atlas retention is 30 days",
                replacement_text="45 days",
            )
        ]
    )
    retriever = RecordingKnowledgeRetriever([])
    branch = _branch(llm, retriever)
    context = _context("Modify that", history)

    with canonical_chat_history_scope(history):
        result = branch.execute(context, _repository())

    assert result.response_type is ResponseType.CLARIFICATION
    assert result.clarification_question is not None
    assert "replace" in result.clarification_question.text.casefold()
    assert len(llm.calls) == 1
    assert retriever.calls == []


def test_modify_preserves_long_chunk_then_commits_without_confirmation_gate() -> None:
    original = "Atlas retention is 30 days"
    replacement = "Atlas retention is 45 days"
    prefix = " ".join(f"Unrelated detail {index}." for index in range(500))
    stored = f"{prefix}\n\n{original}.\n  Keep this unrelated tail marker."
    final_content = stored.replace(original, replacement, 1)
    query = f"Change {original} to {replacement}"
    history = [
        {
            "source": "last_qa",
            "topic_id": "topic-atlas",
            "hop_id": "hop-long",
            "text": "Continue the Atlas record",
            "supporting_questions": [
                {
                    "text": "What retention period should replace 30 days?",
                    "purpose": "resolve_replacement",
                }
            ],
            "clarification_question": None,
            "reminder_supporting_question": None,
            "expected_response_type": "replacement_text_answer",
        }
    ]
    repository = _repository()
    chunk_id = _seed_knowledge(repository, stored)
    llm = ScriptedLLM(
        [
            _extraction(
                action="modify",
                original_text=original,
                replacement_text=replacement,
            ),
            _validation(
                operation="modify",
                decision="PASS",
                selected=[chunk_id],
                assessments=[
                    _assessment(
                        chunk_id,
                        matches=True,
                        matched_text=original,
                    )
                ],
            ),
            {
                "final_content": final_content,
                "confidence": 0.99,
                "reason_summary": "Replaced only the validated detail.",
            },
        ]
    )
    retriever = RecordingKnowledgeRetriever(
        [_retrieval_result(chunk_id, stored, score=0.01)]
    )
    branch = _branch(llm, retriever)
    poisoned_metadata = {
        "knowledge_actions": [{"action": "delete"}],
        "validated_knowledge_actions": [{"action": "delete"}],
        "action_authorization": {"action": "delete"},
        "confirmation_approved": False,
    }
    forbidden_raw_query = "FORBIDDEN_RAW_KNOWLEDGE_MODIFY_QUERY"
    context = replace(
        _context(forbidden_raw_query, history),
        rewritten_query=query,
    )
    context.request.metadata.update(poisoned_metadata)

    with canonical_chat_history_scope(history):
        result = branch.execute(context, repository)

    assert result.response_type is ResponseType.KNOWLEDGE_ACTION
    assert result.actions_pending_confirmation == []
    assert len(llm.calls) == 3
    assert retriever.calls[0]["query"] == original
    validation_payload = _prompt_payload(llm.calls[1])
    finalization_payload = _prompt_payload(llm.calls[2])
    assert "Keep this unrelated tail marker." in json.dumps(validation_payload)
    assert "Keep this unrelated tail marker." in json.dumps(finalization_payload)
    assert (
        validation_payload["knowledge_retrieval"][0]["text_excerpt"]
        == stored
    )
    assert (
        finalization_payload["extra"]["validated_candidate_context"][0]["text"]
        == stored
    )
    assert len(
        finalization_payload["extra"]["validated_candidate_context"]
    ) == 1
    assert (
        finalization_payload["extra"]["validated_candidate_context"][0][
            "matched_text"
        ]
        == original
    )
    assert validation_payload["first_model_response"] == _extraction(
        action="modify",
        original_text=original,
        replacement_text=replacement,
    )
    assert finalization_payload["extra"]["first_model_response"] == (
        validation_payload["first_model_response"]
    )
    assert set(validation_payload) == {
        "first_model_response",
        "knowledge_retrieval",
    }
    serialized_validation = json.dumps(validation_payload)
    assert query not in serialized_validation
    assert "chat_history" not in serialized_validation
    assert "hop-long" not in serialized_validation
    assert finalization_payload["extra"]["extracted_action_content"] == {
        "text_content": "",
        "original_text": original,
        "replacement_text": replacement,
    }
    extraction_supporting_context = _prompt_payload(llm.calls[0])["extra"][
        "supporting_question_context"
    ]
    finalization_supporting_context = finalization_payload["extra"][
        "supporting_question_context"
    ]
    assert extraction_supporting_context
    assert extraction_supporting_context == finalization_supporting_context
    for call in (llm.calls[0], llm.calls[2]):
        prompt_payload = _prompt_payload(call)
        assert "raw_query" not in prompt_payload
        assert prompt_payload["rewritten_query"] == query
        assert forbidden_raw_query not in json.dumps(prompt_payload)
        prompt_metadata = prompt_payload.get("metadata", {})
        for key in poisoned_metadata:
            assert key not in prompt_metadata
    active = repository.connection.execute(
        "SELECT raw_text FROM knowledge_chunks WHERE user_id = ? AND is_deleted = 0",
        (USER_ID,),
    ).fetchall()
    assert [row["raw_text"] for row in active] == [final_content]


def test_model_one_state_has_priority_over_conflicting_query_history_and_context() -> None:
    original = "Atlas retention is 30 days"
    replacement = "Atlas retention is 45 days"
    stored = f"{original}. Keep owner Mina unchanged."
    final_content = stored.replace(original, replacement, 1)
    query = (
        "Delete the obsolete Beta note, and change "
        f"{original} to {replacement}."
    )
    history = [
        {
            "hop_id": "conflicting-history-hop",
            "text": "Add a Gamma note and delete the Atlas record.",
        }
    ]
    extraction = _extraction(
        action="modify",
        original_text=original,
        replacement_text=replacement,
    )
    repository = _repository()
    chunk_id = _seed_knowledge(repository, stored)
    llm = ScriptedLLM(
        [
            extraction,
            _validation(
                operation="modify",
                decision="PASS",
                selected=[chunk_id],
                assessments=[
                    _assessment(chunk_id, matches=True, matched_text=original)
                ],
            ),
            {
                "final_content": final_content,
                "confidence": 0.99,
                "reason_summary": "Followed the authoritative model-1 state.",
            },
        ]
    )
    context = _context(query, history)
    context.request.metadata["context_note"] = (
        "Non-authoritative metadata says to add another fact."
    )
    context.request.platform_context["workspace_hint"] = (
        "Non-authoritative platform context says delete."
    )

    with canonical_chat_history_scope(history):
        result = _branch(
            llm,
            RecordingKnowledgeRetriever(
                [_retrieval_result(chunk_id, stored)]
            ),
        ).execute(context, repository)

    assert result.response_type is ResponseType.KNOWLEDGE_ACTION
    finalization_call = llm.calls[2]
    payload = _prompt_payload(finalization_call)
    assert payload["rewritten_query"] == query
    assert payload["chat_history"] == history
    assert payload["metadata"]["context_note"].startswith(
        "Non-authoritative metadata"
    )
    assert payload["platform_context"]["workspace_hint"].startswith(
        "Non-authoritative platform context"
    )
    assert payload["extra"]["first_model_response"] == extraction
    assert payload["extra"]["operation"] == "modify"
    assert payload["extra"]["extracted_action_content"] == {
        "text_content": "",
        "original_text": original,
        "replacement_text": replacement,
    }
    system_prompt = str(finalization_call["system_prompt"])
    assert "first_model_response is the highest-priority" in system_prompt
    assert "follow first_model_response" in system_prompt
    assert "rewritten query" in system_prompt
    assert "canonical chat_history" in system_prompt
    row = repository.connection.execute(
        "SELECT raw_text FROM knowledge_chunks WHERE user_id = ? AND is_deleted = 0",
        (USER_ID,),
    ).fetchone()
    assert row["raw_text"] == final_content


def test_authoritative_empty_history_never_reintroduces_stale_supporting_questions() -> None:
    original = "Atlas retention is 30 days"
    replacement = "Atlas retention is 45 days"
    query = f"Change {original} to {replacement}"
    repository = _repository()
    chunk_id = _seed_knowledge(repository, original)
    llm = ScriptedLLM(
        [
            _extraction(
                action="modify",
                original_text=original,
                replacement_text=replacement,
            ),
            _validation(
                operation="modify",
                decision="PASS",
                selected=[chunk_id],
                assessments=[
                    _assessment(chunk_id, matches=True, matched_text=original)
                ],
            ),
            {
                "final_content": replacement,
                "confidence": 0.99,
                "reason_summary": "Replaced the validated fact.",
            },
        ]
    )
    context = replace(
        _context(query, []),
        last_qa_state=SimpleNamespace(
            supporting_questions=[
                SimpleNamespace(text="A stale supporting question")
            ]
        ),
    )

    with canonical_chat_history_scope([]):
        result = _branch(
            llm,
            RecordingKnowledgeRetriever(
                [_retrieval_result(chunk_id, original)]
            ),
        ).execute(context, repository)

    assert result.response_type is ResponseType.KNOWLEDGE_ACTION
    assert result.actions_pending_confirmation == []
    assert [call["task"] for call in llm.calls] == [
        LLMTask.KNOWLEDGE_ACTION_EXTRACTION,
        LLMTask.KNOWLEDGE_ACTION_VALIDATION,
        LLMTask.KNOWLEDGE_CONTENT_FINALIZATION,
    ]
    for call in (llm.calls[0], llm.calls[2]):
        payload = _prompt_payload(call)
        assert payload["chat_history"] == []
        assert payload["extra"]["supporting_question_context"] == []
    assert _prompt_payload(llm.calls[2])["extra"][
        "first_model_response"
    ] == _extraction(
        action="modify",
        original_text=original,
        replacement_text=replacement,
    )
    validation_payload = _prompt_payload(llm.calls[1])
    assert set(validation_payload) == {
        "first_model_response",
        "knowledge_retrieval",
    }
    assert "chat_history" not in validation_payload


def test_partial_chunk_delete_fails_closed_to_specific_hitl_without_deleting() -> None:
    target = "Atlas retention is 30 days"
    stored = f"{target}. Unrelated owner is Mina."
    query = f"Delete {target}"
    repository = _repository()
    chunk_id = _seed_knowledge(repository, stored)
    llm = ScriptedLLM(
        [
            _extraction(action="delete", text_content=target),
            _validation(
                operation="delete",
                decision="FAIL",
                selected=[],
                assessments=[
                    _assessment(chunk_id, matches=True, matched_text=target)
                ],
                clarification_question=(
                    "That detail is part of a larger stored item. Should I modify the item and preserve its owner detail?"
                ),
            ),
        ]
    )
    retriever = RecordingKnowledgeRetriever(
        [_retrieval_result(chunk_id, stored, score=0.01)]
    )
    branch = _branch(llm, retriever)

    with canonical_chat_history_scope([]):
        result = branch.execute(_context(query, []), repository)

    assert result.response_type is ResponseType.CLARIFICATION
    assert result.clarification_question is not None
    assert result.clarification_question.text == (
        "That detail is part of a larger stored item. Should I modify the item and preserve its owner detail?"
    )
    assert len(llm.calls) == 2
    row = repository.connection.execute(
        "SELECT is_deleted FROM knowledge_chunks WHERE chunk_id = ?",
        (chunk_id,),
    ).fetchone()
    assert row["is_deleted"] == 0


def test_whole_chunk_delete_pass_commits_without_finalizer_or_confirmation() -> None:
    target = "Atlas temporary codename is Blue Lantern"
    query = f"Delete {target}"
    repository = _repository()
    chunk_id = _seed_knowledge(repository, target)
    llm = ScriptedLLM(
        [
            _extraction(action="delete", text_content=target),
            _validation(
                operation="delete",
                decision="PASS",
                selected=[chunk_id],
                assessments=[
                    _assessment(chunk_id, matches=True, matched_text=target)
                ],
            ),
        ]
    )
    retriever = RecordingKnowledgeRetriever(
        [_retrieval_result(chunk_id, target, score=-0.20)]
    )
    branch = _branch(llm, retriever)

    with canonical_chat_history_scope([]):
        result = branch.execute(_context(query, []), repository)

    assert result.response_type is ResponseType.KNOWLEDGE_ACTION
    assert result.actions_pending_confirmation == []
    assert retriever.calls[0]["query"] == target
    assert [call["task"] for call in llm.calls] == [
        LLMTask.KNOWLEDGE_ACTION_EXTRACTION,
        LLMTask.KNOWLEDGE_ACTION_VALIDATION,
    ]
    row = repository.connection.execute(
        "SELECT is_deleted FROM knowledge_chunks WHERE chunk_id = ?",
        (chunk_id,),
    ).fetchone()
    assert row["is_deleted"] == 1
    outbox_operations = {
        outbox_row["operation"]
        for outbox_row in repository.connection.execute(
            "SELECT operation FROM indexing_outbox WHERE entity_type = ? AND entity_id = ?",
            ("knowledge_chunk", chunk_id),
        ).fetchall()
    }
    assert "delete" in outbox_operations
