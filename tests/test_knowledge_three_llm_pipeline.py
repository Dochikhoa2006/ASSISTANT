from __future__ import annotations

import json
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
from assistant_rag.request_lifecycle import ChatRequestLifecycleExecutor
from assistant_rag.retrieval_validation import KnowledgeRetrievalValidationStrategy
from assistant_rag.settings import ProductionSettings


USER_ID = "knowledge-three-llm-user"


class ScriptedLLM:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def generate_json(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(dict(kwargs))
        if not self.responses:
            raise AssertionError("unexpected extra knowledge LLM call")
        return self.responses.pop(0)

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
) -> KnowledgeFactsBranch:
    config = build_assistant_config(ProductionSettings())
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
        "missing_fields": [],
        "reason_summary": "One grounded knowledge action was extracted.",
    }


def _validation(
    *,
    operation: str,
    result: str,
    selected: list[str],
    assessments: list[dict[str, Any]],
    should_execute: bool,
    requires_hitl: bool = False,
    factuality_concern: bool = False,
    ambiguous: bool = False,
    reason: str = "Knowledge action was validated.",
) -> dict[str, Any]:
    return {
        "operation": operation,
        "validation_result": result,
        "selected_candidate_keys": selected,
        "confidence": 0.99,
        "ambiguous": ambiguous,
        "should_execute": should_execute,
        "requires_hitl": requires_hitl,
        "factuality_concern": factuality_concern,
        "reason_summary": reason,
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
        "matches_target": matches,
        "action_compatible": compatible,
        "confidence": 0.99,
        "matched_fields": ["text"] if matches else [],
        "reason_summary": "Candidate was assessed against the full SQL text.",
        "matched_text": matched_text,
    }


def test_add_runs_exactly_three_llm_stages_with_identical_history_and_commits() -> None:
    fact = "Atlas retention is 30 days"
    query = f"Remember that {fact}"
    history = [{"hop_id": "hop-1", "text": "Atlas planning context"}]
    llm = ScriptedLLM(
        [
            _extraction(action="add", text_content=fact),
            _validation(
                operation="add",
                result="EXECUTE",
                selected=[],
                assessments=[],
                should_execute=True,
            ),
            {
                "final_content": fact,
                "confidence": 0.99,
                "reason_summary": "Preserved the grounded fact exactly.",
            },
        ]
    )
    retriever = RecordingKnowledgeRetriever([])
    repository = _repository()
    branch = _branch(llm, retriever)
    context = _context(query, history)

    with canonical_chat_history_scope(history):
        result = branch.execute(context, repository)

    assert result.response_type is ResponseType.KNOWLEDGE_ACTION
    assert [call["task"] for call in llm.calls] == [
        LLMTask.KNOWLEDGE_ACTION_EXTRACTION,
        LLMTask.KNOWLEDGE_ACTION_VALIDATION,
        LLMTask.KNOWLEDGE_CONTENT_FINALIZATION,
    ]
    assert [call["enforce_min_score"] for call in retriever.calls] == [False]
    for call in llm.calls:
        payload = _prompt_payload(call)
        assert payload["raw_query"] == query
        assert payload["chat_history"] == history

    rows = repository.connection.execute(
        "SELECT raw_text FROM knowledge_chunks WHERE user_id = ? AND is_deleted = 0",
        (USER_ID,),
    ).fetchall()
    assert [row["raw_text"] for row in rows] == [fact]


def test_duplicate_add_uses_low_score_candidate_but_skips_finalizer_write_and_hitl() -> None:
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
                result="SKIP_ALREADY_EXISTS",
                selected=[],
                assessments=[
                    _assessment(chunk_id, matches=True, matched_text=fact)
                ],
                should_execute=False,
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

    assert result.response_type is ResponseType.KNOWLEDGE_ACTION
    assert len(llm.calls) == 2
    assert retriever.calls[0]["enforce_min_score"] is False
    assert repository.connection.execute(
        "SELECT COUNT(*) FROM knowledge_chunks WHERE user_id = ? AND is_deleted = 0",
        (USER_ID,),
    ).fetchone()[0] == 1
    validation_payload = _prompt_payload(llm.calls[1])
    candidate = validation_payload["extra"]["candidate_chunks"][0]
    assert candidate["text_excerpt"] == fact
    assert candidate["rerank_score"] == pytest.approx(0.01)


def test_top_five_full_candidates_reach_validation_and_finalization_without_score_gating() -> None:
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
                result="EXECUTE",
                selected=[],
                assessments=assessments,
                should_execute=True,
            ),
            {
                "final_content": fact,
                "confidence": 0.99,
                "reason_summary": "Preserved the unique fact.",
            },
        ]
    )
    retriever = RecordingKnowledgeRetriever(results)
    branch = _branch(llm, retriever)

    with canonical_chat_history_scope([]):
        result = branch.execute(_context(query, []), repository)

    assert result.response_type is ResponseType.KNOWLEDGE_ACTION
    assert len(llm.calls) == 3
    validation_payload = _prompt_payload(llm.calls[1])
    finalization_payload = _prompt_payload(llm.calls[2])
    assert len(validation_payload["extra"]["candidate_chunks"]) == 5
    assert len(
        finalization_payload["extra"]["validated_candidate_context"]
    ) == 5
    serialized_prompts = json.dumps(
        [validation_payload, finalization_payload]
    )
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
                result="CLARIFY_MISSING_FIELDS",
                selected=[],
                assessments=[],
                should_execute=False,
                requires_hitl=True,
                factuality_concern=True,
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
    assert result.clarification_question is not None
    assert "incorrect" in result.clarification_question.text.casefold()
    assert len(llm.calls) == 2
    assert repository.table_count("knowledge_chunks") == 0


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


def test_modify_preserves_long_chunk_and_confirmation_replay_makes_no_new_llm_calls() -> None:
    original = "Atlas retention is 30 days"
    replacement = "Atlas retention is 45 days"
    prefix = " ".join(f"Unrelated detail {index}." for index in range(500))
    stored = f"{prefix} {original}. Keep this unrelated tail marker."
    final_content = stored.replace(original, replacement, 1)
    query = f"Change {original} to {replacement}"
    history = [{"hop_id": "hop-long", "text": "Continue the Atlas record"}]
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
                result="EXECUTE",
                selected=[chunk_id],
                assessments=[
                    _assessment(
                        chunk_id,
                        matches=True,
                        matched_text=original,
                    )
                ],
                should_execute=True,
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

    with canonical_chat_history_scope(history):
        pending = branch.execute(_context(query, history), repository)

    assert pending.response_type is ResponseType.KNOWLEDGE_ACTION
    assert pending.actions_pending_confirmation
    assert len(llm.calls) == 3
    validation_payload = _prompt_payload(llm.calls[1])
    finalization_payload = _prompt_payload(llm.calls[2])
    assert "Keep this unrelated tail marker." in json.dumps(validation_payload)
    assert "Keep this unrelated tail marker." in json.dumps(finalization_payload)

    token = pending.actions_pending_confirmation[0]["confirmation_token"]
    lifecycle = ChatRequestLifecycleExecutor(
        pipeline=SimpleNamespace(),  # type: ignore[arg-type]
        repository=repository,
    )
    prepared, confirmation_to_mark = lifecycle._hydrate_confirmation(
        ChatRequest(
            user_id=USER_ID,
            raw_query="Confirm",
            confirmation_token=token,
        )
    )
    replay_context = PipelineContext(
        request=prepared,
        rewritten_query="Confirm",
        last_qa_state=None,
        conversation_results=[],
        intent=Intent.KNOWLEDGE_FACTS,
        chat_history=history,
    )
    with canonical_chat_history_scope(history):
        committed = branch.execute(replay_context, repository)

    assert confirmation_to_mark == token
    assert committed.response_type is ResponseType.KNOWLEDGE_ACTION
    assert len(llm.calls) == 3
    active = repository.connection.execute(
        "SELECT raw_text FROM knowledge_chunks WHERE user_id = ? AND is_deleted = 0",
        (USER_ID,),
    ).fetchall()
    assert [row["raw_text"] for row in active] == [final_content]


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
                result="EXECUTE",
                selected=[chunk_id],
                assessments=[
                    _assessment(chunk_id, matches=True, matched_text=target)
                ],
                should_execute=True,
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
    assert "larger stored knowledge item" in result.clarification_question.text
    assert len(llm.calls) == 2
    row = repository.connection.execute(
        "SELECT is_deleted FROM knowledge_chunks WHERE chunk_id = ?",
        (chunk_id,),
    ).fetchone()
    assert row["is_deleted"] == 0


def test_whole_chunk_delete_uses_all_three_stages_then_requires_confirmation() -> None:
    target = "Atlas temporary codename is Blue Lantern"
    query = f"Delete {target}"
    repository = _repository()
    chunk_id = _seed_knowledge(repository, target)
    llm = ScriptedLLM(
        [
            _extraction(action="delete", text_content=target),
            _validation(
                operation="delete",
                result="EXECUTE",
                selected=[chunk_id],
                assessments=[
                    _assessment(chunk_id, matches=True, matched_text=target)
                ],
                should_execute=True,
            ),
            {
                "final_content": target,
                "confidence": 0.99,
                "reason_summary": "Copied the validated whole chunk exactly.",
            },
        ]
    )
    retriever = RecordingKnowledgeRetriever(
        [_retrieval_result(chunk_id, target, score=-0.20)]
    )
    branch = _branch(llm, retriever)

    with canonical_chat_history_scope([]):
        result = branch.execute(_context(query, []), repository)

    assert result.response_type is ResponseType.KNOWLEDGE_ACTION
    assert result.actions_pending_confirmation
    assert [call["task"] for call in llm.calls] == [
        LLMTask.KNOWLEDGE_ACTION_EXTRACTION,
        LLMTask.KNOWLEDGE_ACTION_VALIDATION,
        LLMTask.KNOWLEDGE_CONTENT_FINALIZATION,
    ]
    row = repository.connection.execute(
        "SELECT is_deleted FROM knowledge_chunks WHERE chunk_id = ?",
        (chunk_id,),
    ).fetchone()
    assert row["is_deleted"] == 0
