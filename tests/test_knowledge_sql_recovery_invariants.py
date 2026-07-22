from __future__ import annotations

from dataclasses import replace
import json
from types import SimpleNamespace
from typing import Any, Iterable

import pytest

from assistant_rag.branches import BranchRouter, GeneralResponseBranch
from assistant_rag.bundler import ChatOutput, ResponseBundler
from assistant_rag.canonical_retrieval import (
    KnowledgeRetrievalUnavailableError,
    retrieve_knowledge,
)
from assistant_rag.config import OutboxConfig
from assistant_rag.contracts import (
    ActionValidationResult,
    BranchResult,
    ChatRequest,
    Intent,
    KnowledgeAction,
    LastQAPath,
    LastQAResolution,
    PipelineContext,
    RepositoryActionResult,
    ResponseType,
    RetrievalResult,
    ValidatedKnowledgeAction,
)
from assistant_rag.context_filter import HardRuleContextFilter
from assistant_rag.database import SQLiteRepository
from assistant_rag.llm import LLMTask
from assistant_rag.pipeline import AssistantPipeline
from assistant_rag.retrieval import HybridRetriever


REQUEST_USER = "sql-recovery-owner"
OTHER_USER = "sql-recovery-neighbor"
REQUEST_FACT = "Project Sable's calibration cycle lasts forty-three hours."
OTHER_FACT = "Project Sable's calibration cycle lasts eighty-seven hours."
RECALL_QUERY = "When is Sable due for another calibration cycle?"


class _UnavailableDerivedIndex:
    def __init__(self, *, raises: bool) -> None:
        self.raises = raises

    def search(self, **_kwargs: Any) -> list[RetrievalResult]:
        if self.raises:
            raise RuntimeError("derived store is unavailable")
        return []

    def upsert(self, **_kwargs: Any) -> None:
        return None

    def delete(self, *, entity_id: str) -> None:
        del entity_id

    def clear(self) -> None:
        return None


class _RecordingReranker:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[RetrievalResult], bool]] = []

    def rerank(
        self,
        query: str,
        results: Iterable[RetrievalResult],
        *,
        enforce_min_score: bool = True,
    ) -> list[RetrievalResult]:
        candidates = list(results)
        self.calls.append((query, candidates, enforce_min_score))
        return [
            replace(candidate, rerank_score=0.93, confidence=0.93)
            for candidate in candidates
        ]


class _StaticDerivedIndex:
    def __init__(self, results: Iterable[RetrievalResult]) -> None:
        self.results = list(results)

    def search(self, **_kwargs: Any) -> list[RetrievalResult]:
        return list(self.results)

    def upsert(self, **_kwargs: Any) -> None:
        return None

    def delete(self, *, entity_id: str) -> None:
        del entity_id

    def clear(self) -> None:
        return None


class _NarrowDerivedRetriever:
    """Compatibility fixture intentionally lacking a semantic reranker."""

    def __init__(self, results: Iterable[RetrievalResult]) -> None:
        self.results = list(results)
        self.sql_fallback_candidate_limit = 5

    def retrieve_knowledge(self, **_kwargs: Any) -> list[RetrievalResult]:
        return list(self.results)


class _EntityScoreReranker:
    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.seen_entity_ids: list[list[str]] = []

    def rerank(
        self,
        _query: str,
        results: Iterable[RetrievalResult],
        *,
        enforce_min_score: bool = True,
    ) -> list[RetrievalResult]:
        del enforce_min_score
        candidates = list(results)
        self.seen_entity_ids.append([candidate.entity_id for candidate in candidates])
        return sorted(
            (
                replace(
                    candidate,
                    rerank_score=self.scores.get(candidate.entity_id, 0.31),
                    confidence=self.scores.get(candidate.entity_id, 0.31),
                )
                for candidate in candidates
            ),
            key=lambda candidate: candidate.rerank_score,
            reverse=True,
        )


def _repository() -> SQLiteRepository:
    repository = SQLiteRepository.in_memory()
    repository.initialize_schema()
    return repository


def _seed_fact(
    repository: SQLiteRepository,
    *,
    user_id: str,
    text: str,
) -> str:
    with repository.transaction() as cursor:
        _topic_id, chunk_id, _outbox_job_id = repository.add_knowledge_chunk(
            cursor,
            user_id=user_id,
            title="Operational Notes",
            text=text,
        )
    return chunk_id


@pytest.mark.parametrize("derived_mode", ("raises", "empty"))
def test_canonical_knowledge_retrieval_recovers_owned_sql_facts_through_shared_reranker(
    derived_mode: str,
) -> None:
    repository = _repository()
    requested_chunk_id = _seed_fact(
        repository,
        user_id=REQUEST_USER,
        text=REQUEST_FACT,
    )
    other_chunk_id = _seed_fact(
        repository,
        user_id=OTHER_USER,
        text=OTHER_FACT,
    )
    reranker = _RecordingReranker()
    retriever = HybridRetriever(
        bm25=_UnavailableDerivedIndex(raises=derived_mode == "raises"),
        chroma=_UnavailableDerivedIndex(raises=False),
        reranker=reranker,
    )

    recovered = retrieve_knowledge(
        retriever=retriever,
        repository=repository,
        user_id=REQUEST_USER,
        query=RECALL_QUERY,
    )

    assert [result.entity_id for result in recovered] == [requested_chunk_id]
    assert all(result.entity_id != other_chunk_id for result in recovered)
    assert recovered[0].payload["user_id"] == REQUEST_USER
    assert recovered[0].payload["text"] == REQUEST_FACT
    assert recovered[0].validation_status == "sql_validated"

    sql_rerank_calls = [call for call in reranker.calls if call[1]]
    assert len(sql_rerank_calls) == 1
    rerank_query, rerank_candidates, enforce_min_score = sql_rerank_calls[0]
    assert rerank_query == RECALL_QUERY
    assert enforce_min_score is True
    assert [candidate.entity_id for candidate in rerank_candidates] == [
        requested_chunk_id
    ]
    assert all(
        candidate.entity_id != other_chunk_id for candidate in rerank_candidates
    )
    assert [candidate.payload["text"] for candidate in rerank_candidates] == [
        REQUEST_FACT
    ]


def test_partial_derived_index_hit_cannot_hide_a_better_committed_sql_fact() -> None:
    repository = _repository()
    cached_chunk_id = _seed_fact(
        repository,
        user_id=REQUEST_USER,
        text="The Alder workspace retains completed reports for twelve weeks.",
    )
    committed_chunk_id = _seed_fact(
        repository,
        user_id=REQUEST_USER,
        text="The Alder workspace rotates its signing material every nineteen hours.",
    )
    foreign_chunk_id = _seed_fact(
        repository,
        user_id=OTHER_USER,
        text="The Alder workspace rotates its signing material every seven hours.",
    )
    cached_result = RetrievalResult(
        entity_type="knowledge_chunk",
        entity_id=cached_chunk_id,
        source_store_evidence={"derived_cache": "partial"},
        rerank_score=0.0,
        confidence=0.0,
        validation_status="derived_candidate",
        payload={
            "user_id": REQUEST_USER,
            "is_deleted": False,
            "text": "This payload must be rehydrated from SQL.",
        },
    )
    reranker = _EntityScoreReranker(
        {
            cached_chunk_id: 0.41,
            committed_chunk_id: 0.97,
            foreign_chunk_id: 0.99,
        }
    )
    retriever = HybridRetriever(
        bm25=_StaticDerivedIndex([cached_result]),
        chroma=_StaticDerivedIndex([]),
        reranker=reranker,
    )

    recovered = retrieve_knowledge(
        retriever=retriever,
        repository=repository,
        user_id=REQUEST_USER,
        query="How often does the Alder workspace rotate its signing material?",
    )

    assert recovered
    assert recovered[0].entity_id == committed_chunk_id
    assert recovered[0].payload["user_id"] == REQUEST_USER
    assert recovered[0].validation_status == "sql_validated"
    assert cached_chunk_id in {result.entity_id for result in recovered}
    assert foreign_chunk_id not in {result.entity_id for result in recovered}
    assert any(
        committed_chunk_id in seen_ids for seen_ids in reranker.seen_entity_ids
    )
    assert all(
        foreign_chunk_id not in seen_ids for seen_ids in reranker.seen_entity_ids
    )


def test_completed_outbox_does_not_hide_a_missing_better_sql_fact() -> None:
    repository = _repository()
    chunk_ids = [
        _seed_fact(
            repository,
            user_id=REQUEST_USER,
            text=f"Stable record {position} has completed index fanout.",
        )
        for position in range(2)
    ]
    for chunk_id in chunk_ids:
        row = repository.connection.execute(
            """
            SELECT job_id FROM indexing_outbox
            WHERE entity_type = 'knowledge_chunk' AND entity_id = ?
              AND operation = 'upsert'
            ORDER BY created_at DESC LIMIT 1
            """,
            (chunk_id,),
        ).fetchone()
        assert row is not None
        repository.mark_outbox_job_completed(job_id=row["job_id"])

    cached_id, missing_id = chunk_ids
    derived = RetrievalResult(
        entity_type="knowledge_chunk",
        entity_id=cached_id,
        source_store_evidence={"derived_cache": "healthy"},
        rerank_score=0.0,
        confidence=0.0,
        validation_status="derived_candidate",
        payload={
            "user_id": REQUEST_USER,
            "is_deleted": False,
            "text": "Derived payload is rehydrated from SQL.",
        },
    )
    reranker = _EntityScoreReranker({cached_id: 0.41, missing_id: 0.96})
    recovered = retrieve_knowledge(
        retriever=HybridRetriever(
            bm25=_StaticDerivedIndex([derived]),
            chroma=_StaticDerivedIndex([]),
            reranker=reranker,
            sql_fallback_candidate_limit=3,
        ),
        repository=repository,
        user_id=REQUEST_USER,
        query="Which stable record was selected?",
    )

    assert recovered[0].entity_id == missing_id
    assert {result.entity_id for result in recovered} == {cached_id, missing_id}
    assert any(missing_id in seen_ids for seen_ids in reranker.seen_entity_ids)


def test_narrow_retriever_fails_closed_when_its_derived_coverage_is_partial() -> None:
    repository = _repository()
    cached_id = _seed_fact(
        repository,
        user_id=REQUEST_USER,
        text="The narrow cache contains this authoritative row.",
    )
    missing_id = _seed_fact(
        repository,
        user_id=REQUEST_USER,
        text="The narrow cache omitted this authoritative row.",
    )
    derived = RetrievalResult(
        entity_type="knowledge_chunk",
        entity_id=cached_id,
        source_store_evidence={"derived_cache": "partial"},
        rerank_score=0.91,
        confidence=0.91,
        validation_status="derived_candidate",
        payload={"user_id": REQUEST_USER, "is_deleted": False},
    )

    with pytest.raises(KnowledgeRetrievalUnavailableError):
        retrieve_knowledge(
            retriever=_NarrowDerivedRetriever([derived]),  # type: ignore[arg-type]
            repository=repository,
            user_id=REQUEST_USER,
            query="Which authoritative row answers this request?",
        )

    assert missing_id != cached_id


def test_complete_sql_snapshot_keeps_the_best_fact_across_rerank_batches() -> None:
    repository = _repository()
    chunk_ids = [
        _seed_fact(
            repository,
            user_id=REQUEST_USER,
            text=f"Bounded operational record {position:02d}.",
        )
        for position in range(16)
    ]
    best_id = chunk_ids[0]
    reranker = _EntityScoreReranker({best_id: 0.99})

    recovered = retrieve_knowledge(
        retriever=HybridRetriever(
            bm25=_UnavailableDerivedIndex(raises=False),
            chroma=_UnavailableDerivedIndex(raises=False),
            reranker=reranker,
            sql_fallback_candidate_limit=16,
        ),
        repository=repository,
        user_id=REQUEST_USER,
        query="Which bounded operational record best answers the request?",
    )

    assert recovered[0].entity_id == best_id
    sql_batches = [batch for batch in reranker.seen_entity_ids if batch]
    assert [len(batch) for batch in sql_batches] == [15, 1]
    assert best_id in sql_batches[1]


def test_sql_recovery_uses_repository_bounded_query_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _repository()
    owner_chunk_ids = [
        _seed_fact(
            repository,
            user_id=REQUEST_USER,
            text=f"Record {position} has a distinct immutable payload.",
        )
        for position in range(6)
    ]
    foreign_chunk_id = _seed_fact(
        repository,
        user_id=OTHER_USER,
        text="A separately owned record must never enter recovery candidates.",
    )
    with repository.transaction() as cursor:
        for position, chunk_id in enumerate(owner_chunk_ids, start=1):
            cursor.execute(
                "UPDATE knowledge_chunks SET created_at = ? WHERE chunk_id = ?",
                (f"2026-01-{position:02d}T00:00:00+00:00", chunk_id),
            )
        cursor.execute(
            "UPDATE knowledge_chunks SET is_deleted = 1 WHERE chunk_id = ?",
            (owner_chunk_ids[-1],),
        )
        cursor.execute(
            "UPDATE knowledge_chunks SET created_at = ? WHERE chunk_id = ?",
            ("2099-01-01T00:00:00+00:00", foreign_chunk_id),
        )

    bounded_method = getattr(
        repository,
        "list_knowledge_recovery_candidates",
        None,
    )
    assert callable(bounded_method), (
        "repositories must expose list_knowledge_recovery_candidates(user_id, limit)"
    )

    bounded_calls: list[tuple[str, int, str | None, str | None]] = []

    def record_bounded_call(
        *,
        user_id: str,
        limit: int,
        before_created_at: str | None = None,
        before_chunk_id: str | None = None,
    ) -> list[dict[str, Any]]:
        bounded_calls.append(
            (user_id, limit, before_created_at, before_chunk_id)
        )
        return bounded_method(
            user_id=user_id,
            limit=limit,
            before_created_at=before_created_at,
            before_chunk_id=before_chunk_id,
        )

    def reject_unbounded_listing(**_kwargs: Any) -> list[dict[str, Any]]:
        raise AssertionError("recovery must not load every knowledge fact")

    monkeypatch.setattr(
        repository,
        "list_knowledge_recovery_candidates",
        record_bounded_call,
    )
    monkeypatch.setattr(repository, "list_knowledge_facts", reject_unbounded_listing)
    sql_statements: list[str] = []
    repository.connection.set_trace_callback(sql_statements.append)
    try:
        retriever = HybridRetriever(
            bm25=_UnavailableDerivedIndex(raises=False),
            chroma=_UnavailableDerivedIndex(raises=False),
            reranker=_RecordingReranker(),
            sql_fallback_candidate_limit=5,
        )
        recovered = retrieve_knowledge(
            retriever=retriever,
            repository=repository,
            user_id=REQUEST_USER,
            query="Which records are currently available?",
        )
    finally:
        repository.connection.set_trace_callback(None)

    expected_ids = [
        owner_chunk_ids[4],
        owner_chunk_ids[3],
        owner_chunk_ids[2],
        owner_chunk_ids[1],
        owner_chunk_ids[0],
    ]
    assert bounded_calls == [(REQUEST_USER, 6, None, None)]
    assert [result.entity_id for result in recovered] == expected_ids
    assert all(result.payload["user_id"] == REQUEST_USER for result in recovered)
    assert foreign_chunk_id not in {result.entity_id for result in recovered}
    normalized_statements = [
        " ".join(statement.upper().split()) for statement in sql_statements
    ]
    assert any(
        "FROM KNOWLEDGE_CHUNKS" in statement
        and "ORDER BY C.CREATED_AT DESC" in statement
        and "LIMIT 6" in statement
        for statement in normalized_statements
    )


class _EvidenceSelectingAnswerLLM:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def generate_json(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(dict(kwargs))
        assert kwargs["task"] is LLMTask.ANSWER
        prompt = json.loads(
            str(kwargs["user_prompt"]).removeprefix("Runtime context:\n")
        )
        records = prompt["extra"]["approved_knowledge_records"]
        assert len(records) == 1
        record = records[0]
        payload = {
            "answer_text": record["text"],
            "evidence_references": [
                {
                    "candidate_key": record["candidate_key"],
                    "verbatim_support": record["text"],
                }
            ],
        }
        kwargs["invariant_validator"](payload)
        return payload


def test_saved_fact_survives_unrelated_hop_and_empty_derived_indexes() -> None:
    repository = _repository()
    saved = repository.transactional_knowledge_actions(
        user_id=REQUEST_USER,
        topic_title="Durable preferences",
        raw_user_query="Persist this calibration preference.",
        rewritten_user_query="Persist this calibration preference.",
        response_text="The durable preference was saved.",
        actions=[
            ValidatedKnowledgeAction(
                action=KnowledgeAction.ADD,
                validation_result=ActionValidationResult.EXECUTE,
                knowledge_text=REQUEST_FACT,
                topic_title="Durable preferences",
            )
        ],
    )
    assert saved.committed is True
    _seed_fact(repository, user_id=OTHER_USER, text=OTHER_FACT)
    with repository.transaction() as cursor:
        topic_id = repository.create_topic(
            cursor,
            user_id=REQUEST_USER,
            title="Unrelated exchange",
        )
        repository.append_conversation_hop(
            cursor,
            topic_id=topic_id,
            user_id=REQUEST_USER,
            intent=Intent.GENERAL_RESPONSE.value,
            raw_user_query="Discuss a separate subject.",
            rewritten_user_query="Discuss a separate subject.",
            raw_response="A separate response.",
            response_type=ResponseType.NORMAL.value,
        )

    answer_llm = _EvidenceSelectingAnswerLLM()
    branch = GeneralResponseBranch(
        retriever=HybridRetriever(
            bm25=_UnavailableDerivedIndex(raises=True),
            chroma=_UnavailableDerivedIndex(raises=True),
            reranker=_RecordingReranker(),
        ),
        config=SimpleNamespace(
            retrieval=SimpleNamespace(
                general_response_reminder_statuses=("scheduled", "notified"),
                general_response_reminder_limit=4,
            )
        ),
        context_filter=HardRuleContextFilter(
            allowed_reminder_statuses=("scheduled", "notified"),
            reminder_approved_max_items=4,
            reminder_min_confidence=0.0,
        ),
        llm=answer_llm,  # type: ignore[arg-type]
    )
    result = branch.execute(
        PipelineContext(
            request=ChatRequest(user_id=REQUEST_USER, raw_query=RECALL_QUERY),
            rewritten_query=RECALL_QUERY,
            last_qa_state=None,
            conversation_results=[],
            intent=Intent.GENERAL_RESPONSE,
        ),
        repository,
    )

    assert result.response_type is ResponseType.NORMAL
    assert result.normal_response_text == REQUEST_FACT
    assert len(answer_llm.calls) == 1
    assert OTHER_FACT not in result.normal_response_text


class _FailingWriteIndex:
    def search(self, **_kwargs: Any) -> list[RetrievalResult]:
        return []

    def upsert(self, **_kwargs: Any) -> None:
        raise RuntimeError("derived index rejected the write")

    def delete(self, *, entity_id: str) -> None:
        del entity_id
        raise RuntimeError("derived index rejected the delete")

    def clear(self) -> None:
        return None


class _PipelineRetriever:
    def __init__(self) -> None:
        self.bm25 = _FailingWriteIndex()
        self.chroma = _FailingWriteIndex()

    def retrieve_conversation(self, **_kwargs: Any) -> list[RetrievalResult]:
        return []


class _Store:
    def get(self, _user_id: str) -> None:
        return None

    def save(self, _user_id: str, _state: Any) -> None:
        return None


class _StaticBranch:
    def execute(
        self,
        _context: PipelineContext,
        _repository: SQLiteRepository,
    ) -> BranchResult:
        return BranchResult(
            response_type=ResponseType.KNOWLEDGE_ACTION,
            normal_response_text="The durable update was accepted.",
            knowledge_operation_results=[
                RepositoryActionResult(
                    action_id="committed-action",
                    action_type="add",
                    status="committed",
                    domain_entity_type="knowledge_chunk",
                    domain_entity_id="durable-entity",
                    user_safe_summary="The durable update was accepted.",
                )
            ],
        )


class _Classifier:
    def classify(self, *_args: Any, **_kwargs: Any) -> Intent:
        return Intent.KNOWLEDGE_FACTS


class _PlatformSelector:
    def select(self, _response: Any, _request: ChatRequest) -> dict[str, Any]:
        return {"delivery": {"channel": "none", "status": "not_requested"}}


def test_pipeline_contains_request_scoped_index_failure_after_committed_branch_result() -> None:
    repository = _repository()
    pipeline = AssistantPipeline(
        config=SimpleNamespace(
            context_filter=SimpleNamespace(
                conversation_retrieval_after_last_qa_enabled=True,
                conversation_retrieval_before_intent_enabled=False,
            ),
            outbox=OutboxConfig(
                max_attempts=2,
                batch_size=16,
                retry_backoff_seconds=0,
                processing_timeout_seconds=180,
            ),
        ),
        last_qa_store=_Store(),
        query_rewriter=SimpleNamespace(rewrite=lambda query: query),
        last_qa_resolver=SimpleNamespace(
            resolve=lambda *_args: LastQAResolution(
                path=LastQAPath.NO_LAST_QA,
                rewritten_query="Persist this operational note.",
                state=None,
                did_merge_query=False,
                skip_broad_retrieval=True,
            )
        ),
        retriever=_PipelineRetriever(),  # type: ignore[arg-type]
        context_filter=SimpleNamespace(),
        classifier=_Classifier(),  # type: ignore[arg-type]
        router=BranchRouter({Intent.KNOWLEDGE_FACTS: _StaticBranch()}),
        bundler=ResponseBundler(),
        platform_selector=_PlatformSelector(),  # type: ignore[arg-type]
        chat_output=ChatOutput(),
    )

    response = pipeline.handle(
        ChatRequest(
            user_id=REQUEST_USER,
            raw_query="Persist this operational note.",
        ),
        repository,
    )

    assert response.final_chat_text == "The durable update was accepted."
    assert response.actions_committed == [
        {
            "action_type": "add",
            "domain_entity_type": "knowledge_chunk",
            "domain_entity_id": "durable-entity",
            "summary": "The durable update was accepted.",
        }
    ]
    assert response.warnings == ["request_scoped_index_sync_failed"]
    assert response.conversation_hop_id is not None
    assert repository.table_count("conversation_hops") == 1
    assert repository.connection.execute(
        "SELECT status FROM indexing_outbox"
    ).fetchone()["status"] == "failed"
