from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import json
from types import SimpleNamespace
from typing import Any, Iterable, Iterator

import pytest

import assistant_rag.branches as branches_module
from assistant_rag.branches import GeneralResponseBranch
from assistant_rag.canonical_retrieval import (
    KnowledgeRetrievalUnavailableError,
    retrieve_knowledge,
)
from assistant_rag.config import GeneralPurposeConfig
from assistant_rag.context_filter import ApprovedContext
from assistant_rag.contracts import (
    ChatRequest,
    ContentComposerResult,
    HopWrite,
    Intent,
    PipelineContext,
    ResponseType,
    RetrievalResult,
)
from assistant_rag.database import SQLiteRepository
from assistant_rag.llm import LLMTask
from assistant_rag.retrieval import HybridRetriever


OWNER_ID = "adversarial-context-owner"
EVIDENCE_TEXT = "The Helix workspace uses the umber deployment lane."
EVIDENCE_KEY = "owned-helix-record"


class _Index:
    def __init__(self, results: Iterable[RetrievalResult] = ()) -> None:
        self.results = list(results)

    def search(self, **_kwargs: Any) -> list[RetrievalResult]:
        return list(self.results)

    def upsert(self, **_kwargs: Any) -> None:
        return None

    def delete(self, *, entity_id: str) -> None:
        del entity_id

    def clear(self) -> None:
        return None


class _ScoringReranker:
    def __init__(self, scores: dict[str, float] | None = None) -> None:
        self.scores = scores or {}
        self.calls: list[list[str]] = []

    def rerank(
        self,
        _query: str,
        results: Iterable[RetrievalResult],
        *,
        enforce_min_score: bool = True,
    ) -> list[RetrievalResult]:
        del enforce_min_score
        candidates = list(results)
        self.calls.append([candidate.entity_id for candidate in candidates])
        return sorted(
            (
                replace(
                    candidate,
                    rerank_score=self.scores.get(candidate.entity_id, 0.51),
                    confidence=self.scores.get(candidate.entity_id, 0.51),
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


def _seed_fact(repository: SQLiteRepository, text: str) -> str:
    with repository.transaction() as cursor:
        _topic_id, chunk_id, _job_id = repository.add_knowledge_chunk(
            cursor,
            user_id=OWNER_ID,
            title="Adversarial records",
            text=text,
        )
    return chunk_id


def _complete_latest_upsert(repository: SQLiteRepository, chunk_id: str) -> None:
    row = repository.connection.execute(
        """
        SELECT job_id FROM indexing_outbox
        WHERE entity_type = 'knowledge_chunk'
          AND entity_id = ?
          AND operation = 'upsert'
        ORDER BY created_at DESC, job_id DESC
        LIMIT 1
        """,
        (chunk_id,),
    ).fetchone()
    assert row is not None
    repository.mark_outbox_job_completed(job_id=str(row["job_id"]))


def test_completed_outbox_status_cannot_conceal_a_missing_derived_index_row() -> None:
    repository = _repository()
    cached_id = _seed_fact(
        repository,
        "The Helix workspace archives releases for eleven weeks.",
    )
    missing_id = _seed_fact(repository, EVIDENCE_TEXT)
    _complete_latest_upsert(repository, cached_id)
    _complete_latest_upsert(repository, missing_id)

    cached_candidate = RetrievalResult(
        entity_type="knowledge_chunk",
        entity_id=cached_id,
        source_store_evidence={"derived_cache": "present"},
        rerank_score=0.0,
        confidence=0.0,
        validation_status="derived_candidate",
        payload={
            "user_id": OWNER_ID,
            "is_deleted": False,
            "text": "Derived payload must be replaced by SQL hydration.",
        },
    )
    reranker = _ScoringReranker({cached_id: 0.22, missing_id: 0.98})

    recovered = retrieve_knowledge(
        retriever=HybridRetriever(
            bm25=_Index([cached_candidate]),
            chroma=_Index(),
            reranker=reranker,
            sql_fallback_candidate_limit=8,
        ),
        repository=repository,
        user_id=OWNER_ID,
        query="Which deployment lane is configured for the Helix workspace?",
    )

    assert recovered[0].entity_id == missing_id
    assert recovered[0].payload["text"] == EVIDENCE_TEXT
    assert recovered[0].validation_status == "sql_validated"
    assert any(missing_id in call for call in reranker.calls)


def test_sql_recovery_budget_overflow_fails_before_reranking_sql_history() -> None:
    repository = _repository()
    budget = 7
    for position in range(budget + 9):
        _seed_fact(
            repository,
            f"Immutable record {position:02d} belongs to the requesting owner.",
        )
    reranker = _ScoringReranker()

    with pytest.raises(KnowledgeRetrievalUnavailableError):
        retrieve_knowledge(
            retriever=HybridRetriever(
                bm25=_Index(),
                chroma=_Index(),
                reranker=reranker,
                sql_fallback_candidate_limit=budget,
            ),
            repository=repository,
            user_id=OWNER_ID,
            query="Return the record relevant to this request.",
        )

    # Hybrid derived retrieval may invoke the ranker with no candidates. SQL
    # overflow must be discovered before any non-empty recovery batch is sent.
    assert not any(call for call in reranker.calls)


class _ApprovedFilter:
    def filter(self, **_kwargs: Any) -> ApprovedContext:
        record = {
            "candidate_key": EVIDENCE_KEY,
            "text": EVIDENCE_TEXT,
            "version": 2,
            "rerank_score": 0.97,
            "validation_status": "sql_validated",
        }
        return ApprovedContext(
            knowledge_evidence=[EVIDENCE_TEXT],
            reminder_context=[],
            approved_conversation_history=[],
            rejected_knowledge_ids=[],
            rejected_reminder_ids=[],
            rejected_conversation_ids=[],
            knowledge_records=[record],
        )


class _RecordingRepository:
    @contextmanager
    def transaction(self) -> Iterator[object]:
        yield object()

    def ensure_topic(self, _cursor: object, *, user_id: str, title: str) -> str:
        assert user_id == OWNER_ID
        assert title
        return "adversarial-topic"

    def append_conversation_hop(
        self,
        _cursor: object,
        **kwargs: Any,
    ) -> HopWrite:
        return HopWrite(
            topic_id=str(kwargs["topic_id"]),
            hop_id="adversarial-hop",
            previous_hop_id=kwargs.get("parent_hop_id"),
            outbox_job_id="adversarial-outbox",
        )


class _RecoveryLLM:
    def __init__(self) -> None:
        self.tasks: list[LLMTask] = []

    def generate_json(self, **kwargs: Any) -> dict[str, Any]:
        self.tasks.append(kwargs["task"])
        payload = {
            "answer_text": EVIDENCE_TEXT,
            "evidence_references": [
                {
                    "candidate_key": EVIDENCE_KEY,
                    "verbatim_support": "umber deployment lane",
                }
            ],
        }
        kwargs["invariant_validator"](payload)
        return payload


class _FalseNegativeDependencyLLM:
    def __init__(self) -> None:
        self.tasks: list[LLMTask] = []

    def generate_json(self, **kwargs: Any) -> dict[str, Any]:
        self.tasks.append(kwargs["task"])
        properties = dict(kwargs["schema"].get("properties") or {})
        if "requires_unavailable_context" in properties:
            # The primary classifier is confidently wrong.
            payload = {
                "requires_unavailable_context": False,
                "confidence": 0.99,
                "reason_summary": "Primary semantic decision fixture.",
            }
        elif "answerable_without_unavailable_context" in properties:
            # The mandatory ANSWER stage detects that its own proposed prose
            # cannot be certified without the missing store. The branch must
            # discard answer_text even though this payload is high-confidence.
            payload = {
                "answerable_without_unavailable_context": False,
                "answer_text": "The owner's unverified deployment lane is silver.",
                "confidence": 0.99,
                "reason_summary": "Cross-model answer certificate fixture.",
            }
        else:  # pragma: no cover - makes schema drift fail explicitly
            raise AssertionError(f"Unexpected structured schema: {properties}")
        kwargs["invariant_validator"](payload)
        return payload

    def chat(self, **kwargs: Any) -> str:
        raise AssertionError(
            f"Outage response bypassed its structured certificate: {kwargs}"
        )


def _context(query: str) -> PipelineContext:
    return PipelineContext(
        request=ChatRequest(user_id=OWNER_ID, raw_query=query),
        rewritten_query=query,
        last_qa_state=None,
        conversation_results=[],
        intent=Intent.GENERAL_RESPONSE,
    )


def test_confident_dependency_false_negative_cannot_leak_ungrounded_personal_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        branches_module,
        "retrieve_knowledge",
        lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError("synthetic durable-store outage")
        ),
    )
    monkeypatch.setattr(
        branches_module,
        "retrieve_reminder_candidates",
        lambda **_kwargs: [],
    )

    class _EmptyFilter:
        def filter(self, **_kwargs: Any) -> ApprovedContext:
            return ApprovedContext(
                knowledge_evidence=[],
                reminder_context=[],
                approved_conversation_history=[],
                rejected_knowledge_ids=[],
                rejected_reminder_ids=[],
                rejected_conversation_ids=[],
            )

    llm = _FalseNegativeDependencyLLM()
    ungrounded_claim = "The owner's unverified deployment lane is silver."
    result = GeneralResponseBranch(
        retriever=object(),  # type: ignore[arg-type]
        config=SimpleNamespace(
            retrieval=SimpleNamespace(
                general_response_reminder_statuses=("scheduled", "notified"),
                general_response_reminder_limit=4,
            )
        ),
        context_filter=_EmptyFilter(),  # type: ignore[arg-type]
        llm=llm,  # type: ignore[arg-type]
    ).execute(
        _context("Return the owner's configured deployment lane."),
        _RecordingRepository(),  # type: ignore[arg-type]
    )

    assert result.response_type is ResponseType.NORMAL
    assert result.normal_response_text != ungrounded_claim
    assert ungrounded_claim not in result.normal_response_text
    assert "personal_context_safe_retry" in result.warnings
    assert llm.tasks == [
        LLMTask.RETRIEVAL_VALIDATION,
        LLMTask.ANSWER,
    ]


def test_valid_answer_recovery_discards_all_untrusted_composer_prose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(branches_module, "retrieve_knowledge", lambda **_kwargs: [])
    monkeypatch.setattr(
        branches_module,
        "retrieve_reminder_candidates",
        lambda **_kwargs: [],
    )

    rejected_answer = "The record cannot be accessed."
    unsupported_claim = "The account also has an unrecorded violet access tier."

    class _SmugglingComposer:
        def compose(self, *_args: Any, **_kwargs: Any) -> ContentComposerResult:
            return ContentComposerResult(
                final_response_text=(
                    f"{rejected_answer}\n\n{unsupported_claim}"
                ),
                answer_response_text=rejected_answer,
                tool_trace_summary=json.dumps(
                    {
                        "stage_outcomes": {
                            "answer_generation": {"succeeded": True}
                        }
                    }
                ),
                used_tool_names=("answer_generation",),
                confidence=1.0,
                fallback_used=False,
                reason_summary="adversarial composer fixture",
                content_warnings=(),
            )

    llm = _RecoveryLLM()
    result = GeneralResponseBranch(
        retriever=object(),  # type: ignore[arg-type]
        config=SimpleNamespace(
            retrieval=SimpleNamespace(
                general_response_reminder_statuses=("scheduled", "notified"),
                general_response_reminder_limit=4,
            )
        ),
        context_filter=_ApprovedFilter(),  # type: ignore[arg-type]
        content_composer=_SmugglingComposer(),
        general_purpose_config=GeneralPurposeConfig(),
        llm=llm,  # type: ignore[arg-type]
    ).execute(
        _context("Which deployment lane is configured for Helix?"),
        _RecordingRepository(),  # type: ignore[arg-type]
    )

    assert result.response_type is ResponseType.NORMAL
    assert result.normal_response_text == EVIDENCE_TEXT
    assert unsupported_claim not in result.normal_response_text
    assert llm.tasks == [LLMTask.ANSWER]
