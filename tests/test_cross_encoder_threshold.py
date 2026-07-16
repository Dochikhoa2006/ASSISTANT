from __future__ import annotations

from dataclasses import replace
from typing import Iterable

import pytest

from assistant_rag.chat_history import last_qa_chat_history, select_chat_history
from assistant_rag.contracts import (
    ApprovedConversationContext,
    LastQAState,
    ResponseType,
    RetrievalResult,
)
from assistant_rag.reranking import (
    HTTPReranker,
    SentenceTransformerCrossEncoderReranker,
)
from assistant_rag.retrieval import HybridRetriever
from assistant_rag.settings import RerankerSettings


def _result(
    entity_id: str,
    *,
    entity_type: str = "knowledge_chunk",
    source_confidence: float = 0.99,
) -> RetrievalResult:
    return RetrievalResult(
        entity_type=entity_type,
        entity_id=entity_id,
        source_store_evidence={"test": True},
        rerank_score=0.01,
        confidence=source_confidence,
        validation_status="candidate",
        payload={
            "user_id": "user-1",
            "text": f"document {entity_id}",
            "is_deleted": False,
        },
    )


def _local_reranker(
    scores: list[float],
    *,
    min_score: float = 0.50,
) -> SentenceTransformerCrossEncoderReranker:
    class _Model:
        def predict(
            self,
            pairs: list[tuple[str, str]],
            *,
            batch_size: int,
        ) -> list[float]:
            del batch_size
            assert len(pairs) == len(scores)
            return list(scores)

    # Bypass __post_init__ so this focused unit test never imports or downloads
    # a sentence-transformers model.
    reranker = object.__new__(SentenceTransformerCrossEncoderReranker)
    reranker.settings = replace(RerankerSettings(), min_score=min_score)
    reranker.remote = None
    reranker.model = _Model()
    return reranker


def _assert_thresholded_cross_encoder_results(
    results: list[RetrievalResult],
) -> None:
    assert [result.entity_id for result in results] == ["high", "boundary"]
    assert [result.rerank_score for result in results] == pytest.approx([0.91, 0.50])
    assert len(results) == 2
    assert all(result.rerank_score >= 0.50 for result in results)
    # The confidence exposed after reranking is the cross-encoder confidence,
    # not a higher lexical/vector confidence inherited from the source index.
    assert all(result.confidence == result.rerank_score for result in results)


def test_local_cross_encoder_threshold_is_inclusive_without_backfill() -> None:
    candidates = [
        _result("high"),
        _result("boundary"),
        _result("just-below"),
        _result("low"),
        _result("lowest"),
    ]
    reranker = _local_reranker([0.91, 0.50, 0.499, 0.20, -0.10])

    results = reranker.rerank("query", candidates)

    _assert_thresholded_cross_encoder_results(results)


def test_http_cross_encoder_threshold_is_inclusive_without_backfill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = [
        _result("high"),
        _result("boundary"),
        _result("just-below"),
        _result("low"),
        _result("lowest"),
    ]
    settings = replace(
        RerankerSettings(),
        endpoint_url="http://reranker.invalid",
        min_score=0.50,
    )
    reranker = HTTPReranker(settings)
    monkeypatch.setattr(
        reranker,
        "_score",
        lambda _body: [0.91, 0.50, 0.499, 0.20, -0.10],
    )

    results = reranker.rerank("query", candidates)

    _assert_thresholded_cross_encoder_results(results)


class _StaticIndex:
    def __init__(self, results: list[RetrievalResult]) -> None:
        self.results = results

    def search(
        self,
        *,
        user_id: str,
        query: str,
        entity_type: str,
        limit: int,
    ) -> list[RetrievalResult]:
        del query
        assert user_id == "user-1"
        assert all(result.entity_type == entity_type for result in self.results)
        return list(self.results[:limit])

    def upsert(self, **_kwargs: object) -> None:
        return None

    def delete(self, *, entity_id: str) -> None:
        del entity_id

    def clear(self) -> None:
        return None


class _UnfilteredReranker:
    """Return scored candidates so HybridRetriever's final guard is exercised."""

    def __init__(self, scores_by_id: dict[str, float]) -> None:
        self.scores_by_id = scores_by_id

    def rerank(
        self,
        query: str,
        results: Iterable[RetrievalResult],
        *,
        enforce_min_score: bool = True,
    ) -> list[RetrievalResult]:
        del query
        del enforce_min_score
        scored = [
            replace(result, rerank_score=self.scores_by_id[result.entity_id])
            for result in results
        ]
        return sorted(scored, key=lambda result: result.rerank_score, reverse=True)


@pytest.mark.parametrize(
    ("retrieval_method", "entity_type"),
    (
        ("retrieve_knowledge", "knowledge_chunk"),
        ("retrieve_conversation", "conversation_hop"),
    ),
)
def test_hybrid_retriever_uses_shared_threshold_guard_and_top_five(
    retrieval_method: str,
    entity_type: str,
) -> None:
    candidates = [
        _result(f"item-{index}", entity_type=entity_type)
        for index in range(8)
    ]
    scores = {
        "item-0": 0.99,
        "item-1": 0.90,
        "item-2": 0.80,
        "item-3": 0.70,
        "item-4": 0.60,
        "item-5": 0.50,
        # Both rejected candidates deliberately retain source confidence 0.99;
        # the final guard must use rerank_score, never source confidence.
        "item-6": 0.499,
        "item-7": 0.10,
    }
    retriever = HybridRetriever(
        bm25=_StaticIndex(candidates),
        chroma=_StaticIndex(candidates),
        reranker=_UnfilteredReranker(scores),
        rerank_min_score=0.50,
    )

    results = getattr(retriever, retrieval_method)(user_id="user-1", query="query")

    assert [result.entity_id for result in results] == [
        "item-0",
        "item-1",
        "item-2",
        "item-3",
        "item-4",
    ]
    assert len(results) == 5
    assert all(result.entity_type == entity_type for result in results)
    assert all(result.rerank_score >= 0.50 for result in results)
    assert all(result.confidence == result.rerank_score for result in results)


def _empty_approved_conversation_context() -> ApprovedConversationContext:
    return ApprovedConversationContext(
        approved_conversation_history=[],
        human_supporting_questions=[],
        reminder_supporting_questions=[],
        clarification_question_context=None,
        extracted_expected_response_types=[],
        conversation_retrieval_ran=True,
        conversation_context_status="empty",
        approved_conversation_count=0,
    )


def test_no_conversation_survivor_makes_chat_history_authoritatively_empty() -> None:
    candidates = [
        _result("hop-1", entity_type="conversation_hop"),
        _result("hop-2", entity_type="conversation_hop"),
    ]
    retriever = HybridRetriever(
        bm25=_StaticIndex(candidates),
        chroma=_StaticIndex(candidates),
        reranker=_UnfilteredReranker({"hop-1": 0.499, "hop-2": 0.20}),
        rerank_min_score=0.50,
    )
    last_qa_state = LastQAState(
        last_user_query="Earlier question",
        last_response="Earlier answer",
        response_type=ResponseType.NORMAL,
    )

    conversation_results = retriever.retrieve_conversation(
        user_id="user-1",
        query="unrelated query",
    )
    history = select_chat_history(
        conversation_retrieval=True,
        approved_conversation_context=_empty_approved_conversation_context(),
        last_qa_state=last_qa_state,
    )

    assert conversation_results == []
    assert last_qa_chat_history(last_qa_state) != []
    assert history == []


def test_conversation_retrieval_uses_its_stricter_independent_score_floor() -> None:
    knowledge_candidates = [
        _result("knowledge-boundary", entity_type="knowledge_chunk")
    ]
    conversation_candidates = [
        _result("conversation-below", entity_type="conversation_hop")
    ]
    scores = {
        "knowledge-boundary": 0.49,
        "conversation-below": 0.49,
    }
    knowledge_retriever = HybridRetriever(
        bm25=_StaticIndex(knowledge_candidates),
        chroma=_StaticIndex(knowledge_candidates),
        reranker=_UnfilteredReranker(scores),
        rerank_min_score=0.30,
        conversation_min_confidence_score=0.50,
    )
    conversation_retriever = HybridRetriever(
        bm25=_StaticIndex(conversation_candidates),
        chroma=_StaticIndex(conversation_candidates),
        reranker=_UnfilteredReranker(scores),
        rerank_min_score=0.30,
        conversation_min_confidence_score=0.50,
    )

    assert [
        result.entity_id
        for result in knowledge_retriever.retrieve_knowledge(
            user_id="user-1", query="query"
        )
    ] == ["knowledge-boundary"]
    assert conversation_retriever.retrieve_conversation(
        user_id="user-1", query="query"
    ) == []


def test_local_and_http_rerankers_can_return_below_floor_scores_only_when_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = [_result("high"), _result("low")]
    local = _local_reranker([0.90, 0.05])
    assert [
        item.entity_id
        for item in local.rerank(
            "query",
            candidates,
            enforce_min_score=False,
        )
    ] == ["high", "low"]

    settings = replace(
        RerankerSettings(),
        endpoint_url="http://reranker.invalid",
        min_score=0.50,
    )
    remote = HTTPReranker(settings)
    monkeypatch.setattr(remote, "_score", lambda _body: [0.90, 0.05])
    assert [
        item.entity_id
        for item in remote.rerank(
            "query",
            candidates,
            enforce_min_score=False,
        )
    ] == ["high", "low"]


def test_only_knowledge_retrieval_can_bypass_the_shared_score_floor() -> None:
    knowledge = [
        _result("knowledge-high", entity_type="knowledge_chunk"),
        _result("knowledge-low", entity_type="knowledge_chunk"),
    ]
    scores = {"knowledge-high": 0.90, "knowledge-low": 0.05}
    retriever = HybridRetriever(
        bm25=_StaticIndex(knowledge),
        chroma=_StaticIndex(knowledge),
        reranker=_UnfilteredReranker(scores),
        rerank_min_score=0.50,
        conversation_min_confidence_score=0.50,
    )

    assert [
        item.entity_id
        for item in retriever.retrieve_knowledge(
            user_id="user-1",
            query="query",
        )
    ] == ["knowledge-high"]
    assert [
        item.entity_id
        for item in retriever.retrieve_knowledge(
            user_id="user-1",
            query="query",
            enforce_min_score=False,
        )
    ] == ["knowledge-high", "knowledge-low"]

    conversation = [
        _result("conversation-low", entity_type="conversation_hop")
    ]
    conversation_retriever = HybridRetriever(
        bm25=_StaticIndex(conversation),
        chroma=_StaticIndex(conversation),
        reranker=_UnfilteredReranker({"conversation-low": 0.05}),
        rerank_min_score=0.50,
        conversation_min_confidence_score=0.50,
    )
    assert conversation_retriever.retrieve_conversation(
        user_id="user-1",
        query="query",
    ) == []
