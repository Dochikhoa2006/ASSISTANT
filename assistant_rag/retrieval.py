"""Derived-cache retrieval strategies."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Protocol

from .contracts import RetrievalResult


class SearchIndex(Protocol):
    def search(self, *, user_id: str, query: str, limit: int) -> list[RetrievalResult]:
        ...

    def upsert(
        self,
        *,
        user_id: str,
        entity_type: str,
        entity_id: str,
        text: str,
        metadata: dict[str, str | int | float | bool] | None = None,
    ) -> None:
        ...

    def delete(self, *, entity_id: str) -> None:
        ...

    def clear(self) -> None:
        ...


class Reranker(Protocol):
    def rerank(self, query: str, results: Iterable[RetrievalResult]) -> list[RetrievalResult]:
        ...


@dataclass
class HybridRetriever:
    bm25: SearchIndex
    chroma: SearchIndex
    reranker: Reranker
    rrf_k: int = 60
    lexical_weight: float = 1.0
    semantic_weight: float = 1.0
    rerank_candidate_limit: int = 32
    bm25_top_k: int = 30
    chroma_top_k: int = 30

    def retrieve_conversation(
        self, *, user_id: str, query: str, limit: int, min_confidence: float
    ) -> list[RetrievalResult]:
        return self._retrieve(
            user_id=user_id,
            query=query,
            limit=limit,
            min_confidence=min_confidence,
            allowed_entity_type="conversation_hop",
        )

    def retrieve_knowledge(
        self, *, user_id: str, query: str, limit: int, min_confidence: float
    ) -> list[RetrievalResult]:
        return self._retrieve(
            user_id=user_id,
            query=query,
            limit=limit,
            min_confidence=min_confidence,
            allowed_entity_type="knowledge_chunk",
        )

    def _retrieve(
        self,
        *,
        user_id: str,
        query: str,
        limit: int,
        min_confidence: float,
        allowed_entity_type: str,
    ) -> list[RetrievalResult]:
        merged = self._rrf_merge(
            self.bm25.search(user_id=user_id, query=query, limit=self.bm25_top_k),
            self.chroma.search(user_id=user_id, query=query, limit=self.chroma_top_k),
        )
        reranked = self.reranker.rerank(query, merged[: self.rerank_candidate_limit])
        return [
            item
            for item in reranked
            if item.entity_type == allowed_entity_type and item.confidence >= min_confidence
        ][:limit]

    def _rrf_merge(
        self, lexical: list[RetrievalResult], semantic: list[RetrievalResult]
    ) -> list[RetrievalResult]:
        by_id: dict[str, RetrievalResult] = {}
        score_by_id: defaultdict[str, float] = defaultdict(float)
        for rank, result in enumerate(lexical, start=1):
            by_id[result.entity_id] = result
            score_by_id[result.entity_id] += self.lexical_weight / (self.rrf_k + rank)
        for rank, result in enumerate(semantic, start=1):
            by_id[result.entity_id] = result
            score_by_id[result.entity_id] += self.semantic_weight / (self.rrf_k + rank)
        merged = []
        for entity_id, result in by_id.items():
            merged.append(
                RetrievalResult(
                    entity_type=result.entity_type,
                    entity_id=result.entity_id,
                    source_store_evidence=result.source_store_evidence,
                    rerank_score=score_by_id[entity_id],
                    confidence=max(result.confidence, score_by_id[entity_id]),
                    validation_status=result.validation_status,
                    payload=result.payload,
                )
            )
        return sorted(merged, key=lambda item: item.rerank_score, reverse=True)
