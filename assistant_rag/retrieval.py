"""Derived-cache retrieval strategies."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Protocol

from .contracts import RetrievalResult
from .retrieval_policy import RETRIEVAL_PIPELINE_POLICY, RetrievalPipelinePolicy


class SearchIndex(Protocol):
    def search(
        self, *, user_id: str, query: str, entity_type: str, limit: int
    ) -> list[RetrievalResult]:
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
    rerank_candidate_limit: int = RETRIEVAL_PIPELINE_POLICY.rrf_top_k
    bm25_top_k: int = RETRIEVAL_PIPELINE_POLICY.source_top_k
    chroma_top_k: int = RETRIEVAL_PIPELINE_POLICY.source_top_k

    def __post_init__(self) -> None:
        RetrievalPipelinePolicy(
            source_top_k=self.bm25_top_k,
            rrf_top_k=self.rerank_candidate_limit,
            final_top_k=RETRIEVAL_PIPELINE_POLICY.final_top_k,
        ).validate()
        if self.chroma_top_k != self.bm25_top_k:
            raise ValueError("OpenSearch and ChromaDB candidate limits must be identical")

    def retrieve_conversation(
        self, *, user_id: str, query: str
    ) -> list[RetrievalResult]:
        return self._retrieve(
            user_id=user_id,
            query=query,
            allowed_entity_type="conversation_hop",
        )

    def retrieve_knowledge(
        self, *, user_id: str, query: str
    ) -> list[RetrievalResult]:
        return self._retrieve(
            user_id=user_id,
            query=query,
            allowed_entity_type="knowledge_chunk",
        )

    def _retrieve(
        self,
        *,
        user_id: str,
        query: str,
        allowed_entity_type: str,
    ) -> list[RetrievalResult]:
        merged = self._rrf_merge(
            self.bm25.search(
                user_id=user_id,
                query=query,
                entity_type=allowed_entity_type,
                limit=self.bm25_top_k,
            ),
            self.chroma.search(
                user_id=user_id,
                query=query,
                entity_type=allowed_entity_type,
                limit=self.chroma_top_k,
            ),
        )
        eligible = self._filter_hard_rules(
            user_id=user_id,
            results=merged,
        )
        reranked = self.reranker.rerank(query, eligible[: self.rerank_candidate_limit])
        return self._filter_hard_rules(
            user_id=user_id,
            results=reranked,
        )[: RETRIEVAL_PIPELINE_POLICY.final_top_k]

    @staticmethod
    def _filter_hard_rules(
        *, user_id: str, results: Iterable[RetrievalResult]
    ) -> list[RetrievalResult]:
        """Keep only the retrieval invariants shared by both entity types."""
        approved: list[RetrievalResult] = []
        seen_entity_ids: set[str] = set()
        for result in results:
            payload = result.payload
            if payload.get("user_id") != user_id:
                continue
            if bool(payload.get("is_deleted", False)):
                continue
            if result.entity_id in seen_entity_ids:
                continue
            seen_entity_ids.add(result.entity_id)
            approved.append(result)
        return approved

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
            source_store_evidence = {
                **{
                    key: value
                    for candidate in lexical + semantic
                    if candidate.entity_id == entity_id
                    for key, value in candidate.source_store_evidence.items()
                }
            }
            merged.append(
                RetrievalResult(
                    entity_type=result.entity_type,
                    entity_id=result.entity_id,
                    source_store_evidence=source_store_evidence,
                    rerank_score=score_by_id[entity_id],
                    confidence=max(result.confidence, score_by_id[entity_id]),
                    validation_status=result.validation_status,
                    payload=result.payload,
                )
            )
        return sorted(merged, key=lambda item: item.rerank_score, reverse=True)
