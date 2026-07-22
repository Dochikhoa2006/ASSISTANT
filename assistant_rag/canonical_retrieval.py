"""Canonical derived-index retrieval entry points for every branch."""

from __future__ import annotations

from math import isfinite

from .contracts import RetrievalResult
from .database import AssistantRepository
from .retrieval import HybridRetriever
from .retrieval_policy import RETRIEVAL_PIPELINE_POLICY


class KnowledgeRetrievalUnavailableError(RuntimeError):
    """Authoritative personal-knowledge retrieval could not be completed."""


def _sql_rows_to_candidates(
    *,
    rows: list[dict],
    user_id: str,
) -> list[RetrievalResult]:
    """Build candidates from one bounded active, user-owned SQL page."""

    candidates: list[RetrievalResult] = []
    for row in rows:
        if str(row.get("user_id") or "") != user_id or bool(row.get("is_deleted")):
            continue
        entity_id = str(row.get("chunk_id") or "").strip()
        text = str(row.get("normalized_text") or row.get("raw_text") or "").strip()
        if not entity_id or not text:
            continue
        candidates.append(
            RetrievalResult(
                entity_type="knowledge_chunk",
                entity_id=entity_id,
                source_store_evidence={"sql_recovery": text},
                rerank_score=0.0,
                confidence=0.0,
                validation_status="sql_recovery_candidate",
                payload={
                    "user_id": user_id,
                    "knowledge_topic_id": row.get("knowledge_topic_id"),
                    "chunk_id": entity_id,
                    "source_id": row.get("source_id"),
                    "is_deleted": False,
                    "version": row.get("version"),
                    "text": text,
                },
            )
        )
    return candidates


def _rerank_sql_candidates(
    *,
    retriever: HybridRetriever,
    query: str,
    candidates: list[RetrievalResult],
    enforce_min_score: bool,
) -> list[RetrievalResult]:
    """Apply the canonical semantic ranker to SQL recovery candidates."""

    if not candidates:
        return []
    batch_size = max(
        1,
        int(getattr(retriever, "rerank_candidate_limit", 15)),
    )
    reranked: list[RetrievalResult] = []
    for offset in range(0, len(candidates), batch_size):
        batch = candidates[offset : offset + batch_size]
        if enforce_min_score:
            ranked_batch = retriever.reranker.rerank(query, batch)
        else:
            ranked_batch = retriever.reranker.rerank(
                query,
                batch,
                enforce_min_score=False,
            )
        reranked.extend(ranked_batch)

    threshold = float(getattr(retriever, "rerank_min_score", 0.0))
    approved = [
        result
        for result in reranked
        if result.payload.get("user_id")
        == candidates[0].payload.get("user_id")
        and not bool(result.payload.get("is_deleted"))
        and isfinite(float(result.rerank_score))
        and (
            not enforce_min_score
            or float(result.rerank_score) >= threshold
        )
    ]
    approved.sort(key=lambda result: float(result.rerank_score), reverse=True)
    return approved


def _merge_ranked(
    *groups: list[RetrievalResult],
) -> list[RetrievalResult]:
    by_id: dict[str, RetrievalResult] = {}
    for group in groups:
        for result in group:
            existing = by_id.get(result.entity_id)
            if existing is None or float(result.rerank_score) > float(
                existing.rerank_score
            ):
                by_id[result.entity_id] = result
    return sorted(
        by_id.values(),
        key=lambda result: float(result.rerank_score),
        reverse=True,
    )[: RETRIEVAL_PIPELINE_POLICY.final_top_k]


def _load_complete_sql_snapshot(
    *,
    repository: AssistantRepository,
    user_id: str,
    candidate_budget: int,
) -> list[RetrievalResult]:
    """Load every active owned row, or prove that the budget is insufficient."""

    rows = repository.list_knowledge_recovery_candidates(
        user_id=user_id,
        limit=candidate_budget + 1,
    )
    if len(rows) > candidate_budget:
        raise KnowledgeRetrievalUnavailableError(
            "Authoritative knowledge exceeds the bounded recovery budget"
        )
    candidates = _sql_rows_to_candidates(rows=rows, user_id=user_id)
    if len(candidates) != len(rows):
        raise KnowledgeRetrievalUnavailableError(
            "Authoritative knowledge snapshot contained an invalid row"
        )
    return candidates


def _recover_bounded_sql_knowledge(
    *,
    retriever: HybridRetriever,
    repository: AssistantRepository,
    user_id: str,
    query: str,
    enforce_min_score: bool,
) -> list[RetrievalResult]:
    """Rank a complete authoritative snapshot within a strict total budget.

    A derived-index hit and a completed outbox job prove neither that every
    active row is present in the index nor that the index answered this query
    completely.  Read one extra SQL row to distinguish a complete bounded
    snapshot from an unverifiable partial one.  Overflow fails closed before
    any SQL candidate reaches the semantic reranker.
    """

    candidate_budget = max(
        1,
        int(getattr(retriever, "sql_fallback_candidate_limit", 200)),
    )
    candidates = _load_complete_sql_snapshot(
        repository=repository,
        user_id=user_id,
        candidate_budget=candidate_budget,
    )
    ranked = _rerank_sql_candidates(
        retriever=retriever,
        query=query,
        candidates=candidates,
        enforce_min_score=enforce_min_score,
    )
    return repository.hydrate_knowledge_retrieval_results(
        user_id=user_id,
        results=ranked,
    )


def retrieve_knowledge(
    *,
    retriever: HybridRetriever,
    repository: AssistantRepository,
    user_id: str,
    query: str,
    enforce_min_score: bool = True,
) -> list[RetrievalResult]:
    """Retrieve SQL-validated knowledge with semantic SQL recovery on cache miss."""

    try:
        if enforce_min_score:
            candidates = retriever.retrieve_knowledge(user_id=user_id, query=query)
        else:
            candidates = retriever.retrieve_knowledge(
                user_id=user_id,
                query=query,
                enforce_min_score=False,
            )
        hydrated = repository.hydrate_knowledge_retrieval_results(
            user_id=user_id,
            results=candidates,
        )
    except Exception:
        hydrated = []
    reranker = getattr(retriever, "reranker", None)
    if not callable(getattr(reranker, "rerank", None)):
        # Compatibility for narrow injected retrievers that already returned
        # every active SQL row but do not expose the production semantic
        # ranker. Partial derived results cannot prove coverage and therefore
        # fail closed instead of hiding an authoritative row.
        candidate_budget = max(
            1,
            int(getattr(retriever, "sql_fallback_candidate_limit", 200)),
        )
        authoritative = _load_complete_sql_snapshot(
            repository=repository,
            user_id=user_id,
            candidate_budget=candidate_budget,
        )
        if not authoritative:
            return []
        authoritative_ids = {result.entity_id for result in authoritative}
        hydrated_ids = {result.entity_id for result in hydrated}
        if authoritative_ids == hydrated_ids:
            return _merge_ranked(hydrated)
        raise KnowledgeRetrievalUnavailableError(
            "A semantic ranker is required for incomplete derived retrieval"
        )
    try:
        recovered = _recover_bounded_sql_knowledge(
            retriever=retriever,
            repository=repository,
            user_id=user_id,
            query=query,
            enforce_min_score=enforce_min_score,
        )
    except Exception as exc:
        raise KnowledgeRetrievalUnavailableError(
            "Authoritative knowledge retrieval did not complete"
        ) from exc
    return _merge_ranked(hydrated, recovered)
