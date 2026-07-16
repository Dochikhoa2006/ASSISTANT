"""Canonical derived-index retrieval entry points for every branch."""

from __future__ import annotations

from .contracts import RetrievalResult
from .database import AssistantRepository
from .retrieval import HybridRetriever


def retrieve_knowledge(
    *,
    retriever: HybridRetriever,
    repository: AssistantRepository,
    user_id: str,
    query: str,
    enforce_min_score: bool = True,
) -> list[RetrievalResult]:
    """Run the sole knowledge retrieval pipeline, then hydrate its SQL truth."""
    if enforce_min_score:
        candidates = retriever.retrieve_knowledge(user_id=user_id, query=query)
    else:
        candidates = retriever.retrieve_knowledge(
            user_id=user_id,
            query=query,
            enforce_min_score=False,
        )
    return repository.hydrate_knowledge_retrieval_results(
        user_id=user_id,
        results=candidates,
    )
