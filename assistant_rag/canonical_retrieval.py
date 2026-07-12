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
) -> list[RetrievalResult]:
    """Run the sole knowledge retrieval pipeline, then hydrate its SQL truth."""
    candidates = retriever.retrieve_knowledge(user_id=user_id, query=query)
    return repository.hydrate_knowledge_retrieval_results(
        user_id=user_id,
        results=candidates,
    )
