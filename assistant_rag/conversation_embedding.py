"""Rewritten-query-only serialization for conversation-hop embeddings."""

from __future__ import annotations

import json
from typing import Any, Mapping


CONVERSATION_HOP_EMBEDDING_VERSION = "rewritten_query_hop_v2"
_RAW_USER_QUERY_FIELDS = frozenset(
    {
        "raw_query",
        "raw_user_query",
        "source_raw_user_query",
        "summarized_user_query",
    }
)


def _without_raw_user_queries(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _without_raw_user_queries(item)
            for key, item in value.items()
            if str(key).casefold() not in _RAW_USER_QUERY_FIELDS
        }
    if isinstance(value, list):
        return [_without_raw_user_queries(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_without_raw_user_queries(item) for item in value)
    return value


def conversation_hop_semantic_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    """Project a SQL audit row into safe downstream semantic context."""

    return dict(_without_raw_user_queries(dict(row)))


def serialize_conversation_hop(row: Mapping[str, Any]) -> str:
    """Serialize one hop without audit-only pre-rewrite user text."""
    return json.dumps(
        {
            "embedding_contract": CONVERSATION_HOP_EMBEDDING_VERSION,
            "chunk_index": 0,
            "chunk_count": 1,
            "conversation_hop": conversation_hop_semantic_payload(row),
        },
        ensure_ascii=False,
        default=str,
        separators=(",", ":"),
        sort_keys=True,
    )


def conversation_hop_embedding_metadata(row: Mapping[str, Any]) -> dict[str, str | int | float | bool]:
    """Store retrieval metadata without duplicating/splitting the embedded text."""
    return {
        "user_id": str(row["user_id"]),
        "topic_id": str(row["topic_id"]),
        "hop_id": str(row["hop_id"]),
        "parent_hop_id": str(row.get("parent_hop_id") or ""),
        "root_hop_id": str(row.get("root_hop_id") or ""),
        "branch_id": str(row.get("branch_id") or ""),
        "intent": str(row["intent"]),
        "response_type": str(row["response_type"]),
        "created_at": str(row["created_at"]),
        "is_deleted": False,
        "chunk_index": 0,
        "chunk_count": 1,
        "embedding_contract": CONVERSATION_HOP_EMBEDDING_VERSION,
    }
