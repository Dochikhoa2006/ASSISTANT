"""Lossless, single-document serialization for conversation-hop embeddings."""

from __future__ import annotations

import json
from typing import Any, Mapping


CONVERSATION_HOP_EMBEDDING_VERSION = "whole_hop_v1"


def serialize_conversation_hop(row: Mapping[str, Any]) -> str:
    """Serialize every persisted hop/topic value as one indivisible document."""
    return json.dumps(
        {
            "embedding_contract": CONVERSATION_HOP_EMBEDDING_VERSION,
            "chunk_index": 0,
            "chunk_count": 1,
            "conversation_hop": dict(row),
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
