"""OpenSearch BM25 derived cache."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from .contracts import RetrievalResult
from .settings import OpenSearchSettings


@dataclass
class OpenSearchBM25Index:
    settings: OpenSearchSettings
    client: Any | None = None
    _initialized: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.client is None:
            try:
                from opensearchpy import OpenSearch
            except ImportError as exc:
                raise RuntimeError("Install opensearch-py for OpenSearch BM25 retrieval") from exc
            http_auth = None
            if self.settings.username and self.settings.password:
                http_auth = (self.settings.username, self.settings.password)
            self.client = OpenSearch(
                hosts=[self.settings.url],
                http_auth=http_auth,
                verify_certs=self.settings.verify_certs,
                timeout=self.settings.timeout_seconds,
                max_retries=self.settings.max_retries,
            )

    def initialize(self) -> None:
        for index_name in self._index_names():
            if not self.client.indices.exists(index=index_name):
                self.client.indices.create(index=index_name, body=self._mapping())
        self._ensure_aliases()
        self._initialized = True

    def search(self, *, user_id: str, query: str, limit: int) -> list[RetrievalResult]:
        self._ensure_initialized()
        results: list[RetrievalResult] = []
        for entity_type, index_name in (
            ("conversation_hop", self.settings.conversation_index),
            ("knowledge_chunk", self.settings.knowledge_index),
        ):
            response = self.client.search(
                index=index_name,
                body={
                    "size": limit,
                    "query": {
                        "bool": {
                            "filter": [{"term": {"user_id": user_id}}],
                            "must": [{"match": {"text": query}}],
                        }
                    },
                },
            )
            for hit in response.get("hits", {}).get("hits", []):
                source = hit.get("_source", {})
                score = float(hit.get("_score") or 0.0)
                confidence = score / (score + 1.0) if score > 0 else 0.0
                text = str(source.get("text", ""))
                results.append(
                    RetrievalResult(
                        entity_type=entity_type,
                        entity_id=str(source.get("entity_id") or hit.get("_id")),
                        source_store_evidence={"opensearch": text},
                        rerank_score=confidence,
                        confidence=confidence,
                        validation_status="candidate",
                        payload={"text": text},
                    )
                )
        return sorted(results, key=lambda item: item.rerank_score, reverse=True)[:limit]

    def upsert(self, *, user_id: str, entity_type: str, entity_id: str, text: str) -> None:
        self._ensure_initialized()
        index_name = self._index_for_entity(entity_type)
        self.client.index(
            index=index_name,
            id=entity_id,
            body={
                "entity_id": entity_id,
                "user_id": user_id,
                "entity_type": entity_type,
                "text": text,
                "updated_at": datetime.now(UTC).isoformat(),
            },
        )

    def delete(self, *, entity_id: str) -> None:
        self._ensure_initialized()
        for index_name in self._index_names():
            try:
                self.client.delete(index=index_name, id=entity_id)
            except Exception as exc:
                if getattr(exc, "status_code", None) != 404:
                    raise

    def clear(self) -> None:
        for index_name in self._index_names():
            if self.client.indices.exists(index=index_name):
                self.client.indices.delete(index=index_name)
        self._initialized = False
        self.initialize()

    def counts(self) -> dict[str, int]:
        self._ensure_initialized()
        return {
            self.settings.conversation_index: self._count(self.settings.conversation_index),
            self.settings.knowledge_index: self._count(self.settings.knowledge_index),
        }

    def check(self) -> dict[str, object]:
        reachable = bool(self.client.ping())
        if reachable:
            self.initialize()
        return {
            "backend": "opensearch",
            "url": self.settings.url,
            "reachable": reachable,
            "indexes": self.counts() if reachable else {},
        }

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            self.initialize()

    def _index_for_entity(self, entity_type: str) -> str:
        if entity_type == "conversation_hop":
            return self.settings.conversation_index
        if entity_type == "knowledge_chunk":
            return self.settings.knowledge_index
        raise ValueError("Reminder entities must not be indexed in OpenSearch BM25")

    def _index_names(self) -> tuple[str, str]:
        return (self.settings.conversation_index, self.settings.knowledge_index)

    def _count(self, index_name: str) -> int:
        response = self.client.count(index=index_name)
        return int(response.get("count", 0))

    def _ensure_aliases(self) -> None:
        put_alias = getattr(self.client.indices, "put_alias", None)
        if put_alias is None:
            return
        put_alias(index=self.settings.conversation_index, name=self.settings.conversation_write_alias)
        put_alias(index=self.settings.knowledge_index, name=self.settings.knowledge_write_alias)
        put_alias(
            index=self.settings.conversation_index,
            name=self.settings.reminder_context_alias,
            body={
                "filter": {
                    "bool": {
                        "should": [
                            {"term": {"intent": "reminder"}},
                            {"term": {"response_type": "reminder_action"}},
                            {"term": {"response_type": "reminder_reply"}},
                        ],
                        "minimum_should_match": 1,
                    }
                }
            },
        )

    def _mapping(self) -> dict[str, object]:
        return {
            "settings": {
                "analysis": {
                    "analyzer": {
                        self.settings.analyzer_name: {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": ["lowercase", "asciifolding"],
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "entity_id": {"type": "keyword"},
                    "user_id": {"type": "keyword"},
                    "entity_type": {"type": "keyword"},
                    "topic_id": {"type": "keyword"},
                    "knowledge_topic_id": {"type": "keyword"},
                    "source_id": {"type": "keyword"},
                    "branch_id": {"type": "keyword"},
                    "intent": {"type": "keyword"},
                    "response_type": {"type": "keyword"},
                    "content_hash": {"type": "keyword"},
                    "is_deleted": {"type": "boolean"},
                    "version": {"type": "integer"},
                    "text": {"type": "text", "analyzer": self.settings.analyzer_name},
                    "summary": {"type": "text", "analyzer": self.settings.analyzer_name},
                    "updated_at": {"type": "date"},
                }
            }
        }
