"""ChromaDB persistent derived cache."""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import RetrievalResult
from .embeddings import EmbeddingClient
from .settings import ChromaSettings


@dataclass
class ChromaPersistentVectorIndex:
    settings: ChromaSettings
    embedding_client: EmbeddingClient

    def __post_init__(self) -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError("Install chromadb for production vector retrieval") from exc
        if self.settings.host and self.settings.port:
            self.client = chromadb.HttpClient(host=self.settings.host, port=self.settings.port)
        else:
            self.client = chromadb.PersistentClient(path=self.settings.path)
        self.collections = {
            "conversation_hop": self.client.get_or_create_collection(
                self.settings.conversation_collection
            ),
            "knowledge_chunk": self.client.get_or_create_collection(
                self.settings.knowledge_collection
            ),
        }

    def search(self, *, user_id: str, query: str, limit: int) -> list[RetrievalResult]:
        query_embedding = self.embedding_client.embed([query])[0]
        results: list[RetrievalResult] = []
        for entity_type, collection in self.collections.items():
            payload = collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                where={"user_id": user_id},
                include=["documents", "metadatas", "distances"],
            )
            ids = payload.get("ids", [[]])[0]
            docs = payload.get("documents", [[]])[0]
            distances = payload.get("distances", [[]])[0]
            for entity_id, document, distance in zip(ids, docs, distances, strict=False):
                confidence = 1.0 / (1.0 + float(distance))
                results.append(
                    RetrievalResult(
                        entity_type=entity_type,
                        entity_id=str(entity_id),
                        source_store_evidence={"chroma": document},
                        rerank_score=confidence,
                        confidence=confidence,
                        validation_status="candidate",
                        payload={"text": document},
                    )
                )
        return sorted(results, key=lambda item: item.confidence, reverse=True)[:limit]

    def upsert(self, *, user_id: str, entity_type: str, entity_id: str, text: str) -> None:
        if entity_type not in self.collections:
            raise ValueError("Reminder entities must not be indexed in ChromaDB")
        embedding = self.embedding_client.embed([text])[0]
        self.collections[entity_type].upsert(
            ids=[entity_id],
            documents=[text],
            embeddings=[embedding],
            metadatas=[
                {
                    "user_id": user_id,
                    "entity_type": entity_type,
                    "sql_entity_id": entity_id,
                    "embedding_model": self._embedding_model_name(),
                }
            ],
        )

    def delete(self, *, entity_id: str) -> None:
        for collection in self.collections.values():
            collection.delete(ids=[entity_id])

    def clear(self) -> None:
        for collection_name in (
            self.settings.conversation_collection,
            self.settings.knowledge_collection,
        ):
            try:
                self.client.delete_collection(collection_name)
            except Exception:
                pass
        self.collections = {
            "conversation_hop": self.client.get_or_create_collection(
                self.settings.conversation_collection
            ),
            "knowledge_chunk": self.client.get_or_create_collection(
                self.settings.knowledge_collection
            ),
        }

    def _embedding_model_name(self) -> str:
        settings = getattr(self.embedding_client, "settings", None)
        return str(getattr(settings, "model_name", "unknown"))
