"""SQL-sourced index rebuild and drift checks."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .contracts import DriftIssue, DriftReport, OutboxIndexPayload
from .retrieval import SearchIndex


class IndexOperations:
    def __init__(self, *, repository: Any, bm25: SearchIndex, chroma: SearchIndex):
        self.repository = repository
        self.bm25 = bm25
        self.chroma = chroma

    def rebuild_all_indexes(self) -> int:
        self.bm25.clear()
        self.chroma.clear()
        return self._rebuild(entity_type=None, user_id=None)

    def rebuild_user_indexes(self, *, user_id: str) -> int:
        self._clear_user_scope(user_id)
        return self._rebuild(entity_type=None, user_id=user_id)

    def rebuild_knowledge_index(self, *, user_id: str | None = None) -> int:
        self._clear_scope("knowledge_chunk", user_id)
        return self._rebuild(entity_type="knowledge_chunk", user_id=user_id)

    def rebuild_conversation_index(self, *, user_id: str | None = None) -> int:
        self._clear_scope("conversation_hop", user_id)
        return self._rebuild(entity_type="conversation_hop", user_id=user_id)

    def check_index_drift(self, *, user_id: str | None = None, repair: bool = False) -> dict[str, Any]:
        sql_payloads = self._sql_payloads(user_id=user_id)
        sql_by_type = _count_by_type(sql_payloads)
        bm25_counts = self._index_counts(self.bm25, user_id=user_id)
        chroma_counts = self._index_counts(self.chroma, user_id=user_id)
        issues: list[DriftIssue] = []
        for entity_type, sql_count in sql_by_type.items():
            if bm25_counts.get(entity_type, 0) != sql_count:
                issues.append(DriftIssue("count_mismatch", entity_type, detail=f"bm25={bm25_counts.get(entity_type, 0)} sql={sql_count}"))
            if chroma_counts.get(entity_type, 0) != sql_count:
                issues.append(DriftIssue("count_mismatch", entity_type, detail=f"chroma={chroma_counts.get(entity_type, 0)} sql={sql_count}"))
        for payload in sql_payloads:
            issues.extend(self._metadata_issues(payload, self.bm25, "bm25"))
            issues.extend(self._metadata_issues(payload, self.chroma, "chroma"))
        issues.extend(self._forbidden_doc_issues(self.bm25, "bm25", user_id=user_id))
        issues.extend(self._forbidden_doc_issues(self.chroma, "chroma", user_id=user_id))
        failed_outbox = self._failed_outbox_count()
        if failed_outbox:
            issues.append(DriftIssue("failed_outbox", "indexing_outbox", detail=str(failed_outbox)))
        repaired = False
        if issues and repair:
            self.rebuild_user_indexes(user_id=user_id) if user_id else self.rebuild_all_indexes()
            repaired = True
        report = DriftReport(
            status="drift_detected" if issues else "ok",
            user_id=user_id,
            sql_counts=sql_by_type,
            bm25_counts=bm25_counts,
            chroma_counts=chroma_counts,
            failed_outbox_count=failed_outbox,
            issues=issues,
            repaired=repaired,
        )
        return asdict(report)

    def _rebuild(self, *, entity_type: str | None, user_id: str | None) -> int:
        count = 0
        for payload in self._sql_payloads(user_id=user_id):
            if entity_type and payload.entity_type != entity_type:
                continue
            self.bm25.upsert(
                user_id=payload.user_id,
                entity_type=payload.entity_type,
                entity_id=payload.entity_id,
                text=payload.text,
                metadata=payload.metadata,
            )
            self.chroma.upsert(
                user_id=payload.user_id,
                entity_type=payload.entity_type,
                entity_id=payload.entity_id,
                text=payload.text,
                metadata=payload.metadata,
            )
            count += 1
        return count

    def _sql_payloads(self, *, user_id: str | None) -> list[OutboxIndexPayload]:
        payloads: list[OutboxIndexPayload] = []
        for entity_type, entity_id in self.repository.list_all_outbox_entities():
            try:
                payload = self.repository.load_outbox_entity(entity_type=entity_type, entity_id=entity_id)
            except Exception:
                continue
            if user_id and payload.user_id != user_id:
                continue
            if payload.entity_type not in {"conversation_hop", "knowledge_chunk"}:
                continue
            payloads.append(payload)
        return payloads

    def _clear_user_scope(self, user_id: str) -> None:
        if hasattr(self.bm25, "clear_user"):
            self.bm25.clear_user(user_id=user_id)
        if hasattr(self.chroma, "clear_user"):
            self.chroma.clear_user(user_id=user_id)
        for payload in self._sql_payloads(user_id=user_id):
            self.bm25.delete(entity_id=payload.entity_id)
            self.chroma.delete(entity_id=payload.entity_id)

    def _clear_scope(self, entity_type: str, user_id: str | None) -> None:
        if user_id is None and hasattr(self.bm25, "clear_entity_type") and hasattr(self.chroma, "clear_entity_type"):
            self.bm25.clear_entity_type(entity_type=entity_type)
            self.chroma.clear_entity_type(entity_type=entity_type)
            return
        if user_id is None:
            for payload in self._sql_payloads(user_id=None):
                if payload.entity_type == entity_type:
                    self.bm25.delete(entity_id=payload.entity_id)
                    self.chroma.delete(entity_id=payload.entity_id)
            return
        for payload in self._sql_payloads(user_id=user_id):
            if payload.entity_type == entity_type:
                self.bm25.delete(entity_id=payload.entity_id)
                self.chroma.delete(entity_id=payload.entity_id)

    def _index_counts(self, index: SearchIndex, *, user_id: str | None) -> dict[str, int]:
        if hasattr(index, "count_by_entity_type"):
            return dict(index.count_by_entity_type(user_id=user_id))
        if hasattr(index, "documents"):
            docs = index.documents()
            counts: dict[str, int] = {}
            for doc in docs:
                if user_id and doc.get("user_id") != user_id:
                    continue
                entity_type = str(doc.get("entity_type"))
                counts[entity_type] = counts.get(entity_type, 0) + 1
            return counts
        return {}

    def _metadata_issues(self, payload: OutboxIndexPayload, index: SearchIndex, store_name: str) -> list[DriftIssue]:
        if not hasattr(index, "get_document"):
            return []
        doc = index.get_document(entity_id=payload.entity_id)
        if not doc:
            return [DriftIssue("missing_doc", payload.entity_type, payload.entity_id, f"{store_name} missing")]
        issues: list[DriftIssue] = []
        for key in ("content_hash", "version", "topic_id", "hop_id", "response_type"):
            if key in payload.metadata and str(doc.get(key)) != str(payload.metadata[key]):
                issues.append(DriftIssue("metadata_mismatch", payload.entity_type, payload.entity_id, f"{store_name}:{key}"))
        return issues

    def _forbidden_doc_issues(self, index: SearchIndex, store_name: str, *, user_id: str | None) -> list[DriftIssue]:
        if not hasattr(index, "documents"):
            return []
        issues: list[DriftIssue] = []
        for doc in index.documents():
            if user_id and doc.get("user_id") != user_id:
                continue
            if doc.get("entity_type") not in {"conversation_hop", "knowledge_chunk"}:
                issues.append(DriftIssue("forbidden_doc", str(doc.get("entity_type")), str(doc.get("entity_id")), store_name))
            if doc.get("entity_type") == "knowledge_chunk" and bool(doc.get("is_deleted")):
                issues.append(DriftIssue("deleted_doc_indexed", "knowledge_chunk", str(doc.get("entity_id")), store_name))
        return issues

    def _failed_outbox_count(self) -> int:
        connection = getattr(self.repository, "connection", None)
        if connection is not None:
            row = connection.execute("SELECT COUNT(*) AS total FROM indexing_outbox WHERE status = 'failed'").fetchone()
            return int(row["total"])
        return 0


def _count_by_type(payloads: list[OutboxIndexPayload]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for payload in payloads:
        counts[payload.entity_type] = counts.get(payload.entity_type, 0) + 1
    counts.setdefault("conversation_hop", 0)
    counts.setdefault("knowledge_chunk", 0)
    return counts
