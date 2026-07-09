"""Background indexing from SQL outbox into derived caches."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import sqlite3

from .config import OutboxConfig
from .database import now_iso
from .retrieval import SearchIndex


@dataclass
class BackgroundIndexer:
    connection: sqlite3.Connection
    bm25: SearchIndex
    chroma: SearchIndex
    config: OutboxConfig

    def process_pending(self) -> int:
        self.connection.row_factory = sqlite3.Row
        self._release_stale_processing_jobs()
        processed = 0
        jobs = self._claim_jobs()
        for job in jobs:
            self._process_job(job)
            processed += 1
        return processed

    def _claim_jobs(self) -> list[sqlite3.Row]:
        retry_cutoff = datetime.now(UTC) - timedelta(seconds=self.config.retry_backoff_seconds)
        jobs = self.connection.execute(
            """
            SELECT * FROM indexing_outbox
            WHERE retry_count < ?
              AND (
                status = 'pending'
                OR (status = 'failed' AND updated_at <= ?)
              )
            ORDER BY created_at
            LIMIT ?
            """,
            (self.config.max_attempts, retry_cutoff.isoformat(), self.config.batch_size),
        ).fetchall()
        claimed: list[sqlite3.Row] = []
        for job in jobs:
            cursor = self.connection.execute(
                """
                UPDATE indexing_outbox
                SET status = 'processing', updated_at = ?
                WHERE job_id = ?
                  AND status IN ('pending', 'failed')
                  AND retry_count < ?
                """,
                (now_iso(), job["job_id"], self.config.max_attempts),
            )
            if cursor.rowcount == 1:
                claimed.append(job)
        self.connection.commit()
        return claimed

    def _release_stale_processing_jobs(self) -> None:
        cutoff = datetime.now(UTC) - timedelta(seconds=self.config.processing_timeout_seconds)
        self.connection.execute(
            """
            UPDATE indexing_outbox
            SET status = 'failed',
                retry_count = retry_count + 1,
                error_message = 'Indexing job timed out while processing',
                updated_at = ?
            WHERE status = 'processing' AND updated_at < ? AND retry_count < ?
            """,
            (now_iso(), cutoff.isoformat(), self.config.max_attempts),
        )
        self.connection.commit()

    def _process_job(self, job: sqlite3.Row) -> None:
        try:
            if job["operation"] == "delete":
                self.bm25.delete(entity_id=job["entity_id"])
                self.chroma.delete(entity_id=job["entity_id"])
            else:
                user_id, text = self._load_authoritative_text(
                    entity_type=job["entity_type"], entity_id=job["entity_id"]
                )
                self.bm25.upsert(
                    user_id=user_id,
                    entity_type=job["entity_type"],
                    entity_id=job["entity_id"],
                    text=text,
                )
                self.chroma.upsert(
                    user_id=user_id,
                    entity_type=job["entity_type"],
                    entity_id=job["entity_id"],
                    text=text,
                )
            self.connection.execute(
                """
                UPDATE indexing_outbox
                SET status = 'completed', error_message = NULL, updated_at = ?
                WHERE job_id = ?
                """,
                (now_iso(), job["job_id"]),
            )
            self.connection.commit()
        except Exception as exc:
            self.connection.rollback()
            self.connection.execute(
                """
                UPDATE indexing_outbox
                SET status = 'failed', retry_count = retry_count + 1,
                    error_message = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (str(exc), now_iso(), job["job_id"]),
            )
            self.connection.commit()

    def _load_authoritative_text(self, *, entity_type: str, entity_id: str) -> tuple[str, str]:
        if entity_type == "conversation_hop":
            row = self.connection.execute(
                """
                SELECT user_id, raw_user_query, raw_response
                FROM conversation_hops
                WHERE hop_id = ?
                """,
                (entity_id,),
            ).fetchone()
            if not row:
                raise ValueError("Conversation hop not found for indexing")
            return str(row["user_id"]), f"{row['raw_user_query']} {row['raw_response']}"
        if entity_type == "knowledge_chunk":
            row = self.connection.execute(
                """
                SELECT user_id, normalized_text
                FROM knowledge_chunks
                WHERE chunk_id = ? AND is_deleted = 0
                """,
                (entity_id,),
            ).fetchone()
            if not row:
                raise ValueError("Active knowledge chunk not found for indexing")
            return str(row["user_id"]), str(row["normalized_text"])
        raise ValueError("Unsupported outbox entity type")

    def rebuild_from_sql(self) -> None:
        for index in (self.bm25, self.chroma):
            index.clear()
        for row in self.connection.execute(
            "SELECT hop_id FROM conversation_hops ORDER BY created_at"
        ).fetchall():
            user_id, text = self._load_authoritative_text(
                entity_type="conversation_hop", entity_id=row["hop_id"]
            )
            for index in (self.bm25, self.chroma):
                index.upsert(
                    user_id=user_id,
                    entity_type="conversation_hop",
                    entity_id=row["hop_id"],
                    text=text,
                )
        for row in self.connection.execute(
            "SELECT chunk_id FROM knowledge_chunks WHERE is_deleted = 0 ORDER BY created_at"
        ).fetchall():
            user_id, text = self._load_authoritative_text(
                entity_type="knowledge_chunk", entity_id=row["chunk_id"]
            )
            for index in (self.bm25, self.chroma):
                index.upsert(
                    user_id=user_id,
                    entity_type="knowledge_chunk",
                    entity_id=row["chunk_id"],
                    text=text,
                )
