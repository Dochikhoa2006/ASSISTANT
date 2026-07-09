"""Background indexing from SQL outbox into derived caches."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from .config import OutboxConfig
from .database import AssistantRepository
from .metrics import GLOBAL_METRICS
from .retrieval import SearchIndex

@dataclass
class BackgroundIndexer:
    repository: AssistantRepository
    bm25: SearchIndex
    chroma: SearchIndex
    config: OutboxConfig

    def process_pending(self) -> int:
        self.repository.release_stale_processing_jobs(
            max_attempts=self.config.max_attempts,
            timeout_cutoff=(datetime.now(timezone.utc) - timedelta(seconds=self.config.processing_timeout_seconds)).isoformat(),
        )
        processed = 0
        jobs = self._claim_jobs()
        for job in jobs:
            self._process_job(job)
            processed += 1
        GLOBAL_METRICS.increment("outbox_processed_total", processed)
        return processed

    def _claim_jobs(self) -> list[dict[str, Any]]:
        retry_cutoff = datetime.now(timezone.utc) - timedelta(seconds=self.config.retry_backoff_seconds)
        return self.repository.claim_outbox_jobs(
            max_attempts=self.config.max_attempts,
            batch_size=self.config.batch_size,
            retry_cutoff=retry_cutoff.isoformat(),
        )

    def _process_job(self, job: dict[str, Any]) -> None:
        try:
            if job["operation"] == "delete":
                self.bm25.delete(entity_id=job["entity_id"])
                self.chroma.delete(entity_id=job["entity_id"])
            else:
                payload = self.repository.load_outbox_entity(
                    entity_type=job["entity_type"], entity_id=job["entity_id"]
                )
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
            self.repository.mark_outbox_job_completed(job_id=job["job_id"])
            GLOBAL_METRICS.increment("outbox_job_success_total", entity_type=str(job["entity_type"]))
        except Exception as e:
            self.repository.mark_outbox_job_failed(job_id=job["job_id"], error_message=str(e))
            GLOBAL_METRICS.increment("outbox_job_failure_total", entity_type=str(job.get("entity_type", "unknown")))

    def rebuild_from_sql(self) -> None:
        for index in (self.bm25, self.chroma):
            index.clear()
        
        for entity_type, entity_id in self.repository.list_all_outbox_entities():
            try:
                payload = self.repository.load_outbox_entity(
                    entity_type=entity_type, entity_id=entity_id
                )
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
            except Exception as e:
                pass
