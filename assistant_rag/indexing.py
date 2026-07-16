"""Background indexing from SQL outbox into derived caches."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from time import sleep
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
            if self._process_job(job):
                processed += 1
        GLOBAL_METRICS.increment("outbox_processed_total", processed)
        return processed

    def process_job_ids(self, job_ids: list[str]) -> int:
        """Synchronize the current request's durable jobs immediately.

        The outbox remains authoritative for retries. Jobs already claimed by
        a concurrent worker are deliberately left to that worker, while every
        newly claimed job is written to both indexes before completion.
        """

        requested = set(dict.fromkeys(job_id for job_id in job_ids if job_id))
        remaining = set(requested)
        synchronized: set[str] = set()
        for attempt in range(self.config.max_attempts):
            statuses = self.repository.get_outbox_job_statuses(
                job_ids=list(remaining)
            )
            completed = {
                job_id
                for job_id, status in statuses.items()
                if status == "completed"
            }
            synchronized.update(completed)
            remaining.difference_update(completed)
            if not remaining:
                break

            jobs = self.repository.claim_outbox_jobs_by_ids(
                job_ids=list(remaining),
                max_attempts=self.config.max_attempts,
            )
            for job in jobs:
                if self._process_job(job):
                    synchronized.add(str(job["job_id"]))
                    remaining.discard(str(job["job_id"]))
            if remaining and not jobs and attempt + 1 < self.config.max_attempts:
                # A concurrent worker may own an exact request job. Give it a
                # short bounded window, then verify its durable completion.
                sleep(0.01)

        GLOBAL_METRICS.increment("outbox_processed_total", len(synchronized))
        if remaining:
            statuses = self.repository.get_outbox_job_statuses(
                job_ids=list(remaining)
            )
            raise RuntimeError(
                "Request-scoped index synchronization did not complete for "
                f"jobs {sorted(remaining)} (statuses={statuses})."
            )
        return len(synchronized)

    def _claim_jobs(self) -> list[dict[str, Any]]:
        retry_cutoff = datetime.now(timezone.utc) - timedelta(seconds=self.config.retry_backoff_seconds)
        return self.repository.claim_outbox_jobs(
            max_attempts=self.config.max_attempts,
            batch_size=self.config.batch_size,
            retry_cutoff=retry_cutoff.isoformat(),
        )

    def _process_job(self, job: dict[str, Any]) -> bool:
        try:
            # SQL is the source of truth. Resolve every queued operation against
            # current state so a delayed DELETE cannot remove a reactivated
            # chunk and a delayed UPSERT cannot restore a deleted one.
            try:
                payload = self.repository.load_outbox_entity(
                    entity_type=job["entity_type"], entity_id=job["entity_id"]
                )
            except ValueError:
                payload = None
            if payload is None:
                self.bm25.delete(entity_id=job["entity_id"])
                self.chroma.delete(entity_id=job["entity_id"])
            else:
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
            return True
        except Exception as e:
            self.repository.mark_outbox_job_failed(job_id=job["job_id"], error_message=str(e))
            GLOBAL_METRICS.increment("outbox_job_failure_total", entity_type=str(job.get("entity_type", "unknown")))
            return False

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
