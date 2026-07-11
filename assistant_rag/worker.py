"""Production worker helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import time
import logging

logger = logging.getLogger(__name__)

from .autoscan import ReminderAutoscan
from .config import OutboxConfig
from .database import AssistantRepository
from .indexing import BackgroundIndexer
from .reminder_timing import ReminderTimingPlanner
from .retrieval import SearchIndex
from .settings import ProductionSettings


@dataclass
class ProductionWorker:
    repository: AssistantRepository
    bm25: SearchIndex
    chroma: SearchIndex
    settings: ProductionSettings
    reminder_timing_planner: ReminderTimingPlanner | None = None

    def __post_init__(self) -> None:
        # A production worker must own a real planner.  Keeping construction
        # here makes it impossible to accidentally run autoscan without the
        # timing LLM when callers use the normal worker constructor.
        if self.reminder_timing_planner is None:
            from .production_factory import build_reminder_timing_planner
            self.reminder_timing_planner = build_reminder_timing_planner(self.settings)

    def run_once(self, *, scan_reminders: bool = True) -> dict[str, int]:
        indexer = BackgroundIndexer(
            repository=self.repository,
            bm25=self.bm25,
            chroma=self.chroma,
            config=OutboxConfig(
                max_attempts=self.settings.worker.outbox_max_attempts,
                batch_size=self.settings.worker.outbox_batch_size,
                retry_backoff_seconds=self.settings.worker.outbox_retry_backoff_seconds,
                processing_timeout_seconds=self.settings.worker.outbox_processing_timeout_seconds,
            ),
        )
        try:
            indexed = indexer.process_pending()
            if indexed > 0:
                logger.info(f"Processed {indexed} indexing jobs.")
        except Exception as e:
            logger.error("Exception during background indexing", exc_info=e)
            indexed = 0

        notified_count = 0
        if scan_reminders:
            try:
                catch_up = ReminderAutoscan(
                    self.repository, timing_planner=self.reminder_timing_planner
                ).catch_up_due(
                    now_value=datetime.now(timezone.utc).isoformat()
                )
                notified_count = int(catch_up["notified"])
                if notified_count:
                    logger.info("Scanned and generated %s reminder notifications.", notified_count)
            except Exception as e:
                logger.error("Exception during reminder autoscan", exc_info=e)
                notified_count = 0

        return {"indexed_jobs": indexed, "notified_reminders": notified_count}

    def run_forever(self) -> None:
        logger.info("Starting background worker...")
        last_reminder_scan_at = 0.0
        while True:
            try:
                now = time.monotonic()
                scan_reminders = (
                    now - last_reminder_scan_at
                    >= self.settings.worker.autoscan_interval_seconds
                )
                self.run_once(scan_reminders=scan_reminders)
                if scan_reminders:
                    last_reminder_scan_at = now
            except Exception as e:
                logger.critical("Unexpected failure in worker loop", exc_info=e)
            time.sleep(self.settings.worker.outbox_worker_interval_seconds)
