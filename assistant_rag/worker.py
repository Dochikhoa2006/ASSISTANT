"""Production worker helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import time

from .autoscan import ReminderAutoscan
from .config import OutboxConfig
from .database import SQLRepository
from .indexing import BackgroundIndexer
from .retrieval import SearchIndex
from .settings import ProductionSettings


@dataclass
class ProductionWorker:
    repository: SQLRepository
    bm25: SearchIndex
    chroma: SearchIndex
    settings: ProductionSettings

    def run_once(self) -> dict[str, int]:
        indexer = BackgroundIndexer(
            connection=self.repository.connection,
            bm25=self.bm25,
            chroma=self.chroma,
            config=OutboxConfig(
                max_attempts=self.settings.worker.outbox_max_attempts,
                batch_size=self.settings.worker.outbox_batch_size,
                retry_backoff_seconds=self.settings.worker.outbox_retry_backoff_seconds,
                processing_timeout_seconds=self.settings.worker.outbox_processing_timeout_seconds,
            ),
        )
        indexed = indexer.process_pending()
        notified = ReminderAutoscan(self.repository).scan_due(
            now_value=datetime.now(UTC).isoformat()
        )
        return {"indexed_jobs": indexed, "notified_reminders": len(notified)}

    def run_forever(self) -> None:
        while True:
            self.run_once()
            time.sleep(self.settings.worker.autoscan_interval_seconds)
