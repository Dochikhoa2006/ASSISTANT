from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Iterator, ContextManager
from datetime import datetime
import sqlite3
from .contracts import *

class AssistantRepository(ABC):
    @abstractmethod
    def in_memory(cls) -> 'SQLiteRepository':
        pass

    @abstractmethod
    def persistent(cls, database_path: str, *, enable_wal: bool, busy_timeout_ms: int) -> 'SQLiteRepository':
        pass

    @abstractmethod
    def initialize_schema(self) -> None:
        pass

    @abstractmethod
    def transaction(self) -> Iterator[sqlite3.Cursor]:
        pass

    @abstractmethod
    def ensure_topic(self, cursor: sqlite3.Cursor, *, user_id: str, title: str, topic_summary: str='', state_summary: str='', entities: dict[str, Any] | None=None) -> str:
        pass

    @abstractmethod
    def append_conversation_hop(self, cursor: sqlite3.Cursor, *, topic_id: str, user_id: str, intent: str, raw_user_query: str, rewritten_user_query: str, raw_response: str, response_type: str, supporting_questions: list[str | dict[str, Any]] | None=None, parent_hop_id: str | None=None, branch_id: str | None=None, entities: dict[str, Any] | None=None, state_summary: str='', hop_id: str | None=None) -> HopWrite:
        pass

    @abstractmethod
    def scan_due_reminders(self, *, now_value: str, limit: int = 100) -> list[str]:
        pass

    @abstractmethod
    def load_reminder_reply_context(self, *, user_id: str, reminder_id: str, notification_id: str) -> dict[str, str | None]:
        pass

    @abstractmethod
    def add_knowledge_chunk(self, cursor: sqlite3.Cursor, *, user_id: str, title: str, text: str, source_id: str | None=None, metadata: dict[str, Any] | None=None, replaces_chunk_id: str | None = None, change_reason: str | None = None, modified_by_user_query: str | None = None) -> tuple[str, str, str | None]:
        pass

    @abstractmethod
    def ensure_knowledge_topic(self, cursor: sqlite3.Cursor, *, user_id: str, title: str) -> str:
        pass

    @abstractmethod
    def get_knowledge_chunks_by_ids(self, user_id: str, chunk_ids: list[str], include_deleted: bool=False) -> list[dict[str, Any]]:
        pass

    @abstractmethod
    def soft_delete_knowledge_chunk(self, cursor: sqlite3.Cursor, *, user_id: str, chunk_id: str, expected_version: int | None=None, change_reason: str | None = None, modified_by_user_query: str | None = None) -> str:
        pass

    @abstractmethod
    def create_knowledge_source(self, cursor: sqlite3.Cursor, *, user_id: str, filename: str, file_type: str, content_hash: str, metadata: dict[str, Any] | None = None, processing_status: str = "pending") -> str:
        pass

    @abstractmethod
    def update_knowledge_source_status(self, cursor: sqlite3.Cursor, *, user_id: str, source_id: str, processing_status: str, metadata: dict[str, Any] | None = None) -> None:
        pass

    @abstractmethod
    def list_knowledge_sources(self, *, user_id: str, include_deleted: bool = False) -> list[dict[str, Any]]:
        pass

    @abstractmethod
    def get_knowledge_source(self, *, user_id: str, source_id: str, include_deleted: bool = False) -> dict[str, Any]:
        pass

    @abstractmethod
    def soft_delete_knowledge_source(self, *, user_id: str, source_id: str) -> dict[str, Any]:
        pass

    @abstractmethod
    def reindex_knowledge_source(self, *, user_id: str, source_id: str) -> list[str]:
        pass

    @abstractmethod
    def list_knowledge_facts(self, *, user_id: str, include_deleted: bool = False, source_id: str | None = None) -> list[dict[str, Any]]:
        pass

    @abstractmethod
    def restore_knowledge_chunk(self, *, user_id: str, chunk_id: str) -> str:
        pass

    @abstractmethod
    def update_knowledge_chunk_text(self, *, user_id: str, chunk_id: str, text: str, change_reason: str | None = None, modified_by_user_query: str | None = None) -> dict[str, Any]:
        pass

    @abstractmethod
    def create_generated_artifact(self, *, user_id: str, conversation_hop_id: str | None, file_type: str, filename: str, storage_path: str, storage_url: str, metadata: dict[str, Any] | None = None, expires_at: str | None = None) -> dict[str, Any]:
        pass

    @abstractmethod
    def list_generated_artifacts(self, *, user_id: str, include_deleted: bool = False) -> list[dict[str, Any]]:
        pass

    @abstractmethod
    def get_generated_artifact(self, *, user_id: str, artifact_id: str, include_deleted: bool = False) -> dict[str, Any]:
        pass

    @abstractmethod
    def delete_generated_artifact(self, *, user_id: str, artifact_id: str) -> dict[str, Any]:
        pass

    def record_platform_delivery(self, *, user_id: str, conversation_hop_id: str | None, channel: str, status: str, recipient: str, message: dict[str, Any], error_message: str | None = None) -> dict[str, Any]:
        """Optionally persist safe outbound-delivery metadata.

        Kept as a concrete no-op so existing custom repository implementations
        remain compatible while SQLite/Postgres provide the audit record.
        """
        return {}

    @abstractmethod
    def add_reminder(self, cursor: sqlite3.Cursor, *, user_id: str, source_topic_id: str | None, source_hop_id: str | None, reminder_time: str, raw_reminder: str, reminder_summary: str, subject: str, event_time: str | None=None, supporting_question: str | None=None, supporting_response: str | None=None, user_timezone: str = "UTC", original_time_text: str | None=None, recurrence_rule: str | None=None, recurrence_timezone: str | None=None, next_fire_time: str | None=None, parent_recurring_reminder_id: str | None=None) -> str:
        pass

    @abstractmethod
    def find_active_reminder_duplicates(self, *, user_id: str, subject: str, reminder_time: datetime, statuses: tuple[str, ...] = ("scheduled", "notified"), limit: int = 20) -> dict[str, Any]:
        pass

    @abstractmethod
    def update_reminder_status(self, cursor: sqlite3.Cursor, *, user_id: str, reminder_id: str, status: str, expected_version: int | None=None, expected_status: str | None=None) -> None:
        pass

    @abstractmethod
    def create_notification_if_absent(self, cursor: sqlite3.Cursor, *, user_id: str, reminder_id: str, fire_time: str | None = None) -> str:
        pass

    @abstractmethod
    def insert_outbox_job(self, cursor: sqlite3.Cursor, *, entity_type: OutboxEntityType, entity_id: str, operation: OutboxOperation) -> str:
        pass

    @abstractmethod
    def record_action_audit_noop(self, *, user_id: str, topic_title: str, raw_user_query: str, rewritten_user_query: str, response_text: str, intent: str, response_type: str, parent_hop_id: str | None = None) -> RepositoryTransactionResult:
        pass

    @abstractmethod
    def transactional_knowledge_actions(self, *, user_id: str, topic_title: str, raw_user_query: str, rewritten_user_query: str, response_text: str, actions: list[ValidatedKnowledgeAction], parent_hop_id: str | None = None) -> RepositoryTransactionResult:
        pass

    @abstractmethod
    def transactional_reminder_actions(self, *, user_id: str, topic_title: str, raw_user_query: str, rewritten_user_query: str, response_text: str, actions: list[ValidatedReminderAction], parent_hop_id: str | None = None) -> RepositoryTransactionResult:
        pass

    @abstractmethod
    def list_reminder_candidates(self, user_id: str, statuses: tuple[str, ...], time_window: tuple[datetime, datetime] | None, limit: int) -> list[ReminderCandidateSummary]:
        pass

    @abstractmethod
    def table_count(self, table_name: str) -> int:
        pass

    @abstractmethod
    def list_notifications(self, *, user_id: str, include_deleted: bool=False) -> list[dict[str, Any]]:
        pass

    @abstractmethod
    def update_notification_ui_status(self, *, user_id: str, notification_id: str, ui_status: str) -> dict[str, Any]:
        pass

    @abstractmethod
    def mark_notification_delivery_sent(self, *, user_id: str, notification_id: str) -> dict[str, Any]:
        pass

    @abstractmethod
    def mark_notification_delivery_failed(self, *, user_id: str, notification_id: str, error_message: str, retrying: bool = False) -> dict[str, Any]:
        pass

    @abstractmethod
    def claim_idempotency_key(self, *, user_id: str, idempotency_key: str, payload_hash: str) -> IdempotencyClaimResult:
        pass

    @abstractmethod
    def complete_idempotency_request(self, *, request_id: str, stored_response_json: str) -> None:
        pass

    @abstractmethod
    def fail_idempotency_request(self, *, request_id: str, error_message: str | None = None) -> None:
        pass

    @abstractmethod
    def create_pending_confirmation(self, *, user_id: str, action_type: str, target_entity_type: str, target_entity_id: str | None, proposed_action: dict[str, Any], target_snapshot: dict[str, Any], expires_at: str) -> dict[str, Any]:
        pass

    @abstractmethod
    def load_pending_confirmation(self, *, user_id: str, confirmation_token: str, now_value: str) -> dict[str, Any]:
        pass

    @abstractmethod
    def mark_confirmation_confirmed(self, *, user_id: str, confirmation_token: str) -> None:
        pass

    @abstractmethod
    def list_reminders(self, *, user_id: str, status: str | None=None, from_time: str | None=None, to_time: str | None=None) -> list[dict[str, Any]]:
        pass

    @abstractmethod
    def append_reminder_reply(self, *, user_id: str, reminder_id: str, notification_id: str, reply_text: str, response_text: str) -> HopWrite:
        pass

    @abstractmethod
    def claim_outbox_jobs(self, *, max_attempts: int, batch_size: int, retry_cutoff: str) -> list[dict[str, Any]]:
        pass

    @abstractmethod
    def release_stale_processing_jobs(self, *, max_attempts: int, timeout_cutoff: str) -> None:
        pass

    @abstractmethod
    def mark_outbox_job_completed(self, *, job_id: str) -> None:
        pass

    @abstractmethod
    def mark_outbox_job_failed(self, *, job_id: str, error_message: str) -> None:
        pass

    @abstractmethod
    def load_outbox_entity(self, *, entity_type: str, entity_id: str) -> OutboxIndexPayload:
        pass

    @abstractmethod
    def hydrate_knowledge_retrieval_results(self, *, user_id: str, results: list[RetrievalResult]) -> list[RetrievalResult]:
        pass

    @abstractmethod
    def hydrate_conversation_retrieval_results(self, *, user_id: str, results: list[RetrievalResult]) -> list[RetrievalResult]:
        pass

    @abstractmethod
    def list_all_outbox_entities(self) -> list[tuple[str, str]]:
        pass
