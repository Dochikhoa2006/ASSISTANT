from sqlalchemy import (
    MetaData,
    Table,
    Column,
    String,
    Text,
    Integer,
    DateTime,
    ForeignKey,
    CheckConstraint,
    UniqueConstraint,
    Index,
)

metadata = MetaData()

conversation_topics = Table(
    "conversation_topics",
    metadata,
    Column("topic_id", String, primary_key=True),
    Column("user_id", String, nullable=False),
    Column("title", String, nullable=False),
    Column("topic_summary", Text, nullable=False),
    Column("state_summary", Text, nullable=False),
    Column("entities_json", Text, nullable=False),
    Column("last_hop_id", String, nullable=True),
    Column("status", String, nullable=False),
    Column("created_at", String, nullable=False),
    Column("updated_at", String, nullable=False),
    Column("version", Integer, nullable=False),
    CheckConstraint("status IN ('active', 'archived', 'deleted')"),
)

conversation_hops = Table(
    "conversation_hops",
    metadata,
    Column("hop_id", String, primary_key=True),
    Column("user_id", String, nullable=False),
    Column("topic_id", String, ForeignKey("conversation_topics.topic_id"), nullable=False),
    Column("intent", String, nullable=False),
    Column("raw_user_query", Text, nullable=False),
    Column("rewritten_user_query", Text, nullable=False),
    Column("raw_response", Text, nullable=False),
    Column("response_type", String, nullable=False),
    Column("supporting_questions_json", Text, nullable=False),
    Column("previous_hop_id", String, nullable=True),
    Column("parent_hop_id", String, nullable=True),
    Column("root_hop_id", String, nullable=True),
    Column("branch_id", String, nullable=True),
    Column("depth_from_root", Integer, nullable=False, default=0),
    Column("entities_json", Text, nullable=False),
    Column("outbox_job_id", String, nullable=True),
    Column("created_at", String, nullable=False),
    CheckConstraint(
        "intent IN ('clarification', 'general_response', 'knowledge_facts', 'reminder')",
        name="ck_conversation_hops_intent",
    ),
    CheckConstraint(
        "response_type IN ('clarification', 'normal', 'knowledge_action', 'reminder_action', 'reminder_reply', 'error', 'safe_noop')",
        name="ck_conversation_hops_response_type",
    ),
)

knowledge_topics = Table(
    "knowledge_topics",
    metadata,
    Column("knowledge_topic_id", String, primary_key=True),
    Column("user_id", String, nullable=False),
    Column("title", String, nullable=False),
    Column("description", Text, nullable=False),
    Column("entities_json", Text, nullable=False),
    Column("created_at", String, nullable=False),
    Column("updated_at", String, nullable=False),
    Column("version", Integer, nullable=False),
)

knowledge_sources = Table(
    "knowledge_sources",
    metadata,
    Column("source_id", String, primary_key=True),
    Column("user_id", String, nullable=False),
    Column("filename", String, nullable=False),
    Column("file_type", String, nullable=False),
    Column("upload_time", String, nullable=False),
    Column("processing_status", String, nullable=False),
    Column("content_hash", String, nullable=False),
    Column("metadata_json", Text, nullable=False),
    Column("version", Integer, nullable=False),
    Column("is_deleted", Integer, nullable=False),
    Column("created_at", String, nullable=False),
    Column("updated_at", String, nullable=False),
    CheckConstraint("processing_status IN ('pending', 'processing', 'indexed', 'failed', 'deleted')", name="ck_knowledge_sources_processing_status"),
    CheckConstraint("is_deleted IN (0, 1)", name="ck_knowledge_sources_is_deleted"),
)

knowledge_chunks = Table(
    "knowledge_chunks",
    metadata,
    Column("chunk_id", String, primary_key=True),
    Column("knowledge_topic_id", String, ForeignKey("knowledge_topics.knowledge_topic_id"), nullable=False),
    Column("user_id", String, nullable=False),
    Column("source_id", String, ForeignKey("knowledge_sources.source_id"), nullable=True),
    Column("chunk_index", Integer, nullable=False),
    Column("raw_text", Text, nullable=False),
    Column("normalized_text", Text, nullable=False),
    Column("summary", Text, nullable=False),
    Column("metadata_json", Text, nullable=False),
    Column("content_hash", String, nullable=False),
    Column("is_deleted", Integer, nullable=False),
    Column("created_at", String, nullable=False),
    Column("updated_at", String, nullable=False),
    Column("version", Integer, nullable=False),
    Column("replaces_chunk_id", String, nullable=True),
    Column("replaced_by_chunk_id", String, nullable=True),
    Column("change_reason", Text, nullable=True),
    Column("modified_by_user_query", Text, nullable=True),
    CheckConstraint("is_deleted IN (0, 1)"),
    UniqueConstraint("user_id", "knowledge_topic_id", "content_hash", name="uq_chunks_user_topic_hash"),
)

generated_artifacts = Table(
    "generated_artifacts",
    metadata,
    Column("artifact_id", String, primary_key=True),
    Column("user_id", String, nullable=False),
    Column("conversation_hop_id", String, nullable=True),
    Column("file_type", String, nullable=False),
    Column("filename", String, nullable=False),
    Column("storage_path", Text, nullable=False),
    Column("storage_url", Text, nullable=False),
    Column("metadata_json", Text, nullable=False),
    Column("created_at", String, nullable=False),
    Column("expires_at", String, nullable=True),
    Column("status", String, nullable=False),
    CheckConstraint("file_type IN ('xlsx', 'pdf', 'pptx', 'txt', 'csv')", name="ck_generated_artifacts_file_type"),
    CheckConstraint("status IN ('created', 'deleted', 'failed')", name="ck_generated_artifacts_status"),
)

platform_deliveries = Table(
    "platform_deliveries",
    metadata,
    Column("delivery_id", String, primary_key=True),
    Column("user_id", String, nullable=False),
    Column("conversation_hop_id", String, nullable=True),
    Column("channel", String, nullable=False),
    Column("status", String, nullable=False),
    Column("recipient", String, nullable=False),
    Column("message_json", Text, nullable=False),
    Column("error_message", Text, nullable=True),
    Column("created_at", String, nullable=False),
    CheckConstraint("channel IN ('gmail', 'zalo', 'telegram')", name="ck_platform_deliveries_channel"),
)
Index("idx_platform_deliveries_user_created", platform_deliveries.c.user_id, platform_deliveries.c.created_at)

reminders = Table(
    "reminders",
    metadata,
    Column("reminder_id", String, primary_key=True),
    Column("user_id", String, nullable=False),
    Column("source_topic_id", String, nullable=True),
    Column("source_hop_id", String, nullable=True),
    Column("reminder_time", String, nullable=False),
    Column("raw_reminder", Text, nullable=False),
    Column("reminder_summary", Text, nullable=False),
    Column("subject", String, nullable=False),
    Column("supporting_question", Text, nullable=True),
    Column("supporting_response", Text, nullable=True),
    Column("user_timezone", String, nullable=False, default="UTC"),
    Column("original_time_text", Text, nullable=True),
    Column("recurrence_rule", Text, nullable=True),
    Column("recurrence_timezone", String, nullable=True),
    Column("next_fire_time", String, nullable=True),
    Column("last_fire_time", String, nullable=True),
    Column("parent_recurring_reminder_id", String, nullable=True),
    Column("status", String, nullable=False),
    Column("created_at", String, nullable=False),
    Column("updated_at", String, nullable=False),
    Column("version", Integer, nullable=False),
    CheckConstraint(
        "status IN ('scheduled', 'notified', 'cancelled', 'dismissed', 'completed')",
        name="ck_reminders_status",
    ),
)

reminder_notifications = Table(
    "reminder_notifications",
    metadata,
    Column("notification_id", String, primary_key=True),
    Column("reminder_id", String, ForeignKey("reminders.reminder_id"), nullable=False),
    Column("user_id", String, nullable=False),
    Column("ui_status", String, nullable=False),
    Column("delivery_status", String, nullable=False, default="pending"),
    Column("delivery_attempts", Integer, nullable=False, default=0),
    Column("last_delivery_error", Text, nullable=True),
    Column("created_at", String, nullable=False),
    Column("read_at", String, nullable=True),
    Column("deleted_at", String, nullable=True),
    Column("sent_at", String, nullable=True),
    Column("fire_time", String, nullable=True),
    CheckConstraint("ui_status IN ('unread', 'read', 'deleted')", name="ck_reminder_notifications_ui_status"),
    CheckConstraint(
        "delivery_status IN ('pending', 'sent', 'failed', 'retrying')",
        name="ck_reminder_notifications_delivery_status",
    ),
)

mutation_requests = Table(
    "mutation_requests",
    metadata,
    Column("request_id", String, primary_key=True),
    Column("user_id", String, nullable=False),
    Column("idempotency_key", String, nullable=False),
    Column("payload_hash", String, nullable=False),
    Column("status", String, nullable=False),
    Column("stored_response_json", Text, nullable=True),
    Column("created_at", String, nullable=False),
    Column("updated_at", String, nullable=False),
    CheckConstraint("status IN ('in_progress', 'completed', 'failed')", name="ck_mutation_requests_status"),
    UniqueConstraint("user_id", "idempotency_key", name="uq_mutation_requests_user_key"),
)

pending_action_confirmations = Table(
    "pending_action_confirmations",
    metadata,
    Column("confirmation_token", String, primary_key=True),
    Column("user_id", String, nullable=False),
    Column("action_type", String, nullable=False),
    Column("target_entity_type", String, nullable=False),
    Column("target_entity_id", String, nullable=True),
    Column("proposed_action_json", Text, nullable=False),
    Column("target_snapshot_json", Text, nullable=False),
    Column("expires_at", String, nullable=False),
    Column("status", String, nullable=False),
    Column("created_at", String, nullable=False),
    Column("confirmed_at", String, nullable=True),
    CheckConstraint("status IN ('pending', 'confirmed', 'cancelled', 'expired')", name="ck_pending_action_confirmations_status"),
)

indexing_outbox = Table(
    "indexing_outbox",
    metadata,
    Column("job_id", String, primary_key=True),
    Column("entity_type", String, nullable=False),
    Column("entity_id", String, nullable=False),
    Column("operation", String, nullable=False),
    Column("status", String, nullable=False),
    Column("retry_count", Integer, nullable=False),
    Column("error_message", Text, nullable=True),
    Column("created_at", String, nullable=False),
    Column("updated_at", String, nullable=False),
    CheckConstraint("entity_type IN ('conversation_hop', 'knowledge_chunk')"),
    CheckConstraint("operation IN ('upsert', 'delete')"),
    CheckConstraint("status IN ('pending', 'processing', 'completed', 'failed')"),
)

# Indexes
Index("idx_topics_user_status", conversation_topics.c.user_id, conversation_topics.c.status)
Index("idx_topics_last_hop", conversation_topics.c.user_id, conversation_topics.c.last_hop_id)
Index("idx_hops_topic_prev", conversation_hops.c.user_id, conversation_hops.c.topic_id, conversation_hops.c.previous_hop_id)
Index("idx_hops_topic_parent", conversation_hops.c.user_id, conversation_hops.c.topic_id, conversation_hops.c.parent_hop_id)
Index("idx_hops_user_root", conversation_hops.c.user_id, conversation_hops.c.root_hop_id)
Index("idx_hops_user_branch", conversation_hops.c.user_id, conversation_hops.c.branch_id)
Index("idx_chunks_user_topic_deleted", knowledge_chunks.c.user_id, knowledge_chunks.c.knowledge_topic_id, knowledge_chunks.c.is_deleted)
Index("idx_sources_user_status", knowledge_sources.c.user_id, knowledge_sources.c.processing_status)
Index("idx_sources_user_hash", knowledge_sources.c.user_id, knowledge_sources.c.content_hash)
Index("idx_chunks_user_source", knowledge_chunks.c.user_id, knowledge_chunks.c.source_id)
Index("idx_artifacts_user_status", generated_artifacts.c.user_id, generated_artifacts.c.status)
Index("idx_reminders_user_status_time", reminders.c.user_id, reminders.c.status, reminders.c.reminder_time)
Index("idx_reminders_user_status_next_fire", reminders.c.user_id, reminders.c.status, reminders.c.next_fire_time)
Index("idx_reminders_user_parent_recurring", reminders.c.user_id, reminders.c.parent_recurring_reminder_id)
Index("idx_reminders_user_updated", reminders.c.user_id, reminders.c.updated_at)
Index("idx_reminders_user_created", reminders.c.user_id, reminders.c.created_at)
Index("idx_reminders_user_source_hop", reminders.c.user_id, reminders.c.source_hop_id)
Index("idx_mutation_requests_user_key", mutation_requests.c.user_id, mutation_requests.c.idempotency_key)
Index("idx_confirmations_user_status", pending_action_confirmations.c.user_id, pending_action_confirmations.c.status)
