from __future__ import annotations
"""Shared contracts used by every pipeline stage and branch."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional


class Intent(str, Enum):
    CLARIFICATION = "clarification"
    GENERAL_RESPONSE = "general_response"
    KNOWLEDGE_FACTS = "knowledge_facts"
    REMINDER = "reminder"


class LastQAPath(str, Enum):
    NO_LAST_QA = "no_last_qa"
    CLARIFICATION_CHECK = "clarification_check"
    LATEST_CONTEXT_INTERACTION = "latest_context_interaction"
    BROAD_RETRIEVAL_REQUIRED = "broad_retrieval_required"


class LastQAInteractionType(str, Enum):
    NORMAL_FOLLOW_UP = "normal_follow_up"
    SUPPORTING_QUESTION_ANSWER = "supporting_question_answer"
    CLARIFICATION_ANSWER = "clarification_answer"
    REMINDER_NOTIFICATION_REPLY = "reminder_notification_reply"
    OUTBOUND_MESSAGE_ACTION = "outbound_message_action"


class OutboundFollowUpAction(str, Enum):
    """A semantic action on the one active outbound message envelope."""

    SEND = "send"
    REVISE = "revise"
    REVISE_AND_SEND = "revise_and_send"


class QuestionSource(str, Enum):
    CLARIFICATION_QUESTION = "clarification_question"
    HUMAN_SUPPORTING_QUESTION = "human_supporting_question"
    REMINDER_SUPPORTING_QUESTION = "reminder_supporting_question"
    NONE = "none"


class KnowledgeAction(str, Enum):
    ADD = "add"
    DELETE = "delete"
    MODIFY = "modify"


class ReminderAction(str, Enum):
    ADD = "add"
    DELETE = "delete"
    TURN_ON = "turn_on"
    TURN_OFF = "turn_off"
    MODIFY = "modify"


class ReminderStatus(str, Enum):
    SCHEDULED = "scheduled"
    NOTIFIED = "notified"
    CANCELLED = "cancelled"
    DISMISSED = "dismissed"
    COMPLETED = "completed"


class NotificationUIStatus(str, Enum):
    UNREAD = "unread"
    READ = "read"
    DELETED = "deleted"


class NotificationDeliveryStatus(str, Enum):
    PENDING = "pending"
    SENT = "sent"
    FAILED = "failed"
    RETRYING = "retrying"


class MutationRequestStatus(str, Enum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class ConfirmationStatus(str, Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class KnowledgeSourceStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"
    DELETED = "deleted"


class ArtifactStatus(str, Enum):
    CREATED = "created"
    DELETED = "deleted"
    FAILED = "failed"


class ArtifactFileType(str, Enum):
    XLSX = "xlsx"
    PDF = "pdf"
    PPTX = "pptx"
    TXT = "txt"
    CSV = "csv"


class LifecycleStatus(str, Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"
    EXPIRED = "expired"


class HealthStatus(str, Enum):
    OK = "ok"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class DependencyStatus(str, Enum):
    OK = "ok"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    NOT_CONFIGURED = "not_configured"


class RecurrenceFrequency(str, Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class TraceStage(str, Enum):
    API_RECEIVED = "api_received"
    AUTH_VERIFIED = "auth_verified"
    RATE_LIMIT_CHECKED = "rate_limit_checked"
    REWRITE = "rewrite"
    LAST_QA = "last_qa"
    RETRIEVAL = "retrieval"
    RETRIEVAL_BM25 = "retrieval_bm25"
    RETRIEVAL_CHROMA = "retrieval_chroma"
    RETRIEVAL_MERGE = "retrieval_merge"
    RERANK = "rerank"
    SQL_VALIDATION = "sql_validation"
    CLASSIFICATION = "classification"
    ACTION_DETECTION = "action_detection"
    RISKY_ACTION_VALIDATION = "risky_action_validation"
    BRANCH_EXECUTION = "branch_execution"
    SQL_TRANSACTION = "sql_transaction"
    OUTBOX_ENQUEUE = "outbox_enqueue"
    BUNDLING = "bundling"
    API_RESPONSE = "api_response"
    API_TOTAL = "api_total"


class OperationStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class ResponseType(str, Enum):
    CLARIFICATION = "clarification"
    NORMAL = "normal"
    KNOWLEDGE_ACTION = "knowledge_action"
    REMINDER_ACTION = "reminder_action"
    REMINDER_REPLY = "reminder_reply"
    ERROR = "error"
    SAFE_NOOP = "safe_noop"


class ExpectedResponseType(str, Enum):
    FREE_TEXT_ANSWER              = "free_text_answer"
    YES_NO_ANSWER                 = "yes_no_answer"
    SELECTION_ANSWER              = "selection_answer"
    TIME_OR_DATE_ANSWER           = "time_or_date_answer"
    PREFERENCE_ANSWER             = "preference_answer"
    CONFIRMATION_ANSWER           = "confirmation_answer"
    REPLACEMENT_TEXT_ANSWER       = "replacement_text_answer"
    STRUCTURED_CONTENT_ANSWER     = "structured_content_answer"
    REMINDER_FOLLOWUP_ANSWER      = "reminder_followup_answer"
    HUMAN_SUPPORTING_FOLLOWUP_ANSWER = "human_supporting_followup_answer"
    CLARIFICATION_SLOT_ANSWER     = "clarification_slot_answer"
    UNKNOWN                       = "unknown"


class ActionValidationResult(str, Enum):
    EXECUTE = "execute"
    SKIP_NOT_FOUND = "skip_not_found"
    SKIP_ALREADY_EXISTS = "skip_already_exists"
    CLARIFY_AMBIGUOUS_TARGET = "clarify_ambiguous_target"
    CLARIFY_MISSING_FIELDS = "clarify_missing_fields"
    REJECT_UNSUPPORTED_OPERATION = "reject_unsupported_operation"
    REJECT_UNSAFE_TRANSITION = "reject_unsafe_transition"


class AnswerMode(str, Enum):
    SUPPORT_QUESTION_ANSWER = "support_question_answer"
    NEW_CONVERSATION = "new_conversation"
    FOLLOW_UP_CONVERSATION = "follow_up_conversation"


class OutboxEntityType(str, Enum):
    CONVERSATION_HOP = "conversation_hop"
    KNOWLEDGE_CHUNK = "knowledge_chunk"


class OutboxOperation(str, Enum):
    UPSERT = "upsert"
    DELETE = "delete"


@dataclass(frozen=True)
class GeneratedQuestion:
    text: str
    source: QuestionSource
    purpose: str
    confidence: float
    should_ask: bool = True
    expected_response_type: ExpectedResponseType = ExpectedResponseType.UNKNOWN


@dataclass(frozen=True)
class ChatRequest:
    user_id: str
    raw_query: str
    platform_context: dict[str, Any] = field(default_factory=dict)
    reminder_id: Optional[str] = None
    notification_id: Optional[str] = None
    reply_text: Optional[str] = None
    parent_hop_id: Optional[str] = None
    idempotency_key: Optional[str] = None
    confirmation_token: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    # Appended for positional-call compatibility. It scopes volatile Last-QA
    # state and semantic history so two tabs for the same authenticated user
    # cannot borrow one another's pending action or recent exchange.
    conversation_id: Optional[str] = None

    def __post_init__(self) -> None:
        """Canonicalize cursor identifiers before hashing or repository use."""

        object.__setattr__(self, "user_id", str(self.user_id or "").strip())
        if not self.user_id:
            raise ValueError("user_id must be a non-empty stable identity")
        for field_name in (
            "conversation_id",
            "parent_hop_id",
            "reminder_id",
            "notification_id",
            "idempotency_key",
            "confirmation_token",
        ):
            value = getattr(self, field_name)
            normalized = str(value).strip() if value is not None else ""
            object.__setattr__(self, field_name, normalized or None)


@dataclass(frozen=True)
class OutboundMessageState:
    """Safe temporary state for a user-owned outbound message.

    Credentials and storage paths are intentionally excluded. Artifact IDs are
    rehydrated through the user-scoped repository immediately before delivery.
    """

    channel: str
    status: str
    recipients: tuple[str, ...]
    subject: str
    body: str
    artifact_ids: tuple[str, ...] = field(default_factory=tuple)
    attachment_filenames: tuple[str, ...] = field(default_factory=tuple)
    source_topic_id: Optional[str] = None
    source_hop_id: Optional[str] = None
    excluded_recipients: tuple[str, ...] = field(default_factory=tuple)
    delivered_recipients: tuple[str, ...] = field(default_factory=tuple)
    refused_recipients: tuple[str, ...] = field(default_factory=tuple)


@dataclass
class LastQAState:
    last_user_query: str
    last_response: str
    response_type: ResponseType
    supporting_questions: list[GeneratedQuestion] = field(default_factory=list)
    clarification_question: GeneratedQuestion | None = None
    reminder_supporting_question: GeneratedQuestion | None = None
    linked_topic_id: Optional[str] = None
    linked_hop_id: Optional[str] = None
    expected_response_type: ExpectedResponseType | None = None
    # Canonical SQL-derived context for an exact reminder-notification reply.
    # It is kept separate from the free-form response so the next turn can
    # identify both supporting-question and purpose-driven replies precisely.
    reminder_state: dict[str, Any] | None = None
    reminder_state_hash: Optional[str] = None
    outbound_state: OutboundMessageState | None = None


@dataclass(frozen=True)
class RetrievalResult:
    entity_type: str
    entity_id: str
    source_store_evidence: dict[str, Any]
    rerank_score: float
    confidence: float
    validation_status: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OutboxIndexPayload:
    user_id: str
    entity_type: str
    entity_id: str
    text: str
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)


@dataclass(frozen=True)
class ApprovedConversationContext:
    approved_conversation_history: list[dict[str, Any]]
    human_supporting_questions: list[GeneratedQuestion]
    reminder_supporting_questions: list[GeneratedQuestion]
    clarification_question_context: GeneratedQuestion | None
    extracted_expected_response_types: list[ExpectedResponseType]
    
    conversation_retrieval_ran: bool
    conversation_context_status: Literal["not_run", "approved", "all_rejected", "empty"]
    approved_conversation_count: int
    top_hop_rerank_score: float | None = None

    _internal_selected_topic_candidates: list[str] = field(default_factory=list)
    _internal_selected_hop_candidates: list[str] = field(default_factory=list)
    _rejected_conversation_ids: tuple[str, ...] = field(default_factory=tuple)
    _validation_summary: str = ""


@dataclass
class OperationResult:
    action_type: str
    status: OperationStatus
    user_facing_summary: str
    affected_sql_entity_ids: list[str] = field(default_factory=list)
    database_write_result: dict[str, Any] = field(default_factory=dict)
    indexing_job_result: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RepositoryActionResult:
    action_id: str
    action_type: str
    status: Literal[
        "committed",
        "rolled_back",
        "conflict",
        "validation_failed",
        "not_found",
        "skipped",
        "error",
        "pending_confirmation",
    ]
    domain_entity_type: Literal[
        "knowledge_chunk",
        "knowledge_topic",
        "reminder",
        "conversation_hop",
        "none",
    ] | None = None
    domain_entity_id: Optional[str] = None
    audit_hop_id: Optional[str] = None
    indexing_outbox_ids: tuple[str, ...] = ()
    user_safe_summary: Optional[str] = None
    reason_summary: Optional[str] = None


@dataclass(frozen=True)
class AuthContext:
    user_id: str
    claims: dict[str, Any] = field(default_factory=dict)
    scopes: tuple[str, ...] = ()


@dataclass(frozen=True)
class RateLimitResult:
    allowed: bool
    retry_after_seconds: int = 0


@dataclass(frozen=True)
class IdempotencyClaimResult:
    status: Literal["started", "replay", "in_progress", "conflict", "failed_retry"]
    request_id: str
    stored_response_json: Optional[str] = None
    reason: Optional[str] = None


@dataclass(frozen=True)
class TraceStageSummary:
    stage: str
    latency_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TraceSummary:
    request_id: str
    total_latency_ms: float
    stages: list[TraceStageSummary] = field(default_factory=list)


@dataclass(frozen=True)
class MetricSnapshot:
    counters: dict[str, int]
    gauges: dict[str, float]
    latency_ms: dict[str, dict[str, float]]
    generated_at: str


@dataclass(frozen=True)
class DependencyHealth:
    name: str
    status: DependencyStatus
    detail: str = ""
    latency_ms: float | None = None


@dataclass(frozen=True)
class HealthPayload:
    status: HealthStatus
    dependencies: list[DependencyHealth]
    outbox_pending_count: int = 0
    outbox_failed_count: int = 0


@dataclass(frozen=True)
class DriftIssue:
    kind: str
    entity_type: str
    entity_id: str | None = None
    detail: str = ""


@dataclass(frozen=True)
class DriftReport:
    status: Literal["ok", "drift_detected"]
    user_id: str | None
    sql_counts: dict[str, int]
    bm25_counts: dict[str, int]
    chroma_counts: dict[str, int]
    failed_outbox_count: int
    issues: list[DriftIssue] = field(default_factory=list)
    repaired: bool = False


@dataclass(frozen=True)
class EvaluationReport:
    case_count: int
    top_1_accuracy: float
    top_3_accuracy: float
    wrong_target_rate: float
    clarification_rate: float
    false_mutation_rate: float
    retrieval_empty_rate: float
    average_latency_ms: float
    passed: bool
    cases: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class RepositoryTransactionResult:
    committed: bool
    results: tuple[RepositoryActionResult, ...]
    audit_topic_id: Optional[str] = None
    audit_hop_id: Optional[str] = None
    indexing_outbox_ids: tuple[str, ...] = ()
    error_type: Optional[str] = None
    reason_summary: Optional[str] = None


@dataclass(frozen=True)
class ValidatedKnowledgeAction:
    action: KnowledgeAction
    validation_result: ActionValidationResult

    target_chunk_ids: tuple[str, ...] = ()
    target_topic_ids: tuple[str, ...] = ()

    observed_versions: dict[str, int] = field(default_factory=dict)
    observed_is_deleted: dict[str, bool] = field(default_factory=dict)

    knowledge_text: Optional[str] = None
    replacement_text: Optional[str] = None
    
    # Legacy fields (kept for backward compatibility with ValidatedActionBuilder if it hasn't been updated yet)
    new_text: Optional[str] = None
    target_status: Optional[str] = None
    topic_title: Optional[str] = None
    target_description: Optional[str] = None

    confidence: float = 0.0
    matched_fields: tuple[str, ...] = ()
    reason_summary: Optional[str] = None
    requires_hitl: bool = False
    factuality_concern: bool = False
    hitl_reason: Optional[str] = None
    clarification_question: Optional[str] = None
    pending_confirmation_token: Optional[str] = None


@dataclass(frozen=True)
class RetrievalCandidateAssessment:
    candidate_key: str
    matches_target: bool
    action_compatible: bool
    confidence: float
    matched_fields: tuple[str, ...]
    reason_summary: str
    matched_text: str = ""


@dataclass(frozen=True)
class LLMRetrievalValidationResult:
    operation: str
    validation_result: ActionValidationResult
    selected_candidate_keys: tuple[str, ...]
    confidence: float
    ambiguous: bool
    reason_summary: str
    candidate_assessments: tuple[RetrievalCandidateAssessment, ...]
    should_execute: bool = False
    requires_hitl: bool = False
    factuality_concern: bool = False
    hitl_reason: Optional[str] = None
    clarification_question: Optional[str] = None


@dataclass(frozen=True)
class KnowledgeValidationCandidate:
    candidate_key: str
    knowledge_chunk_id: str
    knowledge_topic_id: str
    text: str
    source_title: Optional[str]
    retrieval_score: float
    rerank_score: float | None
    is_deleted: bool
    user_id: str


@dataclass(frozen=True)
class ReminderValidationCandidate:
    candidate_key: str
    reminder_id: str
    subject: str
    reminder_summary: Optional[str]
    raw_reminder: Optional[str]
    reminder_time: datetime | None
    status: str
    deterministic_score: float
    observed_version: int | None
    observed_status: str


@dataclass(frozen=True)
class LLMKnowledgeCandidatePayload:
    candidate_key: str
    text_excerpt: str
    source_title: Optional[str]
    retrieval_score: float
    rerank_score: float | None
    matched_fields: tuple[str, ...]


@dataclass(frozen=True)
class LLMReminderCandidatePayload:
    candidate_key: str
    subject: str
    reminder_summary: Optional[str]
    reminder_time: datetime | None
    status: str
    deterministic_score: float
    matched_fields: tuple[str, ...]


@dataclass(frozen=True)
class ReminderTargetQuery:
    raw_text: str
    target_subject_terms: tuple[str, ...] = ()
    target_entities: tuple[str, ...] = ()
    target_date: datetime | None = None
    target_time_range: tuple[datetime, datetime] | None = None
    target_daypart: Optional[str] = None


@dataclass(frozen=True)
class ReminderCandidateSummary:
    reminder_id: str
    subject: str
    reminder_summary: str
    raw_reminder: str
    reminder_time: datetime | None
    status: str
    created_at: datetime
    updated_at: datetime | None
    version: int | None
    is_deleted: bool | None = False


@dataclass(frozen=True)
class ReminderCandidateScore:
    candidate_id: str
    exact_score: float = 0.0
    token_score: float = 0.0
    fuzzy_score: float = 0.0
    time_score: float = 0.0
    entity_score: float = 0.0
    status_score: float = 0.0
    recency_score: float = 0.0
    final_score: float = 0.0
    matched_fields: tuple[str, ...] = ()
    reason_summary: Optional[str] = None


@dataclass(frozen=True)
class ReminderTargetResolution:
    validation_result: ActionValidationResult
    target_reminder_ids: tuple[str, ...] = ()
    top_score: ReminderCandidateScore | None = None
    clarification_candidates: tuple[ReminderCandidateSummary, ...] = ()
    chosen_candidate: ReminderCandidateSummary | None = None


@dataclass(frozen=True)
class ValidatedReminderAction:
    action: ReminderAction
    validation_result: ActionValidationResult

    target_reminder_ids: tuple[str, ...] = ()
    observed_status: Optional[str] = None
    observed_version: int | None = None
    observed_reminder_time: datetime | None = None

    subject: Optional[str] = None
    # event_time is the original timestamp supplied for this reminder.  The
    # background timing planner derives reminder_time from it asynchronously.
    event_time: datetime | None = None
    reminder_time: datetime | None = None
    reminder_summary: Optional[str] = None
    raw_reminder: Optional[str] = None
    supporting_question: Optional[str] = None
    supporting_response: Optional[str] = None
    user_timezone: Optional[str] = None
    original_time_text: Optional[str] = None
    recurrence_rule: Optional[str] = None
    recurrence_timezone: Optional[str] = None
    next_fire_time: datetime | None = None
    parent_recurring_reminder_id: Optional[str] = None
    # Legacy callers default to background timing. The reminder mutation
    # pipeline disables it only when an explicit notification timestamp has
    # already been validated and must not be overwritten by autoscan.
    timing_plan_required: bool = True

    replacement_subject: Optional[str] = None
    replacement_time: datetime | None = None
    replacement_summary: Optional[str] = None
    replacement_recurrence_rule: Optional[str] = None
    replacement_recurrence_timezone: Optional[str] = None

    confidence: float = 0.0
    matched_fields: tuple[str, ...] = ()
    reason_summary: Optional[str] = None
    requires_hitl: bool = False
    factuality_concern: bool = False
    hitl_reason: Optional[str] = None
    clarification_question: Optional[str] = None


@dataclass(frozen=True)
class PendingReminderTiming:
    """A reminder claimed by autoscan for one durable timing-plan attempt."""

    reminder_id: str
    user_id: str
    source_time: datetime
    subject: str
    raw_reminder: str
    user_timezone: str
    recurrence_rule: str | None
    version: int


@dataclass(frozen=True)
class PendingReminderSupportingQuestion:
    """A reminder awaiting one durable autoscan question-plan decision."""

    reminder_id: str
    user_id: str
    subject: str
    reminder_summary: str
    raw_reminder: str
    notification_time: datetime | None
    event_time: datetime | None
    user_timezone: str
    recurrence_rule: str | None
    version: int


@dataclass
class BranchResult:
    response_type: ResponseType
    normal_response_text: Optional[str] = None
    clarification_question: GeneratedQuestion | None = None
    human_supporting_questions: list[GeneratedQuestion] = field(default_factory=list)
    reminder_supporting_question: GeneratedQuestion | None = None
    question_source: QuestionSource | None = None
    knowledge_operation_results: list[RepositoryActionResult] | list[OperationResult] = field(default_factory=list)
    reminder_operation_results: list[RepositoryActionResult] | list[OperationResult] = field(default_factory=list)
    human_in_the_loop_result: dict[str, Any] | None = None
    platform_payload: dict[str, Any] = field(default_factory=dict)
    fallback_or_error_message: Optional[str] = None
    database_write_result: dict[str, Any] = field(default_factory=dict)
    indexing_job_result: dict[str, Any] = field(default_factory=dict)
    actions_pending_confirmation: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    linked_topic_id: Optional[str] = None
    linked_hop_id: Optional[str] = None


@dataclass(frozen=True)
class BundledResponse:
    final_chat_text: str
    response_type: ResponseType
    last_qa_state: LastQAState
    platform_payload: dict[str, Any] = field(default_factory=dict)
    persistence_instructions: dict[str, Any] = field(default_factory=dict)
    request_id: Optional[str] = None
    conversation_topic_id: Optional[str] = None
    conversation_hop_id: Optional[str] = None
    actions_committed: list[dict[str, Any]] = field(default_factory=list)
    actions_pending_confirmation: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    trace_summary: TraceSummary | None = None
    # Appended for positional-call compatibility with earlier response fields.
    conversation_id: Optional[str] = None


@dataclass(frozen=True)
class LastQAResolution:
    path: LastQAPath
    rewritten_query: str
    state: LastQAState | None
    did_merge_query: bool
    skip_broad_retrieval: bool
    confidence: float = 0.0

    interaction_type: LastQAInteractionType | None = None
    question_source: QuestionSource = QuestionSource.NONE

    linked_topic_id: Optional[str] = None
    linked_hop_id: Optional[str] = None
    matched_question: Optional[str] = None

    reminder_id: Optional[str] = None
    notification_id: Optional[str] = None
    source_topic_id: Optional[str] = None
    source_hop_id: Optional[str] = None

    missing_context: list[str] = field(default_factory=list)
    diagnostic_context: dict[str, Any] = field(default_factory=dict)

    merge_reason: Optional[str] = None
    skip_reason: Optional[str] = None

    is_authoritative_state: bool = False
    outbound_action: OutboundFollowUpAction | None = None


def validate_last_qa_resolution(resolution: LastQAResolution) -> None:
    if resolution.did_merge_query and resolution.skip_broad_retrieval:
        raise ValueError("Last-QA cannot both merge query and skip broad retrieval.")

    if resolution.did_merge_query:
        assert resolution.path == LastQAPath.CLARIFICATION_CHECK
        assert resolution.interaction_type == LastQAInteractionType.CLARIFICATION_ANSWER
        assert resolution.question_source == QuestionSource.CLARIFICATION_QUESTION
        assert resolution.skip_broad_retrieval is False

    if resolution.skip_broad_retrieval:
        assert resolution.path == LastQAPath.LATEST_CONTEXT_INTERACTION
        assert resolution.did_merge_query is False

    if resolution.interaction_type == LastQAInteractionType.OUTBOUND_MESSAGE_ACTION:
        assert resolution.path == LastQAPath.LATEST_CONTEXT_INTERACTION
        assert resolution.skip_broad_retrieval is True
        assert resolution.state is not None
        assert resolution.state.outbound_state is not None
        assert resolution.outbound_action is not None
        assert resolution.is_authoritative_state is True

    if resolution.path == LastQAPath.BROAD_RETRIEVAL_REQUIRED:
        assert resolution.skip_broad_retrieval is False

    if resolution.path == LastQAPath.NO_LAST_QA:
        assert resolution.state is None
        assert resolution.did_merge_query is False
        assert resolution.skip_broad_retrieval is False

@dataclass(frozen=True)
class PipelineContext:
    request: ChatRequest
    rewritten_query: str
    last_qa_state: LastQAState | None
    conversation_results: list[RetrievalResult]
    intent: Intent
    chat_history: list[dict[str, Any]] = field(default_factory=list)
    conversation_retrieval: bool = False
    chat_history_source: Literal["conversation_retrieval", "last_qa"] = "last_qa"
    last_qa_trace: dict[str, Any] = field(default_factory=dict)
    approved_conversation_context: ApprovedConversationContext | None = None
    # Retained solely for non-semantic lifecycle guards such as suppressing an
    # identical consecutive clarification. It must never become prompt,
    # retrieval, routing, or mutation authority after Last-QA rejects it.
    previous_last_qa_state: LastQAState | None = None


@dataclass(frozen=True)
class HopWrite:
    topic_id: str
    hop_id: str
    previous_hop_id: Optional[str] = None
    outbox_job_id: Optional[str] = None


class GeneralSubBranch(str, Enum):
    SUPPORT_QUESTION_ANSWER  = "support_question_answer"
    CONVERSATION_FOLLOW_UP   = "conversation_follow_up"
    NEW_CONVERSATION_TOPIC   = "new_conversation_topic"


@dataclass(frozen=True)
class SubBranchPromptContext:
    sub_branch: GeneralSubBranch
    persistence_mode: "PersistenceMode"
    chat_history_role: str
    response_goal: str
    database_update_mode: str
    allowed_database_updates: tuple[str, ...]
    prohibited_database_updates: tuple[str, ...]
    expected_response_type: ExpectedResponseType = ExpectedResponseType.UNKNOWN


class PersistenceMode(str, Enum):
    APPEND_TO_EXISTING_TOPIC  = "append_to_existing_topic"
    BRANCH_FROM_EXISTING_HOP  = "branch_from_existing_hop"
    CREATE_NEW_TOPIC          = "create_new_topic"


@dataclass(frozen=True)
class GeneralSubBranchDecision:
    sub_branch: GeneralSubBranch
    confidence: float
    persistence_mode: PersistenceMode
    selected_candidate_ref: Optional[str] = None
    selected_topic_id: Optional[str] = None
    selected_hop_id: Optional[str] = None
    selected_parent_hop_id: Optional[str] = None
    reason_summary: str = ""
    risk_flags: tuple[str, ...] = ()
    missing_context: tuple[str, ...] = ()


@dataclass(frozen=True)
class GeneralResponsePersistencePlan:
    persistence_mode: PersistenceMode
    sub_branch: GeneralSubBranch
    topic_id: Optional[str]
    previous_hop_id: Optional[str]
    parent_hop_id: Optional[str]
    reason_summary: str


@dataclass(frozen=True)
class ContentComposerInput:
    user_id: str
    # Legacy compatibility slot. DeterministicContentComposer overwrites it
    # with rewritten_query before any routing, prompt, or tool can consume it.
    raw_user_query: str
    # Sole semantic query authority for composition and artifact decisions.
    rewritten_query: str
    sub_branch: GeneralSubBranch
    persistence_mode: PersistenceMode
    approved_conversation_history: list[dict[str, Any]]
    human_supporting_questions: list[GeneratedQuestion]
    reminder_supporting_questions: list[GeneratedQuestion]
    extracted_expected_response_types: list[ExpectedResponseType]
    approved_knowledge_evidence: list[str]
    approved_reminder_context: list[dict[str, Any]]
    metadata: dict[str, Any]
    platform_context: dict[str, Any]
    sub_branch_prompt_context: SubBranchPromptContext
    sub_branch_supporting_prompt: str
    repository: Any | None = None
    merged_supporting_detail: str = ""
    # SQL-hydrated, ownership-checked evidence with stable candidate identity.
    # Kept alongside the legacy text list for injected-tool compatibility.
    approved_knowledge_records: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class ContentToolResult:
    tool_name: str
    output_text: str
    confidence: float
    fallback_used: bool
    reason_summary: str
    artifact: dict[str, Any] | None = None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ContentComposerResult:
    final_response_text: str
    tool_trace_summary: str
    used_tool_names: tuple[str, ...]
    confidence: float
    fallback_used: bool
    reason_summary: str
    content_warnings: tuple[str, ...]
    artifacts: tuple[dict[str, Any], ...] = ()
    # Exact output of the mandatory answer model, kept separate from file-stage
    # status text so outbound delivery never has to infer their boundary.
    answer_response_text: str = ""
    # Serialized, query-grounded semantic authorization shared with the
    # post-bundling delivery enforcement boundary.
    semantic_action_decision: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IngestionResult:
    source_id: str
    status: KnowledgeSourceStatus
    chunk_ids: tuple[str, ...] = ()
    outbox_job_ids: tuple[str, ...] = ()
    error_message: str | None = None


@dataclass(frozen=True)
class HumanSupportingDecision:
    should_ask: bool
    question: str
    confidence: float
    question_source: QuestionSource
    expected_response_type: ExpectedResponseType
    reason_summary: str
    risk_flags: tuple[str, ...]
