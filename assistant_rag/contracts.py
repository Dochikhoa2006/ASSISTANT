"""Shared contracts used by every pipeline stage and branch."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Literal


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


class ActionValidationResult(str, Enum):
    EXECUTE = "execute"
    SKIP_NOT_FOUND = "skip_not_found"
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


@dataclass(frozen=True)
class ChatRequest:
    user_id: str
    raw_query: str
    platform_context: dict[str, Any] = field(default_factory=dict)
    reminder_id: str | None = None
    notification_id: str | None = None
    reply_text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LastQAState:
    last_user_query: str
    last_response: str
    response_type: ResponseType
    supporting_questions: list[GeneratedQuestion] = field(default_factory=list)
    clarification_question: GeneratedQuestion | None = None
    reminder_supporting_question: GeneratedQuestion | None = None
    linked_topic_id: str | None = None
    linked_hop_id: str | None = None


@dataclass(frozen=True)
class RetrievalResult:
    entity_type: str
    entity_id: str
    source_store_evidence: dict[str, Any]
    rerank_score: float
    confidence: float
    validation_status: str
    payload: dict[str, Any] = field(default_factory=dict)


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
    ]
    domain_entity_type: Literal[
        "knowledge_chunk",
        "knowledge_topic",
        "reminder",
        "conversation_hop",
        "none",
    ] | None = None
    domain_entity_id: str | None = None
    audit_hop_id: str | None = None
    indexing_outbox_ids: tuple[str, ...] = ()
    user_safe_summary: str | None = None
    reason_summary: str | None = None


@dataclass(frozen=True)
class RepositoryTransactionResult:
    committed: bool
    results: tuple[RepositoryActionResult, ...]
    audit_hop_id: str | None = None
    indexing_outbox_ids: tuple[str, ...] = ()
    error_type: str | None = None
    reason_summary: str | None = None


@dataclass(frozen=True)
class ValidatedKnowledgeAction:
    action: KnowledgeAction
    validation_result: ActionValidationResult

    target_chunk_ids: tuple[str, ...] = ()
    target_topic_ids: tuple[str, ...] = ()

    observed_versions: dict[str, int] = field(default_factory=dict)
    observed_is_deleted: dict[str, bool] = field(default_factory=dict)

    knowledge_text: str | None = None
    replacement_text: str | None = None
    
    # Legacy fields (kept for backward compatibility with ValidatedActionBuilder if it hasn't been updated yet)
    new_text: str | None = None
    target_status: str | None = None
    topic_title: str | None = None
    target_description: str | None = None

    confidence: float = 0.0
    matched_fields: tuple[str, ...] = ()
    reason_summary: str | None = None


@dataclass(frozen=True)
class RetrievalCandidateAssessment:
    candidate_key: str
    matches_target: bool
    action_compatible: bool
    confidence: float
    matched_fields: tuple[str, ...]
    reason_summary: str


@dataclass(frozen=True)
class LLMRetrievalValidationResult:
    operation: str
    validation_result: ActionValidationResult
    selected_candidate_keys: tuple[str, ...]
    confidence: float
    ambiguous: bool
    reason_summary: str
    candidate_assessments: tuple[RetrievalCandidateAssessment, ...]


@dataclass(frozen=True)
class KnowledgeValidationCandidate:
    candidate_key: str
    knowledge_chunk_id: str
    knowledge_topic_id: str
    text: str
    source_title: str | None
    retrieval_score: float
    rerank_score: float | None
    is_deleted: bool
    user_id: str


@dataclass(frozen=True)
class ReminderValidationCandidate:
    candidate_key: str
    reminder_id: str
    subject: str
    reminder_summary: str | None
    raw_reminder: str | None
    reminder_time: datetime | None
    status: str
    deterministic_score: float
    observed_version: int | None
    observed_status: str


@dataclass(frozen=True)
class LLMKnowledgeCandidatePayload:
    candidate_key: str
    text_excerpt: str
    source_title: str | None
    retrieval_score: float
    rerank_score: float | None
    matched_fields: tuple[str, ...]


@dataclass(frozen=True)
class LLMReminderCandidatePayload:
    candidate_key: str
    subject: str
    reminder_summary: str | None
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
    target_daypart: str | None = None


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
    reason_summary: str | None = None


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
    observed_status: str | None = None
    observed_version: int | None = None
    observed_reminder_time: datetime | None = None

    subject: str | None = None
    reminder_time: datetime | None = None
    reminder_summary: str | None = None
    raw_reminder: str | None = None

    replacement_subject: str | None = None
    replacement_time: datetime | None = None
    replacement_summary: str | None = None

    confidence: float = 0.0
    matched_fields: tuple[str, ...] = ()
    reason_summary: str | None = None


@dataclass
class BranchResult:
    response_type: ResponseType
    normal_response_text: str | None = None
    clarification_question: GeneratedQuestion | None = None
    human_supporting_questions: list[GeneratedQuestion] = field(default_factory=list)
    reminder_supporting_question: GeneratedQuestion | None = None
    question_source: QuestionSource | None = None
    knowledge_operation_results: list[RepositoryActionResult] | list[OperationResult] = field(default_factory=list)
    reminder_operation_results: list[RepositoryActionResult] | list[OperationResult] = field(default_factory=list)
    human_in_the_loop_result: dict[str, Any] | None = None
    platform_payload: dict[str, Any] = field(default_factory=dict)
    fallback_or_error_message: str | None = None
    database_write_result: dict[str, Any] = field(default_factory=dict)
    indexing_job_result: dict[str, Any] = field(default_factory=dict)
    linked_topic_id: str | None = None
    linked_hop_id: str | None = None


@dataclass(frozen=True)
class BundledResponse:
    final_chat_text: str
    response_type: ResponseType
    last_qa_state: LastQAState
    persistence_instructions: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LastQAResolution:
    path: LastQAPath
    rewritten_query: str
    state: LastQAState | None
    did_merge_query: bool
    skip_broad_retrieval: bool

    interaction_type: LastQAInteractionType | None = None
    question_source: QuestionSource = QuestionSource.NONE

    linked_topic_id: str | None = None
    linked_hop_id: str | None = None
    matched_question: str | None = None

    reminder_id: str | None = None
    notification_id: str | None = None
    source_topic_id: str | None = None
    source_hop_id: str | None = None

    missing_context: list[str] = field(default_factory=list)
    diagnostic_context: dict[str, Any] = field(default_factory=dict)

    merge_reason: str | None = None
    skip_reason: str | None = None

    is_authoritative_state: bool = False


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
    last_qa_trace: dict[str, Any] = field(default_factory=dict)
