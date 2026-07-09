"""Configuration contracts for the SQL-first assistant.

Runtime values are injected through this module instead of being scattered
through implementation code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .contracts import Intent, ResponseType
from .settings import MutationPartialExecutionPolicy, TargetNotFoundPolicy, UnsupportedActionPolicy


RESPONSE_TYPE_INTENT_MAPPING: dict[ResponseType, Intent] = {
    ResponseType.NORMAL: Intent.GENERAL_RESPONSE,
    ResponseType.KNOWLEDGE_ACTION: Intent.KNOWLEDGE_FACTS,
    ResponseType.REMINDER_ACTION: Intent.REMINDER,
    ResponseType.REMINDER_REPLY: Intent.REMINDER,
}


@dataclass(frozen=True)
class RetrievalConfig:
    conversation_min_confidence: float
    knowledge_min_confidence: float
    max_results: int
    rrf_k: int = 60
    lexical_weight: float = 1.0
    semantic_weight: float = 1.0
    rerank_candidate_limit: int = 32
    general_response_reminder_limit: int = 5
    general_response_reminder_statuses: tuple[str, ...] = ("scheduled", "notified")


@dataclass(frozen=True)
class OutboxConfig:
    max_attempts: int
    batch_size: int
    retry_backoff_seconds: int = 30
    processing_timeout_seconds: int = 300


@dataclass(frozen=True)
class AutoscanConfig:
    interval_seconds: int


@dataclass(frozen=True)
class ClassificationConfig:
    intent_keywords: Mapping[str, tuple[str, ...]]


@dataclass(frozen=True)
class QuestionGenerationConfig:
    enabled: bool = True
    clarification_model: str | None = None
    human_supporting_model: str | None = None
    reminder_supporting_model: str | None = None
    clarification_temperature: float = 0.2
    human_supporting_temperature: float = 0.4
    reminder_supporting_temperature: float = 0.3
    timeout_seconds: float = 30.0
    clarification_max_tokens: int = 150
    human_supporting_max_tokens: int = 150
    reminder_supporting_max_tokens: int = 150
    clarification_retry_count: int = 2
    human_supporting_retry_count: int = 2
    reminder_supporting_retry_count: int = 2
    question_generation_confidence_threshold: float = 0.6
    reminder_supporting_enabled: bool = True
    reminder_supporting_min_confidence: float = 0.6
    human_supporting_max_count: int = 2
    fallback_policy: str = "fallback_message"


@dataclass(frozen=True)
class MutationPolicyConfig:
    partial_execution_policy: MutationPartialExecutionPolicy = MutationPartialExecutionPolicy.ALL_OR_NOTHING
    knowledge_relevance_threshold: float = 0.5
    knowledge_ambiguity_margin: float = 0.1
    knowledge_not_found_policy: TargetNotFoundPolicy = TargetNotFoundPolicy.SKIP_NOT_FOUND
    unsupported_action_policy: UnsupportedActionPolicy = UnsupportedActionPolicy.REJECT_AND_SKIP


@dataclass(frozen=True)
class ReminderTargetResolverConfig:
    reminder_subject_weight: float
    reminder_summary_weight: float
    reminder_raw_text_weight: float
    reminder_time_weight: float
    reminder_entity_weight: float
    reminder_status_weight: float
    reminder_recency_weight: float
    
    reminder_target_relevance_threshold: float
    reminder_target_ambiguity_margin: float
    
    reminder_llm_rerank_enabled: bool
    reminder_llm_rerank_threshold: float
    reminder_llm_rerank_max_candidates: int
    reminder_llm_rerank_model: str | None
    reminder_llm_rerank_json_retry_count: int
    
    reminder_fuzzy_matcher: str
    reminder_target_not_found_policy: TargetNotFoundPolicy
    reminder_target_candidate_limit: int
    
    allowed_reminder_modify_statuses: tuple[str, ...]
    allowed_reminder_turn_on_statuses: tuple[str, ...]
    allowed_reminder_turn_off_statuses: tuple[str, ...]
    allowed_reminder_delete_statuses: tuple[str, ...]
    reminder_delete_status_policy: str


@dataclass(frozen=True)
class RetrievalValidationConfig:
    knowledge_llm_validation_enabled: bool
    knowledge_llm_validation_model: str | None
    knowledge_llm_validation_min_confidence: float
    knowledge_llm_validation_json_retry_count: int
    knowledge_llm_validation_max_candidates: int
    knowledge_llm_validation_failure_policy: str

    reminder_llm_validation_enabled: bool
    reminder_llm_validation_model: str | None
    reminder_llm_validation_min_confidence: float
    reminder_llm_validation_json_retry_count: int
    reminder_llm_validation_max_candidates: int

    destructive_action_requires_unambiguous_target: bool


@dataclass(frozen=True)
class LastQAConfig:
    min_confidence: float = 0.4
    clarification_merge_min_confidence: float = 0.7
    skip_allowed_interaction_types: tuple[str, ...] = (
        "normal_follow_up",
        "human_supporting_question_answer",
        "reminder_notification_reply",
    )
    semantic_match_model: str | None = None
    clarification_merge_model: str | None = None
    json_retry_count: int = 2
    clarification_merge_json_retry_count: int = 2
    enable_reminder_metadata_reply: bool = True
    clarification_merge_enabled: bool = True


@dataclass(frozen=True)
class AssistantConfig:
    retrieval: RetrievalConfig
    outbox: OutboxConfig
    autoscan: AutoscanConfig
    classification: ClassificationConfig
    reminder_resolver: ReminderTargetResolverConfig
    retrieval_validation: RetrievalValidationConfig
    mutation_policy: MutationPolicyConfig = field(default_factory=MutationPolicyConfig)
    question_generation: QuestionGenerationConfig = field(default_factory=QuestionGenerationConfig)
    last_qa: LastQAConfig = field(default_factory=LastQAConfig)
    platform_channels: tuple[str, ...] = field(default_factory=tuple)
    response_type_intent_mapping: dict[ResponseType, Intent] = field(
        default_factory=lambda: dict(RESPONSE_TYPE_INTENT_MAPPING)
    )
    human_in_the_loop_min_confidence: float = 0.6
