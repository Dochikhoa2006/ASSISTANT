"""Configuration contracts for the SQL-first assistant.

Runtime values are injected through this module instead of being scattered
through implementation code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .content_keywords import (
    DOCUMENT_FILE_KEYWORDS,
    EXCEL_FILE_KEYWORDS,
    FILE_CREATION_VERB_KEYWORDS,
    POWERPOINT_FILE_KEYWORDS,
)
from .contracts import Intent, ResponseType
from .settings import KnowledgeChunkSettings, MutationPartialExecutionPolicy, TargetNotFoundPolicy, UnsupportedActionPolicy
from .retrieval_policy import RETRIEVAL_PIPELINE_POLICY, RetrievalPipelinePolicy


RESPONSE_TYPE_INTENT_MAPPING: dict[ResponseType, Intent] = {
    ResponseType.NORMAL: Intent.GENERAL_RESPONSE,
    ResponseType.KNOWLEDGE_ACTION: Intent.KNOWLEDGE_FACTS,
    ResponseType.REMINDER_ACTION: Intent.REMINDER,
    ResponseType.REMINDER_REPLY: Intent.REMINDER,
}


@dataclass(frozen=True)
class RetrievalConfig:
    max_results: int = RETRIEVAL_PIPELINE_POLICY.final_top_k
    bm25_top_k: int = RETRIEVAL_PIPELINE_POLICY.source_top_k
    chroma_top_k: int = RETRIEVAL_PIPELINE_POLICY.source_top_k
    rrf_k: int = 40
    lexical_weight: float = 1.10
    semantic_weight: float = 1.0
    rerank_candidate_limit: int = RETRIEVAL_PIPELINE_POLICY.rrf_top_k
    conversation_min_confidence_score: float = 0.50
    general_response_reminder_limit: int = 4
    general_response_reminder_statuses: tuple[str, ...] = ("scheduled", "notified")

    def __post_init__(self) -> None:
        RetrievalPipelinePolicy(
            source_top_k=self.bm25_top_k,
            rrf_top_k=self.rerank_candidate_limit,
            final_top_k=self.max_results,
        ).validate()
        if self.chroma_top_k != self.bm25_top_k:
            raise ValueError("OpenSearch and ChromaDB candidate limits must be identical")
        if not 0.0 <= self.conversation_min_confidence_score <= 1.0:
            raise ValueError(
                "conversation_min_confidence_score must be in [0.0, 1.0]"
            )


@dataclass(frozen=True)
class OutboxConfig:
    max_attempts: int
    batch_size: int
    retry_backoff_seconds: int = 15
    processing_timeout_seconds: int = 180


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
    clarification_temperature: float = 0.15
    human_supporting_temperature: float = 0.25
    reminder_supporting_temperature: float = 0.2
    timeout_seconds: float = 15.0
    clarification_max_tokens: int = 96
    human_supporting_max_tokens: int = 128
    reminder_supporting_max_tokens: int = 128
    clarification_retry_count: int = 1
    human_supporting_retry_count: int = 1
    reminder_supporting_retry_count: int = 1
    question_generation_confidence_threshold: float = 0.68
    reminder_supporting_enabled: bool = True
    reminder_supporting_min_confidence: float = 0.68
    human_supporting_max_count: int = 1
    fallback_policy: str = "fallback_message"


@dataclass(frozen=True)
class MutationPolicyConfig:
    partial_execution_policy: MutationPartialExecutionPolicy = MutationPartialExecutionPolicy.ALL_OR_NOTHING
    knowledge_relevance_threshold: float = 0.58
    knowledge_ambiguity_margin: float = 0.08
    knowledge_not_found_policy: TargetNotFoundPolicy = TargetNotFoundPolicy.SKIP_NOT_FOUND
    unsupported_action_policy: UnsupportedActionPolicy = UnsupportedActionPolicy.REJECT_AND_SKIP

    def __post_init__(self) -> None:
        if not (0 <= self.knowledge_relevance_threshold <= 1):
            raise ValueError("knowledge_relevance_threshold must be between 0 and 1")
        if not (0 <= self.knowledge_ambiguity_margin <= 1):
            raise ValueError("knowledge_ambiguity_margin must be between 0 and 1")
        if not isinstance(self.partial_execution_policy, MutationPartialExecutionPolicy):
            raise ValueError("partial_execution_policy must be a valid MutationPartialExecutionPolicy")


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

    def __post_init__(self) -> None:
        if not (0 <= self.reminder_target_relevance_threshold <= 1):
            raise ValueError("reminder_target_relevance_threshold must be between 0 and 1")
        if not (0 <= self.reminder_target_ambiguity_margin <= 1):
            raise ValueError("reminder_target_ambiguity_margin must be between 0 and 1")
        valid_statuses = {
            "scheduled",
            "notified",
            "cancelled",
            "dismissed",
            "completed",
        }
        for status_list in [self.allowed_reminder_modify_statuses, self.allowed_reminder_turn_on_statuses, self.allowed_reminder_turn_off_statuses, self.allowed_reminder_delete_statuses]:
            for status in status_list:
                if status not in valid_statuses:
                    raise ValueError(f"Invalid reminder status: {status}")


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

    def __post_init__(self) -> None:
        if not (0 <= self.knowledge_llm_validation_min_confidence <= 1):
            raise ValueError("knowledge_llm_validation_min_confidence must be between 0 and 1")
        if not (0 <= self.reminder_llm_validation_min_confidence <= 1):
            raise ValueError("reminder_llm_validation_min_confidence must be between 0 and 1")


@dataclass(frozen=True)
class LastQAConfig:
    min_confidence: float = 0.80
    skip_broad_retrieval_min_confidence: float = 0.90
    clarification_merge_min_confidence: float = 0.84
    skip_allowed_interaction_types: tuple[str, ...] = (
        "normal_follow_up",
        "supporting_question_answer",
        "reminder_notification_reply",
    )
    semantic_match_model: str | None = None
    clarification_merge_model: str | None = None
    json_retry_count: int = 1
    clarification_merge_json_retry_count: int = 1
    enable_reminder_metadata_reply: bool = True
    clarification_merge_enabled: bool = True


@dataclass(frozen=True)
class ContextFilterConfig:
    reminder_approved_max_items: int = 5
    semantic_context_judge_enabled: bool = False
    semantic_context_judge_failure_policy: str = "use_hard_rule_approved"
    context_filter_debug_diagnostics_enabled: bool = False
    conversation_retrieval_after_last_qa_enabled: bool = True
    conversation_retrieval_before_intent_enabled: bool = True
    expected_response_type_required: bool = True
    expected_response_type_fallback_policy: str = "unknown"
    clarification_expected_response_type_required: bool = True

    def __post_init__(self) -> None:
        if self.expected_response_type_fallback_policy not in {"unknown", "reject"}:
            raise ValueError("expected_response_type_fallback_policy must be 'unknown' or 'reject'")


_ALLOWED_PERSISTENCE_POLICIES = frozenset({"sub_branch_driven"})

@dataclass(frozen=True)
class GeneralPurposeConfig:
    # Sub-branch detector
    general_sub_branch_detector_enabled: bool = True
    general_sub_branch_confidence_threshold: float = 0.65
    support_question_resolution_min_confidence: float = 0.90
    conversation_followup_min_score: float = 0.65
    general_sub_branch_fallback_mode: str = "new_conversation_topic"
    general_sub_branch_detector_json_retry_count: int = 1

    # Content composer
    # Controls the optional Microsoft file stage. The answer_generation stage is
    # unconditional for every general-purpose response.
    content_composer_enabled: bool = True
    content_composer_tool_timeout_seconds: float = 35.0
    content_composer_allowed_tools: tuple[str, ...] = (
        "answer_generation",
        "generate_excel",
        "generate_pdf",
        "generate_pptx",
    )
    content_composer_default_tool: str = "answer_generation"
    content_composer_fallback_tool: str = "answer_generation"

    # Deterministic file-intent signals. A file tool requires at least one
    # rewritten-query match from the verb list and exactly one file-type group.
    file_creation_verb_keywords: tuple[str, ...] = FILE_CREATION_VERB_KEYWORDS
    document_tool_signal_keywords: tuple[str, ...] = DOCUMENT_FILE_KEYWORDS
    excel_tool_signal_keywords: tuple[str, ...] = EXCEL_FILE_KEYWORDS
    pptx_tool_signal_keywords: tuple[str, ...] = POWERPOINT_FILE_KEYWORDS

    # HITL
    hitl_supporting_question_enabled: bool = True
    hitl_supporting_question_confidence_threshold: float = 0.72
    hitl_supporting_question_recent_question_window: int = 2
    hitl_supporting_question_max_length: int = 160
    hitl_supporting_question_safety_mode: str = "standard"

    # Persistence
    general_response_persistence_policy: str = "sub_branch_driven"
    sub_branch_prompt_mode: str = "sub_branch_driven"
    general_response_default_topic_title: str = "General Conversation"

    documents_dir: str | None = None
    artifact_storage_dir: str = "assistant_data/artifacts"
    artifact_download_base_url: str = "/artifacts"

    def __post_init__(self) -> None:
        if not (0.0 <= self.general_sub_branch_confidence_threshold <= 1.0):
            raise ValueError("general_sub_branch_confidence_threshold must be in [0.0, 1.0]")
        if not (0.0 <= self.support_question_resolution_min_confidence <= 1.0):
            raise ValueError(
                "support_question_resolution_min_confidence must be in [0.0, 1.0]"
            )
        if not (0.0 <= self.conversation_followup_min_score <= 1.0):
            raise ValueError("conversation_followup_min_score must be in [0.0, 1.0]")
        if self.content_composer_tool_timeout_seconds <= 0:
            raise ValueError("content_composer_tool_timeout_seconds must be > 0")
        if self.content_composer_default_tool not in self.content_composer_allowed_tools:
            raise ValueError(
                f"content_composer_default_tool '{self.content_composer_default_tool}' "
                f"must be in content_composer_allowed_tools"
            )
        if self.content_composer_fallback_tool not in self.content_composer_allowed_tools:
            raise ValueError(
                f"content_composer_fallback_tool '{self.content_composer_fallback_tool}' "
                f"must be in content_composer_allowed_tools"
            )
        if not (0.0 <= self.hitl_supporting_question_confidence_threshold <= 1.0):
            raise ValueError("hitl_supporting_question_confidence_threshold must be in [0.0, 1.0]")
        if self.hitl_supporting_question_recent_question_window < 0:
            raise ValueError("hitl_supporting_question_recent_question_window must be >= 0")
        if self.hitl_supporting_question_max_length <= 0:
            raise ValueError("hitl_supporting_question_max_length must be > 0")
        if self.general_response_persistence_policy not in _ALLOWED_PERSISTENCE_POLICIES:
            raise ValueError(
                f"general_response_persistence_policy must be one of "
                f"{sorted(_ALLOWED_PERSISTENCE_POLICIES)}"
            )


@dataclass(frozen=True)
class AssistantConfig:
    retrieval: RetrievalConfig
    outbox: OutboxConfig
    autoscan: AutoscanConfig
    classification: ClassificationConfig
    reminder_resolver: ReminderTargetResolverConfig
    retrieval_validation: RetrievalValidationConfig
    knowledge_chunk_settings: KnowledgeChunkSettings = field(default_factory=KnowledgeChunkSettings)
    mutation_policy: MutationPolicyConfig = field(default_factory=MutationPolicyConfig)
    question_generation: QuestionGenerationConfig = field(default_factory=QuestionGenerationConfig)
    last_qa: LastQAConfig = field(default_factory=LastQAConfig)
    context_filter: ContextFilterConfig = field(default_factory=ContextFilterConfig)
    general_purpose: GeneralPurposeConfig = field(default_factory=GeneralPurposeConfig)
    platform_channels: tuple[str, ...] = field(default_factory=tuple)
    default_timezone: str = "UTC"
    reminder_duplicate_similarity_threshold: float = 0.72
    reminder_duplicate_time_window_minutes: int = 45
    confirmation_expiry_minutes: int = 10
    confirmation_high_confidence_threshold: float = 0.92
    response_type_intent_mapping: dict[ResponseType, Intent] = field(
        default_factory=lambda: dict(RESPONSE_TYPE_INTENT_MAPPING)
    )
    human_in_the_loop_min_confidence: float = 0.66
