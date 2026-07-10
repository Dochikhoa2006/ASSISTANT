"""Production settings loaded from environment.

All operational values live here instead of being embedded in business logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import os


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.casefold() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default


def _get_tuple(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = os.getenv(name)
    if not value:
        return default
    return tuple(item.strip() for item in value.split(",") if item.strip())


@dataclass(frozen=True)
class DatabaseSettings:
    path: str = "assistant_data/assistant.sqlite3"
    url: str | None = None
    pool_size: int = 8
    max_overflow: int = 16
    enable_wal: bool = True
    busy_timeout_ms: int = 2000


@dataclass(frozen=True)
class OllamaSettings:
    base_url: str = "http://localhost:11434"
    structured_retry_count: int = 1
    keep_alive: int | str = "30m"

    # Task: QUERY_REWRITE
    model_query_rewrite: str = "qwen3.5:0.8b"
    timeout_query_rewrite: float = 12.0
    num_ctx_query_rewrite: int = 1024
    num_predict_query_rewrite: int | None = 96
    temperature_query_rewrite: float = 0.0

    # Task: LAST_QA
    model_last_qa: str = "qwen3.5:2b"
    timeout_last_qa: float = 18.0
    num_ctx_last_qa: int = 1536
    num_predict_last_qa: int | None = 160
    temperature_last_qa: float = 0.0
    json_retry_count_last_qa: int = 1

    # Task: INTENT
    model_intent: str = "qwen3.5:2b"
    timeout_intent: float = 18.0
    num_ctx_intent: int = 2048
    num_predict_intent: int | None = 96
    temperature_intent: float = 0.0

    # Task: ACTION_EXTRACTION
    model_action_extraction: str = "qwen3.5:4b"
    timeout_action_extraction: float = 30.0
    num_ctx_action_extraction: int = 3072
    num_predict_action_extraction: int | None = 512
    temperature_action_extraction: float = 0.0

    # Task: GENERATE_CLARIFICATION
    model_generate_clarification: str = "qwen3.5:2b"
    timeout_generate_clarification: float = 15.0
    num_ctx_generate_clarification: int = 2048
    num_predict_generate_clarification: int | None = 96
    temperature_generate_clarification: float = 0.15
    json_retry_count_generate_clarification: int = 1

    # Task: GENERATE_HUMAN_SUPPORTING
    model_generate_human_supporting: str = "qwen3.5:2b"
    timeout_generate_human_supporting: float = 15.0
    num_ctx_generate_human_supporting: int = 2048
    num_predict_generate_human_supporting: int | None = 128
    temperature_generate_human_supporting: float = 0.25
    json_retry_count_generate_human_supporting: int = 1

    # Task: GENERATE_REMINDER_SUPPORTING
    model_generate_reminder_supporting: str = "qwen3.5:2b"
    timeout_generate_reminder_supporting: float = 15.0
    num_ctx_generate_reminder_supporting: int = 2048
    num_predict_generate_reminder_supporting: int | None = 128
    temperature_generate_reminder_supporting: float = 0.2
    json_retry_count_generate_reminder_supporting: int = 1

    # Task: CLARIFICATION_MERGE
    model_clarification_merge: str = "qwen3.5:4b"
    timeout_clarification_merge: float = 24.0
    num_ctx_clarification_merge: int = 3072
    num_predict_clarification_merge: int | None = 256
    temperature_clarification_merge: float = 0.0
    json_retry_count_clarification_merge: int = 1

    # Task: ANSWER
    model_answer: str = "qwen3.5:9b"
    timeout_answer: float = 75.0
    num_ctx_answer: int = 8192
    num_predict_answer: int | None = 1024
    temperature_answer: float = 0.22

    # Task: WRITING
    model_writing: str = "qwen3.5:9b"
    timeout_writing: float = 90.0
    num_ctx_writing: int = 8192
    num_predict_writing: int | None = 1536
    temperature_writing: float = 0.38

    # Task: RISKY_ACTION
    model_risky_action: str = "qwen3.5:9b"
    timeout_risky_action: float = 35.0
    num_ctx_risky_action: int = 4096
    num_predict_risky_action: int | None = 384
    temperature_risky_action: float = 0.0
    json_retry_count_risky_action: int = 1

    # Task: RETRIEVAL_VALIDATION
    model_retrieval_validation: str = "qwen3.5:9b"
    timeout_retrieval_validation: float = 35.0
    num_ctx_retrieval_validation: int = 4096
    num_predict_retrieval_validation: int | None = 384
    temperature_retrieval_validation: float = 0.0

    # Task: GENERAL_SUB_BRANCH_DETECTION
    model_general_sub_branch_detection: str = "qwen3.5:2b"
    timeout_general_sub_branch_detection: float = 12.0
    num_ctx_general_sub_branch_detection: int = 1536
    num_predict_general_sub_branch_detection: int | None = 96
    temperature_general_sub_branch_detection: float = 0.0

    # Task: CONTENT_COMPOSER_REACT
    model_content_composer_react: str = "qwen3.5:2b"
    timeout_content_composer_react: float = 15.0
    num_ctx_content_composer_react: int = 2048
    num_predict_content_composer_react: int | None = 128
    temperature_content_composer_react: float = 0.0

    # Task: ACTION_PLANNING
    model_action_planning: str = "qwen3.5:4b"
    timeout_action_planning: float = 35.0
    num_ctx_action_planning: int = 4096
    num_predict_action_planning: int | None = 512
    temperature_action_planning: float = 0.0


@dataclass(frozen=True)
class RetrievalSettings:
    max_results: int = 6
    bm25_top_k: int = 24
    chroma_top_k: int = 24
    min_confidence: float = 0.30
    conversation_min_confidence: float = 0.42
    knowledge_min_confidence: float = 0.38
    rrf_k: int = 40
    lexical_weight: float = 1.10
    semantic_weight: float = 1.0
    rerank_candidate_limit: int = 16


@dataclass(frozen=True)
class OpenSearchSettings:
    url: str = "http://localhost:9200"
    username: str | None = None
    password: str | None = None
    verify_certs: bool = False
    conversation_index: str = "assistant_conversation_hops"
    knowledge_index: str = "assistant_knowledge_chunks"
    conversation_write_alias: str = "assistant_conversation_hops_write"
    knowledge_write_alias: str = "assistant_knowledge_chunks_write"
    reminder_context_alias: str = "assistant_reminder_contexts"
    analyzer_name: str = "assistant_text"
    timeout_seconds: int = 10
    max_retries: int = 2


@dataclass(frozen=True)
class ChromaSettings:
    path: str = "assistant_data/chroma"
    host: str | None = "localhost"
    port: int | None = 8000
    conversation_collection: str = "conversation_hops"
    knowledge_collection: str = "knowledge_chunks"


@dataclass(frozen=True)
class EmbeddingSettings:
    model_name: str = "BAAI/bge-m3"
    device: str | None = None
    batch_size: int = 48
    normalize_embeddings: bool = True
    max_length: int = 4096


@dataclass(frozen=True)
class RerankerSettings:
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    endpoint_url: str | None = None
    timeout_seconds: float = 6.0
    min_score: float = 0.30
    device: str | None = None
    batch_size: int = 24
    max_candidates: int = 20


@dataclass(frozen=True)
class LastQASettings:
    path: str = "assistant_data/last_qa.sqlite3"
    ttl_seconds: int = 1800


@dataclass(frozen=True)
class WorkerSettings:
    outbox_batch_size: int = 64
    outbox_max_attempts: int = 4
    outbox_worker_interval_seconds: int = 3
    outbox_retry_backoff_seconds: int = 15
    outbox_processing_timeout_seconds: int = 180
    autoscan_interval_seconds: int = 30


@dataclass(frozen=True)
class UISettings:
    notification_transport: str = "websocket"
    redis_url: str | None = None


@dataclass(frozen=True)
class APISettings:
    host: str = "127.0.0.1"
    port: int = 8000


@dataclass(frozen=True)
class SafetySettings:
    default_timezone: str = "UTC"
    reminder_duplicate_similarity_threshold: float = 0.72
    reminder_duplicate_time_window_minutes: int = 45
    confirmation_expiry_minutes: int = 10
    confirmation_high_confidence_threshold: float = 0.92
    allow_missing_idempotency_key: bool = True


@dataclass(frozen=True)
class AuthSettings:
    issuer: str | None = None
    audience: str | None = None
    jwks_url: str | None = None
    public_key: str | None = None
    required_scopes: tuple[str, ...] = ()


class MutationPartialExecutionPolicy(str, Enum):
    ALL_OR_NOTHING = "all_or_nothing"
    PARTIAL_ALLOWED = "partial_allowed"


class TargetNotFoundPolicy(str, Enum):
    CLARIFY_ON_NOT_FOUND = "clarify_on_not_found"
    SKIP_NOT_FOUND = "skip_not_found"


class UnsupportedActionPolicy(str, Enum):
    CLARIFY = "clarify"
    REJECT_AND_SKIP = "reject_and_skip"


@dataclass(frozen=True)
class PromptPolicySettings:
    intent_min_confidence: float = 0.55
    action_min_confidence: float = 0.76
    last_qa_min_confidence: float = 0.80
    last_qa_skip_broad_retrieval_min_confidence: float = 0.90
    human_in_the_loop_min_confidence: float = 0.66
    clarification_merge_min_confidence: float = 0.84
    supporting_question_match_threshold: float = 0.74
    skip_broad_retrieval_allowed_relationships: tuple[str, ...] = (
        "supporting_question_answer",
        "normal_follow_up",
        "reminder_reply",
    )
    clarification_merge_enabled: bool = True
    last_qa_enable_reminder_metadata_reply: bool = True
    risky_action_validation_enabled: bool = True
    risky_action_operations: tuple[str, ...] = ("delete", "modify", "turn_off")
    risky_action_confidence_threshold: float = 0.90
    knowledge_modify_requires_replacement_text: bool = True
    context_filter_knowledge_min_confidence: float = 0.38
    context_filter_allowed_reminder_statuses: tuple[str, ...] = ("scheduled", "notified")
    question_generation_enabled: bool = True
    reminder_supporting_question_enabled: bool = True
    question_generation_confidence_threshold: float = 0.68
    reminder_supporting_question_min_confidence: float = 0.68
    human_supporting_question_max_count: int = 1
    question_generation_fallback_policy: str = "fallback_message"
    mutation_partial_execution_policy: MutationPartialExecutionPolicy = MutationPartialExecutionPolicy.ALL_OR_NOTHING
    knowledge_target_relevance_threshold: float = 0.58
    knowledge_target_ambiguity_margin: float = 0.08
    knowledge_target_not_found_policy: TargetNotFoundPolicy = TargetNotFoundPolicy.SKIP_NOT_FOUND
    unsupported_action_policy: UnsupportedActionPolicy = UnsupportedActionPolicy.REJECT_AND_SKIP
    conversation_retrieval_after_last_qa_enabled: bool = True
    conversation_retrieval_before_intent_enabled: bool = True
    expected_response_type_required: bool = True
    expected_response_type_fallback_policy: str = "unknown"
    clarification_expected_response_type_required: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.conversation_retrieval_after_last_qa_enabled, bool):
            raise ValueError("conversation_retrieval_after_last_qa_enabled must be a boolean")
        if not isinstance(self.conversation_retrieval_before_intent_enabled, bool):
            raise ValueError("conversation_retrieval_before_intent_enabled must be a boolean")
        if not isinstance(self.expected_response_type_required, bool):
            raise ValueError("expected_response_type_required must be a boolean")
        if not isinstance(self.clarification_expected_response_type_required, bool):
            raise ValueError("clarification_expected_response_type_required must be a boolean")
        if self.expected_response_type_fallback_policy not in {"unknown", "reject"}:
            raise ValueError("expected_response_type_fallback_policy must be 'unknown' or 'reject'")



@dataclass(frozen=True)
class ReminderTargetResolverSettings:
    reminder_subject_weight: float = 1.25
    reminder_summary_weight: float = 0.65
    reminder_raw_text_weight: float = 0.15
    reminder_time_weight: float = 1.35
    reminder_entity_weight: float = 0.65
    reminder_status_weight: float = 0.15
    reminder_recency_weight: float = 0.20
    
    reminder_target_relevance_threshold: float = 0.78
    reminder_target_ambiguity_margin: float = 0.08
    
    reminder_llm_rerank_enabled: bool = False
    reminder_llm_rerank_threshold: float = 0.90
    reminder_llm_rerank_max_candidates: int = 3
    reminder_llm_rerank_model: str | None = None
    reminder_llm_rerank_json_retry_count: int = 1
    
    reminder_fuzzy_matcher: str = "rapidfuzz"
    reminder_fuzzy_match_threshold: float = 0.82
    reminder_target_not_found_policy: TargetNotFoundPolicy = TargetNotFoundPolicy.SKIP_NOT_FOUND
    reminder_target_candidate_limit: int = 12
    
    allowed_reminder_modify_statuses: tuple[str, ...] = ("scheduled", "notified")
    allowed_reminder_turn_on_statuses: tuple[str, ...] = ("dismissed", "cancelled", "completed")
    allowed_reminder_turn_off_statuses: tuple[str, ...] = ("scheduled", "notified")
    allowed_reminder_delete_statuses: tuple[str, ...] = ("scheduled", "notified", "dismissed", "cancelled", "completed")
    reminder_delete_status_policy: str = "dismissed"

    def __post_init__(self) -> None:
        if self.reminder_subject_weight < 0.0: raise ValueError("reminder_subject_weight < 0.0")
        if self.reminder_summary_weight < 0.0: raise ValueError("reminder_summary_weight < 0.0")
        if self.reminder_raw_text_weight < 0.0: raise ValueError("reminder_raw_text_weight < 0.0")
        if self.reminder_time_weight < 0.0: raise ValueError("reminder_time_weight < 0.0")
        if self.reminder_entity_weight < 0.0: raise ValueError("reminder_entity_weight < 0.0")
        if self.reminder_status_weight < 0.0: raise ValueError("reminder_status_weight < 0.0")
        if self.reminder_recency_weight < 0.0: raise ValueError("reminder_recency_weight < 0.0")

        if not (0.0 <= self.reminder_target_relevance_threshold <= 1.0): raise ValueError("reminder_target_relevance_threshold invalid")
        if not (0.0 <= self.reminder_target_ambiguity_margin <= 1.0): raise ValueError("reminder_target_ambiguity_margin invalid")
        if not (0.0 <= self.reminder_fuzzy_match_threshold <= 1.0): raise ValueError("reminder_fuzzy_match_threshold invalid")
        if not (0.0 <= self.reminder_llm_rerank_threshold <= 1.0): raise ValueError("reminder_llm_rerank_threshold invalid")

        if self.reminder_target_candidate_limit <= 0: raise ValueError("reminder_target_candidate_limit <= 0")
        if self.reminder_llm_rerank_max_candidates <= 0: raise ValueError("reminder_llm_rerank_max_candidates <= 0")
        if self.reminder_llm_rerank_max_candidates > self.reminder_target_candidate_limit:
            raise ValueError("reminder_llm_rerank_max_candidates > reminder_target_candidate_limit")

        if self.reminder_fuzzy_matcher not in {"rapidfuzz", "difflib"}:
            raise ValueError("reminder_fuzzy_matcher invalid")
        if self.reminder_target_not_found_policy not in {TargetNotFoundPolicy.CLARIFY_ON_NOT_FOUND, TargetNotFoundPolicy.SKIP_NOT_FOUND}:
            raise ValueError("reminder_target_not_found_policy invalid")


@dataclass(frozen=True)
class RetrievalValidationSettings:
    knowledge_llm_validation_enabled: bool = True
    knowledge_llm_validation_model: str | None = None
    knowledge_llm_validation_min_confidence: float = 0.86
    knowledge_llm_validation_json_retry_count: int = 1
    knowledge_llm_validation_max_candidates: int = 3
    knowledge_llm_validation_failure_policy: str = "fail_closed"

    reminder_llm_validation_enabled: bool = True
    reminder_llm_validation_model: str | None = None
    reminder_llm_validation_min_confidence: float = 0.86
    reminder_llm_validation_json_retry_count: int = 1
    reminder_llm_validation_max_candidates: int = 3

    destructive_action_requires_unambiguous_target: bool = True

    def __post_init__(self) -> None:
        if not (0.0 <= self.knowledge_llm_validation_min_confidence <= 1.0):
            raise ValueError("knowledge_llm_validation_min_confidence must be between 0.0 and 1.0")
        if not (0.0 <= self.reminder_llm_validation_min_confidence <= 1.0):
            raise ValueError("reminder_llm_validation_min_confidence must be between 0.0 and 1.0")
        if self.knowledge_llm_validation_max_candidates <= 0:
            raise ValueError("knowledge_llm_validation_max_candidates must be > 0")
        if self.reminder_llm_validation_max_candidates <= 0:
            raise ValueError("reminder_llm_validation_max_candidates must be > 0")
        if self.knowledge_llm_validation_json_retry_count < 0:
            raise ValueError("knowledge_llm_validation_json_retry_count must be >= 0")
        if self.reminder_llm_validation_json_retry_count < 0:
            raise ValueError("reminder_llm_validation_json_retry_count must be >= 0")
        if self.knowledge_llm_validation_failure_policy not in {"fail_closed", "deterministic_fallback"}:
            raise ValueError("knowledge_llm_validation_failure_policy must be 'fail_closed' or 'deterministic_fallback'")


@dataclass(frozen=True)
class KnowledgeChunkSettings:
    chunk_size_tokens: int = 560
    chunk_overlap_tokens: int = 80
    min_chunk_tokens: int = 80
    max_chunk_tokens: int = 800

    def __post_init__(self) -> None:
        if self.chunk_size_tokens <= 0:
            raise ValueError("chunk_size_tokens must be > 0")
        if self.chunk_overlap_tokens < 0:
            raise ValueError("chunk_overlap_tokens must be >= 0")
        if self.min_chunk_tokens <= 0:
            raise ValueError("min_chunk_tokens must be > 0")
        if self.max_chunk_tokens < self.min_chunk_tokens:
            raise ValueError("max_chunk_tokens must be >= min_chunk_tokens")


@dataclass(frozen=True)
class ServiceWaitSettings:
    timeout_seconds: int = 120
    probe_timeout_seconds: int = 3
    poll_interval_seconds: int = 1
    auto_pull_ollama_models: bool = True


@dataclass(frozen=True)
class OperationsSettings:
    structured_logs_enabled: bool = True
    log_raw_content: bool = False
    debug_trace_responses: bool = False
    metrics_enabled: bool = True
    health_strict_opensearch: bool = False
    health_strict_chroma: bool = False
    health_strict_ollama: bool = False
    health_strict_redis: bool = False
    health_strict_storage: bool = False
    index_rebuild_batch_size: int = 200
    drift_repair_enabled: bool = False
    recurrence_default_timezone: str = "UTC"
    eval_top_1_threshold: float = 0.78
    eval_top_3_threshold: float = 0.90
    eval_wrong_target_rate_max: float = 0.01
    eval_false_mutation_rate_max: float = 0.0


@dataclass(frozen=True)
class DebugSettings:
    db_path: str = "assistant_data/debug_pipeline.sqlite3"
    user_id: str = "debug-user"
    conversation_min_confidence: float = 0.08
    knowledge_min_confidence: float = 0.10
    max_results: int = 4
    rrf_k: int = 40
    lexical_weight: float = 1.10
    semantic_weight: float = 1.0
    rerank_candidate_limit: int = 8
    outbox_max_attempts: int = 2
    outbox_batch_size: int = 32
    outbox_retry_backoff_seconds: int = 0
    outbox_processing_timeout_seconds: int = 45
    autoscan_interval_seconds: int = 10
    opensearch_conversation_index: str = "debug_assistant_conversation_hops"
    opensearch_knowledge_index: str = "debug_assistant_knowledge_chunks"
    opensearch_conversation_write_alias: str = "debug_assistant_conversation_hops_write"
    opensearch_knowledge_write_alias: str = "debug_assistant_knowledge_chunks_write"
    opensearch_reminder_context_alias: str = "debug_assistant_reminder_contexts"
    chroma_path: str = "assistant_data/debug_chroma"
    chroma_conversation_collection: str = "debug_conversation_hops"
    chroma_knowledge_collection: str = "debug_knowledge_chunks"


@dataclass(frozen=True)
class ContextFilterSettings:
    conversation_min_confidence: float = 0.42
    conversation_approved_max_items: int = 6
    conversation_duplicate_threshold: float = 0.90
    knowledge_approved_max_items: int = 6
    knowledge_duplicate_threshold: float = 0.93
    low_information_text_patterns: tuple[str, ...] = (
        "done", "saved", "updated successfully", "ok", "noted", "sure"
    )
    reminder_approved_max_items: int = 5
    reminder_min_confidence: float = 0.58
    semantic_context_judge_enabled: bool = False
    context_filter_debug_diagnostics_enabled: bool = False
    low_information_min_chars: int = 12


@dataclass(frozen=True)
class GeneralPurposeSettings:
    general_sub_branch_detector_enabled: bool = True
    general_sub_branch_confidence_threshold: float = 0.65
    general_sub_branch_fallback_mode: str = "new_conversation_topic"
    general_sub_branch_detector_json_retry_count: int = 1
    content_composer_enabled: bool = True
    content_composer_max_iterations: int = 2
    content_composer_tool_timeout_seconds: float = 35.0
    content_composer_allowed_tools: tuple[str, ...] = (
        "answer_generation", "generate_excel", "generate_pdf", "generate_pptx",
    )
    content_composer_default_tool: str = "answer_generation"
    content_composer_fallback_tool: str = "answer_generation"
    content_composer_react_enabled: bool = True
    content_composer_short_circuit_single_tool: bool = True
    content_composer_debug_trace_enabled: bool = False
    excel_tool_signal_keywords: tuple[str, ...] = (
        "excel", "spreadsheet", "workbook", "xlsx", "data table",
        "tracker", "kpi dashboard", "financial model", "data grid",
        "rows and columns",
    )
    pdf_tool_signal_keywords: tuple[str, ...] = (
        "pdf", "report", "formal document", "business report", "proposal",
        "memo", "white paper", "executive summary", "structured document",
    )
    pptx_tool_signal_keywords: tuple[str, ...] = (
        "powerpoint", "pptx", "presentation", "slide deck", "slides",
        "pitch deck", "slideshow", "deck", "make slides",
    )
    hitl_supporting_question_enabled: bool = True
    hitl_supporting_question_confidence_threshold: float = 0.72
    hitl_supporting_question_recent_question_window: int = 2
    hitl_supporting_question_max_length: int = 160
    hitl_supporting_question_safety_mode: str = "standard"
    general_response_persistence_policy: str = "sub_branch_driven"
    sub_branch_prompt_mode: str = "sub_branch_driven"
    general_response_default_topic_title: str = "General Conversation"
    documents_dir: str | None = None
    artifact_storage_dir: str = "assistant_data/artifacts"
    artifact_download_base_url: str = "/artifacts"


@dataclass(frozen=True)
class ProductionSettings:
    database: DatabaseSettings = field(default_factory=DatabaseSettings)
    ollama: OllamaSettings = field(default_factory=OllamaSettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    opensearch: OpenSearchSettings = field(default_factory=OpenSearchSettings)
    chroma: ChromaSettings = field(default_factory=ChromaSettings)
    embeddings: EmbeddingSettings = field(default_factory=EmbeddingSettings)
    reranker: RerankerSettings = field(default_factory=RerankerSettings)
    last_qa: LastQASettings = field(default_factory=LastQASettings)
    worker: WorkerSettings = field(default_factory=WorkerSettings)
    ui: UISettings = field(default_factory=UISettings)
    api: APISettings = field(default_factory=APISettings)
    safety: SafetySettings = field(default_factory=SafetySettings)
    auth: AuthSettings = field(default_factory=AuthSettings)
    prompt_policy: PromptPolicySettings = field(default_factory=PromptPolicySettings)
    reminder_resolver: ReminderTargetResolverSettings = field(default_factory=ReminderTargetResolverSettings)
    retrieval_validation: RetrievalValidationSettings = field(default_factory=RetrievalValidationSettings)
    knowledge_chunks: KnowledgeChunkSettings = field(default_factory=KnowledgeChunkSettings)
    service_wait: ServiceWaitSettings = field(default_factory=ServiceWaitSettings)
    operations: OperationsSettings = field(default_factory=OperationsSettings)
    debug: DebugSettings = field(default_factory=DebugSettings)
    context_filter: ContextFilterSettings = field(default_factory=ContextFilterSettings)
    general_purpose: GeneralPurposeSettings = field(default_factory=GeneralPurposeSettings)

    @classmethod
    def from_env(cls) -> "ProductionSettings":
        return cls(
            database=DatabaseSettings(
                path=os.getenv("ASSISTANT_DB_PATH", DatabaseSettings.path),
                url=os.getenv("ASSISTANT_DATABASE_URL") or None,
                pool_size=_get_int("ASSISTANT_DB_POOL_SIZE", DatabaseSettings.pool_size),
                max_overflow=_get_int(
                    "ASSISTANT_DB_MAX_OVERFLOW", DatabaseSettings.max_overflow
                ),
                enable_wal=_get_bool("ASSISTANT_DB_WAL", DatabaseSettings.enable_wal),
                busy_timeout_ms=_get_int(
                    "ASSISTANT_DB_BUSY_TIMEOUT_MS", DatabaseSettings.busy_timeout_ms
                ),
            ),
            ollama=OllamaSettings(
                base_url=os.getenv("OLLAMA_BASE_URL", OllamaSettings.base_url),
                structured_retry_count=_get_int("OLLAMA_STRUCTURED_RETRY_COUNT", OllamaSettings.structured_retry_count),
                keep_alive=-1 if os.getenv("OLLAMA_KEEP_ALIVE", str(OllamaSettings.keep_alive)) == "-1" else os.getenv("OLLAMA_KEEP_ALIVE", OllamaSettings.keep_alive),
                model_query_rewrite=os.getenv("OLLAMA_QUERY_REWRITE_MODEL", OllamaSettings.model_query_rewrite),
                timeout_query_rewrite=_get_float("OLLAMA_QUERY_REWRITE_TIMEOUT", OllamaSettings.timeout_query_rewrite),
                num_ctx_query_rewrite=_get_int("OLLAMA_QUERY_REWRITE_NUM_CTX", OllamaSettings.num_ctx_query_rewrite),
                num_predict_query_rewrite=_get_int("OLLAMA_QUERY_REWRITE_NUM_PREDICT", OllamaSettings.num_predict_query_rewrite) if os.getenv("OLLAMA_QUERY_REWRITE_NUM_PREDICT") else OllamaSettings.num_predict_query_rewrite,
                temperature_query_rewrite=_get_float("OLLAMA_QUERY_REWRITE_TEMPERATURE", OllamaSettings.temperature_query_rewrite),
                model_last_qa=os.getenv("OLLAMA_LAST_QA_MODEL", OllamaSettings.model_last_qa),
                timeout_last_qa=_get_float("OLLAMA_LAST_QA_TIMEOUT", OllamaSettings.timeout_last_qa),
                num_ctx_last_qa=_get_int("OLLAMA_LAST_QA_NUM_CTX", OllamaSettings.num_ctx_last_qa),
                num_predict_last_qa=_get_int("OLLAMA_LAST_QA_NUM_PREDICT", OllamaSettings.num_predict_last_qa) if os.getenv("OLLAMA_LAST_QA_NUM_PREDICT") else OllamaSettings.num_predict_last_qa,
                temperature_last_qa=_get_float("OLLAMA_LAST_QA_TEMPERATURE", OllamaSettings.temperature_last_qa),
                json_retry_count_last_qa=_get_int("OLLAMA_LAST_QA_JSON_RETRY_COUNT", OllamaSettings.json_retry_count_last_qa),
                model_intent=os.getenv("OLLAMA_INTENT_MODEL", OllamaSettings.model_intent),
                timeout_intent=_get_float("OLLAMA_INTENT_TIMEOUT", OllamaSettings.timeout_intent),
                num_ctx_intent=_get_int("OLLAMA_INTENT_NUM_CTX", OllamaSettings.num_ctx_intent),
                num_predict_intent=_get_int("OLLAMA_INTENT_NUM_PREDICT", OllamaSettings.num_predict_intent) if os.getenv("OLLAMA_INTENT_NUM_PREDICT") else OllamaSettings.num_predict_intent,
                temperature_intent=_get_float("OLLAMA_INTENT_TEMPERATURE", OllamaSettings.temperature_intent),
                model_action_extraction=os.getenv("OLLAMA_ACTION_EXTRACTION_MODEL", OllamaSettings.model_action_extraction),
                timeout_action_extraction=_get_float("OLLAMA_ACTION_EXTRACTION_TIMEOUT", OllamaSettings.timeout_action_extraction),
                num_ctx_action_extraction=_get_int("OLLAMA_ACTION_EXTRACTION_NUM_CTX", OllamaSettings.num_ctx_action_extraction),
                num_predict_action_extraction=_get_int("OLLAMA_ACTION_EXTRACTION_NUM_PREDICT", OllamaSettings.num_predict_action_extraction) if os.getenv("OLLAMA_ACTION_EXTRACTION_NUM_PREDICT") else OllamaSettings.num_predict_action_extraction,
                temperature_action_extraction=_get_float("OLLAMA_ACTION_EXTRACTION_TEMPERATURE", OllamaSettings.temperature_action_extraction),
                model_generate_clarification=os.getenv("OLLAMA_GENERATE_CLARIFICATION_MODEL", OllamaSettings.model_generate_clarification),
                timeout_generate_clarification=_get_float("OLLAMA_GENERATE_CLARIFICATION_TIMEOUT", OllamaSettings.timeout_generate_clarification),
                num_ctx_generate_clarification=_get_int("OLLAMA_GENERATE_CLARIFICATION_NUM_CTX", OllamaSettings.num_ctx_generate_clarification),
                num_predict_generate_clarification=_get_int("OLLAMA_GENERATE_CLARIFICATION_NUM_PREDICT", OllamaSettings.num_predict_generate_clarification) if os.getenv("OLLAMA_GENERATE_CLARIFICATION_NUM_PREDICT") else OllamaSettings.num_predict_generate_clarification,
                temperature_generate_clarification=_get_float("OLLAMA_GENERATE_CLARIFICATION_TEMPERATURE", OllamaSettings.temperature_generate_clarification),
                json_retry_count_generate_clarification=_get_int("OLLAMA_GENERATE_CLARIFICATION_JSON_RETRY_COUNT", OllamaSettings.json_retry_count_generate_clarification),
                model_generate_human_supporting=os.getenv("OLLAMA_GENERATE_HUMAN_SUPPORTING_MODEL", OllamaSettings.model_generate_human_supporting),
                timeout_generate_human_supporting=_get_float("OLLAMA_GENERATE_HUMAN_SUPPORTING_TIMEOUT", OllamaSettings.timeout_generate_human_supporting),
                num_ctx_generate_human_supporting=_get_int("OLLAMA_GENERATE_HUMAN_SUPPORTING_NUM_CTX", OllamaSettings.num_ctx_generate_human_supporting),
                num_predict_generate_human_supporting=_get_int("OLLAMA_GENERATE_HUMAN_SUPPORTING_NUM_PREDICT", OllamaSettings.num_predict_generate_human_supporting) if os.getenv("OLLAMA_GENERATE_HUMAN_SUPPORTING_NUM_PREDICT") else OllamaSettings.num_predict_generate_human_supporting,
                temperature_generate_human_supporting=_get_float("OLLAMA_GENERATE_HUMAN_SUPPORTING_TEMPERATURE", OllamaSettings.temperature_generate_human_supporting),
                json_retry_count_generate_human_supporting=_get_int("OLLAMA_GENERATE_HUMAN_SUPPORTING_JSON_RETRY_COUNT", OllamaSettings.json_retry_count_generate_human_supporting),
                model_generate_reminder_supporting=os.getenv("OLLAMA_GENERATE_REMINDER_SUPPORTING_MODEL", OllamaSettings.model_generate_reminder_supporting),
                timeout_generate_reminder_supporting=_get_float("OLLAMA_GENERATE_REMINDER_SUPPORTING_TIMEOUT", OllamaSettings.timeout_generate_reminder_supporting),
                num_ctx_generate_reminder_supporting=_get_int("OLLAMA_GENERATE_REMINDER_SUPPORTING_NUM_CTX", OllamaSettings.num_ctx_generate_reminder_supporting),
                num_predict_generate_reminder_supporting=_get_int("OLLAMA_GENERATE_REMINDER_SUPPORTING_NUM_PREDICT", OllamaSettings.num_predict_generate_reminder_supporting) if os.getenv("OLLAMA_GENERATE_REMINDER_SUPPORTING_NUM_PREDICT") else OllamaSettings.num_predict_generate_reminder_supporting,
                temperature_generate_reminder_supporting=_get_float("OLLAMA_GENERATE_REMINDER_SUPPORTING_TEMPERATURE", OllamaSettings.temperature_generate_reminder_supporting),
                json_retry_count_generate_reminder_supporting=_get_int("OLLAMA_GENERATE_REMINDER_SUPPORTING_JSON_RETRY_COUNT", OllamaSettings.json_retry_count_generate_reminder_supporting),
                model_clarification_merge=os.getenv("OLLAMA_CLARIFICATION_MERGE_MODEL", OllamaSettings.model_clarification_merge),
                timeout_clarification_merge=_get_float("OLLAMA_CLARIFICATION_MERGE_TIMEOUT", OllamaSettings.timeout_clarification_merge),
                num_ctx_clarification_merge=_get_int("OLLAMA_CLARIFICATION_MERGE_NUM_CTX", OllamaSettings.num_ctx_clarification_merge),
                num_predict_clarification_merge=_get_int("OLLAMA_CLARIFICATION_MERGE_NUM_PREDICT", OllamaSettings.num_predict_clarification_merge) if os.getenv("OLLAMA_CLARIFICATION_MERGE_NUM_PREDICT") else OllamaSettings.num_predict_clarification_merge,
                temperature_clarification_merge=_get_float("OLLAMA_CLARIFICATION_MERGE_TEMPERATURE", OllamaSettings.temperature_clarification_merge),
                json_retry_count_clarification_merge=_get_int("OLLAMA_CLARIFICATION_MERGE_JSON_RETRY_COUNT", OllamaSettings.json_retry_count_clarification_merge),
                model_answer=os.getenv("OLLAMA_ANSWER_MODEL", OllamaSettings.model_answer),
                timeout_answer=_get_float("OLLAMA_ANSWER_TIMEOUT", OllamaSettings.timeout_answer),
                num_ctx_answer=_get_int("OLLAMA_ANSWER_NUM_CTX", OllamaSettings.num_ctx_answer),
                num_predict_answer=_get_int("OLLAMA_ANSWER_NUM_PREDICT", OllamaSettings.num_predict_answer) if os.getenv("OLLAMA_ANSWER_NUM_PREDICT") else OllamaSettings.num_predict_answer,
                temperature_answer=_get_float("OLLAMA_ANSWER_TEMPERATURE", OllamaSettings.temperature_answer),
                model_writing=os.getenv("OLLAMA_WRITING_MODEL", OllamaSettings.model_writing),
                timeout_writing=_get_float("OLLAMA_WRITING_TIMEOUT", OllamaSettings.timeout_writing),
                num_ctx_writing=_get_int("OLLAMA_WRITING_NUM_CTX", OllamaSettings.num_ctx_writing),
                num_predict_writing=_get_int("OLLAMA_WRITING_NUM_PREDICT", OllamaSettings.num_predict_writing) if os.getenv("OLLAMA_WRITING_NUM_PREDICT") else OllamaSettings.num_predict_writing,
                temperature_writing=_get_float("OLLAMA_WRITING_TEMPERATURE", OllamaSettings.temperature_writing),
                model_risky_action=os.getenv("OLLAMA_RISKY_ACTION_MODEL", OllamaSettings.model_risky_action),
                timeout_risky_action=_get_float("OLLAMA_RISKY_ACTION_TIMEOUT", OllamaSettings.timeout_risky_action),
                num_ctx_risky_action=_get_int("OLLAMA_RISKY_ACTION_NUM_CTX", OllamaSettings.num_ctx_risky_action),
                num_predict_risky_action=_get_int("OLLAMA_RISKY_ACTION_NUM_PREDICT", OllamaSettings.num_predict_risky_action) if os.getenv("OLLAMA_RISKY_ACTION_NUM_PREDICT") else OllamaSettings.num_predict_risky_action,
                temperature_risky_action=_get_float("OLLAMA_RISKY_ACTION_TEMPERATURE", OllamaSettings.temperature_risky_action),
                json_retry_count_risky_action=_get_int("OLLAMA_RISKY_ACTION_JSON_RETRY_COUNT", OllamaSettings.json_retry_count_risky_action),
                model_retrieval_validation=os.getenv("OLLAMA_RETRIEVAL_VALIDATION_MODEL", OllamaSettings.model_retrieval_validation),
                timeout_retrieval_validation=_get_float("OLLAMA_RETRIEVAL_VALIDATION_TIMEOUT", OllamaSettings.timeout_retrieval_validation),
                num_ctx_retrieval_validation=_get_int("OLLAMA_RETRIEVAL_VALIDATION_NUM_CTX", OllamaSettings.num_ctx_retrieval_validation),
                num_predict_retrieval_validation=_get_int("OLLAMA_RETRIEVAL_VALIDATION_NUM_PREDICT", OllamaSettings.num_predict_retrieval_validation) if os.getenv("OLLAMA_RETRIEVAL_VALIDATION_NUM_PREDICT") else OllamaSettings.num_predict_retrieval_validation,
                temperature_retrieval_validation=_get_float("OLLAMA_RETRIEVAL_VALIDATION_TEMPERATURE", OllamaSettings.temperature_retrieval_validation),
                model_general_sub_branch_detection=os.getenv("OLLAMA_GENERAL_SUB_BRANCH_DETECTION_MODEL", OllamaSettings.model_general_sub_branch_detection),
                timeout_general_sub_branch_detection=_get_float("OLLAMA_GENERAL_SUB_BRANCH_DETECTION_TIMEOUT", OllamaSettings.timeout_general_sub_branch_detection),
                num_ctx_general_sub_branch_detection=_get_int("OLLAMA_GENERAL_SUB_BRANCH_DETECTION_NUM_CTX", OllamaSettings.num_ctx_general_sub_branch_detection),
                num_predict_general_sub_branch_detection=_get_int("OLLAMA_GENERAL_SUB_BRANCH_DETECTION_NUM_PREDICT", OllamaSettings.num_predict_general_sub_branch_detection) if os.getenv("OLLAMA_GENERAL_SUB_BRANCH_DETECTION_NUM_PREDICT") else OllamaSettings.num_predict_general_sub_branch_detection,
                temperature_general_sub_branch_detection=_get_float("OLLAMA_GENERAL_SUB_BRANCH_DETECTION_TEMPERATURE", OllamaSettings.temperature_general_sub_branch_detection),
                model_content_composer_react=os.getenv("OLLAMA_CONTENT_COMPOSER_REACT_MODEL", OllamaSettings.model_content_composer_react),
                timeout_content_composer_react=_get_float("OLLAMA_CONTENT_COMPOSER_REACT_TIMEOUT", OllamaSettings.timeout_content_composer_react),
                num_ctx_content_composer_react=_get_int("OLLAMA_CONTENT_COMPOSER_REACT_NUM_CTX", OllamaSettings.num_ctx_content_composer_react),
                num_predict_content_composer_react=_get_int("OLLAMA_CONTENT_COMPOSER_REACT_NUM_PREDICT", OllamaSettings.num_predict_content_composer_react) if os.getenv("OLLAMA_CONTENT_COMPOSER_REACT_NUM_PREDICT") else OllamaSettings.num_predict_content_composer_react,
                temperature_content_composer_react=_get_float("OLLAMA_CONTENT_COMPOSER_REACT_TEMPERATURE", OllamaSettings.temperature_content_composer_react),
                model_action_planning=os.getenv("OLLAMA_ACTION_PLANNING_MODEL", OllamaSettings.model_action_planning),
                timeout_action_planning=_get_float("OLLAMA_ACTION_PLANNING_TIMEOUT", OllamaSettings.timeout_action_planning),
                num_ctx_action_planning=_get_int("OLLAMA_ACTION_PLANNING_NUM_CTX", OllamaSettings.num_ctx_action_planning),
                num_predict_action_planning=_get_int("OLLAMA_ACTION_PLANNING_NUM_PREDICT", OllamaSettings.num_predict_action_planning) if os.getenv("OLLAMA_ACTION_PLANNING_NUM_PREDICT") else OllamaSettings.num_predict_action_planning,
                temperature_action_planning=_get_float("OLLAMA_ACTION_PLANNING_TEMPERATURE", OllamaSettings.temperature_action_planning),
            ),

            retrieval=RetrievalSettings(
                max_results=_get_int("FINAL_CONTEXT_TOP_K", _get_int("ASSISTANT_MAX_RESULTS", RetrievalSettings.max_results)),
                bm25_top_k=_get_int("BM25_TOP_K", RetrievalSettings.bm25_top_k),
                chroma_top_k=_get_int("CHROMA_TOP_K", RetrievalSettings.chroma_top_k),
                min_confidence=_get_float("RETRIEVAL_MIN_CONFIDENCE", RetrievalSettings.min_confidence),
                conversation_min_confidence=_get_float(
                    "CONVERSATION_CONTEXT_MIN_CONFIDENCE",
                    _get_float("ASSISTANT_CONVERSATION_MIN_CONFIDENCE", RetrievalSettings.conversation_min_confidence),
                ),
                knowledge_min_confidence=_get_float(
                    "KNOWLEDGE_CONTEXT_MIN_CONFIDENCE",
                    _get_float("ASSISTANT_KNOWLEDGE_MIN_CONFIDENCE", RetrievalSettings.knowledge_min_confidence),
                ),
                rrf_k=_get_int("RRF_K", _get_int("ASSISTANT_RRF_K", RetrievalSettings.rrf_k)),
                lexical_weight=_get_float(
                    "ASSISTANT_RRF_LEXICAL_WEIGHT", RetrievalSettings.lexical_weight
                ),
                semantic_weight=_get_float(
                    "ASSISTANT_RRF_SEMANTIC_WEIGHT", RetrievalSettings.semantic_weight
                ),
                rerank_candidate_limit=_get_int(
                    "RERANKER_TOP_K",
                    _get_int("ASSISTANT_RERANK_CANDIDATE_LIMIT", RetrievalSettings.rerank_candidate_limit),
                ),
            ),
            opensearch=OpenSearchSettings(
                url=os.getenv("OPENSEARCH_URL", OpenSearchSettings.url),
                username=os.getenv("OPENSEARCH_USERNAME") or None,
                password=os.getenv("OPENSEARCH_PASSWORD") or None,
                verify_certs=_get_bool(
                    "OPENSEARCH_VERIFY_CERTS", OpenSearchSettings.verify_certs
                ),
                conversation_index=os.getenv(
                    "OPENSEARCH_CONVERSATION_INDEX",
                    OpenSearchSettings.conversation_index,
                ),
                knowledge_index=os.getenv(
                    "OPENSEARCH_KNOWLEDGE_INDEX",
                    OpenSearchSettings.knowledge_index,
                ),
                conversation_write_alias=os.getenv(
                    "OPENSEARCH_CONVERSATION_WRITE_ALIAS",
                    OpenSearchSettings.conversation_write_alias,
                ),
                knowledge_write_alias=os.getenv(
                    "OPENSEARCH_KNOWLEDGE_WRITE_ALIAS",
                    OpenSearchSettings.knowledge_write_alias,
                ),
                reminder_context_alias=os.getenv(
                    "OPENSEARCH_REMINDER_CONTEXT_ALIAS",
                    OpenSearchSettings.reminder_context_alias,
                ),
                analyzer_name=os.getenv(
                    "OPENSEARCH_ANALYZER_NAME", OpenSearchSettings.analyzer_name
                ),
                timeout_seconds=_get_int(
                    "OPENSEARCH_TIMEOUT_SECONDS", OpenSearchSettings.timeout_seconds
                ),
                max_retries=_get_int("OPENSEARCH_MAX_RETRIES", OpenSearchSettings.max_retries),
            ),
            chroma=ChromaSettings(
                path=os.getenv("ASSISTANT_CHROMA_PATH", ChromaSettings.path),
                host=os.getenv("ASSISTANT_CHROMA_HOST", ChromaSettings.host or ""),
                port=_get_int("ASSISTANT_CHROMA_PORT", ChromaSettings.port or 0),
                conversation_collection=os.getenv(
                    "ASSISTANT_CHROMA_CONVERSATION_COLLECTION",
                    ChromaSettings.conversation_collection,
                ),
                knowledge_collection=os.getenv(
                    "ASSISTANT_CHROMA_KNOWLEDGE_COLLECTION",
                    ChromaSettings.knowledge_collection,
                ),
            ),
            embeddings=EmbeddingSettings(
                model_name=os.getenv(
                    "EMBEDDING_MODEL_NAME",
                    os.getenv("ASSISTANT_EMBEDDING_MODEL", EmbeddingSettings.model_name),
                ),
                device=os.getenv("ASSISTANT_EMBEDDING_DEVICE") or None,
                batch_size=_get_int(
                    "EMBEDDING_BATCH_SIZE",
                    _get_int("ASSISTANT_EMBEDDING_BATCH_SIZE", EmbeddingSettings.batch_size),
                ),
                normalize_embeddings=_get_bool(
                    "EMBEDDING_NORMALIZE",
                    _get_bool("ASSISTANT_EMBEDDING_NORMALIZE", EmbeddingSettings.normalize_embeddings),
                ),
                max_length=_get_int("EMBEDDING_MAX_LENGTH", EmbeddingSettings.max_length),
            ),
            reranker=RerankerSettings(
                model_name=os.getenv("ASSISTANT_RERANKER_MODEL", RerankerSettings.model_name),
                endpoint_url=os.getenv("ASSISTANT_RERANKER_ENDPOINT_URL") or None,
                timeout_seconds=_get_float(
                    "ASSISTANT_RERANKER_TIMEOUT_SECONDS",
                    RerankerSettings.timeout_seconds,
                ),
                min_score=_get_float("RERANKER_MIN_SCORE", RerankerSettings.min_score),
                device=os.getenv("ASSISTANT_RERANKER_DEVICE") or None,
                batch_size=_get_int(
                    "RERANKER_BATCH_SIZE",
                    _get_int("ASSISTANT_RERANKER_BATCH_SIZE", RerankerSettings.batch_size),
                ),
                max_candidates=_get_int(
                    "ASSISTANT_RERANKER_MAX_CANDIDATES", RerankerSettings.max_candidates
                ),
            ),
            last_qa=LastQASettings(
                path=os.getenv("ASSISTANT_LAST_QA_PATH", LastQASettings.path),
                ttl_seconds=_get_int("ASSISTANT_LAST_QA_TTL_SECONDS", LastQASettings.ttl_seconds),
            ),
            worker=WorkerSettings(
                outbox_batch_size=_get_int(
                    "OUTBOX_BATCH_SIZE",
                    _get_int("ASSISTANT_OUTBOX_BATCH_SIZE", WorkerSettings.outbox_batch_size),
                ),
                outbox_max_attempts=_get_int(
                    "OUTBOX_MAX_RETRIES",
                    _get_int("ASSISTANT_OUTBOX_MAX_ATTEMPTS", WorkerSettings.outbox_max_attempts),
                ),
                outbox_worker_interval_seconds=_get_int(
                    "OUTBOX_WORKER_INTERVAL_SECONDS",
                    WorkerSettings.outbox_worker_interval_seconds,
                ),
                outbox_retry_backoff_seconds=_get_int(
                    "OUTBOX_RETRY_BACKOFF_SECONDS",
                    _get_int("ASSISTANT_OUTBOX_RETRY_BACKOFF_SECONDS", WorkerSettings.outbox_retry_backoff_seconds),
                ),
                outbox_processing_timeout_seconds=_get_int(
                    "OUTBOX_STALE_PROCESSING_AFTER_SECONDS",
                    _get_int("ASSISTANT_OUTBOX_PROCESSING_TIMEOUT_SECONDS", WorkerSettings.outbox_processing_timeout_seconds),
                ),
                autoscan_interval_seconds=_get_int(
                    "REMINDER_AUTOSCAN_INTERVAL_SECONDS",
                    _get_int("ASSISTANT_AUTOSCAN_INTERVAL_SECONDS", WorkerSettings.autoscan_interval_seconds),
                ),
            ),
            ui=UISettings(
                notification_transport=os.getenv(
                    "ASSISTANT_NOTIFICATION_TRANSPORT",
                    UISettings.notification_transport,
                ),
                redis_url=os.getenv("ASSISTANT_REDIS_URL") or None,
            ),
            api=APISettings(
                host=os.getenv("ASSISTANT_API_HOST", APISettings.host),
                port=_get_int("ASSISTANT_API_PORT", APISettings.port),
            ),
            safety=SafetySettings(
                default_timezone=os.getenv("ASSISTANT_DEFAULT_TIMEZONE", SafetySettings.default_timezone),
                reminder_duplicate_similarity_threshold=_get_float(
                    "ASSISTANT_REMINDER_DUPLICATE_SIMILARITY_THRESHOLD",
                    SafetySettings.reminder_duplicate_similarity_threshold,
                ),
                reminder_duplicate_time_window_minutes=_get_int(
                    "ASSISTANT_REMINDER_DUPLICATE_TIME_WINDOW_MINUTES",
                    SafetySettings.reminder_duplicate_time_window_minutes,
                ),
                confirmation_expiry_minutes=_get_int(
                    "ASSISTANT_CONFIRMATION_EXPIRY_MINUTES",
                    SafetySettings.confirmation_expiry_minutes,
                ),
                confirmation_high_confidence_threshold=_get_float(
                    "ASSISTANT_CONFIRMATION_HIGH_CONFIDENCE_THRESHOLD",
                    SafetySettings.confirmation_high_confidence_threshold,
                ),
                allow_missing_idempotency_key=_get_bool(
                    "ASSISTANT_ALLOW_MISSING_IDEMPOTENCY_KEY",
                    SafetySettings.allow_missing_idempotency_key,
                ),
            ),
            auth=AuthSettings(
                issuer=os.getenv("AUTH_JWT_ISSUER") or None,
                audience=os.getenv("AUTH_JWT_AUDIENCE") or None,
                jwks_url=os.getenv("AUTH_JWKS_URL") or None,
                public_key=os.getenv("AUTH_JWT_PUBLIC_KEY") or None,
                required_scopes=_get_tuple("AUTH_REQUIRED_SCOPES", AuthSettings.required_scopes),
            ),
            prompt_policy=PromptPolicySettings(
                conversation_retrieval_after_last_qa_enabled=_get_bool("ASSISTANT_CONVERSATION_RETRIEVAL_AFTER_LAST_QA", PromptPolicySettings.conversation_retrieval_after_last_qa_enabled),
                conversation_retrieval_before_intent_enabled=_get_bool("ASSISTANT_CONVERSATION_RETRIEVAL_BEFORE_INTENT", PromptPolicySettings.conversation_retrieval_before_intent_enabled),
                expected_response_type_required=_get_bool("ASSISTANT_EXPECTED_RESPONSE_TYPE_REQUIRED", PromptPolicySettings.expected_response_type_required),
                clarification_expected_response_type_required=_get_bool("ASSISTANT_CLARIFICATION_EXPECTED_RESPONSE_TYPE_REQUIRED", PromptPolicySettings.clarification_expected_response_type_required),
                expected_response_type_fallback_policy=os.getenv("ASSISTANT_EXPECTED_RESPONSE_TYPE_FALLBACK_POLICY", PromptPolicySettings.expected_response_type_fallback_policy),
                intent_min_confidence=_get_float(
                    "PROMPT_INTENT_MIN_CONFIDENCE",
                    PromptPolicySettings.intent_min_confidence,
                ),
                action_min_confidence=_get_float(
                    "ACTION_MIN_CONFIDENCE",
                    _get_float("PROMPT_ACTION_MIN_CONFIDENCE", PromptPolicySettings.action_min_confidence),
                ),
                last_qa_min_confidence=_get_float(
                    "LAST_QA_MIN_CONFIDENCE",
                    _get_float("PROMPT_LAST_QA_MIN_CONFIDENCE", PromptPolicySettings.last_qa_min_confidence),
                ),
                last_qa_skip_broad_retrieval_min_confidence=_get_float(
                    "LAST_QA_SKIP_BROAD_RETRIEVAL_MIN_CONFIDENCE",
                    PromptPolicySettings.last_qa_skip_broad_retrieval_min_confidence,
                ),
                human_in_the_loop_min_confidence=_get_float(
                    "PROMPT_HITL_MIN_CONFIDENCE",
                    PromptPolicySettings.human_in_the_loop_min_confidence,
                ),
                clarification_merge_min_confidence=_get_float(
                    "LAST_QA_CLARIFICATION_MERGE_MIN_CONFIDENCE",
                    _get_float("PROMPT_CLARIFICATION_MERGE_MIN_CONFIDENCE", PromptPolicySettings.clarification_merge_min_confidence),
                ),
                supporting_question_match_threshold=_get_float(
                    "PROMPT_SUPPORTING_QUESTION_MATCH_THRESHOLD",
                    PromptPolicySettings.supporting_question_match_threshold,
                ),
                skip_broad_retrieval_allowed_relationships=_get_tuple(
                    "PROMPT_SKIP_BROAD_RETRIEVAL_ALLOWED_RELATIONSHIPS",
                    PromptPolicySettings.skip_broad_retrieval_allowed_relationships,
                ),
                clarification_merge_enabled=_get_bool(
                    "PROMPT_CLARIFICATION_MERGE_ENABLED",
                    PromptPolicySettings.clarification_merge_enabled,
                ),
                last_qa_enable_reminder_metadata_reply=_get_bool(
                    "PROMPT_LAST_QA_ENABLE_REMINDER_METADATA_REPLY",
                    PromptPolicySettings.last_qa_enable_reminder_metadata_reply,
                ),
                risky_action_validation_enabled=_get_bool(
                    "RISKY_ACTION_VALIDATION_ENABLED",
                    _get_bool("PROMPT_RISKY_ACTION_VALIDATION_ENABLED", PromptPolicySettings.risky_action_validation_enabled),
                ),
                risky_action_operations=_get_tuple(
                    "RISKY_ACTION_OPERATIONS",
                    _get_tuple("PROMPT_RISKY_ACTION_OPERATIONS", PromptPolicySettings.risky_action_operations),
                ),
                risky_action_confidence_threshold=_get_float(
                    "RISKY_ACTION_CONFIDENCE_THRESHOLD",
                    _get_float("PROMPT_RISKY_ACTION_CONFIDENCE_THRESHOLD", PromptPolicySettings.risky_action_confidence_threshold),
                ),
                knowledge_modify_requires_replacement_text=_get_bool(
                    "PROMPT_KNOWLEDGE_MODIFY_REQUIRES_REPLACEMENT_TEXT",
                    PromptPolicySettings.knowledge_modify_requires_replacement_text,
                ),
                context_filter_knowledge_min_confidence=_get_float(
                    "KNOWLEDGE_CONTEXT_MIN_CONFIDENCE",
                    _get_float("PROMPT_CONTEXT_FILTER_KNOWLEDGE_MIN_CONFIDENCE", PromptPolicySettings.context_filter_knowledge_min_confidence),
                ),
                context_filter_allowed_reminder_statuses=_get_tuple(
                    "PROMPT_CONTEXT_FILTER_ALLOWED_REMINDER_STATUSES",
                    PromptPolicySettings.context_filter_allowed_reminder_statuses,
                ),
                question_generation_enabled=_get_bool(
                    "PROMPT_QUESTION_GENERATION_ENABLED",
                    PromptPolicySettings.question_generation_enabled,
                ),
                reminder_supporting_question_enabled=_get_bool(
                    "PROMPT_REMINDER_SUPPORTING_QUESTION_ENABLED",
                    PromptPolicySettings.reminder_supporting_question_enabled,
                ),
                question_generation_confidence_threshold=_get_float(
                    "PROMPT_QUESTION_GENERATION_CONFIDENCE_THRESHOLD",
                    PromptPolicySettings.question_generation_confidence_threshold,
                ),
                reminder_supporting_question_min_confidence=_get_float(
                    "PROMPT_REMINDER_SUPPORTING_QUESTION_MIN_CONFIDENCE",
                    PromptPolicySettings.reminder_supporting_question_min_confidence,
                ),
                human_supporting_question_max_count=_get_int(
                    "PROMPT_HUMAN_SUPPORTING_QUESTION_MAX_COUNT",
                    PromptPolicySettings.human_supporting_question_max_count,
                ),
                question_generation_fallback_policy=os.getenv(
                    "PROMPT_QUESTION_GENERATION_FALLBACK_POLICY",
                    PromptPolicySettings.question_generation_fallback_policy,
                ),
                mutation_partial_execution_policy=MutationPartialExecutionPolicy(
                    os.getenv(
                        "PROMPT_MUTATION_PARTIAL_EXECUTION_POLICY",
                        PromptPolicySettings.mutation_partial_execution_policy.value,
                    )
                ),
                knowledge_target_relevance_threshold=_get_float(
                    "PROMPT_KNOWLEDGE_TARGET_RELEVANCE_THRESHOLD",
                    PromptPolicySettings.knowledge_target_relevance_threshold,
                ),
                knowledge_target_ambiguity_margin=_get_float(
                    "PROMPT_KNOWLEDGE_TARGET_AMBIGUITY_MARGIN",
                    PromptPolicySettings.knowledge_target_ambiguity_margin,
                ),
                knowledge_target_not_found_policy=TargetNotFoundPolicy(
                    os.getenv(
                        "PROMPT_KNOWLEDGE_TARGET_NOT_FOUND_POLICY",
                        PromptPolicySettings.knowledge_target_not_found_policy.value,
                    )
                ),
                unsupported_action_policy=UnsupportedActionPolicy(
                    os.getenv(
                        "PROMPT_UNSUPPORTED_ACTION_POLICY",
                        PromptPolicySettings.unsupported_action_policy.value,
                    )
                ),
            ),
            reminder_resolver=ReminderTargetResolverSettings(
                reminder_subject_weight=_get_float("REMINDER_SUBJECT_WEIGHT", ReminderTargetResolverSettings.reminder_subject_weight),
                reminder_summary_weight=_get_float("REMINDER_SUMMARY_WEIGHT", ReminderTargetResolverSettings.reminder_summary_weight),
                reminder_raw_text_weight=_get_float("REMINDER_RAW_TEXT_WEIGHT", ReminderTargetResolverSettings.reminder_raw_text_weight),
                reminder_time_weight=_get_float("REMINDER_TIME_WEIGHT", ReminderTargetResolverSettings.reminder_time_weight),
                reminder_entity_weight=_get_float("REMINDER_ENTITY_WEIGHT", ReminderTargetResolverSettings.reminder_entity_weight),
                reminder_status_weight=_get_float("REMINDER_STATUS_WEIGHT", ReminderTargetResolverSettings.reminder_status_weight),
                reminder_recency_weight=_get_float("REMINDER_RECENCY_WEIGHT", ReminderTargetResolverSettings.reminder_recency_weight),
                reminder_target_relevance_threshold=_get_float("REMINDER_TARGET_MIN_SCORE", _get_float("REMINDER_TARGET_RELEVANCE_THRESHOLD", ReminderTargetResolverSettings.reminder_target_relevance_threshold)),
                reminder_target_ambiguity_margin=_get_float("REMINDER_TARGET_AMBIGUITY_MARGIN", ReminderTargetResolverSettings.reminder_target_ambiguity_margin),
                reminder_llm_rerank_enabled=_get_bool("REMINDER_LLM_RERANK_ENABLED", ReminderTargetResolverSettings.reminder_llm_rerank_enabled),
                reminder_llm_rerank_threshold=_get_float("REMINDER_LLM_RERANK_THRESHOLD", ReminderTargetResolverSettings.reminder_llm_rerank_threshold),
                reminder_llm_rerank_max_candidates=_get_int("REMINDER_LLM_RERANK_MAX_CANDIDATES", ReminderTargetResolverSettings.reminder_llm_rerank_max_candidates),
                reminder_llm_rerank_model=os.getenv("REMINDER_LLM_RERANK_MODEL") or None,
                reminder_llm_rerank_json_retry_count=_get_int("REMINDER_LLM_RERANK_JSON_RETRY_COUNT", ReminderTargetResolverSettings.reminder_llm_rerank_json_retry_count),
                reminder_fuzzy_matcher=os.getenv("REMINDER_FUZZY_MATCHER", ReminderTargetResolverSettings.reminder_fuzzy_matcher),
                reminder_fuzzy_match_threshold=_get_float("REMINDER_FUZZY_MATCH_THRESHOLD", ReminderTargetResolverSettings.reminder_fuzzy_match_threshold),
                reminder_target_not_found_policy=TargetNotFoundPolicy(os.getenv("REMINDER_TARGET_NOT_FOUND_POLICY", ReminderTargetResolverSettings.reminder_target_not_found_policy.value)),
                reminder_target_candidate_limit=_get_int("REMINDER_CANDIDATE_LIMIT", _get_int("REMINDER_TARGET_CANDIDATE_LIMIT", ReminderTargetResolverSettings.reminder_target_candidate_limit)),
                allowed_reminder_modify_statuses=_get_tuple("ALLOWED_REMINDER_MODIFY_STATUSES", ReminderTargetResolverSettings.allowed_reminder_modify_statuses),
                allowed_reminder_turn_on_statuses=_get_tuple("ALLOWED_REMINDER_TURN_ON_STATUSES", ReminderTargetResolverSettings.allowed_reminder_turn_on_statuses),
                allowed_reminder_turn_off_statuses=_get_tuple("ALLOWED_REMINDER_TURN_OFF_STATUSES", ReminderTargetResolverSettings.allowed_reminder_turn_off_statuses),
                allowed_reminder_delete_statuses=_get_tuple("ALLOWED_REMINDER_DELETE_STATUSES", ReminderTargetResolverSettings.allowed_reminder_delete_statuses),
                reminder_delete_status_policy=os.getenv("REMINDER_DELETE_STATUS_POLICY", ReminderTargetResolverSettings.reminder_delete_status_policy),
            ),
            retrieval_validation=RetrievalValidationSettings(
                knowledge_llm_validation_enabled=_get_bool("KNOWLEDGE_LLM_VALIDATION_ENABLED", RetrievalValidationSettings.knowledge_llm_validation_enabled),
                knowledge_llm_validation_model=os.getenv("KNOWLEDGE_LLM_VALIDATION_MODEL") or None,
                knowledge_llm_validation_min_confidence=_get_float("KNOWLEDGE_LLM_VALIDATION_MIN_CONFIDENCE", RetrievalValidationSettings.knowledge_llm_validation_min_confidence),
                knowledge_llm_validation_json_retry_count=_get_int("KNOWLEDGE_LLM_VALIDATION_JSON_RETRY_COUNT", RetrievalValidationSettings.knowledge_llm_validation_json_retry_count),
                knowledge_llm_validation_max_candidates=_get_int("KNOWLEDGE_LLM_VALIDATION_MAX_CANDIDATES", RetrievalValidationSettings.knowledge_llm_validation_max_candidates),
                knowledge_llm_validation_failure_policy=os.getenv("KNOWLEDGE_LLM_VALIDATION_FAILURE_POLICY", RetrievalValidationSettings.knowledge_llm_validation_failure_policy),
                reminder_llm_validation_enabled=_get_bool("REMINDER_LLM_VALIDATION_ENABLED", RetrievalValidationSettings.reminder_llm_validation_enabled),
                reminder_llm_validation_model=os.getenv("REMINDER_LLM_VALIDATION_MODEL") or None,
                reminder_llm_validation_min_confidence=_get_float("REMINDER_LLM_VALIDATION_MIN_CONFIDENCE", RetrievalValidationSettings.reminder_llm_validation_min_confidence),
                reminder_llm_validation_json_retry_count=_get_int("REMINDER_LLM_VALIDATION_JSON_RETRY_COUNT", RetrievalValidationSettings.reminder_llm_validation_json_retry_count),
                reminder_llm_validation_max_candidates=_get_int("REMINDER_LLM_VALIDATION_MAX_CANDIDATES", RetrievalValidationSettings.reminder_llm_validation_max_candidates),
                destructive_action_requires_unambiguous_target=_get_bool("DESTRUCTIVE_ACTION_REQUIRES_UNAMBIGUOUS_TARGET", RetrievalValidationSettings.destructive_action_requires_unambiguous_target),
            ),
            knowledge_chunks=KnowledgeChunkSettings(
                chunk_size_tokens=_get_int("KNOWLEDGE_CHUNK_SIZE_TOKENS", KnowledgeChunkSettings.chunk_size_tokens),
                chunk_overlap_tokens=_get_int("KNOWLEDGE_CHUNK_OVERLAP_TOKENS", KnowledgeChunkSettings.chunk_overlap_tokens),
                min_chunk_tokens=_get_int("KNOWLEDGE_MIN_CHUNK_TOKENS", KnowledgeChunkSettings.min_chunk_tokens),
                max_chunk_tokens=_get_int("KNOWLEDGE_MAX_CHUNK_TOKENS", KnowledgeChunkSettings.max_chunk_tokens),
            ),
            service_wait=ServiceWaitSettings(
                timeout_seconds=_get_int(
                    "ASSISTANT_SERVICE_WAIT_SECONDS",
                    ServiceWaitSettings.timeout_seconds,
                ),
                probe_timeout_seconds=_get_int(
                    "ASSISTANT_SERVICE_PROBE_TIMEOUT_SECONDS",
                    ServiceWaitSettings.probe_timeout_seconds,
                ),
                poll_interval_seconds=_get_int(
                    "ASSISTANT_SERVICE_POLL_INTERVAL_SECONDS",
                    ServiceWaitSettings.poll_interval_seconds,
                ),
                auto_pull_ollama_models=_get_bool(
                    "ASSISTANT_AUTO_PULL_OLLAMA_MODELS",
                    ServiceWaitSettings.auto_pull_ollama_models,
                ),
            ),
            operations=OperationsSettings(
                structured_logs_enabled=_get_bool("ASSISTANT_STRUCTURED_LOGS_ENABLED", OperationsSettings.structured_logs_enabled),
                log_raw_content=_get_bool("ASSISTANT_LOG_RAW_CONTENT", OperationsSettings.log_raw_content),
                debug_trace_responses=_get_bool("ASSISTANT_DEBUG_TRACE_RESPONSES", OperationsSettings.debug_trace_responses),
                metrics_enabled=_get_bool("ASSISTANT_METRICS_ENABLED", OperationsSettings.metrics_enabled),
                health_strict_opensearch=_get_bool("ASSISTANT_HEALTH_STRICT_OPENSEARCH", OperationsSettings.health_strict_opensearch),
                health_strict_chroma=_get_bool("ASSISTANT_HEALTH_STRICT_CHROMA", OperationsSettings.health_strict_chroma),
                health_strict_ollama=_get_bool("ASSISTANT_HEALTH_STRICT_OLLAMA", OperationsSettings.health_strict_ollama),
                health_strict_redis=_get_bool("ASSISTANT_HEALTH_STRICT_REDIS", OperationsSettings.health_strict_redis),
                health_strict_storage=_get_bool("ASSISTANT_HEALTH_STRICT_STORAGE", OperationsSettings.health_strict_storage),
                index_rebuild_batch_size=_get_int("ASSISTANT_INDEX_REBUILD_BATCH_SIZE", OperationsSettings.index_rebuild_batch_size),
                drift_repair_enabled=_get_bool("ASSISTANT_DRIFT_REPAIR_ENABLED", OperationsSettings.drift_repair_enabled),
                recurrence_default_timezone=os.getenv("ASSISTANT_RECURRENCE_DEFAULT_TIMEZONE", OperationsSettings.recurrence_default_timezone),
                eval_top_1_threshold=_get_float("ASSISTANT_EVAL_TOP_1_THRESHOLD", OperationsSettings.eval_top_1_threshold),
                eval_top_3_threshold=_get_float("ASSISTANT_EVAL_TOP_3_THRESHOLD", OperationsSettings.eval_top_3_threshold),
                eval_wrong_target_rate_max=_get_float("ASSISTANT_EVAL_WRONG_TARGET_RATE_MAX", OperationsSettings.eval_wrong_target_rate_max),
                eval_false_mutation_rate_max=_get_float("ASSISTANT_EVAL_FALSE_MUTATION_RATE_MAX", OperationsSettings.eval_false_mutation_rate_max),
            ),
            debug=DebugSettings(
                db_path=os.getenv("ASSISTANT_DEBUG_DB_PATH", DebugSettings.db_path),
                user_id=os.getenv("ASSISTANT_DEBUG_USER_ID", DebugSettings.user_id),
                conversation_min_confidence=_get_float(
                    "ASSISTANT_DEBUG_CONVERSATION_MIN_CONFIDENCE",
                    DebugSettings.conversation_min_confidence,
                ),
                knowledge_min_confidence=_get_float(
                    "ASSISTANT_DEBUG_KNOWLEDGE_MIN_CONFIDENCE",
                    DebugSettings.knowledge_min_confidence,
                ),
                max_results=_get_int("ASSISTANT_DEBUG_MAX_RESULTS", DebugSettings.max_results),
                rrf_k=_get_int("ASSISTANT_DEBUG_RRF_K", DebugSettings.rrf_k),
                lexical_weight=_get_float(
                    "ASSISTANT_DEBUG_RRF_LEXICAL_WEIGHT", DebugSettings.lexical_weight
                ),
                semantic_weight=_get_float(
                    "ASSISTANT_DEBUG_RRF_SEMANTIC_WEIGHT", DebugSettings.semantic_weight
                ),
                rerank_candidate_limit=_get_int(
                    "ASSISTANT_DEBUG_RERANK_CANDIDATE_LIMIT",
                    DebugSettings.rerank_candidate_limit,
                ),
                outbox_max_attempts=_get_int(
                    "ASSISTANT_DEBUG_OUTBOX_MAX_ATTEMPTS",
                    DebugSettings.outbox_max_attempts,
                ),
                outbox_batch_size=_get_int(
                    "ASSISTANT_DEBUG_OUTBOX_BATCH_SIZE",
                    DebugSettings.outbox_batch_size,
                ),
                outbox_retry_backoff_seconds=_get_int(
                    "ASSISTANT_DEBUG_OUTBOX_RETRY_BACKOFF_SECONDS",
                    DebugSettings.outbox_retry_backoff_seconds,
                ),
                outbox_processing_timeout_seconds=_get_int(
                    "ASSISTANT_DEBUG_OUTBOX_PROCESSING_TIMEOUT_SECONDS",
                    DebugSettings.outbox_processing_timeout_seconds,
                ),
                autoscan_interval_seconds=_get_int(
                    "ASSISTANT_DEBUG_AUTOSCAN_INTERVAL_SECONDS",
                    DebugSettings.autoscan_interval_seconds,
                ),
                opensearch_conversation_index=os.getenv(
                    "DEBUG_OPENSEARCH_CONVERSATION_INDEX",
                    DebugSettings.opensearch_conversation_index,
                ),
                opensearch_knowledge_index=os.getenv(
                    "DEBUG_OPENSEARCH_KNOWLEDGE_INDEX",
                    DebugSettings.opensearch_knowledge_index,
                ),
                opensearch_conversation_write_alias=os.getenv(
                    "DEBUG_OPENSEARCH_CONVERSATION_WRITE_ALIAS",
                    DebugSettings.opensearch_conversation_write_alias,
                ),
                opensearch_knowledge_write_alias=os.getenv(
                    "DEBUG_OPENSEARCH_KNOWLEDGE_WRITE_ALIAS",
                    DebugSettings.opensearch_knowledge_write_alias,
                ),
                opensearch_reminder_context_alias=os.getenv(
                    "DEBUG_OPENSEARCH_REMINDER_CONTEXT_ALIAS",
                    DebugSettings.opensearch_reminder_context_alias,
                ),
                chroma_path=os.getenv("DEBUG_CHROMA_PATH", DebugSettings.chroma_path),
                chroma_conversation_collection=os.getenv(
                    "DEBUG_CHROMA_CONVERSATION_COLLECTION",
                    DebugSettings.chroma_conversation_collection,
                ),
                chroma_knowledge_collection=os.getenv(
                    "DEBUG_CHROMA_KNOWLEDGE_COLLECTION",
                    DebugSettings.chroma_knowledge_collection,
                ),
            ),
            context_filter=ContextFilterSettings(
                conversation_min_confidence=_get_float("CONVERSATION_CONTEXT_MIN_CONFIDENCE", _get_float("CONTEXT_FILTER_CONVERSATION_MIN_CONFIDENCE", ContextFilterSettings.conversation_min_confidence)),
                conversation_approved_max_items=_get_int("FINAL_CONTEXT_TOP_K", _get_int("CONTEXT_FILTER_CONVERSATION_MAX_ITEMS", ContextFilterSettings.conversation_approved_max_items)),
                conversation_duplicate_threshold=_get_float("CONTEXT_FILTER_CONVERSATION_DEDUP_THRESHOLD", ContextFilterSettings.conversation_duplicate_threshold),
                knowledge_approved_max_items=_get_int("FINAL_CONTEXT_TOP_K", _get_int("CONTEXT_FILTER_KNOWLEDGE_MAX_ITEMS", ContextFilterSettings.knowledge_approved_max_items)),
                knowledge_duplicate_threshold=_get_float("CONTEXT_FILTER_KNOWLEDGE_DEDUP_THRESHOLD", ContextFilterSettings.knowledge_duplicate_threshold),
                low_information_text_patterns=_get_tuple("CONTEXT_FILTER_LOW_INFO_PATTERNS", ContextFilterSettings.low_information_text_patterns),
                reminder_approved_max_items=_get_int("FINAL_CONTEXT_TOP_K", _get_int("CONTEXT_FILTER_REMINDER_MAX_ITEMS", ContextFilterSettings.reminder_approved_max_items)),
                reminder_min_confidence=_get_float("REMINDER_CONTEXT_MIN_CONFIDENCE", ContextFilterSettings.reminder_min_confidence),
                semantic_context_judge_enabled=_get_bool("CONTEXT_FILTER_SEMANTIC_JUDGE_ENABLED", ContextFilterSettings.semantic_context_judge_enabled),
                context_filter_debug_diagnostics_enabled=_get_bool("CONTEXT_FILTER_DEBUG_DIAGNOSTICS", ContextFilterSettings.context_filter_debug_diagnostics_enabled),
                low_information_min_chars=_get_int("CONTEXT_FILTER_LOW_INFO_MIN_CHARS", ContextFilterSettings.low_information_min_chars),
            ),
            general_purpose=GeneralPurposeSettings(
                general_sub_branch_detector_enabled=_get_bool("GENERAL_SUB_BRANCH_DETECTOR_ENABLED", GeneralPurposeSettings.general_sub_branch_detector_enabled),
                general_sub_branch_confidence_threshold=_get_float("GENERAL_SUB_BRANCH_CONFIDENCE_THRESHOLD", GeneralPurposeSettings.general_sub_branch_confidence_threshold),
                general_sub_branch_fallback_mode=os.getenv("GENERAL_SUB_BRANCH_FALLBACK_MODE", GeneralPurposeSettings.general_sub_branch_fallback_mode),
                general_sub_branch_detector_json_retry_count=_get_int("GENERAL_SUB_BRANCH_DETECTOR_JSON_RETRY_COUNT", GeneralPurposeSettings.general_sub_branch_detector_json_retry_count),
                content_composer_enabled=_get_bool("CONTENT_COMPOSER_ENABLED", GeneralPurposeSettings.content_composer_enabled),
                content_composer_max_iterations=_get_int("CONTENT_COMPOSER_MAX_ITERATIONS", GeneralPurposeSettings.content_composer_max_iterations),
                content_composer_tool_timeout_seconds=_get_float("CONTENT_COMPOSER_TOOL_TIMEOUT_SECONDS", GeneralPurposeSettings.content_composer_tool_timeout_seconds),
                content_composer_allowed_tools=_get_tuple("CONTENT_COMPOSER_ALLOWED_TOOLS", GeneralPurposeSettings.content_composer_allowed_tools),
                content_composer_default_tool=os.getenv("CONTENT_COMPOSER_DEFAULT_TOOL", GeneralPurposeSettings.content_composer_default_tool),
                content_composer_fallback_tool=os.getenv("CONTENT_COMPOSER_FALLBACK_TOOL", GeneralPurposeSettings.content_composer_fallback_tool),
                content_composer_react_enabled=_get_bool("CONTENT_COMPOSER_REACT_ENABLED", GeneralPurposeSettings.content_composer_react_enabled),
                content_composer_debug_trace_enabled=_get_bool("CONTENT_COMPOSER_DEBUG_TRACE_ENABLED", GeneralPurposeSettings.content_composer_debug_trace_enabled),
                excel_tool_signal_keywords=_get_tuple("CONTENT_COMPOSER_EXCEL_KEYWORDS", GeneralPurposeSettings.excel_tool_signal_keywords),
                pdf_tool_signal_keywords=_get_tuple("CONTENT_COMPOSER_PDF_KEYWORDS", GeneralPurposeSettings.pdf_tool_signal_keywords),
                pptx_tool_signal_keywords=_get_tuple("CONTENT_COMPOSER_PPTX_KEYWORDS", GeneralPurposeSettings.pptx_tool_signal_keywords),
                hitl_supporting_question_enabled=_get_bool("HITL_SUPPORTING_QUESTION_ENABLED", GeneralPurposeSettings.hitl_supporting_question_enabled),
                hitl_supporting_question_confidence_threshold=_get_float("HITL_SUPPORTING_QUESTION_CONFIDENCE_THRESHOLD", GeneralPurposeSettings.hitl_supporting_question_confidence_threshold),
                hitl_supporting_question_recent_question_window=_get_int("HITL_SUPPORTING_QUESTION_RECENT_QUESTION_WINDOW", GeneralPurposeSettings.hitl_supporting_question_recent_question_window),
                hitl_supporting_question_max_length=_get_int("HITL_SUPPORTING_QUESTION_MAX_LENGTH", GeneralPurposeSettings.hitl_supporting_question_max_length),
                hitl_supporting_question_safety_mode=os.getenv("HITL_SUPPORTING_QUESTION_SAFETY_MODE", GeneralPurposeSettings.hitl_supporting_question_safety_mode),
                general_response_persistence_policy=os.getenv("GENERAL_RESPONSE_PERSISTENCE_POLICY", GeneralPurposeSettings.general_response_persistence_policy),
                sub_branch_prompt_mode=os.getenv("SUB_BRANCH_PROMPT_MODE", GeneralPurposeSettings.sub_branch_prompt_mode),
                general_response_default_topic_title=os.getenv("GENERAL_RESPONSE_DEFAULT_TOPIC_TITLE", GeneralPurposeSettings.general_response_default_topic_title),
                documents_dir=os.getenv("ASSISTANT_DOCUMENTS_DIR", GeneralPurposeSettings.documents_dir),
                artifact_storage_dir=os.getenv("ASSISTANT_ARTIFACT_STORAGE_DIR", GeneralPurposeSettings.artifact_storage_dir),
                artifact_download_base_url=os.getenv("ASSISTANT_ARTIFACT_DOWNLOAD_BASE_URL", GeneralPurposeSettings.artifact_download_base_url),
            ),
        )
