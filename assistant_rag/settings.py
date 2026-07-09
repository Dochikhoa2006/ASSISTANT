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
    pool_size: int = 5
    max_overflow: int = 10
    enable_wal: bool = True
    busy_timeout_ms: int = 5000


@dataclass(frozen=True)
class OllamaSettings:
    base_url: str = "http://localhost:11434"
    fast_model: str = "llama3.2:3b"
    balanced_model: str = "llama3.2:3b"
    accurate_model: str | None = None
    writing_model: str | None = None
    last_qa_model: str | None = None
    clarification_merge_model: str | None = None
    structured_retry_count: int = 2
    keep_alive: str = "10m"
    default_temperature: float = 0.1
    answer_temperature: float = 0.4
    timeout_seconds: float = 60.0
    last_qa_timeout_seconds: float = 60.0
    clarification_merge_timeout_seconds: float = 60.0
    clarification_question_model: str | None = None
    human_supporting_question_model: str | None = None
    reminder_supporting_question_model: str | None = None
    clarification_question_temperature: float = 0.2
    human_supporting_question_temperature: float = 0.4
    reminder_supporting_question_temperature: float = 0.3
    question_generation_timeout: float = 30.0
    clarification_question_max_tokens: int = 150
    human_supporting_question_max_tokens: int = 150
    reminder_supporting_question_max_tokens: int = 150
    clarification_question_json_retry_count: int = 2
    human_supporting_question_json_retry_count: int = 2
    reminder_supporting_question_json_retry_count: int = 2
    risky_action_model: str | None = None
    risky_action_json_retry_count: int = 2
    last_qa_json_retry_count: int = 2
    clarification_merge_json_retry_count: int = 2


@dataclass(frozen=True)
class RetrievalSettings:
    max_results: int = 6
    conversation_min_confidence: float = 0.10
    knowledge_min_confidence: float = 0.15
    rrf_k: int = 60
    lexical_weight: float = 1.0
    semantic_weight: float = 1.0
    rerank_candidate_limit: int = 24


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
    timeout_seconds: int = 30
    max_retries: int = 3


@dataclass(frozen=True)
class ChromaSettings:
    path: str = "assistant_data/chroma"
    host: str | None = "localhost"
    port: int | None = 8000
    conversation_collection: str = "conversation_hops"
    knowledge_collection: str = "knowledge_chunks"


@dataclass(frozen=True)
class EmbeddingSettings:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str | None = None
    batch_size: int = 32
    normalize_embeddings: bool = True


@dataclass(frozen=True)
class RerankerSettings:
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    endpoint_url: str | None = None
    timeout_seconds: float = 10.0
    device: str | None = None
    batch_size: int = 16
    max_candidates: int = 32


@dataclass(frozen=True)
class LastQASettings:
    path: str = "assistant_data/last_qa.sqlite3"
    ttl_seconds: int = 3600


@dataclass(frozen=True)
class WorkerSettings:
    outbox_batch_size: int = 25
    outbox_max_attempts: int = 5
    outbox_retry_backoff_seconds: int = 30
    outbox_processing_timeout_seconds: int = 300
    autoscan_interval_seconds: int = 10


@dataclass(frozen=True)
class UISettings:
    notification_transport: str = "websocket"
    redis_url: str | None = None


@dataclass(frozen=True)
class APISettings:
    host: str = "127.0.0.1"
    port: int = 8000


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
    intent_min_confidence: float = 0.45
    action_min_confidence: float = 0.55
    last_qa_min_confidence: float = 0.4
    human_in_the_loop_min_confidence: float = 0.6
    clarification_merge_min_confidence: float = 0.7
    supporting_question_match_threshold: float = 0.7
    skip_broad_retrieval_allowed_relationships: tuple[str, ...] = (
        "supporting_question_answer",
        "normal_follow_up",
        "reminder_reply",
    )
    clarification_merge_enabled: bool = True
    last_qa_enable_reminder_metadata_reply: bool = True
    risky_action_validation_enabled: bool = True
    risky_action_operations: tuple[str, ...] = ("delete", "modify")
    risky_action_confidence_threshold: float = 0.7
    knowledge_modify_requires_replacement_text: bool = True
    context_filter_knowledge_min_confidence: float = 0.15
    context_filter_allowed_reminder_statuses: tuple[str, ...] = ("scheduled", "notified")
    question_generation_enabled: bool = True
    reminder_supporting_question_enabled: bool = True
    question_generation_confidence_threshold: float = 0.6
    reminder_supporting_question_min_confidence: float = 0.6
    human_supporting_question_max_count: int = 2
    question_generation_fallback_policy: str = "fallback_message"
    mutation_partial_execution_policy: MutationPartialExecutionPolicy = MutationPartialExecutionPolicy.ALL_OR_NOTHING
    knowledge_target_relevance_threshold: float = 0.5
    knowledge_target_ambiguity_margin: float = 0.1
    knowledge_target_not_found_policy: TargetNotFoundPolicy = TargetNotFoundPolicy.SKIP_NOT_FOUND
    unsupported_action_policy: UnsupportedActionPolicy = UnsupportedActionPolicy.REJECT_AND_SKIP


@dataclass(frozen=True)
class ReminderTargetResolverSettings:
    reminder_subject_weight: float = 1.0
    reminder_summary_weight: float = 0.5
    reminder_raw_text_weight: float = 0.1
    reminder_time_weight: float = 1.0
    reminder_entity_weight: float = 0.5
    reminder_status_weight: float = 0.2
    reminder_recency_weight: float = 0.1
    
    reminder_target_relevance_threshold: float = 0.5
    reminder_target_ambiguity_margin: float = 0.1
    
    reminder_llm_rerank_enabled: bool = False
    reminder_llm_rerank_threshold: float = 0.85
    reminder_llm_rerank_max_candidates: int = 5
    reminder_llm_rerank_model: str | None = None
    reminder_llm_rerank_json_retry_count: int = 2
    
    reminder_fuzzy_matcher: str = "rapidfuzz"
    reminder_target_not_found_policy: TargetNotFoundPolicy = TargetNotFoundPolicy.SKIP_NOT_FOUND
    reminder_target_candidate_limit: int = 20
    
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
    knowledge_llm_validation_enabled: bool = False
    knowledge_llm_validation_model: str | None = None
    knowledge_llm_validation_min_confidence: float = 0.8
    knowledge_llm_validation_json_retry_count: int = 2
    knowledge_llm_validation_max_candidates: int = 5
    knowledge_llm_validation_failure_policy: str = "fail_closed"

    reminder_llm_validation_enabled: bool = False
    reminder_llm_validation_model: str | None = None
    reminder_llm_validation_min_confidence: float = 0.8
    reminder_llm_validation_json_retry_count: int = 2
    reminder_llm_validation_max_candidates: int = 5

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
class ServiceWaitSettings:
    timeout_seconds: int = 180
    probe_timeout_seconds: int = 5
    poll_interval_seconds: int = 2
    auto_pull_ollama_models: bool = True


@dataclass(frozen=True)
class DebugSettings:
    db_path: str = "assistant_data/debug_pipeline.sqlite3"
    user_id: str = "debug-user"
    conversation_min_confidence: float = 0.03
    knowledge_min_confidence: float = 0.05
    max_results: int = 4
    rrf_k: int = 60
    lexical_weight: float = 1.0
    semantic_weight: float = 1.0
    rerank_candidate_limit: int = 12
    outbox_max_attempts: int = 3
    outbox_batch_size: int = 20
    outbox_retry_backoff_seconds: int = 0
    outbox_processing_timeout_seconds: int = 60
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
    prompt_policy: PromptPolicySettings = field(default_factory=PromptPolicySettings)
    reminder_resolver: ReminderTargetResolverSettings = field(default_factory=ReminderTargetResolverSettings)
    retrieval_validation: RetrievalValidationSettings = field(default_factory=RetrievalValidationSettings)
    service_wait: ServiceWaitSettings = field(default_factory=ServiceWaitSettings)
    debug: DebugSettings = field(default_factory=DebugSettings)

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
                fast_model=os.getenv("OLLAMA_FAST_MODEL", OllamaSettings.fast_model),
                balanced_model=os.getenv(
                    "OLLAMA_BALANCED_MODEL", OllamaSettings.balanced_model
                ),
                accurate_model=os.getenv("OLLAMA_ACCURATE_MODEL") or None,
                writing_model=os.getenv("OLLAMA_WRITING_MODEL") or None,
                last_qa_model=os.getenv("OLLAMA_LAST_QA_MODEL") or None,
                clarification_merge_model=os.getenv("OLLAMA_CLARIFICATION_MERGE_MODEL") or None,
                structured_retry_count=_get_int(
                    "OLLAMA_STRUCTURED_RETRY_COUNT",
                    OllamaSettings.structured_retry_count,
                ),
                keep_alive=os.getenv("OLLAMA_KEEP_ALIVE", OllamaSettings.keep_alive),
                default_temperature=_get_float(
                    "OLLAMA_DEFAULT_TEMPERATURE", OllamaSettings.default_temperature
                ),
                answer_temperature=_get_float(
                    "OLLAMA_ANSWER_TEMPERATURE", OllamaSettings.answer_temperature
                ),
                timeout_seconds=_get_float(
                    "OLLAMA_TIMEOUT_SECONDS", OllamaSettings.timeout_seconds
                ),
                last_qa_timeout_seconds=_get_float(
                    "OLLAMA_LAST_QA_TIMEOUT_SECONDS", OllamaSettings.last_qa_timeout_seconds
                ),
                clarification_merge_timeout_seconds=_get_float(
                    "OLLAMA_CLARIFICATION_MERGE_TIMEOUT_SECONDS", OllamaSettings.clarification_merge_timeout_seconds
                ),
                clarification_question_model=os.getenv("OLLAMA_CLARIFICATION_QUESTION_MODEL") or None,
                human_supporting_question_model=os.getenv("OLLAMA_HUMAN_SUPPORTING_QUESTION_MODEL") or None,
                reminder_supporting_question_model=os.getenv("OLLAMA_REMINDER_SUPPORTING_QUESTION_MODEL") or None,
                clarification_question_temperature=_get_float("OLLAMA_CLARIFICATION_QUESTION_TEMPERATURE", OllamaSettings.clarification_question_temperature),
                human_supporting_question_temperature=_get_float("OLLAMA_HUMAN_SUPPORTING_QUESTION_TEMPERATURE", OllamaSettings.human_supporting_question_temperature),
                reminder_supporting_question_temperature=_get_float("OLLAMA_REMINDER_SUPPORTING_QUESTION_TEMPERATURE", OllamaSettings.reminder_supporting_question_temperature),
                question_generation_timeout=_get_float("OLLAMA_QUESTION_GENERATION_TIMEOUT", OllamaSettings.question_generation_timeout),
                clarification_question_max_tokens=_get_int("OLLAMA_CLARIFICATION_QUESTION_MAX_TOKENS", OllamaSettings.clarification_question_max_tokens),
                human_supporting_question_max_tokens=_get_int("OLLAMA_HUMAN_SUPPORTING_QUESTION_MAX_TOKENS", OllamaSettings.human_supporting_question_max_tokens),
                reminder_supporting_question_max_tokens=_get_int("OLLAMA_REMINDER_SUPPORTING_QUESTION_MAX_TOKENS", OllamaSettings.reminder_supporting_question_max_tokens),
                clarification_question_json_retry_count=_get_int("OLLAMA_CLARIFICATION_QUESTION_JSON_RETRY_COUNT", OllamaSettings.clarification_question_json_retry_count),
                human_supporting_question_json_retry_count=_get_int("OLLAMA_HUMAN_SUPPORTING_QUESTION_JSON_RETRY_COUNT", OllamaSettings.human_supporting_question_json_retry_count),
                reminder_supporting_question_json_retry_count=_get_int("OLLAMA_REMINDER_SUPPORTING_QUESTION_JSON_RETRY_COUNT", OllamaSettings.reminder_supporting_question_json_retry_count),
                risky_action_model=os.getenv("OLLAMA_RISKY_ACTION_MODEL") or None,
                risky_action_json_retry_count=_get_int("OLLAMA_RISKY_ACTION_JSON_RETRY_COUNT", OllamaSettings.risky_action_json_retry_count),
                last_qa_json_retry_count=_get_int("OLLAMA_LAST_QA_JSON_RETRY_COUNT", OllamaSettings.last_qa_json_retry_count),
                clarification_merge_json_retry_count=_get_int("OLLAMA_CLARIFICATION_MERGE_JSON_RETRY_COUNT", OllamaSettings.clarification_merge_json_retry_count),
            ),
            retrieval=RetrievalSettings(
                max_results=_get_int("ASSISTANT_MAX_RESULTS", RetrievalSettings.max_results),
                conversation_min_confidence=_get_float(
                    "ASSISTANT_CONVERSATION_MIN_CONFIDENCE",
                    RetrievalSettings.conversation_min_confidence,
                ),
                knowledge_min_confidence=_get_float(
                    "ASSISTANT_KNOWLEDGE_MIN_CONFIDENCE",
                    RetrievalSettings.knowledge_min_confidence,
                ),
                rrf_k=_get_int("ASSISTANT_RRF_K", RetrievalSettings.rrf_k),
                lexical_weight=_get_float(
                    "ASSISTANT_RRF_LEXICAL_WEIGHT", RetrievalSettings.lexical_weight
                ),
                semantic_weight=_get_float(
                    "ASSISTANT_RRF_SEMANTIC_WEIGHT", RetrievalSettings.semantic_weight
                ),
                rerank_candidate_limit=_get_int(
                    "ASSISTANT_RERANK_CANDIDATE_LIMIT",
                    RetrievalSettings.rerank_candidate_limit,
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
                    "ASSISTANT_EMBEDDING_MODEL", EmbeddingSettings.model_name
                ),
                device=os.getenv("ASSISTANT_EMBEDDING_DEVICE") or None,
                batch_size=_get_int(
                    "ASSISTANT_EMBEDDING_BATCH_SIZE", EmbeddingSettings.batch_size
                ),
                normalize_embeddings=_get_bool(
                    "ASSISTANT_EMBEDDING_NORMALIZE",
                    EmbeddingSettings.normalize_embeddings,
                ),
            ),
            reranker=RerankerSettings(
                model_name=os.getenv("ASSISTANT_RERANKER_MODEL", RerankerSettings.model_name),
                endpoint_url=os.getenv("ASSISTANT_RERANKER_ENDPOINT_URL") or None,
                timeout_seconds=_get_float(
                    "ASSISTANT_RERANKER_TIMEOUT_SECONDS",
                    RerankerSettings.timeout_seconds,
                ),
                device=os.getenv("ASSISTANT_RERANKER_DEVICE") or None,
                batch_size=_get_int("ASSISTANT_RERANKER_BATCH_SIZE", RerankerSettings.batch_size),
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
                    "ASSISTANT_OUTBOX_BATCH_SIZE", WorkerSettings.outbox_batch_size
                ),
                outbox_max_attempts=_get_int(
                    "ASSISTANT_OUTBOX_MAX_ATTEMPTS", WorkerSettings.outbox_max_attempts
                ),
                outbox_retry_backoff_seconds=_get_int(
                    "ASSISTANT_OUTBOX_RETRY_BACKOFF_SECONDS",
                    WorkerSettings.outbox_retry_backoff_seconds,
                ),
                outbox_processing_timeout_seconds=_get_int(
                    "ASSISTANT_OUTBOX_PROCESSING_TIMEOUT_SECONDS",
                    WorkerSettings.outbox_processing_timeout_seconds,
                ),
                autoscan_interval_seconds=_get_int(
                    "ASSISTANT_AUTOSCAN_INTERVAL_SECONDS",
                    WorkerSettings.autoscan_interval_seconds,
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
            prompt_policy=PromptPolicySettings(
                intent_min_confidence=_get_float(
                    "PROMPT_INTENT_MIN_CONFIDENCE",
                    PromptPolicySettings.intent_min_confidence,
                ),
                action_min_confidence=_get_float(
                    "PROMPT_ACTION_MIN_CONFIDENCE",
                    PromptPolicySettings.action_min_confidence,
                ),
                last_qa_min_confidence=_get_float(
                    "PROMPT_LAST_QA_MIN_CONFIDENCE",
                    PromptPolicySettings.last_qa_min_confidence,
                ),
                human_in_the_loop_min_confidence=_get_float(
                    "PROMPT_HITL_MIN_CONFIDENCE",
                    PromptPolicySettings.human_in_the_loop_min_confidence,
                ),
                clarification_merge_min_confidence=_get_float(
                    "PROMPT_CLARIFICATION_MERGE_MIN_CONFIDENCE",
                    PromptPolicySettings.clarification_merge_min_confidence,
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
                    "PROMPT_RISKY_ACTION_VALIDATION_ENABLED",
                    PromptPolicySettings.risky_action_validation_enabled,
                ),
                risky_action_operations=_get_tuple(
                    "PROMPT_RISKY_ACTION_OPERATIONS",
                    PromptPolicySettings.risky_action_operations,
                ),
                risky_action_confidence_threshold=_get_float(
                    "PROMPT_RISKY_ACTION_CONFIDENCE_THRESHOLD",
                    PromptPolicySettings.risky_action_confidence_threshold,
                ),
                knowledge_modify_requires_replacement_text=_get_bool(
                    "PROMPT_KNOWLEDGE_MODIFY_REQUIRES_REPLACEMENT_TEXT",
                    PromptPolicySettings.knowledge_modify_requires_replacement_text,
                ),
                context_filter_knowledge_min_confidence=_get_float(
                    "PROMPT_CONTEXT_FILTER_KNOWLEDGE_MIN_CONFIDENCE",
                    PromptPolicySettings.context_filter_knowledge_min_confidence,
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
                reminder_target_relevance_threshold=_get_float("REMINDER_TARGET_RELEVANCE_THRESHOLD", ReminderTargetResolverSettings.reminder_target_relevance_threshold),
                reminder_target_ambiguity_margin=_get_float("REMINDER_TARGET_AMBIGUITY_MARGIN", ReminderTargetResolverSettings.reminder_target_ambiguity_margin),
                reminder_llm_rerank_enabled=_get_bool("REMINDER_LLM_RERANK_ENABLED", ReminderTargetResolverSettings.reminder_llm_rerank_enabled),
                reminder_llm_rerank_threshold=_get_float("REMINDER_LLM_RERANK_THRESHOLD", ReminderTargetResolverSettings.reminder_llm_rerank_threshold),
                reminder_llm_rerank_max_candidates=_get_int("REMINDER_LLM_RERANK_MAX_CANDIDATES", ReminderTargetResolverSettings.reminder_llm_rerank_max_candidates),
                reminder_llm_rerank_model=os.getenv("REMINDER_LLM_RERANK_MODEL") or None,
                reminder_llm_rerank_json_retry_count=_get_int("REMINDER_LLM_RERANK_JSON_RETRY_COUNT", ReminderTargetResolverSettings.reminder_llm_rerank_json_retry_count),
                reminder_fuzzy_matcher=os.getenv("REMINDER_FUZZY_MATCHER", ReminderTargetResolverSettings.reminder_fuzzy_matcher),
                reminder_target_not_found_policy=TargetNotFoundPolicy(os.getenv("REMINDER_TARGET_NOT_FOUND_POLICY", ReminderTargetResolverSettings.reminder_target_not_found_policy.value)),
                reminder_target_candidate_limit=_get_int("REMINDER_TARGET_CANDIDATE_LIMIT", ReminderTargetResolverSettings.reminder_target_candidate_limit),
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
        )
