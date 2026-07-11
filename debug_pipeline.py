from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from getpass import getpass
import json
import logging
import sys
from typing import Any, Callable

from assistant_rag.branch_orchestration import (
    KnowledgeTargetResolver,
    ReminderTargetResolver,
    ValidatedActionBuilder,
)
from assistant_rag.action_detection import LLMActionDetector
from assistant_rag.branches import (
    BranchRouter,
    ClarificationBranch,
    GeneralResponseBranch,
    KnowledgeFactsBranch,
    ReminderBranch,
)
from assistant_rag.bundler import ChatOutput, ResponseBundler
from assistant_rag.classification import LLMLastQAResolver, LastQAResolver, QueryRewriter, can_skip_broad_retrieval
from assistant_rag.config import GeneralPurposeConfig
from assistant_rag.context_filter import HardRuleContextFilter, TwoLayerContextFilter
from assistant_rag.content_composer import AnswerGenerationTool, ContentToolRegistry, ReActContentComposer
from assistant_rag.contracts import (
    BundledResponse,
    ChatRequest,
    ExpectedResponseType,
    GeneratedQuestion,
    Intent,
    LastQAInteractionType,
    LastQAState,
    PipelineContext,
    QuestionSource,
    ResponseType,
    RetrievalResult,
)
from assistant_rag.database import SQLiteRepository, now_iso
from assistant_rag.hybrid_llm import HybridLLMClient
from assistant_rag.last_qa import InMemoryLastQAStore
from assistant_rag.llm import (
    LLMTask,
    OllamaIntentClassifier,
    OllamaLLMClient,
    OllamaModelRouter,
    _structured_attempt_prompt,
    structured_fallback_payload,
    uses_onnx_runtime,
    validate_json_schema,
)
from assistant_rag.observability import JsonLogFormatter, current_trace, new_request_id, start_trace
from assistant_rag.onnx_llm import ONNXLLMClient, _adaptive_max_new_tokens, _sentence_boundary_stop_reason
from assistant_rag.platform import PlatformSelector
from assistant_rag.production_factory import (
    build_assistant_config,
    build_production_pipeline,
    build_production_repository,
)
from assistant_rag.prompts import CONTENT_COMPOSER_REACT_SCHEMA, DEFAULT_PROMPT_REGISTRY
from assistant_rag.settings import ProductionSettings


DEBUG_USER = "debug-user"
logger = logging.getLogger(__name__)


def _debug_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))


def _debug_trace_lines(trace_summary: Any | None) -> list[str]:
    if trace_summary is None:
        return ["Debug trace: unavailable"]

    lines = [
        "Debug trace:",
        f"  request_id: {trace_summary.request_id}",
        f"  total_latency_ms: {trace_summary.total_latency_ms}",
    ]
    if not trace_summary.stages:
        lines.append("  stages: none recorded")
        return lines

    lines.append("  stages:")
    for index, stage in enumerate(trace_summary.stages, start=1):
        lines.append(
            f"    {index:02d}. {stage.stage} | {stage.latency_ms} ms"
        )
        if stage.metadata:
            meta_copy = dict(stage.metadata)
            operations = meta_copy.pop("operations", None)
            if meta_copy:
                lines.append("      metadata:")
                try:
                    formatted = json.dumps(meta_copy, indent=2, default=str)
                    for line in formatted.splitlines():
                        lines.append(f"        {line}")
                except Exception as e:
                    lines.append(f"        {meta_copy}")
            if operations:
                for op in operations:
                    op_name = op.get("operation", "unknown")
                    op_lat = op.get("latency_ms", 0.0)
                    lines.append(f"      - {op_name} | {op_lat} ms")
                    op_meta = {k: v for k, v in op.items() if k not in ("operation", "latency_ms")}
                    if op_meta:
                        try:
                            formatted = json.dumps(op_meta, indent=2, default=str)
                            for line in formatted.splitlines():
                                if line.strip() not in ("{", "}"):
                                    lines.append(f"          {line.strip()}")
                        except Exception:
                            lines.append(f"          {op_meta}")
    return lines


def _debug_response_lines(response: BundledResponse) -> list[str]:
    lines = [
        "Debug response:",
        f"  response_type: {response.response_type.value}",
        f"  conversation_topic_id: {response.conversation_topic_id or 'none'}",
        f"  conversation_hop_id: {response.conversation_hop_id or 'none'}",
    ]

    def _add_json_field(name: str, value: Any) -> None:
        if not value:
            lines.append(f"  {name}: {value}")
            return
        lines.append(f"  {name}:")
        try:
            formatted = json.dumps(value, indent=2, default=str)
            for line in formatted.splitlines():
                lines.append(f"    {line}")
        except Exception:
            lines.append(f"    {value}")

    _add_json_field("warnings", response.warnings)
    _add_json_field("actions_committed", response.actions_committed)
    _add_json_field("actions_pending_confirmation", response.actions_pending_confirmation)

    persistence = response.persistence_instructions or {}
    if persistence:
        _add_json_field("persistence", persistence)
    return lines


def _debug_llm_from_pipeline(pipeline: Any) -> Any | None:
    for component_name in ("query_rewriter", "last_qa_resolver", "classifier"):
        component = getattr(pipeline, component_name, None)
        llm = getattr(component, "llm", None)
        if llm is not None:
            return llm
    return None


def print_runtime_architecture(pipeline: Any, repository: Any) -> None:
    """Print the live objects assembled for the interactive production path."""
    router = getattr(pipeline, "router", None)
    branches = getattr(router, "branches", {}) or getattr(router, "routes", {}) or {}
    branch_lines = [
        f"{getattr(intent, 'value', intent)}={type(branch).__name__}"
        for intent, branch in branches.items()
    ]
    components = (
        "query_rewriter", "last_qa_resolver", "retriever", "context_filter",
        "classifier", "router", "bundler", "platform_selector",
        "post_selector_hitl", "chat_output",
    )
    print("\nLive application architecture (same production factory as Streamlit):")
    print(f"  repository: {type(repository).__name__}")
    for name in components:
        print(f"  {name}: {type(getattr(pipeline, name, None)).__name__}")
    print(f"  branches: {branch_lines or ['unavailable']}")


def configure_debug_logging() -> None:
    """Emit structured, redacted runtime logs to the same terminal as traces."""
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if any(getattr(handler, "_assistant_debug_handler", False) for handler in root.handlers):
        return
    handler = logging.StreamHandler(sys.stderr)
    handler._assistant_debug_handler = True  # type: ignore[attr-defined]
    handler.setFormatter(JsonLogFormatter())
    root.addHandler(handler)


def _debug_llm_lines(llm: Any | None) -> list[str]:
    if llm is None:
        return ["Debug LLM: unavailable"]

    settings = getattr(llm, "settings", None)
    router = getattr(llm, "router", None)
    ollama_client = getattr(llm, "ollama_client", None)
    onnx_client = getattr(llm, "onnx_client", None)
    last_error_by_task = getattr(llm, "last_error_by_task", {}) or {}
    last_errors = {
        getattr(task, "value", str(task)): str(error)
        for task, error in last_error_by_task.items()
    }
    engine = "hybrid_ollama_onnx" if ollama_client is not None and onnx_client is not None else type(llm).__name__
    lines = [
        "Debug LLM:",
        f"  engine: {engine}",
        f"  base_url: {getattr(settings, 'base_url', 'unknown')}",
        f"  keep_alive: {getattr(settings, 'keep_alive', 'unknown')}",
        f"  structured_attempts_for_json: {getattr(settings, 'structured_retry_count', 0) + 1 if settings else 'unknown'}",
    ]
    if onnx_client is not None:
        loaded_models = sorted(getattr(onnx_client, "_models", {}).keys())
        lines.extend(
            [
                f"  onnx_cache_dir: {getattr(onnx_client, 'cache_dir', 'unknown')}",
                f"  onnx_loaded_models: {loaded_models or []}",
            ]
        )
    if router is not None:
        onnx_tasks: list[str] = []
        ollama_tasks: list[str] = []
        for task in LLMTask:
            model = router.model_for_task(task)
            target = onnx_tasks if uses_onnx_runtime(model) else ollama_tasks
            target.append(f"{task.value}={model}")
        lines.extend(
            [
                f"  onnx_routes: {onnx_tasks}",
                f"  ollama_routes: {ollama_tasks}",
            ]
        )
    if not last_errors:
        lines.append("  recent_errors: none")
        return lines

    lines.append("  recent_errors:")
    for task in LLMTask:
        error_text = last_errors.get(task.value)
        if not error_text:
            continue
        payload: dict[str, Any] = {"task": task.value, "last_error": error_text}
        if router is not None:
            decision = router.decision_for_task(task)
            retry_count = getattr(settings, f"json_retry_count_{task.value}", getattr(settings, "structured_retry_count", 0)) if settings else 0
            payload.update(
                {
                    "model": decision.model,
                    "single_call_timeout_seconds": decision.timeout_seconds,
                    "temperature": decision.temperature,
                    "num_ctx": decision.num_ctx,
                    "configured_json_attempts": retry_count + 1,
                    "reason": decision.reason_summary,
                }
            )
        lines.append("    - error payload:")
        try:
            formatted = json.dumps(payload, indent=2, default=str)
            for line in formatted.splitlines():
                lines.append(f"      {line}")
        except Exception:
            lines.append(f"      {payload}")
    return lines


def print_debug_report(response: BundledResponse, llm: Any | None = None) -> None:
    print()
    for line in _debug_trace_lines(response.trace_summary):
        print(line)
    for line in _debug_response_lines(response):
        print(line)
    for line in _debug_llm_lines(llm):
        print(line)


def print_current_debug_trace() -> None:
    trace = current_trace()
    print()
    for line in _debug_trace_lines(trace.summary() if trace else None):
        print(line)


def _debug_runtime_settings(settings: ProductionSettings) -> ProductionSettings:
    """Apply settings.debug to runtime-facing settings for debug entrypoints."""

    debug = settings.debug
    return replace(
        settings,
        database=replace(settings.database, path=debug.db_path),
        retrieval=replace(
            settings.retrieval,
            conversation_min_confidence=debug.conversation_min_confidence,
            knowledge_min_confidence=debug.knowledge_min_confidence,
            max_results=debug.max_results,
            rrf_k=debug.rrf_k,
            lexical_weight=debug.lexical_weight,
            semantic_weight=debug.semantic_weight,
            rerank_candidate_limit=debug.rerank_candidate_limit,
        ),
        worker=replace(
            settings.worker,
            outbox_max_attempts=debug.outbox_max_attempts,
            outbox_batch_size=debug.outbox_batch_size,
            outbox_retry_backoff_seconds=debug.outbox_retry_backoff_seconds,
            outbox_processing_timeout_seconds=debug.outbox_processing_timeout_seconds,
            autoscan_interval_seconds=debug.autoscan_interval_seconds,
        ),
        opensearch=replace(
            settings.opensearch,
            conversation_index=debug.opensearch_conversation_index,
            knowledge_index=debug.opensearch_knowledge_index,
            conversation_write_alias=debug.opensearch_conversation_write_alias,
            knowledge_write_alias=debug.opensearch_knowledge_write_alias,
            reminder_context_alias=debug.opensearch_reminder_context_alias,
        ),
        chroma=replace(
            settings.chroma,
            path=debug.chroma_path,
            conversation_collection=debug.chroma_conversation_collection,
            knowledge_collection=debug.chroma_knowledge_collection,
        ),
    )


class MetadataIntentClassifier:
    def classify(
        self,
        request: ChatRequest,
        rewritten_query: str,
        last_qa_resolution: Any | None = None,
        approved_conversation_context: Any | None = None,
    ) -> Intent:
        return Intent(request.metadata.get("intent", Intent.GENERAL_RESPONSE.value))


class DeterministicQuestionStrategy:
    def generate(self, context: Any, **kwargs: Any) -> GeneratedQuestion:
        expected = context.request.metadata.get(
            "expected_response_type", ExpectedResponseType.FREE_TEXT_ANSWER.value
        )
        try:
            expected_type = ExpectedResponseType(expected)
        except ValueError:
            expected_type = ExpectedResponseType.UNKNOWN
        return GeneratedQuestion(
            text=context.request.metadata.get(
                "clarification_text", "I need one more detail before I can do that."
            ),
            source=QuestionSource.CLARIFICATION_QUESTION,
            purpose="resolve_missing_info",
            confidence=1.0,
            expected_response_type=expected_type,
        )


class OptionalHITLStrategy:
    def evaluate(
        self, context: Any, response_text: str, confidence: float
    ) -> tuple[list[GeneratedQuestion], dict[str, Any] | None]:
        question_text = context.request.metadata.get("supporting_question")
        if not question_text:
            return [], {"triggered": False, "confidence": confidence, "question_count": 0}
        question = GeneratedQuestion(
            text=str(question_text),
            source=QuestionSource.HUMAN_SUPPORTING_QUESTION,
            purpose="optional_context",
            confidence=1.0,
            expected_response_type=ExpectedResponseType.FREE_TEXT_ANSWER,
        )
        return [question], {"triggered": True, "confidence": 1.0, "question_count": 1}


class OptionalReminderSupportingStrategy:
    def generate(self, context: Any, **kwargs: Any) -> GeneratedQuestion | None:
        question_text = context.request.metadata.get("reminder_supporting_question")
        if not question_text:
            return None
        return GeneratedQuestion(
            text=str(question_text),
            source=QuestionSource.REMINDER_SUPPORTING_QUESTION,
            purpose="optional_context",
            confidence=1.0,
            expected_response_type=ExpectedResponseType.REMINDER_FOLLOWUP_ANSWER,
        )


class DeterministicRetriever:
    def __init__(self, repository: SQLiteRepository) -> None:
        self.repository = repository
        self.conversation_calls = 0
        self.knowledge_calls = 0

    def retrieve_conversation(
        self, *, user_id: str, query: str, limit: int, min_confidence: float
    ) -> list[RetrievalResult]:
        self.conversation_calls += 1
        query_norm = _normalize(query)
        rows = self.repository.connection.execute(
            """
            SELECT h.hop_id, h.topic_id, h.user_id, h.raw_user_query,
                    h.raw_response, h.supporting_questions_json, h.entities_json
            FROM conversation_hops h
            WHERE h.user_id = ?
            ORDER BY h.created_at DESC
            """,
            (user_id,),
        ).fetchall()
        results: list[RetrievalResult] = []
        for row in rows:
            text = f"User: {row['raw_user_query']}\nAssistant: {row['raw_response']}"
            score = _token_overlap(query_norm, _normalize(text))
            if score < min_confidence:
                continue
            payload = {
                "hop_id": row["hop_id"],
                "topic_id": row["topic_id"],
                "user_id": row["user_id"],
                "text": text,
                "supporting_questions_json": row["supporting_questions_json"],
                "entities_json": row["entities_json"],
            }
            results.append(
                RetrievalResult(
                    entity_type="conversation_hop",
                    entity_id=row["hop_id"],
                    source_store_evidence={"store": "debug_sql"},
                    rerank_score=score,
                    confidence=score,
                    validation_status="deterministic_debug",
                    payload=payload,
                )
            )
        results.sort(key=lambda item: item.confidence, reverse=True)
        return results[:limit]

    def retrieve_knowledge(
        self, *, user_id: str, query: str, limit: int, min_confidence: float
    ) -> list[RetrievalResult]:
        self.knowledge_calls += 1
        query_norm = _normalize(query)
        rows = self.repository.connection.execute(
            """
            SELECT chunk_id, knowledge_topic_id, user_id, raw_text, summary,
                   is_deleted, version
            FROM knowledge_chunks
            WHERE user_id = ? AND is_deleted = 0
            ORDER BY created_at DESC
            """,
            (user_id,),
        ).fetchall()
        results: list[RetrievalResult] = []
        for row in rows:
            text = str(row["raw_text"])
            score = _token_overlap(query_norm, _normalize(text))
            if score < min_confidence:
                continue
            payload = {
                "chunk_id": row["chunk_id"],
                "topic_id": row["knowledge_topic_id"],
                "user_id": row["user_id"],
                "text": text,
                "summary": row["summary"],
                "is_deleted": bool(row["is_deleted"]),
                "version": row["version"],
            }
            results.append(
                RetrievalResult(
                    entity_type="knowledge_chunk",
                    entity_id=row["chunk_id"],
                    source_store_evidence={"store": "debug_sql"},
                    rerank_score=score,
                    confidence=score,
                    validation_status="deterministic_debug",
                    payload=payload,
                )
            )
        results.sort(key=lambda item: item.confidence, reverse=True)
        return results[:limit]


class ScenarioLLM:
    def chat(self, **kwargs: Any) -> str:
        return "Start with Python basics, practice daily, then build small projects."

    def generate_json(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "thought": "Use the default answer tool for a learning-plan question.",
            "tool_name": "answer_generation",
            "tool_input": {},
            "is_final_answer": True,
        }


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def _token_overlap(query: str, text: str) -> float:
    query_terms = {term for term in query.split() if len(term) > 2}
    text_terms = {term for term in text.split() if len(term) > 2}
    if not query_terms or not text_terms:
        return 0.0
    if query and query in text:
        return 1.0
    return len(query_terms & text_terms) / len(query_terms)


@dataclass
class ScenarioState:
    repository: SQLiteRepository
    pipeline: Any
    retriever: DeterministicRetriever
    user_id: str


@dataclass(frozen=True)
class ScenarioResult:
    name: str
    passed: bool
    details: str


ScenarioFn = Callable[[ProductionSettings], ScenarioResult]


def build_scenario_state(settings: ProductionSettings) -> ScenarioState:
    debug_settings = _debug_runtime_settings(settings)
    config = build_assistant_config(debug_settings)
    repository = SQLiteRepository.in_memory()
    repository.initialize_schema()
    retriever = DeterministicRetriever(repository)
    hard_filter = HardRuleContextFilter(
        knowledge_min_confidence=debug_settings.prompt_policy.context_filter_knowledge_min_confidence,
        allowed_reminder_statuses=debug_settings.prompt_policy.context_filter_allowed_reminder_statuses,
        conversation_min_confidence=debug_settings.context_filter.conversation_min_confidence,
        conversation_approved_max_items=debug_settings.context_filter.conversation_approved_max_items,
        conversation_duplicate_threshold=debug_settings.context_filter.conversation_duplicate_threshold,
        knowledge_approved_max_items=debug_settings.context_filter.knowledge_approved_max_items,
        knowledge_duplicate_threshold=debug_settings.context_filter.knowledge_duplicate_threshold,
        low_information_text_patterns=debug_settings.context_filter.low_information_text_patterns,
        reminder_approved_max_items=debug_settings.context_filter.reminder_approved_max_items,
        reminder_min_confidence=debug_settings.context_filter.reminder_min_confidence,
        low_information_min_chars=debug_settings.context_filter.low_information_min_chars,
    )
    context_filter = TwoLayerContextFilter(hard_rule_filter=hard_filter)
    clarification_strategy = DeterministicQuestionStrategy()
    reminder_resolver = ReminderTargetResolver(config=config)
    knowledge_resolver = KnowledgeTargetResolver(retriever=retriever, config=config)
    validated_builder = ValidatedActionBuilder(
        config=config,
        knowledge_resolver=knowledge_resolver,
        reminder_resolver=reminder_resolver,
    )
    gp_config = GeneralPurposeConfig(
        general_sub_branch_detector_enabled=False,
        content_composer_enabled=False,
        hitl_supporting_question_enabled=False,
    )
    router = BranchRouter(
        {
            Intent.CLARIFICATION: ClarificationBranch(
                prompt_registry=DEFAULT_PROMPT_REGISTRY,
                config=config,
                clarification_strategy=clarification_strategy,
            ),
            Intent.GENERAL_RESPONSE: GeneralResponseBranch(
                retriever=retriever,
                config=config,
                context_filter=context_filter,
                prompt_registry=DEFAULT_PROMPT_REGISTRY,
                hitl_strategy=OptionalHITLStrategy(),
                general_purpose_config=gp_config,
            ),
            Intent.KNOWLEDGE_FACTS: KnowledgeFactsBranch(
                config=config,
                action_detector=None,
                clarification_strategy=clarification_strategy,
                prompt_registry=DEFAULT_PROMPT_REGISTRY,
                validated_action_builder=validated_builder,
                retriever=retriever,
                context_filter=context_filter,
                llm=None,
            ),
            Intent.REMINDER: ReminderBranch(
                config=config,
                action_detector=None,
                clarification_strategy=clarification_strategy,
                reminder_supporting_strategy=OptionalReminderSupportingStrategy(),
                prompt_registry=DEFAULT_PROMPT_REGISTRY,
                validated_action_builder=validated_builder,
                retriever=retriever,
                context_filter=context_filter,
                llm=None,
            ),
        }
    )
    from assistant_rag.pipeline import AssistantPipeline

    pipeline = AssistantPipeline(
        config=config,
        last_qa_store=InMemoryLastQAStore(),
        query_rewriter=QueryRewriter(),
        last_qa_resolver=LastQAResolver(),
        retriever=retriever,
        context_filter=context_filter,
        classifier=MetadataIntentClassifier(),
        router=router,
        bundler=ResponseBundler(DEFAULT_PROMPT_REGISTRY),
        platform_selector=PlatformSelector(),
        chat_output=ChatOutput(),
        prompt_registry=DEFAULT_PROMPT_REGISTRY,
    )
    return ScenarioState(
        repository=repository,
        pipeline=pipeline,
        retriever=retriever,
        user_id=settings.debug.user_id or DEBUG_USER,
    )


def run_request(
    state: ScenarioState,
    query: str,
    *,
    metadata: dict[str, Any] | None = None,
    platform_context: dict[str, Any] | None = None,
) -> BundledResponse:
    return state.pipeline.handle(
        ChatRequest(
            user_id=state.user_id,
            raw_query=query,
            metadata=metadata or {},
            platform_context=platform_context or {},
        ),
        state.repository,
    )


def seed_conversation(
    state: ScenarioState,
    *,
    title: str,
    query: str,
    response: str,
    supporting_questions: list[dict[str, Any]] | None = None,
) -> tuple[str, str]:
    with state.repository.transaction() as cursor:
        topic_id = state.repository.ensure_topic(
            cursor, user_id=state.user_id, title=title
        )
        hop = state.repository.append_conversation_hop(
            cursor,
            topic_id=topic_id,
            user_id=state.user_id,
            intent=Intent.GENERAL_RESPONSE.value,
            raw_user_query=query,
            rewritten_user_query=query,
            raw_response=response,
            response_type=ResponseType.NORMAL.value,
            supporting_questions=supporting_questions or [],
        )
    return topic_id, hop.hop_id


def seed_knowledge(state: ScenarioState, *, title: str, text: str) -> str:
    with state.repository.transaction() as cursor:
        _, chunk_id, _ = state.repository.add_knowledge_chunk(
            cursor, user_id=state.user_id, title=title, text=text
        )
    return chunk_id


def seed_reminder(
    state: ScenarioState,
    *,
    subject: str,
    summary: str,
    reminder_time: datetime,
    status: str = "scheduled",
) -> str:
    topic_id, hop_id = seed_conversation(
        state,
        title="Seeded Reminders",
        query=f"Create reminder for {subject}",
        response=f"Reminder seed for {subject}",
    )
    with state.repository.transaction() as cursor:
        reminder_id = state.repository.add_reminder(
            cursor,
            user_id=state.user_id,
            source_topic_id=topic_id,
            source_hop_id=hop_id,
            reminder_time=reminder_time.isoformat(),
            raw_reminder=summary,
            reminder_summary=summary,
            subject=subject,
        )
        # Scenario fixtures represent reminders whose background timing pass
        # already completed. Real add/modify/turn-on paths deliberately reset
        # this state and are exercised separately.
        cursor.execute(
            """
            UPDATE reminders
            SET event_time = reminder_time, timing_plan_status = 'planned',
                timing_planned_at = ?, timing_plan_reason = 'scenario fixture'
            WHERE reminder_id = ?
            """,
            (now_iso(), reminder_id),
        )
        if status != "scheduled":
            state.repository.update_reminder_status(
                cursor,
                user_id=state.user_id,
                reminder_id=reminder_id,
                status=status,
            )
    return reminder_id


def assert_response(
    response: BundledResponse,
    expected_type: ResponseType,
    expected_text: str | None = None,
) -> None:
    if response.response_type != expected_type:
        raise AssertionError(
            f"expected {expected_type.value}, got {response.response_type.value}: {response.final_chat_text}"
        )
    if expected_text and expected_text.casefold() not in response.final_chat_text.casefold():
        raise AssertionError(
            f"expected text containing {expected_text!r}, got {response.final_chat_text!r}"
        )


def scenario_clarification_direct(settings: ProductionSettings) -> ScenarioResult:
    state = build_scenario_state(settings)
    response = run_request(
        state,
        "Need to update it",
        metadata={
            "intent": Intent.CLARIFICATION.value,
            "clarification_text": "Which item should I update?",
        },
    )
    assert_response(response, ResponseType.CLARIFICATION, "Which item")
    if "Clarification question:" not in response.final_chat_text:
        raise AssertionError(f"clarification question was not explicitly printed: {response.final_chat_text!r}")
    return ScenarioResult("clarification_direct", True, "clarification branch returned a question")


def scenario_general_new_conversation(settings: ProductionSettings) -> ScenarioResult:
    state = build_scenario_state(settings)
    response = run_request(
        state,
        "Explain the release checklist",
        metadata={
            "intent": Intent.GENERAL_RESPONSE.value,
            "normal_response_text": "Use the release checklist in order.",
        },
    )
    assert_response(response, ResponseType.NORMAL, "release checklist")
    if state.repository.table_count("conversation_hops") != 1:
        raise AssertionError("general response did not persist a conversation hop")
    return ScenarioResult("general_new_conversation", True, "new conversation persisted hop and outbox job")


def scenario_general_hitl_supporting_question_printed(settings: ProductionSettings) -> ScenarioResult:
    state = build_scenario_state(settings)
    response = run_request(
        state,
        "Explain the migration plan",
        metadata={
            "intent": Intent.GENERAL_RESPONSE.value,
            "normal_response_text": "Start with schema compatibility, then migrate traffic gradually.",
            "supporting_question": "Which service should I prioritize next?",
        },
    )
    assert_response(response, ResponseType.NORMAL, "schema compatibility")
    if "Supporting question: Which service should I prioritize next?" not in response.final_chat_text:
        raise AssertionError(f"HITL supporting question was not printed: {response.final_chat_text!r}")
    return ScenarioResult(
        "general_hitl_supporting_question_printed",
        True,
        "HITL supporting question is visible in final chat text",
    )


def scenario_general_new_conversation_skips_sub_branch_llm(settings: ProductionSettings) -> ScenarioResult:
    state = build_scenario_state(settings)

    class FailingSubBranchDetector:
        def detect(self, *_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("sub-branch detector should not run for a context-free new conversation")

    branch = state.pipeline.router.branches[Intent.GENERAL_RESPONSE]
    branch.sub_branch_detector = FailingSubBranchDetector()
    branch.general_purpose_config = GeneralPurposeConfig(
        general_sub_branch_detector_enabled=True,
        content_composer_enabled=False,
        hitl_supporting_question_enabled=False,
    )
    response = run_request(
        state,
        "Explain binary search with numbers",
        metadata={
            "intent": Intent.GENERAL_RESPONSE.value,
            "normal_response_text": "Binary search repeatedly halves a sorted list.",
        },
    )
    assert_response(response, ResponseType.NORMAL, "halves")
    return ScenarioResult(
        "general_new_conversation_skips_sub_branch_llm",
        True,
        "context-free general responses choose new topic without sub-branch LLM",
    )


def scenario_answer_adaptive_token_budget(settings: ProductionSettings) -> ScenarioResult:
    simple_prompt = 'Runtime context:\n{"stage":"answer_generation","raw_query":"Explain binary search with numbers","rewritten_query":"Explain binary search with numbers","extra":{"approved_conversation_history":[],"approved_knowledge_evidence":[],"approved_reminder_context":[]}}'
    complex_prompt = 'Runtime context:\n{"stage":"answer_generation","raw_query":"Deeply determine a latency optimization architecture and implementation plan","rewritten_query":"Deeply determine a latency optimization architecture and implementation plan","extra":{"approved_conversation_history":[],"approved_knowledge_evidence":[],"approved_reminder_context":[]}}'
    long_prompt = 'Runtime context:\n{"stage":"answer_generation","raw_query":"Write a comprehensive full specification for the latency optimization","rewritten_query":"Write a comprehensive full specification for the latency optimization","extra":{"approved_conversation_history":[],"approved_knowledge_evidence":[],"approved_reminder_context":[]}}'
    approved_prompt = 'Runtime context:\n{"stage":"answer_generation","raw_query":"Summarize this prior context","rewritten_query":"Summarize this prior context","extra":{"approved_conversation_history":[{"text":"Previous approved context"}],"approved_knowledge_evidence":[],"approved_reminder_context":[]}}'
    if _adaptive_max_new_tokens(LLMTask.ANSWER, simple_prompt, 1024) != 320:
        raise AssertionError("simple teaching answer should use a compact token budget")
    if _adaptive_max_new_tokens(LLMTask.ANSWER, complex_prompt, 1024) != 640:
        raise AssertionError("complex architecture answer should use the bounded complex budget")
    if _adaptive_max_new_tokens(LLMTask.ANSWER, long_prompt, 1024) != 896:
        raise AssertionError("explicit long-form requests should use the high bounded budget")
    if _adaptive_max_new_tokens(LLMTask.ANSWER, approved_prompt, 1024) != 512:
        raise AssertionError("approved-context answers should use the context budget")
    if _adaptive_max_new_tokens(LLMTask.WRITING, simple_prompt, 1536) != 1536:
        raise AssertionError("adaptive answer budget must not affect non-answer tasks")
    if _sentence_boundary_stop_reason("This is a complete enough answer.", 220, 320) is None:
        raise AssertionError("complete simple answers should be eligible for sentence-boundary stop")
    if _sentence_boundary_stop_reason("Too short.", 60, 320) is not None:
        raise AssertionError("short answers should not stop before the budget floor")
    return ScenarioResult("answer_adaptive_token_budget", True, "answer budgets are tiered and can stop at complete sentences")


def scenario_structured_clarification_fallback_policy(settings: ProductionSettings) -> ScenarioResult:
    default_settings = ProductionSettings()
    if default_settings.ollama.num_predict_generate_clarification < 160:
        raise AssertionError("clarification JSON generation needs enough token budget for all required fields")
    if default_settings.ollama.num_predict_generate_human_supporting < 160:
        raise AssertionError("human supporting-question JSON generation needs enough token budget")
    if default_settings.ollama.num_predict_generate_reminder_supporting < 160:
        raise AssertionError("reminder supporting-question JSON generation needs enough token budget")
    if default_settings.ollama.temperature_generate_clarification != 0.0:
        raise AssertionError("clarification JSON generation must use deterministic sampling")
    if default_settings.ollama.model_generate_clarification_fallback != "qwen3.5:4b":
        raise AssertionError("clarification generation needs an Ollama recovery model")

    schema = {
        "type": "object",
        "properties": {
            "question_text": {"type": "string"},
            "question_source": {"type": "string"},
            "purpose": {"type": "string"},
            "confidence": {"type": "number"},
            "should_ask": {"type": "boolean"},
            "expected_response_type": {
                "type": "string",
                "enum": [e.value for e in ExpectedResponseType],
            },
            "reason_summary": {"type": "string"},
        },
        "required": [
            "question_text",
            "question_source",
            "purpose",
            "confidence",
            "should_ask",
            "expected_response_type",
            "reason_summary",
        ],
    }
    payload = structured_fallback_payload(
        task=LLMTask.GENERATE_CLARIFICATION,
        schema=schema,
        user_prompt='Runtime context:\n{"raw_query":"Need to update it"}',
        error=ValueError("synthetic malformed JSON"),
    )
    validate_json_schema(payload, schema)
    if payload["question_text"] != DEFAULT_PROMPT_REGISTRY.message("fallback_message"):
        raise AssertionError("clarification fallback text must come from PromptRegistry")
    return ScenarioResult(
        "structured_clarification_fallback_policy",
        True,
        "clarification structured fallback is schema-valid, registry-backed, and budgeted",
    )


def scenario_mutation_clarification_fast_path(settings: ProductionSettings) -> ScenarioResult:
    """Predictable mutation gaps must not spend latency on planning or question LLMs."""
    state = build_scenario_state(settings)

    class LowConfidenceDetector:
        def __init__(self, missing_fields: list[str]) -> None:
            self.missing_fields = missing_fields

        def detect(self, *_: Any, **__: Any) -> Any:
            return type(
                "Detection",
                (),
                {"requires_clarification": True, "missing_fields": self.missing_fields, "metadata": {}},
            )()

    class MustNotRun:
        def chat(self, **_: Any) -> str:
            raise AssertionError("unused action planning LLM was invoked")

        def generate(self, *_: Any, **__: Any) -> GeneratedQuestion:
            raise AssertionError("predictable mutation clarification invoked question generation")

    knowledge_branch = KnowledgeFactsBranch(
        config=state.pipeline.config,
        action_detector=LowConfidenceDetector(["low_confidence_action_detection"]),
        clarification_strategy=MustNotRun(),
        prompt_registry=DEFAULT_PROMPT_REGISTRY,
        llm=MustNotRun(),
    )
    knowledge_result = knowledge_branch.execute(
        PipelineContext(
            request=ChatRequest(user_id=state.user_id, raw_query="Remember something important."),
            rewritten_query="Remember something important.",
            last_qa_state=None,
            conversation_results=[],
            intent=Intent.KNOWLEDGE_FACTS,
        ),
        state.repository,
    )
    reminder_branch = ReminderBranch(
        config=state.pipeline.config,
        action_detector=LowConfidenceDetector(["reminder_time"]),
        clarification_strategy=MustNotRun(),
        prompt_registry=DEFAULT_PROMPT_REGISTRY,
        llm=MustNotRun(),
    )
    reminder_result = reminder_branch.execute(
        PipelineContext(
            request=ChatRequest(user_id=state.user_id, raw_query="Remind me to call the bank."),
            rewritten_query="Remind me to call the bank.",
            last_qa_state=None,
            conversation_results=[],
            intent=Intent.REMINDER,
        ),
        state.repository,
    )
    if knowledge_result.clarification_question is None or knowledge_result.clarification_question.text != DEFAULT_PROMPT_REGISTRY.message("knowledge_missing_action"):
        raise AssertionError(f"knowledge fast clarification was not deterministic: {knowledge_result}")
    if reminder_result.clarification_question is None or reminder_result.clarification_question.text != DEFAULT_PROMPT_REGISTRY.message("reminder_missing_time"):
        raise AssertionError(f"reminder fast clarification did not request time: {reminder_result}")
    return ScenarioResult("mutation_clarification_fast_path", True, "predictable mutation gaps skip unused action planning and LLM question generation")


def scenario_state_mutation_preflight_bypass(settings: ProductionSettings) -> ScenarioResult:
    """Current state mutations must not consult stale Last-QA or broad conversation retrieval."""
    class MustNotResolveLastQA:
        def resolve(self, *_: Any, **__: Any) -> Any:
            raise AssertionError("state mutation invoked Last-QA resolution")

    state = build_scenario_state(settings)
    state.pipeline.last_qa_resolver = MustNotResolveLastQA()
    response = run_request(
        state,
        "Store the current project retention policy.",
        metadata={
            "intent": Intent.KNOWLEDGE_FACTS.value,
            "knowledge_actions": [{"action": "add", "text": "The project retention policy is current."}],
        },
    )
    assert_response(response, ResponseType.KNOWLEDGE_ACTION, "Added new knowledge")
    if state.retriever.conversation_calls:
        raise AssertionError("state mutation invoked broad conversation retrieval")
    if response.last_qa_state.response_type is not ResponseType.KNOWLEDGE_ACTION:
        raise AssertionError("state mutation did not preserve its knowledge branch")
    return ScenarioResult("state_mutation_preflight_bypass", True, "state mutations bypass stale Last-QA and conversation retrieval")


def scenario_model_backed_action_extraction(settings: ProductionSettings) -> ScenarioResult:
    """Mutation content must come from the configured semantic extractor, not lexical shortcuts."""
    class ScriptedActionLLM:
        def __init__(self) -> None:
            self.calls = 0

        def generate_json(self, **_: Any) -> dict[str, Any]:
            self.calls += 1
            return {
                "intent": Intent.KNOWLEDGE_FACTS.value,
                "confidence": 0.96,
                "knowledge_actions": [
                    {
                        "action": "add",
                        "text": "The weekly release note should include operational risks.",
                        "confidence": 0.96,
                    }
                ],
                "reminder_actions": [],
                "missing_fields": [],
                "risk_flags": [],
                "normalized_entities": {},
            }

    llm = ScriptedActionLLM()
    detector = LLMActionDetector(llm)
    result = detector.detect(
        ChatRequest(
            user_id=DEBUG_USER,
            raw_query="Keep this as a durable project fact: the weekly release note should include operational risks.",
        ),
        "Keep this as a durable project fact: the weekly release note should include operational risks.",
        Intent.KNOWLEDGE_FACTS,
    )
    actions = result.metadata.get("knowledge_actions") or []
    if llm.calls != 1:
        raise AssertionError(f"action extraction bypassed its configured semantic model: {llm.calls} calls")
    if len(actions) != 1 or actions[0].get("text") != "The weekly release note should include operational risks.":
        raise AssertionError(f"action extraction did not preserve the model-extracted fact: {actions}")
    if result.requires_clarification:
        raise AssertionError(f"complete model action was incorrectly sent to clarification: {result}")
    required_rule = "The selected branch is authoritative; never reclassify it."
    if required_rule not in DEFAULT_PROMPT_REGISTRY.system("action_detection"):
        raise AssertionError("action-detection prompt lost its selected-branch rule")
    schema = detector._schema(Intent.KNOWLEDGE_FACTS)
    if "intent" in schema["properties"] or "reminder_actions" in schema["properties"]:
        raise AssertionError("knowledge extraction schema must contain only the selected branch contract")
    malformed_action = {
        "confidence": 0.96,
        "knowledge_actions": [{"action": "persist"}],
        "missing_fields": [],
        "risk_flags": [],
        "normalized_entities": {},
    }
    try:
        validate_json_schema(malformed_action, schema)
    except ValueError:
        pass
    else:
        raise AssertionError("nested unsupported action values must fail structured validation")
    return ScenarioResult("model_backed_action_extraction", True, "complete knowledge content is extracted by the configured semantic model")


def scenario_knowledge_action_recovery(settings: ProductionSettings) -> ScenarioResult:
    """A failed first extraction may recover only through the same semantic action contract."""
    class RecoveryLLM:
        def __init__(self) -> None:
            self.calls = 0

        def generate_json(self, **_: Any) -> dict[str, Any]:
            self.calls += 1
            if self.calls == 1:
                return {
                    "confidence": 0.0,
                    "knowledge_actions": [],
                    "missing_fields": ["target_description"],
                    "risk_flags": ["ambiguous"],
                }
            if self.calls == 2:
                return {"operation": "add"}
            return {"text": "Archived records use a reversible lifecycle."}

    llm = RecoveryLLM()
    result = LLMActionDetector(llm).detect(
        ChatRequest(user_id=DEBUG_USER, raw_query="Store the archival lifecycle policy."),
        "Store the archival lifecycle policy.",
        Intent.KNOWLEDGE_FACTS,
    )
    actions = result.metadata.get("knowledge_actions") or []
    if llm.calls != 3 or result.requires_clarification or actions != [{"action": "add", "text": "Archived records use a reversible lifecycle."}]:
        raise AssertionError(f"knowledge extraction recovery failed: calls={llm.calls}, result={result}")
    if "outer storage operation" not in DEFAULT_PROMPT_REGISTRY.system("knowledge_operation_recovery"):
        raise AssertionError("knowledge recovery prompt lost its outer-operation rule")
    return ScenarioResult("knowledge_action_recovery", True, "failed knowledge extraction recovers through a bounded semantic retry")


def scenario_sql_backed_knowledge_lookup(settings: ProductionSettings) -> ScenarioResult:
    """Explicit lookups must see active user-owned SQL facts before derived indexes catch up."""
    class LookupDetector:
        def detect(self, *_: Any, **__: Any) -> Any:
            return type(
                "LookupDetection",
                (),
                {
                    "metadata": {"knowledge_lookup": True},
                    "requires_clarification": False,
                    "missing_fields": [],
                },
            )()

    state = build_scenario_state(settings)
    fact = "The stored infrastructure locality preference is Tokyo."
    seed_knowledge(state, title="Infrastructure preference", text=fact)
    state.pipeline.router.branches[Intent.KNOWLEDGE_FACTS].action_detector = LookupDetector()
    response = run_request(
        state,
        "What is the stored infrastructure locality preference?",
        metadata={"intent": Intent.KNOWLEDGE_FACTS.value},
    )
    if response.response_type is not ResponseType.NORMAL or response.final_chat_text != fact:
        raise AssertionError(f"SQL-backed lookup did not return the exact stored fact: {response}")
    if state.retriever.knowledge_calls:
        raise AssertionError("explicit SQL-backed lookup incorrectly required derived knowledge retrieval")
    return ScenarioResult("sql_backed_knowledge_lookup", True, "active user-owned SQL knowledge is returned exactly before index catch-up")


def scenario_general_sql_knowledge_fallback(settings: ProductionSettings) -> ScenarioResult:
    """General factual questions must fall back to SQL when derived knowledge retrieval is empty."""
    state = build_scenario_state(settings)
    fact = "The deployment preference for the assistant backend is Tokyo."
    seed_knowledge(state, title="Deployment preference", text=fact)
    state.retriever.retrieve_knowledge = lambda **_: []  # type: ignore[method-assign]
    response = run_request(
        state,
        "What is the deployment preference for the assistant backend?",
    )
    if response.response_type is not ResponseType.NORMAL or response.final_chat_text != fact:
        raise AssertionError(f"general SQL fallback did not return the exact stored fact: {response}")
    return ScenarioResult("general_sql_knowledge_fallback", True, "general factual lookup falls back to active user-owned SQL knowledge")


def scenario_model_backed_knowledge_update(settings: ProductionSettings) -> ScenarioResult:
    """A complete update instruction must retain both its target and replacement."""
    class ScriptedUpdateLLM:
        def __init__(self) -> None:
            self.calls = 0

        def generate_json(self, **_: Any) -> dict[str, Any]:
            self.calls += 1
            return {
                "confidence": 0.97,
                "knowledge_actions": [
                    {
                        "action": "modify",
                        "target_description": "primary data store",
                        "replacement_text": "The primary data store is the production source of truth.",
                    }
                ],
                "missing_fields": [],
                "risk_flags": [],
            }

    llm = ScriptedUpdateLLM()
    detector = LLMActionDetector(llm)
    request = ChatRequest(user_id=DEBUG_USER, raw_query="Update the primary data-store policy to the new source-of-truth statement.")
    result = detector.detect(request, request.raw_query, Intent.KNOWLEDGE_FACTS)
    actions = result.metadata.get("knowledge_actions") or []
    if llm.calls != 1 or result.requires_clarification:
        raise AssertionError(f"complete update was not accepted by action extraction: {result}")
    expected = {
        "action": "modify",
        "target_description": "primary data store",
        "replacement_text": "The primary data store is the production source of truth.",
    }
    if len(actions) != 1 or any(actions[0].get(key) != value for key, value in expected.items()):
        raise AssertionError(f"update extraction lost target or replacement: {actions}")

    state = build_scenario_state(settings)
    seed_knowledge(state, title="Data policy", text="The primary data store is under review.")
    response = run_request(
        state,
        request.raw_query,
        metadata={"intent": Intent.KNOWLEDGE_FACTS.value, "knowledge_actions": actions},
    )
    assert_response(response, ResponseType.KNOWLEDGE_ACTION, "confirm")
    if not response.actions_pending_confirmation:
        raise AssertionError("validated knowledge update must require confirmation before writing")
    return ScenarioResult("model_backed_knowledge_update", True, "complete updates preserve target and replacement, then require confirmation")


def scenario_clarification_schema_echo_recovery(settings: ProductionSettings) -> ScenarioResult:
    """Question generation must request data instances, never induce schema copying."""
    schema = {
        "type": "object",
        "properties": {
            "question_text": {"type": "string"},
            "question_source": {"type": "string"},
            "purpose": {"type": "string"},
            "confidence": {"type": "number"},
            "should_ask": {"type": "boolean"},
            "expected_response_type": {"type": "string", "enum": [e.value for e in ExpectedResponseType]},
            "reason_summary": {"type": "string"},
        },
        "required": [
            "question_text",
            "question_source",
            "purpose",
            "confidence",
            "should_ask",
            "expected_response_type",
            "reason_summary",
        ],
    }
    system_prompt = DEFAULT_PROMPT_REGISTRY.system("question_generation")
    if "Never return, describe, or copy a JSON Schema." not in system_prompt:
        raise AssertionError("question-generation prompt does not forbid schema echoes")
    repair_prompt = _structured_attempt_prompt(
        user_prompt='Runtime context:\n{"stage":"generate_clarification"}',
        schema=schema,
        mode="schema",
    )
    if "not a JSON Schema" not in repair_prompt or '"properties"' in repair_prompt:
        raise AssertionError(f"schema retry prompt is still likely to trigger a schema echo: {repair_prompt}")

    default_settings = ProductionSettings()
    onnx_settings = replace(
        default_settings.ollama,
        model_generate_clarification="microsoft/Phi-4-mini-instruct-onnx",
    )
    router = OllamaModelRouter(onnx_settings)
    onnx_client = ONNXLLMClient(router)
    prompts: list[str] = []
    valid_payload_text = '{"question_text":"Which preference should I update?","question_source":"clarification_question","purpose":"resolve_missing_info","confidence":0.95,"should_ask":true,"expected_response_type":"free_text_answer","reason_summary":"missing target"}'
    responses = [
        '{"type":"object","properties":{}}',
        valid_payload_text,
    ]

    def scripted_generate(task: LLMTask, system_prompt: str, user_prompt: str, decision: Any) -> str:
        prompts.append(user_prompt)
        return responses.pop(0)

    onnx_client._generate = scripted_generate  # type: ignore[method-assign]
    payload = onnx_client.generate_json(
        task=LLMTask.GENERATE_CLARIFICATION,
        system_prompt=system_prompt,
        user_prompt='Runtime context:\n{"stage":"generate_clarification"}',
        schema=schema,
    )
    if payload["question_text"] != "Which preference should I update?" or len(prompts) != 2:
        raise AssertionError(f"ONNX correction retry did not recover the clarification data: {payload}, prompts={len(prompts)}")
    if any('"properties"' in prompt for prompt in prompts):
        raise AssertionError("ONNX clarification retry received the full JSON Schema")

    ollama_client = OllamaLLMClient(onnx_settings, router)
    fallback_onnx = ONNXLLMClient(router)
    hybrid = HybridLLMClient(ollama_client, fallback_onnx)
    def failed_onnx(**_: Any) -> dict[str, Any]:
        fallback_onnx.last_error_by_task[LLMTask.GENERATE_CLARIFICATION] = "structured_fallback_used_after_2_failures: schema echo"
        return structured_fallback_payload(
            task=LLMTask.GENERATE_CLARIFICATION,
            schema=schema,
            user_prompt='Runtime context:\n{}',
            error=ValueError("schema echo"),
        )

    fallback_onnx.generate_json = failed_onnx  # type: ignore[method-assign]
    fallback_calls: list[dict[str, Any]] = []

    def successful_ollama(**kwargs: Any) -> str:
        fallback_calls.append(kwargs)
        return valid_payload_text

    ollama_client._chat_raw = successful_ollama  # type: ignore[method-assign]
    recovered = hybrid.generate_json(
        task=LLMTask.GENERATE_CLARIFICATION,
        system_prompt=system_prompt,
        user_prompt='Runtime context:\n{}',
        schema=schema,
    )
    if recovered["question_text"] != "Which preference should I update?":
        raise AssertionError(f"clarification recovery returned fallback data: {recovered}")
    if not fallback_calls or fallback_calls[0].get("model_override") != onnx_settings.model_generate_clarification_fallback:
        raise AssertionError(f"clarification recovery used the wrong fallback model: {fallback_calls}")
    if LLMTask.GENERATE_CLARIFICATION in hybrid.last_error_by_task:
        raise AssertionError(f"recovered clarification error leaked as active: {hybrid.last_error_by_task}")
    return ScenarioResult("clarification_schema_echo_recovery", True, "clarification retries use data instances and recover through Ollama when ONNX cannot")


def scenario_content_composer_react_structured_policy(settings: ProductionSettings) -> ScenarioResult:
    default_settings = ProductionSettings()
    if default_settings.ollama.num_predict_content_composer_react < 160:
        raise AssertionError("content composer ReAct JSON needs enough token budget for all required fields")

    thought_rules = CONTENT_COMPOSER_REACT_SCHEMA["properties"]["thought"]
    if thought_rules.get("maxLength") != 120:
        raise AssertionError("content composer ReAct thought should stay short enough for small structured models")

    prompt = DEFAULT_PROMPT_REGISTRY.template("content_composer_react")
    if "no more than 12 words" not in " ".join(prompt.decision_rules):
        raise AssertionError("content composer ReAct prompt must cap thought length")

    payload = structured_fallback_payload(
        task=LLMTask.CONTENT_COMPOSER_REACT,
        schema=CONTENT_COMPOSER_REACT_SCHEMA,
        user_prompt='Runtime context:\n{"rewritten_query":"Create an Excel comparison for dog cat duck fur color"}',
        error=ValueError("synthetic malformed JSON"),
    )
    validate_json_schema(payload, CONTENT_COMPOSER_REACT_SCHEMA)
    if payload["tool_name"] != "answer_generation" or not payload["is_final_answer"]:
        raise AssertionError("content composer fallback must finish through answer_generation")
    return ScenarioResult(
        "content_composer_react_structured_policy",
        True,
        "content composer ReAct schema is concise, budgeted, and fallback-safe",
    )


def scenario_structured_fallback_terminal_quiet(settings: ProductionSettings) -> ScenarioResult:
    class BadStructuredLLM(OllamaLLMClient):
        def _chat_raw(self, **_kwargs: Any) -> str:
            return "not json"

    logger = logging.getLogger("assistant_rag.llm")
    records: list[logging.LogRecord] = []

    class CaptureHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = CaptureHandler()
    previous_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        client = BadStructuredLLM(ProductionSettings().ollama, OllamaModelRouter(ProductionSettings().ollama))
        payload = client.generate_json(
            task=LLMTask.CONTENT_COMPOSER_REACT,
            system_prompt="Return JSON.",
            user_prompt='Runtime context:\n{"rewritten_query":"Create an Excel comparison"}',
            schema=CONTENT_COMPOSER_REACT_SCHEMA,
        )
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)

    validate_json_schema(payload, CONTENT_COMPOSER_REACT_SCHEMA)
    terminal_noise = [
        record.getMessage()
        for record in records
        if record.levelno >= logging.WARNING or "Severe failure" in record.getMessage()
    ]
    if terminal_noise:
        raise AssertionError(f"structured fallback emitted terminal-level noise: {terminal_noise}")
    return ScenarioResult(
        "structured_fallback_terminal_quiet",
        True,
        "validated structured fallbacks stay out of warning/error terminal logs",
    )


def scenario_model_routing_policy(settings: ProductionSettings) -> ScenarioResult:
    default_settings = ProductionSettings()
    router = OllamaModelRouter(default_settings.ollama)
    expected_models = {
        LLMTask.QUERY_REWRITE: "qwen3.5:4b",
        LLMTask.LAST_QA: "qwen3.5:4b",
        LLMTask.INTENT: "qwen3.5:4b",
        LLMTask.ACTION_EXTRACTION: "qwen3.5:4b",
        LLMTask.GENERATE_CLARIFICATION: "qwen3.5:4b",
        LLMTask.GENERATE_HUMAN_SUPPORTING: "qwen3.5:4b",
        LLMTask.GENERATE_REMINDER_SUPPORTING: "qwen3.5:4b",
        LLMTask.CLARIFICATION_MERGE: "qwen3.5:4b",
        LLMTask.ANSWER: "microsoft/Phi-4-mini-instruct-onnx",
        LLMTask.WRITING: "microsoft/Phi-4-mini-instruct-onnx",
        LLMTask.RISKY_ACTION: "qwen3.5:4b",
        LLMTask.RETRIEVAL_VALIDATION: "microsoft/Phi-4-mini-instruct-onnx",
        LLMTask.GENERAL_SUB_BRANCH_DETECTION: "qwen3.5:4b",
        LLMTask.CONTENT_COMPOSER_REACT: "qwen3.5:4b",
        LLMTask.ACTION_PLANNING: "microsoft/Phi-4-mini-instruct-onnx",
    }
    actual_models = {task: router.model_for_task(task) for task in expected_models}
    mismatches = {
        task.value: {"expected": expected_model, "actual": actual_models[task]}
        for task, expected_model in expected_models.items()
        if actual_models[task] != expected_model
    }
    if mismatches:
        raise AssertionError(f"model routing mismatches: {mismatches}")

    # Check some basic policy settings
    policy_values = {
        "bm25_top_k": default_settings.retrieval.bm25_top_k,
        "chroma_top_k": default_settings.retrieval.chroma_top_k,
        "rrf_k": default_settings.retrieval.rrf_k,
        "reranker_top_k": default_settings.retrieval.rerank_candidate_limit,
        "reranker_min_score": default_settings.reranker.min_score,
        "reranker_batch_size": default_settings.reranker.batch_size,
        "final_context_top_k": default_settings.retrieval.max_results,
        "retrieval_min_confidence": default_settings.retrieval.min_confidence,
        "knowledge_context_min_confidence": default_settings.retrieval.knowledge_min_confidence,
        "conversation_context_min_confidence": default_settings.retrieval.conversation_min_confidence,
        "last_qa_min_confidence": default_settings.prompt_policy.last_qa_min_confidence,
        "last_qa_clarification_merge_min_confidence": default_settings.prompt_policy.clarification_merge_min_confidence,
        "last_qa_skip_broad_retrieval_min_confidence": default_settings.prompt_policy.last_qa_skip_broad_retrieval_min_confidence,
        "action_min_confidence": default_settings.prompt_policy.action_min_confidence,
        "risky_action_confidence_threshold": default_settings.prompt_policy.risky_action_confidence_threshold,
        "reminder_candidate_limit": default_settings.reminder_resolver.reminder_target_candidate_limit,
        "reminder_target_min_score": default_settings.reminder_resolver.reminder_target_relevance_threshold,
        "reminder_target_ambiguity_margin": default_settings.reminder_resolver.reminder_target_ambiguity_margin,
        "reminder_fuzzy_match_threshold": default_settings.reminder_resolver.reminder_fuzzy_match_threshold,
        "reminder_context_min_confidence": default_settings.context_filter.reminder_min_confidence,
        "knowledge_chunk_size_tokens": default_settings.knowledge_chunks.chunk_size_tokens,
        "knowledge_chunk_overlap_tokens": default_settings.knowledge_chunks.chunk_overlap_tokens,
        "knowledge_min_chunk_tokens": default_settings.knowledge_chunks.min_chunk_tokens,
        "knowledge_max_chunk_tokens": default_settings.knowledge_chunks.max_chunk_tokens,
        "embedding_batch_size": default_settings.embeddings.batch_size,
        "embedding_max_length": default_settings.embeddings.max_length,
        "outbox_batch_size": default_settings.worker.outbox_batch_size,
        "outbox_max_retries": default_settings.worker.outbox_max_attempts,
        "outbox_retry_backoff_seconds": default_settings.worker.outbox_retry_backoff_seconds,
        "outbox_stale_processing_after_seconds": default_settings.worker.outbox_processing_timeout_seconds,
        "outbox_worker_interval_seconds": default_settings.worker.outbox_worker_interval_seconds,
        "reminder_autoscan_interval_seconds": default_settings.worker.autoscan_interval_seconds,
    }
    expected_policy_values = {
        "bm25_top_k": 24,
        "chroma_top_k": 24,
        "rrf_k": 40,
        "reranker_top_k": 16,
        "reranker_min_score": 0.30,
        "reranker_batch_size": 24,
        "final_context_top_k": 6,
        "retrieval_min_confidence": 0.30,
        "knowledge_context_min_confidence": 0.38,
        "conversation_context_min_confidence": 0.42,
        "last_qa_min_confidence": 0.80,
        "last_qa_clarification_merge_min_confidence": 0.84,
        "last_qa_skip_broad_retrieval_min_confidence": 0.90,
        "action_min_confidence": 0.76,
        "risky_action_confidence_threshold": 0.90,
        "reminder_candidate_limit": 12,
        "reminder_target_min_score": 0.78,
        "reminder_target_ambiguity_margin": 0.08,
        "reminder_fuzzy_match_threshold": 0.82,
        "reminder_context_min_confidence": 0.58,
        "knowledge_chunk_size_tokens": 560,
        "knowledge_chunk_overlap_tokens": 80,
        "knowledge_min_chunk_tokens": 80,
        "knowledge_max_chunk_tokens": 800,
        "embedding_batch_size": 48,
        "embedding_max_length": 4096,
        "outbox_batch_size": 64,
        "outbox_max_retries": 4,
        "outbox_retry_backoff_seconds": 15,
        "outbox_stale_processing_after_seconds": 180,
        "outbox_worker_interval_seconds": 3,
        "reminder_autoscan_interval_seconds": 30,
    }
    policy_mismatches = {
        key: {"expected": expected_policy_values[key], "actual": policy_values[key]}
        for key in expected_policy_values
        if policy_values[key] != expected_policy_values[key]
    }
    if policy_mismatches:
        raise AssertionError(f"policy default mismatches: {policy_mismatches}")
    if default_settings.prompt_policy.risky_action_operations != ("delete", "turn_off"):
        raise AssertionError(f"risky action operations mismatch: {default_settings.prompt_policy.risky_action_operations}")
    if not default_settings.retrieval_validation.reminder_llm_validation_enabled:
        raise AssertionError("reminder LLM validation should be enabled by default")
    if not default_settings.retrieval_validation.knowledge_llm_validation_enabled:
        raise AssertionError("knowledge LLM validation should be enabled by default")
    if not default_settings.embeddings.normalize_embeddings:
        raise AssertionError("embedding normalization should be enabled")
    if default_settings.embeddings.model_name != "BAAI/bge-m3":
        raise AssertionError(f"embedding model mismatch: {default_settings.embeddings.model_name}")
    if router.decision_for_task(LLMTask.ANSWER).num_ctx != default_settings.ollama.num_ctx_answer:
        raise AssertionError("answer task should use writing context window")
    if default_settings.ollama.model_intent != "qwen3.5:4b":
        raise AssertionError("intent must use the validated semantic routing model")
    if default_settings.ollama.model_intent_fallback != "qwen3.5:4b":
        raise AssertionError("intent must have a non-ONNX recovery model")
    if default_settings.ollama.preload_onnx_models:
        raise AssertionError("ONNX model preloading must remain opt-in")

    expected_timeouts = {
        LLMTask.QUERY_REWRITE: 12.0,
        LLMTask.LAST_QA: 18.0,
        LLMTask.INTENT: 20.0,
        LLMTask.ACTION_EXTRACTION: 24.0,
        LLMTask.RISKY_ACTION: 35.0,
        LLMTask.RETRIEVAL_VALIDATION: 35.0,
        LLMTask.ANSWER: 75.0,
        LLMTask.WRITING: 90.0,
    }
    timeout_mismatches = {
        task.value: {"expected": expected_timeout, "actual": router.decision_for_task(task).timeout_seconds}
        for task, expected_timeout in expected_timeouts.items()
        if router.decision_for_task(task).timeout_seconds != expected_timeout
    }
    if timeout_mismatches:
        raise AssertionError(f"LLM timeout mismatches: {timeout_mismatches}")
    return ScenarioResult("model_routing_policy", True, "LLM task routing, retrieval, action, context, and worker defaults match policy")


def scenario_intent_branch_ownership_policy(settings: ProductionSettings) -> ScenarioResult:
    """State commands must retain their branch; clarification is context-bound."""
    class ScriptedIntentLLM:
        def __init__(self, payload: dict[str, Any]) -> None:
            self.payload = payload
            self.calls = 0

        def generate_json(self, **_: Any) -> dict[str, Any]:
            self.calls += 1
            return dict(self.payload)

    def classify(
        query: str,
        payload: dict[str, Any],
        *,
        resolution: Any = None,
    ) -> tuple[Intent, int]:
        kind_for_intent = {
            Intent.KNOWLEDGE_FACTS.value: "durable_knowledge",
            Intent.REMINDER.value: "reminder_lifecycle",
            Intent.CLARIFICATION.value: "clarification_reply",
            Intent.GENERAL_RESPONSE.value: "none",
        }
        payload = {
            **payload,
            "operation_kind": payload.get("operation_kind") or kind_for_intent[payload["intent"]],
        }
        llm = ScriptedIntentLLM(payload)
        intent = OllamaIntentClassifier(llm).classify(
            ChatRequest(user_id=DEBUG_USER, raw_query=query),
            query,
            last_qa_resolution=resolution,
        )
        return intent, llm.calls

    knowledge, knowledge_calls = classify(
        "Keep a durable note that weekly reports should be concise.",
        {"intent": "knowledge_facts", "confidence": 1.0, "multi_intent": False, "requires_clarification": False},
    )
    reminder, reminder_calls = classify(
        "Remind me tomorrow at 9 AM to submit expenses.",
        {"intent": "reminder", "confidence": 1.0, "multi_intent": False, "requires_clarification": False},
    )
    ordinary_save, ordinary_save_calls = classify(
        "Save my file as quarterly_report.pdf.",
        {"intent": "general_response", "confidence": 1.0, "multi_intent": False, "requires_clarification": False},
    )
    conversational_reminder, conversational_reminder_calls = classify(
        "Remind me what I said about the deployment plan.",
        {"intent": "general_response", "confidence": 1.0, "multi_intent": False, "requires_clarification": False},
    )
    durable_kind_wins, _ = classify(
        "Retain the deployment preference for future use.",
        {"intent": "reminder", "operation_kind": "durable_knowledge", "confidence": 1.0},
    )
    model_knowledge, _ = classify(
        "Keep a durable note that the Atlas project uses PostgreSQL.",
        {"intent": "knowledge_facts", "confidence": 0.96, "multi_intent": False, "requires_clarification": True},
    )
    general, _ = classify(
        "Explain the tradeoffs between PostgreSQL and SQLite.",
        {"intent": "general_response", "confidence": 0.96, "multi_intent": True, "requires_clarification": True},
    )
    ungrounded_clarification, _ = classify(
        "Help me organize my project notes.",
        {"intent": "clarification", "confidence": 0.98, "multi_intent": False, "requires_clarification": True},
    )
    pending_question = GeneratedQuestion(
        text="Which saved preference should I replace?",
        source=QuestionSource.CLARIFICATION_QUESTION,
        purpose="resolve_missing_info",
        confidence=1.0,
    )
    pending_resolution = type(
        "PendingClarificationResolution",
        (),
        {
            "state": LastQAState(
                last_user_query="Update my preference",
                last_response="Which saved preference should I replace?",
                response_type=ResponseType.CLARIFICATION,
                clarification_question=pending_question,
            )
        },
    )()
    grounded_clarification, _ = classify(
        "The concise-report preference.",
        {"intent": "clarification", "confidence": 0.98, "multi_intent": False, "requires_clarification": True},
        resolution=pending_resolution,
    )

    expected = {
        "explicit knowledge": knowledge,
        "explicit reminder": reminder,
        "ordinary file save": ordinary_save,
        "conversational remind-me question": conversational_reminder,
        "durable operation kind wins over a conflicting label": durable_kind_wins,
        "model-selected knowledge with missing fields": model_knowledge,
        "general request despite advisory flags": general,
        "ungrounded clarification": ungrounded_clarification,
        "grounded clarification": grounded_clarification,
    }
    wanted = {
        "explicit knowledge": Intent.KNOWLEDGE_FACTS,
        "explicit reminder": Intent.REMINDER,
        "ordinary file save": Intent.GENERAL_RESPONSE,
        "conversational remind-me question": Intent.GENERAL_RESPONSE,
        "durable operation kind wins over a conflicting label": Intent.KNOWLEDGE_FACTS,
        "model-selected knowledge with missing fields": Intent.KNOWLEDGE_FACTS,
        "general request despite advisory flags": Intent.GENERAL_RESPONSE,
        "ungrounded clarification": Intent.GENERAL_RESPONSE,
        "grounded clarification": Intent.CLARIFICATION,
    }
    wrong = {label: {"expected": wanted[label].value, "actual": actual.value} for label, actual in expected.items() if actual != wanted[label]}
    if wrong:
        raise AssertionError(f"intent branch ownership regressed: {wrong}")
    if any(calls != 1 for calls in (knowledge_calls, reminder_calls, ordinary_save_calls, conversational_reminder_calls)):
        raise AssertionError("intent routing must use the configured semantic classifier for every request")

    prompt = DEFAULT_PROMPT_REGISTRY.system("intent_classifier")
    required_prompt_rules = (
        "Choose operation_kind before intent; intent must agree with it.",
        "durable_knowledge means the user asks to retain, inspect, change, or remove",
        "reminder_lifecycle means the user asks to create, inspect, change, or remove a scheduled future notification.",
        "a new request is never clarification.",
    )
    missing_rules = [rule for rule in required_prompt_rules if rule not in prompt]
    if missing_rules:
        raise AssertionError(f"intent prompt lost required routing rules: {missing_rules}")
    if "multi_intent" in DEFAULT_PROMPT_REGISTRY.system("intent_classifier"):
        raise AssertionError("intent prompt must not request unused multi_intent output")
    return ScenarioResult("intent_branch_ownership_policy", True, "knowledge, reminders, general requests, and clarification follow semantic branch-ownership rules")


def scenario_last_qa_relationship_policy(settings: ProductionSettings) -> ScenarioResult:
    """Only evidence-backed Last-QA relationships may bypass broad retrieval."""
    config = build_assistant_config(ProductionSettings()).last_qa
    supporting_question = GeneratedQuestion(
        text="Which report format do you prefer?",
        source=QuestionSource.HUMAN_SUPPORTING_QUESTION,
        purpose="optional_context",
        confidence=1.0,
    )
    state = LastQAState(
        last_user_query="Create the weekly report.",
        last_response="I can prepare it. Which report format do you prefer?",
        response_type=ResponseType.NORMAL,
        supporting_questions=[supporting_question],
        linked_topic_id="topic-last-qa",
        linked_hop_id="hop-last-qa",
    )
    standard = {
        "confidence": 0.95,
        "llm_suggested_skip_broad_retrieval": True,
    }
    request = ChatRequest(user_id=DEBUG_USER, raw_query="Use PDF.")
    supporting = {
        **standard,
        "question_source": QuestionSource.HUMAN_SUPPORTING_QUESTION.value,
        "matched_question": supporting_question.text,
    }
    forged_supporting = {**supporting, "matched_question": "What output format should I use?"}
    normal = {
        **standard,
        "question_source": QuestionSource.NONE.value,
        "matched_question": "",
    }
    normal_with_question = {**normal, "matched_question": supporting_question.text}
    reminder_request = ChatRequest(
        user_id=DEBUG_USER,
        raw_query="Done.",
        metadata={
            "reminder_id": "reminder-1",
            "notification_id": "notification-1",
            "source_topic_id": "topic-reminder",
            "source_hop_id": "hop-reminder",
        },
    )
    reminder = {**normal}

    checks = {
        "exact supporting answer": can_skip_broad_retrieval(
            supporting, state, LastQAInteractionType.SUPPORTING_QUESTION_ANSWER, config, request
        ),
        "forged supporting match": can_skip_broad_retrieval(
            forged_supporting, state, LastQAInteractionType.SUPPORTING_QUESTION_ANSWER, config, request
        ),
        "direct normal follow-up": can_skip_broad_retrieval(
            normal, state, LastQAInteractionType.NORMAL_FOLLOW_UP, config, request
        ),
        "normal follow-up carrying a question": can_skip_broad_retrieval(
            normal_with_question, state, LastQAInteractionType.NORMAL_FOLLOW_UP, config, request
        ),
        "metadata-backed reminder reply": can_skip_broad_retrieval(
            reminder, state, LastQAInteractionType.REMINDER_NOTIFICATION_REPLY, config, reminder_request
        ),
        "metadata-free reminder reply": can_skip_broad_retrieval(
            reminder, state, LastQAInteractionType.REMINDER_NOTIFICATION_REPLY, config, request
        ),
        "clarification answer": can_skip_broad_retrieval(
            normal, state, LastQAInteractionType.CLARIFICATION_ANSWER, config, request
        ),
    }
    expected = {
        "exact supporting answer": True,
        "forged supporting match": False,
        "direct normal follow-up": True,
        "normal follow-up carrying a question": False,
        "metadata-backed reminder reply": True,
        "metadata-free reminder reply": False,
        "clarification answer": False,
    }
    wrong = {name: {"expected": expected[name], "actual": actual} for name, actual in checks.items() if actual != expected[name]}
    if wrong:
        raise AssertionError(f"Last-QA retrieval-skip evidence policy regressed: {wrong}")

    prompt = DEFAULT_PROMPT_REGISTRY.system("last_qa")
    required_rules = (
        "Use this precedence: reminder_notification_reply, supporting_question_answer, normal_follow_up",
        "Copy that question verbatim into matched_question.",
        "Text such as 'done', 'yes', or 'thanks' without those IDs is not a reminder reply.",
        "A new standalone request, even on a similar topic, is unrelated.",
        "clarification_merge runs first",
    )
    missing = [rule for rule in required_rules if rule not in prompt]
    if missing:
        raise AssertionError(f"Last-QA prompt lost relationship safeguards: {missing}")
    clarification_prompt = DEFAULT_PROMPT_REGISTRY.system("clarification_merge")
    standalone_rule = "A latest message that starts a distinct standalone request is not a clarification answer"
    if standalone_rule not in clarification_prompt:
        raise AssertionError("clarification merge lost its standalone-request safeguard")
    return ScenarioResult("last_qa_relationship_policy", True, "Last-QA precedence and skip gates require exact supporting-question or reminder evidence")


def scenario_debug_hybrid_llm_compatibility(settings: ProductionSettings) -> ScenarioResult:
    default_settings = ProductionSettings()
    router = OllamaModelRouter(default_settings.ollama)
    ollama_client = OllamaLLMClient(default_settings.ollama, router)
    onnx_client = ONNXLLMClient(router)
    llm = HybridLLMClient(ollama_client, onnx_client)

    onnx_client.last_error_by_task[LLMTask.ANSWER] = "synthetic ONNX compatibility error"
    lines = _debug_llm_lines(llm)
    text = "\n".join(lines)
    required_fragments = (
        "engine: hybrid_ollama_onnx",
        "base_url: http://localhost:11434",
        "onnx_cache_dir: .onnx_models",
        "answer=microsoft/Phi-4-mini-instruct-onnx",
        "query_rewrite=qwen3.5:4b",
        "synthetic ONNX compatibility error",
        '"model": "microsoft/Phi-4-mini-instruct-onnx"',
    )
    missing = [fragment for fragment in required_fragments if fragment not in text]
    if missing:
        raise AssertionError(f"hybrid debug LLM output missing fragments: {missing}\n{text}")
    if "base_url: unknown" in text or "keep_alive: unknown" in text:
        raise AssertionError(f"hybrid debug LLM output regressed to unknown settings:\n{text}")
    return ScenarioResult("debug_hybrid_llm_compatibility", True, "debug report exposes hybrid Ollama/ONNX routes and errors")


def scenario_hybrid_structured_onnx_failover(settings: ProductionSettings) -> ScenarioResult:
    """An unavailable ONNX structured route must recover through its configured Ollama model."""
    default_settings = ProductionSettings()
    onnx_route_settings = replace(
        default_settings.ollama,
        model_intent="microsoft/Phi-4-mini-instruct-onnx",
    )
    router = OllamaModelRouter(onnx_route_settings)
    ollama_client = OllamaLLMClient(onnx_route_settings, router)
    onnx_client = ONNXLLMClient(router)
    llm = HybridLLMClient(ollama_client, onnx_client)
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
    }
    calls: list[dict[str, Any]] = []

    def failed_onnx(**_: Any) -> dict[str, Any]:
        onnx_client.last_error_by_task[LLMTask.INTENT] = "structured_fallback_used_after_2_failures: repository unavailable"
        return {"ok": False}

    def successful_ollama(**kwargs: Any) -> str:
        calls.append(kwargs)
        return '{"ok":true}'

    onnx_client.generate_json = failed_onnx  # type: ignore[method-assign]
    ollama_client._chat_raw = successful_ollama  # type: ignore[method-assign]
    payload = llm.generate_json(
        task=LLMTask.INTENT,
        system_prompt="Return JSON.",
        user_prompt="Runtime context:\n{}",
        schema=schema,
    )
    if payload != {"ok": True}:
        raise AssertionError(f"Ollama fallback result was not returned: {payload}")
    if len(calls) != 1:
        raise AssertionError(f"expected one Ollama fallback attempt, got {len(calls)}")
    if calls[0].get("model_override") != onnx_route_settings.model_intent_fallback:
        raise AssertionError(f"intent fallback used the wrong model: {calls[0]}")
    if LLMTask.INTENT in llm.last_error_by_task:
        raise AssertionError(f"recovered ONNX failure leaked as active error: {llm.last_error_by_task}")
    return ScenarioResult("hybrid_structured_onnx_failover", True, "unavailable ONNX intent route recovers through Ollama without leaking a stale error")


def scenario_onnx_non_retryable_model_error(settings: ProductionSettings) -> ScenarioResult:
    """A bad model ID must not consume every configured structured retry."""
    default_settings = ProductionSettings()
    router = OllamaModelRouter(default_settings.ollama)
    onnx_client = ONNXLLMClient(router)
    attempts = 0
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
    }

    def missing_model(*_: Any, **__: Any) -> str:
        nonlocal attempts
        attempts += 1
        raise RuntimeError("404 Client Error: Repository Not Found")

    onnx_client._generate = missing_model  # type: ignore[method-assign]
    payload = onnx_client.generate_json(
        task=LLMTask.INTENT,
        system_prompt="Return JSON.",
        user_prompt="Runtime context:\n{}",
        schema=schema,
    )
    if payload.get("ok") is not False:
        raise AssertionError(f"unexpected structured fallback payload: {payload}")
    if attempts != 1:
        raise AssertionError(f"non-retryable model error made {attempts} attempts instead of one")
    if "after_1_failures" not in onnx_client.last_error_by_task.get(LLMTask.INTENT, ""):
        raise AssertionError(f"non-retryable failure did not report one attempt: {onnx_client.last_error_by_task}")
    return ScenarioResult("onnx_non_retryable_model_error", True, "missing ONNX repositories stop retrying after the first definitive failure")


def scenario_content_composer_general_react(settings: ProductionSettings) -> ScenarioResult:
    state = build_scenario_state(settings)
    config = GeneralPurposeConfig(
        content_composer_enabled=True,
        content_composer_max_iterations=1,
        content_composer_fallback_tool="answer_generation",
        content_composer_default_tool="answer_generation",
    )
    llm = ScenarioLLM()
    registry = ContentToolRegistry(
        tools=[AnswerGenerationTool(llm=llm, prompt_registry=DEFAULT_PROMPT_REGISTRY)],
        config=config,
    )
    composer = ReActContentComposer(
        registry=registry,
        llm=llm,
        prompt_registry=DEFAULT_PROMPT_REGISTRY,
        config=config,
    )
    from assistant_rag.contracts import (
        ContentComposerInput,
        GeneralSubBranch,
        PersistenceMode,
        SubBranchPromptContext,
    )

    result = composer.compose(
        ContentComposerInput(
            user_id=state.user_id,
            raw_user_query="I want to learn Python from zero.",
            rewritten_query="I want to learn Python from zero.",
            sub_branch=GeneralSubBranch.NEW_CONVERSATION_TOPIC,
            persistence_mode=PersistenceMode.CREATE_NEW_TOPIC,
            approved_conversation_history=[],
            human_supporting_questions=[],
            reminder_supporting_questions=[],
            extracted_expected_response_types=[],
            approved_knowledge_evidence=[],
            approved_reminder_context=[],
            metadata={},
            platform_context={},
            sub_branch_prompt_context=SubBranchPromptContext(
                sub_branch=GeneralSubBranch.NEW_CONVERSATION_TOPIC,
                persistence_mode=PersistenceMode.CREATE_NEW_TOPIC,
                chat_history_role="new conversation",
                response_goal="answer directly",
                database_update_mode="create",
                allowed_database_updates=("conversation_hop_append",),
                prohibited_database_updates=("knowledge_mutation",),
            ),
            sub_branch_supporting_prompt="Answer directly.",
        ),
        config,
    )
    if not result.final_response_text:
        raise AssertionError("content composer returned an empty answer")
    return ScenarioResult("content_composer_general_react", True, "composer handled flattened approved conversation context")


def scenario_general_broad_retrieval_approved(settings: ProductionSettings) -> ScenarioResult:
    state = build_scenario_state(settings)
    seed_conversation(
        state,
        title="Atlas Context",
        query="Project Atlas context",
        response="Atlas deployment requires a freeze window and QA signoff.",
        supporting_questions=[
            {
                "question_text": "Which environment should I focus on?",
                "question_source": QuestionSource.HUMAN_SUPPORTING_QUESTION.value,
                "purpose": "optional_context",
                "confidence": 1.0,
                "expected_response_type": ExpectedResponseType.FREE_TEXT_ANSWER.value,
            }
        ],
    )
    response = run_request(
        state,
        "Continue Atlas context with QA signoff",
        metadata={
            "intent": Intent.GENERAL_RESPONSE.value,
            "normal_response_text": "Continuing from the approved Atlas context.",
        },
    )
    assert_response(response, ResponseType.NORMAL, "approved Atlas")
    if state.retriever.conversation_calls < 1:
        raise AssertionError("broad conversation retrieval did not run")
    return ScenarioResult("general_broad_retrieval_approved", True, "approved conversation retrieval path ran")


def scenario_lastqa_supporting_skip(settings: ProductionSettings) -> ScenarioResult:
    state = build_scenario_state(settings)
    topic_id, hop_id = seed_conversation(
        state,
        title="Answer Draft",
        query="Draft the answer",
        response="Draft created.",
    )
    state.pipeline.last_qa_store.save(
        state.user_id,
        LastQAState(
            last_user_query="Draft the answer",
            last_response="Draft created.",
            response_type=ResponseType.NORMAL,
            supporting_questions=[
                GeneratedQuestion(
                    text="Which format should I use for the answer?",
                    source=QuestionSource.HUMAN_SUPPORTING_QUESTION,
                    purpose="optional_context",
                    confidence=1.0,
                )
            ],
            linked_topic_id=topic_id,
            linked_hop_id=hop_id,
        ),
    )
    before_calls = state.retriever.conversation_calls
    response = run_request(
        state,
        "format",
        metadata={
            "intent": Intent.GENERAL_RESPONSE.value,
            "normal_response_text": "Using that format for the prior answer.",
        },
    )
    assert_response(response, ResponseType.NORMAL, "prior answer")
    if state.retriever.conversation_calls != before_calls:
        raise AssertionError("last-QA supporting-question path should skip broad retrieval")
    return ScenarioResult("lastqa_supporting_skip", True, "latest-context Last-QA path skipped broad retrieval")


def scenario_knowledge_add(settings: ProductionSettings) -> ScenarioResult:
    state = build_scenario_state(settings)
    response = run_request(
        state,
        "Remember Atlas retention is 30 days",
        metadata={
            "intent": Intent.KNOWLEDGE_FACTS.value,
            "knowledge_actions": [
                {
                    "action": "add",
                    "text": "Project Atlas retention is 30 days.",
                    "topic_title": "Project Atlas",
                }
            ],
        },
    )
    assert_response(response, ResponseType.KNOWLEDGE_ACTION, "Added new knowledge")
    if state.repository.table_count("knowledge_chunks") != 1:
        raise AssertionError("knowledge add did not create a chunk")
    entities = state.repository.list_all_outbox_entities()
    for entity_type, entity_id in entities:
        state.repository.load_outbox_entity(entity_type=entity_type, entity_id=entity_id)
    return ScenarioResult("knowledge_add_and_outbox_load", True, "knowledge add and outbox entity loading succeeded")


def scenario_knowledge_modify(settings: ProductionSettings) -> ScenarioResult:
    state = build_scenario_state(settings)
    seed_knowledge(state, title="Project Atlas", text="Project Atlas retention is 30 days.")
    response = run_request(
        state,
        "Change Project Atlas retention to 45 days",
        metadata={
            "intent": Intent.KNOWLEDGE_FACTS.value,
            "knowledge_actions": [
                {
                    "action": "modify",
                    "target_description": "Project Atlas retention 30 days",
                    "replacement_text": "Project Atlas retention is 45 days.",
                }
            ],
        },
    )
    assert_response(response, ResponseType.KNOWLEDGE_ACTION, "confirm")
    if not response.actions_pending_confirmation:
        raise AssertionError("knowledge modify should require confirmation before mutating")
    active = state.repository.connection.execute(
        "SELECT raw_text FROM knowledge_chunks WHERE user_id = ? AND is_deleted = 0",
        (state.user_id,),
    ).fetchall()
    if [row["raw_text"] for row in active] != ["Project Atlas retention is 30 days."]:
        raise AssertionError("knowledge modify mutated before confirmation")
    return ScenarioResult("knowledge_modify", True, "knowledge modify requires confirmation before replacing the active chunk")


def scenario_knowledge_delete_not_found(settings: ProductionSettings) -> ScenarioResult:
    state = build_scenario_state(settings)
    response = run_request(
        state,
        "Delete a missing fact",
        metadata={
            "intent": Intent.KNOWLEDGE_FACTS.value,
            "knowledge_actions": [
                {"action": "delete", "target_description": "missing project codename"}
            ],
        },
    )
    assert_response(response, ResponseType.KNOWLEDGE_ACTION, "No matching knowledge")
    return ScenarioResult("knowledge_delete_not_found", True, "missing target produced safe no-op")


def scenario_knowledge_modify_missing_replacement(settings: ProductionSettings) -> ScenarioResult:
    state = build_scenario_state(settings)
    seed_knowledge(state, title="Project Atlas", text="Project Atlas owner is Mina.")
    response = run_request(
        state,
        "Modify Atlas owner",
        metadata={
            "intent": Intent.KNOWLEDGE_FACTS.value,
            "knowledge_actions": [
                {"action": "modify", "target_description": "Project Atlas owner Mina"}
            ],
            "clarification_text": "What should replace the current knowledge?",
        },
    )
    assert_response(response, ResponseType.CLARIFICATION, "replace")
    return ScenarioResult("knowledge_modify_missing_replacement", True, "missing replacement asks clarification")


def scenario_reminder_add(settings: ProductionSettings) -> ScenarioResult:
    state = build_scenario_state(settings)
    response = run_request(
        state,
        "Remind me to submit payroll tomorrow",
        metadata={
            "intent": Intent.REMINDER.value,
            "reminder_actions": [
                {
                    "action": "add",
                    "subject": "Submit payroll",
                    "reminder_summary": "Submit payroll",
                    "reminder_time": "2026-07-10T09:00:00+00:00",
                }
            ],
        },
    )
    assert_response(response, ResponseType.REMINDER_ACTION, "Added reminder")
    if state.repository.table_count("reminders") != 1:
        raise AssertionError("reminder add did not create a reminder")
    return ScenarioResult("reminder_add", True, "reminder add committed")


def scenario_reminder_supporting_question_printed(settings: ProductionSettings) -> ScenarioResult:
    state = build_scenario_state(settings)
    response = run_request(
        state,
        "Remind me to submit payroll tomorrow",
        metadata={
            "intent": Intent.REMINDER.value,
            "reminder_actions": [
                {
                    "action": "add",
                    "subject": "Submit payroll",
                    "reminder_summary": "Submit payroll",
                    "reminder_time": "2026-07-10T09:00:00+00:00",
                }
            ],
            "reminder_supporting_question": "Should this reminder repeat every payroll cycle?",
        },
    )
    assert_response(response, ResponseType.REMINDER_ACTION, "Added reminder")
    expected = "Reminder supporting question: Should this reminder repeat every payroll cycle?"
    if expected not in response.final_chat_text:
        raise AssertionError(f"reminder supporting question was not printed: {response.final_chat_text!r}")
    return ScenarioResult(
        "reminder_supporting_question_printed",
        True,
        "reminder supporting question is visible in final chat text",
    )


def scenario_reminder_modify(settings: ProductionSettings) -> ScenarioResult:
    state = build_scenario_state(settings)
    seed_reminder(
        state,
        subject="Submit tax form",
        summary="Submit tax form",
        reminder_time=datetime(2026, 7, 10, 9, tzinfo=timezone.utc),
    )
    response = run_request(
        state,
        "Move tax form reminder",
        metadata={
            "intent": Intent.REMINDER.value,
            "reminder_actions": [
                {
                    "action": "modify",
                    "target_description": "Submit tax form",
                    "new_subject": "Submit tax form final",
                    "new_reminder_time": "2026-07-11T10:00:00+00:00",
                }
            ],
        },
    )
    assert_response(response, ResponseType.REMINDER_ACTION, "Modified reminder")
    rows = state.repository.list_reminders(user_id=state.user_id)
    statuses = sorted(row["status"] for row in rows)
    if statuses != ["cancelled", "scheduled"]:
        raise AssertionError(f"expected one cancelled and one scheduled reminder, got {statuses}")
    return ScenarioResult("reminder_modify", True, "reminder modify cancelled old row and created replacement")


def scenario_reminder_turn_off(settings: ProductionSettings) -> ScenarioResult:
    state = build_scenario_state(settings)
    reminder_id = seed_reminder(
        state,
        subject="Call finance",
        summary="Call finance",
        reminder_time=datetime(2026, 7, 10, 9, tzinfo=timezone.utc),
    )
    response = run_request(
        state,
        "Turn off finance reminder",
        metadata={
            "intent": Intent.REMINDER.value,
            "reminder_actions": [
                {"action": "turn_off", "target_description": "Call finance"}
            ],
        },
    )
    assert_response(response, ResponseType.REMINDER_ACTION, "cancelled")
    row = state.repository.list_reminders(user_id=state.user_id)[0]
    if row["reminder_id"] != reminder_id or row["status"] != "cancelled":
        raise AssertionError("turn_off did not cancel the reminder")
    return ScenarioResult("reminder_turn_off", True, "turn_off moved scheduled reminder to cancelled")


def scenario_reminder_turn_on(settings: ProductionSettings) -> ScenarioResult:
    state = build_scenario_state(settings)
    seed_reminder(
        state,
        subject="Renew license",
        summary="Renew license",
        reminder_time=datetime(2026, 7, 10, 9, tzinfo=timezone.utc),
        status="cancelled",
    )
    response = run_request(
        state,
        "Turn on license reminder",
        metadata={
            "intent": Intent.REMINDER.value,
            "reminder_actions": [
                {"action": "turn_on", "target_description": "Renew license"}
            ],
        },
    )
    assert_response(response, ResponseType.REMINDER_ACTION, "scheduled")
    row = state.repository.list_reminders(user_id=state.user_id)[0]
    if row["status"] != "scheduled":
        raise AssertionError("turn_on did not reschedule the reminder")
    return ScenarioResult("reminder_turn_on", True, "turn_on moved cancelled reminder to scheduled")


def scenario_reminder_delete(settings: ProductionSettings) -> ScenarioResult:
    state = build_scenario_state(settings)
    seed_reminder(
        state,
        subject="Legacy review",
        summary="Legacy review",
        reminder_time=datetime(2026, 7, 10, 9, tzinfo=timezone.utc),
    )
    response = run_request(
        state,
        "Delete legacy reminder",
        metadata={
            "intent": Intent.REMINDER.value,
            "reminder_actions": [
                {"action": "delete", "target_description": "Legacy review"}
            ],
        },
    )
    assert_response(response, ResponseType.REMINDER_ACTION, "confirm")
    if not response.actions_pending_confirmation:
        raise AssertionError("reminder delete should require confirmation before mutating")
    row = state.repository.list_reminders(user_id=state.user_id)[0]
    if row["status"] != "scheduled":
        raise AssertionError("delete mutated before confirmation")
    return ScenarioResult("reminder_delete", True, "delete requires confirmation before dismissing the reminder")


def scenario_reminder_missing_time(settings: ProductionSettings) -> ScenarioResult:
    state = build_scenario_state(settings)
    response = run_request(
        state,
        "Remind me to file expenses",
        metadata={
            "intent": Intent.REMINDER.value,
            "reminder_actions": [{"action": "add", "subject": "File expenses"}],
            "clarification_text": "When should I remind you?",
            "expected_response_type": ExpectedResponseType.TIME_OR_DATE_ANSWER.value,
        },
    )
    assert_response(response, ResponseType.CLARIFICATION, "When")
    return ScenarioResult("reminder_missing_time", True, "missing reminder time asks clarification")


def scenario_autoscan_notification_reply(settings: ProductionSettings) -> ScenarioResult:
    state = build_scenario_state(settings)
    reminder_id = seed_reminder(
        state,
        subject="Join standup",
        summary="Join standup",
        reminder_time=datetime.now(timezone.utc) - timedelta(minutes=1),
    )
    notified = state.repository.scan_due_reminders(now_value=now_iso(), limit=10)
    if notified != [reminder_id]:
        raise AssertionError(f"autoscan did not notify expected reminder: {notified}")
    notifications = state.repository.list_notifications(user_id=state.user_id)
    if len(notifications) != 1:
        raise AssertionError("autoscan did not create one notification")
    notification_id = notifications[0]["notification_id"]
    context = state.repository.load_reminder_reply_context(
        user_id=state.user_id,
        reminder_id=reminder_id,
        notification_id=notification_id,
    )
    if context.get("subject") != "Join standup":
        raise AssertionError("reply context did not restore reminder subject")
    write = state.repository.append_reminder_reply(
        user_id=state.user_id,
        reminder_id=reminder_id,
        notification_id=notification_id,
        reply_text="Done",
        response_text="Marked your response.",
    )
    if not write.hop_id:
        raise AssertionError("reminder reply did not append a conversation hop")
    return ScenarioResult("autoscan_notification_reply", True, "autoscan, notification, context load, and reply append passed")


SCENARIOS: tuple[ScenarioFn, ...] = (
    scenario_clarification_direct,
    scenario_general_new_conversation,
    scenario_general_hitl_supporting_question_printed,
    scenario_general_new_conversation_skips_sub_branch_llm,
    scenario_answer_adaptive_token_budget,
    scenario_structured_clarification_fallback_policy,
    scenario_mutation_clarification_fast_path,
    scenario_state_mutation_preflight_bypass,
    scenario_model_backed_action_extraction,
    scenario_knowledge_action_recovery,
    scenario_sql_backed_knowledge_lookup,
    scenario_general_sql_knowledge_fallback,
    scenario_model_backed_knowledge_update,
    scenario_clarification_schema_echo_recovery,
    scenario_content_composer_react_structured_policy,
    scenario_structured_fallback_terminal_quiet,
    scenario_model_routing_policy,
    scenario_intent_branch_ownership_policy,
    scenario_last_qa_relationship_policy,
    scenario_debug_hybrid_llm_compatibility,
    scenario_hybrid_structured_onnx_failover,
    scenario_onnx_non_retryable_model_error,
    scenario_content_composer_general_react,
    scenario_general_broad_retrieval_approved,
    scenario_lastqa_supporting_skip,
    scenario_knowledge_add,
    scenario_knowledge_modify,
    scenario_knowledge_delete_not_found,
    scenario_knowledge_modify_missing_replacement,
    scenario_reminder_add,
    scenario_reminder_supporting_question_printed,
    scenario_reminder_modify,
    scenario_reminder_turn_off,
    scenario_reminder_turn_on,
    scenario_reminder_delete,
    scenario_reminder_missing_time,
    scenario_autoscan_notification_reply,
)


def run_scenario_suite(settings: ProductionSettings, names: set[str] | None = None) -> int:
    selected = []
    for scenario in SCENARIOS:
        scenario_name = scenario.__name__.removeprefix("scenario_")
        if names and scenario_name not in names:
            continue
        selected.append((scenario_name, scenario))
    if names:
        known = {scenario.__name__.removeprefix("scenario_") for scenario in SCENARIOS}
        unknown = sorted(names - known)
        if unknown:
            print(f"Unknown scenario(s): {', '.join(unknown)}", file=sys.stderr)
            return 2

    print(f"Running debug pipeline scenario suite ({len(selected)} scenario(s))...")
    failures: list[ScenarioResult] = []
    for scenario_name, scenario in selected:
        try:
            result = scenario(settings)
        except Exception as exc:
            result = ScenarioResult(scenario_name, False, f"{type(exc).__name__}: {exc}")
        marker = "PASS" if result.passed else "FAIL"
        print(f"[{marker}] {result.name}: {result.details}")
        if not result.passed:
            failures.append(result)

    if failures:
        print(f"\n{len(failures)} scenario(s) failed.", file=sys.stderr)
        return 1
    print("\nAll debug pipeline scenarios passed.")
    return 0


def run_llm_smoke_test(settings: ProductionSettings) -> int:
    """Exercise the configured answer model without initializing retrieval services."""

    smoke_ollama = replace(
        settings.ollama,
        num_ctx_answer=min(settings.ollama.num_ctx_answer, 512),
        num_predict_answer=32,
        temperature_answer=0.0,
    )
    router = OllamaModelRouter(smoke_ollama)
    llm = HybridLLMClient(
        OllamaLLMClient(smoke_ollama, router),
        ONNXLLMClient(router),
    )

    request_id = new_request_id()
    start_trace(request_id)
    try:
        response = llm.chat(
            task=LLMTask.ANSWER,
            system_prompt="You are a concise compatibility smoke test.",
            user_prompt="Reply with the exact words: Phi ONNX smoke test passed.",
        )
    except Exception as exc:
        print_current_debug_trace()
        print(f"LLM smoke test failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        for line in _debug_llm_lines(llm):
            print(line, file=sys.stderr)
        return 1

    print("LLM smoke test response:")
    print(response.strip())
    print()
    for line in _debug_trace_lines(current_trace().summary() if current_trace() else None):
        print(line)
    for line in _debug_llm_lines(llm):
        print(line)
    return 0


def interactive_main(settings: ProductionSettings) -> None:
    configure_debug_logging()
    print("Initializing real SQL-First RAG Assistant Pipeline (CLI Mode)...")
    # Do not apply settings.debug here: interactive debug must construct the
    # identical production architecture and use the same configured stores as
    # Streamlit.  Debug-only deterministic scenarios remain opt-in below.
    pipeline = build_production_pipeline(settings)
    repository = build_production_repository(settings)
    llm = _debug_llm_from_pipeline(pipeline)
    print_runtime_architecture(pipeline, repository)

    user_id = input("User ID [default: default_user]: ").strip() or "default_user"
    gmail_username = input("Gmail username [default: '']: ").strip()
    gmail_app_password = getpass("Gmail app password [default: '']: ")

    print("\nAssistant is ready! Type '/exit' or Ctrl+C to quit.")
    while True:
        try:
            query = input("\nAsk the assistant: ").strip()
            if not query:
                continue
            if query.lower() == "/exit":
                print("Exiting...")
                break

            request_id = new_request_id()
            start_trace(request_id)
            response = pipeline.handle(
                ChatRequest(
                    user_id=user_id,
                    raw_query=query,
                    platform_context={
                        "gmail_username": gmail_username,
                        "gmail_app_password": gmail_app_password,
                    },
                ),
                repository,
            )
            print_debug_report(response, llm=llm)
            logger.info(
                "real_pipeline_request_completed",
                extra={"payload": {
                    "response_type": response.response_type.value,
                    "trace_stage_count": len(response.trace_summary.stages) if response.trace_summary else 0,
                    "conversation_hop_id": response.conversation_hop_id,
                }},
            )
            print(f"\nAssistant:\n{response.final_chat_text}")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as exc:
            print_current_debug_trace()
            print(f"\nError running pipeline: {exc}", file=sys.stderr)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug and scenario-test the assistant pipeline.")
    parser.add_argument(
        "--scenario-suite",
        action="store_true",
        help="Run isolated deterministic component tests (not the real production-app debug path) and exit.",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        default=[],
        help="Run one named scenario from --list-scenarios. May be repeated.",
    )
    parser.add_argument(
        "--list-scenarios",
        action="store_true",
        help="Print deterministic scenario names and exit.",
    )
    parser.add_argument(
        "--llm-smoke-test",
        action="store_true",
        help="Load the configured answer model through the debug LLM wrapper and run one short answer call.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.list_scenarios:
        for scenario in SCENARIOS:
            print(scenario.__name__.removeprefix("scenario_"))
        return 0
    settings = ProductionSettings.from_env()
    if args.llm_smoke_test:
        return run_llm_smoke_test(settings)
    if args.scenario_suite or args.scenario:
        return run_scenario_suite(settings, set(args.scenario) if args.scenario else None)
    interactive_main(settings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
