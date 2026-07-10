from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
import json
import sys
from typing import Any, Callable

from assistant_rag.branch_orchestration import (
    KnowledgeTargetResolver,
    ReminderTargetResolver,
    ValidatedActionBuilder,
)
from assistant_rag.branches import (
    BranchRouter,
    ClarificationBranch,
    GeneralResponseBranch,
    KnowledgeFactsBranch,
    ReminderBranch,
)
from assistant_rag.bundler import ChatOutput, ResponseBundler
from assistant_rag.classification import LastQAResolver, QueryRewriter
from assistant_rag.config import GeneralPurposeConfig
from assistant_rag.context_filter import HardRuleContextFilter, TwoLayerContextFilter
from assistant_rag.content_composer import AnswerGenerationTool, ContentToolRegistry, ReActContentComposer
from assistant_rag.contracts import (
    ActionValidationResult,
    BundledResponse,
    ChatRequest,
    ExpectedResponseType,
    GeneratedQuestion,
    Intent,
    LastQAState,
    QuestionSource,
    ResponseType,
    RetrievalResult,
)
from assistant_rag.database import SQLiteRepository, now_iso
from assistant_rag.last_qa import InMemoryLastQAStore
from assistant_rag.llm import LLMTask, OllamaModelRouter
from assistant_rag.observability import current_trace, new_request_id, start_trace
from assistant_rag.platform import PlatformSelector
from assistant_rag.production_factory import (
    build_assistant_config,
    build_production_pipeline,
    build_production_repository,
)
from assistant_rag.prompts import DEFAULT_PROMPT_REGISTRY
from assistant_rag.settings import ProductionSettings


DEBUG_USER = "debug-user"


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


def _debug_llm_lines(llm: Any | None) -> list[str]:
    if llm is None:
        return ["Debug LLM: unavailable"]

    settings = getattr(llm, "settings", None)
    router = getattr(llm, "router", None)
    last_error_by_task = getattr(llm, "last_error_by_task", {}) or {}
    last_errors = {
        getattr(task, "value", str(task)): str(error)
        for task, error in last_error_by_task.items()
    }
    lines = [
        "Debug LLM:",
        f"  base_url: {getattr(settings, 'base_url', 'unknown')}",
        f"  keep_alive: {getattr(settings, 'keep_alive', 'unknown')}",
        f"  structured_attempts_for_json: {getattr(settings, 'structured_retry_count', 0) + 1 if settings else 'unknown'}",
    ]
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
            payload.update(
                {
                    "model": decision.model,
                    "single_call_timeout_seconds": decision.timeout_seconds,
                    "temperature": decision.temperature,
                    "num_ctx": decision.num_ctx,
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
            "reason_summary": "General learning-plan answer.",
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


def scenario_model_routing_policy(settings: ProductionSettings) -> ScenarioResult:
    default_settings = ProductionSettings()
    router = OllamaModelRouter(default_settings.ollama)
    expected_models = {
        LLMTask.QUERY_REWRITE: "qwen3.5:0.8b",
        LLMTask.LAST_QA: "qwen3.5:2b",
        LLMTask.INTENT: "qwen3.5:2b",
        LLMTask.ACTION_EXTRACTION: "qwen3.5:4b",
        LLMTask.GENERATE_CLARIFICATION: "qwen3.5:2b",
        LLMTask.GENERATE_HUMAN_SUPPORTING: "qwen3.5:2b",
        LLMTask.GENERATE_REMINDER_SUPPORTING: "qwen3.5:2b",
        LLMTask.CLARIFICATION_MERGE: "qwen3.5:4b",
        LLMTask.ANSWER: "qwen3.5:9b",
        LLMTask.WRITING: "qwen3.5:9b",
        LLMTask.RISKY_ACTION: "qwen3.5:9b",
        LLMTask.RETRIEVAL_VALIDATION: "qwen3.5:9b",
        LLMTask.GENERAL_SUB_BRANCH_DETECTION: "qwen3.5:2b",
        LLMTask.CONTENT_COMPOSER_REACT: "qwen3.5:2b",
        LLMTask.ACTION_PLANNING: "qwen3.5:4b",
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
    if default_settings.prompt_policy.risky_action_operations != ("delete", "modify", "turn_off"):
        raise AssertionError(f"risky action operations mismatch: {default_settings.prompt_policy.risky_action_operations}")
    if not default_settings.retrieval_validation.reminder_llm_validation_enabled:
        raise AssertionError("reminder LLM validation should be enabled by default")
    if not default_settings.retrieval_validation.knowledge_llm_validation_enabled:
        raise AssertionError("knowledge LLM validation should be enabled by default")
    if not default_settings.embeddings.normalize_embeddings:
        raise AssertionError("embedding normalization should be enabled")
    if default_settings.embeddings.model_name != "BAAI/bge-m3":
        raise AssertionError(f"embedding model mismatch: {default_settings.embeddings.model_name}")
    if router.decision_for_task(LLMTask.ANSWER).num_ctx != 8192:
        raise AssertionError("answer task should use writing context window")

    expected_timeouts = {
        LLMTask.QUERY_REWRITE: 12.0,
        LLMTask.LAST_QA: 18.0,
        LLMTask.INTENT: 18.0,
        LLMTask.ACTION_EXTRACTION: 30.0,
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
    scenario_model_routing_policy,
    scenario_content_composer_general_react,
    scenario_general_broad_retrieval_approved,
    scenario_lastqa_supporting_skip,
    scenario_knowledge_add,
    scenario_knowledge_modify,
    scenario_knowledge_delete_not_found,
    scenario_knowledge_modify_missing_replacement,
    scenario_reminder_add,
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


def interactive_main(settings: ProductionSettings) -> None:
    print("Initializing SQL-First RAG Assistant Pipeline (CLI Mode)...")
    debug_settings = _debug_runtime_settings(settings)
    pipeline = build_production_pipeline(debug_settings)
    repository = build_production_repository(debug_settings)
    llm = _debug_llm_from_pipeline(pipeline)

    user_id = input(f"User ID [default: {debug_settings.debug.user_id}]: ").strip() or debug_settings.debug.user_id
    gmail_username = input("Gmail username [default: '']: ").strip()

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
                    },
                ),
                repository,
            )
            print_debug_report(response, llm=llm)
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
        help="Run deterministic architecture/path scenarios and exit.",
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.list_scenarios:
        for scenario in SCENARIOS:
            print(scenario.__name__.removeprefix("scenario_"))
        return 0
    settings = ProductionSettings.from_env()
    if args.scenario_suite or args.scenario:
        return run_scenario_suite(settings, set(args.scenario) if args.scenario else None)
    interactive_main(settings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
