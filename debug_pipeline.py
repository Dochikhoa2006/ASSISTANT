from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass, fields, replace
from datetime import datetime, timedelta, timezone
from getpass import getpass
import imaplib
import json
import logging
import re
import smtplib
import sys
from tempfile import TemporaryDirectory
import zipfile
from typing import Any, Callable
from types import SimpleNamespace
from pathlib import Path

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
from assistant_rag.classification import LLMLastQAResolver, LastQAResolver, QueryRewriter, can_skip_broad_retrieval
from assistant_rag.config import GeneralPurposeConfig
from assistant_rag.context_filter import HardRuleContextFilter, TwoLayerContextFilter
from assistant_rag.retrieval_policy import (
    DEFAULT_CROSS_ENCODER_MIN_SCORE,
    RETRIEVAL_PIPELINE_POLICY,
)
from assistant_rag.content_composer import (
    AnswerGenerationTool,
    ContentToolRegistry,
    GenerateExcelTool,
    DeterministicContentComposer,
)
from assistant_rag.contracts import (
    BundledResponse,
    ChatRequest,
    ExpectedResponseType,
    GeneratedQuestion,
    HumanSupportingDecision,
    Intent,
    LastQAInteractionType,
    LastQAPath,
    LastQAState,
    PipelineContext,
    QuestionSource,
    ResponseType,
    RetrievalResult,
)
from assistant_rag.database import SQLiteRepository, now_iso
from assistant_rag.hybrid_llm import HybridLLMClient
from assistant_rag.knowledge_mutation import (
    KnowledgeContentFinalizationStrategy,
    KnowledgeMutationPipeline,
    LLMKnowledgeActionDetector,
)
from assistant_rag.reminder_mutation import (
    REMINDER_EDITABLE_FIELDS,
    LLMReminderActionDetector,
    ReminderActionValidationStrategy,
    ReminderContentFinalizationStrategy,
    ReminderMutationPipeline,
)
from assistant_rag.general_sub_branch import GeneralSubBranchDetector
from assistant_rag.last_qa import InMemoryLastQAStore
from assistant_rag.llm import (
    LLMTask,
    OllamaIntentClassifier,
    OllamaLLMClient,
    OllamaModelRouter,
    _structured_attempt_prompt,
    is_structured_fallback,
    llm_trace_stage_name,
    llm_trace_stage_name_for_prompt,
    structured_fallback_payload,
    uses_onnx_runtime,
    validate_json_schema,
)
from assistant_rag.observability import JsonLogFormatter, current_trace, new_request_id, start_trace
from assistant_rag.onnx_llm import ONNXLLMClient, _adaptive_max_new_tokens, _sentence_boundary_stop_reason
from assistant_rag.ops_cli import LocalMemoryIndex
from assistant_rag.platform import GmailSender, PlatformSelector
from assistant_rag.production_factory import (
    build_assistant_config,
    build_production_runtime,
)
from assistant_rag.prompts import CONTENT_COMPOSER_REACT_SCHEMA, DEFAULT_PROMPT_REGISTRY
from assistant_rag.request_lifecycle import (
    ChatRequestLifecycleExecutor,
    RequestLifecycleConflict,
)
from assistant_rag.retrieval_validation import KnowledgeRetrievalValidationStrategy
from assistant_rag.settings import (
    CAPABLE_LLM_MODEL,
    FAST_LLM_MODEL,
    ProductionSettings,
)
from assistant_rag.semantic_actions import grounded_semantic_action_from_payload


DEBUG_USER = "debug-user"
logger = logging.getLogger(__name__)


def _debug_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))


def _scenario_semantic_payload(
    query: str,
    *,
    message_operation: str = "none",
    channel: str = "none",
    message_evidence: str | None = None,
    recipients: tuple[str, ...] = (),
    excluded_recipients: tuple[str, ...] = (),
    global_cancellation: bool = False,
    cancellation_evidence: str | None = None,
    file_type: str = "none",
    file_evidence: str | None = None,
    file_type_evidence: str | None = None,
) -> dict[str, Any]:
    raw = {
        "message": {
            "operation": message_operation,
            "channel": channel,
            "recipient_update": "replace",
            "recipients": [
                {"value": item, "disposition": "include", "evidence": item}
                for item in recipients
            ] + [
                {"value": item, "disposition": "exclude", "evidence": item}
                for item in excluded_recipients
            ],
            "global_cancellation": global_cancellation,
            "authorization_evidence": [message_evidence] if message_evidence else [],
            "cancellation_evidence": (
                [cancellation_evidence] if cancellation_evidence else []
            ),
            "artifact_reference": "none",
            "copy_revision": False,
            "confidence": 0.99,
        },
        "file": {
            "operation": "create" if file_type != "none" else "none",
            "file_type": file_type,
            "authorization_evidence": [file_evidence] if file_evidence else [],
            "type_evidence": [file_type_evidence] if file_type_evidence else [],
            "confidence": 0.99,
        },
        "reason_summary": "debug semantic fixture",
    }
    return grounded_semantic_action_from_payload(
        raw, canonical_query=query
    ).to_payload()


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
    platform_payload = response.platform_payload or {}
    platform_delivery = {
        key: platform_payload[key]
        for key in ("platform_selection", "delivery", "draft")
        if key in platform_payload
    }
    if platform_delivery:
        _add_json_field("platform_delivery", platform_delivery)
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
        "chat_output",
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


class DeterministicGeneralHITLStrategy:
    """Debug-only stand-in for the production general-HITL decision contract."""

    def evaluate(self, *, context: Any, **_: Any) -> HumanSupportingDecision:
        question_text = context.request.metadata.get("supporting_question")
        if not question_text:
            return HumanSupportingDecision(
                should_ask=False,
                question="",
                confidence=1.0,
                question_source=QuestionSource.HUMAN_SUPPORTING_QUESTION,
                expected_response_type=ExpectedResponseType.UNKNOWN,
                reason_summary="No test supporting question supplied.",
                risk_flags=(),
            )
        return HumanSupportingDecision(
            should_ask=True,
            question=str(question_text),
            confidence=1.0,
            question_source=QuestionSource.HUMAN_SUPPORTING_QUESTION,
            expected_response_type=ExpectedResponseType.FREE_TEXT_ANSWER,
            reason_summary="Deterministic debug decision.",
            risk_flags=(),
        )


class DeterministicRetriever:
    def __init__(
        self,
        repository: SQLiteRepository,
        *,
        min_score: float = DEFAULT_CROSS_ENCODER_MIN_SCORE,
        conversation_min_score: float = 0.50,
    ) -> None:
        self.repository = repository
        self.min_score = min_score
        self.conversation_min_score = conversation_min_score
        self.conversation_calls = 0
        self.knowledge_calls = 0
        # The deterministic retriever still reads SQL directly, but exposing
        # the two derived stores makes every scenario exercise the production
        # request-scoped outbox fanout contract as well.
        self.bm25 = LocalMemoryIndex()
        self.chroma = LocalMemoryIndex()

    def retrieve_conversation(
        self, *, user_id: str, query: str
    ) -> list[RetrievalResult]:
        self.conversation_calls += 1
        query_norm = _normalize(query)
        rows = self.repository.connection.execute(
            """
            SELECT h.hop_id, h.topic_id, h.user_id, h.rewritten_user_query,
                    h.raw_response, h.supporting_questions_json, h.entities_json
            FROM conversation_hops h
            WHERE h.user_id = ?
            ORDER BY h.created_at DESC
            """,
            (user_id,),
        ).fetchall()
        results: list[RetrievalResult] = []
        for row in rows:
            text = (
                f"User: {row['rewritten_user_query']}\n"
                f"Assistant: {row['raw_response']}"
            )
            score = _token_overlap(query_norm, _normalize(text))
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
        results.sort(key=lambda item: item.rerank_score, reverse=True)
        return [
            result
            for result in results
            if result.rerank_score >= self.conversation_min_score
        ][: RETRIEVAL_PIPELINE_POLICY.final_top_k]

    def retrieve_knowledge(
        self,
        *,
        user_id: str,
        query: str,
        enforce_min_score: bool = True,
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
        results.sort(key=lambda item: item.rerank_score, reverse=True)
        return [
            result
            for result in results
            if not enforce_min_score or result.rerank_score >= self.min_score
        ][: RETRIEVAL_PIPELINE_POLICY.final_top_k]


class ScenarioLLM:
    def chat(self, **kwargs: Any) -> str:
        prompt = str(kwargs.get("user_prompt") or "")
        prefix = "Runtime context:\n"
        if prompt.startswith(prefix):
            try:
                payload = json.loads(prompt[len(prefix) :])
            except Exception:
                payload = {}
            metadata = payload.get("metadata") or {}
            scripted_response = metadata.get("normal_response_text")
            if scripted_response:
                return str(scripted_response)
        return "Start with Python basics, practice daily, then build small projects."

    def generate_json(self, **kwargs: Any) -> dict[str, Any]:
        task = kwargs.get("task")
        prompt = str(kwargs.get("user_prompt") or "")
        prefix = "Runtime context:\n"
        payload = json.loads(prompt[len(prefix) :]) if prompt.startswith(prefix) else {}
        if task is LLMTask.ANSWER:
            records = list(
                (payload.get("extra") or {}).get(
                    "approved_knowledge_records"
                )
                or []
            )
            for record in records:
                candidate_key = str(record.get("candidate_key") or "").strip()
                evidence_text = str(record.get("text") or "").strip()
                if candidate_key and evidence_text:
                    return {
                        "answer_text": evidence_text,
                        "evidence_references": [
                            {
                                "candidate_key": candidate_key,
                                "verbatim_support": evidence_text,
                            }
                        ],
                    }
        if task is LLMTask.KNOWLEDGE_ACTION_EXTRACTION:
            raw_query = str(
                payload.get("rewritten_query") or payload.get("raw_query") or ""
            ).strip()
            extra = payload.get("extra") or {}
            confirmed = list(
                extra.get("trusted_confirmation_action_context") or []
            )
            confirmed_action = confirmed[0] if len(confirmed) == 1 else {}
            if confirmed_action:
                action_name = str(confirmed_action.get("action") or "")
                if action_name == "add":
                    text_content = str(
                        confirmed_action.get("knowledge_text")
                        or confirmed_action.get("new_text")
                        or ""
                    )
                    original_text = ""
                    replacement_text = ""
                elif action_name == "delete":
                    text_content = str(
                        confirmed_action.get("target_description") or ""
                    )
                    original_text = ""
                    replacement_text = ""
                else:
                    text_content = ""
                    original_text = str(
                        confirmed_action.get("target_description") or ""
                    )
                    replacement_text = str(
                        confirmed_action.get("replacement_text")
                        or confirmed_action.get("new_text")
                        or ""
                    )
            else:
                folded = raw_query.casefold()
                if re.match(r"^(?:delete|remove|erase|forget)\b", folded):
                    action_name = "delete"
                elif re.match(
                    r"^(?:change|modify|update|edit|replace|revise)\b",
                    folded,
                ):
                    action_name = "modify"
                else:
                    action_name = "add"
                body = re.sub(
                    r"(?i)^(?:remember(?:\s+that)?|save|store|record|add)\s+",
                    "",
                    raw_query,
                ).strip(" .")
                text_content = body if action_name == "add" else ""
                original_text = ""
                replacement_text = ""
                if action_name == "delete":
                    text_content = re.sub(
                        r"(?i)^(?:delete|remove|erase|forget)\s+",
                        "",
                        raw_query,
                    ).strip(" .")
                elif action_name == "modify":
                    body = re.sub(
                        r"(?i)^(?:change|modify|update|edit|replace|revise)\s+",
                        "",
                        raw_query,
                    ).strip(" .")
                    match = re.match(r"(?is)^(.+?)\s+(?:to|with)\s+(.+)$", body)
                    original_text = match.group(1).strip() if match else body
                    replacement_text = match.group(2).strip() if match else ""
            return {
                "action": action_name,
                "text_content": text_content,
                "original_text": original_text,
                "replacement_text": replacement_text,
                "confidence": 0.99,
                "missing_fields": (
                    ["replacement_text"]
                    if action_name == "modify" and not replacement_text
                    else ["text_content"]
                    if action_name in {"add", "delete"} and not text_content
                    else []
                ),
                "reason_summary": "Scenario LLM selected one knowledge action.",
            }

        if task is LLMTask.KNOWLEDGE_ACTION_VALIDATION:
            first_response = payload.get("first_model_response") or {}
            operation = str(first_response.get("action") or "")
            proposed = (
                str(first_response.get("text_content") or "")
                if operation == "add"
                else ""
            )
            target = (
                str(first_response.get("text_content") or "")
                if operation == "delete"
                else str(first_response.get("original_text") or "")
            )
            replacement = str(first_response.get("replacement_text") or "")
            candidates = list(payload.get("knowledge_retrieval") or [])
            assessments: list[dict[str, Any]] = []
            selected: list[str] = []
            for candidate in candidates:
                key = str(candidate.get("candidate_key") or "")
                text = str(candidate.get("text_excerpt") or "")
                matched_text = ""
                matches = False
                if operation == "add":
                    matches = bool(proposed) and (
                        proposed.casefold() in text.casefold()
                        or text.casefold() in proposed.casefold()
                    )
                    matched_text = text if matches else ""
                elif operation == "modify":
                    replacement_value = re.search(
                        r"\b\d+\s+days?\b", replacement, re.I
                    )
                    old_value = re.search(r"\b\d+\s+days?\b", text, re.I)
                    replacement_is_only_value = bool(
                        replacement_value
                        and replacement.strip(" .").casefold()
                        == replacement_value.group(0).casefold()
                    )
                    if replacement_is_only_value and old_value:
                        matched_text = old_value.group(0)
                        matches = True
                    elif target and target.casefold() in text.casefold():
                        start = text.casefold().index(target.casefold())
                        matched_text = text[start : start + len(target)]
                        matches = True
                    else:
                        target_terms = set(
                            re.findall(r"[a-z0-9]+", target.casefold())
                        )
                        text_terms = set(
                            re.findall(r"[a-z0-9]+", text.casefold())
                        )
                        meaningful_target = {
                            term for term in target_terms if len(term) > 2
                        }
                        if meaningful_target and len(
                            meaningful_target & text_terms
                        ) / len(meaningful_target) >= 0.4:
                            matched_text = text
                            matches = True
                elif target and target.casefold() in text.casefold():
                    start = text.casefold().index(target.casefold())
                    target_key = " ".join(
                        re.findall(r"[a-z0-9]+", target.casefold())
                    )
                    text_key = " ".join(
                        re.findall(r"[a-z0-9]+", text.casefold())
                    )
                    matched_text = (
                        text
                        if operation == "delete" and target_key == text_key
                        else text[start : start + len(target)]
                    )
                    matches = True
                if matches and operation in {"delete", "modify"} and not selected:
                    selected = [key]
                assessments.append(
                    {
                        "candidate_key": key,
                        "confidence": 0.99 if matches else 0.95,
                        "matched_text": matched_text,
                    }
                )
            if operation == "modify" and not replacement:
                decision = "FAIL"
                clarification_question = (
                    "What should replace that stored fact?"
                )
                selected = []
            elif operation == "add":
                duplicate = any(item["matched_text"] for item in assessments)
                decision = "FAIL" if duplicate else "PASS"
                clarification_question = (
                    "That fact is already stored. What different fact should I add?"
                    if duplicate
                    else ""
                )
                selected = []
            elif selected and not (
                operation == "delete"
                and " ".join(
                    re.findall(
                        r"[a-z0-9]+",
                        next(
                            item["matched_text"]
                            for item in assessments
                            if item["candidate_key"] == selected[0]
                        ).casefold(),
                    )
                )
                != " ".join(
                    re.findall(
                        r"[a-z0-9]+",
                        next(
                            str(candidate.get("text_excerpt") or "")
                            for candidate in candidates
                            if str(candidate.get("candidate_key") or "")
                            == selected[0]
                        ).casefold(),
                    )
                )
            ):
                decision = "PASS"
                clarification_question = ""
            else:
                decision = "FAIL"
                selected = []
                clarification_question = (
                    "Which complete stored knowledge item should I delete?"
                    if operation == "delete"
                    else "Which stored fact should I modify, and what should replace it?"
                )
            return {
                "operation": operation,
                "decision": decision,
                "selected_candidate_keys": selected,
                "confidence": 0.99,
                "clarification_question": clarification_question,
                "reason_summary": "Deterministic debug validation.",
                "candidate_assessments": assessments,
            }

        if task is LLMTask.KNOWLEDGE_CONTENT_FINALIZATION:
            extra = payload.get("extra") or {}
            operation = str(extra.get("operation") or "")
            extracted = extra.get("extracted_action_content") or {}
            candidates = list(extra.get("validated_candidate_context") or [])
            validation = extra.get("validation_result") or {}
            selected = candidates[0] if len(candidates) == 1 else {}
            final_content = str(selected.get("text") or "")
            matched_text = str(selected.get("matched_text") or "")
            replacement = str(extracted.get("replacement_text") or "")
            if operation == "modify":
                final_content = final_content.replace(matched_text, replacement, 1)
            return {
                "final_content": final_content,
                "confidence": 0.99,
                "reason_summary": "Deterministic debug finalization.",
            }

        if task is LLMTask.REMINDER_ACTION_EXTRACTION:
            raw_query = str(
                payload.get("rewritten_query") or payload.get("raw_query") or ""
            )
            extra = payload.get("extra") or {}
            confirmed = list(
                extra.get("trusted_confirmation_action_context") or []
            )
            source = dict(confirmed[0]) if len(confirmed) == 1 else {}
            if not source:
                folded = raw_query.casefold()
                if re.match(r"^\s*(?:turn|switch)\s+off\b", folded):
                    action_name = "turn_off"
                    toggle_direction = ""
                elif re.match(r"^\s*(?:turn|switch)\s+on\b", folded):
                    action_name = "turn_on"
                    toggle_direction = ""
                elif re.match(r"^\s*(?:delete|remove|cancel)\b", folded):
                    action_name = "delete"
                    toggle_direction = ""
                elif re.match(
                    r"^\s*(?:modify|change|update|edit|move)\b", folded
                ):
                    action_name = "modify"
                    toggle_direction = ""
                else:
                    action_name = "add"
                    toggle_direction = ""
                source = {
                    "action": action_name,
                    "toggle_direction": toggle_direction,
                }
                if action_name == "add":
                    subject = re.sub(
                        r"(?i)^\s*(?:remind\s+me\s+to|set\s+a\s+reminder\s+to|add\s+a\s+reminder\s+to)\s+",
                        "",
                        raw_query,
                    ).strip(" .")
                    subject = re.split(
                        r"(?i)\s+(?:tomorrow|at\s+\d{4}-\d{2}-\d{2}t)",
                        subject,
                        maxsplit=1,
                    )[0].strip(" .")
                    iso_match = re.search(
                        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:Z|[+-]\d{2}:\d{2})",
                        raw_query,
                        re.I,
                    )
                    source.update(
                        {
                            "subject": subject,
                            "reminder_summary": subject,
                            "raw_reminder": subject,
                            "notification_time": (
                                iso_match.group(0)
                                if iso_match
                                else "2035-04-05T09:30:00+00:00"
                                if "tomorrow" in folded
                                else ""
                            ),
                            "user_timezone": "UTC",
                            "original_time_text": (
                                iso_match.group(0)
                                if iso_match
                                else "tomorrow"
                                if "tomorrow" in folded
                                else ""
                            ),
                        }
                    )
                elif action_name == "modify":
                    target_match = re.search(
                        r"(?i)^\s*(?:modify|change|update|edit|move)\s+(?:the\s+)?(.+?)\s+reminder(?:\s*:|\s+to|\s*$)",
                        raw_query,
                    )
                    subject_match = re.search(
                        r"(?i)(?:subject|title)\s+to\s+(.+?)(?:\s+and\s+the\s+notification|\s+and\s+notification|$)",
                        raw_query,
                    )
                    time_match = re.search(
                        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:Z|[+-]\d{2}:\d{2})",
                        raw_query,
                        re.I,
                    )
                    source["target_description"] = (
                        target_match.group(1).strip() if target_match else ""
                    )
                    if subject_match:
                        source["new_subject"] = subject_match.group(1).strip(" .")
                    if time_match:
                        source["new_reminder_time"] = time_match.group(0)
                        source["original_time_text"] = time_match.group(0)
                        source["user_timezone"] = "UTC"
                else:
                    target = re.sub(
                        r"(?i)^\s*(?:delete|remove|cancel|turn\s+on|turn\s+off|switch\s+on|switch\s+off)\s+(?:the\s+)?",
                        "",
                        raw_query,
                    )
                    source["target_description"] = re.sub(
                        r"(?i)\s+reminder\s*$", "", target
                    ).strip(" .")
            stored_action = str(source.get("action") or "add")
            action = (
                str(source.get("toggle_direction") or "")
                if stored_action == "toggle"
                else stored_action
            )
            record = {field: "" for field in REMINDER_EDITABLE_FIELDS}
            retrieval_text = str(
                source.get("retrieval_text")
                or source.get("target_description")
                or source.get("subject")
                or source.get("reminder_summary")
                or source.get("raw_reminder")
                or ""
            ).strip()
            changed_fields = [str(item) for item in source.get("changed_fields") or []]

            if action == "add":
                record.update(
                    {
                        "subject": str(source.get("subject") or retrieval_text).strip(),
                        "reminder_summary": str(
                            source.get("reminder_summary")
                            or source.get("subject")
                            or retrieval_text
                        ).strip(),
                        "raw_reminder": str(source.get("raw_reminder") or raw_query).strip(),
                        "notification_time": str(
                            source.get("notification_time")
                            or source.get("reminder_time")
                            or ""
                        ).strip(),
                        "event_time": str(source.get("event_time") or "").strip(),
                        "user_timezone": str(
                            source.get("user_timezone")
                            or (payload.get("platform_context") or {}).get("timezone")
                            or "UTC"
                        ).strip(),
                        "original_time_text": str(
                            source.get("original_time_text") or raw_query
                        ).strip(),
                        "recurrence_rule": str(source.get("recurrence_rule") or "").strip(),
                        "recurrence_timezone": str(
                            source.get("recurrence_timezone") or ""
                        ).strip(),
                        "supporting_question": str(
                            source.get("supporting_question") or ""
                        ).strip(),
                        "supporting_response": str(
                            source.get("supporting_response") or ""
                        ).strip(),
                    }
                )
                changed_fields = []
            elif action == "modify":
                replacement_keys = {
                    "subject": ("subject", "new_subject", "replacement_subject"),
                    "reminder_summary": (
                        "reminder_summary",
                        "new_summary",
                        "replacement_summary",
                    ),
                    "raw_reminder": ("raw_reminder", "replacement_raw_reminder"),
                    "notification_time": (
                        "notification_time",
                        "new_reminder_time",
                        "replacement_time",
                    ),
                    "event_time": ("event_time", "new_event_time"),
                    "user_timezone": ("user_timezone",),
                    "original_time_text": ("original_time_text",),
                    "recurrence_rule": (
                        "recurrence_rule",
                        "replacement_recurrence_rule",
                    ),
                    "recurrence_timezone": (
                        "recurrence_timezone",
                        "replacement_recurrence_timezone",
                    ),
                    "supporting_question": ("supporting_question",),
                    "supporting_response": ("supporting_response",),
                }
                for field, keys in replacement_keys.items():
                    present_key = next((key for key in keys if key in source), None)
                    if present_key is None:
                        continue
                    record[field] = str(source.get(present_key) or "").strip()
                    if field not in changed_fields:
                        changed_fields.append(field)
                if record["notification_time"] or record["event_time"]:
                    record["original_time_text"] = str(
                        source.get("original_time_text") or raw_query
                    ).strip()
                    if "original_time_text" not in changed_fields:
                        changed_fields.append("original_time_text")

            time_fields = {
                field for field in ("notification_time", "event_time") if record[field]
            }
            time_semantics = (
                "both"
                if len(time_fields) == 2
                else next(iter(time_fields))
                if time_fields
                else "unchanged"
            )
            missing_fields: list[str] = []
            if not retrieval_text:
                missing_fields.append("retrieval_text")
            if action == "add":
                if not record["subject"]:
                    missing_fields.append("subject")
                if not (record["notification_time"] or record["event_time"]):
                    missing_fields.append("notification_time")
            elif action == "modify" and not changed_fields:
                missing_fields.append("changed_fields")
            model_fields = tuple(
                field
                for field in REMINDER_EDITABLE_FIELDS
                if field not in {"supporting_question", "supporting_response"}
            )
            supplied_fields = (
                [field for field in model_fields if record[field]]
                if action == "add"
                else [
                    field
                    for field in changed_fields
                    if field in model_fields
                ]
                if action == "modify"
                else []
            )
            return {
                "action": action,
                "retrieval_text": retrieval_text,
                "field_values": [
                    {"field": field, "value": record[field]}
                    for field in supplied_fields
                ],
                "confidence": 0.99 if not missing_fields else 0.0,
            }

        if task is LLMTask.REMINDER_ACTION_VALIDATION:
            first_response = payload.get("first_model_response") or {}
            operation = str(first_response.get("action") or "")
            target = str(first_response.get("retrieval_text") or "").strip()
            field_values = {
                str(item.get("field") or ""): str(item.get("value") or "")
                for item in first_response.get("field_values") or []
                if isinstance(item, dict)
            }
            candidates = list(payload.get("reminder_retrieval") or [])
            assessments: list[dict[str, Any]] = []
            semantic_matches: list[str] = []
            for candidate in candidates:
                candidate_key = str(candidate.get("candidate_key") or "")
                matched_field = ""
                matched_text = ""
                for field in REMINDER_EDITABLE_FIELDS:
                    value = str(candidate.get(field) or "")
                    if not value or not target:
                        continue
                    folded_value = value.casefold()
                    folded_target = target.casefold()
                    if folded_target in folded_value:
                        start = folded_value.index(folded_target)
                        matched_field = field
                        matched_text = value[start : start + len(target)]
                        break
                    target_terms = set(re.findall(r"[a-z0-9]+", folded_target))
                    value_terms = set(re.findall(r"[a-z0-9]+", folded_value))
                    if target_terms and len(target_terms & value_terms) / len(target_terms) >= 0.6:
                        matched_field = field
                        matched_text = value
                        break
                matches = bool(matched_field)
                if matches:
                    semantic_matches.append(candidate_key)
                assessments.append(
                    {
                        "candidate_key": candidate_key,
                        "confidence": 0.99 if matches else 0.95,
                        "evidence_field": matched_field if matches else "",
                        "matched_text": matched_text,
                    }
                )

            has_time = bool(
                field_values.get("notification_time")
                or field_values.get("event_time")
            )
            extraction_complete = (
                bool(target and field_values.get("subject") and has_time)
                if operation == "add"
                else bool(target and field_values)
                if operation == "modify"
                else bool(target and not field_values)
            )
            if not extraction_complete:
                decision = "FAIL"
                selected = []
                clarification_question = (
                    "When should I remind you?"
                    if operation == "add" and not has_time
                    else "Which exact reminder and change should I use?"
                )
            elif operation == "add" and not semantic_matches:
                decision = "PASS"
                selected: list[str] = []
                clarification_question = ""
            elif operation != "add" and len(semantic_matches) == 1:
                decision = "PASS"
                selected = [semantic_matches[0]]
                clarification_question = ""
            else:
                decision = "FAIL"
                selected = []
                clarification_question = (
                    "That reminder already exists. What should be different?"
                    if operation == "add"
                    else "Which exact reminder should I change?"
                )
            return {
                "validation_result": decision,
                "selected_candidate_keys": selected,
                "confidence": 0.99,
                "clarification_question": clarification_question,
                "candidate_assessments": assessments,
            }

        if task is LLMTask.REMINDER_CONTENT_FINALIZATION:
            first_response = (payload.get("extra") or {}).get(
                "first_model_response"
            ) or payload.get("first_model_response") or {}
            return {
                "approved": bool(
                    first_response.get("action") == "modify"
                    and first_response.get("field_values")
                ),
                "confidence": 0.99,
            }

        raise AssertionError("deterministic content composer must not request a ReAct decision")


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
    retriever = DeterministicRetriever(
        repository,
        min_score=debug_settings.reranker.min_score,
        conversation_min_score=(
            debug_settings.retrieval.conversation_min_confidence_score
        ),
    )
    hard_filter = HardRuleContextFilter(
        allowed_reminder_statuses=debug_settings.prompt_policy.context_filter_allowed_reminder_statuses,
        reminder_approved_max_items=debug_settings.context_filter.reminder_approved_max_items,
        reminder_min_confidence=debug_settings.context_filter.reminder_min_confidence,
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
    knowledge_llm = ScenarioLLM()
    knowledge_validator = KnowledgeRetrievalValidationStrategy(
        config=config.retrieval_validation,
        llm=knowledge_llm,
        prompts=DEFAULT_PROMPT_REGISTRY,
    )
    knowledge_pipeline = KnowledgeMutationPipeline(
        retriever=retriever,
        config=config,
        validator=knowledge_validator,
        finalizer=KnowledgeContentFinalizationStrategy(
            llm=knowledge_llm,
            prompts=DEFAULT_PROMPT_REGISTRY,
            min_confidence=(
                config.retrieval_validation.knowledge_llm_validation_min_confidence
            ),
        ),
    )
    reminder_llm = ScenarioLLM()
    reminder_validator = ReminderActionValidationStrategy(
        config=config,
        llm=reminder_llm,
        prompts=DEFAULT_PROMPT_REGISTRY,
    )
    reminder_pipeline = ReminderMutationPipeline(
        config=config,
        validator=reminder_validator,
        finalizer=ReminderContentFinalizationStrategy(
            llm=reminder_llm,
            prompts=DEFAULT_PROMPT_REGISTRY,
            min_confidence=(
                config.retrieval_validation.reminder_llm_validation_min_confidence
            ),
        ),
    )
    gp_config = GeneralPurposeConfig(
        general_sub_branch_detector_enabled=True,
        content_composer_enabled=False,
        hitl_supporting_question_enabled=True,
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
                general_hitl_strategy=DeterministicGeneralHITLStrategy(),
                general_purpose_config=gp_config,
                llm=ScenarioLLM(),
                sub_branch_detector=GeneralSubBranchDetector(
                    ScenarioLLM(), DEFAULT_PROMPT_REGISTRY
                ),
            ),
            Intent.KNOWLEDGE_FACTS: KnowledgeFactsBranch(
                config=config,
                action_detector=LLMKnowledgeActionDetector(
                    llm=knowledge_llm,
                    prompts=DEFAULT_PROMPT_REGISTRY,
                    min_confidence=debug_settings.prompt_policy.action_min_confidence,
                ),
                clarification_strategy=clarification_strategy,
                prompt_registry=DEFAULT_PROMPT_REGISTRY,
                validated_action_builder=validated_builder,
                retriever=retriever,
                context_filter=context_filter,
                llm=knowledge_llm,
                knowledge_mutation_pipeline=knowledge_pipeline,
            ),
            Intent.REMINDER: ReminderBranch(
                config=config,
                action_detector=LLMReminderActionDetector(
                    llm=reminder_llm,
                    prompts=DEFAULT_PROMPT_REGISTRY,
                    min_confidence=debug_settings.prompt_policy.action_min_confidence,
                    default_timezone=config.default_timezone,
                ),
                prompt_registry=DEFAULT_PROMPT_REGISTRY,
                retriever=retriever,
                context_filter=context_filter,
                llm=reminder_llm,
                reminder_mutation_pipeline=reminder_pipeline,
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
    request_metadata = metadata or {}
    response = state.pipeline.handle(
        ChatRequest(
            user_id=state.user_id,
            raw_query=query,
            metadata=request_metadata,
            platform_context=platform_context or {},
        ),
        state.repository,
    )
    selected_intent = Intent(
        request_metadata.get("intent", Intent.GENERAL_RESPONSE.value)
    )
    if selected_intent not in {
        Intent.CLARIFICATION,
        Intent.GENERAL_RESPONSE,
        Intent.KNOWLEDGE_FACTS,
        Intent.REMINDER,
    }:
        return response
    if not response.conversation_topic_id or not response.conversation_hop_id:
        raise AssertionError(
            f"{selected_intent.value} returned without a persisted conversation hop"
        )
    persisted = state.repository.connection.execute(
        """
        SELECT topic_id, intent, response_type
        FROM conversation_hops
        WHERE user_id = ? AND hop_id = ?
        """,
        (state.user_id, response.conversation_hop_id),
    ).fetchone()
    if persisted is None:
        raise AssertionError(
            f"{selected_intent.value} hop is missing from SQL source truth"
        )
    if (
        str(persisted["topic_id"]) != response.conversation_topic_id
        or str(persisted["intent"]) != selected_intent.value
    ):
        raise AssertionError(
            f"{selected_intent.value} hop identity does not match SQL source truth"
        )
    for store_name, store in (
        ("OpenSearch", state.retriever.bm25),
        ("ChromaDB", state.retriever.chroma),
    ):
        indexed = store.get_document(entity_id=response.conversation_hop_id)
        if indexed is None or indexed.get("entity_type") != "conversation_hop":
            raise AssertionError(
                f"{selected_intent.value} hop was not synchronized to {store_name}"
            )
    outbox = state.repository.connection.execute(
        """
        SELECT status FROM indexing_outbox
        WHERE entity_type = 'conversation_hop' AND entity_id = ?
        ORDER BY created_at DESC LIMIT 1
        """,
        (response.conversation_hop_id,),
    ).fetchone()
    if outbox is None or str(outbox["status"]) != "completed":
        raise AssertionError(
            f"{selected_intent.value} hop outbox job was not completed"
        )
    return response


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


def assert_knowledge_audit_binding(
    state: ScenarioState,
    response: BundledResponse,
    *,
    action: str,
    knowledge_topic_id: str,
    knowledge_chunk_id: str,
) -> None:
    row = state.repository.connection.execute(
        """
        SELECT entities_json FROM conversation_hops
        WHERE user_id = ? AND hop_id = ?
        """,
        (state.user_id, response.conversation_hop_id),
    ).fetchone()
    if row is None:
        raise AssertionError("knowledge audit hop is missing from SQL")
    entities = json.loads(str(row["entities_json"] or "{}"))
    knowledge_entities = entities.get("knowledge") or []
    if len(knowledge_entities) != 1:
        raise AssertionError(
            f"knowledge audit hop must bind exactly one action: {knowledge_entities!r}"
        )
    binding = knowledge_entities[0]
    if (
        binding.get("action") != action
        or binding.get("knowledge_topic_id") != knowledge_topic_id
        or binding.get("knowledge_chunk_id") != knowledge_chunk_id
    ):
        raise AssertionError(
            f"knowledge audit binding does not match its SQL mutation: {binding!r}"
        )


def scenario_clarification_direct(settings: ProductionSettings) -> ScenarioResult:
    state = build_scenario_state(settings)
    response = run_request(
        state,
        "Need to update it",
        metadata={
            "intent": Intent.CLARIFICATION.value,
            "clarification_text": "Which item should I update?",
            "clarification_question": GeneratedQuestion(
                text="Which stale item was requested previously?",
                source=QuestionSource.CLARIFICATION_QUESTION,
                purpose="stale_metadata_question",
                confidence=1.0,
            ),
        },
    )
    assert_response(response, ResponseType.CLARIFICATION, "Which item")
    if "Clarification question:" not in response.final_chat_text:
        raise AssertionError(f"clarification question was not explicitly printed: {response.final_chat_text!r}")
    return ScenarioResult(
        "clarification_direct",
        True,
        "clarification branch returned and persisted its question",
    )


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

    class ForbiddenPlatformSelector:
        def select(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
            raise AssertionError("PlatformSelector ran while HITL owned the question")

    state.pipeline.platform_selector = ForbiddenPlatformSelector()  # type: ignore[assignment]
    response = run_request(
        state,
        "Send an email about the migration plan.",
        metadata={
            "intent": Intent.GENERAL_RESPONSE.value,
            "normal_response_text": (
                "Subject: Migration plan\n\n"
                "Start with schema compatibility, then migrate traffic gradually."
            ),
            "supporting_question": "Which recipient email address should receive it?",
        },
    )
    assert_response(response, ResponseType.NORMAL, "schema compatibility")
    if "Supporting question: Which recipient email address should receive it?" not in response.final_chat_text:
        raise AssertionError(f"HITL supporting question was not printed: {response.final_chat_text!r}")
    if response.platform_payload.get("platform_selection", {}).get("source") != "bypassed_active_branch_question":
        raise AssertionError("active HITL did not bypass PlatformSelector")
    if response.platform_payload.get("delivery", {}).get("status") != "deferred_by_active_question":
        raise AssertionError("HITL-owned turn exposed a competing platform delivery state")

    class SupportingAnswerLLM:
        def generate_json(self, **kwargs: Any) -> dict[str, Any]:
            properties = kwargs.get("schema", {}).get("properties", {})
            if "matched_question_index" not in properties:
                raise AssertionError("supporting-answer scenario received an unrelated schema")
            return {
                "relationship": "supporting_question_answer",
                "matched_question_index": 0,
                "confidence": 1.0,
            }

    state.pipeline.last_qa_resolver = LLMLastQAResolver(
        llm=SupportingAnswerLLM(),
        config=state.pipeline.config.last_qa,
        prompt_registry=DEFAULT_PROMPT_REGISTRY,
    )
    combined_delivery_query = (
        "Send an email about the migration plan.\n"
        "Resolved required context: ops@example.com"
    )
    resumed_semantic = grounded_semantic_action_from_payload(
        _scenario_semantic_payload(
            combined_delivery_query,
            message_operation="send",
            channel="gmail",
            message_evidence="Send an email",
            recipients=("ops@example.com",),
        ),
        canonical_query=combined_delivery_query,
    )

    class ResolvedSemanticAnalyzer:
        def analyze(self, query: str, **_kwargs: Any) -> Any:
            if query != combined_delivery_query:
                raise AssertionError(f"unexpected resolved platform query: {query!r}")
            return resumed_semantic

    state.pipeline.platform_selector = PlatformSelector(
        llm=None,
        semantic_analyzer=ResolvedSemanticAnalyzer(),  # type: ignore[arg-type]
    )
    resumed = run_request(
        state,
        "ops@example.com",
        metadata={
            "intent": Intent.GENERAL_RESPONSE.value,
            "normal_response_text": (
                "Subject: Migration plan\n\n"
                "Start with schema compatibility, then migrate traffic gradually."
            ),
        },
    )
    if resumed.platform_payload.get("platform_selection", {}).get("channel") != "gmail":
        raise AssertionError("the HITL answer did not resume the deferred Gmail request")
    if resumed.platform_payload.get("draft", {}).get("recipients") != ["ops@example.com"]:
        raise AssertionError("the HITL answer was not bound as the deferred recipient")
    delivery = resumed.platform_payload.get("delivery", {})
    if delivery.get("status") != "needs_input" or "question" in delivery or not delivery.get("notice"):
        raise AssertionError("resumed delivery did not remain a notice-only platform operation")
    return ScenarioResult(
        "general_hitl_supporting_question_printed",
        True,
        "one HITL question bypasses PlatformSelector and its answer resumes delivery",
    )


def scenario_general_hitl_disabled_does_not_fallback(settings: ProductionSettings) -> ScenarioResult:
    state = build_scenario_state(settings)
    branch = state.pipeline.router.branches[Intent.GENERAL_RESPONSE]

    class FailingLegacyHITL:
        def evaluate(self, *_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("disabled general HITL must not invoke the legacy generic strategy")

    branch.general_purpose_config = replace(
        branch.general_purpose_config,
        hitl_supporting_question_enabled=False,
    )
    branch.general_hitl_strategy = None
    branch.hitl_strategy = FailingLegacyHITL()
    response = run_request(
        state,
        "Explain the migration plan",
        metadata={
            "intent": Intent.GENERAL_RESPONSE.value,
            "normal_response_text": "Start with schema compatibility, then migrate traffic gradually.",
            "supporting_question": "This must never be injected while general HITL is disabled.",
        },
    )
    assert_response(response, ResponseType.NORMAL, "schema compatibility")
    if "Supporting question:" in response.final_chat_text:
        raise AssertionError("disabled general HITL emitted a supporting question")
    return ScenarioResult(
        "general_hitl_disabled_does_not_fallback",
        True,
        "disabled general HITL does not invoke or fall back to the legacy strategy",
    )


def scenario_user_entrypoint_runtime_parity(settings: ProductionSettings) -> ScenarioResult:
    """Both user-facing entrypoints must use the identical production runtime."""
    root = Path(__file__).resolve().parent
    for entrypoint in (root / "streamlit_app.py", root / "debug_pipeline.py"):
        source = entrypoint.read_text(encoding="utf-8")
        calls = {
            node.func.id
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        if "build_production_runtime" not in calls:
            raise AssertionError(f"{entrypoint.name} does not build the shared production runtime")
        if {"build_production_pipeline", "build_production_repository"} & calls:
            raise AssertionError(f"{entrypoint.name} bypasses the shared production runtime")

    from assistant_rag import production_factory as factory

    shared_llm = object()
    pipeline = SimpleNamespace(platform_selector=SimpleNamespace(llm=shared_llm))
    repository = object()
    received_llm: list[Any] = []
    original_pipeline = factory.build_production_pipeline
    original_repository = factory.build_production_repository
    original_planner = factory.build_reminder_timing_planner
    try:
        factory.build_production_pipeline = lambda _settings: pipeline
        factory.build_production_repository = lambda _settings: repository

        def build_planner(_settings: Any, *, llm: Any = None) -> Any:
            received_llm.append(llm)
            return SimpleNamespace()

        factory.build_reminder_timing_planner = build_planner
        runtime = factory.build_production_runtime(settings)
    finally:
        factory.build_production_pipeline = original_pipeline
        factory.build_production_repository = original_repository
        factory.build_reminder_timing_planner = original_planner

    if runtime.pipeline is not pipeline or runtime.repository is not repository or received_llm != [shared_llm]:
        raise AssertionError("production runtime did not preserve the single shared pipeline, repository, and LLM")
    try:
        factory.build_production_pipeline(
            replace(settings, model_warmup=replace(settings.model_warmup, enabled=False))
        )
    except ValueError as exc:
        if "MODEL_WARMUP" not in str(exc):
            raise AssertionError(f"model-warmup guard reported the wrong failure: {exc}") from exc
    else:
        raise AssertionError("production pipeline accepted a startup configuration with model warm-up disabled")
    return ScenarioResult(
        "user_entrypoint_runtime_parity",
        True,
        "Streamlit and interactive debug share one fully warmed production runtime contract",
    )


def scenario_streamlit_terminal_failure_containment(
    _settings: ProductionSettings,
) -> ScenarioResult:
    """An escaped backend failure must not crash chat or ask forever."""

    import streamlit_app

    request = ChatRequest(
        user_id="streamlit-debug-failure",
        raw_query="Handle this request.",
        idempotency_key="streamlit-debug-failure-key",
    )
    original_execute = streamlit_app._execute_user_request
    try:
        for error in (
            RequestLifecycleConflict("already in progress"),
            RuntimeError("backend unavailable"),
        ):
            def fail(**_kwargs: Any) -> Any:
                raise error

            streamlit_app._execute_user_request = fail
            execution, failure = streamlit_app._execute_user_request_safely(
                pipeline=object(),
                repository=object(),
                request=request,
            )
            if execution is not None or failure is None:
                raise AssertionError("Streamlit did not contain a terminal request failure")
            if failure.get("response_type") != ResponseType.ERROR.value:
                raise AssertionError(f"Streamlit failure was not typed as error: {failure!r}")
            if "?" in str(failure.get("content") or ""):
                raise AssertionError(f"Streamlit failure created a clarification loop: {failure!r}")
    finally:
        streamlit_app._execute_user_request = original_execute

    return ScenarioResult(
        "streamlit_terminal_failure_containment",
        True,
        "Streamlit preserves chat on lifecycle/backend failures without asking another clarification",
    )


def scenario_platform_gmail_multi_recipient_delivery(settings: ProductionSettings) -> ScenarioResult:
    delivery_query = (
        "Send this by Gmail to alice@example.com and bob@example.com."
    )
    class ScriptedPlatformLLM:
        def __init__(self, mode: str) -> None:
            self.mode = mode
            self.calls = 0

        def generate_json(self, **_kwargs: Any) -> dict[str, Any]:
            self.calls += 1
            if self.calls == 1:
                return {"channel": "gmail", "confidence": 1.0}
            return {
                "recipients": ["alice@example.com", "bob@example.com", "alice@example.com"],
                "subject": "Migration update",
                "body": "The migration is ready.",
                "mode": self.mode,
            }

    class FakeSMTP:
        instances: list[Any] = []

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            self.login_args: tuple[str, str] | None = None
            self.message: Any | None = None
            self.__class__.instances.append(self)

        def __enter__(self) -> "FakeSMTP":
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def login(self, username: str, password: str) -> None:
            self.login_args = (username, password)

        def send_message(self, message: Any) -> dict[str, Any]:
            self.message = message
            return {}

    class FakeIMAP:
        instances: list[Any] = []

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            self.login_args: tuple[str, str] | None = None
            self.append_args: tuple[Any, ...] | None = None
            self.__class__.instances.append(self)

        def login(self, username: str, password: str) -> tuple[str, list[bytes]]:
            self.login_args = (username, password)
            return "OK", []

        def list(self) -> tuple[str, list[bytes]]:
            return "OK", [b'(\\HasNoChildren \\Drafts) "/" "[Gmail]/Drafts"']

        def append(self, *args: Any) -> tuple[str, list[bytes]]:
            self.append_args = args
            return "OK", []

        def logout(self) -> tuple[str, list[bytes]]:
            return "BYE", []

    bundled = BundledResponse(
        final_chat_text="The migration is ready.",
        response_type=ResponseType.NORMAL,
        last_qa_state=LastQAState(
            last_user_query=delivery_query,
            last_response="",
            response_type=ResponseType.NORMAL,
        ),
        platform_payload={
            "semantic_action_decision": _scenario_semantic_payload(
                delivery_query,
                message_operation="send",
                channel="gmail",
                message_evidence="Send this by Gmail",
                recipients=("alice@example.com", "bob@example.com"),
            )
        },
    )

    def response_for_query(
        response: BundledResponse,
        query: str,
        semantic_payload: dict[str, Any],
    ) -> BundledResponse:
        return replace(
            response,
            last_qa_state=replace(
                response.last_qa_state,
                last_user_query=query,
            ),
            platform_payload={
                **response.platform_payload,
                "semantic_action_decision": semantic_payload,
            },
        )
    original_smtp_ssl = smtplib.SMTP_SSL
    try:
        smtplib.SMTP_SSL = FakeSMTP  # type: ignore[assignment]
        direct_delivery_llm = ScriptedPlatformLLM("send")
        send_selector = PlatformSelector(
            llm=direct_delivery_llm,
            senders={"gmail": GmailSender()},
        )
        sent = send_selector.select(
            bundled,
            ChatRequest(
                user_id="platform-test",
                raw_query=delivery_query,
                platform_context={
                    "gmail_username": "sender@example.com",
                    "gmail_app_password": "app-password-for-test",
                },
            ),
        )
        state = build_scenario_state(settings)
        pipeline_initial_semantic = grounded_semantic_action_from_payload(
            bundled.platform_payload["semantic_action_decision"],
            canonical_query=delivery_query,
        )
        follow_up_query = "Please transmit the active draft now."
        pipeline_follow_up_semantic = grounded_semantic_action_from_payload(
            _scenario_semantic_payload(
                follow_up_query,
                message_operation="send",
                channel="gmail",
                message_evidence="transmit the active draft",
            ),
            canonical_query=follow_up_query,
        )

        class PipelineSemanticAnalyzer:
            def analyze(self, query: str, **_kwargs: Any) -> Any:
                if query == delivery_query:
                    return pipeline_initial_semantic
                if query == follow_up_query:
                    return pipeline_follow_up_semantic
                raise AssertionError(f"unexpected pipeline delivery query: {query!r}")

        state.pipeline.platform_selector = PlatformSelector(
            llm=None,
            semantic_analyzer=PipelineSemanticAnalyzer(),  # type: ignore[arg-type]
            senders={"gmail": GmailSender()},
        )
        pipeline_response = run_request(
            state,
            delivery_query,
            metadata={
                "intent": Intent.GENERAL_RESPONSE.value,
                "normal_response_text": "The migration is ready.",
            },
            platform_context={
                "gmail_username": "sender@example.com",
                "gmail_app_password": "app-password-for-test",
            },
        )
    finally:
        smtplib.SMTP_SSL = original_smtp_ssl  # type: ignore[assignment]

    if sent["delivery"]["status"] != "sent" or sent["delivery"]["recipients"] != ["alice@example.com", "bob@example.com"]:
        raise AssertionError(f"multi-recipient Gmail send failed: {sent!r}")
    if direct_delivery_llm.calls:
        raise AssertionError(
            "explicit Gmail delivery invoked a redundant platform LLM call"
        )
    if len(FakeSMTP.instances) != 2:
        raise AssertionError(
            "selector and full pipeline did not each create exactly one SMTP "
            f"delivery: count={len(FakeSMTP.instances)}; "
            f"pipeline_payload={pipeline_response.platform_payload!r}"
        )
    smtp_client = FakeSMTP.instances[0]
    if smtp_client.login_args != ("sender@example.com", "app-password-for-test"):
        raise AssertionError("Gmail sender did not use the supplied UI/debug credentials")
    if str(smtp_client.message["To"]) != "alice@example.com, bob@example.com":
        raise AssertionError("Gmail sender did not address every extracted recipient")
    if "app-password-for-test" in json.dumps(sent):
        raise AssertionError("Gmail app password leaked into the public delivery payload")
    if pipeline_response.final_chat_text != "Sent via Gmail to alice@example.com, bob@example.com.":
        raise AssertionError(f"full pipeline did not report all Gmail recipients: {pipeline_response.final_chat_text!r}")
    delivery_count = state.repository.connection.execute(
        "SELECT COUNT(*) AS total FROM platform_deliveries"
    ).fetchone()["total"]
    if int(delivery_count) != 1:
        raise AssertionError("full pipeline did not audit the Gmail delivery exactly once")
    if "app-password-for-test" in json.dumps(pipeline_response.platform_payload):
        raise AssertionError("Gmail app password leaked through the full pipeline payload")
    debug_output = "\n".join(_debug_response_lines(pipeline_response))
    if "platform_delivery" not in debug_output or "alice@example.com, bob@example.com" not in debug_output:
        raise AssertionError("debug_pipeline did not expose the safe Gmail delivery envelope")
    if "app-password-for-test" in debug_output:
        raise AssertionError("Gmail app password leaked through debug_pipeline output")

    # A grounded email-writing decision must enter Gmail preparation even if
    # no platform model is available at the delivery boundary.
    # This is still draft-only: a write request cannot authorize SMTP sending.
    exact_request = (
        "Write an email to inform a day-off to vaiojjr@gmail.com "
        "koffdo75@gmail.com and do not send it to minh@gmail.com."
    )
    day_off_response = BundledResponse(
        final_chat_text=(
            "To inform the recipients about your day off, use this email:\n\n"
            "Subject: Day Off Notification\n\n"
            "Dear Vaiojjr and Koffdo,\n\n"
            "I will be unavailable tomorrow because I am taking a day off.\n\n"
            "Best regards,\n[Your Name]"
        ),
        response_type=ResponseType.NORMAL,
        last_qa_state=LastQAState(
            last_user_query=exact_request,
            last_response="",
            response_type=ResponseType.NORMAL,
        ),
        platform_payload={
            "semantic_action_decision": _scenario_semantic_payload(
                exact_request,
                message_operation="prepare",
                channel="gmail",
                message_evidence="Write an email",
                recipients=(
                    "vaiojjr@gmail.com",
                    "koffdo75@gmail.com",
                    "minh@gmail.com",
                ),
                excluded_recipients=("minh@gmail.com",),
            )
        },
    )
    deterministic_draft = PlatformSelector(llm=None).select(
        day_off_response,
        ChatRequest(user_id="platform-test", raw_query=exact_request),
    )
    if deterministic_draft["platform_selection"] != {
        "channel": "gmail",
        "confidence": 0.99,
        "source": "grounded_semantic_contract",
    }:
        raise AssertionError(f"explicit email request did not select Gmail: {deterministic_draft!r}")
    if deterministic_draft["delivery"]["status"] != "draft_ready":
        raise AssertionError(f"email-writing request did not produce a draft: {deterministic_draft!r}")
    if deterministic_draft["draft"]["recipients"] != ["vaiojjr@gmail.com", "koffdo75@gmail.com"]:
        raise AssertionError("explicit email request lost one or more recipients")
    if deterministic_draft["draft"]["subject"] != "Day Off Notification":
        raise AssertionError("email subject was not recovered from the bundled draft")
    if not deterministic_draft["draft"]["body"].startswith("Dear Vaiojjr and Koffdo,"):
        raise AssertionError("email body included explanatory text instead of the drafted message")

    # The composing response bundle owns the authored envelope. Supporting
    # query evidence validates/fills it but must not override its recipient
    # order, subject, or body, and the bundle cannot invent a new recipient.
    bundle_priority_query = (
        "Draft an email to alice@example.com and bob@example.com; "
        "subject: query fallback subject."
    )
    bundle_priority_response = replace(
        day_off_response,
        final_chat_text=(
            "To: mallory@example.com, bob@example.com, alice@example.com\n"
            "Subject: Bundled composer subject\n\n"
            "Dear team,\n\nBundled composer body."
        ),
        last_qa_state=replace(
            day_off_response.last_qa_state,
            last_user_query=bundle_priority_query,
        ),
        platform_payload={
            "outbound_message": {
                "recipients": [
                    "mallory@example.com",
                    "bob@example.com",
                    "alice@example.com",
                ],
                "subject": "Bundled composer subject",
                "body": "Dear team,\n\nBundled composer body.",
            },
            "semantic_action_decision": _scenario_semantic_payload(
                bundle_priority_query,
                message_operation="prepare",
                channel="gmail",
                message_evidence="Draft an email",
                recipients=("alice@example.com", "bob@example.com"),
            )
        },
    )
    bundle_priority_draft = PlatformSelector(llm=None).select(
        bundle_priority_response,
        ChatRequest(user_id="platform-test", raw_query=bundle_priority_query),
    )
    if bundle_priority_draft["draft"]["recipients"] != [
        "bob@example.com",
        "alice@example.com",
    ]:
        raise AssertionError(
            "response-bundler recipient priority or validation regressed"
        )
    if bundle_priority_draft["draft"]["subject"] != "Bundled composer subject":
        raise AssertionError("supporting input overrode the bundled email subject")
    if bundle_priority_draft["draft"]["body"] != (
        "Dear team,\n\nBundled composer body."
    ):
        raise AssertionError("supporting input overrode the bundled email body")

    class ExplodingDraftGmailSender(GmailSender):
        def create_draft(self, _payload: dict[str, Any], _platform_context: dict[str, Any]) -> dict[str, Any]:
            raise AssertionError("a write-email request must not save a remote Gmail draft")

    local_draft_with_credentials = PlatformSelector(
        llm=None,
        senders={"gmail": ExplodingDraftGmailSender()},
    ).select(
        day_off_response,
        ChatRequest(
            user_id="platform-test",
            raw_query=exact_request,
            platform_context={
                "gmail_username": "sender@example.com",
                "gmail_app_password": "app-password-for-test",
            },
        ),
    )
    if local_draft_with_credentials["delivery"]["status"] != "draft_ready":
        raise AssertionError("writing an email with saved credentials unexpectedly attempted Gmail IMAP")
    ordinary_query = "Is vaiojjr@gmail.com an email address I should use?"
    ordinary_address_question = PlatformSelector(llm=None).select(
        replace(
            day_off_response,
            last_qa_state=replace(
                day_off_response.last_qa_state,
                last_user_query=ordinary_query,
            ),
            platform_payload={
                "semantic_action_decision": _scenario_semantic_payload(
                    ordinary_query
                )
            },
        ),
        ChatRequest(
            user_id="platform-test",
            raw_query=ordinary_query,
        ),
    )
    if ordinary_address_question["delivery"]["channel"] != "none":
        raise AssertionError("an ordinary email-address question incorrectly entered Gmail delivery")

    state = build_scenario_state(settings)
    day_off_semantic = grounded_semantic_action_from_payload(
        day_off_response.platform_payload["semantic_action_decision"],
        canonical_query=exact_request,
    )
    active_send_query = "Please transmit the active draft now."
    active_send_semantic = grounded_semantic_action_from_payload(
        _scenario_semantic_payload(
            active_send_query,
            message_operation="send",
            channel="gmail",
            message_evidence="transmit the active draft",
        ),
        canonical_query=active_send_query,
    )

    class DayOffSemanticAnalyzer:
        def analyze(self, query: str, **_kwargs: Any) -> Any:
            if query == exact_request:
                return day_off_semantic
            if query == active_send_query:
                return active_send_semantic
            raise AssertionError(f"unexpected day-off platform query: {query!r}")

    state.pipeline.platform_selector = PlatformSelector(
        llm=None,
        semantic_analyzer=DayOffSemanticAnalyzer(),  # type: ignore[arg-type]
    )
    deterministic_pipeline_response = run_request(
        state,
        exact_request,
        metadata={
            "intent": Intent.GENERAL_RESPONSE.value,
            "normal_response_text": day_off_response.final_chat_text,
            "semantic_action_decision": day_off_response.platform_payload[
                "semantic_action_decision"
            ],
        },
    )
    if deterministic_pipeline_response.platform_payload["delivery"]["status"] != "draft_ready":
        raise AssertionError(
            "full pipeline did not preserve the grounded Gmail draft: "
            f"{deterministic_pipeline_response.platform_payload!r}"
        )
    if not deterministic_pipeline_response.final_chat_text.startswith("Gmail draft is ready for review"):
        raise AssertionError(f"full pipeline did not surface Gmail draft status: {deterministic_pipeline_response.final_chat_text!r}")
    if "It has not been sent." not in deterministic_pipeline_response.final_chat_text:
        raise AssertionError("local Gmail drafts did not clearly state that no email was sent")
    if deterministic_pipeline_response.last_qa_state.supporting_questions:
        raise AssertionError("an actionable explicit Gmail request emitted an unnecessary HITL question")
    active_draft = deterministic_pipeline_response.last_qa_state.outbound_state
    if active_draft is None:
        raise AssertionError("full pipeline did not retain the safe outbound envelope in Last-QA")
    if active_draft.recipients != ("vaiojjr@gmail.com", "koffdo75@gmail.com"):
        raise AssertionError("Last-QA outbound state lost one or more draft recipients")

    # The next turn is resolved semantically against the active envelope. It
    # must neither rediscover the addresses from this short instruction nor
    # enter broad conversation retrieval.
    smtp_count_before_follow_up = len(FakeSMTP.instances)

    class SemanticOutboundScenarioLLM:
        def generate_json(self, **kwargs: Any) -> dict[str, Any]:
            properties = kwargs.get("schema", {}).get("properties", {})
            if "outbound_action" not in properties:
                raise AssertionError("outbound scenario received an unrelated schema")
            return {"outbound_action": "send", "confidence": 1.0}

    state.pipeline.last_qa_resolver = LLMLastQAResolver(
        llm=SemanticOutboundScenarioLLM(),
        config=state.pipeline.config.last_qa,
        prompt_registry=DEFAULT_PROMPT_REGISTRY,
    )
    try:
        smtplib.SMTP_SSL = FakeSMTP  # type: ignore[assignment]
        semantic_follow_up = run_request(
            state,
            "Please transmit the active draft now.",
            metadata={
                "semantic_action_decision": _scenario_semantic_payload(
                    "Please transmit the active draft now.",
                    message_operation="send",
                    channel="gmail",
                    message_evidence="transmit the active draft",
                )
            },
            platform_context={
                "gmail_username": "sender@example.com",
                "gmail_app_password": "app-password-for-test",
            },
        )
    finally:
        smtplib.SMTP_SSL = original_smtp_ssl  # type: ignore[assignment]
    if semantic_follow_up.platform_payload["delivery"]["status"] != "sent":
        raise AssertionError(
            f"semantic outbound follow-up did not send the active draft: {semantic_follow_up.platform_payload!r}"
        )
    if semantic_follow_up.platform_payload["platform_selection"].get("source") != "authoritative_outbound_last_qa":
        raise AssertionError("short follow-up bypassed the authoritative Last-QA outbound path")
    if len(FakeSMTP.instances) != smtp_count_before_follow_up + 1:
        raise AssertionError("semantic follow-up did not perform exactly one SMTP dispatch")
    smtp_count_after_follow_up = len(FakeSMTP.instances)
    follow_up_message = FakeSMTP.instances[-1].message
    if str(follow_up_message["To"]) != "vaiojjr@gmail.com, koffdo75@gmail.com":
        raise AssertionError("semantic follow-up did not preserve the active recipient envelope")
    if str(follow_up_message["Subject"]) != active_draft.subject:
        raise AssertionError("semantic follow-up changed the active draft subject")
    if follow_up_message.get_body(preferencelist=("plain",)).get_content().strip() != active_draft.body:
        raise AssertionError("semantic follow-up changed the active draft body")
    sent_outbound = semantic_follow_up.last_qa_state.outbound_state
    if sent_outbound is None or sent_outbound.status != "sent":
        raise AssertionError("successfully sent outbound state was not retained as immutable history")

    # A grounded cancellation cannot be overridden by an unrelated extractor
    # response or turned into an SMTP side effect.
    original_imap_ssl = imaplib.IMAP4_SSL
    draft_query = (
        "Save a Gmail draft update to alice@example.com and bob@example.com; "
        "do not send."
    )
    try:
        imaplib.IMAP4_SSL = FakeIMAP  # type: ignore[assignment]
        draft_selector = PlatformSelector(llm=ScriptedPlatformLLM("send"), senders={"gmail": GmailSender()})
        drafted = draft_selector.select(
            response_for_query(
                bundled,
                draft_query,
                _scenario_semantic_payload(
                    draft_query,
                    message_operation="save_draft",
                    channel="gmail",
                    message_evidence="Save a Gmail draft",
                    recipients=("alice@example.com", "bob@example.com"),
                ),
            ),
            ChatRequest(
                user_id="platform-test",
                raw_query=draft_query,
                platform_context={
                    "gmail_username": "sender@example.com",
                    "gmail_app_password": "app-password-for-test",
                },
            ),
        )
    finally:
        imaplib.IMAP4_SSL = original_imap_ssl  # type: ignore[assignment]
    if drafted["delivery"]["status"] != "draft_saved" or drafted["draft"]["recipients"] != ["alice@example.com", "bob@example.com"]:
        raise AssertionError(f"multi-recipient Gmail draft failed: {drafted!r}")
    if len(FakeSMTP.instances) != smtp_count_after_follow_up:
        raise AssertionError("draft mode unexpectedly invoked Gmail SMTP sending")
    if len(FakeIMAP.instances) != 1 or FakeIMAP.instances[0].login_args != ("sender@example.com", "app-password-for-test"):
        raise AssertionError("Gmail draft did not use the supplied UI/debug credentials")
    if not FakeIMAP.instances[0].append_args or FakeIMAP.instances[0].append_args[0] != "[Gmail]/Drafts":
        raise AssertionError("Gmail draft was not appended to the provider draft mailbox")
    if FakeIMAP.instances[0].append_args[1] != "(\\Draft)":
        raise AssertionError("Gmail draft append did not use the IMAP draft flag")
    draft_bytes = FakeIMAP.instances[0].append_args[3]
    if b"alice@example.com, bob@example.com" not in draft_bytes:
        raise AssertionError("saved Gmail draft did not include every extracted recipient")

    # The Streamlit credential check must verify the IMAP path used by draft
    # creation, not merely the SMTP path used for sends.
    try:
        smtplib.SMTP_SSL = FakeSMTP  # type: ignore[assignment]
        imaplib.IMAP4_SSL = FakeIMAP  # type: ignore[assignment]
        valid_credentials, validation_message = GmailSender().validate_credentials(
            "sender@example.com", "app-password-for-test"
        )
        smtp_valid, smtp_validation_message = GmailSender().validate_smtp_credentials(
            "sender@example.com", "app-password-for-test"
        )
    finally:
        smtplib.SMTP_SSL = original_smtp_ssl  # type: ignore[assignment]
        imaplib.IMAP4_SSL = original_imap_ssl  # type: ignore[assignment]
    if not valid_credentials or validation_message != "Gmail SMTP and IMAP draft access are valid.":
        raise AssertionError(f"credential validation did not verify both Gmail protocols: {validation_message!r}")

    if not smtp_valid or smtp_validation_message != "Gmail sending credentials are valid.":
        raise AssertionError(f"SMTP-only credential validation was not available: {smtp_validation_message!r}")

    class RejectingIMAP(FakeIMAP):
        def login(self, username: str, password: str) -> tuple[str, list[bytes]]:
            super().login(username, password)
            raise imaplib.IMAP4.error("AUTHENTICATIONFAILED")

    try:
        smtplib.SMTP_SSL = FakeSMTP  # type: ignore[assignment]
        imaplib.IMAP4_SSL = RejectingIMAP  # type: ignore[assignment]
        valid_credentials, validation_message = GmailSender().validate_credentials(
            "sender@example.com", "app-password-for-test"
        )
        failed_draft = PlatformSelector(
            llm=ScriptedPlatformLLM("send"), senders={"gmail": GmailSender()}
        ).select(
            response_for_query(
                bundled,
                draft_query,
                _scenario_semantic_payload(
                    draft_query,
                    message_operation="save_draft",
                    channel="gmail",
                    message_evidence="Save a Gmail draft",
                    recipients=("alice@example.com", "bob@example.com"),
                ),
            ),
            ChatRequest(
                user_id="platform-test",
                raw_query=draft_query,
                platform_context={
                    "gmail_username": "sender@example.com",
                    "gmail_app_password": "app-password-for-test",
                },
            ),
        )
    finally:
        smtplib.SMTP_SSL = original_smtp_ssl  # type: ignore[assignment]
        imaplib.IMAP4_SSL = original_imap_ssl  # type: ignore[assignment]
    if valid_credentials or "Gmail IMAP draft access failed" not in validation_message:
        raise AssertionError(f"IMAP validation failure was not surfaced safely: {validation_message!r}")
    failure_notice = failed_draft["delivery"].get("notice", "")
    if failed_draft["delivery"]["status"] != "failed" or "Gmail rejected the IMAP sign-in" not in failure_notice:
        raise AssertionError(f"draft failure hid the actionable IMAP reason: {failed_draft!r}")

    # A generated artifact must be visible to Gmail extraction and travel as a
    # MIME attachment when the user explicitly asks to send that file.
    attachment_bundled = replace(
        bundled,
        platform_payload={
            "artifacts": [{
                "artifact_id": "platform-test-artifact",
                "filename": "generated_notes.txt",
                "storage_path": str(Path(__file__)),
            }],
        },
    )
    attachment_query = "Send the generated file to alice@example.com."
    try:
        smtplib.SMTP_SSL = FakeSMTP  # type: ignore[assignment]
        attached_delivery = PlatformSelector(
            llm=ScriptedPlatformLLM("send"), senders={"gmail": GmailSender()}
        ).select(
            response_for_query(
                attachment_bundled,
                attachment_query,
                _scenario_semantic_payload(
                    attachment_query,
                    message_operation="send",
                    channel="gmail",
                    message_evidence="Send the generated file",
                    recipients=("alice@example.com",),
                ),
            ),
            ChatRequest(
                user_id="platform-test",
                raw_query=attachment_query,
                platform_context={
                    "gmail_username": "sender@example.com",
                    "gmail_app_password": "app-password-for-test",
                },
            ),
        )
    finally:
        smtplib.SMTP_SSL = original_smtp_ssl  # type: ignore[assignment]
    if attached_delivery["delivery"]["status"] != "sent":
        raise AssertionError(f"explicit file delivery did not reach Gmail: {attached_delivery!r}")
    attachment_message = FakeSMTP.instances[-1].message
    if not str(attachment_message["Subject"]) or not list(attachment_message.iter_attachments()):
        raise AssertionError(
            "generated artifact was not attached to the Gmail message: "
            f"{attached_delivery!r}; attachment_count={len(list(attachment_message.iter_attachments()))}"
        )

    # Microsoft-tool artifacts must be attached without relying on generic
    # attach/file/document wording, and the raw request remains authoritative
    # when an incomplete extractor returns only one of multiple recipients.
    class IncompleteRecipientPlatformLLM:
        def __init__(self, mode: str) -> None:
            self.mode = mode

        def generate_json(self, **kwargs: Any) -> dict[str, Any]:
            if "platform selector" in str(kwargs.get("system_prompt") or ""):
                return {"channel": "gmail", "confidence": 1.0}
            return {
                "recipients": ["alice@example.com"],
                "subject": "Migration workbook",
                "body": "The migration workbook is ready.",
                "mode": self.mode,
            }

    from email import policy
    from email.parser import BytesParser

    workbook_bytes = b"debug-generated-workbook-mime-payload"
    with TemporaryDirectory(prefix="assistant-debug-gmail-artifact-") as artifact_dir:
        workbook_path = Path(artifact_dir) / "migration_budget.xlsx"
        workbook_path.write_bytes(workbook_bytes)
        workbook_bundled = replace(
            bundled,
            final_chat_text="Subject: Migration workbook\n\nThe migration workbook is ready.",
            platform_payload={
                "artifacts": [{
                    "artifact_id": "platform-test-workbook",
                    "file_type": "xlsx",
                    "filename": workbook_path.name,
                    "storage_path": str(workbook_path),
                    "storage_url": f"/artifacts/{workbook_path.name}",
                    "status": "created",
                }],
            },
        )
        credentials = {
            "gmail_username": "sender@example.com",
            "gmail_app_password": "app-password-for-test",
        }
        workbook_send_query = (
            "Create an Excel workbook with the migration budget, then email it to "
            "alice@example.com and bob@example.com now."
        )
        workbook_draft_query = (
            "Create an Excel workbook with the migration budget, then save a Gmail draft to "
            "alice@example.com and bob@example.com; do not send."
        )

        try:
            smtplib.SMTP_SSL = FakeSMTP  # type: ignore[assignment]
            workbook_sent = PlatformSelector(
                llm=IncompleteRecipientPlatformLLM("draft"),
                senders={"gmail": GmailSender()},
            ).select(
                response_for_query(
                    workbook_bundled,
                    workbook_send_query,
                    _scenario_semantic_payload(
                        workbook_send_query,
                        message_operation="send",
                        channel="gmail",
                        message_evidence="email it",
                        recipients=("alice@example.com", "bob@example.com"),
                    ),
                ),
                ChatRequest(
                    user_id="platform-test",
                    raw_query=workbook_send_query,
                    platform_context=credentials,
                ),
            )
        finally:
            smtplib.SMTP_SSL = original_smtp_ssl  # type: ignore[assignment]

        expected_recipients = ["alice@example.com", "bob@example.com"]
        if workbook_sent["delivery"]["status"] != "sent":
            raise AssertionError(f"compound workbook email was not sent: {workbook_sent!r}")
        if workbook_sent["delivery"]["recipients"] != expected_recipients:
            raise AssertionError("incomplete extractor dropped an explicit Gmail recipient")
        sent_workbook_message = FakeSMTP.instances[-1].message
        sent_workbook_attachments = list(sent_workbook_message.iter_attachments())
        if str(sent_workbook_message["To"]) != ", ".join(expected_recipients):
            raise AssertionError("compound workbook email did not address every raw recipient")
        if len(sent_workbook_attachments) != 1:
            raise AssertionError("compound workbook email did not include exactly one generated artifact")
        if sent_workbook_attachments[0].get_filename() != workbook_path.name:
            raise AssertionError("compound workbook email changed the generated artifact filename")
        if sent_workbook_attachments[0].get_content_type() != (
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        ):
            raise AssertionError("SMTP MIME attachment did not preserve the Excel content type")
        if sent_workbook_attachments[0].get_payload(decode=True) != workbook_bytes:
            raise AssertionError("SMTP MIME attachment bytes did not match the generated workbook")

        try:
            imaplib.IMAP4_SSL = FakeIMAP  # type: ignore[assignment]
            workbook_drafted = PlatformSelector(
                llm=IncompleteRecipientPlatformLLM("send"),
                senders={"gmail": GmailSender()},
            ).select(
                response_for_query(
                    workbook_bundled,
                    workbook_draft_query,
                    _scenario_semantic_payload(
                        workbook_draft_query,
                        message_operation="save_draft",
                        channel="gmail",
                        message_evidence="save a Gmail draft",
                        recipients=("alice@example.com", "bob@example.com"),
                    ),
                ),
                ChatRequest(
                    user_id="platform-test",
                    raw_query=workbook_draft_query,
                    platform_context=credentials,
                ),
            )
        finally:
            imaplib.IMAP4_SSL = original_imap_ssl  # type: ignore[assignment]

        if workbook_drafted["delivery"]["status"] != "draft_saved":
            raise AssertionError(f"compound workbook Gmail draft was not saved: {workbook_drafted!r}")
        if workbook_drafted["delivery"]["recipients"] != expected_recipients:
            raise AssertionError("saved workbook draft dropped an explicit Gmail recipient")
        saved_workbook_bytes = FakeIMAP.instances[-1].append_args[3]
        saved_workbook_message = BytesParser(policy=policy.default).parsebytes(saved_workbook_bytes)
        saved_workbook_attachments = list(saved_workbook_message.iter_attachments())
        if str(saved_workbook_message["To"]) != ", ".join(expected_recipients):
            raise AssertionError("saved workbook draft did not address every raw recipient")
        if len(saved_workbook_attachments) != 1:
            raise AssertionError("saved workbook draft did not include exactly one generated artifact")
        if saved_workbook_attachments[0].get_filename() != workbook_path.name:
            raise AssertionError("saved workbook draft changed the generated artifact filename")
        if saved_workbook_attachments[0].get_content_type() != (
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        ):
            raise AssertionError("IMAP MIME attachment did not preserve the Excel content type")
        if saved_workbook_attachments[0].get_payload(decode=True) != workbook_bytes:
            raise AssertionError("IMAP MIME attachment bytes did not match the generated workbook")

    smtp_connections_before_missing_credentials = len(FakeSMTP.instances)
    missing_credentials_selector = PlatformSelector(
        llm=ScriptedPlatformLLM("send"),
        senders={"gmail": GmailSender()},
    )
    missing_credentials = missing_credentials_selector.select(
        bundled,
        ChatRequest(
            user_id="platform-test",
            raw_query="Send this by Gmail to alice@example.com and bob@example.com.",
        ),
    )
    if missing_credentials["delivery"]["status"] != "needs_input":
        raise AssertionError("Gmail send without credentials was not safely blocked")
    if "question" in missing_credentials["delivery"] or not missing_credentials["delivery"].get("notice"):
        raise AssertionError("PlatformSelector turned delivery configuration into a conversational question")
    if len(FakeSMTP.instances) != smtp_connections_before_missing_credentials:
        raise AssertionError("missing Gmail credentials unexpectedly invoked SMTP sending")
    return ScenarioResult(
        "platform_gmail_multi_recipient_delivery",
        True,
        "Gmail send uses SMTP for all recipients and Gmail draft uses IMAP without sending",
    )


def scenario_general_new_conversation_skips_sub_branch_llm(settings: ProductionSettings) -> ScenarioResult:
    state = build_scenario_state(settings)

    class FailingSubBranchLLM:
        def chat(self, **_kwargs: Any) -> str:
            raise AssertionError("deterministic sub-branch selection must not call the LLM")

        def generate_json(self, **_kwargs: Any) -> dict[str, Any]:
            raise AssertionError("deterministic sub-branch selection must not call the LLM")

    branch = state.pipeline.router.branches[Intent.GENERAL_RESPONSE]
    branch.sub_branch_detector = GeneralSubBranchDetector(
        FailingSubBranchLLM(), DEFAULT_PROMPT_REGISTRY
    )
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
        "detector chose a context-free new topic without calling its retained LLM dependency",
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


def scenario_llm_answer_generation_trace(settings: ProductionSettings) -> ScenarioResult:
    class TraceableAnswerLLM(OllamaLLMClient):
        def _chat_raw(self, **_kwargs: Any) -> str:
            return "Synthetic answer used only to verify trace output."

    recorder = start_trace(new_request_id())
    client = TraceableAnswerLLM(settings.ollama, OllamaModelRouter(settings.ollama))
    config = GeneralPurposeConfig()
    composer = DeterministicContentComposer(
        registry=ContentToolRegistry(
            tools=[
                AnswerGenerationTool(
                    llm=client,
                    prompt_registry=DEFAULT_PROMPT_REGISTRY,
                ),
                GenerateExcelTool(
                    llm=client,
                    prompt_registry=DEFAULT_PROMPT_REGISTRY,
                    config=config,
                ),
            ],
            config=config,
        )
    )
    from assistant_rag.contracts import (
        ContentComposerInput,
        GeneralSubBranch,
        PersistenceMode,
        SubBranchPromptContext,
    )

    query = "Create an Excel workbook with Task, Owner, and Status columns."
    result = composer.compose(
        ContentComposerInput(
            user_id=DEBUG_USER,
            raw_user_query=query,
            rewritten_query=query,
            sub_branch=GeneralSubBranch.NEW_CONVERSATION_TOPIC,
            persistence_mode=PersistenceMode.CREATE_NEW_TOPIC,
            approved_conversation_history=[],
            human_supporting_questions=[],
            reminder_supporting_questions=[],
            extracted_expected_response_types=[],
            approved_knowledge_evidence=[],
            approved_reminder_context=[],
            metadata={
                "semantic_action_decision": _scenario_semantic_payload(
                    query,
                    file_type="xlsx",
                    file_evidence="Create",
                    file_type_evidence="Excel workbook",
                )
            },
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
    if result.used_tool_names != ("answer_generation", "generate_excel"):
        raise AssertionError(
            "composer did not drive mandatory answer generation before Microsoft "
            f"writing: {result.used_tool_names}"
        )
    summary = recorder.summary()
    stage_names = [stage.stage for stage in summary.stages]
    if stage_names != [
        "llm_answer_generation",
        "llm_writing_microsoft_tool",
    ]:
        raise AssertionError(f"general-purpose LLM trace labels regressed: {stage_names}")
    if llm_trace_stage_name(LLMTask.ANSWER, engine="onnx") != "llm_answer_generation":
        raise AssertionError("ONNX answer generation does not share the canonical trace label")
    if llm_trace_stage_name_for_prompt(
        LLMTask.WRITING,
        'Runtime context:\n{"stage":"content_tool_answer_generation"}',
        engine="onnx",
    ) != "llm_writing_microsoft_tool":
        raise AssertionError("ONNX Microsoft writing does not share the canonical trace label")
    debug_trace = "\n".join(_debug_trace_lines(summary))
    for required_stage in (
        "llm_answer_generation",
        "llm_writing_microsoft_tool",
    ):
        if required_stage not in debug_trace:
            raise AssertionError(f"debug trace formatter omitted {required_stage}")
    return ScenarioResult(
        "llm_answer_generation_trace",
        True,
        "general-purpose answer and optional Microsoft writing use stable ordered trace labels",
    )


def scenario_structured_clarification_fallback_policy(settings: ProductionSettings) -> ScenarioResult:
    default_settings = ProductionSettings()
    if default_settings.ollama.num_predict_generate_clarification < 160:
        raise AssertionError("clarification JSON generation needs enough token budget for all required fields")
    if default_settings.ollama.num_predict_generate_human_supporting < 160:
        raise AssertionError("human supporting-question JSON generation needs enough token budget")
    if default_settings.ollama.temperature_generate_clarification != 0.0:
        raise AssertionError("clarification JSON generation must use deterministic sampling")
    if default_settings.ollama.model_generate_clarification_fallback != "qwen3.5:4b":
        raise AssertionError("clarification generation needs an Ollama recovery model")
    answer_prompt = DEFAULT_PROMPT_REGISTRY.system("answer_generation")
    if "dedicated HITL stage owns every conversational question" not in answer_prompt:
        raise AssertionError("answer generation can still compete with HITL question ownership")
    question_prompt = DEFAULT_PROMPT_REGISTRY.system("question_generation")
    required_question_rules = (
        "required user-provided fact is missing",
        "return should_ask=false for complete or explicit requests",
        "delivery credentials",
        "next-step suggestions",
    )
    missing_question_rules = [
        rule for rule in required_question_rules if rule not in question_prompt
    ]
    if missing_question_rules:
        raise AssertionError(
            f"question necessity policy regressed: {missing_question_rules}"
        )

    schema = {
        "type": "object",
        "properties": {
            "question_text": {"type": "string"},
            "confidence": {"type": "number"},
            "expected_response_type": {
                "type": "string",
                "enum": [e.value for e in ExpectedResponseType],
            },
        },
        "required": [
            "question_text",
            "confidence",
            "expected_response_type",
        ],
    }
    payload = structured_fallback_payload(
        task=LLMTask.GENERATE_CLARIFICATION,
        schema=schema,
        user_prompt='Runtime context:\n{"raw_query":"Need to update it"}',
        error=ValueError("synthetic malformed JSON"),
    )
    validate_json_schema(payload, schema)
    if not is_structured_fallback(payload):
        raise AssertionError("structured fallback must retain out-of-band provenance")
    if payload["question_text"] or payload["confidence"] != 0.0:
        raise AssertionError("technical fallback must not impersonate a clarification decision")
    for task in (
        LLMTask.KNOWLEDGE_ACTION_VALIDATION,
        LLMTask.KNOWLEDGE_CONTENT_FINALIZATION,
        LLMTask.REMINDER_ACTION_VALIDATION,
        LLMTask.REMINDER_CONTENT_FINALIZATION,
        LLMTask.RETRIEVAL_VALIDATION,
    ):
        fallback_model = getattr(
            default_settings.ollama,
            f"model_{task.value}_fallback",
            None,
        )
        if not fallback_model or uses_onnx_runtime(fallback_model):
            raise AssertionError(
                f"{task.value} needs a non-ONNX structured recovery model"
            )
    return ScenarioResult(
        "structured_clarification_fallback_policy",
        True,
        "terminal structured fallback is marked, non-questioning, and cross-engine recoverable",
    )


def scenario_mutation_clarification_fast_path(settings: ProductionSettings) -> ScenarioResult:
    """Mutation gaps must not spend latency on removed or unnecessary stages."""
    state = build_scenario_state(settings)

    class LowConfidenceDetector:
        def __init__(self, missing_fields: list[str]) -> None:
            self.missing_fields = missing_fields

        def detect(self, *_: Any, **__: Any) -> Any:
            return type(
                "Detection",
                (),
                {
                    "requires_clarification": True,
                    "missing_fields": self.missing_fields,
                    "metadata": {},
                    "failure_kind": "semantic_gap",
                },
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
    if (
        reminder_result.response_type is not ResponseType.SAFE_NOOP
        or reminder_result.clarification_question is not None
    ):
        raise AssertionError(
            "reminder model-1 rejection must use the non-generative safe path: "
            f"{reminder_result}"
        )
    return ScenarioResult(
        "mutation_clarification_fast_path",
        True,
        "mutation gaps skip unused planning and the removed reminder clarification generator",
    )


def scenario_last_qa_mandatory_before_classification(settings: ProductionSettings) -> ScenarioResult:
    """Every request must load and resolve Last-QA before its only classification."""
    state = build_scenario_state(settings)
    events: list[str] = []

    class RecordingRewriter:
        def rewrite(self, query: str) -> str:
            events.append("rewrite")
            return QueryRewriter().rewrite(query)

    class RecordingLastQAStore:
        def __init__(self, delegate: Any) -> None:
            self.delegate = delegate

        def get(self, user_id: str) -> LastQAState | None:
            events.append("last_qa_state")
            return self.delegate.get(user_id)

        def save(self, user_id: str, value: LastQAState) -> None:
            self.delegate.save(user_id, value)

    class RecordingLastQAResolver:
        def resolve(
            self, request: ChatRequest, rewritten_query: str, last_state: LastQAState | None
        ) -> Any:
            events.append("last_qa_resolver")
            return LastQAResolver().resolve(request, rewritten_query, last_state)

    class RecordingClassifier:
        def classify(self, request: ChatRequest, *_: Any, **__: Any) -> Intent:
            events.append("classification")
            return Intent(request.metadata["intent"])

    state.pipeline.query_rewriter = RecordingRewriter()
    state.pipeline.last_qa_store = RecordingLastQAStore(state.pipeline.last_qa_store)
    state.pipeline.last_qa_resolver = RecordingLastQAResolver()
    state.pipeline.classifier = RecordingClassifier()
    start_trace(new_request_id())
    response = run_request(
        state,
        "Store the current project retention policy.",
        metadata={
            "intent": Intent.KNOWLEDGE_FACTS.value,
            "knowledge_actions": [{"action": "add", "text": "The project retention policy is current."}],
        },
    )
    assert_response(response, ResponseType.KNOWLEDGE_ACTION, "Added new knowledge")
    expected_prefix = ["rewrite", "last_qa_state", "last_qa_resolver", "classification"]
    if events[:4] != expected_prefix:
        raise AssertionError(f"mandatory Last-QA ordering regressed: {events}")
    if events.count("last_qa_resolver") != 1 or events.count("classification") != 1:
        raise AssertionError(f"resolver and classifier must each run exactly once: {events}")
    stage_names = [stage.stage for stage in response.trace_summary.stages] if response.trace_summary else []
    required_order = ["rewrite", "last_qa_resolution", "classification"]
    positions = [stage_names.index(name) for name in required_order]
    if positions != sorted(positions):
        raise AssertionError(f"pipeline stage contract regressed: {stage_names}")
    return ScenarioResult(
        "last_qa_mandatory_before_classification",
        True,
        "rewrite, Last-QA state load, Last-QA resolution, and one classification ran in order",
    )


def scenario_llm_first_action_extraction(settings: ProductionSettings) -> ScenarioResult:
    """The selected knowledge branch must begin with its extraction model."""
    detector = LLMKnowledgeActionDetector(
        llm=ScenarioLLM(),
        prompts=DEFAULT_PROMPT_REGISTRY,
    )
    query = "Remember that the weekly release note includes operational risks."
    result = detector.detect(
        ChatRequest(user_id=DEBUG_USER, raw_query=query),
        query,
        Intent.KNOWLEDGE_FACTS,
    )
    actions = result.metadata.get("knowledge_actions") or []
    if len(actions) != 1 or actions[0].get("action") != "add":
        raise AssertionError(f"LLM extraction selected the wrong action: {actions}")
    if "weekly release note" not in str(actions[0].get("text")):
        raise AssertionError(f"LLM add extraction lost supplied content: {actions}")
    if result.requires_clarification:
        raise AssertionError(f"explicit single action was incorrectly rejected: {result}")
    return ScenarioResult(
        "llm_first_action_extraction",
        True,
        "the knowledge extraction LLM selected exactly one grounded action",
    )


def scenario_llm_extraction_owns_single_action(settings: ProductionSettings) -> ScenarioResult:
    """Raw keyword multiplicity must not override the model's one-action schema."""
    detector = LLMKnowledgeActionDetector(
        llm=ScenarioLLM(),
        prompts=DEFAULT_PROMPT_REGISTRY,
    )
    result = detector.detect(
        ChatRequest(user_id=DEBUG_USER, raw_query="Delete the old policy and add a new policy."),
        "Delete the old policy and add a new policy.",
        Intent.KNOWLEDGE_FACTS,
    )
    actions = result.metadata.get("knowledge_actions") or []
    if result.requires_clarification or len(actions) != 1 or actions[0].get("action") != "delete":
        raise AssertionError(f"LLM extraction did not retain exactly one action: {result}")
    return ScenarioResult(
        "llm_extraction_owns_single_action",
        True,
        "one schema-valid LLM action was not vetoed by raw keyword counting",
    )


def scenario_read_only_knowledge_general_route(settings: ProductionSettings) -> ScenarioResult:
    """Knowledge reads must be answered only by the general-purpose branch."""
    class MustNotRunMutationDetector:
        def detect(self, *_: Any, **__: Any) -> Any:
            raise AssertionError("read-only knowledge query reached the mutation detector")

    state = build_scenario_state(settings)
    fact = "The stored infrastructure locality preference is Tokyo."
    seed_knowledge(state, title="Infrastructure preference", text=fact)
    state.pipeline.router.branches[
        Intent.KNOWLEDGE_FACTS
    ].action_detector = MustNotRunMutationDetector()
    response = run_request(
        state,
        "What is the stored infrastructure locality preference?",
        metadata={
            # The intent detector owns the read-versus-mutation distinction;
            # the router must dispatch its selected general branch directly.
            "intent": Intent.GENERAL_RESPONSE.value,
            "normal_response_text": fact,
        },
    )
    if response.response_type is not ResponseType.NORMAL or response.final_chat_text != fact:
        raise AssertionError(f"general knowledge answer did not return the stored fact: {response}")
    if state.retriever.knowledge_calls != 1:
        raise AssertionError("general-purpose knowledge answer did not use canonical retrieval")
    return ScenarioResult(
        "read_only_knowledge_general_route",
        True,
        "read-only knowledge lookup was owned exclusively by general-purpose",
    )


def scenario_general_sql_knowledge_fallback(settings: ProductionSettings) -> ScenarioResult:
    """General responses must use one canonical knowledge retrieval without a SQL bypass."""
    state = build_scenario_state(settings)
    fact = "The deployment preference for the assistant backend is Tokyo."
    seed_knowledge(state, title="Deployment preference", text=fact)
    response = run_request(
        state,
        "What is the deployment preference for the assistant backend?",
        metadata={"normal_response_text": fact},
    )
    if response.response_type is not ResponseType.NORMAL or response.final_chat_text != fact:
        raise AssertionError(f"general canonical retrieval response did not preserve the provided answer: {response}")
    if state.retriever.knowledge_calls != 1:
        raise AssertionError("general response did not use exactly one canonical knowledge retrieval")
    return ScenarioResult("general_sql_knowledge_fallback", True, "general response used one canonical knowledge retrieval")


def scenario_read_only_reminder_general_route(settings: ProductionSettings) -> ScenarioResult:
    """Reminder reads must be answered only by the general-purpose branch."""

    class MustNotRunMutationDetector:
        def detect(self, *_: Any, **__: Any) -> Any:
            raise AssertionError("read-only reminder query reached the mutation detector")

    state = build_scenario_state(settings)
    seed_reminder(
        state,
        subject="Submit payroll",
        summary="Submit payroll",
        reminder_time=datetime(2026, 7, 20, 9, tzinfo=timezone.utc),
    )
    state.pipeline.router.branches[
        Intent.REMINDER
    ].action_detector = MustNotRunMutationDetector()
    answer = "The Submit payroll reminder is scheduled for July 20 at 9 AM UTC."
    response = run_request(
        state,
        "When is my Submit payroll reminder?",
        metadata={
            "intent": Intent.GENERAL_RESPONSE.value,
            "normal_response_text": answer,
        },
    )
    assert_response(response, ResponseType.NORMAL, answer)
    reminders = state.repository.list_reminders(user_id=state.user_id)
    if len(reminders) != 1 or reminders[0]["status"] != "scheduled":
        raise AssertionError("read-only reminder answer changed reminder state")
    return ScenarioResult(
        "read_only_reminder_general_route",
        True,
        "read-only reminder lookup was owned exclusively by general-purpose",
    )


def scenario_llm_first_knowledge_update(settings: ProductionSettings) -> ScenarioResult:
    """A complete update must be extracted by the branch model before retrieval."""
    request = ChatRequest(user_id=DEBUG_USER, raw_query="Update the primary data-store policy to the new source-of-truth statement.")
    state = build_scenario_state(settings)
    seed_knowledge(state, title="Data policy", text="The primary data store is under review.")
    response = run_request(
        state,
        request.raw_query,
        metadata={"intent": Intent.KNOWLEDGE_FACTS.value},
    )
    assert_response(response, ResponseType.KNOWLEDGE_ACTION)
    if response.actions_pending_confirmation:
        raise AssertionError("knowledge PASS created a forbidden downstream confirmation")
    active = state.repository.connection.execute(
        "SELECT raw_text FROM knowledge_chunks WHERE user_id = ? AND is_deleted = 0",
        (state.user_id,),
    ).fetchall()
    if len(active) != 1 or "new source-of-truth statement" not in active[0]["raw_text"]:
        raise AssertionError("validated knowledge update did not commit directly")
    return ScenarioResult(
        "llm_first_knowledge_update",
        True,
        "the branch models preserved the update and LLM2 PASS committed it directly",
    )


def scenario_shared_confirmation_lifecycle(settings: ProductionSettings) -> ScenarioResult:
    """Knowledge PASS commits directly while request replay stays idempotent."""

    state = build_scenario_state(settings)
    seed_knowledge(
        state,
        title="Project Atlas",
        text="Project Atlas retention is 30 days.",
    )
    lifecycle = ChatRequestLifecycleExecutor(
        pipeline=state.pipeline,
        repository=state.repository,
    )
    query = "Change Project Atlas retention to 45 days"
    request = ChatRequest(
        user_id=state.user_id,
        raw_query=query,
        idempotency_key="scenario-direct-knowledge-pass",
        metadata={"intent": Intent.KNOWLEDGE_FACTS.value},
    )
    initial = lifecycle.execute(
        request,
        fallback_request_id="scenario-direct-knowledge-pass",
    )
    # The pipeline enriches request metadata in place. Replay with a fresh,
    # semantically identical request so the lifecycle compares the original
    # client payload instead of its runtime-enriched object.
    replayed = lifecycle.execute(
        ChatRequest(
            user_id=state.user_id,
            raw_query=query,
            idempotency_key="scenario-direct-knowledge-pass",
            metadata={"intent": Intent.KNOWLEDGE_FACTS.value},
        ),
        fallback_request_id="scenario-direct-knowledge-replay",
    )
    active = state.repository.connection.execute(
        """
        SELECT raw_text FROM knowledge_chunks
        WHERE user_id = ? AND is_deleted = 0
        """,
        (state.user_id,),
    ).fetchall()
    pending_count = state.repository.connection.execute(
        "SELECT COUNT(*) FROM pending_action_confirmations"
    ).fetchone()[0]
    if initial.response is None or not initial.response.actions_committed:
        raise AssertionError("knowledge PASS did not commit directly")
    if initial.response.actions_pending_confirmation:
        raise AssertionError("knowledge PASS created a pending confirmation")
    if not replayed.replayed or replayed.response is not None:
        raise AssertionError("repeated direct mutation did not use idempotent replay")
    if [row["raw_text"] for row in active] != [
        "Project Atlas retention is 45 days."
    ]:
        raise AssertionError(f"direct mutation produced incorrect knowledge: {active!r}")
    if pending_count != 0:
        raise AssertionError("direct knowledge mutation persisted a confirmation row")
    return ScenarioResult(
        "shared_confirmation_lifecycle",
        True,
        "knowledge PASS committed once without confirmation and replayed safely",
    )


def scenario_clarification_schema_echo_recovery(settings: ProductionSettings) -> ScenarioResult:
    """Question generation must request data instances, never induce schema copying."""
    schema = {
        "type": "object",
        "properties": {
            "question_text": {"type": "string"},
            "confidence": {"type": "number"},
            "expected_response_type": {"type": "string", "enum": [e.value for e in ExpectedResponseType]},
        },
        "required": [
            "question_text",
            "confidence",
            "expected_response_type",
        ],
    }
    system_prompt = DEFAULT_PROMPT_REGISTRY.system("question_generation")
    if "Never return, describe, or copy a JSON Schema." not in system_prompt:
        raise AssertionError("question-generation prompt does not forbid schema echoes")
    repair_prompt = _structured_attempt_prompt(
        user_prompt='Runtime context:\n{"stage":"generate_clarification"}',
        schema=schema,
        mode="schema",
        is_retry=True,
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
    valid_payload_text = '{"question_text":"Which preference should I update?","confidence":0.95,"expected_response_type":"free_text_answer"}'
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
        LLMTask.QUERY_REWRITE: FAST_LLM_MODEL,
        LLMTask.LAST_QA: CAPABLE_LLM_MODEL,
        LLMTask.INTENT: FAST_LLM_MODEL,
        LLMTask.ACTION_EXTRACTION: FAST_LLM_MODEL,
        LLMTask.KNOWLEDGE_ACTION_EXTRACTION: FAST_LLM_MODEL,
        LLMTask.KNOWLEDGE_ACTION_VALIDATION: CAPABLE_LLM_MODEL,
        LLMTask.KNOWLEDGE_CONTENT_FINALIZATION: CAPABLE_LLM_MODEL,
        LLMTask.REMINDER_ACTION_EXTRACTION: FAST_LLM_MODEL,
        LLMTask.REMINDER_ACTION_VALIDATION: CAPABLE_LLM_MODEL,
        LLMTask.REMINDER_CONTENT_FINALIZATION: CAPABLE_LLM_MODEL,
        LLMTask.GENERATE_CLARIFICATION: FAST_LLM_MODEL,
        LLMTask.GENERATE_HUMAN_SUPPORTING: FAST_LLM_MODEL,
        LLMTask.CLARIFICATION_MERGE: FAST_LLM_MODEL,
        LLMTask.ANSWER: CAPABLE_LLM_MODEL,
        LLMTask.WRITING: CAPABLE_LLM_MODEL,
        LLMTask.RISKY_ACTION: FAST_LLM_MODEL,
        LLMTask.RETRIEVAL_VALIDATION: CAPABLE_LLM_MODEL,
        LLMTask.GENERAL_SUB_BRANCH_DETECTION: FAST_LLM_MODEL,
        LLMTask.CONTENT_COMPOSER_REACT: FAST_LLM_MODEL,
        LLMTask.ACTION_PLANNING: CAPABLE_LLM_MODEL,
    }
    actual_models = {task: router.model_for_task(task) for task in expected_models}
    mismatches = {
        task.value: {"expected": expected_model, "actual": actual_models[task]}
        for task, expected_model in expected_models.items()
        if actual_models[task] != expected_model
    }
    if mismatches:
        raise AssertionError(f"model routing mismatches: {mismatches}")
    configured_models = {
        str(getattr(default_settings.ollama, field_info.name))
        for field_info in fields(default_settings.ollama)
        if field_info.name.startswith("model_")
        and getattr(default_settings.ollama, field_info.name)
    }
    if configured_models != {FAST_LLM_MODEL, CAPABLE_LLM_MODEL}:
        raise AssertionError(
            f"production must resolve exactly the two-model pool: {configured_models}"
        )
    if any(uses_onnx_runtime(model) for model in actual_models.values()):
        raise AssertionError("default production routes must not retain ONNX LLM weights")
    if default_settings.ollama.keep_alive != "5m":
        raise AssertionError("idle Ollama weights should expire after five minutes")
    if default_settings.ollama.model_last_qa_fallback != FAST_LLM_MODEL:
        raise AssertionError("Last-QA needs a local cross-model recovery route")
    if default_settings.ollama.json_retry_count_last_qa != 0:
        raise AssertionError(
            "Last-QA should fail over models instead of repeating the same malformed output"
        )

    # Check some basic policy settings
    policy_values = {
        "bm25_top_k": default_settings.retrieval.bm25_top_k,
        "chroma_top_k": default_settings.retrieval.chroma_top_k,
        "rrf_k": default_settings.retrieval.rrf_k,
        "reranker_top_k": default_settings.retrieval.rerank_candidate_limit,
        "reranker_min_score": default_settings.reranker.min_score,
        "conversation_min_confidence_score": default_settings.retrieval.conversation_min_confidence_score,
        "reranker_batch_size": default_settings.reranker.batch_size,
        "final_context_top_k": default_settings.retrieval.max_results,
        "last_qa_min_confidence": default_settings.prompt_policy.last_qa_min_confidence,
        "last_qa_clarification_merge_min_confidence": default_settings.prompt_policy.clarification_merge_min_confidence,
        "last_qa_skip_broad_retrieval_min_confidence": default_settings.prompt_policy.last_qa_skip_broad_retrieval_min_confidence,
        "support_question_resolution_min_confidence": default_settings.general_purpose.support_question_resolution_min_confidence,
        "conversation_followup_min_score": default_settings.general_purpose.conversation_followup_min_score,
        "action_min_confidence": default_settings.prompt_policy.action_min_confidence,
        "knowledge_validation_candidate_limit": default_settings.retrieval_validation.knowledge_llm_validation_max_candidates,
        "reminder_validation_candidate_limit": default_settings.retrieval_validation.reminder_llm_validation_max_candidates,
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
        "bm25_top_k": RETRIEVAL_PIPELINE_POLICY.source_top_k,
        "chroma_top_k": RETRIEVAL_PIPELINE_POLICY.source_top_k,
        "rrf_k": 40,
        "reranker_top_k": RETRIEVAL_PIPELINE_POLICY.rrf_top_k,
        "reranker_min_score": 0.30,
        "conversation_min_confidence_score": 0.50,
        "reranker_batch_size": 24,
        "final_context_top_k": RETRIEVAL_PIPELINE_POLICY.final_top_k,
        "last_qa_min_confidence": 0.80,
        "last_qa_clarification_merge_min_confidence": 0.84,
        "last_qa_skip_broad_retrieval_min_confidence": 0.90,
        "support_question_resolution_min_confidence": 0.90,
        "conversation_followup_min_score": 0.65,
        "action_min_confidence": 0.76,
        "knowledge_validation_candidate_limit": 5,
        "reminder_validation_candidate_limit": 3,
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
    task_capacity = {
        LLMTask.QUERY_REWRITE: (1024, 128),
        LLMTask.LAST_QA: (2048, 128),
        LLMTask.INTENT: (1536, 64),
        LLMTask.ACTION_EXTRACTION: (1536, 160),
        LLMTask.KNOWLEDGE_ACTION_EXTRACTION: (8192, 2048),
        LLMTask.KNOWLEDGE_ACTION_VALIDATION: (8192, 1024),
        LLMTask.KNOWLEDGE_CONTENT_FINALIZATION: (12288, 2048),
        LLMTask.REMINDER_ACTION_EXTRACTION: (8192, 2048),
        LLMTask.REMINDER_ACTION_VALIDATION: (12288, 1024),
        LLMTask.REMINDER_CONTENT_FINALIZATION: (4096, 128),
        LLMTask.GENERATE_CLARIFICATION: (1536, 160),
        LLMTask.GENERATE_HUMAN_SUPPORTING: (2048, 160),
        LLMTask.CLARIFICATION_MERGE: (1536, 256),
        LLMTask.ANSWER: (4096, 1024),
        LLMTask.WRITING: (4096, 1024),
        LLMTask.RISKY_ACTION: (1024, 192),
        LLMTask.RETRIEVAL_VALIDATION: (4096, 512),
        LLMTask.GENERAL_SUB_BRANCH_DETECTION: (1024, 96),
        LLMTask.CONTENT_COMPOSER_REACT: (768, 160),
        LLMTask.ACTION_PLANNING: (2048, 256),
    }
    for task, expected in task_capacity.items():
        decision = router.decision_for_task(task)
        if (decision.num_ctx, decision.num_predict) != expected:
            raise AssertionError(
                f"{task.value} capacity mismatch: "
                f"{(decision.num_ctx, decision.num_predict)}"
            )
    if default_settings.ollama.model_intent != FAST_LLM_MODEL:
        raise AssertionError("intent must use the validated semantic routing model")
    if default_settings.ollama.model_intent_fallback != FAST_LLM_MODEL:
        raise AssertionError("intent must have a non-ONNX recovery model")
    if default_settings.ollama.preload_onnx_models:
        raise AssertionError("ONNX model preloading must remain opt-in")

    expected_timeouts = {
        LLMTask.QUERY_REWRITE: 12.0,
        LLMTask.LAST_QA: 30.0,
        LLMTask.INTENT: 20.0,
        LLMTask.ACTION_EXTRACTION: 24.0,
        LLMTask.KNOWLEDGE_ACTION_EXTRACTION: 24.0,
        LLMTask.KNOWLEDGE_ACTION_VALIDATION: 45.0,
        LLMTask.KNOWLEDGE_CONTENT_FINALIZATION: 90.0,
        LLMTask.REMINDER_ACTION_EXTRACTION: 36.0,
        LLMTask.REMINDER_ACTION_VALIDATION: 45.0,
        LLMTask.REMINDER_CONTENT_FINALIZATION: 90.0,
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
        intent_value = str(
            payload.get("intent") or Intent.GENERAL_RESPONSE.value
        )
        payload = {
            "intent": intent_value,
            "confidence": payload.get("confidence", 0.0),
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
    final_intent_wins, _ = classify(
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
        "final intent ignores an obsolete conflicting alias": final_intent_wins,
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
        "final intent ignores an obsolete conflicting alias": Intent.REMINDER,
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
        "Choose one final branch name directly",
        "knowledge_facts is mutation-only",
        "reminder is mutation-only",
        "Every request to search, find, list, show, inspect, look up, retrieve, recall, read, or answer",
        "a new request is never clarification.",
    )
    missing_rules = [rule for rule in required_prompt_rules if rule not in prompt]
    if missing_rules:
        raise AssertionError(f"intent prompt lost required routing rules: {missing_rules}")
    if "multi_intent" in DEFAULT_PROMPT_REGISTRY.system("intent_classifier"):
        raise AssertionError("intent prompt must not request unused multi_intent output")
    return ScenarioResult("intent_branch_ownership_policy", True, "knowledge, reminders, general requests, and clarification follow semantic branch-ownership rules")


def scenario_last_qa_relationship_policy(settings: ProductionSettings) -> ScenarioResult:
    """Only state-bound evidence may bypass retrieval; the model gets a minimal decision."""
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
        "relationship": LastQAInteractionType.NORMAL_FOLLOW_UP.value,
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

    class RelationshipScenarioLLM:
        def __init__(self, relationship: str, confidence: float) -> None:
            self.relationship = relationship
            self.confidence = confidence

        def generate_json(self, **kwargs: Any) -> dict[str, Any]:
            properties = kwargs.get("schema", {}).get("properties", {})
            if set(properties) != {
                "relationship",
                "matched_question_index",
                "confidence",
            }:
                raise AssertionError(
                    f"Last-QA relationship schema regressed: {set(properties)}"
                )
            if "latest_exchange" not in kwargs.get("user_prompt", ""):
                raise AssertionError("Last-QA model did not receive the latest exchange")
            return {
                "relationship": self.relationship,
                "matched_question_index": -1,
                "confidence": self.confidence,
            }

    latest_resolution = LLMLastQAResolver(
        llm=RelationshipScenarioLLM("normal_follow_up", 0.96),
        config=config,
        prompt_registry=DEFAULT_PROMPT_REGISTRY,
    ).resolve(
        ChatRequest(user_id=DEBUG_USER, raw_query="Explain why that is required."),
        "Explain why that is required.",
        state,
    )
    if (
        latest_resolution.path is not LastQAPath.LATEST_CONTEXT_INTERACTION
        or latest_resolution.interaction_type
        is not LastQAInteractionType.NORMAL_FOLLOW_UP
        or not latest_resolution.skip_broad_retrieval
    ):
        raise AssertionError(
            f"clear latest-context follow-up was underrated: {latest_resolution!r}"
        )

    unrelated_resolution = LLMLastQAResolver(
        llm=RelationshipScenarioLLM("unrelated_or_uncertain", 0.99),
        config=config,
        prompt_registry=DEFAULT_PROMPT_REGISTRY,
    ).resolve(
        ChatRequest(user_id=DEBUG_USER, raw_query="Explain photosynthesis."),
        "Explain photosynthesis.",
        state,
    )
    if unrelated_resolution.path is not LastQAPath.BROAD_RETRIEVAL_REQUIRED:
        raise AssertionError(
            f"unrelated request incorrectly reused latest context: {unrelated_resolution!r}"
        )

    prompt = DEFAULT_PROMPT_REGISTRY.system("last_qa")
    required_rules = (
        "with relationship and confidence",
        "also return matched_question_index",
        "When matched_question_index is present, use -1",
        "Do not classify source-verified reminder-notification replies, clarification answers, or outbound-message actions",
        "topical similarity without dependency",
        "normal_follow_up",
    )
    missing = [rule for rule in required_rules if rule not in prompt]
    if missing:
        raise AssertionError(f"Last-QA prompt lost relationship safeguards: {missing}")
    clarification_prompt = DEFAULT_PROMPT_REGISTRY.system("clarification_merge")
    standalone_rule = "A latest message that starts a distinct standalone request is not a clarification answer"
    if standalone_rule not in clarification_prompt:
        raise AssertionError("clarification merge lost its standalone-request safeguard")
    return ScenarioResult(
        "last_qa_relationship_policy",
        True,
        "Last-QA recognizes high-confidence latest-exchange continuity while keeping unrelated or weak evidence on broad retrieval",
    )


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
        "onnx_loaded_models: []",
        "onnx_routes: []",
        "answer=qwen3.5:9b",
        "last_qa=qwen3.5:9b",
        "query_rewrite=qwen3.5:4b",
        "synthetic ONNX compatibility error",
        '"model": "qwen3.5:9b"',
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


def scenario_content_composer_deterministic(settings: ProductionSettings) -> ScenarioResult:
    state = build_scenario_state(settings)
    with TemporaryDirectory(prefix="assistant-debug-artifacts-") as artifact_storage_dir:
        config = GeneralPurposeConfig(
            content_composer_enabled=True,
            content_composer_fallback_tool="answer_generation",
            content_composer_default_tool="answer_generation",
            artifact_storage_dir=artifact_storage_dir,
        )
        llm = ScenarioLLM()
        registry = ContentToolRegistry(
            tools=[AnswerGenerationTool(llm=llm, prompt_registry=DEFAULT_PROMPT_REGISTRY)],
            config=config,
        )
        composer = DeterministicContentComposer(
            registry=registry,
        )
        from assistant_rag.contracts import (
            ContentComposerInput,
            GeneralSubBranch,
            PersistenceMode,
            SubBranchPromptContext,
        )

        composer_input = ContentComposerInput(
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
        )
        result = composer.compose(composer_input, config)
        if not result.final_response_text:
            raise AssertionError("content composer returned an empty answer")

        artifact_registry = ContentToolRegistry(
            tools=[
                AnswerGenerationTool(llm=llm, prompt_registry=DEFAULT_PROMPT_REGISTRY),
                GenerateExcelTool(llm=llm, prompt_registry=DEFAULT_PROMPT_REGISTRY, config=config),
            ],
            config=config,
        )
        artifact_composer = DeterministicContentComposer(
            registry=artifact_registry,
        )
        artifact_result = artifact_composer.compose(
            replace(
                composer_input,
                raw_user_query="Give me an Excel file to track project tasks.",
                rewritten_query="Give me an Excel file to track project tasks.",
                metadata={
                    "semantic_action_decision": _scenario_semantic_payload(
                        "Give me an Excel file to track project tasks.",
                        file_type="xlsx",
                        file_evidence="Give me",
                        file_type_evidence="Excel file",
                    )
                },
                repository=state.repository,
            ),
            config,
        )
        if artifact_result.used_tool_names != (
            "answer_generation",
            "generate_excel",
        ) or len(artifact_result.artifacts) != 1:
            raise AssertionError(
                "file-only Excel request did not execute answer generation followed "
                f"by exactly one artifact tool: {artifact_result!r}"
            )
        artifact_path = Path(str(artifact_result.artifacts[0].get("storage_path") or ""))
        if not artifact_path.is_file() or not zipfile.is_zipfile(artifact_path):
            raise AssertionError("generated Excel artifact was not a downloadable workbook")
    return ScenarioResult("content_composer_deterministic", True, "every general-purpose request runs answer generation before an optional downloadable Microsoft artifact")


def scenario_general_broad_retrieval_approved(settings: ProductionSettings) -> ScenarioResult:
    state = build_scenario_state(settings)
    topic_id, parent_hop_id = seed_conversation(
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
    branch = state.pipeline.router.branches[Intent.GENERAL_RESPONSE]
    branch.sub_branch_detector = GeneralSubBranchDetector(
        llm=ScenarioLLM(),
        prompt_registry=DEFAULT_PROMPT_REGISTRY,
    )
    branch.general_purpose_config = GeneralPurposeConfig(
        general_sub_branch_detector_enabled=True,
        content_composer_enabled=False,
        hitl_supporting_question_enabled=False,
    )
    response = run_request(
        state,
        "Atlas context QA signoff",
        metadata={
            "intent": Intent.GENERAL_RESPONSE.value,
            "normal_response_text": "Continuing from the approved Atlas context.",
        },
    )
    assert_response(response, ResponseType.NORMAL, "approved Atlas")
    if state.retriever.conversation_calls < 1:
        raise AssertionError("broad conversation retrieval did not run")
    if response.conversation_topic_id != topic_id:
        raise AssertionError(
            "approved broad retrieval did not append to the selected topic: "
            f"expected={topic_id!r}, actual={response.conversation_topic_id!r}"
        )
    persisted_hop = state.repository.connection.execute(
        """
        SELECT topic_id, previous_hop_id, parent_hop_id, entities_json
        FROM conversation_hops
        WHERE hop_id = ? AND user_id = ?
        """,
        (response.conversation_hop_id, state.user_id),
    ).fetchone()
    if persisted_hop is None:
        raise AssertionError("general follow-up response hop was not persisted")
    persisted_entities = json.loads(str(persisted_hop["entities_json"]))
    expected_persistence = {
        "topic_id": topic_id,
        "previous_hop_id": parent_hop_id,
        "parent_hop_id": parent_hop_id,
        "sub_branch": "conversation_follow_up",
    }
    actual_persistence = {
        "topic_id": persisted_hop["topic_id"],
        "previous_hop_id": persisted_hop["previous_hop_id"],
        "parent_hop_id": persisted_hop["parent_hop_id"],
        "sub_branch": persisted_entities.get("sub_branch"),
    }
    if actual_persistence != expected_persistence:
        raise AssertionError(
            "approved broad retrieval did not preserve append/topic/parent/sub-branch persistence: "
            f"expected={expected_persistence!r}, actual={actual_persistence!r}"
        )
    return ScenarioResult(
        "general_broad_retrieval_approved",
        True,
        "real deterministic detector appended the follow-up to the approved topic and parent hop",
    )


def scenario_general_broad_retrieval_below_followup_gate(
    settings: ProductionSettings,
) -> ScenarioResult:
    """A retrievable but weak hop must not own follow-up persistence."""
    state = build_scenario_state(settings)
    old_topic_id, _ = seed_conversation(
        state,
        title="Weak Atlas Context",
        query="Project Atlas context",
        response="Atlas deployment requires a freeze window and QA signoff.",
    )
    response = run_request(
        state,
        # Deterministic debug score: 3 matching terms / 5 query terms = 0.60.
        # This passes retrieval's 0.50 floor but fails the 0.65 follow-up gate.
        "Continue Atlas context signoff today",
        metadata={
            "intent": Intent.GENERAL_RESPONSE.value,
            "normal_response_text": "Starting a fresh discussion.",
        },
    )
    assert_response(response, ResponseType.NORMAL, "fresh discussion")
    if state.retriever.conversation_calls != 1:
        raise AssertionError("weak-hop scenario did not run broad retrieval exactly once")
    if response.conversation_topic_id == old_topic_id:
        raise AssertionError("a 0.60 hop incorrectly owned follow-up persistence")
    persisted = state.repository.connection.execute(
        """
        SELECT previous_hop_id, parent_hop_id, entities_json
        FROM conversation_hops
        WHERE hop_id = ? AND user_id = ?
        """,
        (response.conversation_hop_id, state.user_id),
    ).fetchone()
    if persisted is None:
        raise AssertionError("weak-hop new-topic response was not persisted")
    entities = json.loads(str(persisted["entities_json"]))
    if (
        persisted["previous_hop_id"] is not None
        or persisted["parent_hop_id"] is not None
        or entities.get("sub_branch") != "new_conversation_topic"
    ):
        raise AssertionError(
            "below-threshold retrieval did not produce isolated new-topic persistence"
        )
    return ScenarioResult(
        "general_broad_retrieval_below_followup_gate",
        True,
        "a 0.60 retrieved hop passed retrieval but safely fell back to a new topic",
    )


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


def scenario_lastqa_support_below_resolution_gate(
    settings: ProductionSettings,
) -> ScenarioResult:
    """A stale 0.80 support match must not append to the linked topic."""
    state = build_scenario_state(settings)
    old_topic_id, old_hop_id = seed_conversation(
        state,
        title="Stale Supporting Context",
        query="Prepare the report",
        response="A draft is ready.",
    )
    state.pipeline.last_qa_store.save(
        state.user_id,
        LastQAState(
            last_user_query="Prepare the report",
            last_response="A draft is ready.",
            response_type=ResponseType.NORMAL,
            supporting_questions=[
                GeneratedQuestion(
                    text="Report format should use now",
                    source=QuestionSource.HUMAN_SUPPORTING_QUESTION,
                    purpose="optional_context",
                    confidence=1.0,
                )
            ],
            linked_topic_id=old_topic_id,
            linked_hop_id=old_hop_id,
        ),
    )
    before_calls = state.retriever.conversation_calls
    response = run_request(
        state,
        # Deterministic Last-QA overlap: 4/5 = 0.80. The resolver recognizes
        # it, but the sub-branch's stricter 0.90 ownership gate rejects it.
        "Report format should use today",
        metadata={
            "intent": Intent.GENERAL_RESPONSE.value,
            "normal_response_text": "Treating this as a fresh request.",
        },
    )
    assert_response(response, ResponseType.NORMAL, "fresh request")
    if state.retriever.conversation_calls != before_calls:
        raise AssertionError("authoritative Last-QA support path unexpectedly retrieved")
    if response.conversation_topic_id == old_topic_id:
        raise AssertionError("a 0.80 stale support match incorrectly appended to its old topic")
    persisted = state.repository.connection.execute(
        """
        SELECT previous_hop_id, parent_hop_id, entities_json
        FROM conversation_hops
        WHERE hop_id = ? AND user_id = ?
        """,
        (response.conversation_hop_id, state.user_id),
    ).fetchone()
    if persisted is None:
        raise AssertionError("stale-support new-topic response was not persisted")
    entities = json.loads(str(persisted["entities_json"]))
    if (
        persisted["previous_hop_id"] is not None
        or persisted["parent_hop_id"] is not None
        or entities.get("sub_branch") != "new_conversation_topic"
    ):
        raise AssertionError(
            "below-threshold support match did not produce isolated new-topic persistence"
        )
    return ScenarioResult(
        "lastqa_support_below_resolution_gate",
        True,
        "a 0.80 supporting-question match could not reuse stale topic ownership",
    )


def scenario_knowledge_add(settings: ProductionSettings) -> ScenarioResult:
    state = build_scenario_state(settings)
    response = run_request(
        state,
        "Remember Atlas retention is 30 days",
        metadata={
            "intent": Intent.KNOWLEDGE_FACTS.value,
            "topic_title": "Project Atlas",
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
    chunk = state.repository.connection.execute(
        """
        SELECT chunk_id, knowledge_topic_id, raw_text, is_deleted
        FROM knowledge_chunks WHERE user_id = ?
        """,
        (state.user_id,),
    ).fetchone()
    topic = state.repository.connection.execute(
        """
        SELECT title, version FROM knowledge_topics
        WHERE user_id = ? AND knowledge_topic_id = ?
        """,
        (state.user_id, chunk["knowledge_topic_id"]),
    ).fetchone()
    if (
        chunk is None
        or topic is None
        or str(topic["title"]) != "Project Atlas"
        or int(topic["version"]) != 2
        or bool(chunk["is_deleted"])
    ):
        raise AssertionError("knowledge add did not update its exact SQL topic/chunk")
    assert_knowledge_audit_binding(
        state,
        response,
        action="add",
        knowledge_topic_id=str(chunk["knowledge_topic_id"]),
        knowledge_chunk_id=str(chunk["chunk_id"]),
    )
    for store_name, store in (
        ("OpenSearch", state.retriever.bm25),
        ("ChromaDB", state.retriever.chroma),
    ):
        indexed = store.get_document(entity_id=str(chunk["chunk_id"]))
        if indexed is None or indexed.get("entity_type") != "knowledge_chunk":
            raise AssertionError(f"knowledge add was not synchronized to {store_name}")
    entities = state.repository.list_all_outbox_entities()
    for entity_type, entity_id in entities:
        state.repository.load_outbox_entity(entity_type=entity_type, entity_id=entity_id)
    return ScenarioResult("knowledge_add", True, "knowledge add and outbox entity loading succeeded")


def scenario_knowledge_modify(settings: ProductionSettings) -> ScenarioResult:
    state = build_scenario_state(settings)
    original_chunk_id = seed_knowledge(
        state,
        title="Project Atlas",
        text="Project Atlas retention is 30 days.",
    )
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
    assert_response(response, ResponseType.KNOWLEDGE_ACTION)
    if response.actions_pending_confirmation:
        raise AssertionError("knowledge modify PASS created a confirmation gate")
    active = state.repository.connection.execute(
        """
        SELECT chunk_id, knowledge_topic_id, raw_text, replaces_chunk_id
        FROM knowledge_chunks WHERE user_id = ? AND is_deleted = 0
        """,
        (state.user_id,),
    ).fetchall()
    if len(active) != 1 or active[0]["raw_text"] != "Project Atlas retention is 45 days.":
        raise AssertionError("knowledge modify PASS did not commit the finalized chunk")
    replacement = active[0]
    original = state.repository.connection.execute(
        """
        SELECT knowledge_topic_id, is_deleted, replaced_by_chunk_id
        FROM knowledge_chunks WHERE user_id = ? AND chunk_id = ?
        """,
        (state.user_id, original_chunk_id),
    ).fetchone()
    if (
        original is None
        or not bool(original["is_deleted"])
        or str(original["knowledge_topic_id"])
        != str(replacement["knowledge_topic_id"])
        or str(original["replaced_by_chunk_id"]) != str(replacement["chunk_id"])
        or str(replacement["replaces_chunk_id"]) != original_chunk_id
        or state.repository.table_count("knowledge_topics") != 1
    ):
        raise AssertionError("knowledge modify changed topic identity or version linkage")
    assert_knowledge_audit_binding(
        state,
        response,
        action="modify",
        knowledge_topic_id=str(replacement["knowledge_topic_id"]),
        knowledge_chunk_id=str(replacement["chunk_id"]),
    )
    for store_name, store in (
        ("OpenSearch", state.retriever.bm25),
        ("ChromaDB", state.retriever.chroma),
    ):
        if store.get_document(entity_id=original_chunk_id) is not None:
            raise AssertionError(f"modified knowledge remained active in {store_name}")
        if store.get_document(entity_id=str(replacement["chunk_id"])) is None:
            raise AssertionError(f"modified knowledge was not synchronized to {store_name}")
    return ScenarioResult(
        "knowledge_modify",
        True,
        "knowledge modify PASS committed immediately after finalization",
    )


def scenario_knowledge_delete(settings: ProductionSettings) -> ScenarioResult:
    state = build_scenario_state(settings)
    chunk_id = seed_knowledge(
        state,
        title="Project Atlas",
        text="Project Atlas retention is 30 days.",
    )
    before = state.repository.connection.execute(
        """
        SELECT knowledge_topic_id FROM knowledge_chunks
        WHERE user_id = ? AND chunk_id = ?
        """,
        (state.user_id, chunk_id),
    ).fetchone()
    response = run_request(
        state,
        "Delete Project Atlas retention is 30 days",
        metadata={"intent": Intent.KNOWLEDGE_FACTS.value},
    )
    assert_response(response, ResponseType.KNOWLEDGE_ACTION, "Deleted knowledge")
    deleted = state.repository.connection.execute(
        """
        SELECT knowledge_topic_id, is_deleted FROM knowledge_chunks
        WHERE user_id = ? AND chunk_id = ?
        """,
        (state.user_id, chunk_id),
    ).fetchone()
    if (
        before is None
        or deleted is None
        or not bool(deleted["is_deleted"])
        or str(deleted["knowledge_topic_id"]) != str(before["knowledge_topic_id"])
        or state.repository.table_count("knowledge_topics") != 1
    ):
        raise AssertionError("knowledge delete did not update its exact SQL topic/chunk")
    assert_knowledge_audit_binding(
        state,
        response,
        action="delete",
        knowledge_topic_id=str(deleted["knowledge_topic_id"]),
        knowledge_chunk_id=chunk_id,
    )
    for store_name, store in (
        ("OpenSearch", state.retriever.bm25),
        ("ChromaDB", state.retriever.chroma),
    ):
        if store.get_document(entity_id=chunk_id) is not None:
            raise AssertionError(f"deleted knowledge remained active in {store_name}")
    return ScenarioResult(
        "knowledge_delete",
        True,
        "knowledge delete preserved topic identity and removed both derived chunks",
    )


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
    assert_response(
        response,
        ResponseType.CLARIFICATION,
        "Which complete stored knowledge item",
    )
    return ScenarioResult(
        "knowledge_delete_not_found",
        True,
        "validator FAIL returned its immediate target clarification",
    )


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


def scenario_knowledge_three_llm_pipeline(settings: ProductionSettings) -> ScenarioResult:
    state = build_scenario_state(settings)
    original = "Atlas retention is 30 days"
    replacement = "Atlas retention is 45 days"
    prefix = " ".join(f"Unrelated detail {index}." for index in range(500))
    stored = f"{prefix} {original}. Preserve this tail marker."
    final_content = stored.replace(original, replacement, 1)
    chunk_id = seed_knowledge(state, title="Project Atlas", text=stored)
    query = f"Change {original} to {replacement}"

    class StageLLM:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []
            self.responses = [
                {
                    "action": "modify",
                    "text_content": "",
                    "original_text": original,
                    "replacement_text": replacement,
                    "confidence": 0.99,
                },
                {
                    "decision": "PASS",
                    "selected_candidate_keys": [chunk_id],
                    "confidence": 0.99,
                    "clarification_question": "",
                    "candidate_assessments": [
                        {
                            "candidate_key": chunk_id,
                            "confidence": 0.99,
                            "matched_text": original,
                        }
                    ],
                },
                {
                    "final_content": final_content,
                    "confidence": 0.99,
                },
            ]

        def generate_json(self, **kwargs: Any) -> dict[str, Any]:
            self.calls.append(dict(kwargs))
            if not self.responses:
                raise AssertionError("unexpected extra knowledge model call")
            return self.responses.pop(0)

        def chat(self, **_kwargs: Any) -> str:
            raise AssertionError("knowledge mutation stages require JSON")

    llm = StageLLM()
    config = state.pipeline.config
    validator = KnowledgeRetrievalValidationStrategy(
        config=config.retrieval_validation,
        llm=llm,
        prompts=DEFAULT_PROMPT_REGISTRY,
    )
    finalizer = KnowledgeContentFinalizationStrategy(
        llm=llm,
        prompts=DEFAULT_PROMPT_REGISTRY,
        min_confidence=(
            config.retrieval_validation.knowledge_llm_validation_min_confidence
        ),
    )
    state.pipeline.router.branches[Intent.KNOWLEDGE_FACTS] = KnowledgeFactsBranch(
        config=config,
        action_detector=LLMKnowledgeActionDetector(
            llm=llm,
            prompts=DEFAULT_PROMPT_REGISTRY,
            min_confidence=settings.prompt_policy.action_min_confidence,
        ),
        prompt_registry=DEFAULT_PROMPT_REGISTRY,
        knowledge_mutation_pipeline=KnowledgeMutationPipeline(
            retriever=state.retriever,
            config=config,
            validator=validator,
            finalizer=finalizer,
        ),
    )

    # A normal knowledge retrieval would reject this exact deterministic score;
    # the mutation-only bypass must still deliver it to SQL validation.
    state.retriever.min_score = 1.01
    lifecycle = ChatRequestLifecycleExecutor(
        pipeline=state.pipeline,
        repository=state.repository,
    )
    initial = lifecycle.execute(
        ChatRequest(
            user_id=state.user_id,
            raw_query=query,
            metadata={"intent": Intent.KNOWLEDGE_FACTS.value},
            idempotency_key="knowledge-three-llm-initial",
        ),
        fallback_request_id="knowledge-three-llm-initial",
    )
    if initial.response is None:
        raise AssertionError("initial knowledge mutation unexpectedly replayed")
    assert_response(initial.response, ResponseType.KNOWLEDGE_ACTION)
    if initial.response.actions_pending_confirmation:
        raise AssertionError("validated modify created a forbidden confirmation")
    expected_tasks = [
        LLMTask.KNOWLEDGE_ACTION_EXTRACTION,
        LLMTask.KNOWLEDGE_ACTION_VALIDATION,
        LLMTask.KNOWLEDGE_CONTENT_FINALIZATION,
    ]
    if [call["task"] for call in llm.calls] != expected_tasks:
        raise AssertionError("knowledge model stages did not execute exactly once in order")
    prompt_payloads = [
        json.loads(str(call["user_prompt"]).removeprefix("Runtime context:\n"))
        for call in llm.calls
    ]
    canonical_history = prompt_payloads[0]["chat_history"]
    for payload in (prompt_payloads[0], prompt_payloads[2]):
        if (
            payload.get("rewritten_query") != query
            or payload["chat_history"] != canonical_history
            or "raw_query" in payload
        ):
            raise AssertionError("knowledge extraction/finalization lost query/history")
    if set(prompt_payloads[1]) != {
        "first_model_response",
        "knowledge_retrieval",
    }:
        raise AssertionError("knowledge validation prompt was not strictly isolated")
    if query in json.dumps(prompt_payloads[1]) or "chat_history" in prompt_payloads[1]:
        raise AssertionError("knowledge validation received forbidden query/history")
    if "Preserve this tail marker." not in json.dumps(prompt_payloads[1:]):
        raise AssertionError("long SQL chunk tail was truncated before validation/finalization")
    active = state.repository.connection.execute(
        "SELECT raw_text FROM knowledge_chunks WHERE user_id = ? AND is_deleted = 0",
        (state.user_id,),
    ).fetchall()
    if [row["raw_text"] for row in active] != [final_content]:
        raise AssertionError("direct modify did not preserve the full finalized chunk")
    return ScenarioResult(
        "knowledge_three_llm_pipeline",
        True,
        "isolated validation and direct post-finalization commit passed",
    )


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
        "Modify the Submit tax form reminder: change the subject to Submit tax form final and the notification time to 2026-07-11T10:00:00+00:00",
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
        "Turn off the Call finance reminder",
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
        "Turn on the Renew license reminder",
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
        "Delete the Legacy review reminder",
        metadata={
            "intent": Intent.REMINDER.value,
            "reminder_actions": [
                {"action": "delete", "target_description": "Legacy review"}
            ],
        },
    )
    assert_response(response, ResponseType.REMINDER_ACTION, "dismissed")
    if response.actions_pending_confirmation:
        raise AssertionError("validated reminder delete reintroduced a confirmation stage")
    row = state.repository.list_reminders(user_id=state.user_id)[0]
    if row["status"] != "dismissed":
        raise AssertionError("validated delete did not dismiss the reminder")
    return ScenarioResult(
        "reminder_delete",
        True,
        "validated delete dismissed the reminder without a removed confirmation stage",
    )


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
    return ScenarioResult(
        "reminder_missing_time",
        True,
        "model 2 returned the immediate missing-time clarification",
    )


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
    scenario_general_hitl_disabled_does_not_fallback,
    scenario_user_entrypoint_runtime_parity,
    scenario_streamlit_terminal_failure_containment,
    scenario_platform_gmail_multi_recipient_delivery,
    scenario_general_new_conversation_skips_sub_branch_llm,
    scenario_answer_adaptive_token_budget,
    scenario_llm_answer_generation_trace,
    scenario_structured_clarification_fallback_policy,
    scenario_mutation_clarification_fast_path,
    scenario_last_qa_mandatory_before_classification,
    scenario_llm_first_action_extraction,
    scenario_llm_extraction_owns_single_action,
    scenario_read_only_knowledge_general_route,
    scenario_general_sql_knowledge_fallback,
    scenario_read_only_reminder_general_route,
    scenario_llm_first_knowledge_update,
    scenario_shared_confirmation_lifecycle,
    scenario_clarification_schema_echo_recovery,
    scenario_content_composer_react_structured_policy,
    scenario_structured_fallback_terminal_quiet,
    scenario_model_routing_policy,
    scenario_intent_branch_ownership_policy,
    scenario_last_qa_relationship_policy,
    scenario_debug_hybrid_llm_compatibility,
    scenario_hybrid_structured_onnx_failover,
    scenario_onnx_non_retryable_model_error,
    scenario_content_composer_deterministic,
    scenario_general_broad_retrieval_approved,
    scenario_general_broad_retrieval_below_followup_gate,
    scenario_lastqa_supporting_skip,
    scenario_lastqa_support_below_resolution_gate,
    scenario_knowledge_add,
    scenario_knowledge_modify,
    scenario_knowledge_delete,
    scenario_knowledge_delete_not_found,
    scenario_knowledge_modify_missing_replacement,
    scenario_knowledge_three_llm_pipeline,
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


def run_llm_smoke_test(settings: ProductionSettings) -> int:
    """Exercise the configured answer model from the fully warmed production runtime."""

    runtime = build_production_runtime(settings)
    llm = _debug_llm_from_pipeline(runtime.pipeline)
    if llm is None:
        print("LLM smoke test failed: production runtime has no routed LLM", file=sys.stderr)
        return 1

    request_id = new_request_id()
    start_trace(request_id)
    try:
        response = llm.chat(
            task=LLMTask.ANSWER,
            system_prompt="You are a concise compatibility smoke test.",
            user_prompt="Reply with the exact words: Qwen two-model smoke test passed.",
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
    runtime = build_production_runtime(settings)
    pipeline = runtime.pipeline
    repository = runtime.repository
    llm = _debug_llm_from_pipeline(pipeline)
    print_runtime_architecture(pipeline, repository)

    try:
        catch_up = runtime.reminder_autoscan.catch_up_due(
            now_value=datetime.now(timezone.utc).isoformat(),
            batch_size=100,
        )
        print(f"Startup reminder catch-up: {catch_up}")
    except Exception as exc:
        logger.exception("debug reminder autoscan catch-up failed")
        print(f"Startup reminder catch-up unavailable: {type(exc).__name__}", file=sys.stderr)

    user_id = input("User ID [default: default_user]: ").strip() or "default_user"
    gmail_username = input("Gmail username [default: '']: ").strip()
    gmail_app_password = getpass("Gmail app password [default: '']: ")
    lifecycle = ChatRequestLifecycleExecutor(
        pipeline=pipeline,
        repository=repository,
    )
    pending_confirmation_queries: dict[str, str] = {}

    print(
        "\nAssistant is ready! Type '/exit' or Ctrl+C to quit. "
        "Use '/confirm [token]' for a pending change."
    )
    while True:
        try:
            query = input("\nAsk the assistant: ").strip()
            if not query:
                continue
            if query.lower() == "/exit":
                print("Exiting...")
                break

            confirmation_token: str | None = None
            request_query = query
            if query.casefold().startswith("/confirm"):
                parts = query.split(maxsplit=1)
                if len(parts) == 2 and parts[1].strip():
                    confirmation_token = parts[1].strip()
                elif pending_confirmation_queries:
                    confirmation_token = next(
                        reversed(pending_confirmation_queries)
                    )
                else:
                    print("No pending confirmation token is available.")
                    continue
                request_query = pending_confirmation_queries.get(
                    confirmation_token,
                    "Confirm the pending action.",
                )

            idempotency_key = (
                f"debug-confirm:{confirmation_token}"
                if confirmation_token
                else new_request_id()
            )
            request_id = new_request_id(idempotency_key)
            start_trace(request_id)
            execution = lifecycle.execute(
                ChatRequest(
                    user_id=user_id,
                    raw_query=request_query,
                    confirmation_token=confirmation_token,
                    idempotency_key=idempotency_key,
                    platform_context={
                        "gmail_username": gmail_username,
                        "gmail_app_password": gmail_app_password,
                    },
                ),
                fallback_request_id=request_id,
            )
            if execution.replayed:
                if confirmation_token:
                    pending_confirmation_queries.pop(confirmation_token, None)
                print("\nAssistant (idempotent replay):")
                print(execution.payload.get("final_chat_text", "Request already completed."))
                continue
            response = execution.response
            if response is None:
                raise RuntimeError("Fresh lifecycle execution returned no response")
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
            if confirmation_token and response.response_type is not ResponseType.ERROR:
                pending_confirmation_queries.pop(confirmation_token, None)
            for confirmation in response.actions_pending_confirmation:
                token = str(confirmation.get("confirmation_token") or "").strip()
                if not token:
                    continue
                pending_confirmation_queries[token] = request_query
                print(f"Pending confirmation token: {token}")
                print(f"Run /confirm {token} to apply this change exactly once.")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except RequestLifecycleConflict as exc:
            print(f"\nRequest not executed: {exc}", file=sys.stderr)
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
