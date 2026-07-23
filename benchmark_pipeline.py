#!/usr/bin/env python3
"""
pipeline_benchmark_v1
=====================
A standalone, zero-mock benchmark that drives the **real production pipeline**
for a batch of user queries and emits one structured JSON object per query as
JSON Lines (JSONL) to stdout, with diagnostic logs going to stderr.

Usage
-----
    python benchmark_pipeline.py --input benchmark_queries.json

Options
-------
    --input PATH            JSON file with query list or object (required)
    --user-id ID            Override benchmark user ID (default: benchmark-user)
    --output PATH           Write JSONL output to file instead of stdout
    --continue-on-error     Emit error records instead of aborting on failure
    --warmup                Run one silent warmup query before the measured batch
    --text-preview-chars N  Max chars for knowledge text preview (default: 200)

Input formats
-------------
    ["First query", "Second query"]

    {"user_id": "benchmark-user", "queries": ["First query", "Second query"]}
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import statistics
import sys
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

# ---------------------------------------------------------------------------
# Logging - diagnostics to stderr only; stdout is reserved for JSON records
# ---------------------------------------------------------------------------
_handler = logging.StreamHandler(sys.stderr)
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
logging.basicConfig(handlers=[_handler], level=logging.INFO)
logger = logging.getLogger("benchmark_pipeline")

BENCHMARK_VERSION = "pipeline_benchmark_v1"
DEFAULT_USER_ID = "benchmark-user"
DEFAULT_TEXT_PREVIEW_CHARS = 200


# ---------------------------------------------------------------------------
# Safe JSON serializer - handles dataclasses, enums, tuples, datetimes
# ---------------------------------------------------------------------------

def _safe_default(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if hasattr(obj, "value"):  # Enum
        return obj.value
    if isinstance(obj, tuple):
        return list(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, set):
        return sorted(obj)
    return str(obj)


def _to_json(obj: Any) -> str:
    return json.dumps(obj, default=_safe_default, ensure_ascii=False, separators=(",", ":"))


def _emit_record(record: dict[str, Any], output_file: Any) -> None:
    line = _to_json(record) + "\n"
    output_file.write(line)
    output_file.flush()


# ---------------------------------------------------------------------------
# Tiny percentile helper - no scipy/numpy required
# ---------------------------------------------------------------------------

def _percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(sorted_data) - 1)
    return sorted_data[lo] + (sorted_data[hi] - sorted_data[lo]) * (k - lo)


def _safe_float(value: Any) -> float:
    try:
        v = float(value)
        return v if math.isfinite(v) else 0.0
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# Per-request observation bag
# ---------------------------------------------------------------------------

@dataclass
class RequestObservation:
    """Mutable bag populated by observer hooks during one pipeline call."""

    # Intent: populated by the wrapped classifier
    intent_branch: str | None = None
    intent_confidence: float | None = None
    intent_reason_summary: str | None = None

    # Conversation retrieval
    conv_retrieval_executed: bool = False
    conv_retrieval_skipped_reason: str | None = None
    conv_retrieval_raw_count: int = 0
    conv_retrieval_results: list[dict[str, Any]] = field(default_factory=list)

    # Knowledge retrieval
    knowledge_retrieval_executed: bool = False
    knowledge_retrieval_skipped_reason: str | None = None
    knowledge_query_used: str | None = None
    knowledge_raw_count: int = 0
    knowledge_results: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Observer proxies
# ---------------------------------------------------------------------------

class _ObservingIntentClassifier:
    """
    Wraps OllamaIntentClassifier to extract the per-request LLM confidence
    score from the pipeline's own intent schema payload.

    The wrapper monkey-patches llm.generate_json for the single INTENT task
    call, captures the JSON payload once, then immediately restores the
    original method. It never alters the returned intent or any LLM state.
    """

    def __init__(self, real_classifier: Any, observation: RequestObservation) -> None:
        self._real = real_classifier
        self._obs = observation

    def classify(self, request: Any, rewritten_query: str, **kwargs: Any) -> Any:
        captured_payload: dict[str, Any] = {}

        llm = getattr(self._real, "llm", None)
        if llm is not None:
            original_generate_json = llm.generate_json

            def _intercepting_generate_json(task: Any, **call_kwargs: Any) -> Any:
                result = original_generate_json(task=task, **call_kwargs)
                try:
                    from assistant_rag.llm import LLMTask
                    if task is LLMTask.INTENT and isinstance(result, dict):
                        captured_payload.update(result)
                except Exception:
                    pass
                return result

            llm.generate_json = _intercepting_generate_json
            try:
                intent = self._real.classify(request, rewritten_query, **kwargs)
            finally:
                llm.generate_json = original_generate_json
        else:
            intent = self._real.classify(request, rewritten_query, **kwargs)

        self._obs.intent_branch = intent.value if hasattr(intent, "value") else str(intent)
        raw_confidence = captured_payload.get("confidence")
        if raw_confidence is not None:
            try:
                c = float(raw_confidence)
                self._obs.intent_confidence = c if math.isfinite(c) else None
            except (TypeError, ValueError):
                pass
        self._obs.intent_reason_summary = None
        return intent


class _ObservingRetriever:
    """
    Wraps HybridRetriever.retrieve_knowledge to capture the query used,
    raw candidate count, and returned results.
    retrieve_conversation is passed through unchanged.
    """

    def __init__(self, real_retriever: Any, observation: RequestObservation, preview_chars: int) -> None:
        self._real = real_retriever
        self._obs = observation
        self._preview_chars = preview_chars

    def retrieve_knowledge(self, *, user_id: str, query: str, **kwargs: Any) -> Any:
        self._obs.knowledge_retrieval_executed = True
        self._obs.knowledge_query_used = query
        results = self._real.retrieve_knowledge(user_id=user_id, query=query, **kwargs)
        self._obs.knowledge_raw_count = len(results)
        self._obs.knowledge_results = [
            self._format_knowledge_result(rank, r)
            for rank, r in enumerate(results, start=1)
        ]
        return results

    def retrieve_conversation(self, *, user_id: str, query: str, **kwargs: Any) -> Any:
        return self._real.retrieve_conversation(user_id=user_id, query=query, **kwargs)

    def _format_knowledge_result(self, rank: int, result: Any) -> dict[str, Any]:
        payload: dict[str, Any] = getattr(result, "payload", {}) or {}
        evidence: dict[str, Any] = {}
        raw_evidence = getattr(result, "source_store_evidence", {}) or {}
        # Redact raw text from evidence, keep only structural keys
        for k, v in raw_evidence.items():
            lower_k = k.lower()
            if any(p in lower_k for p in ("text", "content", "body", "document", "raw")):
                continue
            evidence[k] = v

        text_raw = str(payload.get("text") or "").strip()
        preview = text_raw[: self._preview_chars] + ("..." if len(text_raw) > self._preview_chars else "")

        return {
            "rank": rank,
            "entity_id": str(getattr(result, "entity_id", "") or ""),
            "knowledge_topic_id": payload.get("knowledge_topic_id"),
            "source_id": payload.get("source_id"),
            "rerank_score": _safe_float(getattr(result, "rerank_score", 0.0)),
            "confidence": _safe_float(getattr(result, "confidence", 0.0)),
            "validation_status": str(getattr(result, "validation_status", "") or ""),
            "source_store_evidence": evidence,
            "approved_for_context": getattr(result, "validation_status", "") == "sql_validated",
            "text_preview": preview,
        }

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


class _ConversationCapturingRouter:
    """
    Thin wrapper around BranchRouter that inspects the PipelineContext passed
    to route() and extracts conversation retrieval results from it.
    Runs inside the production pipeline - no second retrieval occurs.
    """

    def __init__(self, real_router: Any, observation: RequestObservation) -> None:
        self._real = real_router
        self._obs = observation

    def route(self, context: Any, repository: Any) -> Any:
        conv_results = list(getattr(context, "conversation_results", []) or [])
        conv_ran = bool(getattr(context, "conversation_retrieval", False))

        approved_ctx = getattr(context, "approved_conversation_context", None)
        if approved_ctx is not None:
            conv_ran = conv_ran or bool(getattr(approved_ctx, "conversation_retrieval_ran", False))

        if conv_ran:
            self._obs.conv_retrieval_executed = True
            self._obs.conv_retrieval_raw_count = len(conv_results)
            self._obs.conv_retrieval_results = [
                _format_conv_result(rank, r, approved_ctx)
                for rank, r in enumerate(conv_results, start=1)
            ]
        else:
            self._obs.conv_retrieval_executed = False
            lq_trace = dict(getattr(context, "last_qa_trace", {}) or {})
            skip_reason = (
                lq_trace.get("skip_reason")
                or ("authoritative_last_qa_link" if lq_trace.get("skip_broad_retrieval") else None)
                or "configuration_disabled"
            )
            self._obs.conv_retrieval_skipped_reason = str(skip_reason)

        return self._real.route(context, repository)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


def _format_conv_result(rank: int, result: Any, approved_ctx: Any) -> dict[str, Any]:
    payload: dict[str, Any] = getattr(result, "payload", {}) or {}
    evidence: dict[str, Any] = {}
    raw_evidence = getattr(result, "source_store_evidence", {}) or {}
    for k, v in raw_evidence.items():
        lower_k = k.lower()
        if any(p in lower_k for p in ("embedding", "vector", "text", "content", "body")):
            continue
        evidence[k] = v

    entity_id = str(getattr(result, "entity_id", "") or "")

    # topic_id may be in payload or nested entities_json
    topic_id = payload.get("topic_id")
    if topic_id is None:
        entities_raw = payload.get("entities_json") or {}
        if isinstance(entities_raw, str):
            try:
                entities_raw = json.loads(entities_raw)
            except Exception:
                entities_raw = {}
        topic_id = entities_raw.get("topic_id") if isinstance(entities_raw, dict) else None

    return {
        "rank": rank,
        "entity_id": entity_id,
        "topic_id": topic_id,
        "rerank_score": _safe_float(getattr(result, "rerank_score", 0.0)),
        "confidence": _safe_float(getattr(result, "confidence", 0.0)),
        "validation_status": str(getattr(result, "validation_status", "") or ""),
        "source_store_evidence": evidence,
    }


# ---------------------------------------------------------------------------
# Instrumented pipeline wrapper
# ---------------------------------------------------------------------------

class _InstrumentedPipeline:
    """
    Wraps AssistantPipeline.handle() to inject observing proxies per-request
    and restore all originals in a finally block.
    """

    def __init__(self, real_pipeline: Any, preview_chars: int) -> None:
        self._pipeline = real_pipeline
        self._preview_chars = preview_chars

    def handle_observed(self, request: Any, repository: Any) -> tuple[Any, RequestObservation]:
        """Call handle() with fresh observers; return (response, observation)."""
        obs = RequestObservation()
        pipeline = self._pipeline

        orig_classifier = pipeline.classifier
        orig_retriever = pipeline.retriever
        orig_router = pipeline.router

        obs_classifier = _ObservingIntentClassifier(orig_classifier, obs)
        obs_retriever = _ObservingRetriever(orig_retriever, obs, self._preview_chars)
        obs_router = _ConversationCapturingRouter(orig_router, obs)

        # Patch retriever inside branches that hold a direct reference
        patched_branches: list[tuple[Any, str, Any]] = []
        router_routes = getattr(orig_router, "_routes", None) or {}
        for branch_obj in router_routes.values():
            if hasattr(branch_obj, "retriever"):
                patched_branches.append((branch_obj, "retriever", branch_obj.retriever))
                branch_obj.retriever = obs_retriever
            for sub_attr in ("knowledge_mutation_pipeline", "validated_action_builder"):
                sub = getattr(branch_obj, sub_attr, None)
                if sub is not None and hasattr(sub, "retriever"):
                    patched_branches.append((sub, "retriever", sub.retriever))
                    sub.retriever = obs_retriever

        pipeline.classifier = obs_classifier
        pipeline.retriever = obs_retriever
        pipeline.router = obs_router
        try:
            response = pipeline.handle(request, repository)
        finally:
            pipeline.classifier = orig_classifier
            pipeline.retriever = orig_retriever
            pipeline.router = orig_router
            for obj, attr, orig_val in patched_branches:
                setattr(obj, attr, orig_val)

        return response, obs


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _extract_conv_retrieval(response: Any, obs: RequestObservation) -> dict[str, Any]:
    """
    Build conversation retrieval section.
    Prefers direct observer data; falls back to TraceSummary stages on
    the BundledResponse if the router wrapper did not capture results
    (e.g. broad retrieval was skipped before the router was reached).
    """
    if obs.conv_retrieval_executed or obs.conv_retrieval_skipped_reason:
        return {
            "executed": obs.conv_retrieval_executed,
            "skipped_reason": obs.conv_retrieval_skipped_reason,
            "candidate_count": obs.conv_retrieval_raw_count,
            "selected_count": len(obs.conv_retrieval_results),
            "results": obs.conv_retrieval_results,
        }

    trace = getattr(response, "trace_summary", None)
    if trace is None:
        return {
            "executed": False,
            "skipped_reason": "trace_unavailable",
            "candidate_count": 0,
            "selected_count": 0,
            "results": [],
        }

    stages: list[Any] = list(getattr(trace, "stages", []) or [])
    gate_stage = next(
        (s for s in stages if getattr(s, "stage", "") == "conversation_retrieval_gate"),
        None,
    )
    if gate_stage is None:
        return {
            "executed": False,
            "skipped_reason": "no_conversation_retrieval_gate_in_trace",
            "candidate_count": 0,
            "selected_count": 0,
            "results": [],
        }

    gate_metadata: dict[str, Any] = dict(getattr(gate_stage, "metadata", {}) or {})
    will_run = bool(gate_metadata.get("will_run", False))

    if not will_run:
        last_qa_stage = next(
            (s for s in stages if getattr(s, "stage", "") in ("last_qa_resolution", "last_qa")),
            None,
        )
        skip_meta: dict[str, Any] = dict(getattr(last_qa_stage, "metadata", {}) or {}) if last_qa_stage else {}
        skip_reason = (
            skip_meta.get("decision_reason")
            or ("last_qa_skipped_broad_retrieval" if skip_meta.get("skip_broad_retrieval") else None)
            or "configuration_disabled"
        )
        return {
            "executed": False,
            "skipped_reason": str(skip_reason),
            "candidate_count": 0,
            "selected_count": 0,
            "results": [],
        }

    validation_stage = next(
        (
            s for s in stages
            if getattr(s, "stage", "") == "sql_validation"
            and dict(getattr(s, "metadata", {}) or {}).get("entity_type") == "conversation_hop"
        ),
        None,
    )
    val_meta = dict(getattr(validation_stage, "metadata", {}) or {}) if validation_stage else {}
    candidate_count = int(val_meta.get("candidate_count", 0))

    return {
        "executed": True,
        "skipped_reason": None,
        "candidate_count": candidate_count,
        "selected_count": 0,
        "results": [],
    }


def _build_knowledge_retrieval(obs: RequestObservation) -> dict[str, Any]:
    if not obs.knowledge_retrieval_executed:
        return {
            "executed": False,
            "skipped_reason": obs.knowledge_retrieval_skipped_reason
            or "intent_did_not_require_knowledge_retrieval",
            "query_used": None,
            "candidate_count": 0,
            "selected_count": 0,
            "results": [],
        }
    return {
        "executed": True,
        "query_used": obs.knowledge_query_used,
        "candidate_count": obs.knowledge_raw_count,
        "selected_count": len(obs.knowledge_results),
        "results": obs.knowledge_results,
    }


def _build_response_section(response: Any) -> dict[str, Any]:
    if response is None:
        return {}

    last_qa = getattr(response, "last_qa_state", None)

    def _qq(q: Any) -> str | None:
        if q is None:
            return None
        text = getattr(q, "text", None) or (q if isinstance(q, str) else None)
        return str(text).strip() if text else None

    clarification_q = _qq(getattr(last_qa, "clarification_question", None)) if last_qa else None
    reminder_sq = _qq(getattr(last_qa, "reminder_supporting_question", None)) if last_qa else None
    supporting_qs = [
        _qq(q)
        for q in (getattr(last_qa, "supporting_questions", []) or [])
        if _qq(q)
    ]

    last_qa_safe: dict[str, Any] = {}
    if last_qa is not None:
        rt = getattr(last_qa, "response_type", None)
        last_qa_safe = {
            "response_type": rt.value if hasattr(rt, "value") else str(rt) if rt is not None else None,
            "linked_topic_id": getattr(last_qa, "linked_topic_id", None),
            "linked_hop_id": getattr(last_qa, "linked_hop_id", None),
            "clarification_question": clarification_q,
            "reminder_supporting_question": reminder_sq,
            "supporting_question_count": len(getattr(last_qa, "supporting_questions", []) or []),
        }

    raw_platform: dict[str, Any] = dict(getattr(response, "platform_payload", {}) or {})
    platform_safe: dict[str, Any] = {}
    _SENSITIVE = ("token", "secret", "password", "key", "credential", "path", "raw_")
    for k, v in raw_platform.items():
        if any(s in k.lower() for s in _SENSITIVE):
            continue
        platform_safe[k] = v

    artifacts_raw = raw_platform.get("artifacts") or []
    artifact_ids = [
        str(item.get("artifact_id") or "")
        for item in (artifacts_raw if isinstance(artifacts_raw, list) else [])
        if isinstance(item, dict) and item.get("artifact_id")
    ]

    rt_val = getattr(response, "response_type", None)
    response_type_str = (
        rt_val.value if hasattr(rt_val, "value") else str(rt_val)
    ) if rt_val is not None else None

    return {
        "final_chat_text": str(getattr(response, "final_chat_text", "") or ""),
        "response_type": response_type_str,
        "clarification_question": clarification_q,
        "reminder_supporting_question": reminder_sq,
        "supporting_questions": supporting_qs,
        "actions_committed": list(getattr(response, "actions_committed", []) or []),
        "actions_pending": list(getattr(response, "actions_pending_confirmation", []) or []),
        "warnings": list(getattr(response, "warnings", []) or []),
        "conversation_topic_id": getattr(response, "conversation_topic_id", None),
        "conversation_hop_id": getattr(response, "conversation_hop_id", None),
        "artifact_ids": artifact_ids,
        "platform_payload": platform_safe,
        "last_qa_state": last_qa_safe,
    }


# ---------------------------------------------------------------------------
# Single-query benchmark runner
# ---------------------------------------------------------------------------

def run_single_query(
    *,
    query: str,
    query_index: int,
    user_id: str,
    instrumented: _InstrumentedPipeline,
    repository: Any,
) -> dict[str, Any]:
    """Execute one measured query and return the full benchmark record."""

    request_id = uuid.uuid4().hex
    idempotency_key = f"benchmark-{uuid.uuid4().hex}"

    from assistant_rag.contracts import ChatRequest
    from assistant_rag.request_lifecycle import ChatRequestLifecycleExecutor
    from assistant_rag.observability import start_trace

    request = ChatRequest(
        user_id=user_id,
        raw_query=query,
        idempotency_key=idempotency_key,
    )

    start_trace(request_id)

    started_utc = datetime.now(timezone.utc)
    t_start_ns = time.perf_counter_ns()

    # Use a thin shim that delegates to handle_observed
    class _ShimPipeline:
        def __init__(self, instr: _InstrumentedPipeline) -> None:
            self._instr = instr
            self._obs: RequestObservation | None = None

        def handle(self, req: Any, repo: Any) -> Any:
            response, obs = self._instr.handle_observed(req, repo)
            self._obs = obs
            return response

    shim = _ShimPipeline(instrumented)

    executor = ChatRequestLifecycleExecutor(
        pipeline=shim,  # type: ignore[arg-type]
        repository=repository,
    )
    execution = executor.execute(request, fallback_request_id=request_id)

    t_end_ns = time.perf_counter_ns()
    ended_utc = datetime.now(timezone.utc)

    latency_ms = (t_end_ns - t_start_ns) / 1_000_000.0

    response = execution.response
    obs = shim._obs or RequestObservation()

    latency_section = {
        "end_to_end_ms": round(latency_ms, 3),
        "started_at": started_utc.isoformat(),
        "completed_at": ended_utc.isoformat(),
    }

    intent_section = {
        "branch": obs.intent_branch,
        "confidence": obs.intent_confidence,
        "reason_summary": obs.intent_reason_summary,
    }

    return {
        "benchmark_version": BENCHMARK_VERSION,
        "query_index": query_index,
        "user_id": user_id,
        "request_id": getattr(execution, "request_id", None) or request_id,
        "user_query": query,
        "status": "success",
        "latency": latency_section,
        "intent": intent_section,
        "response": _build_response_section(response),
        "conversation_retrieval": _extract_conv_retrieval(response, obs),
        "knowledge_retrieval": _build_knowledge_retrieval(obs),
        "error": None,
    }


# ---------------------------------------------------------------------------
# Error record builder
# ---------------------------------------------------------------------------

def _build_error_record(
    *,
    query: str,
    query_index: int,
    user_id: str,
    exc: BaseException,
    request_id: str | None = None,
) -> dict[str, Any]:
    error_type = type(exc).__name__
    safe_message = repr(exc)[:300]
    logger.error("Query %d failed: %s: %s", query_index, error_type, safe_message)
    return {
        "benchmark_version": BENCHMARK_VERSION,
        "query_index": query_index,
        "user_id": user_id,
        "request_id": request_id or "",
        "user_query": query,
        "status": "error",
        "latency": None,
        "intent": None,
        "response": None,
        "conversation_retrieval": None,
        "knowledge_retrieval": None,
        "error": {
            "type": error_type,
            "message": safe_message,
        },
    }


# ---------------------------------------------------------------------------
# Batch summary
# ---------------------------------------------------------------------------

def _build_summary(records: list[dict[str, Any]], total: int) -> dict[str, Any]:
    successful = [r for r in records if r.get("status") == "success"]
    latencies = [
        r["latency"]["end_to_end_ms"]
        for r in successful
        if r.get("latency") and r["latency"].get("end_to_end_ms") is not None
    ]

    intent_counts: dict[str, int] = {}
    conv_executed = 0
    know_executed = 0

    for r in successful:
        branch = (r.get("intent") or {}).get("branch")
        if branch:
            intent_counts[branch] = intent_counts.get(branch, 0) + 1
        if (r.get("conversation_retrieval") or {}).get("executed"):
            conv_executed += 1
        if (r.get("knowledge_retrieval") or {}).get("executed"):
            know_executed += 1

    latency_stats: dict[str, float] = {}
    if latencies:
        latency_stats = {
            "minimum": round(min(latencies), 3),
            "maximum": round(max(latencies), 3),
            "mean": round(statistics.mean(latencies), 3),
            "median": round(statistics.median(latencies), 3),
            "p95": round(_percentile(latencies, 95), 3),
        }

    return {
        "record_type": "benchmark_summary",
        "total_queries": total,
        "successful_queries": len(successful),
        "failed_queries": total - len(successful),
        "latency_ms": latency_stats,
        "intent_counts": intent_counts,
        "conversation_retrieval_executed": conv_executed,
        "knowledge_retrieval_executed": know_executed,
    }


# ---------------------------------------------------------------------------
# Input parsing
# ---------------------------------------------------------------------------

def _load_queries(path: str) -> tuple[str, list[str]]:
    """Return (user_id_from_file_or_empty, queries)."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, list):
        return "", [str(q) for q in data]
    if isinstance(data, dict):
        queries = [str(q) for q in data.get("queries", [])]
        file_user_id = str(data.get("user_id", "")).strip()
        return file_user_id, queries
    raise ValueError(f"Unsupported input format in {path}; expected list or object.")


# ---------------------------------------------------------------------------
# Runtime bootstrap
# ---------------------------------------------------------------------------

def _build_runtime(preview_chars: int) -> tuple[_InstrumentedPipeline, Any]:
    """Build the exact production runtime. Returns (instrumented_pipeline, repository)."""
    logger.info("Loading ProductionSettings from environment...")
    from assistant_rag.settings import ProductionSettings
    from assistant_rag.production_factory import (
        build_production_pipeline,
        build_production_repository,
    )

    settings = ProductionSettings.from_env()

    logger.info("Building production repository...")
    repository = build_production_repository(settings)

    logger.info("Building production pipeline (includes model warmup if configured)...")
    pipeline = build_production_pipeline(settings)

    instrumented = _InstrumentedPipeline(pipeline, preview_chars)
    return instrumented, repository


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------

def _run_warmup(*, instrumented: _InstrumentedPipeline, repository: Any, user_id: str) -> None:
    logger.info("Running warmup query (not measured)...")
    from assistant_rag.contracts import ChatRequest
    from assistant_rag.request_lifecycle import ChatRequestLifecycleExecutor
    from assistant_rag.observability import start_trace

    warmup_id = uuid.uuid4().hex
    start_trace(warmup_id)
    warmup_request = ChatRequest(
        user_id=user_id,
        raw_query="warmup",
        idempotency_key=f"warmup-{warmup_id}",
    )
    # Use the real pipeline directly for warmup; no observation needed
    real_pipeline = instrumented._pipeline
    try:
        executor = ChatRequestLifecycleExecutor(pipeline=real_pipeline, repository=repository)
        executor.execute(warmup_request, fallback_request_id=warmup_id)
        logger.info("Warmup complete.")
    except Exception as exc:
        logger.warning("Warmup query raised %s: %s", type(exc).__name__, exc)


# ---------------------------------------------------------------------------
# Resource cleanup
# ---------------------------------------------------------------------------

def _shutdown(instrumented: _InstrumentedPipeline, repository: Any) -> None:
    pipeline = instrumented._pipeline
    for attr in ("retriever", "bm25", "chroma"):
        target = getattr(pipeline, attr, None)
        candidates = [target]
        if target is not None:
            for sub_attr in ("bm25", "chroma"):
                sub = getattr(target, sub_attr, None)
                if sub is not None:
                    candidates.append(sub)
        for sub in candidates:
            if sub is None:
                continue
            for method in ("close", "shutdown", "disconnect"):
                fn = getattr(sub, method, None)
                if callable(fn):
                    try:
                        fn()
                    except Exception:
                        pass

    for method in ("close", "shutdown", "dispose"):
        fn = getattr(repository, method, None)
        if callable(fn):
            try:
                fn()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="benchmark_pipeline.py",
        description=(
            "Run a batch of queries through the real production pipeline "
            "and emit structured JSON benchmark records."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input",
        required=True,
        metavar="PATH",
        help="JSON file with query list or {user_id, queries} object.",
    )
    parser.add_argument(
        "--user-id",
        default="",
        metavar="ID",
        help=f"Benchmark user ID (overrides file; default: {DEFAULT_USER_ID}).",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help="Write JSONL records to PATH instead of stdout.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Emit error records instead of aborting on query failure.",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Send one silent warmup query before the measured batch.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append records to --output file instead of overwriting.",
    )
    parser.add_argument(
        "--text-preview-chars",
        type=int,
        default=DEFAULT_TEXT_PREVIEW_CHARS,
        metavar="N",
        help=f"Max chars for knowledge text preview (default: {DEFAULT_TEXT_PREVIEW_CHARS}).",
    )
    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Load queries
    try:
        file_user_id, queries = _load_queries(args.input)
    except Exception as exc:
        logger.error("Failed to load input file %s: %s", args.input, exc)
        sys.exit(1)

    if not queries:
        logger.error("Input file contains no queries. Exiting.")
        sys.exit(1)

    user_id = args.user_id.strip() or file_user_id or DEFAULT_USER_ID
    logger.info("Benchmark user: %s | queries: %d", user_id, len(queries))

    # Build runtime
    try:
        instrumented, repository = _build_runtime(args.text_preview_chars)
    except Exception:
        logger.error("Failed to initialise production runtime:\n%s", traceback.format_exc())
        sys.exit(1)

    # Open output
    if args.output:
        try:
            mode = "a" if args.append else "w"
            output_file = open(args.output, mode, encoding="utf-8")
        except OSError as exc:
            logger.error("Cannot open output file %s: %s", args.output, exc)
            sys.exit(1)
    else:
        output_file = sys.stdout

    # Warmup
    if args.warmup:
        try:
            _run_warmup(instrumented=instrumented, repository=repository, user_id=user_id)
        except Exception as exc:
            logger.warning("Warmup raised %s: %s", type(exc).__name__, exc)

    # Batch execution
    records: list[dict[str, Any]] = []
    try:
        for idx, query in enumerate(queries):
            logger.info("Running query %d/%d: %r...", idx + 1, len(queries), query[:80])
            try:
                record = run_single_query(
                    query=query,
                    query_index=idx,
                    user_id=user_id,
                    instrumented=instrumented,
                    repository=repository,
                )
            except Exception as exc:
                if args.continue_on_error:
                    record = _build_error_record(
                        query=query,
                        query_index=idx,
                        user_id=user_id,
                        exc=exc,
                    )
                else:
                    logger.error(
                        "Query %d failed (use --continue-on-error to skip):\n%s",
                        idx,
                        traceback.format_exc(),
                    )
                    sys.exit(1)

            records.append(record)
            _emit_record(record, output_file)

            lat = (record.get("latency") or {}).get("end_to_end_ms")
            branch = (record.get("intent") or {}).get("branch")
            logger.info(
                "Query %d done | status=%s | intent=%s | latency=%.1f ms",
                idx,
                record.get("status"),
                branch,
                lat or 0.0,
            )

        # Summary
        summary = _build_summary(records, total=len(queries))
        _emit_record(summary, output_file)
        p95 = (summary.get("latency_ms") or {}).get("p95") or 0.0
        logger.info(
            "Benchmark complete: %d/%d successful | p95 latency=%.1f ms",
            summary["successful_queries"],
            summary["total_queries"],
            p95,
        )

    finally:
        _shutdown(instrumented, repository)
        if args.output and output_file is not sys.stdout:
            try:
                output_file.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
