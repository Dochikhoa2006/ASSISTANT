"""Ollama-backed LLM integration."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import Enum
import json
import logging
import time
from typing import Any, Protocol
from urllib import error, request

from .contracts import Intent, ResponseType
from .metrics import GLOBAL_METRICS
from .observability import StageTimer, current_trace
from .prompts import DEFAULT_PROMPT_REGISTRY, PromptContext, PromptRegistry
from .settings import OllamaSettings, PromptPolicySettings

logger = logging.getLogger(__name__)


def _has_pending_clarification(last_qa_resolution: Any) -> bool:
    state = getattr(last_qa_resolution, "state", None)
    response_type = getattr(state, "response_type", None)
    response_value = getattr(response_type, "value", response_type)
    return bool(
        state
        and getattr(state, "clarification_question", None)
        and response_value == ResponseType.CLARIFICATION.value
    )


def _is_authoritative_outbound_action(last_qa_resolution: Any) -> bool:
    interaction = getattr(last_qa_resolution, "interaction_type", None)
    interaction_value = getattr(interaction, "value", interaction)
    return bool(
        last_qa_resolution
        and interaction_value == "outbound_message_action"
        and getattr(last_qa_resolution, "outbound_action", None) is not None
        and getattr(last_qa_resolution, "is_authoritative_state", False)
    )


class LLMTask(str, Enum):
    QUERY_REWRITE = "query_rewrite"
    LAST_QA = "last_qa"
    INTENT = "intent"
    ACTION_EXTRACTION = "action_extraction"
    KNOWLEDGE_ACTION_EXTRACTION = "knowledge_action_extraction"
    KNOWLEDGE_ACTION_VALIDATION = "knowledge_action_validation"
    KNOWLEDGE_CONTENT_FINALIZATION = "knowledge_content_finalization"
    REMINDER_ACTION_EXTRACTION = "reminder_action_extraction"
    REMINDER_ACTION_VALIDATION = "reminder_action_validation"
    REMINDER_CONTENT_FINALIZATION = "reminder_content_finalization"
    GENERATE_CLARIFICATION = "generate_clarification"
    GENERATE_HUMAN_SUPPORTING = "generate_human_supporting"
    CLARIFICATION_MERGE = "clarification_merge"
    ANSWER = "answer"
    WRITING = "writing"
    RISKY_ACTION = "risky_action"
    RETRIEVAL_VALIDATION = "retrieval_validation"
    GENERAL_SUB_BRANCH_DETECTION = "general_sub_branch_detection"
    CONTENT_COMPOSER_REACT = "content_composer_react"
    ACTION_PLANNING = "action_planning"


def llm_trace_stage_name(
    task: LLMTask,
    *,
    engine: str | None = None,
    pipeline_stage: str | None = None,
) -> str:
    """Return the stable trace label for an LLM task.

    Answer generation historically appeared as ``llm_answer`` (and
    ``llm_answer_onnx``), which obscured its relationship to the canonical
    ``answer_generation`` pipeline stage.  Keep every other established label
    intact while making answer-generation timing consistent across engines.
    """

    if task is LLMTask.ANSWER:
        return "llm_answer_generation"
    if (
        task is LLMTask.WRITING
        and str(pipeline_stage or "").strip() == "content_tool_answer_generation"
    ):
        return "llm_writing_microsoft_tool"
    suffix = f"_{engine}" if engine else ""
    return f"llm_{task.value}{suffix}"


@dataclass(frozen=True)
class ModelDecision:
    task: LLMTask
    model: str
    temperature: float
    timeout_seconds: float
    num_ctx: int | None
    num_predict: int | None
    reason_summary: str


class LLMClient(Protocol):
    def generate_json(
        self,
        *,
        task: LLMTask,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any],
        model_override: str | None = None,
        fallback_for: str | None = None,
    ) -> dict[str, Any]:
        ...

    def chat(self, *, task: LLMTask, system_prompt: str, user_prompt: str) -> str:
        ...


class StructuredFallbackPayload(dict[str, Any]):
    """Schema-compatible terminal recovery with out-of-band provenance.

    The marker deliberately lives on the mapping object rather than inside it,
    so a minimal model schema does not need diagnostic fields and strict
    downstream key guards continue to see only the declared output contract.
    """

    __slots__ = ("task", "reason")

    def __init__(
        self,
        payload: dict[str, Any],
        *,
        task: LLMTask,
        reason: str,
    ) -> None:
        super().__init__(payload)
        self.task = task
        self.reason = reason


def is_structured_fallback(payload: Any) -> bool:
    """Return whether a structured result is terminal recovery, not model data."""

    return isinstance(payload, StructuredFallbackPayload)


@dataclass(frozen=True)
class OllamaModelRouter:
    settings: OllamaSettings

    def decision_for_task(self, task: LLMTask) -> ModelDecision:
        return ModelDecision(
            task=task,
            model=self.model_for_task(task),
            temperature=self.temperature_for_task(task),
            timeout_seconds=self.timeout_for_task(task),
            num_ctx=self.num_ctx_for_task(task),
            num_predict=self.num_predict_for_task(task),
            reason_summary=f"Task-specific configuration for {task.value}",
        )

    def model_for_task(self, task: LLMTask) -> str:
        return getattr(self.settings, f"model_{task.value}")

    def temperature_for_task(self, task: LLMTask) -> float:
        return getattr(self.settings, f"temperature_{task.value}")

    def num_ctx_for_task(self, task: LLMTask) -> int | None:
        return getattr(self.settings, f"num_ctx_{task.value}")

    def num_predict_for_task(self, task: LLMTask) -> int | None:
        return getattr(self.settings, f"num_predict_{task.value}")

    def timeout_for_task(self, task: LLMTask) -> float:
        return getattr(self.settings, f"timeout_{task.value}")


@dataclass
class OllamaLLMClient:
    settings: OllamaSettings
    router: OllamaModelRouter
    last_error_by_task: dict[LLMTask, str] = field(default_factory=dict)

    def warmup_models(self) -> list[str]:
        """Ask Ollama to load every configured local model and keep it resident."""
        models: list[str] = []
        for field_info in fields(self.settings):
            if not field_info.name.startswith("model_"):
                continue
            model = str(getattr(self.settings, field_info.name) or "")
            if model and not uses_onnx_runtime(model) and model not in models:
                models.append(model)

        for model in models:
            self._request_json(
                "/api/generate",
                {
                    "model": model,
                    "prompt": "",
                    "stream": False,
                    "keep_alive": self.settings.keep_alive,
                },
                timeout=max(60.0, max(self.router.timeout_for_task(task) for task in LLMTask)),
            )
        return models

    def generate_json(
        self,
        *,
        task: LLMTask,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any],
        model_override: str | None = None,
        fallback_for: str | None = None,
    ) -> dict[str, Any]:
        with StageTimer(llm_trace_stage_name_for_prompt(task, user_prompt)):
            last_error: Exception | None = None
            retry_count = getattr(self.settings, f"json_retry_count_{task.value}", self.settings.structured_retry_count)
            attempt_errors: list[str] = []
            total_attempts = retry_count + 1
            attempt_plan = _structured_attempt_plan(total_attempts)
            for attempt_index, attempt_mode in enumerate(attempt_plan, start=1):
                try:
                    attempt_prompt = _structured_attempt_prompt(
                        user_prompt=user_prompt,
                        schema=schema,
                        mode=attempt_mode,
                    )
                    raw = self._chat_raw(
                        task=task,
                        system_prompt=system_prompt,
                        user_prompt=attempt_prompt,
                        format_schema=_structured_attempt_format(schema, attempt_mode),
                        model_override=model_override,
                        fallback_for=fallback_for,
                        expect_json=True,
                        attempt_index=attempt_index,
                        total_attempts=len(attempt_plan),
                        attempt_mode=attempt_mode,
                    )
                    payload = parse_json_object(raw)
                    payload, normalized = normalize_structured_output(
                        payload,
                        schema,
                    )
                    validate_json_schema(payload, schema)
                    trace = current_trace()
                    if trace and trace._active_timers:
                        timer = trace._active_timers[-1]
                        timer.metadata["structured_attempts"] = attempt_index
                        timer.metadata["structured_configured_attempts"] = total_attempts
                        if normalized:
                            timer.metadata["structured_output_normalized"] = True
                        if attempt_errors:
                            timer.metadata["structured_previous_failures"] = len(attempt_errors)
                            timer.metadata["structured_last_retry_error"] = attempt_errors[-1]
                    self.last_error_by_task.pop(task, None)
                    return payload
                except Exception as exc:
                    GLOBAL_METRICS.increment("llm_json_parse_failures_total", task=task.value)
                    last_error = exc
                    attempt_errors.append(f"attempt {attempt_index}/{len(attempt_plan)} [{attempt_mode}]: {type(exc).__name__}: {exc}")
                    self.last_error_by_task[task] = attempt_errors[-1]
                    logger.debug("LLM JSON parsing failed on attempt %d/%d for task %s (mode: %s): %s", attempt_index, len(attempt_plan), task.value, attempt_mode, exc)
            
            GLOBAL_METRICS.increment("llm_failures_total", task=task.value)
            payload = structured_fallback_payload(
                task=task,
                schema=schema,
                user_prompt=user_prompt,
                error=last_error,
            )
            validate_json_schema(payload, schema)
            _log_structured_fallback(
                engine="Ollama",
                task=task,
                attempt_count=len(attempt_plan),
                attempt_errors=attempt_errors,
            )
            trace = current_trace()
            if trace and trace._active_timers:
                timer = trace._active_timers[-1]
                timer.metadata["structured_fallback_used"] = True
                timer.metadata["structured_attempts"] = len(attempt_plan)
                timer.metadata["structured_configured_attempts"] = total_attempts
                timer.metadata["structured_failed_attempts"] = len(attempt_errors)
                timer.metadata["structured_fallback_reason"] = str(last_error)
                timer.metadata["structured_retry_errors_tail"] = attempt_errors[-3:]
            self.last_error_by_task[task] = f"structured_fallback_used_after_{len(attempt_errors)}_failures: {last_error}"
            return payload

    def chat(self, *, task: LLMTask, system_prompt: str, user_prompt: str) -> str:
        with StageTimer(llm_trace_stage_name_for_prompt(task, user_prompt)):
            try:
                response = self._chat_raw(
                    task=task,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    format_schema=None,
                )
            except Exception as exc:
                fallback_model = self._fallback_model_for_task(task)
                primary_model = self.router.model_for_task(task)
                if not fallback_model or fallback_model == primary_model:
                    GLOBAL_METRICS.increment("llm_failures_total", task=task.value)
                    self.last_error_by_task[task] = str(exc)
                    raise
                try:
                    response = self._chat_raw(
                        task=task,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        format_schema=None,
                        model_override=fallback_model,
                        fallback_for=primary_model,
                    )
                except Exception as fallback_exc:
                    GLOBAL_METRICS.increment("llm_failures_total", task=task.value)
                    self.last_error_by_task[task] = (
                        f"{primary_model} failed: {exc}; fallback {fallback_model} failed: {fallback_exc}"
                    )
                    logger.error("Severe failure: LLM chat failed for task %s on both primary (%s) and fallback (%s). Errors: %s, %s", task.value, primary_model, fallback_model, exc, fallback_exc)
                    raise
            self.last_error_by_task.pop(task, None)
            return response

    def list_models(self) -> list[str]:
        payload = self._get_json("/api/tags")
        return [str(model["name"]) for model in payload.get("models", [])]

    def check(self) -> dict[str, Any]:
        started = time.perf_counter()
        models = self.list_models()
        configured_models = self._configured_models()
        model_checks = {
            task_name: model in models
            for task_name, model in configured_models.items()
            if not uses_onnx_runtime(model)
        }
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        return {
            "base_url": self.settings.base_url,
            "reachable": True,
            "installed_models": models,
            "configured_models": configured_models,
            "model_checks": model_checks,
            "latency_ms": latency_ms,
        }

    def _configured_models(self) -> dict[str, str]:
        configured: dict[str, str] = {}
        for field_info in fields(self.settings):
            if not field_info.name.startswith("model_"):
                continue
            model = getattr(self.settings, field_info.name)
            if model:
                configured[field_info.name.removeprefix("model_")] = str(model)
        return configured

    def _chat_raw(
        self,
        *,
        task: LLMTask,
        system_prompt: str,
        user_prompt: str,
        format_schema: dict[str, Any] | str | None,
        model_override: str | None = None,
        fallback_for: str | None = None,
        expect_json: bool = False,
        attempt_index: int | None = None,
        total_attempts: int | None = None,
        attempt_mode: str | None = None,
    ) -> str:
        decision = self.router.decision_for_task(task)
        model = model_override or decision.model
        attempt_suffix = ""
        if attempt_index is not None and total_attempts is not None:
            mode_text = f", mode: {attempt_mode}" if attempt_mode else ""
            attempt_suffix = f", attempt: {attempt_index}/{total_attempts}{mode_text}"
        log_task_name = llm_trace_stage_name_for_prompt(
            task,
            user_prompt,
        ).removeprefix("llm_")
        if fallback_for:
            print(f"LLM used: {model} (task: {log_task_name}, fallback_for: {fallback_for}{attempt_suffix})")
        else:
            print(f"LLM used: {model} (task: {log_task_name}{attempt_suffix})")
        body: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "keep_alive": self.settings.keep_alive,
            "options": {
                "temperature": decision.temperature,
            },
        }
        if self.settings.disable_thinking:
            body["think"] = False
        if decision.num_ctx:
            body["options"]["num_ctx"] = decision.num_ctx
        if decision.num_predict:
            body["options"]["num_predict"] = decision.num_predict
        if format_schema is not None:
            body["format"] = format_schema
        payload = self._request_json("/api/chat", body, timeout=decision.timeout_seconds)
        
        trace = current_trace()
        if trace and trace._active_timers:
            timer = trace._active_timers[-1]
            timer.metadata["model"] = model
            if fallback_for:
                timer.metadata["fallback_for"] = fallback_for
            if attempt_index is not None and total_attempts is not None:
                timer.metadata["current_attempt"] = attempt_index
                timer.metadata["total_attempts"] = total_attempts
            if attempt_mode:
                timer.metadata["attempt_mode"] = attempt_mode
            if "prompt_eval_count" in payload:
                timer.metadata["input_count"] = payload["prompt_eval_count"]
            if "eval_count" in payload:
                timer.metadata["output_count"] = payload["eval_count"]
                
        msg = payload.get("message", {})
        content = str(msg.get("content", ""))
        thinking = str(msg.get("thinking", ""))
        if expect_json and not content.strip():
            if thinking.strip():
                raise ValueError("Structured LLM returned thinking but no JSON content")
            raise ValueError("Structured LLM returned empty JSON content")
        if not content.strip() and thinking:
            return thinking.strip()
        return content

    def _fallback_model_for_task(self, task: LLMTask) -> str | None:
        fallback = getattr(self.settings, f"model_{task.value}_fallback", None)
        return str(fallback) if fallback else None

    def _request_json(self, path: str, body: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        url = f"{self.settings.base_url.rstrip('/')}{path}"
        data = json.dumps(body).encode("utf-8")
        req = request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        if timeout is None:
            timeout = self._control_plane_timeout_seconds()
        try:
            with request.urlopen(req, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.URLError as exc:
            raise ConnectionError(f"Ollama request failed for {url}: {exc}") from exc

    def _get_json(self, path: str) -> dict[str, Any]:
        url = f"{self.settings.base_url.rstrip('/')}{path}"
        req = request.Request(url, method="GET")
        try:
            with request.urlopen(req, timeout=self._control_plane_timeout_seconds()) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.URLError as exc:
            raise ConnectionError(f"Ollama request failed for {url}: {exc}") from exc

    def _control_plane_timeout_seconds(self) -> float:
        return min(10.0, max(3.0, self.settings.timeout_query_rewrite))


def parse_json_object(raw: str) -> dict[str, Any]:
    from .json_repair import advanced_parse_json
    return advanced_parse_json(raw)


def _structured_attempt_plan(configured_attempts: int) -> list[str]:
    """Use varied request strategies instead of repeating one broken shape."""
    if configured_attempts <= 1:
        return ["json"]
    if configured_attempts == 2:
        return ["json", "schema"]
    return ["json", "schema", "plain"]


def _structured_attempt_format(schema: dict[str, Any], mode: str) -> dict[str, Any] | str | None:
    if mode == "schema":
        return schema
    if mode == "json":
        return "json"
    return None


def _structured_attempt_prompt(*, user_prompt: str, schema: dict[str, Any], mode: str) -> str:
    template = json.dumps(_schema_default_object(schema), indent=2)
    rules = []
    properties = schema.get("properties") or {}
    for key, prop in properties.items():
        if "enum" in prop:
            rules.append(f"- '{key}' must be one of: {json.dumps(prop['enum'])}")
        if prop.get("type") == "string" and prop.get("maxLength"):
            rules.append(f"- '{key}' must be at most {int(prop['maxLength'])} characters")
            
    rules_text = "\n".join(rules)
    if rules_text:
        rules_text = f"\nConstraints:\n{rules_text}"

    retry_instruction = (
        "Correction: the previous output was invalid. Return a JSON data object, not a JSON Schema. "
        "Never output keys such as 'type', 'properties', 'required', 'items', or '$schema'.\n"
        if mode == "schema"
        else ""
    )
    return (
        f"{user_prompt}\n\n"
        f"{retry_instruction}"
        "Structured output instruction:\n"
        "Return exactly one JSON object and nothing else. No markdown, no code fence, no prose.\n"
        f"Return a data instance with exactly this shape:\n{template}{rules_text}"
    )


def _extract_json_object(text: str) -> str | None:
    from .json_repair import _extract_json_object as advanced_extract
    return advanced_extract(text)


def validate_json_schema(payload: dict[str, Any], schema: dict[str, Any]) -> None:
    """Validate the complete structured-output contract, including nested actions.

    Action schemas carry their safety-critical operation enum inside array items.
    A root-only check would let an invalid nested operation reach mutation
    validation, where it becomes an opaque no-op instead of consuming the
    configured structured-output retry.
    """
    _validate_json_value(payload, schema, path="$")


def normalize_structured_output(
    payload: dict[str, Any],
    schema: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    """Apply only lossless, schema-authorized repairs before validation.

    This removes keys only when ``additionalProperties`` explicitly forbids
    them, canonicalizes an enum string only when casing identifies exactly one
    declared value, and converts integral JSON numbers to integers. Missing or
    semantically contradictory values remain failures and consume the normal
    bounded recovery path.
    """

    normalized = _normalize_structured_value(payload, schema)
    if not isinstance(normalized, dict):
        return payload, False
    return normalized, not _same_json_shape_and_value(normalized, payload)


def _same_json_shape_and_value(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _same_json_shape_and_value(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _same_json_shape_and_value(a, b) for a, b in zip(left, right)
        )
    return left == right


def _normalize_structured_value(value: Any, rules: dict[str, Any]) -> Any:
    enum_values = rules.get("enum")
    if isinstance(value, str) and isinstance(enum_values, list):
        matches = [
            candidate
            for candidate in enum_values
            if isinstance(candidate, str)
            and candidate.casefold() == value.casefold()
        ]
        if len(matches) == 1:
            value = matches[0]

    expected_type = rules.get("type")
    if (
        expected_type == "integer"
        and isinstance(value, float)
        and not isinstance(value, bool)
        and value.is_integer()
    ):
        value = int(value)

    if isinstance(value, dict) and expected_type == "object":
        properties = rules.get("properties") or {}
        allowed_keys = set(properties)
        keys = (
            [key for key in value if key in allowed_keys]
            if rules.get("additionalProperties") is False
            else list(value)
        )
        return {
            key: _normalize_structured_value(value[key], properties[key])
            if key in properties
            else value[key]
            for key in keys
        }

    if isinstance(value, list) and isinstance(rules.get("items"), dict):
        return [
            _normalize_structured_value(item, rules["items"])
            for item in value
        ]
    return value


def _validate_json_value(value: Any, rules: dict[str, Any], *, path: str) -> None:
    expected_type = rules.get("type")
    if expected_type == "string" and not isinstance(value, str):
        raise ValueError(f"Structured output field must be string: {path}")
    if expected_type == "number" and (isinstance(value, bool) or not isinstance(value, (int, float))):
        raise ValueError(f"Structured output field must be number: {path}")
    if expected_type == "integer" and (isinstance(value, bool) or not isinstance(value, int)):
        raise ValueError(f"Structured output field must be integer: {path}")
    if expected_type == "boolean" and not isinstance(value, bool):
        raise ValueError(f"Structured output field must be boolean: {path}")
    if expected_type == "array" and not isinstance(value, list):
        raise ValueError(f"Structured output field must be array: {path}")
    if expected_type == "object" and not isinstance(value, dict):
        raise ValueError(f"Structured output field must be object: {path}")

    enum_values = rules.get("enum")
    if enum_values is not None and value not in enum_values:
        raise ValueError(f"Structured output field has unsupported value at {path}: {value!r}")

    if isinstance(value, dict):
        required = rules.get("required") or []
        for key in required:
            if key not in value:
                raise ValueError(f"Structured output missing required field: {path}.{key}")
        for key, nested_rules in (rules.get("properties") or {}).items():
            if key in value:
                _validate_json_value(value[key], nested_rules, path=f"{path}.{key}")
    elif isinstance(value, list) and isinstance(rules.get("items"), dict):
        item_rules = rules["items"]
        for index, item in enumerate(value):
            _validate_json_value(item, item_rules, path=f"{path}[{index}]")


def structured_fallback_payload(
    *,
    task: LLMTask,
    schema: dict[str, Any],
    user_prompt: str,
    error: Exception | None,
) -> dict[str, Any]:
    payload = _schema_default_object(schema)
    context = _prompt_context_payload(user_prompt)
    reason = f"structured fallback after invalid LLM JSON: {error}"

    if task == LLMTask.QUERY_REWRITE:
        query = str(context.get("raw_query") or context.get("rewritten_query") or "").strip()
        payload.update({"rewritten_query": query})
    elif task == LLMTask.LAST_QA:
        properties = schema.get("properties") or {}
        if "outbound_action" in properties:
            payload.update({"outbound_action": "none", "confidence": 0.0})
        elif "matched_question_index" in properties:
            payload.update({"matched_question_index": -1, "confidence": 0.0})
        else:
            payload.update({
                "interaction_type": "ambiguous",
                "question_source": "none",
                "matched_question": "",
                "confidence": 0.0,
            })
    elif task == LLMTask.INTENT:
        payload.update({
            "intent": Intent.GENERAL_RESPONSE.value,
            "confidence": 1.0,
        })
    elif task == LLMTask.REMINDER_ACTION_EXTRACTION:
        payload.update({
            "action": "add",
            "retrieval_text": "",
            "field_values": [],
            "confidence": 0.0,
        })
    elif task in {
        LLMTask.ACTION_EXTRACTION,
        LLMTask.KNOWLEDGE_ACTION_EXTRACTION,
    }:
        if "text_content" in (schema.get("properties") or {}):
            payload.update({
                "action": "add",
                "text_content": "",
                "original_text": "",
                "replacement_text": "",
                "confidence": 0.0,
                "missing_fields": ["llm_structured_fallback"],
                "reason_summary": reason,
            })
        else:
            intent = str(context.get("intent") or Intent.GENERAL_RESPONSE.value)
            payload.update({
                "intent": intent if intent in {item.value for item in Intent} else Intent.GENERAL_RESPONSE.value,
                "confidence": 0.0,
                "knowledge_actions": [],
                "reminder_actions": [],
                "missing_fields": ["llm_structured_fallback"],
                "risk_flags": ["safe_fallback_no_actions"],
                "normalized_entities": {},
                "reason_summary": reason,
            })
    elif task == LLMTask.RISKY_ACTION:
        payload.update({"results": []})
    elif task == LLMTask.GENERATE_CLARIFICATION:
        payload.update({
            # Technical recovery must never impersonate a semantic decision to
            # ask the user something. Consumers use the out-of-band fallback
            # marker and this empty fail-closed shape to pause cleanly.
            "question_text": "",
            "question_source": "clarification_question",
            "purpose": "resolve_missing_info",
            "confidence": 0.0,
            "should_ask": False,
            "expected_response_type": "unknown",
            "reason_summary": reason,
        })
    elif task == LLMTask.GENERATE_HUMAN_SUPPORTING:
        payload.update({"questions": []})
    elif task == LLMTask.CLARIFICATION_MERGE:
        payload.update({
            "answered_clarification": False,
            "merged_query": str(context.get("rewritten_query") or ""),
            "confidence": 0.0,
            "missing_context": ["clarification_merge_unavailable"],
            "reason_summary": reason,
        })
    elif task == LLMTask.GENERAL_SUB_BRANCH_DETECTION:
        payload.update({
            "sub_branch": "new_conversation_topic",
            "confidence": 0.55,
            "persistence_mode": "create_new_topic",
            "selected_candidate_ref": "",
        })
    elif task == LLMTask.CONTENT_COMPOSER_REACT:
        payload.update({
            "thought": "Fallback to standard answer generation.",
            "tool_name": "answer_generation",
            "confidence": 1.0,
            "is_final_answer": True,
        })
    elif task == LLMTask.KNOWLEDGE_ACTION_VALIDATION:
        payload.update({
            "operation": _first_enum(schema, "operation") or "add",
            "decision": "FAIL",
            "selected_candidate_keys": [],
            "confidence": 0.0,
            "clarification_question": "",
            "reason_summary": reason,
            "candidate_assessments": [],
        })
    elif task == LLMTask.REMINDER_ACTION_VALIDATION:
        payload.update({
            "validation_result": "FAIL",
            "selected_candidate_keys": [],
            "confidence": 0.0,
            "clarification_question": "",
            "candidate_assessments": [],
        })
    elif task == LLMTask.RETRIEVAL_VALIDATION:
        operation = _first_enum(schema, "operation") or "delete"
        payload.update({
            "operation": operation,
            "validation_result": "CLARIFY_MISSING_FIELDS",
            "selected_candidate_keys": [],
            "confidence": 0.0,
            "ambiguous": True,
            "reason_summary": reason,
            "candidate_assessments": [],
        })
        if "should_execute" in (schema.get("properties") or {}):
            payload.update({
                "should_execute": False,
                "requires_hitl": True,
                "factuality_concern": False,
            })
        if "hitl_reason" in (schema.get("properties") or {}):
            payload["hitl_reason"] = "llm_structured_fallback"
    elif task == LLMTask.REMINDER_CONTENT_FINALIZATION:
        payload.update({
            "approved": False,
            "confidence": 0.0,
        })
    elif task == LLMTask.ACTION_PLANNING:
        payload.update({"confidence": 0.0, "reason_summary": reason})
        properties = schema.get("properties") or {}
        channel_values = (properties.get("channel") or {}).get("enum") or []
        if "none" in channel_values:
            payload["channel"] = "none"
        mode_values = (properties.get("mode") or {}).get("enum") or []
        if "draft" in mode_values:
            payload["mode"] = "draft"
        if "should_ask" in properties:
            payload["should_ask"] = False
        if "needs_clarification" in properties:
            payload["needs_clarification"] = True
    elif task in {
        LLMTask.CONTENT_COMPOSER_REACT,
        LLMTask.ANSWER,
        LLMTask.WRITING,
        LLMTask.KNOWLEDGE_CONTENT_FINALIZATION,
    }:
        payload.update({"reason_summary": reason})

    if "sheet_name" in payload:
        payload.update({"sheet_name": "Sheet1", "columns": [], "suggested_rows": [], "confidence": 0.0, "reason_summary": reason})
    if "title" in payload and "sections" in payload:
        payload.update({"title": "Untitled", "sections": [], "confidence": 0.0, "reason_summary": reason})
    if "presentation_title" in payload:
        payload.update({"presentation_title": "Untitled", "slides": [], "confidence": 0.0, "reason_summary": reason})

    # Fallbacks obey the same bounded output surface as the real model. This
    # prevents obsolete diagnostic keys from re-expanding a deliberately
    # minimal schema or tripping a downstream response guard.
    allowed_fields = set((schema.get("properties") or {}).keys())
    filtered_payload = {
        key: value
        for key, value in payload.items()
        if key in allowed_fields
    }
    return StructuredFallbackPayload(
        filtered_payload,
        task=task,
        reason=reason,
    )


def _log_structured_fallback(
    *,
    engine: str,
    task: LLMTask,
    attempt_count: int,
    attempt_errors: list[str],
) -> None:
    logger.debug(
        "%s structured JSON generation fell back after %d attempts for task %s. Errors: %s",
        engine,
        attempt_count,
        task.value,
        attempt_errors,
    )


def _schema_default_object(schema: dict[str, Any]) -> dict[str, Any]:
    properties = schema.get("properties") or {}
    required = schema.get("required") or properties.keys()
    return {
        key: _schema_default_value(properties.get(key, {}))
        for key in required
    }


def _schema_default_value(rules: dict[str, Any]) -> Any:
    if "default" in rules:
        return rules["default"]

    enum_values = rules.get("enum")
    if enum_values:
        return enum_values[0]

    expected_type = rules.get("type")
    if expected_type == "string":
        return ""
    if expected_type == "number":
        return 0.0
    if expected_type == "integer":
        return 0
    if expected_type == "boolean":
        return False
    if expected_type == "array":
        return []
    if expected_type == "object":
        return _schema_default_object(rules) if rules.get("properties") else {}
    return None


def _prompt_context_payload(user_prompt: str) -> dict[str, Any]:
    prefix = "Runtime context:\n"
    text = user_prompt.removeprefix(prefix).strip()
    try:
        payload = json.loads(text)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def llm_trace_stage_name_for_prompt(
    task: LLMTask,
    user_prompt: str,
    *,
    engine: str | None = None,
) -> str:
    """Resolve a stable task label using only the declared prompt stage."""
    pipeline_stage = _prompt_context_payload(user_prompt).get("stage")
    return llm_trace_stage_name(
        task,
        engine=engine,
        pipeline_stage=str(pipeline_stage or ""),
    )


def _first_enum(schema: dict[str, Any], key: str) -> str | None:
    rules = (schema.get("properties") or {}).get(key) or {}
    values = rules.get("enum") or []
    return str(values[0]) if values else None


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def uses_onnx_runtime(model_name: str) -> bool:
    normalized = model_name.casefold()
    return "onnx" in normalized or normalized.startswith("microsoft/phi")


def _truncate_intent_text(value: Any, max_chars: int = 700) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "...<truncated>"


def _compact_intent_question(question: Any) -> dict[str, Any]:
    if isinstance(question, dict):
        compact = {
            "text": question.get("text") or question.get("question_text"),
            "source": _enum_value(question.get("source") or question.get("question_source")),
            "purpose": question.get("purpose"),
            "expected_response_type": _enum_value(question.get("expected_response_type")),
        }
    else:
        compact = {
            "text": getattr(question, "text", None),
            "source": _enum_value(getattr(question, "source", None)),
            "purpose": getattr(question, "purpose", None),
            "expected_response_type": _enum_value(getattr(question, "expected_response_type", None)),
        }
    if compact.get("text"):
        compact["text"] = _truncate_intent_text(compact["text"], max_chars=220)
    return {key: value for key, value in compact.items() if value not in (None, "", [], {})}


def _compact_intent_history_item(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        return {"text": _truncate_intent_text(item)}

    compact = {
        "role": item.get("role") or "conversation_hop",
        "text": item.get("text") or item.get("summary") or item.get("raw_text"),
        "topic_id": item.get("topic_id"),
        "hop_id": item.get("hop_id"),
        "expected_response_type": _enum_value(item.get("expected_response_type")),
        "created_at": item.get("created_at"),
        "confidence": item.get("confidence"),
    }
    if compact.get("text"):
        compact["text"] = _truncate_intent_text(compact["text"])

    supporting_questions = item.get("supporting_questions") or []
    if isinstance(supporting_questions, list) and supporting_questions:
        compact["supporting_questions"] = [
            _compact_intent_question(question)
            for question in supporting_questions[:3]
        ]

    return {key: value for key, value in compact.items() if value not in (None, "", [], {})}


def build_intent_conversation_extra(
    approved_conversation_context: Any = None,
    last_qa_resolution: Any = None,
) -> dict[str, Any]:
    """Build bounded chat-history context for the intent routing prompt."""
    last_qa_path = "unknown"
    if last_qa_resolution is not None:
        last_qa_path = str(_enum_value(getattr(last_qa_resolution, "path", "unknown")))

    if approved_conversation_context is None:
        return {
            "conversation_context_status": "not_run",
            "conversation_retrieval_ran": False,
            "has_approved_conversation": False,
            "approved_conversation_count": 0,
            "last_qa_path": last_qa_path,
        }

    history = list(getattr(approved_conversation_context, "approved_conversation_history", []) or [])
    expected_types = [
        _enum_value(item)
        for item in list(getattr(approved_conversation_context, "extracted_expected_response_types", []) or [])[:4]
    ]

    return {
        "conversation_context_status": getattr(approved_conversation_context, "conversation_context_status", "unknown"),
        "conversation_retrieval_ran": bool(getattr(approved_conversation_context, "conversation_retrieval_ran", False)),
        "has_approved_conversation": bool(history),
        "approved_conversation_count": int(getattr(approved_conversation_context, "approved_conversation_count", len(history)) or 0),
        "last_qa_path": last_qa_path,
        "approved_conversation_history": [
            _compact_intent_history_item(item)
            for item in history[:3]
        ],
        "human_supporting_questions": [
            _compact_intent_question(question)
            for question in list(getattr(approved_conversation_context, "human_supporting_questions", []) or [])[:2]
        ],
        "reminder_supporting_questions": [
            _compact_intent_question(question)
            for question in list(getattr(approved_conversation_context, "reminder_supporting_questions", []) or [])[:2]
        ],
        "extracted_expected_response_types": expected_types,
    }


def intent_classification_schema() -> dict[str, Any]:
    """Return the minimal final-branch contract shared by both classifiers."""

    return {
        "type": "object",
        "properties": {
            "intent": {
                "type": "string",
                # The safe non-mutating route is also the generated example
                # and fallback default, avoiding first-enum mutation bias.
                "enum": [
                    Intent.GENERAL_RESPONSE.value,
                    Intent.KNOWLEDGE_FACTS.value,
                    Intent.REMINDER.value,
                    Intent.CLARIFICATION.value,
                ],
                "default": Intent.GENERAL_RESPONSE.value,
            },
            "confidence": {"type": "number"},
        },
        "required": ["intent", "confidence"],
    }


class OllamaIntentClassifier:
    def __init__(
        self,
        llm: LLMClient,
        prompt_registry: PromptRegistry = DEFAULT_PROMPT_REGISTRY,
        min_confidence: float = PromptPolicySettings.intent_min_confidence,
    ) -> None:
        self.llm = llm
        self.prompt_registry = prompt_registry
        self.min_confidence = min_confidence

    def classify(
        self, 
        request: Any, 
        rewritten_query: str,
        last_qa_resolution: Any = None,
        approved_conversation_context: Any = None,
        **kwargs
    ) -> Intent:
        explicit_intent = request.metadata.get("intent")
        if explicit_intent:
            try:
                return Intent(explicit_intent)
            except ValueError:
                pass
        if _is_authoritative_outbound_action(last_qa_resolution):
            return Intent.GENERAL_RESPONSE
        schema = intent_classification_schema()
        try:
            payload = self.llm.generate_json(
                task=LLMTask.INTENT,
                system_prompt=self.prompt_registry.system("intent_classifier"),
                user_prompt=self.prompt_registry.user(
                    PromptContext(
                        stage="intent_classifier",
                        user_id=request.user_id,
                        rewritten_query=rewritten_query,
                        metadata=request.metadata,
                        platform_context=request.platform_context,
                        extra=build_intent_conversation_extra(
                            approved_conversation_context=approved_conversation_context,
                            last_qa_resolution=last_qa_resolution,
                        ),
                    )
                ),
                schema=schema,
            )
            if is_structured_fallback(payload):
                return Intent.GENERAL_RESPONSE
            validate_json_schema(payload, schema)
        except Exception:
            return Intent.GENERAL_RESPONSE
        if float(payload.get("confidence", 0.0)) < self.min_confidence:
            return Intent.GENERAL_RESPONSE

        intent = Intent(str(payload["intent"]))
        if intent is Intent.CLARIFICATION and not _has_pending_clarification(last_qa_resolution):
            return Intent.GENERAL_RESPONSE
        return intent
