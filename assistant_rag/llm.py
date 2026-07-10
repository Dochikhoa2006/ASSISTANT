"""Ollama-backed LLM integration."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import Enum
import json
import logging
import time
from typing import Any, Protocol
from urllib import error, request

from .contracts import Intent
from .metrics import GLOBAL_METRICS
from .observability import StageTimer, current_trace
from .prompts import DEFAULT_PROMPT_REGISTRY, PromptContext, PromptRegistry
from .settings import OllamaSettings, PromptPolicySettings

logger = logging.getLogger(__name__)


class LLMTask(str, Enum):
    QUERY_REWRITE = "query_rewrite"
    LAST_QA = "last_qa"
    INTENT = "intent"
    ACTION_EXTRACTION = "action_extraction"
    GENERATE_CLARIFICATION = "generate_clarification"
    GENERATE_HUMAN_SUPPORTING = "generate_human_supporting"
    GENERATE_REMINDER_SUPPORTING = "generate_reminder_supporting"
    CLARIFICATION_MERGE = "clarification_merge"
    ANSWER = "answer"
    WRITING = "writing"
    RISKY_ACTION = "risky_action"
    RETRIEVAL_VALIDATION = "retrieval_validation"
    GENERAL_SUB_BRANCH_DETECTION = "general_sub_branch_detection"
    CONTENT_COMPOSER_REACT = "content_composer_react"
    ACTION_PLANNING = "action_planning"


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
    ) -> dict[str, Any]:
        ...

    def chat(self, *, task: LLMTask, system_prompt: str, user_prompt: str) -> str:
        ...


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

    def generate_json(
        self,
        *,
        task: LLMTask,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        with StageTimer(f"llm_{task.value}"):
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
                        expect_json=True,
                        attempt_index=attempt_index,
                        total_attempts=len(attempt_plan),
                        attempt_mode=attempt_mode,
                    )
                    payload = parse_json_object(raw)
                    validate_json_schema(payload, schema)
                    trace = current_trace()
                    if trace and trace._active_timers:
                        timer = trace._active_timers[-1]
                        timer.metadata["structured_attempts"] = attempt_index
                        timer.metadata["structured_configured_attempts"] = total_attempts
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
        with StageTimer(f"llm_{task.value}"):
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
        if fallback_for:
            print(f"LLM used: {model} (task: {task.value}, fallback_for: {fallback_for}{attempt_suffix})")
        else:
            print(f"LLM used: {model} (task: {task.value}{attempt_suffix})")
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
        if task == LLMTask.ANSWER:
            return self.settings.model_answer_fallback
        return None

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
    stripped = raw.strip()
    if not stripped:
        raise ValueError("Expected JSON object, got empty LLM response")
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as exc:
        extracted = _extract_json_object(stripped)
        if extracted is None:
            preview = stripped[:240].replace("\n", "\\n")
            raise ValueError(f"Expected JSON object, got non-JSON response: {preview!r}") from exc
        try:
            parsed = json.loads(extracted)
        except json.JSONDecodeError as extracted_exc:
            preview = extracted[:240].replace("\n", "\\n")
            raise ValueError(f"Extracted invalid JSON object from LLM response: {preview!r}") from extracted_exc
    if not isinstance(parsed, dict):
        if isinstance(parsed, str) and "{" in parsed:
            return parse_json_object(parsed)
        raise ValueError("Expected JSON object")
    return parsed


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
    if mode == "schema":
        return user_prompt

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

    return (
        f"{user_prompt}\n\n"
        "Structured output repair instruction:\n"
        "Return exactly one JSON object and nothing else. No markdown, no code fence, no prose.\n"
        f"The JSON object must have exactly this shape:\n{template}{rules_text}"
    )


def _extract_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape = False
    for idx, char in enumerate(text[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:idx + 1]

    return None


def validate_json_schema(payload: dict[str, Any], schema: dict[str, Any]) -> None:
    required = schema.get("required") or []
    for key in required:
        if key not in payload:
            raise ValueError(f"Structured output missing required field: {key}")
    properties = schema.get("properties") or {}
    for key, rules in properties.items():
        if key not in payload:
            continue
        value = payload[key]
        expected_type = rules.get("type")
        if expected_type == "string" and not isinstance(value, str):
            raise ValueError(f"Structured output field must be string: {key}")
        if expected_type == "number" and (isinstance(value, bool) or not isinstance(value, (int, float))):
            raise ValueError(f"Structured output field must be number: {key}")
        if expected_type == "boolean" and not isinstance(value, bool):
            raise ValueError(f"Structured output field must be boolean: {key}")
        if expected_type == "array" and not isinstance(value, list):
            raise ValueError(f"Structured output field must be array: {key}")
        if expected_type == "object" and not isinstance(value, dict):
            raise ValueError(f"Structured output field must be object: {key}")
        enum_values = rules.get("enum")
        if enum_values is not None and value not in enum_values:
            raise ValueError(f"Structured output field has unsupported value: {key}")


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
        payload.update({
            "interaction_detected": False,
            "interaction_type": "",
            "question_source": "none",
            "matched_question": "",
            "confidence": 0.0,
            "llm_suggested_skip_broad_retrieval": False,
        })
    elif task == LLMTask.INTENT:
        payload.update({
            "intent": Intent.GENERAL_RESPONSE.value,
            "confidence": 1.0,
            "multi_intent": False,
            "requires_clarification": False,
            "reason_summary": reason,
        })
    elif task == LLMTask.ACTION_EXTRACTION:
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
            "question_text": DEFAULT_PROMPT_REGISTRY.message("fallback_message"),
            "question_source": "clarification_question",
            "purpose": "resolve_missing_info",
            "confidence": 1.0,
            "should_ask": True,
            "expected_response_type": "unknown",
            "reason_summary": reason,
        })
    elif task == LLMTask.GENERATE_HUMAN_SUPPORTING:
        payload.update({"questions": []})
    elif task == LLMTask.GENERATE_REMINDER_SUPPORTING:
        payload.update({
            "question_text": "",
            "question_source": "reminder_supporting_question",
            "purpose": "reminder_followup",
            "confidence": 0.0,
            "should_ask": False,
            "expected_response_type": "unknown",
            "reason_summary": reason,
        })
    elif task == LLMTask.CLARIFICATION_MERGE:
        payload.update({
            "answered_clarification": False,
            "merged_query": str(context.get("rewritten_query") or context.get("raw_query") or ""),
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
    elif task == LLMTask.ACTION_PLANNING:
        payload.update({"confidence": 0.0, "reason_summary": reason})
    elif task in {LLMTask.CONTENT_COMPOSER_REACT, LLMTask.ANSWER, LLMTask.WRITING}:
        payload.update({"reason_summary": reason})

    if "sheet_name" in payload:
        payload.update({"sheet_name": "Sheet1", "columns": [], "suggested_rows": [], "confidence": 0.0, "reason_summary": reason})
    if "title" in payload and "sections" in payload:
        payload.update({"title": "Untitled", "sections": [], "confidence": 0.0, "reason_summary": reason})
    if "presentation_title" in payload:
        payload.update({"presentation_title": "Untitled", "slides": [], "confidence": 0.0, "reason_summary": reason})

    return payload


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
            return Intent(explicit_intent)
        schema = {
            "type": "object",
            "properties": {
                "intent": {
                    "type": "string",
                    "enum": [
                        "clarification",
                        "general_response",
                        "knowledge_facts",
                        "reminder",
                    ],
                },
                "confidence": {"type": "number"},
                "multi_intent": {"type": "boolean"},
                "requires_clarification": {"type": "boolean"},
            },
            "required": [
                "intent",
                "confidence",
                "multi_intent",
                "requires_clarification",
            ],
        }
        try:
            payload = self.llm.generate_json(
                task=LLMTask.INTENT,
                system_prompt=self.prompt_registry.system("intent_classifier"),
                user_prompt=self.prompt_registry.user(
                    PromptContext(
                        stage="intent_classifier",
                        user_id=request.user_id,
                        raw_query=request.raw_query,
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
            validate_json_schema(payload, schema)
        except Exception:
            return Intent.GENERAL_RESPONSE
        if (
            float(payload.get("confidence", 0.0)) < self.min_confidence
            or bool(payload.get("requires_clarification"))
            or bool(payload.get("multi_intent"))
        ):
            return Intent.CLARIFICATION
        return Intent(str(payload["intent"]))
