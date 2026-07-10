"""Ollama-backed LLM integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
import time
from typing import Any, Protocol
from urllib import error, request

from .contracts import Intent
from .metrics import GLOBAL_METRICS
from .observability import StageTimer, current_trace
from .prompts import DEFAULT_PROMPT_REGISTRY, PromptContext, PromptRegistry
from .settings import OllamaSettings, PromptPolicySettings


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
        model = self.model_for_task(task)
        temperature = self.temperature_for_task(task)
        reasons = {
            LLMTask.QUERY_REWRITE: "fast rewrite with low variance",
            LLMTask.LAST_QA: "fast temporary-context decision",
            LLMTask.INTENT: "fast branch classification",
            LLMTask.ACTION_EXTRACTION: "balanced structured action extraction",
            LLMTask.GENERATE_CLARIFICATION: "targeted clarification question",
            LLMTask.GENERATE_HUMAN_SUPPORTING: "optional human supporting questions",
            LLMTask.GENERATE_REMINDER_SUPPORTING: "reminder-specific follow-up actions",
            LLMTask.CLARIFICATION_MERGE: "semantic merge for clarification answers",
            LLMTask.ANSWER: "writing model for final user-facing text",
            LLMTask.WRITING: "writing model for long-form generated content",
            LLMTask.RISKY_ACTION: "accurate model for risky mutation analysis",
            LLMTask.GENERAL_SUB_BRANCH_DETECTION: "classification for general response sub-branch",
            LLMTask.CONTENT_COMPOSER_REACT: "ReAct loop for tool selection and reasoning",
            LLMTask.ACTION_PLANNING: "planning actions for knowledge and reminder branches",
        }
        
        return ModelDecision(
            task=task,
            model=model,
            temperature=temperature,
            timeout_seconds=self.timeout_for_task(task),
            num_ctx=self.num_ctx_for_task(task),
            num_predict=self.num_predict_for_task(task),
            reason_summary=reasons.get(task, "configured model policy"),
        )

    def model_for_task(self, task: LLMTask) -> str:
        if self.settings.heavy_production_enabled and self.settings.heavy_production_model:
            if task in {
                LLMTask.ANSWER,
                LLMTask.WRITING,
                LLMTask.RISKY_ACTION,
                LLMTask.RETRIEVAL_VALIDATION,
                LLMTask.ACTION_PLANNING,
            }:
                return self.settings.heavy_production_model
        if task is LLMTask.LAST_QA:
            return self.settings.last_qa_model or self.settings.fast_model
        if task is LLMTask.INTENT:
            return self.settings.intent_model or self.settings.balanced_model
        if task is LLMTask.ACTION_EXTRACTION:
            return self.settings.action_extraction_model or self.settings.balanced_model
        if task is LLMTask.CLARIFICATION_MERGE:
            return self.settings.clarification_merge_model or self.settings.balanced_model
        if task is LLMTask.GENERATE_CLARIFICATION:
            return self.settings.clarification_question_model or self.settings.balanced_model
        if task is LLMTask.GENERATE_HUMAN_SUPPORTING:
            return self.settings.human_supporting_question_model or self.settings.balanced_model
        if task is LLMTask.GENERATE_REMINDER_SUPPORTING:
            return self.settings.reminder_supporting_question_model or self.settings.balanced_model
        if task in {LLMTask.QUERY_REWRITE, LLMTask.GENERAL_SUB_BRANCH_DETECTION, LLMTask.CONTENT_COMPOSER_REACT}:
            if task is LLMTask.GENERAL_SUB_BRANCH_DETECTION and self.settings.general_sub_branch_detector_model:
                return self.settings.general_sub_branch_detector_model
            return self.settings.fast_model
        if task in {LLMTask.RISKY_ACTION}:
            return self.settings.risky_action_model or self.settings.accurate_model or self.settings.balanced_model
        if task in {LLMTask.ANSWER, LLMTask.WRITING}:
            return self.settings.writing_model or self.settings.balanced_model
        if task is LLMTask.ACTION_PLANNING:
            return self.settings.balanced_model
        return self.settings.balanced_model

    def temperature_for_task(self, task: LLMTask) -> float:
        if task is LLMTask.QUERY_REWRITE:
            return self.settings.query_rewrite_temperature
        if task is LLMTask.LAST_QA:
            return self.settings.last_qa_temperature
        if task is LLMTask.INTENT:
            return self.settings.intent_classifier_temperature
        if task in {LLMTask.ACTION_EXTRACTION, LLMTask.ACTION_PLANNING}:
            return self.settings.action_detection_temperature
        if task is LLMTask.RISKY_ACTION:
            return self.settings.risky_action_temperature
        if task is LLMTask.ANSWER:
            return self.settings.answer_temperature
        if task is LLMTask.WRITING:
            return self.settings.writing_temperature
        if task is LLMTask.GENERATE_CLARIFICATION:
            return self.settings.clarification_question_temperature
        if task is LLMTask.GENERATE_HUMAN_SUPPORTING:
            return self.settings.human_supporting_question_temperature
        if task is LLMTask.GENERATE_REMINDER_SUPPORTING:
            return self.settings.reminder_supporting_question_temperature
        return self.settings.default_temperature

    def num_ctx_for_task(self, task: LLMTask) -> int | None:
        if self.settings.heavy_production_enabled and self.settings.heavy_production_model:
            if task in {
                LLMTask.ANSWER,
                LLMTask.RISKY_ACTION,
                LLMTask.RETRIEVAL_VALIDATION,
                LLMTask.ACTION_PLANNING,
            }:
                return self.settings.num_ctx_accurate
        if task in {LLMTask.QUERY_REWRITE, LLMTask.LAST_QA, LLMTask.GENERAL_SUB_BRANCH_DETECTION, LLMTask.CONTENT_COMPOSER_REACT}:
            return self.settings.num_ctx_fast
        if task in {LLMTask.INTENT, LLMTask.ACTION_EXTRACTION, LLMTask.ACTION_PLANNING, LLMTask.CLARIFICATION_MERGE, LLMTask.GENERATE_CLARIFICATION, LLMTask.GENERATE_HUMAN_SUPPORTING, LLMTask.GENERATE_REMINDER_SUPPORTING}:
            return self.settings.num_ctx_balanced
        if task in {LLMTask.RISKY_ACTION, LLMTask.RETRIEVAL_VALIDATION}:
            return self.settings.num_ctx_accurate
        if task in {LLMTask.ANSWER, LLMTask.WRITING}:
            return self.settings.num_ctx_writing
        return None

    def num_predict_for_task(self, task: LLMTask) -> int | None:
        if task in {LLMTask.QUERY_REWRITE, LLMTask.LAST_QA, LLMTask.GENERAL_SUB_BRANCH_DETECTION, LLMTask.INTENT, LLMTask.CONTENT_COMPOSER_REACT}:
            return self.settings.num_predict_fast
        if task in {LLMTask.ANSWER, LLMTask.WRITING}:
            return self.settings.num_predict_writing
        return None

    def timeout_for_task(self, task: LLMTask) -> float:
        if task in {LLMTask.QUERY_REWRITE, LLMTask.LAST_QA, LLMTask.GENERAL_SUB_BRANCH_DETECTION, LLMTask.CONTENT_COMPOSER_REACT}:
            return self.settings.timeout_fast
        if task in {LLMTask.INTENT, LLMTask.ACTION_EXTRACTION, LLMTask.ACTION_PLANNING, LLMTask.CLARIFICATION_MERGE, LLMTask.GENERATE_CLARIFICATION, LLMTask.GENERATE_HUMAN_SUPPORTING, LLMTask.GENERATE_REMINDER_SUPPORTING}:
            return self.settings.timeout_balanced
        if task is LLMTask.RISKY_ACTION:
            return self.settings.timeout_risky_action
        if task is LLMTask.RETRIEVAL_VALIDATION:
            return self.settings.timeout_accurate
        if task in {LLMTask.ANSWER, LLMTask.WRITING}:
            return self.settings.timeout_writing
        return self.settings.timeout_seconds


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
            for _ in range(self.settings.structured_retry_count + 1):
                try:
                    raw = self._chat_raw(
                        task=task,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        format_schema=schema,
                    )
                    payload = parse_json_object(raw)
                    validate_json_schema(payload, schema)
                    self.last_error_by_task.pop(task, None)
                    return payload
                except Exception as exc:
                    GLOBAL_METRICS.increment("llm_json_parse_failures_total", task=task.value)
                    last_error = exc
                    self.last_error_by_task[task] = str(exc)
            GLOBAL_METRICS.increment("llm_failures_total", task=task.value)
            raise ValueError(f"Ollama structured output failed: {last_error}")

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
                GLOBAL_METRICS.increment("llm_failures_total", task=task.value)
                self.last_error_by_task[task] = str(exc)
                raise
            self.last_error_by_task.pop(task, None)
            return response

    def list_models(self) -> list[str]:
        payload = self._get_json("/api/tags")
        return [str(model["name"]) for model in payload.get("models", [])]

    def check(self) -> dict[str, Any]:
        started = time.perf_counter()
        models = self.list_models()
        model_checks = {
            "fast_model": self.settings.fast_model in models,
            "balanced_model": self.settings.balanced_model in models,
            "intent_model": (
                True
                if self.settings.intent_model is None
                else self.settings.intent_model in models
            ),
            "action_extraction_model": (
                True
                if self.settings.action_extraction_model is None
                else self.settings.action_extraction_model in models
            ),
            "accurate_model": (
                True
                if self.settings.accurate_model is None
                else self.settings.accurate_model in models
            ),
            "risky_action_model": (
                True
                if self.settings.risky_action_model is None
                else self.settings.risky_action_model in models
            ),
            "writing_model": (
                True
                if self.settings.writing_model is None
                else self.settings.writing_model in models
            ),
            "last_qa_model": (
                True
                if self.settings.last_qa_model is None
                else self.settings.last_qa_model in models
            ),
        }
        if self.settings.heavy_production_enabled and self.settings.heavy_production_model:
            model_checks["heavy_production_model"] = self.settings.heavy_production_model in models
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        return {
            "base_url": self.settings.base_url,
            "reachable": True,
            "installed_models": models,
            "model_checks": model_checks,
            "latency_ms": latency_ms,
        }

    def _chat_raw(
        self,
        *,
        task: LLMTask,
        system_prompt: str,
        user_prompt: str,
        format_schema: dict[str, Any] | None,
    ) -> str:
        decision = self.router.decision_for_task(task)
        body: dict[str, Any] = {
            "model": decision.model,
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
            if "prompt_eval_count" in payload:
                timer.metadata["input_count"] = payload["prompt_eval_count"]
            if "eval_count" in payload:
                timer.metadata["output_count"] = payload["eval_count"]
                
        msg = payload.get("message", {})
        content = str(msg.get("content", ""))
        thinking = str(msg.get("thinking", ""))
        if not content.strip() and thinking:
            return thinking.strip()
        return content

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
            timeout = self.settings.timeout_seconds
        try:
            with request.urlopen(req, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.URLError as exc:
            raise ConnectionError(f"Ollama request failed for {url}: {exc}") from exc

    def _get_json(self, path: str) -> dict[str, Any]:
        url = f"{self.settings.base_url.rstrip('/')}{path}"
        req = request.Request(url, method="GET")
        try:
            with request.urlopen(req, timeout=self.settings.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.URLError as exc:
            raise ConnectionError(f"Ollama request failed for {url}: {exc}") from exc


def parse_json_object(raw: str) -> dict[str, Any]:
    stripped = raw.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()
    parsed = json.loads(stripped)
    if not isinstance(parsed, dict):
        raise ValueError("Expected JSON object")
    return parsed


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
