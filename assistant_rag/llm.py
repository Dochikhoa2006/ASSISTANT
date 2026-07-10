"""Ollama-backed LLM integration."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
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
            for _ in range(retry_count + 1):
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
        configured_models = self._configured_models()
        model_checks = {
            task_name: model in models
            for task_name, model in configured_models.items()
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
        format_schema: dict[str, Any] | None,
    ) -> str:
        decision = self.router.decision_for_task(task)
        print(f"LLM used: {decision.model} (task: {task.value})")
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
