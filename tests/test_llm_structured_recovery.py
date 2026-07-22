from __future__ import annotations

from types import MethodType
from typing import Any

import pytest

from assistant_rag.hybrid_llm import HybridLLMClient
from assistant_rag.llm import (
    LLMTask,
    OllamaLLMClient,
    OllamaModelRouter,
    StructuredOutputInvariantError,
    is_structured_fallback,
    normalize_structured_output,
    structured_fallback_payload,
    validate_json_schema,
)
from assistant_rag.observability import start_trace
from assistant_rag.prompts import REMINDER_ACTION_EXTRACTION_SCHEMA
from assistant_rag.settings import CAPABLE_LLM_MODEL, FAST_LLM_MODEL, OllamaSettings


def test_structured_normalization_is_lossless_and_schema_authorized() -> None:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["decision", "lead_minutes", "assessment"],
        "properties": {
            "decision": {"type": "string", "enum": ["PASS", "FAIL"]},
            "lead_minutes": {"type": "integer"},
            "assessment": {
                "type": "object",
                "additionalProperties": False,
                "required": ["state"],
                "properties": {
                    "state": {"type": "string", "enum": ["ON", "OFF"]}
                },
            },
        },
    }

    normalized, changed = normalize_structured_output(
        {
            "decision": "pass",
            "lead_minutes": 30.0,
            "assessment": {"state": "on", "explanation": "redundant"},
            "reasoning": "forbidden by the declared schema",
        },
        schema,
    )

    assert changed
    assert normalized == {
        "decision": "PASS",
        "lead_minutes": 30,
        "assessment": {"state": "ON"},
    }
    integral_only, integral_changed = normalize_structured_output(
        {"lead_minutes": 30.0},
        {
            "type": "object",
            "properties": {"lead_minutes": {"type": "integer"}},
        },
    )
    assert integral_changed
    assert type(integral_only["lead_minutes"]) is int


def test_structured_fallback_provenance_does_not_expand_schema_or_ask() -> None:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["question_text", "confidence"],
        "properties": {
            "question_text": {"type": "string"},
            "confidence": {"type": "number"},
        },
    }

    payload = structured_fallback_payload(
        task=LLMTask.GENERATE_CLARIFICATION,
        schema=schema,
        user_prompt="Runtime context:\n{}",
        error=ValueError("invalid output"),
    )

    assert is_structured_fallback(payload)
    assert set(payload) == set(schema["properties"])
    assert payload == {"question_text": "", "confidence": 0.0}


def test_last_qa_fallback_matches_dynamic_no_question_schema() -> None:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["relationship", "confidence"],
        "properties": {
            "relationship": {
                "type": "string",
                "enum": ["unrelated_or_uncertain", "normal_follow_up"],
            },
            "confidence": {"type": "number"},
        },
    }

    payload = structured_fallback_payload(
        task=LLMTask.LAST_QA,
        schema=schema,
        user_prompt="Runtime context:\n{}",
        error=ValueError("invalid output"),
    )

    assert is_structured_fallback(payload)
    assert payload == {
        "relationship": "unrelated_or_uncertain",
        "confidence": 0.0,
    }


class _UnusedOnnx:
    def __init__(self) -> None:
        self.last_error_by_task: dict[LLMTask, str] = {}


def _reminder_extraction_payload(
    *,
    retrieval_text: str = "submit the expense report",
) -> dict[str, Any]:
    return {
        "action": "add",
        "retrieval_text": retrieval_text,
        "field_values": {
            "subject": "Submit the expense report",
            "raw_reminder": "Remind me to submit the expense report",
            "user_timezone": "",
            "original_time_text": "",
        },
        "confidence": 0.96,
    }


def test_reminder_extraction_valid_first_result_uses_one_schema_attempt() -> None:
    settings = OllamaSettings()
    ollama = OllamaLLMClient(
        settings=settings,
        router=OllamaModelRouter(settings),
    )
    calls: list[dict[str, Any]] = []

    def scripted_chat_raw(self: OllamaLLMClient, **kwargs: Any) -> str:
        del self
        calls.append(kwargs)
        return (
            '{"action":"add","retrieval_text":"submit the expense report",'
            '"field_values":{"subject":"Submit the expense report",'
            '"raw_reminder":"Remind me to submit the expense report",'
            '"user_timezone":"","original_time_text":""},'
            '"confidence":0.96}'
        )

    ollama._chat_raw = MethodType(scripted_chat_raw, ollama)  # type: ignore[method-assign]

    payload = ollama.generate_json(
        task=LLMTask.REMINDER_ACTION_EXTRACTION,
        system_prompt="system",
        user_prompt='Runtime context:\n{"query":"Remind me to submit the expense report"}',
        schema=REMINDER_ACTION_EXTRACTION_SCHEMA,
    )

    assert payload == _reminder_extraction_payload()
    assert len(calls) == 1
    assert calls[0]["attempt_mode"] == "schema"
    assert calls[0]["format_schema"] == REMINDER_ACTION_EXTRACTION_SCHEMA
    assert calls[0]["attempt_index"] == 1
    assert calls[0]["total_attempts"] == 1
    assert "previous output was invalid" not in calls[0]["user_prompt"]
    assert "native JSON Schema" in calls[0]["user_prompt"]
    assert "exactly this shape" not in calls[0]["user_prompt"]
    assert '"retrieval_text": ""' not in calls[0]["user_prompt"]


def test_reminder_extraction_does_not_coerce_legacy_field_value_objects() -> None:
    legacy_payload = {
        "action": "add",
        "retrieval_text": "submit the expense report",
        "field_values": [
            {"subject": "Submit the expense report"},
            {"raw_reminder": "Remind me to submit the expense report"},
            {"notification_time": "2026-07-19T02:00:00Z"},
        ],
        "confidence": 0.96,
    }

    with pytest.raises(
        ValueError,
        match=r"field must be object: \$\.field_values",
    ):
        validate_json_schema(legacy_payload, REMINDER_ACTION_EXTRACTION_SCHEMA)

    normalized, changed = normalize_structured_output(
        legacy_payload,
        REMINDER_ACTION_EXTRACTION_SCHEMA,
    )
    assert not changed
    assert normalized["field_values"] == legacy_payload["field_values"]
    with pytest.raises(
        ValueError,
        match=r"field must be object: \$\.field_values",
    ):
        validate_json_schema(normalized, REMINDER_ACTION_EXTRACTION_SCHEMA)


def test_reminder_extraction_invariant_failure_recovers_once_with_capable_model() -> None:
    settings = OllamaSettings()
    assert settings.json_retry_count_reminder_action_extraction == 0
    ollama = OllamaLLMClient(
        settings=settings,
        router=OllamaModelRouter(settings),
    )
    calls: list[dict[str, Any]] = []
    validations: list[str] = []

    def scripted_chat_raw(self: OllamaLLMClient, **kwargs: Any) -> str:
        del self
        calls.append(kwargs)
        if kwargs.get("fallback_for"):
            return (
                '{"action":"add","retrieval_text":"submit the expense report",'
                '"field_values":{"subject":"Submit the expense report",'
                '"raw_reminder":"Remind me to submit the expense report",'
                '"user_timezone":"","original_time_text":""},'
                '"confidence":0.96}'
            )
        return (
            '{"action":"add","retrieval_text":"","field_values":'
            '{"subject":"Submit the expense report","user_timezone":"",'
            '"original_time_text":""},'
            '"confidence":0.93}'
        )

    def require_grounded_retrieval(payload: dict[str, Any]) -> None:
        retrieval_text = str(payload["retrieval_text"]).strip()
        validations.append(retrieval_text)
        if not retrieval_text:
            raise StructuredOutputInvariantError(
                "retrieval_text must be grounded in the request"
            )

    ollama._chat_raw = MethodType(scripted_chat_raw, ollama)  # type: ignore[method-assign]
    hybrid = HybridLLMClient(ollama, _UnusedOnnx())  # type: ignore[arg-type]

    payload = hybrid.generate_json(
        task=LLMTask.REMINDER_ACTION_EXTRACTION,
        system_prompt="system",
        user_prompt='Runtime context:\n{"query":"Remind me to submit the expense report"}',
        schema=REMINDER_ACTION_EXTRACTION_SCHEMA,
        invariant_validator=require_grounded_retrieval,
    )

    assert payload == _reminder_extraction_payload()
    assert validations == ["", "submit the expense report"]
    assert len(calls) == 2
    assert [call["attempt_mode"] for call in calls] == ["schema", "schema"]
    assert [call["total_attempts"] for call in calls] == [1, 1]
    assert calls[0]["model_override"] is None
    assert calls[1]["model_override"] == CAPABLE_LLM_MODEL
    assert calls[1]["fallback_for"] == FAST_LLM_MODEL
    assert "retrieval_text must be grounded" in calls[1]["user_prompt"]
    assert LLMTask.REMINDER_ACTION_EXTRACTION not in ollama.last_error_by_task


@pytest.mark.parametrize("fallback_succeeds", (True, False))
def test_last_qa_cross_model_recovery_is_one_json_then_one_schema_attempt(
    fallback_succeeds: bool,
) -> None:
    settings = OllamaSettings()
    ollama = OllamaLLMClient(
        settings=settings,
        router=OllamaModelRouter(settings),
    )
    calls: list[dict[str, Any]] = []

    def scripted_chat_raw(self: OllamaLLMClient, **kwargs: Any) -> str:
        del self
        calls.append(kwargs)
        if kwargs.get("fallback_for") and fallback_succeeds:
            return '{"relationship":"normal_follow_up","confidence":0.96}'
        return "not-json"

    ollama._chat_raw = MethodType(scripted_chat_raw, ollama)  # type: ignore[method-assign]
    hybrid = HybridLLMClient(ollama, _UnusedOnnx())  # type: ignore[arg-type]
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["relationship", "confidence"],
        "properties": {
            "relationship": {
                "type": "string",
                "enum": ["unrelated_or_uncertain", "normal_follow_up"],
            },
            "confidence": {"type": "number"},
        },
    }

    payload = hybrid.generate_json(
        task=LLMTask.LAST_QA,
        system_prompt="system",
        user_prompt='Runtime context:\n{"stage":"last_qa"}',
        schema=schema,
    )

    assert len(calls) == 2
    assert [call["attempt_mode"] for call in calls] == ["json", "schema"]
    assert calls[0]["model_override"] is None
    assert calls[1]["model_override"] == settings.model_last_qa_fallback
    assert calls[1]["fallback_for"] == settings.model_last_qa
    if fallback_succeeds:
        assert payload == {
            "relationship": "normal_follow_up",
            "confidence": 0.96,
        }
        assert LLMTask.LAST_QA not in ollama.last_error_by_task
    else:
        assert is_structured_fallback(payload)
        assert payload == {
            "relationship": "unrelated_or_uncertain",
            "confidence": 0.0,
        }


class _Router:
    def model_for_task(self, task: LLMTask) -> str:
        return "microsoft/Phi-4-mini-instruct-onnx"


class _FallbackOllama:
    def __init__(self) -> None:
        self.router = _Router()
        self.settings = OllamaSettings()
        self.last_error_by_task: dict[LLMTask, str] = {}
        self.calls: list[dict[str, Any]] = []

    def _fallback_model_for_task(self, task: LLMTask) -> str | None:
        return getattr(self.settings, f"model_{task.value}_fallback", None)

    def generate_json(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {"approved": True, "confidence": 0.99}


class _FailingOnnx:
    def __init__(self) -> None:
        self.last_error_by_task: dict[LLMTask, str] = {}

    def generate_json(self, **kwargs: Any) -> dict[str, Any]:
        task = kwargs["task"]
        self.last_error_by_task[task] = "synthetic structured failure"
        return structured_fallback_payload(
            task=task,
            schema=kwargs["schema"],
            user_prompt=kwargs["user_prompt"],
            error=ValueError("synthetic structured failure"),
        )


def test_hybrid_recovers_onnx_finalization_through_configured_ollama_model() -> None:
    ollama = _FallbackOllama()
    onnx = _FailingOnnx()
    client = HybridLLMClient(ollama, onnx)  # type: ignore[arg-type]

    payload = client.generate_json(
        task=LLMTask.KNOWLEDGE_CONTENT_FINALIZATION,
        system_prompt="system",
        user_prompt="Runtime context:\n{}",
        schema={
            "type": "object",
            "required": ["approved", "confidence"],
            "properties": {
                "approved": {"type": "boolean"},
                "confidence": {"type": "number"},
            },
        },
    )

    assert payload == {"approved": True, "confidence": 0.99}
    assert len(ollama.calls) == 1
    assert (
        ollama.calls[0]["model_override"]
        == OllamaSettings.model_knowledge_content_finalization_fallback
    )
    assert LLMTask.KNOWLEDGE_CONTENT_FINALIZATION not in onnx.last_error_by_task


def test_every_capable_task_has_fast_cross_model_recovery() -> None:
    settings = OllamaSettings()
    for task in (
        LLMTask.LAST_QA,
        LLMTask.KNOWLEDGE_ACTION_VALIDATION,
        LLMTask.KNOWLEDGE_CONTENT_FINALIZATION,
        LLMTask.REMINDER_ACTION_VALIDATION,
        LLMTask.REMINDER_CONTENT_FINALIZATION,
        LLMTask.ANSWER,
        LLMTask.WRITING,
        LLMTask.RETRIEVAL_VALIDATION,
        LLMTask.ACTION_PLANNING,
    ):
        primary = getattr(settings, f"model_{task.value}")
        fallback = getattr(settings, f"model_{task.value}_fallback")
        assert primary == CAPABLE_LLM_MODEL
        assert fallback == FAST_LLM_MODEL


class _FailingChatOnnx:
    def __init__(self) -> None:
        self.last_error_by_task: dict[LLMTask, str] = {}

    def chat(self, **kwargs: Any) -> str:
        task = kwargs["task"]
        self.last_error_by_task[task] = "synthetic ONNX model-resolution failure"
        raise RuntimeError("synthetic ONNX model-resolution failure")


class _FallbackChatOllama:
    def __init__(self) -> None:
        self.router = _Router()
        self.settings = OllamaSettings()
        self.last_error_by_task: dict[LLMTask, str] = {}
        self.calls: list[dict[str, Any]] = []

    def _fallback_model_for_task(self, _task: LLMTask) -> str:
        return "qwen3.5:4b"

    def _chat_raw(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        return "recovered answer"


def test_hybrid_chat_records_one_logical_answer_stage_across_onnx_recovery() -> None:
    ollama = _FallbackChatOllama()
    onnx = _FailingChatOnnx()
    client = HybridLLMClient(ollama, onnx)  # type: ignore[arg-type]
    trace = start_trace("hybrid-answer-recovery")

    response = client.chat(
        task=LLMTask.ANSWER,
        system_prompt="system",
        user_prompt='Runtime context:\n{"stage":"answer_generation"}',
    )

    assert response == "recovered answer"
    assert len(ollama.calls) == 1
    assert ollama.calls[0]["fallback_for"] == "microsoft/Phi-4-mini-instruct-onnx"
    assert LLMTask.ANSWER not in onnx.last_error_by_task
    assert [stage.stage for stage in trace.summary().stages] == [
        "llm_answer_generation"
    ]
