from __future__ import annotations

from typing import Any

from assistant_rag.hybrid_llm import HybridLLMClient
from assistant_rag.llm import (
    LLMTask,
    is_structured_fallback,
    normalize_structured_output,
    structured_fallback_payload,
)
from assistant_rag.settings import OllamaSettings


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


def test_every_onnx_mutation_stage_has_cross_engine_recovery() -> None:
    settings = OllamaSettings()
    for task in (
        LLMTask.KNOWLEDGE_ACTION_VALIDATION,
        LLMTask.KNOWLEDGE_CONTENT_FINALIZATION,
        LLMTask.REMINDER_ACTION_VALIDATION,
        LLMTask.REMINDER_CONTENT_FINALIZATION,
        LLMTask.RETRIEVAL_VALIDATION,
    ):
        primary = getattr(settings, f"model_{task.value}")
        fallback = getattr(settings, f"model_{task.value}_fallback")
        assert "onnx" in primary.casefold()
        assert fallback
        assert "onnx" not in fallback.casefold()
