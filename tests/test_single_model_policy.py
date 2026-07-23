from __future__ import annotations

from dataclasses import fields, replace
from types import MethodType
from typing import Any

import pytest

from assistant_rag.llm import LLMTask, OllamaLLMClient, OllamaModelRouter, uses_onnx_runtime
from assistant_rag.onnx_llm import ONNXLLMClient
from assistant_rag.service_wait import configured_ollama_models
from assistant_rag.settings import (
    OllamaSettings,
    PRODUCTION_LLM_MODEL,
    ProductionSettings,
    ReminderTargetResolverSettings,
    RetrievalValidationSettings,
)


def _configured_models(settings: OllamaSettings) -> list[str]:
    models: list[str] = []
    for field_info in fields(settings):
        if not field_info.name.startswith("model_"):
            continue
        model = getattr(settings, field_info.name)
        if model and model not in models:
            models.append(model)
    return models


def test_production_defaults_use_exactly_one_qwen_model() -> None:
    settings = ProductionSettings()
    router = OllamaModelRouter(settings.ollama)

    assert _configured_models(settings.ollama) == [PRODUCTION_LLM_MODEL]
    assert {
        router.model_for_task(task)
        for task in LLMTask
    } == {PRODUCTION_LLM_MODEL}
    assert all(
        getattr(settings.ollama, field_info.name) is None
        for field_info in fields(settings.ollama)
        if field_info.name.startswith("model_")
        and field_info.name.endswith("_fallback")
    )
    assert not any(
        uses_onnx_runtime(router.model_for_task(task))
        for task in LLMTask
    )
    assert configured_ollama_models(settings) == [PRODUCTION_LLM_MODEL]


def test_default_warmup_loads_one_ollama_model_and_no_onnx_model() -> None:
    settings = ProductionSettings().ollama
    router = OllamaModelRouter(settings)
    ollama = OllamaLLMClient(settings, router)
    requests: list[dict[str, Any]] = []

    def record_request(
        self: OllamaLLMClient,
        path: str,
        body: dict[str, Any],
        *,
        timeout: float,
    ) -> dict[str, Any]:
        del self
        requests.append({"path": path, "body": body, "timeout": timeout})
        return {}

    ollama._request_json = MethodType(record_request, ollama)  # type: ignore[method-assign]

    assert ollama.warmup_models() == [PRODUCTION_LLM_MODEL]
    assert [request["body"]["model"] for request in requests] == [
        PRODUCTION_LLM_MODEL,
    ]
    assert all(
        request["body"]["keep_alive"] == settings.keep_alive
        for request in requests
    )
    assert ONNXLLMClient(router).preload_models() == []


def test_qwen_request_explicitly_disables_thinking_and_uses_task_budgets() -> None:
    settings = ProductionSettings().ollama
    ollama = OllamaLLMClient(settings, OllamaModelRouter(settings))
    requests: list[dict[str, Any]] = []

    def record_request(
        self: OllamaLLMClient,
        path: str,
        body: dict[str, Any],
        *,
        timeout: float,
    ) -> dict[str, Any]:
        del self
        requests.append({"path": path, "body": body, "timeout": timeout})
        return {"message": {"content": "ready"}}

    ollama._request_json = MethodType(record_request, ollama)  # type: ignore[method-assign]

    assert ollama._chat_raw(
        task=LLMTask.ANSWER,
        system_prompt="system",
        user_prompt="user",
        format_schema=None,
    ) == "ready"
    assert len(requests) == 1
    request = requests[0]
    assert request["path"] == "/api/chat"
    assert request["body"]["model"] == PRODUCTION_LLM_MODEL
    assert request["body"]["think"] is False
    assert request["body"]["options"] == {
        "temperature": settings.temperature_answer,
        "num_ctx": settings.num_ctx_answer,
        "num_predict": settings.num_predict_answer,
    }
    assert request["timeout"] == settings.timeout_answer


@pytest.mark.parametrize("field_name", ("model_answer", "model_answer_fallback"))
def test_production_rejects_any_other_primary_or_fallback_model(
    field_name: str,
) -> None:
    invalid_model_settings = replace(
        OllamaSettings(),
        **{field_name: "other-model:latest"},
    )

    with pytest.raises(ValueError, match="only the qwen3.5:2b"):
        ProductionSettings(ollama=invalid_model_settings)


def test_production_rejects_an_empty_primary_model_route() -> None:
    with pytest.raises(ValueError, match="missing primary routes"):
        ProductionSettings(
            ollama=replace(OllamaSettings(), model_answer=""),
        )


def test_environment_cannot_replace_the_production_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OLLAMA_ANSWER_MODEL", "other-model:latest")

    with pytest.raises(ValueError, match="only the qwen3.5:2b"):
        ProductionSettings.from_env()


@pytest.mark.parametrize(
    ("settings_override", "expected_name"),
    (
        (
            {
                "retrieval_validation": replace(
                    RetrievalValidationSettings(),
                    knowledge_llm_validation_model="other-model:latest",
                )
            },
            "KNOWLEDGE_LLM_VALIDATION_MODEL",
        ),
        (
            {
                "retrieval_validation": replace(
                    RetrievalValidationSettings(),
                    reminder_llm_validation_model="other-model:latest",
                )
            },
            "REMINDER_LLM_VALIDATION_MODEL",
        ),
        (
            {
                "reminder_resolver": replace(
                    ReminderTargetResolverSettings(),
                    reminder_llm_rerank_model="other-model:latest",
                )
            },
            "REMINDER_LLM_RERANK_MODEL",
        ),
    ),
)
def test_production_rejects_non_qwen_call_time_model_overrides(
    settings_override: dict[str, object],
    expected_name: str,
) -> None:
    with pytest.raises(ValueError, match=expected_name):
        ProductionSettings(**settings_override)  # type: ignore[arg-type]


def test_call_time_model_overrides_may_use_the_single_production_model() -> None:
    settings = ProductionSettings(
        retrieval_validation=replace(
            RetrievalValidationSettings(),
            knowledge_llm_validation_model=PRODUCTION_LLM_MODEL,
            reminder_llm_validation_model=PRODUCTION_LLM_MODEL,
        ),
        reminder_resolver=replace(
            ReminderTargetResolverSettings(),
            reminder_llm_rerank_model=PRODUCTION_LLM_MODEL,
        ),
    )

    assert (
        settings.retrieval_validation.knowledge_llm_validation_model
        == PRODUCTION_LLM_MODEL
    )


def test_task_context_and_output_budgets_match_qwen_2b_policy() -> None:
    router = OllamaModelRouter(ProductionSettings().ollama)
    expected = {
        LLMTask.QUERY_REWRITE: (2048, 128),
        LLMTask.LAST_QA: (4096, 128),
        LLMTask.INTENT: (4096, 64),
        LLMTask.ACTION_EXTRACTION: (4096, 160),
        LLMTask.KNOWLEDGE_ACTION_EXTRACTION: (8192, 2048),
        LLMTask.KNOWLEDGE_ACTION_VALIDATION: (16384, 1024),
        LLMTask.KNOWLEDGE_CONTENT_FINALIZATION: (16384, 2048),
        LLMTask.REMINDER_ACTION_EXTRACTION: (8192, 2048),
        LLMTask.REMINDER_ACTION_VALIDATION: (16384, 1024),
        LLMTask.REMINDER_CONTENT_FINALIZATION: (8192, 128),
        LLMTask.GENERATE_CLARIFICATION: (4096, 160),
        LLMTask.GENERATE_HUMAN_SUPPORTING: (4096, 160),
        LLMTask.CLARIFICATION_MERGE: (4096, 256),
        LLMTask.ANSWER: (8192, 1536),
        LLMTask.WRITING: (8192, 2048),
        LLMTask.RISKY_ACTION: (4096, 192),
        LLMTask.RETRIEVAL_VALIDATION: (8192, 512),
        LLMTask.GENERAL_SUB_BRANCH_DETECTION: (2048, 96),
        LLMTask.CONTENT_COMPOSER_REACT: (2048, 160),
        LLMTask.ACTION_PLANNING: (4096, 256),
    }

    assert {
        task: (
            router.decision_for_task(task).num_ctx,
            router.decision_for_task(task).num_predict,
        )
        for task in LLMTask
    } == expected
