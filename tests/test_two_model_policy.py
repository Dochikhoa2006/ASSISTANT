from __future__ import annotations

from dataclasses import fields, replace
from types import MethodType
from typing import Any

import pytest

from assistant_rag.llm import LLMTask, OllamaLLMClient, OllamaModelRouter, uses_onnx_runtime
from assistant_rag.onnx_llm import ONNXLLMClient
from assistant_rag.service_wait import configured_ollama_models
from assistant_rag.settings import (
    CAPABLE_LLM_MODEL,
    FAST_LLM_MODEL,
    OllamaSettings,
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


def test_production_defaults_use_exactly_the_warmed_two_model_pool() -> None:
    settings = ProductionSettings()
    router = OllamaModelRouter(settings.ollama)

    assert set(_configured_models(settings.ollama)) == {
        FAST_LLM_MODEL,
        CAPABLE_LLM_MODEL,
    }
    assert {
        router.model_for_task(task)
        for task in LLMTask
    } == {FAST_LLM_MODEL, CAPABLE_LLM_MODEL}
    assert not any(
        uses_onnx_runtime(router.model_for_task(task))
        for task in LLMTask
    )
    assert configured_ollama_models(settings) == [
        FAST_LLM_MODEL,
        CAPABLE_LLM_MODEL,
    ]


def test_default_warmup_deduplicates_two_ollama_models_and_no_onnx_model() -> None:
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

    assert ollama.warmup_models() == [FAST_LLM_MODEL, CAPABLE_LLM_MODEL]
    assert [request["body"]["model"] for request in requests] == [
        FAST_LLM_MODEL,
        CAPABLE_LLM_MODEL,
    ]
    assert all(
        request["body"]["keep_alive"] == settings.keep_alive
        for request in requests
    )
    assert ONNXLLMClient(router).preload_models() == []


def test_production_rejects_a_third_primary_or_fallback_model() -> None:
    three_model_settings = replace(
        OllamaSettings(),
        model_answer="third-model:latest",
    )

    with pytest.raises(ValueError, match="at most two generative LLM models"):
        ProductionSettings(ollama=three_model_settings)


def test_environment_cannot_reintroduce_a_third_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OLLAMA_ANSWER_MODEL", "third-model:latest")

    with pytest.raises(ValueError, match="at most two generative LLM models"):
        ProductionSettings.from_env()


@pytest.mark.parametrize(
    ("settings_override", "expected_name"),
    (
        (
            {
                "retrieval_validation": replace(
                    RetrievalValidationSettings(),
                    knowledge_llm_validation_model="third-model:latest",
                )
            },
            "KNOWLEDGE_LLM_VALIDATION_MODEL",
        ),
        (
            {
                "retrieval_validation": replace(
                    RetrievalValidationSettings(),
                    reminder_llm_validation_model="third-model:latest",
                )
            },
            "REMINDER_LLM_VALIDATION_MODEL",
        ),
        (
            {
                "reminder_resolver": replace(
                    ReminderTargetResolverSettings(),
                    reminder_llm_rerank_model="third-model:latest",
                )
            },
            "REMINDER_LLM_RERANK_MODEL",
        ),
    ),
)
def test_production_rejects_unwarmed_call_time_model_overrides(
    settings_override: dict[str, object],
    expected_name: str,
) -> None:
    with pytest.raises(ValueError, match=expected_name):
        ProductionSettings(**settings_override)  # type: ignore[arg-type]


def test_call_time_model_overrides_may_reuse_the_warmed_pool() -> None:
    settings = ProductionSettings(
        retrieval_validation=replace(
            RetrievalValidationSettings(),
            knowledge_llm_validation_model=CAPABLE_LLM_MODEL,
            reminder_llm_validation_model=FAST_LLM_MODEL,
        ),
        reminder_resolver=replace(
            ReminderTargetResolverSettings(),
            reminder_llm_rerank_model=FAST_LLM_MODEL,
        ),
    )

    assert settings.retrieval_validation.knowledge_llm_validation_model == CAPABLE_LLM_MODEL


def test_task_context_and_output_budgets_match_the_two_model_policy() -> None:
    router = OllamaModelRouter(ProductionSettings().ollama)
    expected = {
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

    assert {
        task: (
            router.decision_for_task(task).num_ctx,
            router.decision_for_task(task).num_predict,
        )
        for task in LLMTask
    } == expected
