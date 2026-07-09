from __future__ import annotations

import logging
from typing import Any

from assistant_rag.config import GeneralPurposeConfig
from assistant_rag.content_composer import AnswerGenerationTool, ContentToolRegistry, ReActContentComposer
from assistant_rag.contracts import (
    ChatRequest,
    ContentComposerInput,
    ContentComposerResult,
    ContentToolResult,
    ExpectedResponseType,
    GeneralSubBranch,
    Intent,
    PersistenceMode,
    PipelineContext,
    QuestionSource,
    SubBranchPromptContext,
)
from assistant_rag.generation import LLMGeneralHITLStrategy
from assistant_rag.llm import LLMTask
from assistant_rag.prompts import DEFAULT_PROMPT_REGISTRY


class TimeoutPlannerLLM:
    def generate_json(
        self,
        *,
        task: LLMTask,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        raise ValueError("Ollama structured output failed: timed out")

    def chat(self, *, task: LLMTask, system_prompt: str, user_prompt: str) -> str:
        return "Fallback answer."


class FailingAnswerLLM:
    def generate_json(
        self,
        *,
        task: LLMTask,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        raise AssertionError("generate_json should not be called by single-tool answer fallback")

    def chat(self, *, task: LLMTask, system_prompt: str, user_prompt: str) -> str:
        raise TimeoutError("timed out")


class SecondaryTool:
    @property
    def name(self) -> str:
        return "generate_excel"

    @property
    def description(self) -> str:
        return "Secondary test tool."

    def can_handle(self, composer_input: ContentComposerInput, config: GeneralPurposeConfig) -> bool:
        return False

    def execute(self, composer_input: ContentComposerInput, config: GeneralPurposeConfig) -> ContentToolResult:
        return ContentToolResult(
            tool_name=self.name,
            output_text="Secondary output.",
            confidence=1.0,
            fallback_used=False,
            reason_summary="Secondary tool executed.",
        )


def _composer_input() -> ContentComposerInput:
    return ContentComposerInput(
        user_id="u1",
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
            expected_response_type=ExpectedResponseType.UNKNOWN,
        ),
        sub_branch_supporting_prompt="Answer directly.",
    )


def _pipeline_context() -> PipelineContext:
    return PipelineContext(
        request=ChatRequest(user_id="u1", raw_query="I want to learn Python from zero."),
        rewritten_query="I want to learn Python from zero.",
        last_qa_state=None,
        conversation_results=[],
        intent=Intent.GENERAL_RESPONSE,
    )


def _composer_result() -> ContentComposerResult:
    return ContentComposerResult(
        final_response_text="Fallback answer.",
        tool_trace_summary="",
        used_tool_names=("answer_generation",),
        confidence=0.0,
        fallback_used=True,
        reason_summary="Fallback answer used.",
        content_warnings=(),
    )


def _composer() -> ReActContentComposer:
    config = GeneralPurposeConfig(content_composer_max_iterations=1)
    llm = TimeoutPlannerLLM()
    registry = ContentToolRegistry(
        tools=[
            AnswerGenerationTool(llm=llm, prompt_registry=DEFAULT_PROMPT_REGISTRY),
            SecondaryTool(),
        ],
        config=config,
    )
    return ReActContentComposer(registry, llm, DEFAULT_PROMPT_REGISTRY, config)


def test_content_composer_react_timeout_falls_back_without_warning_or_traceback(caplog: Any) -> None:
    caplog.set_level(logging.WARNING, logger="assistant_rag.content_composer")

    result = _composer().compose(_composer_input(), GeneralPurposeConfig(content_composer_max_iterations=1))

    assert result.final_response_text == "Fallback answer."
    assert result.fallback_used is True
    assert caplog.records == []


def test_answer_generation_failure_uses_prompt_registry_message_and_warning() -> None:
    config = GeneralPurposeConfig(content_composer_max_iterations=1)
    registry = ContentToolRegistry(
        tools=[AnswerGenerationTool(llm=FailingAnswerLLM(), prompt_registry=DEFAULT_PROMPT_REGISTRY)],
        config=config,
    )
    composer = ReActContentComposer(registry, FailingAnswerLLM(), DEFAULT_PROMPT_REGISTRY, config)

    result = composer.compose(_composer_input(), config)

    assert result.final_response_text == DEFAULT_PROMPT_REGISTRY.message("answer_model_unavailable")
    assert result.fallback_used is True
    assert result.content_warnings == ("answer_model_unavailable",)
    assert result.reason_summary.startswith("answer_generation_failed:")


def test_general_hitl_timeout_falls_back_without_warning_or_traceback(caplog: Any) -> None:
    caplog.set_level(logging.WARNING, logger="assistant_rag.generation")

    decision = LLMGeneralHITLStrategy(
        TimeoutPlannerLLM(),
        DEFAULT_PROMPT_REGISTRY,
        GeneralPurposeConfig(hitl_supporting_question_enabled=True),
    ).evaluate(_pipeline_context(), _composer_result())

    assert decision.should_ask is False
    assert decision.question_source is QuestionSource.HUMAN_SUPPORTING_QUESTION
    assert caplog.records == []
