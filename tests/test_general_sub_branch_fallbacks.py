from __future__ import annotations

import logging
from typing import Any

from assistant_rag.config import GeneralPurposeConfig
from assistant_rag.contracts import ChatRequest, GeneralSubBranch, Intent, PersistenceMode, PipelineContext
from assistant_rag.general_sub_branch import GeneralSubBranchDetector
from assistant_rag.llm import LLMTask
from assistant_rag.prompts import DEFAULT_PROMPT_REGISTRY


class TimeoutLLM:
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
        raise AssertionError("chat should not be called by the sub-branch detector")


def _context() -> PipelineContext:
    return PipelineContext(
        request=ChatRequest(user_id="u1", raw_query="I want to learn Python from zero."),
        rewritten_query="I want to learn Python from zero.",
        last_qa_state=None,
        conversation_results=[],
        intent=Intent.GENERAL_RESPONSE,
    )


def _detector() -> GeneralSubBranchDetector:
    return GeneralSubBranchDetector(TimeoutLLM(), DEFAULT_PROMPT_REGISTRY)


def test_general_sub_branch_timeout_falls_back_without_warning_or_traceback(caplog: Any) -> None:
    caplog.set_level(logging.WARNING, logger="assistant_rag.general_sub_branch")

    decision = _detector().detect(_context(), GeneralPurposeConfig())

    assert decision.sub_branch is GeneralSubBranch.NEW_CONVERSATION_TOPIC
    assert decision.persistence_mode is PersistenceMode.CREATE_NEW_TOPIC
    assert decision.confidence == 0.0
    assert caplog.records == []


def test_general_sub_branch_timeout_keeps_debug_breadcrumb_without_traceback(caplog: Any) -> None:
    caplog.set_level(logging.DEBUG, logger="assistant_rag.general_sub_branch")

    decision = _detector().detect(_context(), GeneralPurposeConfig())

    assert decision.sub_branch is GeneralSubBranch.NEW_CONVERSATION_TOPIC
    assert decision.persistence_mode is PersistenceMode.CREATE_NEW_TOPIC
    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.DEBUG
    assert caplog.records[0].exc_info is None
    assert caplog.messages == [
        "General sub-branch detection failed: Ollama structured output failed: timed out"
    ]
