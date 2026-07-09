from __future__ import annotations

import logging
from typing import Any

from assistant_rag.classification import LLMQueryRewriter
from assistant_rag.llm import LLMTask


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
        raise AssertionError("chat should not be called by the query rewriter")


def test_query_rewriter_timeout_falls_back_without_warning_or_traceback(caplog: Any) -> None:
    caplog.set_level(logging.WARNING, logger="assistant_rag.classification")

    rewritten = LLMQueryRewriter(TimeoutLLM()).rewrite("  I want to learn Python from zero.  ")

    assert rewritten == "I want to learn Python from zero."
    assert caplog.records == []


def test_query_rewriter_timeout_keeps_debug_breadcrumb_without_traceback(caplog: Any) -> None:
    caplog.set_level(logging.DEBUG, logger="assistant_rag.classification")

    rewritten = LLMQueryRewriter(TimeoutLLM()).rewrite("  I want to learn Python from zero.  ")

    assert rewritten == "I want to learn Python from zero."
    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.DEBUG
    assert caplog.records[0].exc_info is None
    assert caplog.messages == [
        "Query rewrite LLM fallback engaged: Ollama structured output failed: timed out"
    ]
