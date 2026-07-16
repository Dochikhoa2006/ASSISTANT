from __future__ import annotations

import json
from typing import Any

from assistant_rag.branches import ClarificationBranch
from assistant_rag.config import QuestionGenerationConfig
from assistant_rag.contracts import (
    ChatRequest,
    ExpectedResponseType,
    GeneratedQuestion,
    Intent,
    PipelineContext,
    QuestionSource,
    ResponseType,
)
from assistant_rag.generation import LLMClarificationStrategy
from assistant_rag.llm import LLMTask
from assistant_rag.prompts import DEFAULT_PROMPT_REGISTRY


def _question(text: str) -> GeneratedQuestion:
    return GeneratedQuestion(
        text=text,
        source=QuestionSource.CLARIFICATION_QUESTION,
        purpose="resolve_missing_info",
        confidence=1.0,
        expected_response_type=ExpectedResponseType.FREE_TEXT_ANSWER,
    )


def _context(*, stale_question: GeneratedQuestion) -> PipelineContext:
    return PipelineContext(
        request=ChatRequest(
            user_id="clarification-user",
            raw_query="Please change it to the new value.",
            metadata={"clarification_question": stale_question},
        ),
        rewritten_query="Change the unspecified item to the new value.",
        last_qa_state=None,
        conversation_results=[],
        intent=Intent.CLARIFICATION,
        chat_history=[
            {
                "role": "conversation_hop",
                "raw_user_query": "FORBIDDEN_RAW_HISTORY_QUERY",
                "rewritten_user_query": "Earlier current context",
                "raw_response": "Earlier response",
            }
        ],
    )


class RecordingStrategy:
    def __init__(self, generated: GeneratedQuestion) -> None:
        self.generated = generated
        self.calls: list[tuple[PipelineContext, dict[str, Any]]] = []

    def generate(
        self,
        context: PipelineContext,
        **kwargs: Any,
    ) -> GeneratedQuestion:
        self.calls.append((context, kwargs))
        return self.generated


def test_clarification_branch_always_generates_instead_of_reusing_metadata() -> None:
    stale = _question("Which stale item should I change?")
    generated = _question("Which item should I change in this request?")
    context = _context(stale_question=stale)
    strategy = RecordingStrategy(generated)

    result = ClarificationBranch(
        clarification_strategy=strategy,
    ).execute(context, repository=object())

    assert len(strategy.calls) == 1
    assert strategy.calls[0] == (context, {})
    assert result.response_type is ResponseType.CLARIFICATION
    assert result.clarification_question is generated
    assert result.clarification_question is not stale


class RecordingClarificationLLM:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def generate_json(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {
            "question_text": "Which current item should be changed?",
            "question_source": QuestionSource.CLARIFICATION_QUESTION.value,
            "purpose": "resolve_current_turn",
            "confidence": 1.0,
            "should_ask": True,
            "expected_response_type": ExpectedResponseType.FREE_TEXT_ANSWER.value,
            "reason_summary": "The current turn does not name the target.",
        }


def test_llm_clarification_prompt_uses_current_turn_and_excludes_stale_metadata() -> None:
    stale_text = "Which stale item should I change?"
    context = _context(stale_question=_question(stale_text))
    llm = RecordingClarificationLLM()
    strategy = LLMClarificationStrategy(
        llm=llm,
        prompt_registry=DEFAULT_PROMPT_REGISTRY,
        config=QuestionGenerationConfig(),
    )

    generated = strategy.generate(context)

    assert generated is not None
    assert generated.text == "Which current item should be changed?"
    assert len(llm.calls) == 1
    call = llm.calls[0]
    assert call["task"] is LLMTask.GENERATE_CLARIFICATION
    runtime = json.loads(call["user_prompt"].removeprefix("Runtime context:\n"))
    assert "raw_query" not in runtime
    assert runtime["rewritten_query"] == context.rewritten_query
    assert runtime["chat_history"] == [
        {
            "role": "conversation_hop",
            "rewritten_user_query": "Earlier current context",
            "raw_response": "Earlier response",
        }
    ]
    assert runtime["extra"]["task_type"] == "clarification"
    assert "current turn" in runtime["extra"]["generation_instruction"]
    assert context.request.raw_query not in call["user_prompt"]
    assert "FORBIDDEN_RAW_HISTORY_QUERY" not in call["user_prompt"]
    assert "raw_user_query" not in call["user_prompt"]
    assert stale_text not in call["user_prompt"]
