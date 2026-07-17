from __future__ import annotations

import json
from typing import Any

from assistant_rag.branches import BranchRouter, ClarificationBranch
from assistant_rag.config import QuestionGenerationConfig
from assistant_rag.contracts import (
    ChatRequest,
    ExpectedResponseType,
    GeneratedQuestion,
    Intent,
    LastQAState,
    PipelineContext,
    QuestionSource,
    ResponseType,
)
from assistant_rag.generation import LLMClarificationStrategy, _question_list_schema
from assistant_rag.llm import LLMTask, structured_fallback_payload
from assistant_rag.prompts import DEFAULT_PROMPT_REGISTRY


def _question(text: str) -> GeneratedQuestion:
    return GeneratedQuestion(
        text=text,
        source=QuestionSource.CLARIFICATION_QUESTION,
        purpose="resolve_missing_info",
        confidence=1.0,
        expected_response_type=ExpectedResponseType.FREE_TEXT_ANSWER,
    )


def test_included_human_question_does_not_repeat_should_ask_decision() -> None:
    item_schema = _question_list_schema()["properties"]["questions"]["items"]

    assert "should_ask" not in item_schema["properties"]
    assert item_schema["required"] == [
        "question_text",
        "confidence",
        "expected_response_type",
    ]


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
            "confidence": 1.0,
            "expected_response_type": ExpectedResponseType.FREE_TEXT_ANSWER.value,
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
    assert "should_ask" not in call["schema"]["properties"]
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


class StructuredFallbackLLM:
    def generate_json(self, **kwargs: Any) -> dict[str, Any]:
        return structured_fallback_payload(
            task=kwargs["task"],
            schema=kwargs["schema"],
            user_prompt=kwargs["user_prompt"],
            error=ValueError("synthetic invalid JSON"),
        )


def test_structured_fallback_never_becomes_a_generic_clarification() -> None:
    context = _context(stale_question=_question("Which stale item?"))
    strategy = LLMClarificationStrategy(
        llm=StructuredFallbackLLM(),
        prompt_registry=DEFAULT_PROMPT_REGISTRY,
        config=QuestionGenerationConfig(),
    )

    result = ClarificationBranch(
        clarification_strategy=strategy,
    ).execute(context, repository=object())

    assert result.response_type is ResponseType.ERROR
    assert result.clarification_question is None
    assert "generic question" in result.fallback_or_error_message


class StaticClarificationBranch:
    prompt_registry = DEFAULT_PROMPT_REGISTRY

    def __init__(self, question: GeneratedQuestion) -> None:
        self.question = question

    def execute(self, context: PipelineContext, repository: object):
        from assistant_rag.contracts import BranchResult

        return BranchResult(
            response_type=ResponseType.CLARIFICATION,
            clarification_question=self.question,
        )


def test_identical_consecutive_clarification_is_safely_suppressed() -> None:
    repeated = _question("Which item should I change?")
    context = _context(stale_question=repeated)
    context = PipelineContext(
        **{
            **context.__dict__,
            "previous_last_qa_state": LastQAState(
                last_user_query="Change it.",
                last_response=repeated.text,
                response_type=ResponseType.CLARIFICATION,
                clarification_question=repeated,
            ),
        }
    )

    result = BranchRouter(
        {Intent.CLARIFICATION: StaticClarificationBranch(repeated)}
    ).route(context, repository=object())

    assert result.response_type is ResponseType.SAFE_NOOP
    assert result.clarification_question is None
    assert result.normal_response_text
    assert "clarification_repeat_suppressed" in result.warnings
