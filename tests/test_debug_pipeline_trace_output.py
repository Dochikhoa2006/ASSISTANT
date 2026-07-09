from __future__ import annotations

from types import SimpleNamespace

from assistant_rag.contracts import (
    BundledResponse,
    LastQAState,
    ResponseType,
    TraceStageSummary,
    TraceSummary,
)
from assistant_rag.llm import LLMTask, ModelDecision
from assistant_rag.prompts import DEFAULT_PROMPT_REGISTRY

from debug_pipeline import _debug_llm_lines, _debug_response_lines, _debug_trace_lines


def test_debug_trace_lines_print_each_stage_with_metadata() -> None:
    summary = TraceSummary(
        request_id="req-1",
        total_latency_ms=12.345,
        stages=[
            TraceStageSummary(
                stage="rewrite",
                latency_ms=1.2,
                metadata={"raw_query": "[REDACTED]"},
            ),
            TraceStageSummary(
                stage="branch_execution",
                latency_ms=9.8,
                metadata={"intent": "general_response"},
            ),
        ],
    )

    lines = _debug_trace_lines(summary)

    assert lines == [
        "Debug trace:",
        "  request_id: req-1",
        "  total_latency_ms: 12.345",
        "  stages:",
        '    01. rewrite | 1.2 ms | metadata={"raw_query":"[REDACTED]"}',
        '    02. branch_execution | 9.8 ms | metadata={"intent":"general_response"}',
    ]


def test_debug_response_lines_show_runtime_diagnostics() -> None:
    fallback_text = DEFAULT_PROMPT_REGISTRY.message("answer_model_unavailable")
    response = BundledResponse(
        final_chat_text=fallback_text,
        response_type=ResponseType.NORMAL,
        last_qa_state=LastQAState(
            last_user_query="I want to learn Python from zero.",
            last_response=fallback_text,
            response_type=ResponseType.NORMAL,
        ),
        conversation_topic_id="topic-1",
        conversation_hop_id="hop-1",
        warnings=["answer_model_unavailable"],
        persistence_instructions={"database_write_result": {"conversation_hop_id": "hop-1"}},
    )

    lines = _debug_response_lines(response)

    assert "  response_type: normal" in lines
    assert "  conversation_topic_id: topic-1" in lines
    assert "  conversation_hop_id: hop-1" in lines
    assert '  warnings: ["answer_model_unavailable"]' in lines
    assert '  persistence: {"database_write_result":{"conversation_hop_id":"hop-1"}}' in lines


class FakeRouter:
    def decision_for_task(self, task: LLMTask) -> ModelDecision:
        return ModelDecision(
            task=task,
            model=f"model-for-{task.value}",
            temperature=0.25,
            timeout_seconds=12.5,
            num_ctx=4096,
            reason_summary=f"reason-for-{task.value}",
        )


def test_debug_llm_lines_show_runtime_model_policy_and_recent_errors() -> None:
    llm = SimpleNamespace(
        settings=SimpleNamespace(
            base_url="http://ollama.local",
            keep_alive="1m",
            structured_retry_count=4,
        ),
        router=FakeRouter(),
        last_error_by_task={
            LLMTask.QUERY_REWRITE: "timed out",
            LLMTask.ANSWER: "connection refused",
        },
    )

    lines = _debug_llm_lines(llm)

    assert lines[:5] == [
        "Debug LLM:",
        "  base_url: http://ollama.local",
        "  keep_alive: 1m",
        "  structured_attempts_for_json: 5",
        "  recent_errors:",
    ]
    assert (
        '    - {"last_error":"timed out","model":"model-for-query_rewrite",'
        '"num_ctx":4096,"reason":"reason-for-query_rewrite",'
        '"single_call_timeout_seconds":12.5,"task":"query_rewrite","temperature":0.25}'
    ) in lines
    assert (
        '    - {"last_error":"connection refused","model":"model-for-answer",'
        '"num_ctx":4096,"reason":"reason-for-answer",'
        '"single_call_timeout_seconds":12.5,"task":"answer","temperature":0.25}'
    ) in lines
