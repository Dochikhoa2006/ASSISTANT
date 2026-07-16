"""Canonical request-scoped chat history for every downstream prompt."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from enum import Enum
from typing import Any, Iterator, Mapping

from .contracts import ApprovedConversationContext, LastQAState


CHAT_HISTORY_PROMPT_RULE = (
    "Always inspect and use chat_history for task-relevant continuity and reference "
    "resolution. The current query remains authoritative for new intent and actions; "
    "history must never invent a mutation. An empty chat_history is authoritative and "
    "means no prior conversation context is available."
)


_CANONICAL_CHAT_HISTORY: ContextVar[tuple[dict[str, Any], ...]] = ContextVar(
    "assistant_canonical_chat_history",
    default=(),
)
_RAW_USER_QUERY_KEYS = frozenset(
    {
        "raw_query",
        "raw_user_query",
        "source_raw_user_query",
        "summarized_user_query",
    }
)


def _without_raw_user_queries(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _without_raw_user_queries(item)
            for key, item in value.items()
            if str(key).casefold() not in _RAW_USER_QUERY_KEYS
        }
    if isinstance(value, list):
        return [_without_raw_user_queries(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_without_raw_user_queries(item) for item in value)
    return value


def rewritten_only_chat_history(
    chat_history: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Quarantine audit-only raw user queries from semantic history."""

    sanitized = _without_raw_user_queries(deepcopy(chat_history))
    return [dict(item) for item in sanitized if isinstance(item, Mapping)]


def _enum_value(value: Any) -> Any:
    return value.value if isinstance(value, Enum) else value


def _question_payload(question: Any) -> dict[str, Any] | None:
    if question is None:
        return None
    return {
        "text": str(getattr(question, "text", "")),
        "source": _enum_value(getattr(question, "source", None)),
        "purpose": str(getattr(question, "purpose", "")),
        "confidence": float(getattr(question, "confidence", 0.0)),
        "should_ask": bool(getattr(question, "should_ask", True)),
        "expected_response_type": _enum_value(
            getattr(question, "expected_response_type", None)
        ),
    }


def last_qa_chat_history(state: LastQAState | None) -> list[dict[str, Any]]:
    """Represent the complete Last-QA hop using the retrieved-hop shape."""

    if state is None:
        return []
    supporting_questions = [
        payload
        for question in state.supporting_questions
        if (payload := _question_payload(question)) is not None
    ]
    return [
        {
            "source": "last_qa",
            "role": "conversation_hop",
            "rewritten_user_query": state.last_user_query,
            "raw_response": state.last_response,
            "response_type": _enum_value(state.response_type),
            "supporting_questions": supporting_questions,
            "clarification_question": _question_payload(state.clarification_question),
            "reminder_supporting_question": _question_payload(
                state.reminder_supporting_question
            ),
            "topic_id": state.linked_topic_id,
            "hop_id": state.linked_hop_id,
            "expected_response_type": _enum_value(state.expected_response_type),
        }
    ]


def select_chat_history(
    *,
    conversation_retrieval: bool,
    approved_conversation_context: ApprovedConversationContext | None,
    last_qa_state: LastQAState | None,
) -> list[dict[str, Any]]:
    """Apply the non-fallback history precedence required by the pipeline.

    Once conversation retrieval runs, its post-validation result is authoritative.
    An empty/all-rejected result therefore stays empty and never falls back to
    Last-QA.  Last-QA is used only when conversation retrieval did not run.
    """

    if conversation_retrieval:
        if approved_conversation_context is None:
            return []
        return rewritten_only_chat_history(
            approved_conversation_context.approved_conversation_history
        )
    return last_qa_chat_history(last_qa_state)


@contextmanager
def canonical_chat_history_scope(
    chat_history: list[dict[str, Any]],
) -> Iterator[None]:
    """Expose one immutable-per-request history value to every prompt builder."""

    token = _CANONICAL_CHAT_HISTORY.set(
        tuple(rewritten_only_chat_history(chat_history))
    )
    try:
        yield
    finally:
        _CANONICAL_CHAT_HISTORY.reset(token)


def current_chat_history() -> list[dict[str, Any]]:
    """Return a defensive copy of the active request's canonical history."""

    return deepcopy(list(_CANONICAL_CHAT_HISTORY.get()))


def supporting_question_context(
    chat_history: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Expose question-bearing continuity without bypassing history precedence.

    This projection is deliberately derived only from the already-resolved
    canonical history.  In particular, an authoritative empty history remains
    empty instead of falling back to stale Last-QA or rejected retrieval data.
    """

    projected: list[dict[str, Any]] = []
    for hop in deepcopy(chat_history):
        if not isinstance(hop, Mapping):
            continue
        questions = hop.get("supporting_questions")
        if not isinstance(questions, list):
            questions = []
        clarification = hop.get("clarification_question")
        reminder_question = hop.get("reminder_supporting_question")
        expected_response_type = hop.get("expected_response_type")
        if not (
            questions
            or clarification
            or reminder_question
            or expected_response_type
        ):
            continue
        projected.append(
            {
                "source": hop.get("source"),
                "topic_id": hop.get("topic_id"),
                "hop_id": hop.get("hop_id"),
                "supporting_questions": questions,
                "clarification_question": clarification,
                "reminder_supporting_question": reminder_question,
                "expected_response_type": expected_response_type,
            }
        )
    return projected


def inject_chat_history(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Attach canonical history to direct JSON prompts that bypass PromptRegistry."""

    enriched = dict(payload)
    enriched["chat_history"] = current_chat_history()
    return enriched
