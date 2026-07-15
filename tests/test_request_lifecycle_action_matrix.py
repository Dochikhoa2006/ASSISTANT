from __future__ import annotations

import pytest

from assistant_rag.action_detection import DeterministicActionDetector
from assistant_rag.action_keywords import (
    ADD_ACTION_KEYWORDS,
    DELETE_ACTION_KEYWORDS,
    MODIFY_ACTION_KEYWORDS,
    TURN_OFF_ACTION_KEYWORDS,
    TURN_ON_ACTION_KEYWORDS,
)
from assistant_rag.contracts import ChatRequest, Intent
from assistant_rag.request_lifecycle import looks_destructive, looks_like_mutation


ACTION_KEYWORD_CASES = (
    *((Intent.KNOWLEDGE_FACTS, keyword, "delete") for keyword in DELETE_ACTION_KEYWORDS),
    *((Intent.KNOWLEDGE_FACTS, keyword, "modify") for keyword in MODIFY_ACTION_KEYWORDS),
    *((Intent.KNOWLEDGE_FACTS, keyword, "add") for keyword in ADD_ACTION_KEYWORDS),
    *((Intent.REMINDER, keyword, "delete") for keyword in DELETE_ACTION_KEYWORDS),
    *((Intent.REMINDER, keyword, "modify") for keyword in MODIFY_ACTION_KEYWORDS),
    *((Intent.REMINDER, keyword, "add") for keyword in ADD_ACTION_KEYWORDS),
    *((Intent.REMINDER, keyword, "turn_on") for keyword in TURN_ON_ACTION_KEYWORDS),
    *((Intent.REMINDER, keyword, "turn_off") for keyword in TURN_OFF_ACTION_KEYWORDS),
)


@pytest.mark.parametrize(("intent", "keyword", "expected_action"), ACTION_KEYWORD_CASES)
def test_every_authorized_action_keyword_enters_the_matching_request_lifecycle(
    intent: Intent,
    keyword: str,
    expected_action: str,
) -> None:
    query = f"Please {keyword} the requested item."
    request = ChatRequest(user_id="lifecycle-matrix", raw_query=query)

    detection = DeterministicActionDetector().detect(request, query, intent)

    action_key = "knowledge_actions" if intent is Intent.KNOWLEDGE_FACTS else "reminder_actions"
    assert not detection.requires_clarification, (intent, keyword, detection)
    assert detection.metadata[action_key][0]["action"] == expected_action
    assert looks_like_mutation(request), (intent, keyword, expected_action)
    if expected_action in {"delete", "modify", "turn_off"}:
        assert looks_destructive(request), (intent, keyword, expected_action)


@pytest.mark.parametrize(
    "query",
    (
        "Please discuss the reminder policy.",
        "The recordkeeper updated the changelog.",
        "This removable memo is informational.",
    ),
)
def test_non_action_words_do_not_enter_the_mutation_lifecycle(query: str) -> None:
    request = ChatRequest(user_id="lifecycle-matrix", raw_query=query)

    assert not looks_like_mutation(request)
    assert not looks_destructive(request)


@pytest.mark.parametrize("query", ("Please save this fact.", "Please enable the reminder."))
def test_unambiguously_non_destructive_actions_are_not_destructive(query: str) -> None:
    request = ChatRequest(user_id="lifecycle-matrix", raw_query=query)

    assert looks_like_mutation(request)
    assert not looks_destructive(request)


def test_confirmation_enters_both_mutation_and_destructive_lifecycles() -> None:
    request = ChatRequest(
        user_id="lifecycle-matrix",
        raw_query="Yes, confirm it.",
        confirmation_token="verified-pending-action",
    )

    assert looks_like_mutation(request)
    assert looks_destructive(request)


def test_independent_multi_action_request_is_conservatively_lifecycle_limited() -> None:
    request = ChatRequest(
        user_id="lifecycle-matrix",
        raw_query="Delete the old reminder and create a new reminder.",
    )

    assert looks_like_mutation(request)
    assert looks_destructive(request)
