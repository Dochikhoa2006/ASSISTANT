from __future__ import annotations

import pytest

from assistant_rag.action_keywords import (
    ADD_ACTION_KEYWORDS,
    DELETE_ACTION_KEYWORDS,
    MODIFY_ACTION_KEYWORDS,
    TURN_OFF_ACTION_KEYWORDS,
    TURN_ON_ACTION_KEYWORDS,
)
from assistant_rag.contracts import ChatRequest, Intent
from assistant_rag.request_lifecycle import looks_destructive, looks_like_mutation
from assistant_rag.request_policy import (
    has_destructive_mutation_policy_signal,
    has_explicit_mutation_policy_signal,
)


REQUEST_POLICY_KEYWORD_CASES = (
    *((Intent.KNOWLEDGE_FACTS, keyword, True) for keyword in DELETE_ACTION_KEYWORDS),
    *((Intent.KNOWLEDGE_FACTS, keyword, True) for keyword in MODIFY_ACTION_KEYWORDS),
    *((Intent.KNOWLEDGE_FACTS, keyword, False) for keyword in ADD_ACTION_KEYWORDS),
    *((Intent.REMINDER, keyword, True) for keyword in DELETE_ACTION_KEYWORDS),
    *((Intent.REMINDER, keyword, True) for keyword in MODIFY_ACTION_KEYWORDS),
    *((Intent.REMINDER, keyword, False) for keyword in ADD_ACTION_KEYWORDS),
    *((Intent.REMINDER, keyword, False) for keyword in TURN_ON_ACTION_KEYWORDS),
    *((Intent.REMINDER, keyword, True) for keyword in TURN_OFF_ACTION_KEYWORDS),
)


@pytest.mark.parametrize(
    ("intent", "keyword", "expected_destructive"),
    REQUEST_POLICY_KEYWORD_CASES,
)
def test_every_policy_keyword_sets_only_request_lifecycle_signals(
    intent: Intent,
    keyword: str,
    expected_destructive: bool,
) -> None:
    query = f"Please {keyword} the requested item."
    request = ChatRequest(user_id="lifecycle-matrix", raw_query=query)

    assert has_explicit_mutation_policy_signal(request, intent), (intent, keyword)
    assert has_destructive_mutation_policy_signal(
        request, intent
    ) is expected_destructive
    assert looks_like_mutation(request), (intent, keyword)
    if expected_destructive:
        assert looks_destructive(request), (intent, keyword)


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
