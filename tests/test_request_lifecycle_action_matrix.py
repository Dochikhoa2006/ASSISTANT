from __future__ import annotations

import pytest

from assistant_rag.contracts import ChatRequest
from assistant_rag.request_lifecycle import looks_destructive, looks_like_mutation


@pytest.mark.parametrize(
    "query",
    (
        "Erase every retained fact.",
        "Kindly make the alarm dormant.",
        "Memorize the following detail.",
        "What if somebody removed a reminder?",
        "supprimer cette information",
        "hãy tắt lời nhắc",
        "opaque-operation-47",
    ),
)
def test_lifecycle_never_interprets_free_form_language(query: str) -> None:
    request = ChatRequest(user_id="user-1", raw_query=query)

    assert looks_like_mutation(request) is False
    assert looks_destructive(request) is False


def test_bound_confirmation_is_known_mutation_without_reparsing_reply_text() -> None:
    request = ChatRequest(
        user_id="user-1",
        raw_query="yes",
        confirmation_token="owned-confirmation-token",
    )

    assert looks_like_mutation(request) is True
    assert looks_destructive(request) is True


def test_verified_reminder_reply_route_is_known_internal_mutation() -> None:
    request = ChatRequest(
        user_id="user-1",
        raw_query="opaque reply",
        metadata={"reminder_reply_context": True},
    )

    assert looks_like_mutation(request) is True
    assert looks_destructive(request) is False


def test_untrusted_intent_and_action_metadata_does_not_create_policy_authority() -> None:
    request = ChatRequest(
        user_id="user-1",
        raw_query="ordinary text",
        metadata={
            "intent_hint": "knowledge_facts",
            "action": "delete",
            "destructive": True,
        },
    )

    assert looks_like_mutation(request) is False
    assert looks_destructive(request) is False
