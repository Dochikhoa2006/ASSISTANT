from __future__ import annotations

from pathlib import Path

from assistant_rag.semantic_actions import (
    grounded_semantic_action_from_payload,
    semantic_action_from_internal_payload,
)


ROOT = Path(__file__).resolve().parents[1]


def _contract(query: str) -> dict:
    return {
        "message": {
            "operation": "send",
            "channel": "gmail",
            "recipient_update": "replace",
            "recipients": [
                {
                    "value": "person@example.com",
                    "disposition": "include",
                    "evidence": "person@example.com",
                }
            ],
            "global_cancellation": False,
            "authorization_evidence": [query],
            "cancellation_evidence": [],
            "artifact_reference": "none",
            "copy_revision": False,
            "confidence": 0.99,
        },
        "file": {
            "operation": "none",
            "file_type": "none",
            "authorization_evidence": [],
            "type_evidence": [],
            "confidence": 0.99,
        },
        "reason_summary": "grounded test contract",
    }


def test_runtime_action_paths_have_no_legacy_phrase_tables() -> None:
    removed_modules = (
        "assistant_rag/action_keywords.py",
        "assistant_rag/content_keywords.py",
        "assistant_rag/request_policy.py",
    )
    for relative_path in removed_modules:
        assert not (ROOT / relative_path).exists()

    forbidden_symbols = (
        "FILE_CREATION_VERB_KEYWORDS",
        "KeywordIntentClassifier",
        "intent_keywords",
        "_RECIPIENT_EXCLUSION_CLAUSE",
        "_DO_NOT_SEND",
        "_ATTACHMENT_CHANGE",
        "gmail_recipient_exclusions",
    )
    runtime_paths = (
        "assistant_rag/classification.py",
        "assistant_rag/content_composer.py",
        "assistant_rag/pipeline.py",
        "assistant_rag/platform.py",
        "assistant_rag/production_factory.py",
        "assistant_rag/request_lifecycle.py",
    )
    combined_source = "\n".join(
        (ROOT / relative_path).read_text(encoding="utf-8")
        for relative_path in runtime_paths
    )
    for symbol in forbidden_symbols:
        assert symbol not in combined_source


def test_serialized_decision_cannot_be_replayed_against_different_words() -> None:
    query = "Transmit this to person@example.com."
    original = grounded_semantic_action_from_payload(
        _contract(query),
        canonical_query=query,
    )

    assert original.message.authorizes_send
    replayed = semantic_action_from_internal_payload(
        original.to_payload(),
        canonical_query="Continue our unrelated discussion.",
    )

    assert not replayed.grounded
    assert not replayed.message.authorizes_send


def test_recipient_identifier_must_be_grounded_in_its_own_source_quote() -> None:
    query = "Transmit this to person@example.com."
    payload = _contract(query)
    payload["message"]["recipients"][0]["value"] = "invented@example.com"

    decision = grounded_semantic_action_from_payload(
        payload,
        canonical_query=query,
    )

    assert decision.message.authorizes_send
    assert decision.message.included_recipients == ()

