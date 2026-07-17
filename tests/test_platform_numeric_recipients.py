from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from assistant_rag.contracts import BundledResponse, ChatRequest, LastQAState, ResponseType
from assistant_rag.platform import (
    PLATFORM_RECIPIENT_EXTRACTION_SCHEMA,
    PlatformSelector,
)


_RAW_QUERY_SENTINEL = "RAW_SENTINEL audit-only ingress text."


class ScriptedPlatformLLM:
    def __init__(self, channel: str, extraction: dict[str, Any]) -> None:
        self.channel = channel
        self.extraction = extraction
        self.calls: list[dict[str, Any]] = []

    def generate_json(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if "platform selector" in kwargs["system_prompt"]:
            return {"channel": self.channel, "confidence": 1.0}
        return dict(self.extraction)


@dataclass
class RecordingSender:
    calls: list[tuple[dict[str, Any], dict[str, Any]]] = field(default_factory=list)

    def send(self, payload: dict[str, Any], platform_context: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((dict(payload), dict(platform_context)))
        return {
            "status": "sent",
            "provider": payload["channel"],
            "recipient": payload["recipient"],
        }


def _bundled_response(rewritten_query: str) -> BundledResponse:
    return BundledResponse(
        final_chat_text="The requested update is ready.",
        response_type=ResponseType.NORMAL,
        last_qa_state=LastQAState(
            last_user_query=rewritten_query,
            last_response="",
            response_type=ResponseType.NORMAL,
        ),
    )


@pytest.mark.parametrize(
    ("channel", "recipient", "rewritten_query", "platform_context"),
    (
        (
            "telegram",
            -1001234567890,
            "Send this message via Telegram to chat ID -1001234567890.",
            {"telegram_bot_token": "test-token"},
        ),
        (
            "zalo",
            84987654321,
            "Send this message via Zalo to user ID 84987654321.",
            {"zalo_access_token": "test-token", "zalo_api_url": "https://example.test/zalo"},
        ),
    ),
)
def test_numeric_recipient_ids_are_routed_for_non_email_channels(
    channel: str,
    recipient: int,
    rewritten_query: str,
    platform_context: dict[str, str],
) -> None:
    sender = RecordingSender()
    llm = ScriptedPlatformLLM(
        channel,
        {"recipients": [str(recipient)]},
    )
    selector = PlatformSelector(
        llm=llm,
        senders={channel: sender},
    )

    result = selector.select(
        _bundled_response(rewritten_query),
        ChatRequest(
            user_id="platform-test",
            raw_query=_RAW_QUERY_SENTINEL,
            platform_context=platform_context,
        ),
    )

    expected_recipient = str(recipient)
    assert result["delivery"] == {
        "channel": channel,
        "status": "sent",
        "recipient": expected_recipient,
        "recipients": [expected_recipient],
        "provider": channel,
    }
    assert len(sender.calls) == 1
    sent_payload, sent_context = sender.calls[0]
    assert sent_payload["recipient"] == expected_recipient
    assert sent_payload["recipients"] == [expected_recipient]
    assert sent_payload["body"] == "The requested update is ready."
    assert sent_payload["mode"] == "send"
    assert sent_context == platform_context
    assert len(llm.calls) == 2
    assert llm.calls[1]["schema"] == PLATFORM_RECIPIENT_EXTRACTION_SCHEMA
    assert set(llm.calls[1]["schema"]["properties"]) == {"recipients"}


def test_gmail_recipient_validation_remains_email_only() -> None:
    rewritten_query = "Send an email to alice@example.com with the update."
    sender = RecordingSender()
    selector = PlatformSelector(
        llm=ScriptedPlatformLLM(
            "gmail",
            {
                "recipients": [123456, "alice@example.com"],
                "subject": "Update",
                "body": "The requested update is ready.",
                "mode": "send",
            },
        ),
        senders={"gmail": sender},
    )

    result = selector.select(
        _bundled_response(rewritten_query),
        ChatRequest(
            user_id="platform-test",
            raw_query=_RAW_QUERY_SENTINEL,
            platform_context={
                "gmail_username": "sender@example.com",
                "gmail_app_password": "test-password",
            },
        ),
    )

    assert result["delivery"]["status"] == "sent"
    assert result["delivery"]["recipients"] == ["alice@example.com"]
    assert len(sender.calls) == 1
    sent_payload, _ = sender.calls[0]
    assert sent_payload["recipient"] == "alice@example.com"
    assert sent_payload["recipients"] == ["alice@example.com"]
