from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from assistant_rag.contracts import (
    BundledResponse,
    ChatRequest,
    LastQAState,
    OutboundFollowUpAction,
    OutboundMessageState,
    ResponseType,
)
from assistant_rag.platform import PlatformSelector
from assistant_rag.semantic_actions import grounded_semantic_action_from_payload


def _semantic_payload(
    query: str,
    *,
    operation: str = "none",
    channel: str = "none",
    action_quote: str | None = None,
    include: tuple[str, ...] = (),
    exclude: tuple[str, ...] = (),
    recipient_update: str = "preserve",
    global_cancellation: bool = False,
    cancellation_quote: str | None = None,
    artifact_reference: str = "none",
    copy_revision: bool = False,
    file_operation: str = "none",
    file_type: str = "none",
    file_action_quote: str | None = None,
    file_type_quote: str | None = None,
    confidence: float = 0.99,
) -> dict[str, Any]:
    recipients = [
        {"value": value, "disposition": "include", "evidence": value}
        for value in include
    ] + [
        {"value": value, "disposition": "exclude", "evidence": value}
        for value in exclude
    ]
    raw = {
        "message": {
            "operation": operation,
            "channel": channel,
            "recipient_update": recipient_update,
            "recipients": recipients,
            "global_cancellation": global_cancellation,
            "authorization_evidence": [action_quote] if action_quote else [],
            "cancellation_evidence": (
                [cancellation_quote] if cancellation_quote else []
            ),
            "artifact_reference": artifact_reference,
            "copy_revision": copy_revision,
            "confidence": confidence,
        },
        "file": {
            "operation": file_operation,
            "file_type": file_type,
            "authorization_evidence": (
                [file_action_quote] if file_action_quote else []
            ),
            "type_evidence": [file_type_quote] if file_type_quote else [],
            "confidence": confidence,
        },
        "reason_summary": f"fixture:{query}",
    }
    return grounded_semantic_action_from_payload(
        raw, canonical_query=query
    ).to_payload()


def _bundled(
    query: str,
    semantic_payload: dict[str, Any],
    *,
    artifacts: list[dict[str, Any]] | None = None,
    platform_payload: dict[str, Any] | None = None,
    response_type: ResponseType = ResponseType.NORMAL,
    text: str = "Subject: Team update\n\nHello team,\n\nHere is the complete update.",
) -> BundledResponse:
    payload = dict(platform_payload or {})
    payload["semantic_action_decision"] = semantic_payload
    if artifacts is not None:
        payload["artifacts"] = artifacts
    return BundledResponse(
        final_chat_text=text,
        response_type=response_type,
        last_qa_state=LastQAState(
            last_user_query=query,
            last_response=text,
            response_type=response_type,
        ),
        platform_payload=payload,
    )


def _artifact(tmp_path: Path, suffix: str, content: bytes = b"payload") -> dict[str, Any]:
    path = tmp_path / f"generated{suffix}"
    path.write_bytes(content)
    return {
        "artifact_id": f"artifact-{suffix.lstrip('.')}",
        "file_type": suffix.lstrip("."),
        "filename": path.name,
        "storage_path": str(path),
        "storage_url": f"/artifacts/{path.name}",
        "status": "created",
    }


@dataclass
class RecordingSender:
    status: str = "sent"
    send_calls: list[dict[str, Any]] = field(default_factory=list)
    draft_calls: list[dict[str, Any]] = field(default_factory=list)

    def send(self, payload: dict[str, Any], _context: dict[str, Any]) -> dict[str, Any]:
        self.send_calls.append(payload)
        return {"status": self.status, "provider": "gmail"}

    def create_draft(
        self, payload: dict[str, Any], _context: dict[str, Any]
    ) -> dict[str, Any]:
        self.draft_calls.append(payload)
        return {"status": "draft_saved", "provider": "gmail"}


def _gmail_context() -> dict[str, str]:
    return {
        "gmail_username": "sender@example.com",
        "gmail_app_password": "app-password",
    }


@pytest.mark.parametrize(
    "query",
    (
        "Send an email to alice@example.com.",
        "Draft a message for alice@example.com.",
        "Please re-attach that presentation and deliver it.",
        "How do I send mail to alice@example.com?",
        "hãy gửi thư cho alice@example.com",
    ),
)
def test_natural_language_alone_never_routes_or_authorizes(query: str) -> None:
    response = _bundled(query, _semantic_payload(query))

    result = PlatformSelector(llm=None).select(
        response, ChatRequest(user_id="user-1", raw_query=query)
    )

    assert result["platform_selection"]["channel"] == "none"
    assert result["delivery"]["status"] == "not_requested"


@pytest.mark.parametrize(
    ("query", "action_quote"),
    (
        ("Convey this electronically to alice@example.com.", "Convey this electronically"),
        ("Chuyển nội dung này cho alice@example.com.", "Chuyển nội dung này"),
        ("Opaque operation α for alice@example.com.", "Opaque operation α"),
    ),
)
def test_unseen_phrasings_route_from_grounded_meaning_not_words(
    query: str, action_quote: str
) -> None:
    semantic = _semantic_payload(
        query,
        operation="prepare",
        channel="gmail",
        action_quote=action_quote,
        include=("alice@example.com",),
        recipient_update="replace",
    )

    result = PlatformSelector(llm=None).select(
        _bundled(query, semantic),
        ChatRequest(user_id="user-1", raw_query="untrusted raw ingress"),
    )

    assert result["platform_selection"] == {
        "channel": "gmail",
        "confidence": 0.99,
        "source": "grounded_semantic_contract",
    }
    assert result["delivery"]["status"] == "draft_ready"
    assert result["draft"]["recipients"] == ["alice@example.com"]


def test_recipient_polarity_is_enforced_from_grounded_contract() -> None:
    query = "Prepare the notice for alice@example.com and bob@example.com, excluding mallory@example.com."
    semantic = _semantic_payload(
        query,
        operation="prepare",
        channel="gmail",
        action_quote="Prepare the notice",
        include=("alice@example.com", "bob@example.com", "mallory@example.com"),
        exclude=("mallory@example.com",),
        recipient_update="replace",
    )

    result = PlatformSelector(llm=None).select(
        _bundled(query, semantic), ChatRequest(user_id="user-1", raw_query=query)
    )

    assert result["draft"]["recipients"] == [
        "alice@example.com",
        "bob@example.com",
    ]
    assert result["draft"]["excluded_recipients"] == ["mallory@example.com"]


def test_global_cancellation_always_downgrades_send_to_local_draft() -> None:
    query = "Transmit this to alice@example.com; hold the operation entirely."
    semantic = _semantic_payload(
        query,
        operation="send",
        channel="gmail",
        action_quote="Transmit this",
        include=("alice@example.com",),
        global_cancellation=True,
        cancellation_quote="hold the operation entirely",
    )
    sender = RecordingSender()

    result = PlatformSelector(
        llm=None, senders={"gmail": sender}
    ).select(
        _bundled(query, semantic),
        ChatRequest(
            user_id="user-1", raw_query=query, platform_context=_gmail_context()
        ),
    )

    assert result["delivery"]["status"] == "draft_ready"
    assert sender.send_calls == []


def test_send_requires_grounded_authorization_recipients_and_credentials() -> None:
    query = "Dispatch the completed notice to alice@example.com and bob@example.com."
    semantic = _semantic_payload(
        query,
        operation="send",
        channel="gmail",
        action_quote="Dispatch the completed notice",
        include=("alice@example.com", "bob@example.com"),
        recipient_update="replace",
    )
    sender = RecordingSender()
    selector = PlatformSelector(llm=None, senders={"gmail": sender})

    missing = selector.select(
        _bundled(query, semantic), ChatRequest(user_id="user-1", raw_query=query)
    )
    sent = selector.select(
        _bundled(query, semantic),
        ChatRequest(
            user_id="user-1", raw_query=query, platform_context=_gmail_context()
        ),
    )

    assert missing["delivery"]["status"] == "needs_input"
    assert sent["delivery"]["status"] == "sent"
    assert sender.send_calls[-1]["recipients"] == [
        "alice@example.com",
        "bob@example.com",
    ]


def test_save_draft_is_distinct_from_local_preparation() -> None:
    query = "Persist this prepared message for alice@example.com in the provider mailbox."
    semantic = _semantic_payload(
        query,
        operation="save_draft",
        channel="gmail",
        action_quote="Persist this prepared message",
        include=("alice@example.com",),
    )
    sender = RecordingSender()
    selector = PlatformSelector(llm=None, senders={"gmail": sender})

    local = selector.select(
        _bundled(query, semantic), ChatRequest(user_id="user-1", raw_query=query)
    )
    saved = selector.select(
        _bundled(query, semantic),
        ChatRequest(
            user_id="user-1", raw_query=query, platform_context=_gmail_context()
        ),
    )

    assert local["delivery"]["status"] == "draft_ready"
    assert saved["delivery"]["status"] == "draft_saved"
    assert len(sender.draft_calls) == 1


@pytest.mark.parametrize(
    ("recipient", "evidence"),
    (
        ("invented@example.com", "alice@example.com"),
        ("invalid@gmail", "invalid@gmail"),
    ),
)
def test_invented_or_syntactically_invalid_recipient_is_rejected(
    recipient: str, evidence: str
) -> None:
    query = f"Prepare a note for {evidence}."
    raw = _semantic_payload(
        query,
        operation="prepare",
        channel="gmail",
        action_quote="Prepare a note",
        include=(recipient,),
    )

    result = PlatformSelector(llm=None).select(
        _bundled(query, raw), ChatRequest(user_id="user-1", raw_query=query)
    )

    assert result["delivery"]["status"] == "needs_input"


def test_bundle_may_order_but_cannot_invent_or_drop_grounded_recipients() -> None:
    query = "Prepare this for alice@example.com and bob@example.com."
    semantic = _semantic_payload(
        query,
        operation="prepare",
        channel="gmail",
        action_quote="Prepare this",
        include=("alice@example.com", "bob@example.com"),
    )
    response = _bundled(
        query,
        semantic,
        platform_payload={
            "outbound_message": {
                "recipients": ["bob@example.com", "mallory@example.com"],
                "subject": "Complete subject",
                "body": "Complete body from both composition stages.",
            }
        },
    )

    result = PlatformSelector(llm=None).select(
        response, ChatRequest(user_id="user-1", raw_query=query)
    )

    assert result["draft"]["recipients"] == [
        "bob@example.com",
        "alice@example.com",
    ]
    assert result["draft"]["subject"] == "Complete subject"
    assert result["draft"]["body"] == "Complete body from both composition stages."


@pytest.mark.parametrize("suffix", (".xlsx", ".pdf", ".pptx"))
def test_every_generated_microsoft_artifact_is_attached(
    tmp_path: Path, suffix: str
) -> None:
    query = "Prepare the generated deliverable for alice@example.com."
    semantic = _semantic_payload(
        query,
        operation="prepare",
        channel="gmail",
        action_quote="Prepare the generated deliverable",
        include=("alice@example.com",),
    )
    artifact = _artifact(tmp_path, suffix)

    result = PlatformSelector(llm=None).select(
        _bundled(query, semantic, artifacts=[artifact]),
        ChatRequest(user_id="user-1", raw_query=query),
    )

    assert result["delivery"]["status"] == "draft_ready"
    assert [item["artifact_id"] for item in result["draft"]["attachments"]] == [
        artifact["artifact_id"]
    ]


def test_unavailable_declared_artifact_blocks_before_send(tmp_path: Path) -> None:
    query = "Dispatch the generated deliverable to alice@example.com."
    semantic = _semantic_payload(
        query,
        operation="send",
        channel="gmail",
        action_quote="Dispatch the generated deliverable",
        include=("alice@example.com",),
    )
    missing = {
        "artifact_id": "missing-artifact",
        "file_type": "pptx",
        "filename": "missing.pptx",
        "storage_path": str(tmp_path / "missing.pptx"),
        "status": "created",
    }
    sender = RecordingSender()

    result = PlatformSelector(llm=None, senders={"gmail": sender}).select(
        _bundled(query, semantic, artifacts=[missing]),
        ChatRequest(
            user_id="user-1", raw_query=query, platform_context=_gmail_context()
        ),
    )

    assert result["delivery"]["status"] == "failed"
    assert sender.send_calls == []


def test_authoritative_send_followup_reuses_exact_owned_envelope(
    tmp_path: Path,
) -> None:
    query = "Proceed with the pending transmission."
    semantic = _semantic_payload(
        query,
        operation="send",
        channel="gmail",
        action_quote="Proceed with the pending transmission",
    )
    artifact = _artifact(tmp_path, ".pdf")
    state = OutboundMessageState(
        channel="gmail",
        status="draft_ready",
        recipients=("alice@example.com",),
        subject="Owned subject",
        body="Owned complete body",
        artifact_ids=(artifact["artifact_id"],),
        attachment_filenames=(artifact["filename"],),
        source_topic_id="topic-1",
        source_hop_id="hop-1",
    )
    sender = RecordingSender()

    result = PlatformSelector(llm=None, senders={"gmail": sender}).select_with_outbound_context(
        _bundled(query, semantic),
        ChatRequest(
            user_id="user-1", raw_query=query, platform_context=_gmail_context()
        ),
        outbound_state=state,
        outbound_action=OutboundFollowUpAction.SEND,
        available_artifacts=[artifact],
    )

    assert result["delivery"]["status"] == "sent"
    assert sender.send_calls[0]["subject"] == "Owned subject"
    assert sender.send_calls[0]["body"] == "Owned complete body"
    assert sender.send_calls[0]["attachments"][0]["artifact_id"] == artifact["artifact_id"]


def test_outbound_send_action_without_current_grounded_send_is_held() -> None:
    query = "Continue our discussion."
    state = OutboundMessageState(
        channel="gmail",
        status="draft_ready",
        recipients=("alice@example.com",),
        subject="Owned subject",
        body="Owned body",
    )
    sender = RecordingSender()

    result = PlatformSelector(llm=None, senders={"gmail": sender}).select_with_outbound_context(
        _bundled(query, _semantic_payload(query)),
        ChatRequest(user_id="user-1", raw_query=query),
        outbound_state=state,
        outbound_action=OutboundFollowUpAction.SEND,
    )

    assert result["delivery"]["status"] == "pending_review"
    assert sender.send_calls == []


def test_outbound_recipient_replacement_uses_semantic_polarity() -> None:
    query = "Revise the envelope for bob@example.com, excluding alice@example.com."
    semantic = _semantic_payload(
        query,
        operation="revise",
        channel="gmail",
        action_quote="Revise the envelope",
        include=("bob@example.com",),
        exclude=("alice@example.com",),
        recipient_update="replace",
    )
    state = OutboundMessageState(
        channel="gmail",
        status="draft_ready",
        recipients=("alice@example.com",),
        subject="Owned subject",
        body="Owned body",
    )

    result = PlatformSelector(llm=None).select_with_outbound_context(
        _bundled(
            query,
            semantic,
            platform_payload={
                "outbound_message": {
                    "recipients": ["bob@example.com"],
                    "subject": "Revised subject",
                    "body": "Revised body",
                }
            },
        ),
        ChatRequest(user_id="user-1", raw_query=query),
        outbound_state=state,
        outbound_action=OutboundFollowUpAction.REVISE,
    )

    assert result["delivery"]["status"] == "draft_ready"
    assert result["draft"]["recipients"] == ["bob@example.com"]


def test_existing_attachment_reference_selects_owned_current_file(
    tmp_path: Path,
) -> None:
    query = "Associate the current slide file with the pending message."
    semantic = _semantic_payload(
        query,
        operation="revise",
        channel="gmail",
        action_quote="Associate the current slide file",
        artifact_reference="current",
        file_operation="reuse",
        file_type="pptx",
        file_action_quote="current slide file",
        file_type_quote="slide file",
    )
    pptx = _artifact(tmp_path, ".pptx")
    pdf = _artifact(tmp_path, ".pdf")
    state = OutboundMessageState(
        channel="gmail",
        status="draft_ready",
        recipients=("alice@example.com",),
        subject="Owned subject",
        body="Owned body",
        artifact_ids=(pptx["artifact_id"], pdf["artifact_id"]),
        attachment_filenames=(pptx["filename"], pdf["filename"]),
    )

    result = PlatformSelector(llm=None).select_with_outbound_context(
        _bundled(query, semantic),
        ChatRequest(user_id="user-1", raw_query=query),
        outbound_state=state,
        outbound_action=OutboundFollowUpAction.REVISE,
        available_artifacts=[pptx, pdf],
    )

    assert [item["artifact_id"] for item in result["draft"]["attachments"]] == [
        pptx["artifact_id"]
    ]
    assert result["draft"]["subject"] == "Owned subject"
    assert result["draft"]["body"] == "Owned body"


def test_answer_generation_failure_is_a_hard_delivery_stop() -> None:
    query = "Dispatch this to alice@example.com."
    semantic = _semantic_payload(
        query,
        operation="send",
        channel="gmail",
        action_quote="Dispatch this",
        include=("alice@example.com",),
    )
    sender = RecordingSender()
    response = _bundled(
        query,
        semantic,
        platform_payload={"content_composition": {"answer_succeeded": False}},
    )

    result = PlatformSelector(llm=None, senders={"gmail": sender}).select(
        response,
        ChatRequest(
            user_id="user-1", raw_query=query, platform_context=_gmail_context()
        ),
    )

    assert result["delivery"]["status"] == "failed"
    assert sender.send_calls == []


def test_error_response_preserves_artifact_but_blocks_external_delivery(
    tmp_path: Path,
) -> None:
    query = "Dispatch this to alice@example.com."
    artifact = _artifact(tmp_path, ".pptx")
    semantic = _semantic_payload(
        query,
        operation="send",
        channel="gmail",
        action_quote="Dispatch this",
        include=("alice@example.com",),
    )

    result = PlatformSelector(llm=None).select(
        _bundled(
            query,
            semantic,
            artifacts=[artifact],
            response_type=ResponseType.ERROR,
        ),
        ChatRequest(user_id="user-1", raw_query=query),
    )

    assert result["artifacts"] == [artifact]
    assert result["delivery"]["status"] == "not_requested"
