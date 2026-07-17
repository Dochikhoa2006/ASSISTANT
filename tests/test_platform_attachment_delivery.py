from __future__ import annotations

from dataclasses import dataclass, field
from email import policy
from email.message import EmailMessage
from email.parser import BytesParser
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
from assistant_rag.llm import LLMTask, structured_fallback_payload
from assistant_rag.platform import (
    PLATFORM_CHANNEL_SELECTION_SCHEMA,
    GmailSender,
    PlatformSelector,
)


_RECIPIENTS = ["alice@example.com", "bob@example.com"]
_RAW_QUERY_SENTINEL = "RAW_SENTINEL audit-only ingress text."


class IncompletePlatformLLM:
    """Omit Bob and invent Mallory so rewritten-query authority is exercised."""

    def __init__(self, mode: str = "send") -> None:
        self.mode = mode

    def generate_json(self, **_kwargs: Any) -> dict[str, Any]:
        return {
            "recipients": ["alice@example.com", "mallory@example.com"],
            "subject": "Generated Microsoft artifact",
            "body": "The generated artifact is ready.",
            "mode": self.mode,
        }


class UnexpectedPlatformLLM:
    def generate_json(self, **_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("deterministic platform routing must not need an LLM call")


@dataclass
class RecordingSender:
    send_calls: list[dict[str, Any]] = field(default_factory=list)
    draft_calls: list[dict[str, Any]] = field(default_factory=list)

    def send(
        self,
        payload: dict[str, Any],
        _context: dict[str, Any],
    ) -> dict[str, Any]:
        self.send_calls.append(payload)
        return {"status": "sent", "provider": "gmail"}

    def create_draft(
        self,
        payload: dict[str, Any],
        _context: dict[str, Any],
    ) -> dict[str, Any]:
        self.draft_calls.append(payload)
        return {"status": "draft_saved", "provider": "gmail"}


def _bundled(
    artifact: dict[str, Any] | None,
    *,
    rewritten_query: str = "Create and deliver the artifact.",
    final_chat_text: str | None = None,
    platform_payload: dict[str, Any] | None = None,
) -> BundledResponse:
    text = final_chat_text or (
        "Subject: Generated Microsoft artifact\n\n"
        "The generated artifact is ready."
    )
    payload = dict(platform_payload or {})
    if artifact is not None:
        payload["artifacts"] = [artifact]
    return BundledResponse(
        final_chat_text=text,
        response_type=ResponseType.NORMAL,
        last_qa_state=LastQAState(
            last_user_query=rewritten_query,
            last_response=text,
            response_type=ResponseType.NORMAL,
        ),
        platform_payload=payload,
    )


def _artifact(
    tmp_path: Path,
    *,
    suffix: str,
    payload: bytes,
) -> dict[str, Any]:
    path = tmp_path / f"generated{suffix}"
    path.write_bytes(payload)
    return {
        "artifact_id": f"artifact-{suffix.lstrip('.')}",
        "file_type": suffix.lstrip("."),
        "filename": path.name,
        "storage_path": str(path),
        "storage_url": f"/artifacts/{path.name}",
        "status": "created",
    }


@pytest.mark.parametrize(
    "query",
    (
        "Email it to alice@example.com and bob@example.com.",
        "Email the generated workbook to alice@example.com and bob@example.com.",
        "Email the .pptx to alice@example.com and bob@example.com.",
        "Send the .xlsx to alice@example.com and bob@example.com.",
    ),
)
def test_microsoft_artifact_email_phrases_route_to_gmail_deterministically(
    query: str,
) -> None:
    result = PlatformSelector(llm=None).select(
        _bundled(None, rewritten_query=query),
        ChatRequest(user_id="user-1", raw_query=_RAW_QUERY_SENTINEL),
    )

    assert result["platform_selection"] == {
        "channel": "gmail",
        "confidence": 1.0,
        "source": "deterministic_explicit_email_request",
    }
    assert result["draft"]["recipients"] == _RECIPIENTS


def test_explicit_gmail_delivery_skips_redundant_message_extraction_llm() -> None:
    query = "Email the update to alice@example.com and bob@example.com."

    result = PlatformSelector(llm=UnexpectedPlatformLLM()).select(
        _bundled(None, rewritten_query=query),
        ChatRequest(user_id="user-1", raw_query=_RAW_QUERY_SENTINEL),
    )

    assert result["platform_selection"]["channel"] == "gmail"
    assert result["draft"]["recipients"] == _RECIPIENTS
    assert result["draft"]["subject"] == "Generated Microsoft artifact"
    assert result["draft"]["body"] == "The generated artifact is ready."


def test_gmail_envelope_prioritizes_bundled_recipient_order_subject_and_body() -> None:
    query = (
        "Draft an email to alice@example.com and bob@example.com; "
        "subject: Supporting-input subject."
    )
    result = PlatformSelector(llm=UnexpectedPlatformLLM()).select(
        _bundled(
            None,
            rewritten_query=query,
            final_chat_text=(
                "To: bob@example.com, alice@example.com\n"
                "Subject: Composer-owned subject\n\n"
                "Dear team,\n\nComposer-owned body."
            ),
        ),
        ChatRequest(user_id="user-1", raw_query=_RAW_QUERY_SENTINEL),
    )

    assert result["draft"]["recipients"] == [
        "bob@example.com",
        "alice@example.com",
    ]
    assert result["draft"]["subject"] == "Composer-owned subject"
    assert result["draft"]["body"] == "Dear team,\n\nComposer-owned body."


def test_gmail_bundle_cannot_invent_or_drop_approved_recipients() -> None:
    query = "Draft an email to alice@example.com and bob@example.com."
    result = PlatformSelector(llm=None).select(
        _bundled(
            None,
            rewritten_query=query,
            final_chat_text=(
                "To: mallory@example.com, alice@example.com\n"
                "Subject: Approved update\n\nThe approved update is ready."
            ),
        ),
        ChatRequest(user_id="user-1", raw_query=_RAW_QUERY_SENTINEL),
    )

    assert result["draft"]["recipients"] == [
        "alice@example.com",
        "bob@example.com",
    ]
    assert "mallory@example.com" not in result["draft"]["recipient"]


def test_structured_bundled_envelope_precedes_plain_text_fields() -> None:
    query = "Draft an email to alice@example.com and bob@example.com."
    result = PlatformSelector(llm=None).select(
        _bundled(
            None,
            rewritten_query=query,
            final_chat_text=(
                "To: alice@example.com, bob@example.com\n"
                "Subject: Plain-text subject\n\nPlain-text body."
            ),
            platform_payload={
                "outbound_message": {
                    "recipients": ["bob@example.com", "alice@example.com"],
                    "subject": "Structured bundle subject",
                    "body": "Structured bundle body.",
                }
            },
        ),
        ChatRequest(user_id="user-1", raw_query=_RAW_QUERY_SENTINEL),
    )

    assert result["draft"]["recipients"] == [
        "bob@example.com",
        "alice@example.com",
    ]
    assert result["draft"]["subject"] == "Structured bundle subject"
    assert result["draft"]["body"] == "Structured bundle body."


def test_platform_formatter_cannot_override_bundled_envelope_or_artifact(
    tmp_path: Path,
) -> None:
    artifact = _artifact(tmp_path, suffix=".pdf", payload=b"%PDF bundled")
    replacement = _artifact(
        tmp_path,
        suffix=".xlsx",
        payload=b"formatter replacement",
    )
    query = "Draft an email to alice@example.com and bob@example.com."

    class OverridingFormatter:
        def format(
            self,
            _response: BundledResponse,
            _request: ChatRequest,
        ) -> dict[str, Any]:
            return {
                "recipients": ["mallory@example.com"],
                "subject": "Formatter subject",
                "body": "Formatter body.",
                "artifacts": [replacement],
            }

    selector = PlatformSelector(llm=None)
    selector.register("gmail", OverridingFormatter())
    result = selector.select(
        _bundled(
            artifact,
            rewritten_query=query,
            final_chat_text=(
                "To: bob@example.com, alice@example.com\n"
                "Subject: Bundled subject\n\nBundled body."
            ),
        ),
        ChatRequest(user_id="user-1", raw_query=_RAW_QUERY_SENTINEL),
    )

    assert result["draft"]["recipients"] == [
        "bob@example.com",
        "alice@example.com",
    ]
    assert result["draft"]["subject"] == "Bundled subject"
    assert result["draft"]["body"] == "Bundled body."
    assert result["draft"]["attachments"][0]["artifact_id"] == artifact["artifact_id"]
    assert result["artifacts"] == [artifact]


def test_ordinary_response_skips_platform_selection_llm() -> None:
    result = PlatformSelector(llm=UnexpectedPlatformLLM()).select(
        _bundled(None, rewritten_query="Explain the approved project update."),
        ChatRequest(user_id="user-1", raw_query=_RAW_QUERY_SENTINEL),
    )

    assert result["platform_selection"] == {
        "channel": "none",
        "confidence": 1.0,
        "source": "deterministic_no_platform_request",
    }
    assert result["delivery"] == {"channel": "none", "status": "not_requested"}


def test_platform_channel_structured_fallback_is_fail_closed() -> None:
    assert set(PLATFORM_CHANNEL_SELECTION_SCHEMA["properties"]) == {"channel"}
    fallback = structured_fallback_payload(
        task=LLMTask.ACTION_PLANNING,
        schema=PLATFORM_CHANNEL_SELECTION_SCHEMA,
        user_prompt="{}",
        error=ValueError("invalid channel output"),
    )

    assert fallback == {"channel": "none"}


def test_email_address_question_does_not_route_to_gmail() -> None:
    rewritten_query = "Is alice@example.com an email address?"
    result = PlatformSelector(llm=None).select(
        _bundled(None, rewritten_query=rewritten_query),
        ChatRequest(
            user_id="user-1",
            raw_query="RAW_SENTINEL Send an email to raw-only@example.com now.",
        ),
    )

    assert result["platform_selection"]["channel"] == "none"


@pytest.mark.parametrize(
    "query",
    (
        "Create the workbook, but do not email it to alice@example.com and bob@example.com.",
        "Create the workbook; don't mail it to alice@example.com and bob@example.com.",
    ),
)
def test_negative_email_delivery_language_never_authorizes_send(query: str) -> None:
    sender = RecordingSender()
    result = PlatformSelector(
        llm=IncompletePlatformLLM("send"),
        senders={"gmail": sender},
    ).select(
        _bundled(None, rewritten_query=query),
        ChatRequest(
            user_id="user-1",
            raw_query="RAW_SENTINEL Send an email to raw-only@example.com now.",
        ),
    )

    assert result["platform_selection"]["channel"] == "gmail"
    assert result["delivery"]["status"] == "draft_ready"
    assert result["draft"]["mode"] == "draft"
    assert result["draft"]["recipients"] == _RECIPIENTS
    assert sender.send_calls == []


def test_error_response_keeps_ui_artifact_but_blocks_external_delivery(
    tmp_path: Path,
) -> None:
    artifact = _artifact(tmp_path, suffix=".xlsx", payload=b"recoverable-workbook")
    error_response = _bundled(artifact)
    error_response = BundledResponse(
        final_chat_text="The conversation write failed safely.",
        response_type=ResponseType.ERROR,
        last_qa_state=LastQAState(
            last_user_query="Create and send the workbook.",
            last_response="The conversation write failed safely.",
            response_type=ResponseType.ERROR,
        ),
        platform_payload=error_response.platform_payload,
    )
    sender = RecordingSender()

    result = PlatformSelector(
        llm=IncompletePlatformLLM("send"),
        senders={"gmail": sender},
    ).select(
        error_response,
        ChatRequest(
            user_id="user-1",
            raw_query=(
                "Email the workbook to alice@example.com and bob@example.com."
            ),
            platform_context={
                "gmail_username": "sender@example.com",
                "gmail_app_password": "app-password",
            },
        ),
    )

    assert result["platform_selection"]["source"] == "safe_fallback_error_response"
    assert result["delivery"] == {"channel": "none", "status": "not_requested"}
    assert result["artifacts"] == [artifact]
    assert sender.send_calls == []
    assert sender.draft_calls == []


def test_platform_delivery_authorization_uses_only_rewritten_query() -> None:
    selector = PlatformSelector(llm=None)

    authorized = selector.select(
        _bundled(
            None,
            rewritten_query=(
                "REWRITTEN_SENTINEL Email it to alice@example.com and bob@example.com."
            ),
        ),
        ChatRequest(user_id="user-1", raw_query=_RAW_QUERY_SENTINEL),
    )
    blocked = selector.select(
        _bundled(
            None,
            rewritten_query="REWRITTEN_SENTINEL Explain the completed artifact.",
        ),
        ChatRequest(
            user_id="user-1",
            raw_query=(
                "RAW_SENTINEL Send the artifact to alice@example.com and bob@example.com."
            ),
        ),
    )

    assert authorized["platform_selection"]["channel"] == "gmail"
    assert authorized["draft"]["recipients"] == _RECIPIENTS
    assert blocked["platform_selection"]["channel"] == "none"
    assert blocked["delivery"] == {"channel": "none", "status": "not_requested"}


@pytest.mark.parametrize(
    ("file_phrase", "suffix", "mime_type"),
    (
        (
            "Excel workbook",
            ".xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ),
        ("PDF report", ".pdf", "application/pdf"),
        (
            "PowerPoint presentation",
            ".pptx",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ),
    ),
)
def test_local_gmail_draft_automatically_includes_every_microsoft_artifact(
    tmp_path: Path,
    file_phrase: str,
    suffix: str,
    mime_type: str,
) -> None:
    del mime_type
    artifact = _artifact(tmp_path, suffix=suffix, payload=f"bytes-{suffix}".encode())
    sender = RecordingSender()
    rewritten_query = (
        f"Create a {file_phrase} and draft an email to "
        "alice@example.com; bob@example.com; and ALICE@example.com."
    )
    result = PlatformSelector(
        llm=IncompletePlatformLLM("draft"),
        senders={"gmail": sender},
    ).select(
        _bundled(artifact, rewritten_query=rewritten_query),
        ChatRequest(
            user_id="user-1",
            raw_query=_RAW_QUERY_SENTINEL,
        ),
    )

    assert result["platform_selection"]["channel"] == "gmail"
    assert result["delivery"]["status"] == "draft_ready"
    assert result["delivery"]["recipients"] == _RECIPIENTS
    assert result["draft"]["recipients"] == _RECIPIENTS
    assert [item["filename"] for item in result["draft"]["attachments"]] == [
        artifact["filename"]
    ]
    assert "storage_path" not in result["draft"]["attachments"][0]
    assert result["artifacts"][0]["storage_path"] == artifact["storage_path"]
    assert sender.send_calls == []
    assert sender.draft_calls == []


class CaptureSMTP:
    instances: list["CaptureSMTP"] = []

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.message: EmailMessage | None = None
        self.login_args: tuple[str, str] | None = None
        type(self).instances.append(self)

    def __enter__(self) -> "CaptureSMTP":
        return self

    def __exit__(self, *_args: Any) -> None:
        return None

    def login(self, username: str, password: str) -> None:
        self.login_args = (username, password)

    def send_message(self, message: EmailMessage) -> dict[str, object]:
        self.message = message
        return {}


class CaptureIMAP:
    instances: list["CaptureIMAP"] = []

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.appended: bytes | None = None
        type(self).instances.append(self)

    def login(self, _username: str, _password: str) -> tuple[str, list[bytes]]:
        return "OK", []

    def list(self) -> tuple[str, list[bytes]]:
        return "OK", [b'(\\HasNoChildren \\Drafts) "/" "[Gmail]/Drafts"']

    def append(
        self,
        _mailbox: str,
        _flags: str,
        _date: str,
        message: bytes,
    ) -> tuple[str, list[bytes]]:
        self.appended = message
        return "OK", []

    def logout(self) -> tuple[str, list[bytes]]:
        return "BYE", []


def _assert_wire_message(
    message: EmailMessage,
    *,
    artifact: dict[str, Any],
    expected_bytes: bytes,
    expected_mime: str,
) -> None:
    assert str(message["To"]) == ", ".join(_RECIPIENTS)
    attachments = list(message.iter_attachments())
    assert len(attachments) == 1
    assert attachments[0].get_filename() == artifact["filename"]
    assert attachments[0].get_content_type() == expected_mime
    assert attachments[0].get_payload(decode=True) == expected_bytes


def test_authoritative_outbound_follow_up_sends_exact_envelope_and_pdf(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pdf_bytes = b"%PDF-1.4 exact follow-up attachment"
    artifact = _artifact(tmp_path, suffix=".pdf", payload=pdf_bytes)
    outbound = OutboundMessageState(
        channel="gmail",
        status="draft_ready",
        recipients=tuple(_RECIPIENTS),
        subject="Health absence notice",
        body="I am unwell and will remain home today.",
        artifact_ids=(artifact["artifact_id"],),
        attachment_filenames=(artifact["filename"],),
        source_topic_id="topic-mail",
        source_hop_id="hop-mail",
    )
    response = _bundled(
        None,
        rewritten_query="Carry out the pending outbound operation.",
    )
    CaptureSMTP.instances.clear()
    monkeypatch.setattr("assistant_rag.platform.smtplib.SMTP_SSL", CaptureSMTP)

    result = PlatformSelector(
        llm=UnexpectedPlatformLLM(),
        senders={"gmail": GmailSender()},
    ).select_with_outbound_context(
        response,
        ChatRequest(
            user_id="user-1",
            raw_query=_RAW_QUERY_SENTINEL,
            platform_context={
                "gmail_username": "sender@example.com",
                "gmail_app_password": "app-password",
            },
        ),
        outbound_state=outbound,
        outbound_action=OutboundFollowUpAction.SEND,
        available_artifacts=[artifact],
    )

    assert result["platform_selection"]["source"] == "authoritative_outbound_last_qa"
    assert result["delivery"]["status"] == "sent"
    assert result["draft"]["subject"] == outbound.subject
    assert result["draft"]["body"] == outbound.body
    assert "storage_path" not in result["draft"]["attachments"][0]
    message = CaptureSMTP.instances[-1].message
    assert message is not None
    _assert_wire_message(
        message,
        artifact=artifact,
        expected_bytes=pdf_bytes,
        expected_mime="application/pdf",
    )


def test_outbound_revision_adds_generated_pdf_then_follow_up_sends_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pdf_bytes = b"%PDF-1.4 medicine details"
    artifact = _artifact(tmp_path, suffix=".pdf", payload=pdf_bytes)

    class RevisionLLM:
        calls: list[dict[str, Any]] = []

        def generate_json(self, **kwargs: Any) -> dict[str, Any]:
            self.calls.append(kwargs)
            return {
                "recipients": list(_RECIPIENTS),
                "subject": "Medical absence notice",
                "body": (
                    "I am sick and need to remain home. The attached PDF "
                    "contains the medicine details."
                ),
                "artifact_ids": [artifact["artifact_id"]],
            }

    initial = OutboundMessageState(
        channel="gmail",
        status="draft_ready",
        recipients=tuple(_RECIPIENTS),
        subject="Day-off notice",
        body="I will be away today.",
    )
    response = _bundled(
        artifact,
        rewritten_query=(
            "Update the message to explain that I am sick, and include the "
            "generated PDF with my medicine details."
        ),
    )
    selector = PlatformSelector(llm=RevisionLLM(), senders={"gmail": GmailSender()})
    revised = selector.select_with_outbound_context(
        response,
        ChatRequest(user_id="user-1", raw_query=_RAW_QUERY_SENTINEL),
        outbound_state=initial,
        outbound_action=OutboundFollowUpAction.REVISE,
        available_artifacts=[artifact],
        new_artifact_ids=[artifact["artifact_id"]],
    )

    assert revised["delivery"]["status"] == "draft_ready"
    assert revised["draft"]["subject"] == "Medical absence notice"
    assert "medicine details" in revised["draft"]["body"]
    assert revised["draft"]["attachments"] == [
        {
            "artifact_id": artifact["artifact_id"],
            "filename": artifact["filename"],
            "storage_url": artifact["storage_url"],
            "file_type": "pdf",
        }
    ]

    revised_state = OutboundMessageState(
        channel="gmail",
        status="draft_ready",
        recipients=tuple(revised["draft"]["recipients"]),
        subject=revised["draft"]["subject"],
        body=revised["draft"]["body"],
        artifact_ids=(artifact["artifact_id"],),
        attachment_filenames=(artifact["filename"],),
    )
    CaptureSMTP.instances.clear()
    monkeypatch.setattr("assistant_rag.platform.smtplib.SMTP_SSL", CaptureSMTP)
    sent = selector.select_with_outbound_context(
        _bundled(None, rewritten_query="Transmit the active message now."),
        ChatRequest(
            user_id="user-1",
            raw_query=_RAW_QUERY_SENTINEL,
            platform_context={
                "gmail_username": "sender@example.com",
                "gmail_app_password": "app-password",
            },
        ),
        outbound_state=revised_state,
        outbound_action=OutboundFollowUpAction.SEND,
        available_artifacts=[artifact],
    )

    assert sent["delivery"]["status"] == "sent"
    message = CaptureSMTP.instances[-1].message
    assert message is not None
    _assert_wire_message(
        message,
        artifact=artifact,
        expected_bytes=pdf_bytes,
        expected_mime="application/pdf",
    )


def test_outbound_revision_structured_failure_never_sends_or_asks_in_a_loop() -> None:
    class FallbackRevisionLLM:
        def generate_json(self, **kwargs: Any) -> dict[str, Any]:
            return structured_fallback_payload(
                task=LLMTask.WRITING,
                schema=kwargs["schema"],
                user_prompt=kwargs["user_prompt"],
                error=ValueError("invalid revision output"),
            )

    sender = RecordingSender()
    state = OutboundMessageState(
        channel="gmail",
        status="draft_ready",
        recipients=("alice@example.com",),
        subject="Existing subject",
        body="Existing body.",
    )

    result = PlatformSelector(
        llm=FallbackRevisionLLM(),
        senders={"gmail": sender},
    ).select_with_outbound_context(
        _bundled(None, rewritten_query="Change the active message."),
        ChatRequest(user_id="user-1", raw_query=_RAW_QUERY_SENTINEL),
        outbound_state=state,
        outbound_action=OutboundFollowUpAction.REVISE_AND_SEND,
    )

    assert result["delivery"]["status"] == "pending_review"
    assert "?" not in result["delivery"]["notice"]
    assert "question" not in result["delivery"]
    assert result["draft"]["subject"] == state.subject
    assert result["draft"]["body"] == state.body
    assert sender.send_calls == []
    assert sender.draft_calls == []


@pytest.mark.parametrize(
    ("file_phrase", "suffix", "expected_mime"),
    (
        (
            "Excel workbook",
            ".xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ),
        ("PDF report", ".pdf", "application/pdf"),
        (
            "PowerPoint presentation",
            ".pptx",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ),
    ),
)
@pytest.mark.parametrize("delivery_mode", ("send", "saved_draft"))
def test_multi_recipient_gmail_wire_payload_contains_generated_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    file_phrase: str,
    suffix: str,
    expected_mime: str,
    delivery_mode: str,
) -> None:
    expected_bytes = f"exact-generated-{suffix}-bytes".encode()
    artifact = _artifact(tmp_path, suffix=suffix, payload=expected_bytes)
    credentials = {
        "gmail_username": "sender@example.com",
        "gmail_app_password": "app-password",
    }
    if delivery_mode == "send":
        CaptureSMTP.instances.clear()
        monkeypatch.setattr("assistant_rag.platform.smtplib.SMTP_SSL", CaptureSMTP)
        query = (
            f"Create a {file_phrase} and email it to alice@example.com and "
            "bob@example.com now."
        )
    else:
        CaptureIMAP.instances.clear()
        monkeypatch.setattr("assistant_rag.platform.imaplib.IMAP4_SSL", CaptureIMAP)
        query = (
            f"Create a {file_phrase}, then save a Gmail draft email to "
            "alice@example.com and bob@example.com; do not send."
        )

    result = PlatformSelector(
        # The rewritten query, not an under-authorizing model extraction, owns send mode.
        llm=IncompletePlatformLLM("draft"),
        senders={"gmail": GmailSender()},
    ).select(
        _bundled(artifact, rewritten_query=query),
        ChatRequest(
            user_id="user-1",
            raw_query=_RAW_QUERY_SENTINEL,
            platform_context=credentials,
        ),
    )

    expected_status = "sent" if delivery_mode == "send" else "draft_saved"
    assert result["delivery"]["status"] == expected_status
    assert result["delivery"]["recipients"] == _RECIPIENTS
    assert result["draft"]["recipients"] == _RECIPIENTS
    assert result["draft"]["attachments"][0]["filename"] == artifact["filename"]
    assert "storage_path" not in result["draft"]["attachments"][0]

    if delivery_mode == "send":
        message = CaptureSMTP.instances[-1].message
        assert message is not None
    else:
        appended = CaptureIMAP.instances[-1].appended
        assert appended is not None
        message = BytesParser(policy=policy.default).parsebytes(appended)
    _assert_wire_message(
        message,
        artifact=artifact,
        expected_bytes=expected_bytes,
        expected_mime=expected_mime,
    )


@pytest.mark.parametrize("delivery_mode", ("send", "saved_draft"))
def test_unavailable_declared_artifact_blocks_gmail_before_dispatch(
    tmp_path: Path,
    delivery_mode: str,
) -> None:
    missing_path = tmp_path / "missing.xlsx"
    artifact = {
        "artifact_id": "artifact-missing",
        "file_type": "xlsx",
        "filename": missing_path.name,
        "storage_path": str(missing_path),
        "status": "created",
    }
    sender = RecordingSender()
    query = (
        "Create an Excel workbook and email it to alice@example.com and bob@example.com."
        if delivery_mode == "send"
        else "Create an Excel workbook and save a Gmail draft email to alice@example.com and bob@example.com; do not send."
    )
    result = PlatformSelector(
        llm=IncompletePlatformLLM("send"),
        senders={"gmail": sender},
    ).select(
        _bundled(artifact, rewritten_query=query),
        ChatRequest(
            user_id="user-1",
            raw_query=_RAW_QUERY_SENTINEL,
            platform_context={
                "gmail_username": "sender@example.com",
                "gmail_app_password": "app-password",
            },
        ),
    )

    assert result["delivery"]["status"] == "failed"
    assert "missing.xlsx" in result["delivery"]["notice"]
    assert "question" not in result["delivery"]
    assert sender.send_calls == []
    assert sender.draft_calls == []
    assert result["artifacts"][0]["storage_path"] == str(missing_path)
