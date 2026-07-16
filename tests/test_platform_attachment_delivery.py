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
    ResponseType,
)
from assistant_rag.platform import GmailSender, PlatformSelector


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
) -> BundledResponse:
    text = "Subject: Generated Microsoft artifact\n\nThe generated artifact is ready."
    return BundledResponse(
        final_chat_text=text,
        response_type=ResponseType.NORMAL,
        last_qa_state=LastQAState(
            last_user_query=rewritten_query,
            last_response=text,
            response_type=ResponseType.NORMAL,
        ),
        platform_payload={"artifacts": [artifact]} if artifact is not None else {},
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
    assert "missing.xlsx" in result["delivery"]["question"]
    assert sender.send_calls == []
    assert sender.draft_calls == []
    assert result["artifacts"][0]["storage_path"] == str(missing_path)
