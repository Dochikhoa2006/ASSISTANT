"""Post-bundling platform delivery routing."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from email.message import EmailMessage
from pathlib import Path
import imaplib
import json
import logging
import re
import smtplib
import time
from typing import Any, Protocol
from urllib import request as urlrequest

from .artifacts import artifact_mime_type, resolve_generated_artifacts
from .contracts import BundledResponse, ChatRequest, ResponseType
from .llm import LLMClient, LLMTask
from .chat_history import CHAT_HISTORY_PROMPT_RULE, inject_chat_history


logger = logging.getLogger(__name__)


_CHANNELS = ("gmail", "zalo", "telegram")
# A terminal period is ordinary sentence punctuation, not part of an address.
# Do not reject otherwise-valid recipients written at the end of a sentence.
_EMAIL = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+(?![\w+-])")
_SEND_WORDS = re.compile(r"\b(send|deliver|email|mail|message|nhắn|gửi)\b", re.I)
_DIRECT_SEND_WORDS = re.compile(r"\b(send|deliver|gửi)\b", re.I)
_DO_NOT_SEND = re.compile(
    r"\b(?:do\s+not|don't|not\s+to|without)\s+(?:send|deliver|email|mail|gửi)\b",
    re.I,
)
_DRAFT_WORDS = re.compile(r"\b(compose|draft|write|prepare|soạn)\b", re.I)
_EMAIL_MESSAGE_WORDS = re.compile(r"\b(?:email|mail|message)\b", re.I)
_DIRECT_EMAIL_VERB = re.compile(r"\b(?:email|mail)\s+(?:to\s+)?[\w.+-]+@", re.I)
_EMAIL_OBJECT_DELIVERY = re.compile(
    r"\b(?:email|mail)\b(?:"
    r"\s+(?:it|this|that|them)\s+to\b"
    r"|\s+(?:[\w'-]+\s+){0,6}\.?"
    r"(?:file|document|workbook|spreadsheet|worksheet|pdf|report|presentation|deck|slides|"
    r"powerpoint|excel|xlsx|pptx|docx)"
    r"\s+to\b)",
    re.I,
)
_FILE_DELIVERY_WORDS = re.compile(
    r"\b(?:file|attachment|document|report|excel|spreadsheet|workbook|worksheet|xlsx|"
    r"pdf|powerpoint|presentation|deck|slides|pptx|docx)\b",
    re.I,
)
_SUBJECT_LINE = re.compile(r"^\s*subject\s*:\s*(.+?)\s*$", re.I | re.M)
_EMAIL_SALUTATION = re.compile(r"^\s*(?:dear|hello|hi)\b", re.I | re.M)
_SAVE_GMAIL_DRAFT = re.compile(
    r"\b(?:save|store|create|add|put)\b[\s\w'-]{0,48}\b(?:gmail(?:'s)?\s+)?drafts?\b"
    r"|\b(?:gmail(?:'s)?\s+)?drafts?\b[\s\w'-]{0,32}\b(?:save|store|create|add|put)\b",
    re.I,
)


class PlatformFormatter(Protocol):
    def format(self, response: BundledResponse, request: ChatRequest) -> dict[str, Any]:
        ...


class DeliverySender(Protocol):
    def send(self, payload: dict[str, Any], platform_context: dict[str, Any]) -> dict[str, Any]:
        ...


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _recipient_emails(value: Any) -> list[str]:
    """Return unique, syntactically valid email recipients in supplied order."""
    values = value if isinstance(value, (list, tuple, set)) else [value]
    recipients: list[str] = []
    seen: set[str] = set()
    for item in values:
        for email in _EMAIL.findall(_clean(item)):
            key = email.casefold()
            if key not in seen:
                seen.add(key)
                recipients.append(email)
    return recipients


def _message_recipients(payload: dict[str, Any]) -> list[str]:
    return _recipient_emails(payload.get("recipients") or payload.get("recipient"))


def _recipient_identifiers(value: Any) -> list[str]:
    """Return unique, non-empty platform recipient identifiers in order."""
    values = value if isinstance(value, (list, tuple, set)) else [value]
    recipients: list[str] = []
    seen: set[str] = set()
    for item in values:
        recipient = _clean(item)
        key = recipient.casefold()
        if recipient and key not in seen:
            seen.add(key)
            recipients.append(recipient)
    return recipients


def _channel_message_recipients(channel: str, payload: dict[str, Any]) -> list[str]:
    """Apply email validation only to Gmail; other channels use opaque IDs."""
    if channel == "gmail":
        return _message_recipients(payload)
    return _recipient_identifiers(payload.get("recipients") or payload.get("recipient"))


def _gmail_recipient_candidates(text: str) -> list[str]:
    """Extract addresses only from explicit recipient clauses in the request."""
    recipients: list[str] = []
    seen: set[str] = set()
    clause_pattern = re.compile(
        r"\b(?:to|cc|bcc|recipient(?:s)?\s*(?:are|:)?|email(?:s)?(?:\s+to|\s*:|\s+))\s+"
        r"(.+?)(?=\b(?:to|cc|bcc|using|via|with|from|subject|body|username|app\s+password|credential)\b|\.(?:\s|$)|\n|$)",
        re.I | re.S,
    )
    for match in clause_pattern.finditer(text):
        for email in _recipient_emails(match.group(1)):
            key = email.casefold()
            if key not in seen:
                seen.add(key)
                recipients.append(email)
    if recipients:
        return recipients
    # With no credential language, email addresses in an explicit Gmail
    # delivery request are safe fallback recipient candidates. The selector
    # has already established that this is a delivery operation.
    if not re.search(r"\b(?:username|app\s+password|credential)\b", text, re.I):
        return _recipient_emails(text)
    return []


def _is_explicit_email_message_request(text: str) -> bool:
    """Recognize a request to prepare or send an email with stated recipients.

    This deliberately requires both a real email address and message-writing
    language. It does not route ordinary questions that merely mention an
    address, and it never decides whether a message may be sent.
    """
    has_message_action = bool(
        (_DRAFT_WORDS.search(text) or _DIRECT_SEND_WORDS.search(text))
        and _EMAIL_MESSAGE_WORDS.search(text)
    )
    has_file_delivery_action = bool(_DIRECT_SEND_WORDS.search(text) and _FILE_DELIVERY_WORDS.search(text))
    return bool(
        _recipient_emails(text)
        and (
            has_message_action
            or has_file_delivery_action
            or _DIRECT_EMAIL_VERB.search(text)
            or _EMAIL_OBJECT_DELIVERY.search(text)
        )
    )


def _requests_gmail_draft_save(text: str) -> bool:
    """Return whether the user explicitly authorized saving to Gmail Drafts.

    Writing or drafting an email is local preparation. Persisting it to a
    provider mailbox is a separate external action and must be requested
    directly, even when Gmail credentials are available in the UI.
    """
    return bool(_SAVE_GMAIL_DRAFT.search(text))


def _subject_from_bundled_response(text: str) -> str:
    match = _SUBJECT_LINE.search(text or "")
    return _clean(match.group(1)) if match else ""


def _body_from_bundled_response(text: str) -> str:
    """Prefer the drafted message portion when the answer contains one."""
    response = _clean(text)
    salutation = _EMAIL_SALUTATION.search(response)
    if salutation:
        return response[salutation.start():].strip()
    return response


def _safe_imap_failure_reason(exc: Exception) -> str:
    """Describe an IMAP failure without exposing provider responses or secrets."""
    if isinstance(exc, imaplib.IMAP4.error):
        return "Gmail rejected the IMAP sign-in or draft-mailbox request"
    if isinstance(exc, RuntimeError):
        return "Gmail rejected the draft-mailbox operation"
    if isinstance(exc, (TimeoutError, OSError)):
        return "the Gmail IMAP service could not be reached"
    return "the Gmail IMAP draft service returned an unexpected error"


def _public_artifacts(payload: dict[str, Any]) -> list[dict[str, str]]:
    resolved, _ = resolve_generated_artifacts(payload.get("artifacts"))
    return resolved


@dataclass
class GmailSender:
    host: str = "smtp.gmail.com"
    port: int = 465
    imap_host: str = "imap.gmail.com"
    imap_port: int = 993
    draft_mailbox: str = "[Gmail]/Drafts"
    timeout_seconds: float = 20.0

    def validate_smtp_credentials(self, username: str, app_password: str) -> tuple[bool, str]:
        """Verify the credentials needed to send Gmail messages.

        This intentionally does not contact IMAP. Sending and saving a remote
        Gmail draft are distinct operations, and the UI's normal credential
        check should stay fast and side-effect free.
        """
        username = _clean(username)
        app_password = "".join(_clean(app_password).split())
        if not username or not app_password:
            return False, "Enter both a Gmail username and an app password."
        try:
            with smtplib.SMTP_SSL(self.host, self.port, timeout=self.timeout_seconds) as client:
                client.login(username, app_password)
        except Exception as exc:
            logger.warning("Gmail SMTP credential validation failed: %s", type(exc).__name__)
            return False, "Gmail SMTP sign-in failed. Verify the Gmail username and app password."
        return True, "Gmail sending credentials are valid."

    def validate_credentials(self, username: str, app_password: str) -> tuple[bool, str]:
        """Verify both SMTP sending and optional IMAP Gmail-draft access."""
        valid, message = self.validate_smtp_credentials(username, app_password)
        if not valid:
            return False, message
        username = _clean(username)
        app_password = "".join(_clean(app_password).split())

        client: Any | None = None
        try:
            client = imaplib.IMAP4_SSL(
                self.imap_host,
                self.imap_port,
                timeout=self.timeout_seconds,
            )
            client.login(username, app_password)
            status, _ = client.list()
            if _clean(status).upper() != "OK":
                raise RuntimeError("Gmail did not allow draft mailbox listing.")
        except Exception as exc:
            logger.warning("Gmail IMAP credential validation failed: %s", type(exc).__name__)
            return False, f"Gmail IMAP draft access failed: {_safe_imap_failure_reason(exc)}."
        finally:
            if client is not None:
                try:
                    client.logout()
                except Exception:
                    pass
        return True, "Gmail SMTP and IMAP draft access are valid."

    def send(self, payload: dict[str, Any], platform_context: dict[str, Any]) -> dict[str, Any]:
        username = _clean(platform_context.get("gmail_username"))
        app_password = "".join(_clean(platform_context.get("gmail_app_password")).split())
        if not username or not app_password:
            raise ValueError("Gmail username and app password are required before sending.")
        recipients = _message_recipients(payload)
        if not recipients:
            raise ValueError("At least one valid Gmail recipient is required before sending.")
        message = self._email_message(payload, username, recipients)
        with smtplib.SMTP_SSL(self.host, self.port, timeout=self.timeout_seconds) as client:
            client.login(username, app_password)
            refused = client.send_message(message)
        if refused:
            refused_keys = {str(item).casefold() for item in refused}
            delivered = [item for item in recipients if item.casefold() not in refused_keys]
            return {
                "status": "partial_failure",
                "provider": "gmail",
                "recipient": ", ".join(recipients),
                "recipients": recipients,
                "delivered_recipients": delivered,
                "refused_recipients": [item for item in recipients if item.casefold() in refused_keys],
            }
        return {
            "status": "sent",
            "provider": "gmail",
            "recipient": ", ".join(recipients),
            "recipients": recipients,
        }

    def create_draft(self, payload: dict[str, Any], platform_context: dict[str, Any]) -> dict[str, Any]:
        username = _clean(platform_context.get("gmail_username"))
        app_password = "".join(_clean(platform_context.get("gmail_app_password")).split())
        if not username or not app_password:
            raise ValueError("Gmail username and app password are required before saving a Gmail draft.")
        recipients = _message_recipients(payload)
        if not recipients:
            raise ValueError("At least one valid Gmail recipient is required before saving a draft.")
        message = self._email_message(payload, username, recipients)
        client: Any | None = None
        try:
            client = imaplib.IMAP4_SSL(
                self.imap_host,
                self.imap_port,
                timeout=self.timeout_seconds,
            )
            client.login(username, app_password)
            mailbox = self._resolve_draft_mailbox(client)
            status, _ = client.append(
                mailbox,
                "(\\Draft)",
                imaplib.Time2Internaldate(time.time()),
                message.as_bytes(),
            )
            if _clean(status).upper() != "OK":
                raise RuntimeError("Gmail rejected the draft append operation.")
        finally:
            if client is not None:
                try:
                    client.logout()
                except Exception:
                    pass
        return {
            "status": "draft_saved",
            "provider": "gmail",
            "recipient": ", ".join(recipients),
            "recipients": recipients,
        }

    def _email_message(self, payload: dict[str, Any], username: str, recipients: list[str]) -> EmailMessage:
        attachments, unavailable = resolve_generated_artifacts(payload.get("attachments"))
        if unavailable:
            raise ValueError(
                "Generated attachment files are unavailable: " + ", ".join(unavailable)
            )
        message = EmailMessage()
        message["From"] = username
        message["To"] = ", ".join(recipients)
        message["Subject"] = payload["subject"]
        message.set_content(payload["body"])
        for artifact in attachments:
            path = Path(artifact["storage_path"])
            maintype, subtype = artifact_mime_type(artifact["filename"]).split("/", 1)
            message.add_attachment(
                path.read_bytes(),
                maintype=maintype,
                subtype=subtype,
                filename=artifact["filename"],
            )
        return message

    def _resolve_draft_mailbox(self, client: Any) -> str:
        try:
            status, rows = client.list()
            if _clean(status).upper() == "OK":
                for raw_row in rows or []:
                    row = raw_row.decode("utf-8", errors="replace") if isinstance(raw_row, bytes) else _clean(raw_row)
                    if "\\Drafts" not in row:
                        continue
                    quoted = re.search(r'"([^"]+)"\s*$', row)
                    if quoted:
                        return quoted.group(1)
                    return row.rsplit(" ", 1)[-1].strip('"')
        except Exception:
            pass
        return self.draft_mailbox


@dataclass
class TelegramSender:
    timeout_seconds: float = 20.0

    def send(self, payload: dict[str, Any], platform_context: dict[str, Any]) -> dict[str, Any]:
        token = _clean(platform_context.get("telegram_bot_token"))
        if not token:
            raise ValueError("A Telegram bot token is required before sending.")
        endpoint = f"https://api.telegram.org/bot{token}/sendMessage"
        data = json.dumps({"chat_id": payload["recipient"], "text": payload["body"]}).encode("utf-8")
        req = urlrequest.Request(endpoint, data=data, headers={"Content-Type": "application/json"}, method="POST")
        with urlrequest.urlopen(req, timeout=self.timeout_seconds) as response:
            result = json.loads(response.read().decode("utf-8"))
        if not result.get("ok"):
            raise RuntimeError("Telegram rejected the message.")
        return {"status": "sent", "provider": "telegram", "recipient": payload["recipient"]}


@dataclass
class ZaloSender:
    timeout_seconds: float = 20.0

    def send(self, payload: dict[str, Any], platform_context: dict[str, Any]) -> dict[str, Any]:
        token = _clean(platform_context.get("zalo_access_token"))
        endpoint = _clean(platform_context.get("zalo_api_url"))
        if not token or not endpoint:
            raise ValueError("A Zalo access token and approved Zalo API URL are required before sending.")
        # The endpoint is supplied by the approved Zalo integration because OA
        # message payload requirements differ by account/product.
        data = json.dumps({"recipient": payload["recipient"], "message": {"text": payload["body"]}}).encode("utf-8")
        req = urlrequest.Request(endpoint, data=data, headers={"Content-Type": "application/json", "access_token": token}, method="POST")
        with urlrequest.urlopen(req, timeout=self.timeout_seconds) as response:
            result = json.loads(response.read().decode("utf-8"))
        if result.get("error") not in (None, 0):
            raise RuntimeError("Zalo rejected the message.")
        return {"status": "sent", "provider": "zalo", "recipient": payload["recipient"]}


@dataclass
class PlatformSelector:
    """Choose a platform and prepare delivery state.

    Explicit email-message requests with literal email recipients always enter
    the Gmail preparation flow. All other channel selection is model-driven.
    A local draft is the default; saving to Gmail Drafts or sending both require
    an explicit external-action request.
    """

    llm: LLMClient | None = None
    formatters: dict[str, PlatformFormatter] = field(default_factory=dict)
    senders: dict[str, DeliverySender] = field(default_factory=lambda: {
        "gmail": GmailSender(), "telegram": TelegramSender(), "zalo": ZaloSender(),
    })

    def register(self, channel: str, formatter: PlatformFormatter) -> None:
        self.formatters[channel] = formatter

    def select(self, response: BundledResponse, request: ChatRequest) -> dict[str, Any]:
        base = dict(response.platform_payload)
        context = request.platform_context or {}
        rewritten_query = response.last_qa_state.last_user_query
        if response.response_type is ResponseType.ERROR:
            base["platform_selection"] = {
                "channel": "none",
                "confidence": 1.0,
                "source": "safe_fallback_error_response",
            }
            return self._hitl_passthrough(base, response)
        selection = self._choose_channel(response, rewritten_query)
        channel = selection["channel"]
        base["platform_selection"] = selection
        if channel == "none":
            return self._hitl_passthrough(base, response)

        formatter = self.formatters.get(channel)
        if formatter:
            generated_artifacts = base.get("artifacts")
            had_generated_artifacts = "artifacts" in base
            # Preserve the formatter protocol while preventing extensions from
            # observing or acting on pre-rewrite text through ChatRequest.
            formatter_request = replace(request, raw_query=rewritten_query)
            base.update(formatter.format(response, formatter_request))
            if had_generated_artifacts:
                # A channel formatter may shape text, but it cannot replace or
                # discard files created by the content-composition stage.
                base["artifacts"] = generated_artifacts
            else:
                base.pop("artifacts", None)

        message = self._extract(
            channel,
            response,
            base,
            rewritten_query,
        )
        unavailable_attachments = list(message.get("unavailable_attachments") or [])
        if unavailable_attachments:
            return self._delivery_hold(
                base,
                "The message was not sent or drafted because these generated "
                "attachments are unavailable: "
                + ", ".join(unavailable_attachments)
                + ". Regenerate the files and try again.",
                channel=channel,
                draft=message,
                status="failed",
            )
        missing = self._missing_fields(channel, message, context)
        if missing:
            return self._delivery_hold(
                base,
                f"To prepare the {channel.title()} message, please provide: {', '.join(missing)}.",
                channel=channel,
                draft=message,
            )

        if message["mode"] == "draft":
            creator = getattr(self.senders.get(channel), "create_draft", None)
            has_credentials = bool(
                _clean(context.get("gmail_username"))
                and _clean(context.get("gmail_app_password"))
            )
            if (
                channel == "gmail"
                and _requests_gmail_draft_save(rewritten_query)
                and callable(creator)
                and has_credentials
            ):
                try:
                    draft_dispatch = creator(message, context)
                except Exception as exc:
                    logger.warning("Gmail draft save failed: %s", type(exc).__name__)
                    return self._delivery_result(
                        base,
                        channel=channel,
                        status="failed",
                        message=message,
                        question=(
                            "The Gmail draft could not be saved because "
                            f"{_safe_imap_failure_reason(exc)}. Check the Gmail app password "
                            "and IMAP access, then try again."
                        ),
                    )
                return self._delivery_result(
                    base,
                    channel=channel,
                    status="draft_saved",
                    message=message,
                    provider=_clean(draft_dispatch.get("provider")) or channel,
                    dispatch=draft_dispatch,
                )
            return self._delivery_result(
                base,
                channel=channel,
                status="draft_ready",
                message=message,
            )

        # A send is authorized only by an explicit send-mode extraction and a
        # complete, channel-specific credential set. The sender receives the
        # credentials from platform_context; they are never placed in the
        # delivery payload, drafts, history, or audit message.
        try:
            dispatch = self._send(channel, message, context)
        except Exception:
            return self._delivery_result(
                base,
                channel=channel,
                status="failed",
                message=message,
                question=(
                    f"The {channel.title()} message could not be sent. "
                    "Check the delivery credentials and recipient details, then try again."
                ),
            )
        if dispatch.get("status") != "sent":
            refused = list(dispatch.get("refused_recipients") or [])
            delivered = list(dispatch.get("delivered_recipients") or [])
            return self._delivery_result(
                base,
                channel=channel,
                status="partial_failure",
                message=message,
                provider=_clean(dispatch.get("provider")) or channel,
                question=(
                    f"The {channel.title()} message was delivered to "
                    f"{', '.join(delivered) or 'no recipients'}, but was refused for "
                    f"{', '.join(refused) or 'one or more recipients'}."
                ),
                dispatch=dispatch,
            )
        return self._delivery_result(
            base,
            channel=channel,
            status="sent",
            message=message,
            provider=_clean(dispatch.get("provider")) or channel,
            dispatch=dispatch,
        )

    def _choose_channel(
        self,
        response: BundledResponse,
        rewritten_query: str,
    ) -> dict[str, Any]:
        """Choose Gmail deterministically for explicit email messages, else use the LLM."""
        if _is_explicit_email_message_request(rewritten_query):
            return {
                "channel": "gmail",
                "confidence": 1.0,
                "source": "deterministic_explicit_email_request",
            }
        if self.llm is None:
            return {"channel": "none", "confidence": 0.0, "source": "safe_fallback_no_llm"}
        schema = {
            "type": "object",
            "required": ["channel", "confidence"],
            "properties": {
                "channel": {"type": "string", "enum": ["gmail", "zalo", "telegram", "none"]},
                "confidence": {"type": "number"},
            },
        }
        try:
            plan = self.llm.generate_json(
                task=LLMTask.ACTION_PLANNING,
                system_prompt=(
                    "You are the platform selector in an assistant pipeline. "
                    "For every turn choose exactly one channel: gmail, zalo, telegram, or none. "
                    "Choose a messaging channel only when the user explicitly asks the assistant "
                    "to prepare or send the bundled answer to a recipient through that channel. "
                    "Choose none for ordinary questions, explanations, drafts without a delivery channel, "
                    "or questions about how a platform works. Never choose a channel only because the "
                    f"answer mentions it.\n\n{CHAT_HISTORY_PROMPT_RULE}"
                ),
                user_prompt=json.dumps(
                    inject_chat_history({
                        "rewritten_query": rewritten_query,
                        "bundled_response": response.final_chat_text,
                        "available_platforms": list(_CHANNELS),
                    })
                ),
                schema=schema,
            )
            candidate = _clean(plan.get("channel")).casefold()
            if candidate in (*_CHANNELS, "none"):
                confidence = plan.get("confidence", 0.0)
                try:
                    confidence = max(0.0, min(1.0, float(confidence)))
                except (TypeError, ValueError):
                    confidence = 0.0
                return {"channel": candidate, "confidence": confidence, "source": "llm"}
        except Exception:
            pass
        # Selection failures are safe: continue to the HITL pass-through path
        # rather than accidentally preparing or sending a message.
        return {"channel": "none", "confidence": 0.0, "source": "safe_fallback_llm_error"}

    def _extract(
        self,
        channel: str,
        response: BundledResponse,
        base: dict[str, Any],
        rewritten_query: str,
    ) -> dict[str, Any]:
        text = rewritten_query
        extracted: dict[str, Any] = {}
        if self.llm is not None:
            schema = {"type": "object", "required": ["subject", "body", "mode"], "properties": {"recipient": {"type": "string"}, "recipients": {"type": "array", "items": {"type": "string"}}, "subject": {"type": "string"}, "body": {"type": "string"}, "mode": {"type": "string", "enum": ["send", "draft"]}}}
            try:
                extracted = self.llm.generate_json(
                    task=LLMTask.ACTION_PLANNING,
                    system_prompt=(
                        f"You extract a {channel} message only from explicit user-provided facts. "
                        "Do not invent recipients, subject, body, or attachments. Return every explicitly "
                        "requested recipient in recipients. Mode is send only for an explicit request to "
                        f"send now; otherwise draft.\n\n{CHAT_HISTORY_PROMPT_RULE}"
                    ),
                    user_prompt=json.dumps(
                        inject_chat_history({
                            "rewritten_query": text,
                            "bundled_response": response.final_chat_text,
                            "available_artifacts": [
                                {
                                    "artifact_id": artifact.get("artifact_id"),
                                    "filename": artifact.get("filename"),
                                }
                                for artifact in _public_artifacts(base)
                            ],
                        })
                    ),
                    schema=schema,
                )
            except Exception:
                extracted = {}
        allowed_gmail_recipients = _gmail_recipient_candidates(text) if channel == "gmail" else []
        recipient_parser = _recipient_emails if channel == "gmail" else _recipient_identifiers
        recipients = recipient_parser(extracted.get("recipients"))
        for recipient in recipient_parser(extracted.get("recipient")):
            if recipient.casefold() not in {item.casefold() for item in recipients}:
                recipients.append(recipient)
        if channel == "gmail":
            # Literal addresses in the rewritten request are authoritative.
            # The LLM extractor may neither invent addresses nor narrow the
            # explicit ordered recipient set.
            recipients = allowed_gmail_recipients
        # The bundled answer is the canonical body unless a platform-specific
        # extractor safely supplied a body. This makes artifact/general answers
        # usable by all delivery platforms without a second answer generator.
        body = _clean(extracted.get("body")) or _body_from_bundled_response(response.final_chat_text)
        subject = _clean(extracted.get("subject"))
        if not subject:
            subject_match = re.search(r"\bsubject\s*[:=-]\s*([^\n.;]+)", text, re.I)
            subject = subject_match.group(1).strip(" '\"") if subject_match else ""
        if not subject:
            subject = _subject_from_bundled_response(response.final_chat_text)
        attachments: list[dict[str, str]] = []
        unavailable_attachments: list[str] = []
        if channel == "gmail":
            attachments, unavailable_attachments = resolve_generated_artifacts(
                base.get("artifacts")
            )
        if not subject and attachments:
            subject = "Requested file"
        mode = _clean(extracted.get("mode"))
        if _DO_NOT_SEND.search(text):
            mode = "draft"
        elif _DIRECT_SEND_WORDS.search(text):
            mode = "send"
        elif _DRAFT_WORDS.search(text):
            mode = "draft"
        elif _DIRECT_EMAIL_VERB.search(text) or _EMAIL_OBJECT_DELIVERY.search(text):
            mode = "send"
        elif mode not in {"send", "draft"}:
            mode = "send" if _SEND_WORDS.search(text) and not _DRAFT_WORDS.search(text) else "draft"
        return {
            "channel": channel,
            "recipient": ", ".join(recipients),
            "recipients": recipients,
            "subject": subject,
            "body": body,
            "mode": mode,
            "attachments": attachments,
            "unavailable_attachments": unavailable_attachments,
        }

    @staticmethod
    def _missing_fields(channel: str, message: dict[str, Any], context: dict[str, Any]) -> list[str]:
        missing = [field for field in ("recipient", "body") if not (_channel_message_recipients(channel, message) if field == "recipient" else _clean(message.get(field)))]
        if channel == "gmail":
            if not _message_recipients(message):
                missing = [item for item in missing if item != "recipient"] + ["at least one valid recipient email address"]
            if not _clean(message.get("subject")):
                missing.append("subject")
            if message.get("mode") == "send" and not _clean(context.get("gmail_username")):
                missing.append("Gmail username")
            if message.get("mode") == "send" and not _clean(context.get("gmail_app_password")):
                missing.append("Gmail app password")
        elif message.get("mode") == "send":
            credential = "telegram_bot_token" if channel == "telegram" else "zalo_access_token and Zalo API URL"
            if channel == "telegram" and not _clean(context.get("telegram_bot_token")):
                missing.append(credential)
            if channel == "zalo" and (not _clean(context.get("zalo_access_token")) or not _clean(context.get("zalo_api_url"))):
                missing.append(credential)
        return list(dict.fromkeys(missing))

    def _send(self, channel: str, message: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        sender = self.senders.get(channel)
        if sender is None:
            raise ValueError(f"No sender is configured for {channel}.")
        recipients = _channel_message_recipients(channel, message)
        if channel == "gmail":
            return sender.send(message, context)

        # Non-email channels conventionally accept one recipient per request.
        # Keep the selector contract uniform by dispatching each extracted
        # recipient in order and failing the operation if any dispatch fails.
        for recipient in recipients:
            sender.send({**message, "recipient": recipient, "recipients": [recipient]}, context)
        return {"status": "sent", "provider": channel, "recipient": ", ".join(recipients), "recipients": recipients}

    def _delivery_result(
        self,
        base: dict[str, Any],
        *,
        channel: str,
        status: str,
        message: dict[str, Any],
        provider: str | None = None,
        question: str | None = None,
        dispatch: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        base["delivery"] = {
            "channel": channel,
            "status": status,
            "recipient": message["recipient"],
            "recipients": list(message["recipients"]),
        }
        if provider:
            base["delivery"]["provider"] = provider
        if question:
            base["delivery"]["question"] = question
        if dispatch:
            for key in ("delivered_recipients", "refused_recipients"):
                if key in dispatch:
                    base["delivery"][key] = list(dispatch[key])
        base["draft"] = self._public_message(message)
        return base

    @staticmethod
    def _public_message(message: dict[str, Any]) -> dict[str, Any]:
        public = {key: value for key, value in message.items() if key != "storage_path"}
        public["attachments"] = [
            {key: value for key, value in attachment.items() if key != "storage_path"}
            for attachment in message.get("attachments", [])
        ]
        return public

    def _delivery_hold(self, base: dict[str, Any], question: str, *, channel: str | None = None, draft: dict[str, Any] | None = None, status: str = "needs_input") -> dict[str, Any]:
        """Return a platform-specific delivery requirement."""
        base.update({
            "delivery": {
                "channel": channel or "none",
                "status": status,
                "question": question,
            },
        })
        if draft:
            base["draft"] = self._public_message(draft)
        return base

    @staticmethod
    def _hitl_passthrough(base: dict[str, Any], response: BundledResponse) -> dict[str, Any]:
        """The ``none`` route carries no platform delivery requirement."""
        base.update({
            "text": response.final_chat_text,
            "delivery": {"channel": "none", "status": "not_requested"},
        })
        return base


class PlainTextFormatter:
    def format(self, response: BundledResponse, request: ChatRequest) -> dict[str, Any]:
        return {"text": response.final_chat_text}
