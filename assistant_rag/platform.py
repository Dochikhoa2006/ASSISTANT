"""Post-bundling platform delivery routing."""

from __future__ import annotations

from dataclasses import dataclass, field
from email.message import EmailMessage
from pathlib import Path
import imaplib
import json
import re
import smtplib
import time
from typing import Any, Protocol
from urllib import request as urlrequest

from .contracts import BundledResponse, ChatRequest
from .llm import LLMClient, LLMTask


_CHANNELS = ("gmail", "zalo", "telegram")
_EMAIL = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+(?![\w.+-])")
_SEND_WORDS = re.compile(r"\b(send|deliver|email|mail|message|nhắn|gửi)\b", re.I)
_DIRECT_SEND_WORDS = re.compile(r"\b(send|deliver|gửi)\b", re.I)
_DO_NOT_SEND = re.compile(r"\b(?:do\s+not|don't|not\s+to|without)\s+(?:send|deliver|gửi)\b", re.I)
_DRAFT_WORDS = re.compile(r"\b(compose|draft|write|prepare|soạn)\b", re.I)


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


def _gmail_recipient_candidates(text: str) -> list[str]:
    """Extract addresses only from explicit recipient clauses in the request."""
    recipients: list[str] = []
    seen: set[str] = set()
    clause_pattern = re.compile(
        r"\b(?:to|cc|bcc|recipient(?:s)?\s*(?:are|:)?|email(?:s)?(?:\s+to|\s*:|\s+))\s+"
        r"(.+?)(?=\b(?:to|cc|bcc|using|via|with|from|subject|body|username|app\s+password|credential)\b|\.(?:\s|$)|;|\n|$)",
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


def _public_artifacts(payload: dict[str, Any]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for item in payload.get("artifacts", []) or []:
        if not isinstance(item, dict):
            continue
        path = _clean(item.get("storage_path"))
        filename = _clean(item.get("filename"))
        # Artifacts must have been created by the content tool and have a real
        # local file; never accept an arbitrary path supplied in a chat request.
        if path and filename and Path(path).is_file():
            result.append({
                "artifact_id": _clean(item.get("artifact_id")),
                "filename": filename,
                "storage_path": path,
                "storage_url": _clean(item.get("storage_url")),
            })
    return result


@dataclass
class GmailSender:
    host: str = "smtp.gmail.com"
    port: int = 465
    imap_host: str = "imap.gmail.com"
    imap_port: int = 993
    draft_mailbox: str = "[Gmail]/Drafts"
    timeout_seconds: float = 20.0

    def validate_credentials(self, username: str, app_password: str) -> tuple[bool, str]:
        username = _clean(username)
        app_password = "".join(_clean(app_password).split())
        if not username or not app_password:
            return False, "Enter both a Gmail username and an app password."
        try:
            with smtplib.SMTP_SSL(self.host, self.port, timeout=self.timeout_seconds) as client:
                client.login(username, app_password)
            return True, "Gmail credentials are valid."
        except Exception as exc:
            return False, f"Gmail sign-in failed: {exc}"

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
        client = imaplib.IMAP4_SSL(
            self.imap_host,
            self.imap_port,
            timeout=self.timeout_seconds,
        )
        try:
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
        message = EmailMessage()
        message["From"] = username
        message["To"] = ", ".join(recipients)
        message["Subject"] = payload["subject"]
        message.set_content(payload["body"])
        for artifact in payload.get("attachments", []):
            path = Path(artifact["storage_path"])
            message.add_attachment(
                path.read_bytes(),
                maintype="application",
                subtype="octet-stream",
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

    ``delivery_requested`` is intentionally not read here.  It was a legacy
    keyword/flag activation gate and made the selector conditional.  The model
    below is now the single decision point on every completed assistant turn.
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
        selection = self._choose_channel(response, request)
        channel = selection["channel"]
        base["platform_selection"] = selection
        if channel == "none":
            return self._hitl_passthrough(base, response)

        formatter = self.formatters.get(channel)
        if formatter:
            base.update(formatter.format(response, request))

        message = self._extract(channel, response, request, base)
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
            if channel == "gmail" and callable(creator) and has_credentials:
                try:
                    draft_dispatch = creator(message, context)
                except Exception:
                    return self._delivery_result(
                        base,
                        channel=channel,
                        status="failed",
                        message=message,
                        question=(
                            "The Gmail draft could not be saved. Check the Gmail credentials "
                            "and account IMAP access, then try again."
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

    def _choose_channel(self, response: BundledResponse, request: ChatRequest) -> dict[str, Any]:
        """Run the selector LLM for every query; never infer from keywords."""
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
                    "answer mentions it."
                ),
                user_prompt=json.dumps({
                    "user_query": request.raw_query,
                    "bundled_response": response.final_chat_text,
                    "available_platforms": list(_CHANNELS),
                }),
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

    def _extract(self, channel: str, response: BundledResponse, request: ChatRequest, base: dict[str, Any]) -> dict[str, Any]:
        text = request.raw_query
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
                        "send now; otherwise draft."
                    ),
                    user_prompt=json.dumps({"user_query": text, "bundled_response": response.final_chat_text, "available_artifacts": [{"artifact_id": a.get("artifact_id"), "filename": a.get("filename")} for a in _public_artifacts(base)]}),
                    schema=schema,
                )
            except Exception:
                extracted = {}
        allowed_gmail_recipients = _gmail_recipient_candidates(text) if channel == "gmail" else []
        allowed_gmail_keys = {item.casefold() for item in allowed_gmail_recipients}
        recipients = _recipient_emails(extracted.get("recipients"))
        for recipient in _recipient_emails(extracted.get("recipient")):
            if recipient.casefold() not in {item.casefold() for item in recipients}:
                recipients.append(recipient)
        if channel == "gmail":
            recipients = [item for item in recipients if item.casefold() in allowed_gmail_keys]
            if not recipients:
                recipients = allowed_gmail_recipients
        # The bundled answer is the canonical body unless a platform-specific
        # extractor safely supplied a body. This makes artifact/general answers
        # usable by all delivery platforms without a second answer generator.
        body = _clean(extracted.get("body")) or response.final_chat_text
        subject = _clean(extracted.get("subject"))
        if not subject:
            subject_match = re.search(r"\bsubject\s*[:=-]\s*([^\n.;]+)", text, re.I)
            subject = subject_match.group(1).strip(" '\"") if subject_match else ""
        mode = _clean(extracted.get("mode"))
        if _DO_NOT_SEND.search(text):
            mode = "draft"
        elif _DIRECT_SEND_WORDS.search(text):
            mode = "send"
        elif _DRAFT_WORDS.search(text):
            mode = "draft"
        elif mode not in {"send", "draft"}:
            mode = "send" if _SEND_WORDS.search(text) and not _DRAFT_WORDS.search(text) else "draft"
        attachments = _public_artifacts(base) if re.search(r"\b(attach|attachment|file|document|đính kèm)\b", text, re.I) else []
        return {
            "channel": channel,
            "recipient": ", ".join(recipients),
            "recipients": recipients,
            "subject": subject,
            "body": body,
            "mode": mode,
            "attachments": attachments,
        }

    @staticmethod
    def _missing_fields(channel: str, message: dict[str, Any], context: dict[str, Any]) -> list[str]:
        missing = [field for field in ("recipient", "body") if not (_message_recipients(message) if field == "recipient" else _clean(message.get(field)))]
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
        recipients = _message_recipients(message)
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
