"""Post-bundling platform delivery routing."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from email.message import EmailMessage
from hashlib import sha256
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
from .contracts import (
    BundledResponse,
    ChatRequest,
    OutboundFollowUpAction,
    OutboundMessageState,
    ResponseType,
)
from .llm import (
    LLMClient,
    LLMTask,
    is_structured_fallback,
    validate_json_schema,
)
from .prompts import DEFAULT_PROMPT_REGISTRY, PromptContext, PromptRegistry
from .semantic_actions import (
    SemanticActionAnalyzer,
    SemanticActionDecision,
    semantic_action_from_internal_payload,
)


logger = logging.getLogger(__name__)


class DeliveryOutcomeUnknown(RuntimeError):
    """A provider may have accepted an operation before transport failure."""


OUTBOUND_REVISION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["recipients", "subject", "body", "artifact_ids"],
    "properties": {
        "recipients": {"type": "array", "items": {"type": "string"}},
        "subject": {"type": "string"},
        "body": {"type": "string"},
        "artifact_ids": {"type": "array", "items": {"type": "string"}},
    },
}


# A terminal period is ordinary sentence punctuation, not part of an address.
# Do not reject otherwise-valid recipients written at the end of a sentence.
_EMAIL_ADDRESS_PATTERN = r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"
_EMAIL = re.compile(
    rf"(?<![\w.+-]){_EMAIL_ADDRESS_PATTERN}(?![\w+-])"
)
_SUBJECT_LINE = re.compile(r"^\s*subject\s*:\s*(.+?)\s*$", re.I | re.M)
_BUNDLED_ENVELOPE_LINE = re.compile(
    r"^\s*(?:to|cc|bcc|from|recipients?|subject)\s*:\s*.*$",
    re.I,
)
_EMAIL_SALUTATION = re.compile(r"^\s*(?:dear|hello|hi)\b", re.I | re.M)


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


def _answer_generation_failed(response: BundledResponse) -> bool:
    payload = response.platform_payload
    composition = payload.get("content_composition") if isinstance(payload, dict) else None
    return bool(
        isinstance(composition, dict)
        and composition.get("answer_succeeded") is False
    )


def _composed_email_envelope(
    response: BundledResponse,
) -> tuple[str, str] | None:
    """Return valid answer-stage email copy, excluding delivery meta prose."""
    structured_subject = _clean(_bundled_message_value(response, "subject"))
    structured_body = _clean(_bundled_message_value(response, "body"))
    if (
        structured_subject
        and structured_body
    ):
        return structured_subject, structured_body

    text = _answer_generation_text(response)
    subject_match = _SUBJECT_LINE.search(text)
    if not subject_match:
        return None
    subject = _clean(subject_match.group(1))
    body = _body_from_bundled_response(text, response.platform_payload.get("artifacts"))
    if not subject or not body:
        return None
    return subject, body


def _artifact_type(item: dict[str, Any]) -> str:
    explicit = _clean(item.get("file_type")).casefold().lstrip(".")
    if explicit:
        return explicit
    return Path(_clean(item.get("filename"))).suffix.casefold().lstrip(".")


def _dedupe_artifact_ids(
    artifact_ids: list[str],
    artifacts_by_id: dict[str, dict[str, Any]],
) -> list[str]:
    selected: list[str] = []
    seen: set[str] = set()
    for artifact_id in artifact_ids:
        item = artifacts_by_id.get(artifact_id)
        if item is None:
            continue
        identity = (
            _clean(item.get("storage_path")).casefold()
            or _clean(item.get("storage_url")).casefold()
            or f"{_clean(item.get('filename')).casefold()}:{_artifact_type(item)}"
            or artifact_id.casefold()
        )
        if identity in seen:
            continue
        seen.add(identity)
        selected.append(artifact_id)
    return selected


def _referenced_artifact_ids(
    *,
    semantic_decision: SemanticActionDecision,
    artifacts_by_id: dict[str, dict[str, Any]],
    current_ids: list[str],
    new_artifact_ids: list[str],
) -> list[str] | None:
    """Resolve a model-classified artifact reference against validated IDs."""
    reference = semantic_decision.message.artifact_reference
    if reference == "none":
        return None
    requested_type = semantic_decision.file.file_type

    def relevant(artifact_id: str) -> bool:
        item = artifacts_by_id.get(artifact_id)
        return bool(
            item is not None
            and (
                requested_type == "none"
                or _artifact_type(item) == requested_type
            )
        )

    generated = [artifact_id for artifact_id in new_artifact_ids if relevant(artifact_id)]
    if reference == "newly_created":
        return _dedupe_artifact_ids(generated[-1:], artifacts_by_id)

    if reference == "current":
        current = [artifact_id for artifact_id in current_ids if relevant(artifact_id)]
        return _dedupe_artifact_ids(current, artifacts_by_id)

    if generated and semantic_decision.file.operation == "create":
        return _dedupe_artifact_ids([generated[-1]], artifacts_by_id)

    current = [artifact_id for artifact_id in current_ids if relevant(artifact_id)]
    if current:
        return _dedupe_artifact_ids(current, artifacts_by_id)

    new_keys = set(new_artifact_ids)
    prior = [
        artifact_id
        for artifact_id in artifacts_by_id
        if artifact_id not in new_keys and relevant(artifact_id)
    ]
    if prior:
        return _dedupe_artifact_ids([prior[-1]], artifacts_by_id)

    if generated:
        return _dedupe_artifact_ids([generated[-1]], artifacts_by_id)
    return []


def _bundled_message_value(response: BundledResponse, field_name: str) -> Any:
    """Read a composed envelope field without consulting selector extensions.

    ``final_chat_text`` is the normal composer output. The optional structured
    containers make the same boundary forward-compatible if a composer supplies
    fields directly, without adding another LLM extraction schema or call.
    """
    payload = response.platform_payload
    if not isinstance(payload, dict):
        return None
    for source in (
        payload.get("outbound_message"),
        payload.get("message"),
        payload,
    ):
        if isinstance(source, dict) and source.get(field_name) not in (None, "", []):
            return source.get(field_name)
    return None


def _answer_generation_text(response: BundledResponse) -> str:
    """Return the typed answer-stage output, with legacy bundle fallback."""
    payload = response.platform_payload
    if isinstance(payload, dict):
        answer_text = _clean(payload.get("answer_generation_text"))
        if answer_text:
            return answer_text
    return response.final_chat_text or ""


def _bundled_gmail_recipient_candidates(response: BundledResponse) -> list[str]:
    """Read only typed composer envelope fields, never prose body mentions."""
    return _recipient_emails(
        _bundled_message_value(response, "recipients")
        or _bundled_message_value(response, "recipient")
    )


def _prioritized_approved_recipients(
    preferred: list[str],
    approved: list[str],
) -> list[str]:
    """Prefer bundled order while rejecting inventions and filling omissions."""
    approved_by_key = {recipient.casefold(): recipient for recipient in approved}
    recipients: list[str] = []
    seen: set[str] = set()
    for recipient in preferred:
        key = recipient.casefold()
        if key in approved_by_key and key not in seen:
            recipients.append(approved_by_key[key])
            seen.add(key)
    # The composer may omit one of several explicit recipients. Preserve every
    # approved address rather than allowing the response to narrow delivery.
    for recipient in approved:
        key = recipient.casefold()
        if key not in seen:
            recipients.append(recipient)
            seen.add(key)
    return recipients


def _subject_from_bundled_response(text: str) -> str:
    match = _SUBJECT_LINE.search(text or "")
    if match:
        return _clean(match.group(1))
    # A model call is unnecessary merely to manufacture a subject. Prefer the
    # first concise, non-salutation line of the already approved bundled text.
    for raw_line in (text or "").splitlines():
        line = _clean(raw_line)
        if (
            not line
            or _BUNDLED_ENVELOPE_LINE.match(raw_line)
            or _EMAIL_SALUTATION.match(line)
        ):
            continue
        return line[:120].rstrip(" .,:;-—")
    return ""


def _body_from_bundled_response(
    text: str,
    artifacts: Any = None,
) -> str:
    """Return all authored message copy while removing only envelope syntax."""
    del artifacts
    response = _clean(text)
    subject = _SUBJECT_LINE.search(response)
    if subject:
        response = response[subject.end():].strip()
    response = "\n".join(
        line
        for line in response.splitlines()
        if not _BUNDLED_ENVELOPE_LINE.match(line)
    ).strip()
    response = re.sub(r"\n{3,}", "\n\n", response)
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
            try:
                refused = client.send_message(message)
            except (TimeoutError, OSError, smtplib.SMTPServerDisconnected) as exc:
                raise DeliveryOutcomeUnknown(
                    "Gmail delivery acknowledgement was not received"
                ) from exc
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
            try:
                status, _ = client.append(
                    mailbox,
                    "(\\Draft)",
                    imaplib.Time2Internaldate(time.time()),
                    message.as_bytes(),
                )
            except (TimeoutError, OSError) as exc:
                raise DeliveryOutcomeUnknown(
                    "Gmail draft acknowledgement was not received"
                ) from exc
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
        dispatch_id = _clean(payload.get("dispatch_id"))
        if dispatch_id:
            domain = username.rsplit("@", 1)[-1] if "@" in username else "assistant.local"
            message["Message-ID"] = (
                f"<assistant-{sha256(dispatch_id.encode('utf-8')).hexdigest()[:32]}@{domain}>"
            )
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
    semantic_analyzer: SemanticActionAnalyzer | None = None
    prompt_registry: PromptRegistry = field(default_factory=lambda: DEFAULT_PROMPT_REGISTRY)
    formatters: dict[str, PlatformFormatter] = field(default_factory=dict)
    senders: dict[str, DeliverySender] = field(default_factory=lambda: {
        "gmail": GmailSender(), "telegram": TelegramSender(), "zalo": ZaloSender(),
    })

    def __post_init__(self) -> None:
        if self.semantic_analyzer is None and self.llm is not None:
            self.semantic_analyzer = SemanticActionAnalyzer(llm=self.llm)

    def _semantic_decision(
        self,
        response: BundledResponse,
        rewritten_query: str,
    ) -> SemanticActionDecision:
        decision = semantic_action_from_internal_payload(
            (response.platform_payload or {}).get("semantic_action_decision"),
            canonical_query=rewritten_query,
        )
        if not decision.grounded and self.semantic_analyzer is not None:
            decision = self.semantic_analyzer.analyze(rewritten_query)
        return decision

    def register(self, channel: str, formatter: PlatformFormatter) -> None:
        self.formatters[channel] = formatter

    def select(self, response: BundledResponse, request: ChatRequest) -> dict[str, Any]:
        base = dict(response.platform_payload)
        context = request.platform_context or {}
        rewritten_query = response.last_qa_state.last_user_query
        if _answer_generation_failed(response):
            base["platform_selection"] = {
                "channel": "none",
                "confidence": 1.0,
                "source": "safe_fallback_answer_generation_failed",
            }
            return self._delivery_hold(
                base,
                "No message was sent or drafted because answer composition did not complete successfully.",
                channel="none",
                status="failed",
            )
        if response.response_type is ResponseType.ERROR:
            base["platform_selection"] = {
                "channel": "none",
                "confidence": 1.0,
                "source": "safe_fallback_error_response",
            }
            return self._hitl_passthrough(base, response)
        semantic_decision = self._semantic_decision(response, rewritten_query)
        base["semantic_action_decision"] = semantic_decision.to_payload()
        selection = self._choose_channel(semantic_decision)
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
            rewritten_query,
            semantic_decision=semantic_decision,
            dispatch_id=request.idempotency_key,
        )
        return self._complete_delivery(
            base,
            message=message,
            context=context,
            rewritten_query=rewritten_query,
            semantic_decision=semantic_decision,
        )

    def select_with_outbound_context(
        self,
        response: BundledResponse,
        request: ChatRequest,
        *,
        outbound_state: OutboundMessageState | None,
        outbound_action: OutboundFollowUpAction | None,
        available_artifacts: list[dict[str, Any]] | None = None,
        new_artifact_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Apply an authoritative Last-QA action to the active envelope."""
        if outbound_state is None or outbound_action is None:
            return self.select(response, request)

        base = dict(response.platform_payload)
        if _answer_generation_failed(response):
            base["platform_selection"] = {
                "channel": outbound_state.channel,
                "confidence": 1.0,
                "source": "safe_fallback_answer_generation_failed",
            }
            return self._delivery_hold(
                base,
                "No message was sent or drafted because answer composition did not complete successfully.",
                channel=outbound_state.channel,
                status="failed",
            )
        base["platform_selection"] = {
            "channel": outbound_state.channel,
            "confidence": 1.0,
            "source": "authoritative_outbound_last_qa",
        }
        rewritten_query = response.last_qa_state.last_user_query
        semantic_decision = self._semantic_decision(response, rewritten_query)
        base["semantic_action_decision"] = semantic_decision.to_payload()
        if outbound_action in {
            OutboundFollowUpAction.SEND,
            OutboundFollowUpAction.REVISE_AND_SEND,
        } and not semantic_decision.message.authorizes_send:
            return self._delivery_hold(
                base,
                "The active message was not sent because this turn did not contain a current, explicit send instruction.",
                channel=outbound_state.channel,
                draft=self._message_from_outbound_state(
                    outbound_state,
                    artifacts=list(available_artifacts or []),
                    mode="draft",
                ),
                status="pending_review",
            )
        artifacts = list(available_artifacts or [])
        message = self._message_from_outbound_state(
            outbound_state,
            artifacts=artifacts,
            mode=(
                "send"
                if outbound_action
                in {
                    OutboundFollowUpAction.SEND,
                    OutboundFollowUpAction.REVISE_AND_SEND,
                }
                else "draft"
            ),
        )
        message["dispatch_id"] = request.idempotency_key
        if outbound_action in {
            OutboundFollowUpAction.REVISE,
            OutboundFollowUpAction.REVISE_AND_SEND,
        }:
            message = self._revise_outbound_message(
                message,
                rewritten_query=rewritten_query,
                artifacts=artifacts,
                new_artifact_ids=list(new_artifact_ids or []),
                authoritative_envelope=(
                    _composed_email_envelope(response)
                    if (
                        outbound_state.channel == "gmail"
                        and semantic_decision.message.copy_revision
                    )
                    else None
                ),
                semantic_decision=semantic_decision,
            )
            if bool(message.pop("_revision_failed", False)):
                return self._delivery_hold(
                    base,
                    "I kept the existing message text and recipients because the requested revision could not be validated. Any generated files remain available, and no message was sent.",
                    channel=outbound_state.channel,
                    draft=message,
                    status="pending_review",
                )
        return self._complete_delivery(
            base,
            message=message,
            context=request.platform_context or {},
            rewritten_query=response.last_qa_state.last_user_query,
            semantic_decision=semantic_decision,
        )

    def _message_from_outbound_state(
        self,
        state: OutboundMessageState,
        *,
        artifacts: list[dict[str, Any]],
        mode: str,
    ) -> dict[str, Any]:
        artifacts_by_id = {
            str(item.get("artifact_id") or ""): item
            for item in artifacts
            if isinstance(item, dict) and item.get("artifact_id")
        }
        declared = [
            artifacts_by_id[artifact_id]
            for artifact_id in state.artifact_ids
            if artifact_id in artifacts_by_id
        ]
        attachments, unavailable = resolve_generated_artifacts(declared)
        missing_ids = [
            artifact_id
            for artifact_id in state.artifact_ids
            if artifact_id not in artifacts_by_id
        ]
        for index, artifact_id in enumerate(state.artifact_ids):
            if artifact_id not in missing_ids:
                continue
            filename = (
                state.attachment_filenames[index]
                if index < len(state.attachment_filenames)
                and state.attachment_filenames[index]
                else artifact_id
            )
            if filename not in unavailable:
                unavailable.append(filename)
        excluded_keys = {
            recipient.casefold() for recipient in state.excluded_recipients
        }
        recipients = [
            recipient
            for recipient in state.recipients
            if recipient.casefold() not in excluded_keys
        ]
        return {
            "channel": state.channel,
            "recipient": ", ".join(recipients),
            "recipients": recipients,
            "subject": state.subject,
            "body": state.body,
            "mode": mode,
            "attachments": attachments,
            "unavailable_attachments": list(dict.fromkeys(unavailable)),
            "excluded_recipients": list(state.excluded_recipients),
        }

    def _revise_outbound_message(
        self,
        message: dict[str, Any],
        *,
        rewritten_query: str,
        artifacts: list[dict[str, Any]],
        new_artifact_ids: list[str],
        authoritative_envelope: tuple[str, str] | None,
        semantic_decision: SemanticActionDecision,
    ) -> dict[str, Any]:
        artifacts_by_id = {
            str(item.get("artifact_id") or ""): item
            for item in artifacts
            if isinstance(item, dict) and item.get("artifact_id")
        }
        current_ids = [
            str(item.get("artifact_id") or "")
            for item in message.get("attachments", [])
            if isinstance(item, dict) and item.get("artifact_id")
        ]
        referenced_ids = _referenced_artifact_ids(
            semantic_decision=semantic_decision,
            artifacts_by_id=artifacts_by_id,
            current_ids=current_ids,
            new_artifact_ids=new_artifact_ids,
        )
        selected_ids = (
            referenced_ids
            if referenced_ids is not None
            else _dedupe_artifact_ids(
                list(dict.fromkeys([*current_ids, *new_artifact_ids])),
                artifacts_by_id,
            )
        )
        channel = str(message.get("channel") or "gmail")
        preserve_existing_copy = bool(
            not semantic_decision.message.copy_revision
            and authoritative_envelope is None
        )
        revised: dict[str, Any] = {}
        # Attaching a referenced file is a deterministic envelope mutation. It
        # must retain the already-approved subject/body and does not need a
        # writing model to paraphrase or manufacture replacement copy.
        if not preserve_existing_copy and self.llm is not None:
            try:
                revised = self.llm.generate_json(
                    task=LLMTask.WRITING,
                    system_prompt=self.prompt_registry.system("outbound_revision"),
                    user_prompt=self.prompt_registry.user(
                        PromptContext(
                            stage="outbound_revision",
                            rewritten_query=rewritten_query,
                            extra={
                                "active_outbound_message": {
                                    "recipients": list(message.get("recipients") or []),
                                    "subject": str(message.get("subject") or ""),
                                    "body": str(message.get("body") or ""),
                                    "artifact_ids": current_ids,
                                },
                                "available_artifacts": [
                                    {
                                        "artifact_id": artifact_id,
                                        "filename": str(item.get("filename") or ""),
                                        "is_new_this_turn": artifact_id
                                        in set(new_artifact_ids),
                                    }
                                    for artifact_id, item in artifacts_by_id.items()
                                ],
                            },
                        )
                    ),
                    schema=OUTBOUND_REVISION_SCHEMA,
                )
                if is_structured_fallback(revised):
                    revised = {}
                elif revised:
                    validate_json_schema(revised, OUTBOUND_REVISION_SCHEMA)
                    if not (
                        _clean(revised.get("subject"))
                        and _clean(revised.get("body"))
                        and isinstance(revised.get("recipients"), list)
                        and isinstance(revised.get("artifact_ids"), list)
                    ):
                        revised = {}
            except Exception as exc:
                logger.warning("Outbound draft revision failed: %s", type(exc).__name__)
                revised = {}

        recipient_parser = (
            _recipient_emails if channel == "gmail" else _recipient_identifiers
        )
        existing_recipients = recipient_parser(message.get("recipients") or [])
        proposed_recipients = recipient_parser(revised.get("recipients"))
        if channel == "gmail":
            included = list(semantic_decision.message.included_recipients)
            excluded = list(semantic_decision.message.excluded_recipients)
            excluded_keys = {
                recipient.casefold() for recipient in excluded
            }
            if semantic_decision.message.recipient_update == "replace" and included:
                recipients = _recipient_emails(included)
            elif semantic_decision.message.recipient_update == "add" and included:
                recipients = _recipient_emails([*existing_recipients, *included])
            elif semantic_decision.message.recipient_update == "remove":
                recipients = [
                    recipient
                    for recipient in existing_recipients
                    if recipient.casefold() not in excluded_keys
                ]
            else:
                allowed_keys = {
                    recipient.casefold() for recipient in existing_recipients
                }
                recipients = [
                    recipient
                    for recipient in proposed_recipients
                    if recipient.casefold() in allowed_keys
                ] or existing_recipients
            recipients = [
                recipient
                for recipient in recipients
                if recipient.casefold() not in excluded_keys
            ]
        else:
            recipients = proposed_recipients or existing_recipients

        proposed_ids = [
            str(value)
            for value in revised.get("artifact_ids", [])
            if str(value) in artifacts_by_id
        ] if isinstance(revised.get("artifact_ids"), list) else []
        if referenced_ids is None and not preserve_existing_copy:
            selected_ids = _dedupe_artifact_ids(
                list(
                    dict.fromkeys(
                        [*(proposed_ids or selected_ids), *new_artifact_ids]
                    )
                ),
                artifacts_by_id,
            )
        declared = [
            artifacts_by_id[artifact_id]
            for artifact_id in selected_ids
            if artifact_id in artifacts_by_id
        ]
        attachments, unavailable = resolve_generated_artifacts(declared)
        if authoritative_envelope is not None:
            subject, body = authoritative_envelope
        else:
            subject = _clean(revised.get("subject")) or _clean(message.get("subject"))
            body = _clean(revised.get("body")) or _clean(message.get("body"))
        revision_failed = bool(not recipients) or bool(
            not preserve_existing_copy
            and not revised
            and authoritative_envelope is None
        )
        return {
            **message,
            "recipient": ", ".join(recipients),
            "recipients": recipients,
            "subject": subject,
            "body": body,
            "attachments": attachments,
            "unavailable_attachments": unavailable,
            "_revision_failed": revision_failed,
        }

    def _complete_delivery(
        self,
        base: dict[str, Any],
        *,
        message: dict[str, Any],
        context: dict[str, Any],
        rewritten_query: str,
        semantic_decision: SemanticActionDecision,
    ) -> dict[str, Any]:
        channel = str(message.get("channel") or "none")
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
                (
                    f"{channel.title()} delivery was not attempted because the required "
                    f"delivery configuration is incomplete: {', '.join(missing)}."
                ),
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
                and semantic_decision.message.operation == "save_draft"
                and callable(creator)
                and has_credentials
            ):
                try:
                    draft_dispatch = creator(message, context)
                except DeliveryOutcomeUnknown:
                    return self._delivery_result(
                        base,
                        channel=channel,
                        status="delivery_unknown",
                        message=message,
                        notice=(
                            "Gmail may have saved the draft, but its acknowledgement was lost. "
                            "Check Gmail Drafts before attempting the operation again."
                        ),
                    )
                except Exception as exc:
                    logger.warning("Gmail draft save failed: %s", type(exc).__name__)
                    return self._delivery_result(
                        base,
                        channel=channel,
                        status="failed",
                        message=message,
                        notice=(
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
        except DeliveryOutcomeUnknown:
            return self._delivery_result(
                base,
                channel=channel,
                status="delivery_unknown",
                message=message,
                notice=(
                    f"The {channel.title()} provider may have accepted the message, "
                    "but its acknowledgement was lost. Check sent messages before any manual retry."
                ),
            )
        except Exception:
            return self._delivery_result(
                base,
                channel=channel,
                status="failed",
                message=message,
                notice=(
                    f"The {channel.title()} message could not be sent. "
                    "Review the chatbot delivery configuration and recipient details before retrying."
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
                notice=(
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
        semantic_decision: SemanticActionDecision,
    ) -> dict[str, Any]:
        """Choose a channel only from the already-grounded semantic contract."""
        message = semantic_decision.message
        if not semantic_decision.grounded or not message.requests_message:
            return {
                "channel": "none",
                "confidence": message.confidence,
                "source": "safe_fallback_no_grounded_message_action",
            }
        return {
            "channel": message.channel,
            "confidence": message.confidence,
            "source": "grounded_semantic_contract",
        }

    def _extract(
        self,
        channel: str,
        response: BundledResponse,
        rewritten_query: str,
        *,
        semantic_decision: SemanticActionDecision,
        dispatch_id: str | None = None,
    ) -> dict[str, Any]:
        del rewritten_query
        included = list(semantic_decision.message.included_recipients)
        excluded = list(semantic_decision.message.excluded_recipients)
        allowed_gmail_recipients = included if channel == "gmail" else []
        if channel == "gmail":
            allowed_gmail_recipients = _recipient_emails(
                [
                    *allowed_gmail_recipients,
                    *(
                        response.platform_payload.get(
                            "approved_resolved_recipients", []
                        )
                        if isinstance(
                            response.platform_payload.get(
                                "approved_resolved_recipients", []
                            ),
                            list,
                        )
                        else []
                    ),
                ]
            )
            excluded_keys = {item.casefold() for item in excluded}
            allowed_gmail_recipients = [
                item
                for item in allowed_gmail_recipients
                if item.casefold() not in excluded_keys
            ]
        recipient_parser = _recipient_emails if channel == "gmail" else _recipient_identifiers
        recipients = recipient_parser(included)
        if channel == "gmail":
            recipients = _prioritized_approved_recipients(
                _bundled_gmail_recipient_candidates(response),
                allowed_gmail_recipients,
            )
        # Structured bundle fields win over its plain-text envelope, and both
        # win over supporting inputs. This keeps composition authoritative and
        # avoids latency plus corruption from a redundant extraction LLM.
        bundled_artifacts = response.platform_payload.get("artifacts")
        answer_text = _answer_generation_text(response)
        body = _clean(_bundled_message_value(response, "body")) or (
            _body_from_bundled_response(
                answer_text,
                bundled_artifacts,
            )
        )
        bundled_subject = _clean(_bundled_message_value(response, "subject"))
        subject_line = _SUBJECT_LINE.search(answer_text)
        if not bundled_subject and subject_line:
            bundled_subject = _clean(subject_line.group(1))
        subject = bundled_subject
        if not subject:
            subject = _subject_from_bundled_response(answer_text)
        attachments: list[dict[str, str]] = []
        unavailable_attachments: list[str] = []
        if channel == "gmail":
            attachments, unavailable_attachments = resolve_generated_artifacts(
                bundled_artifacts
            )
        if not subject and attachments:
            subject = "Requested file"
        mode = "send" if semantic_decision.message.authorizes_send else "draft"
        return {
            "channel": channel,
            "recipient": ", ".join(recipients),
            "recipients": recipients,
            "subject": subject,
            "body": body,
            "mode": mode,
            "attachments": attachments,
            "unavailable_attachments": unavailable_attachments,
            "excluded_recipients": (
                excluded if channel == "gmail" else []
            ),
            "dispatch_id": dispatch_id,
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
        # Track each outcome so a later retry targets only failures instead of
        # duplicating already delivered messages.
        delivered: list[str] = []
        refused: list[str] = []
        for recipient in recipients:
            try:
                result = sender.send(
                    {**message, "recipient": recipient, "recipients": [recipient]},
                    context,
                )
            except Exception:
                refused.append(recipient)
                continue
            if isinstance(result, dict) and result.get("status") not in (None, "sent"):
                refused.append(recipient)
            else:
                delivered.append(recipient)
        return {
            "status": "sent" if not refused else "partial_failure",
            "provider": channel,
            "recipient": ", ".join(recipients),
            "recipients": recipients,
            "delivered_recipients": delivered,
            "refused_recipients": refused,
        }

    def _delivery_result(
        self,
        base: dict[str, Any],
        *,
        channel: str,
        status: str,
        message: dict[str, Any],
        provider: str | None = None,
        notice: str | None = None,
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
        if notice:
            base["delivery"]["notice"] = notice
        if dispatch and (
            status != "sent" or dispatch.get("refused_recipients")
        ):
            for key in ("delivered_recipients", "refused_recipients"):
                if key in dispatch:
                    base["delivery"][key] = list(dispatch[key])
        base["draft"] = self._public_message(message)
        return base

    @staticmethod
    def _public_message(message: dict[str, Any]) -> dict[str, Any]:
        public = {
            key: value
            for key, value in message.items()
            if key not in {"storage_path", "dispatch_id"}
        }
        public["attachments"] = [
            {key: value for key, value in attachment.items() if key != "storage_path"}
            for attachment in message.get("attachments", [])
        ]
        return public

    def _delivery_hold(self, base: dict[str, Any], notice: str, *, channel: str | None = None, draft: dict[str, Any] | None = None, status: str = "needs_input") -> dict[str, Any]:
        """Return non-conversational delivery state without creating a question."""
        base.update({
            "delivery": {
                "channel": channel or "none",
                "status": status,
                "notice": notice,
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
