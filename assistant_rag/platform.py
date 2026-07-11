"""Post-bundling delivery routing and the single post-selector HITL gate.

Every completed turn passes through :class:`PlatformSelector`.  Its LLM receives
the user's query and final bundled response and chooses exactly one of
``gmail``, ``zalo``, ``telegram`` or ``none``.  The selector only prepares
delivery state; it never decides which user-facing HITL question is displayed.

:class:`PostSelectorHITL` is the only component that turns that state into a
question.  It asks either one normal, high-value supporting question on the
``none`` route, or one delivery-specific question on a selected-platform route.
The two scopes never overwrite or combine with one another.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from email.message import EmailMessage
from pathlib import Path
import json
import re
import smtplib
from typing import Any, Protocol
from urllib import request as urlrequest

from .contracts import BundledResponse, ChatRequest
from .llm import LLMClient, LLMTask


_CHANNELS = ("gmail", "zalo", "telegram")
_EMAIL = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+(?![\w.+-])")
_SEND_WORDS = re.compile(r"\b(send|deliver|email|mail|message|nhắn|gửi)\b", re.I)
_DRAFT_WORDS = re.compile(r"\b(compose|draft|write|prepare|soạn)\b", re.I)


class PlatformFormatter(Protocol):
    def format(self, response: BundledResponse, request: ChatRequest) -> dict[str, Any]:
        ...


class DeliverySender(Protocol):
    def send(self, payload: dict[str, Any], platform_context: dict[str, Any]) -> dict[str, Any]:
        ...


def _clean(value: Any) -> str:
    return str(value or "").strip()


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
    timeout_seconds: float = 20.0

    def validate_credentials(self, username: str, app_password: str) -> tuple[bool, str]:
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
        app_password = _clean(platform_context.get("gmail_app_password"))
        if not username or not app_password:
            raise ValueError("Gmail username and app password are required before sending.")
        message = EmailMessage()
        message["From"] = username
        message["To"] = payload["recipient"]
        message["Subject"] = payload["subject"]
        message.set_content(payload["body"])
        for artifact in payload.get("attachments", []):
            path = Path(artifact["storage_path"])
            message.add_attachment(path.read_bytes(), maintype="application", subtype="octet-stream", filename=artifact["filename"])
        with smtplib.SMTP_SSL(self.host, self.port, timeout=self.timeout_seconds) as client:
            client.login(username, app_password)
            refused = client.send_message(message)
        if refused:
            raise RuntimeError(f"Gmail refused recipient(s): {', '.join(refused)}")
        return {"status": "sent", "provider": "gmail", "recipient": payload["recipient"]}


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

        # The platform tool has now extracted a complete message.  It must
        # always be reviewed before delivery; sender invocation belongs to a
        # subsequent, explicitly approved action.
        return self._delivery_hold(
            base,
            f"Please review the {channel.title()} message below and confirm before it is sent.",
            channel=channel,
            draft=message,
            status="pending_review",
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
            schema = {"type": "object", "required": ["recipient", "subject", "body", "mode"], "properties": {"recipient": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}, "mode": {"type": "string", "enum": ["send", "draft"]}}}
            try:
                extracted = self.llm.generate_json(
                    task=LLMTask.ACTION_PLANNING,
                    system_prompt=f"You extract a {channel} message only from explicit user-provided facts. Do not invent recipient, subject, body, or attachments. Mode is send only for an explicit request to send now; otherwise draft.",
                    user_prompt=json.dumps({"user_query": text, "bundled_response": response.final_chat_text, "available_artifacts": [{"artifact_id": a.get("artifact_id"), "filename": a.get("filename")} for a in _public_artifacts(base)]}),
                    schema=schema,
                )
            except Exception:
                extracted = {}
        recipient = _clean(extracted.get("recipient"))
        if not recipient and channel == "gmail":
            found = _EMAIL.findall(text)
            recipient = found[0] if found else ""
        # The bundled answer is the canonical body unless a platform-specific
        # extractor safely supplied a body. This makes artifact/general answers
        # usable by all delivery platforms without a second answer generator.
        body = _clean(extracted.get("body")) or response.final_chat_text
        subject = _clean(extracted.get("subject"))
        if not subject:
            subject_match = re.search(r"\bsubject\s*[:=-]\s*([^\n.;]+)", text, re.I)
            subject = subject_match.group(1).strip(" '\"") if subject_match else ""
        mode = _clean(extracted.get("mode"))
        if mode not in {"send", "draft"}:
            mode = "send" if _SEND_WORDS.search(text) and not _DRAFT_WORDS.search(text) else "draft"
        attachments = _public_artifacts(base) if re.search(r"\b(attach|attachment|file|document|đính kèm)\b", text, re.I) else []
        return {"channel": channel, "recipient": recipient, "subject": subject, "body": body, "mode": mode, "attachments": attachments}

    @staticmethod
    def _missing_fields(channel: str, message: dict[str, Any], context: dict[str, Any]) -> list[str]:
        missing = [field for field in ("recipient", "body") if not _clean(message.get(field))]
        if channel == "gmail":
            if not _EMAIL.fullmatch(_clean(message.get("recipient"))):
                missing = [item for item in missing if item != "recipient"] + ["a valid recipient email address"]
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

    @staticmethod
    def _public_message(message: dict[str, Any]) -> dict[str, Any]:
        public = {key: value for key, value in message.items() if key != "storage_path"}
        public["attachments"] = [
            {key: value for key, value in attachment.items() if key != "storage_path"}
            for attachment in message.get("attachments", [])
        ]
        return public

    def _delivery_hold(self, base: dict[str, Any], question: str, *, channel: str | None = None, draft: dict[str, Any] | None = None, status: str = "needs_input") -> dict[str, Any]:
        """Return delivery state only; PostSelectorHITL owns user-facing HITL."""
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
        """The ``none`` route carries no delivery question into the HITL gate."""
        base.update({
            "text": response.final_chat_text,
            "delivery": {"channel": "none", "status": "not_requested"},
        })
        return base


@dataclass
class PostSelectorHITL:
    """Choose one user-facing HITL scope after platform selection.

    Optional supporting questions never interrupt the user.  The only normal
    questions admitted are action-blocking or safety-critical ones.  Such a
    question has priority over a delivery prompt because a routing choice must
    never hide a required clarification.  Otherwise a selected platform may
    ask only about that draft's state (recipient, credentials, or review).
    """

    def apply(self, response: BundledResponse, platform_payload: dict[str, Any]) -> dict[str, Any]:
        payload = dict(platform_payload)
        delivery = dict(payload.get("delivery") or {})
        channel = _clean(delivery.get("channel")).casefold() or "none"
        critical = self._critical_normal_question(response)

        # A required clarification is the one exception to delivery-first
        # routing: it blocks the underlying task, so no delivery review can be
        # meaningful until the user resolves it.
        if critical:
            question, source = critical
            already_in_answer = self._contains_question(response.final_chat_text, question)
            if channel in _CHANNELS:
                delivery.pop("question", None)
                delivery["deferred_by"] = "critical_normal_hitl"
                payload["delivery"] = delivery
            else:
                payload["delivery"] = {"channel": "none", "status": "not_requested"}
            payload.update({
                "text": response.final_chat_text if already_in_answer else question,
                "hitl": {
                    "required": True,
                    "scope": "general",
                    "stage": "after_platform_selector",
                    "question": question,
                    "source": source,
                    "append_to_answer": not already_in_answer,
                    "reason": "blocking_or_safety_critical_question",
                },
            })
            return payload

        if channel in _CHANNELS:
            question = _clean(delivery.pop("question", ""))
            if not question:
                # A selected route must never silently send or appear complete.
                question = f"Please review the {channel.title()} delivery details before sending."
            payload["delivery"] = delivery
            payload.update({
                "text": question,
                "hitl": {
                    "required": True,
                    "scope": "delivery",
                    "stage": "after_platform_selector",
                    "question": question,
                },
            })
            return payload

        # The general HITL component has still evaluated this turn, but no
        # optional question is surfaced.  This avoids low-value interruptions.
        hitl: dict[str, Any] = {
            "required": False,
            "scope": "general",
            "stage": "after_platform_selector",
            "reason": "no_blocking_or_safety_critical_question",
        }
        payload.update({
            "delivery": {"channel": "none", "status": "not_requested"},
            "text": response.final_chat_text,
            "hitl": hitl,
        })
        return payload

    @staticmethod
    def _critical_normal_question(response: BundledResponse) -> tuple[str, str] | None:
        """Return one explicitly mandatory normal question, if any.

        Normal supporting questions use ``optional_context`` by default and
        are intentionally suppressed.  A caller can opt in only with a
        ``must_ask`` flag or an action-blocking/safety-critical purpose.
        """
        clarification = response.last_qa_state.clarification_question
        clarification_text = PostSelectorHITL._candidate_text(clarification)
        if clarification_text:
            return clarification_text, "clarification"

        candidates = [
            *list(response.last_qa_state.supporting_questions or []),
            response.last_qa_state.reminder_supporting_question,
        ]
        ranked: list[tuple[float, str]] = []
        for candidate in candidates:
            text, _purpose, mandatory, raw_confidence = PostSelectorHITL._candidate_details(candidate)
            if not text or not mandatory:
                continue
            try:
                confidence = float(raw_confidence)
            except (TypeError, ValueError):
                confidence = 0.0
            ranked.append((confidence, text))
        if not ranked:
            return None
        return max(ranked, key=lambda item: item[0])[1], "mandatory_supporting_question"

    @staticmethod
    def _candidate_text(candidate: Any) -> str:
        return PostSelectorHITL._candidate_details(candidate)[0]

    @staticmethod
    def _candidate_details(candidate: Any) -> tuple[str, str, bool, Any]:
        critical_purposes = {"action_blocking", "required_clarification", "safety_critical"}
        if isinstance(candidate, dict):
            text = _clean(candidate.get("text") or candidate.get("question_text"))
            purpose = _clean(candidate.get("purpose")).casefold()
            mandatory = bool(candidate.get("must_ask") or candidate.get("required")) or purpose in critical_purposes
            return text, purpose, mandatory, candidate.get("confidence", 0.0)
        text = _clean(getattr(candidate, "text", candidate if isinstance(candidate, str) else ""))
        purpose = _clean(getattr(candidate, "purpose", "")).casefold()
        mandatory = bool(getattr(candidate, "must_ask", False) or getattr(candidate, "required", False)) or purpose in critical_purposes
        return text, purpose, mandatory, getattr(candidate, "confidence", 0.0)

    @staticmethod
    def _contains_question(answer: str, question: str) -> bool:
        normalize = lambda value: " ".join(str(value).casefold().split())
        return normalize(question) in normalize(answer)


class PlainTextFormatter:
    def format(self, response: BundledResponse, request: ChatRequest) -> dict[str, Any]:
        return {"text": response.final_chat_text}
