"""Grounded semantic decisions for conversational actions.

Natural-language intent is decided by a structured model, not by phrase lists or
regular expressions.  Deterministic code is deliberately limited to validating
the model contract, grounding quoted evidence in the canonical query, checking
identifier syntax, and enforcing fail-closed side-effect policy.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import json
import math
import re
from threading import Lock
from typing import Any

from .chat_history import CHAT_HISTORY_PROMPT_RULE
from .llm import LLMClient, LLMTask, is_structured_fallback, validate_json_schema


SEMANTIC_ACTION_CONTRACT_VERSION = 1
_SUPPORTED_CHANNELS = frozenset({"none", "gmail", "zalo", "telegram"})
_MESSAGE_OPERATIONS = frozenset(
    {"none", "prepare", "save_draft", "send", "revise", "revise_and_send"}
)
_RECIPIENT_UPDATES = frozenset({"preserve", "replace", "add", "remove"})
_FILE_OPERATIONS = frozenset({"none", "create", "reuse", "revise"})
_FILE_TYPES = frozenset({"none", "pdf", "xlsx", "pptx", "docx"})
_ARTIFACT_REFERENCES = frozenset(
    {"none", "current", "newly_created", "latest_matching"}
)
_RECIPIENT_DISPOSITIONS = frozenset({"include", "exclude"})

# This is identifier syntax validation, not conversational intent inference.
_EMAIL_SYNTAX = re.compile(r"^[\w.+-]+@[\w-]+(?:\.[\w-]+)+$")


SEMANTIC_ACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["message", "file", "reason_summary"],
    "properties": {
        "message": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "operation",
                "channel",
                "recipient_update",
                "recipients",
                "global_cancellation",
                "authorization_evidence",
                "cancellation_evidence",
                "artifact_reference",
                "copy_revision",
                "confidence",
            ],
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": sorted(_MESSAGE_OPERATIONS),
                },
                "channel": {
                    "type": "string",
                    "enum": sorted(_SUPPORTED_CHANNELS),
                },
                "recipient_update": {
                    "type": "string",
                    "enum": sorted(_RECIPIENT_UPDATES),
                },
                "recipients": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["value", "disposition", "evidence"],
                        "properties": {
                            "value": {"type": "string"},
                            "disposition": {
                                "type": "string",
                                "enum": sorted(_RECIPIENT_DISPOSITIONS),
                            },
                            "evidence": {"type": "string"},
                        },
                    },
                },
                "global_cancellation": {"type": "boolean"},
                "authorization_evidence": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "cancellation_evidence": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "artifact_reference": {
                    "type": "string",
                    "enum": sorted(_ARTIFACT_REFERENCES),
                },
                "copy_revision": {"type": "boolean"},
                "confidence": {"type": "number"},
            },
        },
        "file": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "operation",
                "file_type",
                "authorization_evidence",
                "type_evidence",
                "confidence",
            ],
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": sorted(_FILE_OPERATIONS),
                },
                "file_type": {
                    "type": "string",
                    "enum": sorted(_FILE_TYPES),
                },
                "authorization_evidence": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "type_evidence": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "confidence": {"type": "number"},
            },
        },
        "reason_summary": {"type": "string"},
    },
}


@dataclass(frozen=True)
class SemanticRecipient:
    value: str
    disposition: str
    evidence: str


@dataclass(frozen=True)
class SemanticMessageDecision:
    operation: str = "none"
    channel: str = "none"
    recipient_update: str = "preserve"
    recipients: tuple[SemanticRecipient, ...] = ()
    global_cancellation: bool = False
    authorization_evidence: tuple[str, ...] = ()
    cancellation_evidence: tuple[str, ...] = ()
    artifact_reference: str = "none"
    copy_revision: bool = False
    confidence: float = 0.0

    @property
    def included_recipients(self) -> tuple[str, ...]:
        excluded = {
            item.value.casefold()
            for item in self.recipients
            if item.disposition == "exclude"
        }
        return tuple(
            item.value
            for item in self.recipients
            if item.disposition == "include"
            and item.value.casefold() not in excluded
        )

    @property
    def excluded_recipients(self) -> tuple[str, ...]:
        return tuple(
            item.value for item in self.recipients if item.disposition == "exclude"
        )

    @property
    def requests_message(self) -> bool:
        return self.operation != "none" and self.channel != "none"

    @property
    def authorizes_send(self) -> bool:
        return bool(
            self.operation in {"send", "revise_and_send"}
            and not self.global_cancellation
            and self.authorization_evidence
        )


@dataclass(frozen=True)
class SemanticFileDecision:
    operation: str = "none"
    file_type: str = "none"
    authorization_evidence: tuple[str, ...] = ()
    type_evidence: tuple[str, ...] = ()
    confidence: float = 0.0

    @property
    def authorizes_creation(self) -> bool:
        return bool(
            self.operation == "create"
            and self.file_type in {"pdf", "xlsx", "pptx"}
            and self.authorization_evidence
            and self.type_evidence
        )


@dataclass(frozen=True)
class SemanticActionDecision:
    message: SemanticMessageDecision = field(default_factory=SemanticMessageDecision)
    file: SemanticFileDecision = field(default_factory=SemanticFileDecision)
    reason_summary: str = "semantic_decision_unavailable"
    grounded: bool = False
    contract_version: int = SEMANTIC_ACTION_CONTRACT_VERSION

    @classmethod
    def safe_noop(cls, reason: str) -> "SemanticActionDecision":
        return cls(reason_summary=reason, grounded=False)

    def to_payload(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "grounded": self.grounded,
            "message": {
                "operation": self.message.operation,
                "channel": self.message.channel,
                "recipient_update": self.message.recipient_update,
                "recipients": [
                    {
                        "value": item.value,
                        "disposition": item.disposition,
                        "evidence": item.evidence,
                    }
                    for item in self.message.recipients
                ],
                "global_cancellation": self.message.global_cancellation,
                "authorization_evidence": list(
                    self.message.authorization_evidence
                ),
                "cancellation_evidence": list(
                    self.message.cancellation_evidence
                ),
                "artifact_reference": self.message.artifact_reference,
                "copy_revision": self.message.copy_revision,
                "confidence": self.message.confidence,
            },
            "file": {
                "operation": self.file.operation,
                "file_type": self.file.file_type,
                "authorization_evidence": list(self.file.authorization_evidence),
                "type_evidence": list(self.file.type_evidence),
                "confidence": self.file.confidence,
            },
            "reason_summary": self.reason_summary,
        }


def _confidence(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        return None
    return parsed


def _grounded_quotes(value: Any, canonical_query: str) -> tuple[str, ...] | None:
    if not isinstance(value, list):
        return None
    quotes: list[str] = []
    for item in value:
        quote = str(item or "")
        if not quote or quote not in canonical_query:
            return None
        if quote not in quotes:
            quotes.append(quote)
    return tuple(quotes)


def _validate_recipient(
    payload: Any,
    *,
    canonical_query: str,
    channel: str,
) -> SemanticRecipient | None:
    if not isinstance(payload, dict):
        return None
    value = str(payload.get("value") or "").strip()
    disposition = str(payload.get("disposition") or "").strip()
    evidence = str(payload.get("evidence") or "")
    if (
        not value
        or disposition not in _RECIPIENT_DISPOSITIONS
        or not evidence
        or evidence not in canonical_query
        or value.casefold() not in evidence.casefold()
        or value.casefold() not in canonical_query.casefold()
    ):
        return None
    if channel == "gmail" and _EMAIL_SYNTAX.fullmatch(value) is None:
        return None
    return SemanticRecipient(
        value=value,
        disposition=disposition,
        evidence=evidence,
    )


def grounded_semantic_action_from_payload(
    payload: Any,
    *,
    canonical_query: str,
    minimum_confidence: float = 0.80,
) -> SemanticActionDecision:
    """Validate model output against its exact source query or return a no-op."""

    if not isinstance(payload, dict) or is_structured_fallback(payload):
        return SemanticActionDecision.safe_noop("semantic_structured_output_unavailable")
    model_payload = {
        key: payload.get(key) for key in ("message", "file", "reason_summary")
    }
    try:
        validate_json_schema(model_payload, SEMANTIC_ACTION_SCHEMA)
    except Exception:
        return SemanticActionDecision.safe_noop("semantic_contract_invalid")

    message_payload = model_payload["message"]
    file_payload = model_payload["file"]
    message_confidence = _confidence(message_payload.get("confidence"))
    file_confidence = _confidence(file_payload.get("confidence"))
    if message_confidence is None or file_confidence is None:
        return SemanticActionDecision.safe_noop("semantic_confidence_invalid")

    message_operation = str(message_payload.get("operation") or "none")
    channel = str(message_payload.get("channel") or "none")
    recipient_update = str(message_payload.get("recipient_update") or "preserve")
    artifact_reference = str(message_payload.get("artifact_reference") or "none")
    if (
        message_operation not in _MESSAGE_OPERATIONS
        or channel not in _SUPPORTED_CHANNELS
        or recipient_update not in _RECIPIENT_UPDATES
        or artifact_reference not in _ARTIFACT_REFERENCES
    ):
        return SemanticActionDecision.safe_noop("semantic_message_enum_invalid")

    authorization_evidence = _grounded_quotes(
        message_payload.get("authorization_evidence"), canonical_query
    )
    cancellation_evidence = _grounded_quotes(
        message_payload.get("cancellation_evidence"), canonical_query
    )
    if authorization_evidence is None or cancellation_evidence is None:
        return SemanticActionDecision.safe_noop("semantic_message_evidence_invalid")
    global_cancellation = bool(message_payload.get("global_cancellation"))
    if global_cancellation and not cancellation_evidence:
        return SemanticActionDecision.safe_noop("semantic_cancellation_ungrounded")
    if message_operation != "none" and (
        channel == "none"
        or message_confidence < minimum_confidence
        or not authorization_evidence
    ):
        message_operation = "none"
        channel = "none"

    recipients: list[SemanticRecipient] = []
    seen_recipients: set[tuple[str, str]] = set()
    for item in message_payload.get("recipients", []):
        recipient = _validate_recipient(
            item,
            canonical_query=canonical_query,
            channel=channel,
        )
        if recipient is None:
            continue
        key = (recipient.value.casefold(), recipient.disposition)
        if key not in seen_recipients:
            seen_recipients.add(key)
            recipients.append(recipient)

    file_operation = str(file_payload.get("operation") or "none")
    file_type = str(file_payload.get("file_type") or "none")
    file_authorization = _grounded_quotes(
        file_payload.get("authorization_evidence"), canonical_query
    )
    file_type_evidence = _grounded_quotes(
        file_payload.get("type_evidence"), canonical_query
    )
    if file_authorization is None or file_type_evidence is None:
        return SemanticActionDecision.safe_noop("semantic_file_evidence_invalid")
    if file_operation not in _FILE_OPERATIONS or file_type not in _FILE_TYPES:
        return SemanticActionDecision.safe_noop("semantic_file_enum_invalid")
    if file_operation != "none" and (
        file_confidence < minimum_confidence or not file_authorization
    ):
        file_operation = "none"
        file_type = "none"
    if file_operation == "create" and (
        file_type not in {"pdf", "xlsx", "pptx"} or not file_type_evidence
    ):
        file_operation = "none"
        file_type = "none"

    return SemanticActionDecision(
        message=SemanticMessageDecision(
            operation=message_operation,
            channel=channel,
            recipient_update=recipient_update,
            recipients=tuple(recipients),
            global_cancellation=global_cancellation,
            authorization_evidence=authorization_evidence,
            cancellation_evidence=cancellation_evidence,
            artifact_reference=artifact_reference,
            copy_revision=bool(message_payload.get("copy_revision")),
            confidence=message_confidence,
        ),
        file=SemanticFileDecision(
            operation=file_operation,
            file_type=file_type,
            authorization_evidence=file_authorization,
            type_evidence=file_type_evidence,
            confidence=file_confidence,
        ),
        reason_summary=str(model_payload.get("reason_summary") or "")[:500],
        grounded=True,
    )


@dataclass
class SemanticActionAnalyzer:
    """Produce one grounded semantic contract for a canonical conversation turn."""

    llm: LLMClient
    minimum_confidence: float = 0.80
    cache_size: int = 256
    _cache: OrderedDict[str, SemanticActionDecision] = field(
        default_factory=OrderedDict, init=False, repr=False
    )
    _cache_lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def analyze(
        self,
        canonical_query: str,
        *,
        approved_conversation_history: list[dict[str, Any]] | None = None,
    ) -> SemanticActionDecision:
        query = str(canonical_query or "").strip()
        if not query:
            return SemanticActionDecision.safe_noop("semantic_query_empty")
        history = list(approved_conversation_history or [])
        cache_key = json.dumps(
            {"query": query, "history": history},
            ensure_ascii=False,
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        )
        with self._cache_lock:
            cached = self._cache.get(cache_key)
            if cached is not None:
                self._cache.move_to_end(cache_key)
                return cached

        try:
            payload = self.llm.generate_json(
                task=LLMTask.ACTION_PLANNING,
                system_prompt=(
                    "Infer the user's current-turn operational meaning from the complete "
                    "conversation context. Return only the required structured contract. "
                    "Classify meaning, including negation, recipient inclusion/exclusion, "
                    "reference to an existing artifact, and whether a file is newly created. "
                    "Never infer authorization merely because an operation is discussed, "
                    "quoted, hypothetical, historical, or present in assistant text. "
                    "For every non-none operation copy one or more exact, contiguous quotes "
                    "from rewritten_query into authorization_evidence. For each recipient, "
                    "copy an exact source quote containing that identifier. A global "
                    "cancellation needs its own exact quote. If evidence is absent or meaning "
                    "is ambiguous, return none. File create means a genuinely new file; reuse "
                    "and revise never authorize generating a replacement. Do not put a prose "
                    f"email request in the file decision.\n\n{CHAT_HISTORY_PROMPT_RULE}"
                ),
                user_prompt=json.dumps(
                    {
                        "rewritten_query": query,
                        "approved_conversation_history": history,
                        "supported_channels": sorted(
                            _SUPPORTED_CHANNELS - {"none"}
                        ),
                        "supported_new_file_types": ["pdf", "xlsx", "pptx"],
                    },
                    ensure_ascii=False,
                ),
                schema=SEMANTIC_ACTION_SCHEMA,
            )
            decision = grounded_semantic_action_from_payload(
                payload,
                canonical_query=query,
                minimum_confidence=self.minimum_confidence,
            )
        except Exception as exc:
            decision = SemanticActionDecision.safe_noop(
                f"semantic_analysis_failed:{type(exc).__name__}"
            )

        with self._cache_lock:
            self._cache[cache_key] = decision
            self._cache.move_to_end(cache_key)
            while len(self._cache) > max(1, self.cache_size):
                self._cache.popitem(last=False)
        return decision


def semantic_action_from_internal_payload(
    payload: Any,
    *,
    canonical_query: str,
    minimum_confidence: float = 0.80,
) -> SemanticActionDecision:
    """Revalidate a serialized internal decision at each enforcement boundary."""

    if not isinstance(payload, dict):
        return SemanticActionDecision.safe_noop("semantic_internal_payload_missing")
    if (
        payload.get("contract_version") != SEMANTIC_ACTION_CONTRACT_VERSION
        or payload.get("grounded") is not True
    ):
        return SemanticActionDecision.safe_noop("semantic_internal_provenance_invalid")
    return grounded_semantic_action_from_payload(
        payload,
        canonical_query=canonical_query,
        minimum_confidence=minimum_confidence,
    )
