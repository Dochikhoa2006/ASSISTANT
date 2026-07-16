"""Deterministic action authorization before knowledge/reminder execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import re
import unicodedata
from typing import Any, Protocol

from .action_keywords import (
    ADD_ACTION_KEYWORDS,
    DELETE_ACTION_KEYWORDS,
    MODIFY_ACTION_KEYWORDS,
    TURN_OFF_ACTION_KEYWORDS,
    TURN_ON_ACTION_KEYWORDS,
)
from .contracts import ChatRequest, Intent


_ACTION_METADATA_KEYS = {
    Intent.KNOWLEDGE_FACTS: "knowledge_actions",
    Intent.REMINDER: "reminder_actions",
}


@dataclass(frozen=True)
class ActionKeywordMatch:
    action: str
    keyword: str
    start: int
    end: int


@dataclass(frozen=True)
class ActionKeywordDecision:
    selected_action: str | None
    matched_actions: tuple[str, ...]
    matched_keywords: tuple[str, ...]
    reason_summary: str


@dataclass(frozen=True)
class ActionDetectionResult:
    intent: Intent
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)
    missing_fields: list[str] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)

    @property
    def requires_clarification(self) -> bool:
        return self.confidence < 0.5 or bool(self.missing_fields)


class ActionDetector(Protocol):
    def detect(self, request: ChatRequest, rewritten_query: str, intent: Intent) -> ActionDetectionResult:
        ...


class NoOpActionDetector:
    def detect(self, request: ChatRequest, rewritten_query: str, intent: Intent) -> ActionDetectionResult:
        return ActionDetectionResult(intent=intent, confidence=1.0, metadata={})


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold()


@lru_cache(maxsize=512)
def _keyword_pattern(keyword: str) -> re.Pattern[str]:
    normalized = " ".join(_normalize(keyword).split())
    if not normalized:
        return re.compile(r"(?!x)x")
    body = re.escape(normalized).replace(r"\ ", r"\s+")
    prefix = r"(?<![\w-])" if normalized[0].isalnum() or normalized[0] == "_" else ""
    suffix = r"(?![\w-])" if normalized[-1].isalnum() or normalized[-1] == "_" else ""
    return re.compile(f"{prefix}{body}{suffix}")


def _action_groups(intent: Intent) -> tuple[tuple[str, tuple[str, ...]], ...]:
    common = (
        ("delete", DELETE_ACTION_KEYWORDS),
        ("modify", MODIFY_ACTION_KEYWORDS),
        ("add", ADD_ACTION_KEYWORDS),
    )
    if intent is Intent.REMINDER:
        return common + (
            ("turn_on", TURN_ON_ACTION_KEYWORDS),
            ("turn_off", TURN_OFF_ACTION_KEYWORDS),
        )
    return common if intent is Intent.KNOWLEDGE_FACTS else ()


def _effective_action_matches(raw_user_query: str, intent: Intent) -> tuple[ActionKeywordMatch, ...]:
    text = _normalize(raw_user_query)
    matches = [
        ActionKeywordMatch(action=action, keyword=keyword, start=match.start(), end=match.end())
        for action, keywords in _action_groups(intent)
        for keyword in keywords
        for match in _keyword_pattern(keyword).finditer(text)
    ]
    # A longer explicit phrase owns a contained cross-category keyword. This makes
    # "cancel and delete" a delete and "set to inactive" a turn-off, while separate
    # action phrases still remain ambiguous and fail closed.
    effective = [
        candidate
        for candidate in matches
        if not any(
            other.action != candidate.action
            and other.start <= candidate.start
            and other.end >= candidate.end
            and (other.end - other.start) > (candidate.end - candidate.start)
            for other in matches
        )
    ]
    return tuple(effective)


def classify_action_request(raw_user_query: str, intent: Intent) -> ActionKeywordDecision:
    """Classify only explicit action words from the raw query."""

    matches = _effective_action_matches(raw_user_query, intent)
    matched_actions = tuple(
        action for action, _keywords in _action_groups(intent) if any(item.action == action for item in matches)
    )
    matched_keywords = tuple(
        dict.fromkeys(
            item.keyword for item in sorted(matches, key=lambda item: (item.start, item.end, item.keyword))
        )
    )
    if not matched_actions:
        return ActionKeywordDecision(
            selected_action=None,
            matched_actions=(),
            matched_keywords=(),
            reason_summary="missing_action_keyword",
        )
    if len(matched_actions) != 1:
        return ActionKeywordDecision(
            selected_action=None,
            matched_actions=matched_actions,
            matched_keywords=matched_keywords,
            reason_summary="ambiguous_action_keywords",
        )
    return ActionKeywordDecision(
        selected_action=matched_actions[0],
        matched_actions=matched_actions,
        matched_keywords=matched_keywords,
        reason_summary=f"selected_{matched_actions[0]}_action",
    )


def _clean_remainder(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip(" \t\r\n,.:;!?-")
    cleaned = re.sub(
        r"^(?:(?:please|kindly)\s+|(?:can|could|would|will)\s+you\s+|i\s+(?:want|need)\s+you\s+to\s+)+",
        "",
        cleaned,
        flags=re.I,
    )
    return cleaned.strip(" \t\r\n,.:;!?-")


def _request_remainder(raw_user_query: str, intent: Intent, action: str) -> str:
    candidates = [item for item in _effective_action_matches(raw_user_query, intent) if item.action == action]
    if not candidates:
        return ""
    chosen = min(candidates, key=lambda item: (-(item.end - item.start), item.start))
    return _clean_remainder(f"{raw_user_query[:chosen.start]} {raw_user_query[chosen.end:]}")


def _split_replacement(text: str) -> tuple[str, str | None]:
    parts = re.split(r"\s+(?:with|to)\s+", text, maxsplit=1, flags=re.I)
    if len(parts) != 2:
        return text, None
    target, replacement = (_clean_remainder(value) for value in parts)
    return target, replacement or None


def _build_action_payload(raw_user_query: str, intent: Intent, action: str) -> dict[str, Any]:
    remainder = _request_remainder(raw_user_query, intent, action)
    payload: dict[str, Any] = {"action": action, "confidence": 1.0}
    if intent is Intent.KNOWLEDGE_FACTS:
        if action == "add":
            payload["text"] = remainder
        elif action == "modify":
            target, replacement = _split_replacement(remainder)
            payload["target_description"] = target
            if replacement:
                payload["replacement_text"] = replacement
        else:
            payload["target_description"] = remainder
        return payload

    if action == "add":
        payload.update({"subject": remainder, "raw_reminder": raw_user_query})
    elif action == "modify":
        target, replacement = _split_replacement(remainder)
        payload["target_description"] = target
        if replacement:
            payload["new_subject"] = replacement
    else:
        payload["target_description"] = remainder
    return payload


def action_payload_is_authorized(
    request: ChatRequest,
    intent: Intent,
    actions: list[Any],
) -> tuple[bool, str]:
    """Validate branch payload cardinality and its raw-query authorization."""

    if len(actions) != 1:
        return False, "action_count_must_equal_one"
    action = actions[0]
    action_name = (
        str(action.get("action") or "")
        if isinstance(action, dict)
        else str(getattr(getattr(action, "action", None), "value", getattr(action, "action", "")) or "")
    ).casefold()
    supported = {name for name, _keywords in _action_groups(intent)}
    if action_name not in supported:
        return False, "unsupported_branch_action"

    # A confirmation token is verified and its prevalidated action loaded by the API
    # before branch execution. It authorizes replay of exactly that one stored action.
    if (
        request.confirmation_token
        and request.metadata.get("confirmation_approved")
        and request.metadata.get("validated_knowledge_actions" if intent is Intent.KNOWLEDGE_FACTS else "validated_reminder_actions")
    ):
        authorization = request.metadata.get("action_authorization") or {}
        if (
            authorization.get("intent") == intent.value
            and str(authorization.get("action") or "").casefold() == action_name
        ):
            return True, "authorized_confirmed_action"
        return False, "confirmed_action_authorization_mismatch"

    decision = classify_action_request(request.raw_query, intent)
    if decision.selected_action != action_name:
        return False, decision.reason_summary
    return True, decision.reason_summary


_MUTATION_REQUEST_PREFIX = re.compile(
    r"^(?:"
    r"(?:can|could|would|will)\s+you(?:\s+(?:please|kindly))?(?:\s+help\s+me(?:\s+to)?)?"
    r"|(?:i|we)\s+(?:want|need|would\s+like)(?:\s+you)?\s+to"
    r"|(?:i|we|you)\s+(?:should|must|need\s+to)"
    r"|need\s+to|let(?:'s|\s+us)|help\s+me(?:\s+to)?|make\s+sure\s+to"
    r"|i\s+authorize\s+you\s+to|go\s+ahead\s+and"
    r")$"
)
_READ_ONLY_ACTION_OPENING = re.compile(
    r"^(?:remember|remind\s+me)\s+"
    r"(?:what|who|where|why|how|which|whether|if)\b"
    r"|^update\s+me\s+(?:on|about)\b"
)


def request_has_explicit_mutation(request: ChatRequest, intent: Intent) -> bool:
    """Return whether a state branch owns this exact raw user request.

    Knowledge and reminder branches are mutation-only. Read/search/list/lookup
    requests remain general-purpose even if they discuss a previous action such
    as "what did I add?". A verified pending confirmation is the sole exception
    because its action was already authorized and stored by the lifecycle layer.
    """

    if intent not in _ACTION_METADATA_KEYS:
        return False

    metadata = request.metadata or {}
    validated_key = (
        "validated_knowledge_actions"
        if intent is Intent.KNOWLEDGE_FACTS
        else "validated_reminder_actions"
    )
    if (
        request.confirmation_token
        and metadata.get("confirmation_approved")
        and metadata.get(validated_key)
    ):
        confirmed, reason = action_payload_is_authorized(
            request,
            intent,
            list(metadata.get(validated_key) or []),
        )
        if confirmed and reason == "authorized_confirmed_action":
            return True

    matches = _effective_action_matches(request.raw_query, intent)
    if not matches:
        return False

    normalized_query = _normalize(request.raw_query).strip()
    if _READ_ONLY_ACTION_OPENING.search(normalized_query):
        return False

    first_match = min(matches, key=lambda item: (item.start, item.end))
    prefix = normalized_query[: first_match.start].strip(" \t\r\n,.:;!?-")
    while True:
        cleaned = re.sub(
            r"^(?:please|kindly|just|now)(?:\s+|$)", "", prefix
        ).strip()
        if cleaned == prefix:
            break
        prefix = cleaned
    if not prefix:
        return True
    return bool(_MUTATION_REQUEST_PREFIX.fullmatch(prefix))


def enforce_mutation_only_intent(request: ChatRequest, intent: Intent) -> Intent:
    """Move non-mutating knowledge/reminder requests to general-purpose."""

    if intent in _ACTION_METADATA_KEYS and not request_has_explicit_mutation(
        request, intent
    ):
        return Intent.GENERAL_RESPONSE
    return intent


@dataclass
class DeterministicActionDetector:
    """Authorize one branch action without invoking a model."""

    def detect(self, request: ChatRequest, rewritten_query: str, intent: Intent) -> ActionDetectionResult:
        if intent not in _ACTION_METADATA_KEYS:
            return ActionDetectionResult(intent=intent, confidence=1.0, metadata={})

        decision = classify_action_request(request.raw_query, intent)
        if not request_has_explicit_mutation(request, intent):
            return ActionDetectionResult(
                intent=intent,
                confidence=0.0,
                metadata={},
                missing_fields=["action_keyword"],
                risk_flags=[
                    "read_only_query_requires_general_response"
                    if decision.matched_actions
                    else decision.reason_summary
                ],
            )

        action_key = _ACTION_METADATA_KEYS[intent]
        supplied_actions = list(request.metadata.get(action_key) or [])
        confirmed, confirmation_reason = action_payload_is_authorized(request, intent, supplied_actions)
        if confirmed and confirmation_reason == "authorized_confirmed_action":
            return ActionDetectionResult(
                intent=intent,
                confidence=1.0,
                metadata={action_key: supplied_actions},
            )

        if decision.selected_action is None:
            missing = "single_action" if decision.matched_actions else "action_keyword"
            return ActionDetectionResult(
                intent=intent,
                confidence=0.0,
                metadata={},
                missing_fields=[missing],
                risk_flags=[decision.reason_summary],
            )

        if len(supplied_actions) > 1:
            return ActionDetectionResult(
                intent=intent,
                confidence=0.0,
                metadata={},
                missing_fields=["single_action"],
                risk_flags=["multiple_action_payloads_rejected"],
            )
        if supplied_actions:
            if not isinstance(supplied_actions[0], dict):
                return ActionDetectionResult(
                    intent=intent,
                    confidence=0.0,
                    metadata={},
                    missing_fields=["single_action"],
                    risk_flags=["invalid_action_payload_rejected"],
                )
            supplied_name = str(supplied_actions[0].get("action") or "").casefold()
            if supplied_name != decision.selected_action:
                return ActionDetectionResult(
                    intent=intent,
                    confidence=0.0,
                    metadata={},
                    missing_fields=["action_keyword"],
                    risk_flags=["action_payload_keyword_mismatch"],
                )
            action_payload = dict(supplied_actions[0])
        else:
            action_payload = _build_action_payload(request.raw_query, intent, decision.selected_action)

        action_payload["action"] = decision.selected_action
        action_payload.setdefault("confidence", 1.0)
        return ActionDetectionResult(
            intent=intent,
            confidence=1.0,
            metadata={
                action_key: [action_payload],
                "action_authorization": {
                    "intent": intent.value,
                    "action": decision.selected_action,
                    "matched_keywords": list(decision.matched_keywords),
                    "reason_summary": decision.reason_summary,
                },
            },
        )
