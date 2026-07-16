"""Conservative request-policy signals for entrypoint lifecycle controls.

These helpers exist only so API and UI entrypoints can apply mutation and
destructive-request rate limits before the semantic pipeline runs. They never
route a request, choose a branch action, or construct an action payload. The
knowledge and reminder branches obtain their action and content exclusively
from their first-phase LLM extractors.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import re
import unicodedata

from .action_keywords import (
    ADD_ACTION_KEYWORDS,
    DELETE_ACTION_KEYWORDS,
    MODIFY_ACTION_KEYWORDS,
    TURN_OFF_ACTION_KEYWORDS,
    TURN_ON_ACTION_KEYWORDS,
)
from .contracts import ChatRequest, Intent


_SUPPORTED_POLICY_INTENTS = frozenset({Intent.KNOWLEDGE_FACTS, Intent.REMINDER})
_SUPPORTED_CONFIRMATION_ACTION_NAMES = {
    Intent.KNOWLEDGE_FACTS: frozenset({"add", "delete", "modify"}),
}


@dataclass(frozen=True)
class _PolicyKeywordMatch:
    destructive: bool
    start: int
    end: int


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


def _policy_keyword_groups(
    intent: Intent,
) -> tuple[tuple[bool, tuple[str, ...]], ...]:
    common = (
        (True, DELETE_ACTION_KEYWORDS),
        (True, MODIFY_ACTION_KEYWORDS),
        (False, ADD_ACTION_KEYWORDS),
    )
    if intent is Intent.REMINDER:
        return common + (
            (False, TURN_ON_ACTION_KEYWORDS),
            (True, TURN_OFF_ACTION_KEYWORDS),
        )
    return common if intent is Intent.KNOWLEDGE_FACTS else ()


def _effective_policy_matches(
    raw_user_query: str,
    intent: Intent,
) -> tuple[_PolicyKeywordMatch, ...]:
    text = _normalize(raw_user_query)
    matches = [
        _PolicyKeywordMatch(
            destructive=destructive,
            start=match.start(),
            end=match.end(),
        )
        for destructive, keywords in _policy_keyword_groups(intent)
        for keyword in keywords
        for match in _keyword_pattern(keyword).finditer(text)
    ]
    # A longer policy phrase owns a contained phrase with the opposite risk
    # class. For example, "set to active" is non-destructive even though it
    # contains the destructive modify phrase "set to".
    return tuple(
        candidate
        for candidate in matches
        if not any(
            other.destructive != candidate.destructive
            and other.start <= candidate.start
            and other.end >= candidate.end
            and (other.end - other.start) > (candidate.end - candidate.start)
            for other in matches
        )
    )


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


def _has_bound_confirmation_policy_signal(
    request: ChatRequest,
    intent: Intent,
) -> bool:
    """Check lifecycle binding without authorizing or returning the action."""

    metadata = request.metadata or {}
    if intent not in _SUPPORTED_CONFIRMATION_ACTION_NAMES:
        return False
    actions = list(metadata.get("validated_knowledge_actions") or [])
    if (
        not request.confirmation_token
        or not metadata.get("confirmation_approved")
        or len(actions) != 1
    ):
        return False

    candidate = actions[0]
    candidate_name = (
        str(candidate.get("action") or "")
        if isinstance(candidate, dict)
        else str(
            getattr(
                getattr(candidate, "action", None),
                "value",
                getattr(candidate, "action", ""),
            )
            or ""
        )
    ).casefold()
    if candidate_name not in _SUPPORTED_CONFIRMATION_ACTION_NAMES[intent]:
        return False
    authorization = metadata.get("action_authorization") or {}
    return (
        authorization.get("intent") == intent.value
        and str(authorization.get("action") or "").casefold() == candidate_name
    )


def has_explicit_mutation_policy_signal(
    request: ChatRequest,
    intent: Intent,
) -> bool:
    """Return a rate-limit signal without selecting or authorizing an action."""

    if intent not in _SUPPORTED_POLICY_INTENTS:
        return False

    if _has_bound_confirmation_policy_signal(request, intent):
        return True

    matches = _effective_policy_matches(request.raw_query, intent)
    if not matches:
        return False

    normalized_query = _normalize(request.raw_query).strip()
    if _READ_ONLY_ACTION_OPENING.search(normalized_query):
        return False

    first_match = min(matches, key=lambda item: (item.start, item.end))
    prefix = normalized_query[: first_match.start].strip(" \t\r\n,.:;!?-")
    while True:
        cleaned = re.sub(
            r"^(?:please|kindly|just|now)(?:\s+|$)",
            "",
            prefix,
        ).strip()
        if cleaned == prefix:
            break
        prefix = cleaned
    if not prefix:
        return True
    return bool(_MUTATION_REQUEST_PREFIX.fullmatch(prefix))


def has_destructive_mutation_policy_signal(
    request: ChatRequest,
    intent: Intent,
) -> bool:
    """Return a destructive-rate-limit signal without exposing an action."""

    if not has_explicit_mutation_policy_signal(request, intent):
        return False
    return any(
        match.destructive
        for match in _effective_policy_matches(request.raw_query, intent)
    )
