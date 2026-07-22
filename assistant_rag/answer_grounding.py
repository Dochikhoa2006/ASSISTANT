"""Evidence-bound answer generation for SQL-authoritative personal knowledge."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite
import unicodedata
from typing import Any, Iterable

from .llm import (
    LLMClient,
    LLMTask,
    StructuredOutputInvariantError,
    is_structured_fallback,
)
from .prompts import PromptContext, PromptRegistry


GROUNDED_ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "answer_text": {"type": "string"},
        "evidence_references": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "candidate_key": {"type": "string"},
                    "verbatim_support": {"type": "string"},
                },
                "required": ["candidate_key", "verbatim_support"],
            },
        },
    },
    "required": ["answer_text", "evidence_references"],
}


@dataclass(frozen=True)
class AnswerGenerationOutcome:
    text: str
    model_succeeded: bool
    grounded_candidate_keys: tuple[str, ...] = ()
    fallback_used: bool = False


def _canonical_text(value: Any) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(normalized.split())


def _record_map(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    approved: dict[str, dict[str, Any]] = {}
    for raw in records:
        record = dict(raw)
        candidate_key = str(record.get("candidate_key") or "").strip()
        text = str(record.get("text") or "").strip()
        if candidate_key and text and candidate_key not in approved:
            approved[candidate_key] = record
    return approved


def safe_evidence_answer(
    records: Iterable[dict[str, Any]],
    legacy_evidence: Iterable[str] = (),
) -> str:
    """Return the strongest single approved fact when generation is untrusted."""

    text, _keys = _safe_evidence_selection(records, legacy_evidence)
    return text


def _safe_evidence_selection(
    records: Iterable[dict[str, Any]],
    legacy_evidence: Iterable[str] = (),
) -> tuple[str, tuple[str, ...]]:
    """Select one SQL-approved record without exposing tangential candidates."""

    ranked: list[tuple[float, int, str, str]] = []
    for position, raw in enumerate(records):
        candidate_key = str(raw.get("candidate_key") or "").strip()
        value = str(raw.get("text") or "").strip()
        if not value:
            continue
        try:
            score = float(raw.get("rerank_score", float("-inf")))
        except (TypeError, ValueError):
            score = float("-inf")
        if not isfinite(score):
            score = float("-inf")
        ranked.append((score, -position, candidate_key, value))
    if ranked:
        _score, _position, candidate_key, value = max(ranked)
        return value, ((candidate_key,) if candidate_key else ())

    for raw in legacy_evidence:
        value = str(raw).strip()
        if value:
            return value, ()
    return "", ()


def is_grounded_answer_text(
    answer_text: str,
    records: Iterable[dict[str, Any]],
) -> bool:
    """Accept only whole approved SQL records without decontextualization."""

    answer = _canonical_text(answer_text)
    if not answer:
        return False
    stored = [
        _canonical_text(record.get("text"))
        for record in records
        if _canonical_text(record.get("text"))
    ]
    if any(answer == value for value in stored):
        return True
    # Multiple facts may be returned without invented connective prose, but
    # every output unit must still be an exact approved record.
    return any(
        answer == _canonical_text("\n".join(sequence))
        for sequence in (
            [value for value in stored],
            [value for value in reversed(stored)],
        )
        if sequence
    )


def _validate_grounded_payload(
    payload: dict[str, Any],
    *,
    approved_by_key: dict[str, dict[str, Any]],
) -> None:
    answer_text = str(payload.get("answer_text") or "").strip()
    if not answer_text:
        raise StructuredOutputInvariantError("answer_text must be non-empty")
    answer_canonical = _canonical_text(answer_text)
    references = payload.get("evidence_references")
    if not isinstance(references, list) or not references:
        raise StructuredOutputInvariantError(
            "At least one approved evidence reference is required."
        )

    seen: set[str] = set()
    selected_records: list[dict[str, Any]] = []
    for reference in references:
        if not isinstance(reference, dict):
            raise StructuredOutputInvariantError(
                "Every evidence reference must be an object."
            )
        candidate_key = str(reference.get("candidate_key") or "").strip()
        if candidate_key not in approved_by_key or candidate_key in seen:
            raise StructuredOutputInvariantError(
                "Evidence references must be unique supplied candidate keys."
            )
        seen.add(candidate_key)
        selected_records.append(approved_by_key[candidate_key])
        support = _canonical_text(reference.get("verbatim_support"))
        stored = _canonical_text(approved_by_key[candidate_key].get("text"))
        if not support or support not in stored or support not in answer_canonical:
            raise StructuredOutputInvariantError(
                "verbatim_support must occur in both the selected SQL evidence and answer_text."
            )

        # Reject token-sized citations that could be attached to an unrelated
        # response. This is language-agnostic shape validation, not a vocabulary
        # rule: multi-token evidence needs a multi-token supporting span, while
        # scripts without spaces need a minimally substantive character span.
        stored_units = stored.split()
        support_units = support.split()
        if len(stored_units) > 1 and len(support_units) < 2:
            raise StructuredOutputInvariantError(
                "verbatim_support is too small to ground a multi-token evidence record."
            )
        if len(stored_units) == 1 and len(stored) >= 8 and len(support) < 4:
            raise StructuredOutputInvariantError(
                "verbatim_support is too small to ground this evidence record."
            )
    if not is_grounded_answer_text(answer_text, selected_records):
        raise StructuredOutputInvariantError(
            "answer_text must preserve the whole selected SQL evidence record."
        )


def generate_answer(
    *,
    llm: LLMClient | None,
    prompt_registry: PromptRegistry,
    prompt_context: PromptContext,
    approved_knowledge_records: Iterable[dict[str, Any]] = (),
    approved_knowledge_evidence: Iterable[str] = (),
) -> AnswerGenerationOutcome:
    """Generate text, enforcing supplied SQL evidence references when present."""

    records = list(approved_knowledge_records)
    approved_by_key = _record_map(records)
    fallback, fallback_keys = _safe_evidence_selection(
        records,
        approved_knowledge_evidence,
    )
    if llm is None:
        return AnswerGenerationOutcome(
            text=fallback or prompt_registry.message("answer_model_unavailable"),
            model_succeeded=False,
            grounded_candidate_keys=fallback_keys,
            fallback_used=True,
        )

    if not approved_by_key:
        try:
            output = str(
                llm.chat(
                    task=LLMTask.ANSWER,
                    system_prompt=prompt_registry.system("answer_generation"),
                    user_prompt=prompt_registry.user(prompt_context),
                )
                or ""
            ).strip()
        except Exception:
            output = ""
        if output:
            return AnswerGenerationOutcome(text=output, model_succeeded=True)
        return AnswerGenerationOutcome(
            text=fallback or prompt_registry.message("answer_model_unavailable"),
            model_succeeded=False,
            grounded_candidate_keys=fallback_keys,
            fallback_used=True,
        )

    grounded_context = replace(
        prompt_context,
        extra={
            **dict(prompt_context.extra),
            "approved_knowledge_records": list(approved_by_key.values()),
            "grounded_answer_contract": {
                "answer_text": "direct user-facing answer",
                "evidence_references": [
                    {
                        "candidate_key": "one supplied candidate_key",
                        "verbatim_support": (
                            "a meaningful exact span present in both that evidence text "
                            "and answer_text"
                        ),
                    }
                ],
            },
        },
    )
    try:
        payload = llm.generate_json(
            task=LLMTask.ANSWER,
            system_prompt=prompt_registry.system("answer_generation"),
            user_prompt=prompt_registry.user(grounded_context),
            schema=GROUNDED_ANSWER_SCHEMA,
            invariant_validator=lambda value: _validate_grounded_payload(
                value,
                approved_by_key=approved_by_key,
            ),
        )
        if is_structured_fallback(payload):
            raise StructuredOutputInvariantError(
                "Grounded answer generation used terminal structured recovery."
            )
        _validate_grounded_payload(payload, approved_by_key=approved_by_key)
        keys = tuple(
            str(reference["candidate_key"])
            for reference in payload["evidence_references"]
        )
        return AnswerGenerationOutcome(
            text=str(payload["answer_text"]).strip(),
            model_succeeded=True,
            grounded_candidate_keys=keys,
        )
    except Exception:
        return AnswerGenerationOutcome(
            text=fallback or prompt_registry.message("answer_model_unavailable"),
            model_succeeded=False,
            grounded_candidate_keys=fallback_keys,
            fallback_used=True,
        )
