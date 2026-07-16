"""Reminder mutation orchestration; model 3 is reserved for MODIFY only."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
import json
from math import isfinite
import re
from typing import Any
import unicodedata
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .action_detection import ActionDetectionResult
from .chat_history import current_chat_history
from .config import AssistantConfig
from .contracts import (
    ActionValidationResult,
    ChatRequest,
    Intent,
    LLMRetrievalValidationResult,
    PipelineContext,
    ReminderAction,
    RetrievalCandidateAssessment,
    ValidatedReminderAction,
)
from .database import AssistantRepository
from .llm import LLMClient, LLMTask
from .prompts import (
    PromptContext,
    PromptRegistry,
    REMINDER_ACTION_EXTRACTION_SCHEMA,
    REMINDER_ACTION_VALIDATION_SCHEMA,
    REMINDER_CONTENT_FINALIZATION_SCHEMA,
)


REMINDER_EDITABLE_FIELDS: tuple[str, ...] = (
    "subject",
    "reminder_summary",
    "raw_reminder",
    "notification_time",
    "event_time",
    "user_timezone",
    "original_time_text",
    "recurrence_rule",
    "recurrence_timezone",
    "supporting_question",
    "supporting_response",
)
_REMINDER_MODEL_MUTATION_FIELDS = tuple(
    field
    for field in REMINDER_EDITABLE_FIELDS
    if field not in {"supporting_question", "supporting_response"}
)

_TEXT_FIELDS = {
    "subject",
    "reminder_summary",
    "raw_reminder",
    "original_time_text",
    "recurrence_rule",
}
_TIME_FIELDS = {"notification_time", "event_time"}
_CLEARABLE_FIELDS = {
    "reminder_summary",
    "raw_reminder",
    "original_time_text",
    "recurrence_rule",
    "recurrence_timezone",
}
_EXTERNAL_REMINDER_ACTIONS = {
    ReminderAction.ADD.value,
    ReminderAction.DELETE.value,
    ReminderAction.MODIFY.value,
    ReminderAction.TURN_ON.value,
    ReminderAction.TURN_OFF.value,
}
_EXTRACTED_ACTION_SOURCE = "extracted_action"
_SELECTED_CANDIDATE_SOURCE = "selected_candidate"
_REMINDER_EXTRACTION_FIELDS = {
    "action",
    "retrieval_text",
    "field_values",
    "confidence",
}


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _normalized(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split()).strip(
        " \t\r\n\"'`.,;:!?"
    )


def _is_grounded(value: str, sources: list[str]) -> bool:
    needle = _normalized(value)
    return bool(needle) and any(needle in _normalized(source) for source in sources)


def _parse_datetime(value: str) -> datetime | None:
    if not value:
        return None
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except (TypeError, ValueError):
        return None
    return parsed


def _is_normalized_utc_timestamp(value: str) -> bool:
    """Validate model-produced time without changing or repairing its value."""
    parsed = _parse_datetime(value)
    return bool(
        parsed is not None
        and parsed.tzinfo is not None
        and parsed.utcoffset() == timedelta(0)
    )


def _is_valid_iana_timezone(value: str) -> bool:
    try:
        ZoneInfo(value)
    except (ZoneInfoNotFoundError, ValueError):
        return False
    return True


def _decode_field_values(
    raw_values: Any,
) -> tuple[list[str], dict[str, str]] | None:
    """Parse the compact model-1 field list into one deterministic flat record."""
    if not isinstance(raw_values, list):
        return None
    field_order: list[str] = []
    values = {field: "" for field in REMINDER_EDITABLE_FIELDS}
    for item in raw_values:
        if not isinstance(item, dict) or set(item) != {"field", "value"}:
            return None
        field = _text(item["field"])
        value = item["value"]
        if (
            field not in _REMINDER_MODEL_MUTATION_FIELDS
            or field in field_order
            or not isinstance(value, str)
        ):
            return None
        field_order.append(field)
        values[field] = _text(value)
    return field_order, values


def _time_semantics_for(
    *,
    action_name: str,
    supplied_fields: list[str],
    values: dict[str, str],
) -> str:
    relevant = (
        set(supplied_fields) & _TIME_FIELDS
        if action_name == ReminderAction.MODIFY.value
        else {field for field in _TIME_FIELDS if values[field]}
    )
    if relevant == _TIME_FIELDS:
        return "both"
    if relevant:
        return next(iter(relevant))
    return "unchanged"


@dataclass(frozen=True)
class ReminderMutationCandidate:
    candidate_key: str
    status: str
    version: int
    deterministic_score: float
    fields: dict[str, str]
    next_fire_time: str = ""
    parent_recurring_reminder_id: str = ""
    timing_plan_status: str = "pending"

    @property
    def reminder_time(self) -> datetime | None:
        return _parse_datetime(self.fields["notification_time"])


@dataclass(frozen=True)
class ReminderFinalizationResult:
    final_reminder: dict[str, str]
    confidence: float
    reason_summary: str


class LLMReminderActionDetector:
    """Extract exactly one reminder action for the already selected branch."""

    def __init__(
        self,
        *,
        llm: LLMClient,
        prompts: PromptRegistry,
        min_confidence: float = 0.76,
        default_timezone: str = "UTC",
    ) -> None:
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("Reminder action extraction confidence must be in [0, 1]")
        self.llm = llm
        self.prompts = prompts
        self.min_confidence = min_confidence
        self.default_timezone = default_timezone

    def detect(
        self,
        request: ChatRequest,
        rewritten_query: str,
        intent: Intent,
    ) -> ActionDetectionResult:
        runtime_now_utc = datetime.now(timezone.utc).isoformat()
        extraction_metadata = {
            key: value
            for key, value in request.metadata.items()
            if key
            not in {
                "intent",
                "reminder_actions",
                "reminder_action_extraction_response",
                "validated_reminder_actions",
                "action_authorization",
                "confirmation_approved",
            }
        }
        chat_history = current_chat_history()
        try:
            payload = self.llm.generate_json(
                task=LLMTask.REMINDER_ACTION_EXTRACTION,
                system_prompt=self.prompts.system("reminder_action_extraction"),
                user_prompt=self.prompts.user(
                    PromptContext(
                        stage="reminder_action_extraction",
                        user_id=request.user_id,
                        rewritten_query=rewritten_query,
                        intent=intent.value,
                        metadata=extraction_metadata,
                        platform_context=request.platform_context,
                        chat_history=chat_history,
                        extra={
                            "allowed_actions": [
                                "add",
                                "delete",
                                "modify",
                                "turn_on",
                                "turn_off",
                            ],
                            "cardinality": "exactly_one",
                            "editable_fields": list(_REMINDER_MODEL_MUTATION_FIELDS),
                            "action_content_contract": {
                                "add": {
                                    "target": "retrieval_text",
                                    "field_values": "all explicitly supplied new reminder fields",
                                },
                                "delete": {
                                    "target": "retrieval_text",
                                    "field_values": "empty",
                                },
                                "modify": {
                                    "target": "retrieval_text",
                                    "field_values": "only replacements or explicit clearings on the same reminder",
                                },
                                "turn_on": {
                                    "target": "retrieval_text",
                                    "field_values": "empty",
                                },
                                "turn_off": {
                                    "target": "retrieval_text",
                                    "field_values": "empty",
                                },
                            },
                            "field_semantics": {
                                "subject": "title or name",
                                "reminder_summary": "concise summary",
                                "raw_reminder": "body, content, instructions, or note",
                                "notification_time": "normalized UTC ISO-8601 instant when to notify the user",
                                "event_time": "normalized UTC ISO-8601 instant when the event or deadline occurs",
                                "user_timezone": "trusted IANA time zone used to interpret and normalize time",
                                "original_time_text": "verbatim current-turn time expression",
                                "recurrence_rule": "repeat schedule",
                                "recurrence_timezone": "time zone governing recurrence",
                            },
                            "context_policy": {
                                "current_turn_defines_action_and_new_values": True,
                                "history_resolves_existing_target_only": True,
                                "platform_context_resolves_trusted_timezone": True,
                                "runtime_now_resolves_relative_time": True,
                                "metadata_cannot_supply_action_payload": True,
                            },
                            "runtime_now_utc": runtime_now_utc,
                            "default_timezone": self.default_timezone,
                            "time_normalization_contract": {
                                "owner": "reminder_action_extraction_model",
                                "output_timezone": "UTC",
                                "output_format": "ISO-8601 with explicit +00:00 offset",
                                "required_companion_fields": [
                                    "user_timezone",
                                    "original_time_text",
                                ],
                                "omit_unresolvable_time_instead_of_guessing": True,
                            },
                        },
                    )
                ),
                schema=REMINDER_ACTION_EXTRACTION_SCHEMA,
            )
        except Exception:
            return self._failed("reminder_action_extraction_failed", ["action"])

        if not isinstance(payload, dict) or set(payload) != _REMINDER_EXTRACTION_FIELDS:
            return self._failed("invalid_reminder_action_extraction", ["action"])
        if not isinstance(payload["action"], str) or not isinstance(
            payload["retrieval_text"], str
        ):
            return self._failed("invalid_reminder_action_extraction", ["action"])
        if (
            isinstance(payload["confidence"], bool)
            or not isinstance(payload["confidence"], (int, float))
        ):
            return self._failed("invalid_reminder_action_extraction", ["action"])

        try:
            extracted_action_name = payload["action"].strip().casefold()
            retrieval_text = payload["retrieval_text"].strip()
            confidence = float(payload["confidence"])
        except (KeyError, TypeError, ValueError):
            return self._failed("invalid_reminder_action_extraction", ["action"])
        decoded = _decode_field_values(payload["field_values"])
        if decoded is None:
            return self._failed("invalid_reminder_action_extraction", ["action"])
        supplied_fields, values = decoded

        if (
            extracted_action_name not in _EXTERNAL_REMINDER_ACTIONS
            or not isfinite(confidence)
            or not 0.0 <= confidence <= 1.0
        ):
            return self._failed("invalid_reminder_action_extraction", ["action"])

        action_name = extracted_action_name
        changed_fields = (
            list(supplied_fields)
            if action_name == ReminderAction.MODIFY.value
            else []
        )
        time_semantics = _time_semantics_for(
            action_name=action_name,
            supplied_fields=supplied_fields,
            values=values,
        )
        toggle_direction = action_name if action_name in {
            ReminderAction.TURN_ON.value,
            ReminderAction.TURN_OFF.value,
        } else ""

        current_sources = [rewritten_query]
        history_source = json.dumps(chat_history, ensure_ascii=False, default=str)
        retrieval_sources = (
            list(current_sources)
            if action_name == ReminderAction.ADD.value
            else current_sources + [history_source]
        )
        if retrieval_text and not _is_grounded(retrieval_text, retrieval_sources):
            return self._failed(
                "reminder_target_not_grounded_in_request_context",
                ["target_description"],
            )

        # Guard 1 owns provenance because model 2 is intentionally isolated
        # from the request and history. It does not own action completeness,
        # calendar validity, or action/field compatibility; model 2 receives
        # this best-effort state and judges those interactions.
        grounded_fields = {
            field for field in supplied_fields if values[field]
        }
        content_sources = current_sources
        ungrounded = [
            field
            for field in grounded_fields & _TEXT_FIELDS
            if not _is_grounded(values[field], content_sources)
        ]
        if ungrounded:
            return self._failed(
                "reminder_content_not_grounded_in_current_turn",
                ungrounded,
            )

        allowed_timezones = {
            self.default_timezone,
            _text(request.platform_context.get("timezone")),
        }
        for field in ("user_timezone", "recurrence_timezone"):
            if values[field] and not (
                values[field] in allowed_timezones
                or _is_grounded(values[field], content_sources)
            ):
                return self._failed(
                    "reminder_timezone_not_grounded",
                    ["timezone"],
                )

        action_payload: dict[str, Any] = {
            "action": action_name,
            "extracted_action": extracted_action_name,
            "toggle_direction": toggle_direction,
            "retrieval_text": retrieval_text,
            "changed_fields": changed_fields,
            "time_semantics": time_semantics,
            "runtime_now_utc": runtime_now_utc,
            "confidence": confidence,
            **values,
        }
        return ActionDetectionResult(
            intent=Intent.REMINDER,
            confidence=confidence,
            metadata={
                "reminder_actions": [action_payload],
                "reminder_action_extraction_response": {
                    "action": extracted_action_name,
                    "retrieval_text": retrieval_text,
                    "field_values": [
                        {"field": field, "value": values[field]}
                        for field in supplied_fields
                    ],
                    "confidence": confidence,
                },
                "action_authorization": {
                    "intent": Intent.REMINDER.value,
                    "action": action_name,
                    "extracted_action": extracted_action_name,
                    "toggle_direction": toggle_direction,
                    "source": "llm_action_extraction",
                    "reason_summary": "selected_by_reminder_action_extraction_llm",
                },
            },
        )

    @staticmethod
    def _validate_shape(
        *,
        action_name: str,
        retrieval_text: str,
        changed_fields: list[str],
        values: dict[str, str],
        time_semantics: str,
        allow_incomplete: bool = False,
    ) -> tuple[str, list[str]] | None:
        if not retrieval_text and not allow_incomplete:
            return "missing_reminder_retrieval_text", ["target_description"]
        nonempty = {field for field, value in values.items() if value}
        if action_name == ReminderAction.ADD.value:
            missing = []
            if not values["subject"] and not allow_incomplete:
                missing.append("subject")
            if not values["raw_reminder"] and not allow_incomplete:
                missing.append("reminder_content")
            if (
                not (values["notification_time"] or values["event_time"])
                and not allow_incomplete
            ):
                missing.append("reminder_time")
            if changed_fields:
                missing.append("reminder_fields")
            time_fields = nonempty & _TIME_FIELDS
            expected_semantics = (
                "both"
                if time_fields == _TIME_FIELDS
                else next(iter(time_fields))
                if time_fields
                else "unchanged"
            )
            if time_semantics != expected_semantics:
                missing.append("reminder_time")
            return (
                ("invalid_add_reminder_content_contract", missing)
                if missing
                else None
            )
        if action_name in {
            ReminderAction.DELETE.value,
            ReminderAction.TURN_ON.value,
            ReminderAction.TURN_OFF.value,
        }:
            if changed_fields or nonempty or time_semantics != "unchanged":
                return (
                    "invalid_target_only_reminder_content_contract",
                    ["target_description"],
                )
            return None

        if not changed_fields and not allow_incomplete:
            return "modify_reminder_requires_changed_fields", ["reminder_update"]
        unexpected = nonempty - set(changed_fields)
        missing_values = [
            field
            for field in changed_fields
            if not values[field] and field not in _CLEARABLE_FIELDS
        ]
        if unexpected or (missing_values and not allow_incomplete):
            return (
                "invalid_modify_reminder_content_contract",
                missing_values or sorted(unexpected),
            )
        time_changed = set(changed_fields) & _TIME_FIELDS
        expected_semantics = (
            "both"
            if time_changed == _TIME_FIELDS
            else next(iter(time_changed))
            if time_changed
            else "unchanged"
        )
        if time_semantics != expected_semantics:
            return "invalid_reminder_time_semantics", ["reminder_time"]
        return None

    @staticmethod
    def _failed(reason: str, missing_fields: list[str]) -> ActionDetectionResult:
        return ActionDetectionResult(
            intent=Intent.REMINDER,
            confidence=0.0,
            missing_fields=missing_fields,
            risk_flags=[reason],
        )


class ReminderActionValidationStrategy:
    """Validate every bounded SQL reminder candidate for every reminder action."""

    def __init__(
        self,
        *,
        config: AssistantConfig,
        llm: LLMClient,
        prompts: PromptRegistry,
    ) -> None:
        if not config.retrieval_validation.reminder_llm_validation_enabled:
            raise ValueError("Reminder mutation validation must remain LLM-enabled")
        self.config = config
        self.llm = llm
        self.prompts = prompts

    def validate(
        self,
        *,
        context: PipelineContext,
        action_payload: dict[str, Any],
        candidates: list[ReminderMutationCandidate],
        first_model_response: dict[str, Any] | None = None,
    ) -> LLMRetrievalValidationResult:
        operation = _text(action_payload.get("action")).casefold()
        # ``context`` remains in the compatibility signature, but no request,
        # query, metadata, platform, or history value may enter LLM2. The
        # PromptContext stage also enforces this boundary with an exact key
        # allowlist, matching the knowledge validator's isolation pattern.
        isolated_first_response = self._validate_first_model_response(
            first_model_response,
            requested_operation=operation,
        )
        if isolated_first_response is None:
            return self._rejection(
                operation,
                "Reminder validation received an invalid first-model response.",
                "invalid_first_model_response",
            )
        decoded_first = _decode_field_values(isolated_first_response["field_values"])
        if decoded_first is None:
            return self._rejection(
                operation,
                "Reminder validation could not decode the first-model fields.",
                "invalid_first_model_response",
            )
        supplied_fields, extracted_values = decoded_first
        changed_fields = (
            supplied_fields if operation == ReminderAction.MODIFY.value else []
        )
        time_semantics = _time_semantics_for(
            action_name=operation,
            supplied_fields=supplied_fields,
            values=extracted_values,
        )
        complete_shape = self._first_model_response_ready(
            isolated_first_response,
        )
        raw_action_confidence = action_payload.get("confidence")
        state_matches_action_payload = (
            isolated_first_response["retrieval_text"]
            == _text(action_payload.get("retrieval_text"))
            and not isinstance(raw_action_confidence, bool)
            and isinstance(raw_action_confidence, (int, float))
            and isolated_first_response["confidence"]
            == float(raw_action_confidence)
            and changed_fields
            == [_text(field) for field in action_payload.get("changed_fields", [])]
            and time_semantics == _text(action_payload.get("time_semantics"))
            and all(
                extracted_values[field] == _text(action_payload.get(field))
                for field in REMINDER_EDITABLE_FIELDS
            )
        )
        extraction_ready_for_execution = (
            complete_shape and state_matches_action_payload
        )

        reminder_retrieval = [
            {
                "candidate_key": candidate.candidate_key,
                "status": candidate.status,
                "version": candidate.version,
                "deterministic_score": candidate.deterministic_score,
                "next_fire_time": candidate.next_fire_time,
                "parent_recurring_reminder_id": (
                    candidate.parent_recurring_reminder_id
                ),
                "timing_plan_status": candidate.timing_plan_status,
                **candidate.fields,
            }
            for candidate in candidates
        ]
        try:
            raw = self.llm.generate_json(
                task=LLMTask.REMINDER_ACTION_VALIDATION,
                system_prompt=self.prompts.system("reminder_action_validation"),
                user_prompt=self.prompts.user(
                    PromptContext(
                        stage="reminder_action_validation",
                        extra={
                            "first_model_response": isolated_first_response,
                            "reminder_retrieval": reminder_retrieval,
                        },
                    )
                ),
                schema=REMINDER_ACTION_VALIDATION_SCHEMA,
                model_override=(
                    self.config.retrieval_validation.reminder_llm_validation_model
                ),
            )
        except Exception:
            return self._rejection(
                operation,
                "Reminder validation model failed safely.",
                "internal_pipeline_failure",
            )
        return self._parse(
            raw=raw,
            operation=operation,
            action_payload=action_payload,
            candidates=candidates,
            extraction_ready_for_execution=extraction_ready_for_execution,
        )

    @staticmethod
    def _validate_first_model_response(
        response: dict[str, Any] | None,
        *,
        requested_operation: str,
    ) -> dict[str, Any] | None:
        if not isinstance(response, dict) or set(response) != _REMINDER_EXTRACTION_FIELDS:
            return None
        if not isinstance(response["action"], str) or not isinstance(
            response["retrieval_text"], str
        ):
            return None
        raw_confidence = response["confidence"]
        if (
            isinstance(raw_confidence, bool)
            or not isinstance(raw_confidence, (int, float))
            or not isfinite(float(raw_confidence))
            or not 0.0 <= float(raw_confidence) <= 1.0
        ):
            return None

        extracted_action = _text(response["action"]).casefold()
        retrieval_text = _text(response["retrieval_text"])
        decoded = _decode_field_values(response["field_values"])
        if decoded is None or extracted_action not in _EXTERNAL_REMINDER_ACTIONS:
            return None
        supplied_fields, values = decoded
        if extracted_action != requested_operation:
            return None
        return {
            "action": extracted_action,
            "retrieval_text": retrieval_text,
            "field_values": [
                {"field": field, "value": values[field]}
                for field in supplied_fields
            ],
            "confidence": float(raw_confidence),
        }

    @staticmethod
    def _first_model_response_ready(response: dict[str, Any]) -> bool:
        """Apply execution-only invariants after model 2 has made its decision."""
        decoded = _decode_field_values(response["field_values"])
        if decoded is None:
            return False
        supplied_fields, values = decoded
        action_name = _text(response["action"]).casefold()
        changed_fields = (
            supplied_fields
            if action_name == ReminderAction.MODIFY.value
            else []
        )
        time_semantics = _time_semantics_for(
            action_name=action_name,
            supplied_fields=supplied_fields,
            values=values,
        )
        if LLMReminderActionDetector._validate_shape(
            action_name=action_name,
            retrieval_text=_text(response["retrieval_text"]),
            changed_fields=changed_fields,
            values=values,
            time_semantics=time_semantics,
            allow_incomplete=False,
        ) is not None:
            return False
        if any(
            values[field] and not _is_normalized_utc_timestamp(values[field])
            for field in _TIME_FIELDS
        ):
            return False
        if any(
            values[field] and not _is_valid_iana_timezone(values[field])
            for field in ("user_timezone", "recurrence_timezone")
        ):
            return False
        return not (
            any(values[field] for field in _TIME_FIELDS)
            and (
                not values["original_time_text"]
                or not values["user_timezone"]
            )
        )

    def _parse(
        self,
        *,
        raw: dict[str, Any],
        operation: str,
        action_payload: dict[str, Any],
        candidates: list[ReminderMutationCandidate],
        extraction_ready_for_execution: bool,
    ) -> LLMRetrievalValidationResult:
        """Guard 2: validate model 2's response, selection, and SQL evidence.

        Exact MODIFY merging belongs to finalization and status/version writes
        belong to the repository, so this guard leaves those later invariants
        to their owning stages.
        """
        candidate_map = {
            candidate.candidate_key: candidate for candidate in candidates
        }
        expected_root_fields = {
            "validation_result",
            "selected_candidate_keys",
            "confidence",
            "clarification_question",
            "candidate_assessments",
        }
        expected_assessment_fields = {
            "candidate_key",
            "match_kind",
            "confidence",
            "evidence_field",
            "matched_text",
        }
        try:
            if not isinstance(raw, dict) or set(raw) != expected_root_fields:
                raise ValueError("unexpected reminder validation fields")
            raw_result = raw["validation_result"]
            raw_selected = raw["selected_candidate_keys"]
            raw_confidence = raw["confidence"]
            raw_question = raw["clarification_question"]
            raw_assessments = raw["candidate_assessments"]
            if (
                raw_result not in {"PASS", "FAIL"}
                or not isinstance(raw_selected, list)
                or not all(isinstance(value, str) for value in raw_selected)
                or isinstance(raw_confidence, bool)
                or not isinstance(raw_confidence, (int, float))
                or not isinstance(raw_question, str)
                or not isinstance(raw_assessments, list)
            ):
                raise ValueError("invalid reminder validation field types")
            requested_action = ReminderAction(operation)
            assessments_list: list[RetrievalCandidateAssessment] = []
            for item in raw_assessments:
                if (
                    not isinstance(item, dict)
                    or set(item) != expected_assessment_fields
                ):
                    raise ValueError("invalid reminder assessment fields")
                candidate_key = _text(item["candidate_key"])
                match_kind = item["match_kind"]
                item_confidence = item["confidence"]
                evidence_field = item["evidence_field"]
                matched_text = item["matched_text"]
                if (
                    not isinstance(item["candidate_key"], str)
                    or match_kind not in {"NONE", "TARGET", "EQUIVALENT"}
                    or isinstance(item_confidence, bool)
                    or not isinstance(item_confidence, (int, float))
                    or not isinstance(evidence_field, str)
                    or not isinstance(matched_text, str)
                    or candidate_key not in candidate_map
                ):
                    raise ValueError("invalid reminder assessment values")
                if operation == ReminderAction.ADD.value:
                    if match_kind == "TARGET":
                        raise ValueError("ADD assessments use EQUIVALENT")
                elif match_kind == "EQUIVALENT":
                    raise ValueError("target actions use TARGET")

                candidate = candidate_map[candidate_key]
                matches_target = match_kind != "NONE"
                if matches_target:
                    if (
                        evidence_field not in REMINDER_EDITABLE_FIELDS
                        or not matched_text
                        or matched_text not in candidate.fields[evidence_field]
                    ):
                        raise ValueError("ungrounded reminder match evidence")
                    matched_fields = (evidence_field,)
                else:
                    if evidence_field or matched_text:
                        raise ValueError("non-match included evidence")
                    matched_fields = ()
                assessments_list.append(
                    RetrievalCandidateAssessment(
                        candidate_key=candidate_key,
                        matches_target=matches_target,
                        action_compatible=(
                            candidate.status
                            in self._allowed_statuses(requested_action)
                        ),
                        confidence=float(item_confidence),
                        matched_fields=matched_fields,
                        matched_text=matched_text,
                        reason_summary=match_kind.casefold(),
                    )
                )
            assessments = tuple(assessments_list)
            is_pass = raw_result == "PASS"
            validation_result = (
                ActionValidationResult.EXECUTE
                if is_pass
                else ActionValidationResult.CLARIFY_MISSING_FIELDS
            )
            result = LLMRetrievalValidationResult(
                operation=operation,
                validation_result=validation_result,
                selected_candidate_keys=tuple(
                    _text(value) for value in raw_selected
                ),
                confidence=float(raw_confidence),
                ambiguous=False,
                should_execute=is_pass,
                requires_hitl=not is_pass,
                factuality_concern=False,
                hitl_reason=(
                    None if is_pass else "reminder_validation_fail"
                ),
                reason_summary=(
                    "Reminder action passed semantic validation."
                    if is_pass
                    else "Reminder validation requires user clarification."
                ),
                candidate_assessments=assessments,
                clarification_question=raw_question.strip(),
            )
        except (KeyError, TypeError, ValueError):
            return self._rejection(
                operation,
                "Reminder validation returned an invalid contract.",
                "invalid_validation_contract",
            )

        minimum = self.config.retrieval_validation.reminder_llm_validation_min_confidence
        if (
            result.operation != operation
            or not isfinite(result.confidence)
            or not 0.0 <= result.confidence <= 1.0
            or len(set(result.selected_candidate_keys))
            != len(result.selected_candidate_keys)
            or any(key not in candidate_map for key in result.selected_candidate_keys)
        ):
            return self._rejection(
                operation,
                "Reminder validation confidence or selection was unsafe.",
                "unsafe_validation_selection",
            )

        assessment_keys = [item.candidate_key for item in result.candidate_assessments]
        if len(set(assessment_keys)) != len(assessment_keys):
            return self._rejection(
                operation,
                "A reminder candidate cannot be assessed more than once.",
                "duplicate_candidate_assessments",
            )

        requested_action = ReminderAction(operation)
        for assessment in result.candidate_assessments:
            candidate = candidate_map[assessment.candidate_key]
            expected_compatible = candidate.status in self._allowed_statuses(
                requested_action
            )
            if (
                not isfinite(assessment.confidence)
                or not 0.0 <= assessment.confidence <= 1.0
                or assessment.action_compatible != expected_compatible
                or any(
                    field not in REMINDER_EDITABLE_FIELDS
                    for field in assessment.matched_fields
                )
            ):
                return self._rejection(
                    operation,
                    "Reminder candidate evidence was invalid.",
                    "invalid_candidate_evidence",
                )
            if assessment.matches_target:
                if (
                    len(assessment.matched_fields) != 1
                    or not assessment.matched_text
                    or assessment.matched_text
                    not in candidate.fields[assessment.matched_fields[0]]
                ):
                    return self._rejection(
                        operation,
                        "Reminder match evidence was not present in SQL fields.",
                        "ungrounded_candidate_match",
                    )
            elif assessment.matched_fields or assessment.matched_text:
                return self._rejection(
                    operation,
                    "A non-match cannot contain match evidence.",
                    "contradictory_candidate_evidence",
                )

        is_execute = result.validation_result is ActionValidationResult.EXECUTE
        if not is_execute:
            # FAIL is already non-executable. As in the knowledge guard,
            # partial candidate coverage and low confidence are acceptable,
            # but the model must provide exactly one usable user question.
            if (
                result.selected_candidate_keys
                or not (result.clarification_question or "").strip()
            ):
                return self._rejection(
                    operation,
                    "A failed reminder decision requires one question and no target.",
                    "invalid_failure_contract",
                )
            return result

        if result.clarification_question:
            return self._rejection(
                operation,
                "A passed reminder decision cannot include a question.",
                "invalid_pass_question",
            )
        if result.confidence < minimum:
            return self._rejection(
                operation,
                "Reminder validation confidence was below the execution threshold.",
                "unsafe_validation_confidence",
            )
        if set(assessment_keys) != set(candidate_map):
            return self._rejection(
                operation,
                "Every reminder candidate must be assessed before a semantic decision.",
                "incomplete_candidate_assessments",
            )

        if not extraction_ready_for_execution:
            return self._rejection(
                operation,
                "The extracted reminder state was not execution-ready.",
                "incomplete_first_model_response",
            )

        semantic_matches = {
            assessment.candidate_key
            for assessment in result.candidate_assessments
            if assessment.matches_target
            and assessment.confidence >= minimum
        }
        strong_matches = {
            assessment.candidate_key
            for assessment in result.candidate_assessments
            if assessment.candidate_key in semantic_matches
            and assessment.action_compatible
        }
        if operation == ReminderAction.ADD.value:
            if result.selected_candidate_keys or strong_matches:
                return self._rejection(
                    operation,
                    "Add duplicate evidence was inconsistent.",
                    "duplicate_conflict",
                )
        else:
            if (
                len(result.selected_candidate_keys) != 1
                or strong_matches != {result.selected_candidate_keys[0]}
                or semantic_matches != {result.selected_candidate_keys[0]}
            ):
                return self._rejection(
                    operation,
                    "Reminder target was not unique.",
                    "ambiguous_target",
                )
            selected_candidate = candidate_map[result.selected_candidate_keys[0]]
            if (
                requested_action is ReminderAction.MODIFY
                and not self._modify_changes_candidate(
                    action_payload,
                    selected_candidate,
                )
            ):
                return self._rejection(
                    operation,
                    "A no-op reminder modification cannot pass validation.",
                    "invalid_no_op_pass",
                )

        return result

    @staticmethod
    def _modify_changes_candidate(
        action_payload: dict[str, Any],
        candidate: ReminderMutationCandidate,
    ) -> bool:
        changed_fields = {
            _text(field) for field in action_payload.get("changed_fields") or []
        }
        original_time_is_control = bool(
            "original_time_text" in changed_fields and changed_fields & _TIME_FIELDS
        )
        for field in changed_fields:
            field_name = _text(field)
            if field_name == "original_time_text" and original_time_is_control:
                continue
            proposed = _text(action_payload.get(field_name))
            current = candidate.fields.get(field_name, "")
            if field_name in _TIME_FIELDS:
                proposed_time = _parse_datetime(proposed)
                current_time = _parse_datetime(current)
                if proposed_time and current_time and proposed_time == current_time:
                    continue
            if proposed != current:
                return True
        return False

    def _allowed_statuses(self, action: ReminderAction) -> tuple[str, ...]:
        resolver = self.config.reminder_resolver
        if action is ReminderAction.ADD:
            return ("scheduled", "notified")
        if action is ReminderAction.MODIFY:
            return resolver.allowed_reminder_modify_statuses
        if action is ReminderAction.DELETE:
            return resolver.allowed_reminder_delete_statuses
        if action is ReminderAction.TURN_ON:
            return resolver.allowed_reminder_turn_on_statuses
        return resolver.allowed_reminder_turn_off_statuses

    @staticmethod
    def _no_op_statuses(action: ReminderAction) -> tuple[str, ...]:
        if action is ReminderAction.TURN_ON:
            return ("scheduled", "notified")
        if action is ReminderAction.TURN_OFF:
            return ("cancelled", "dismissed", "completed")
        return ()

    @staticmethod
    def _rejection(
        operation: str,
        reason: str,
        _failure_code: str,
    ) -> LLMRetrievalValidationResult:
        return LLMRetrievalValidationResult(
            operation=operation,
            validation_result=ActionValidationResult.REJECT_UNSAFE_TRANSITION,
            selected_candidate_keys=(),
            confidence=0.0,
            ambiguous=False,
            reason_summary=reason,
            candidate_assessments=(),
            should_execute=False,
            requires_hitl=False,
            factuality_concern=False,
            # Any call to this helper represents an invalid input/model
            # contract or an unsafe PASS rejected by guard 2. A valid semantic
            # FAIL returns earlier with reminder_validation_fail and its user
            # question, so guard failures must surface as pipeline errors.
            hitl_reason="internal_pipeline_failure",
        )


class ReminderContentFinalizationStrategy:
    """Ask model 3 to approve only model 1's validated MODIFY state."""

    def __init__(
        self,
        *,
        llm: LLMClient,
        prompts: PromptRegistry,
        min_confidence: float,
    ) -> None:
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("Reminder finalization confidence must be in [0, 1]")
        self.llm = llm
        self.prompts = prompts
        self.min_confidence = min_confidence

    def finalize(
        self,
        *,
        context: PipelineContext,
        action_payload: dict[str, Any],
        first_model_response: dict[str, Any] | None,
        candidates: list[ReminderMutationCandidate],
        selected: ReminderMutationCandidate | None,
        validation: LLMRetrievalValidationResult,
    ) -> ReminderFinalizationResult | None:
        operation = _text(action_payload.get("action")).casefold()
        # Match the knowledge branch's model-3 boundary: only MODIFY may enter
        # finalization, and it must carry exactly one validated SQL candidate.
        if (
            operation != ReminderAction.MODIFY.value
            or selected is None
            or validation.operation != operation
            or validation.validation_result is not ActionValidationResult.EXECUTE
            or not validation.should_execute
            or validation.selected_candidate_keys != (selected.candidate_key,)
            or sum(
                candidate.candidate_key == selected.candidate_key
                for candidate in candidates
            )
            != 1
        ):
            return None
        isolated_first_response = (
            ReminderActionValidationStrategy._validate_first_model_response(
                first_model_response,
                requested_operation=operation,
            )
        )
        if isolated_first_response is None:
            return None
        expected_bindings = self._expected_bindings(
            operation=operation,
            action_payload=action_payload,
            selected=selected,
        )
        if expected_bindings is None:
            return None
        try:
            raw = self.llm.generate_json(
                task=LLMTask.REMINDER_CONTENT_FINALIZATION,
                system_prompt=self.prompts.system("reminder_content_finalization"),
                user_prompt=self.prompts.user(
                    PromptContext(
                        stage="reminder_content_finalization",
                        extra={
                            "first_model_response": isolated_first_response,
                        },
                    )
                ),
                schema=REMINDER_CONTENT_FINALIZATION_SCHEMA,
            )
            if not isinstance(raw, dict) or set(raw) != {"approved", "confidence"}:
                return None
            approved = raw["approved"]
            if not isinstance(approved, bool):
                return None
            confidence = float(raw["confidence"])
        except Exception:
            return None
        if (
            not approved
            or not isfinite(confidence)
            or not 0.0 <= confidence <= 1.0
            or confidence < self.min_confidence
        ):
            return None
        final_record = self._merge_bound_record(
            bindings=expected_bindings,
            action_payload=action_payload,
            selected=selected,
        )
        if final_record is None:
            return None
        return ReminderFinalizationResult(
            final_reminder=final_record,
            confidence=confidence,
            reason_summary="Model 3 approved the isolated model-1 MODIFY state.",
        )

    @staticmethod
    def _expected_bindings(
        *,
        operation: str,
        action_payload: dict[str, Any],
        selected: ReminderMutationCandidate | None,
    ) -> dict[str, str] | None:
        if operation != ReminderAction.MODIFY.value or selected is None:
            return None
        expected = {
            field: _SELECTED_CANDIDATE_SOURCE
            for field in REMINDER_EDITABLE_FIELDS
        }
        changed = [_text(field) for field in action_payload.get("changed_fields", [])]
        if (
            not changed
            or len(changed) != len(set(changed))
            or any(field not in expected for field in changed)
        ):
            return None
        for field in changed:
            expected[field] = _EXTRACTED_ACTION_SOURCE
        if all(
            _text(action_payload.get(field)) == selected.fields[field]
            for field in changed
        ):
            return None
        return expected

    @staticmethod
    def _merge_bound_record(
        *,
        bindings: dict[str, str],
        action_payload: dict[str, Any],
        selected: ReminderMutationCandidate | None,
    ) -> dict[str, str] | None:
        final_record: dict[str, str] = {}
        for field in REMINDER_EDITABLE_FIELDS:
            source = bindings.get(field)
            if source == _EXTRACTED_ACTION_SOURCE:
                final_record[field] = _text(action_payload.get(field))
            elif source == _SELECTED_CANDIDATE_SOURCE and selected is not None:
                final_record[field] = selected.fields[field]
            else:
                return None
        return final_record


class ReminderMutationPipeline:
    """Run guard 1, validation/guard 2, and finalization for one mutation.

    Model 1 runs in ``LLMReminderActionDetector`` before this object. Guard 1
    proves that its canonical response exactly produced the downstream action
    state. Model 2 then validates SQL-only reminder candidates and its strategy
    guards that response. Finalization is deterministic for non-MODIFY actions
    and model-3-backed for MODIFY. SQL execution remains in ``ReminderBranch``.
    """

    def __init__(
        self,
        *,
        config: AssistantConfig,
        validator: ReminderActionValidationStrategy,
        finalizer: ReminderContentFinalizationStrategy,
    ) -> None:
        self.config = config
        self.validator = validator
        self.finalizer = finalizer

    def build_action(
        self,
        *,
        context: PipelineContext,
        action_payload: dict[str, Any],
        repository: AssistantRepository,
    ) -> ValidatedReminderAction:
        action_name = _text(action_payload.get("action")).casefold()
        if action_name not in {item.value for item in ReminderAction}:
            return self._rejection(
                ReminderAction.ADD,
                action_payload,
                "Unsupported extracted reminder action.",
            )
        action = ReminderAction(action_name)
        retrieval_text = _text(action_payload.get("retrieval_text"))
        first_model_response = context.request.metadata.get(
            "reminder_action_extraction_response"
        )
        guarded_first_response = self._guard_first_model_state(
            action=action,
            action_payload=action_payload,
            first_model_response=(
                dict(first_model_response)
                if isinstance(first_model_response, dict)
                else None
            ),
        )
        if guarded_first_response is None:
            return self._technical_failure(
                action,
                action_payload,
                "Reminder guard 1 rejected inconsistent model-1 state.",
            )

        statuses = tuple(
            dict.fromkeys(
                self.validator._allowed_statuses(action)
                + self.validator._no_op_statuses(action)
            )
        )
        try:
            full_rows = [
                row
                for row in repository.list_reminders(user_id=context.request.user_id)
                if _text(row.get("status")) in statuses
            ]
            candidates = [
                self._candidate_from_row(
                    str(row["reminder_id"]),
                    row,
                    retrieval_text,
                )
                for row in full_rows
            ]
            candidates.sort(
                key=lambda item: (item.deterministic_score, item.candidate_key),
                reverse=True,
            )
            candidates = candidates[
                : self.config.reminder_resolver.reminder_target_candidate_limit
            ]
            candidates = candidates[
                : self.config.retrieval_validation.reminder_llm_validation_max_candidates
            ]
        except Exception:
            return self._technical_failure(
                action,
                action_payload,
                "Reminder candidate retrieval failed safely.",
            )

        validation = self.validator.validate(
            context=context,
            action_payload=action_payload,
            candidates=candidates,
            first_model_response=guarded_first_response,
        )
        if validation.validation_result is not ActionValidationResult.EXECUTE:
            return ValidatedReminderAction(
                action=action,
                validation_result=validation.validation_result,
                confidence=validation.confidence,
                reason_summary=validation.reason_summary,
                requires_hitl=validation.requires_hitl,
                factuality_concern=validation.factuality_concern,
                hitl_reason=validation.hitl_reason,
                clarification_question=validation.clarification_question,
            )

        candidate_map = {candidate.candidate_key: candidate for candidate in candidates}
        selected = None
        if action is not ReminderAction.ADD:
            if len(validation.selected_candidate_keys) != 1:
                return self._rejection(
                    action,
                    action_payload,
                    "Reminder validation did not select exactly one SQL target.",
                )
            selected = candidate_map.get(validation.selected_candidate_keys[0])
            if selected is None:
                return self._rejection(
                    action,
                    action_payload,
                    "Reminder validation selected an unavailable SQL target.",
                )
        elif validation.selected_candidate_keys:
            return self._rejection(
                action,
                action_payload,
                "Reminder add validation selected an existing target.",
            )

        finalized: ReminderFinalizationResult | None = None
        if action is ReminderAction.MODIFY:
            finalized = self.finalizer.finalize(
                context=context,
                action_payload=action_payload,
                first_model_response=guarded_first_response,
                candidates=candidates,
                selected=selected,
                validation=validation,
            )
            if finalized is None:
                return self._technical_failure(
                    action,
                    action_payload,
                    "Reminder content finalization failed safely.",
                )
        return self._validated_action(
            action=action,
            action_payload=action_payload,
            selected=selected,
            validation=validation,
            finalized=finalized,
        )

    @staticmethod
    def _guard_first_model_state(
        *,
        action: ReminderAction,
        action_payload: dict[str, Any],
        first_model_response: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Guard 1: prove every downstream action field came from model 1.

        Completeness, action compatibility, normalized-time requirements, and
        candidate semantics remain model 2/guard 2 responsibilities.
        """
        isolated = ReminderActionValidationStrategy._validate_first_model_response(
            first_model_response,
            requested_operation=action.value,
        )
        if isolated is None:
            return None
        decoded = _decode_field_values(isolated["field_values"])
        raw_confidence = action_payload.get("confidence")
        if (
            decoded is None
            or isinstance(raw_confidence, bool)
            or not isinstance(raw_confidence, (int, float))
            or float(raw_confidence) != isolated["confidence"]
            or _text(action_payload.get("retrieval_text"))
            != isolated["retrieval_text"]
            or _text(action_payload.get("extracted_action")).casefold()
            != action.value
        ):
            return None
        supplied_fields, values = decoded
        expected_changed_fields = (
            supplied_fields if action is ReminderAction.MODIFY else []
        )
        expected_time_semantics = _time_semantics_for(
            action_name=action.value,
            supplied_fields=supplied_fields,
            values=values,
        )
        expected_toggle_direction = (
            action.value
            if action in {ReminderAction.TURN_ON, ReminderAction.TURN_OFF}
            else ""
        )
        if (
            [_text(field) for field in action_payload.get("changed_fields") or []]
            != expected_changed_fields
            or _text(action_payload.get("time_semantics"))
            != expected_time_semantics
            or _text(action_payload.get("toggle_direction"))
            != expected_toggle_direction
            or any(
                _text(action_payload.get(field)) != values[field]
                for field in REMINDER_EDITABLE_FIELDS
            )
        ):
            return None
        return isolated

    @staticmethod
    def _candidate_from_row(
        reminder_id: str,
        row: dict[str, Any],
        retrieval_text: str,
    ) -> ReminderMutationCandidate:
        fields = {
            "subject": _text(row.get("subject")),
            "reminder_summary": _text(row.get("reminder_summary")),
            "raw_reminder": _text(row.get("raw_reminder")),
            "notification_time": _text(row.get("reminder_time")),
            "event_time": _text(row.get("event_time")),
            "user_timezone": _text(row.get("user_timezone")),
            "original_time_text": _text(row.get("original_time_text")),
            "recurrence_rule": _text(row.get("recurrence_rule")),
            "recurrence_timezone": _text(row.get("recurrence_timezone")),
            "supporting_question": _text(row.get("supporting_question")),
            "supporting_response": _text(row.get("supporting_response")),
        }
        target = _normalized(retrieval_text)
        normalized_values = [_normalized(value) for value in fields.values() if value]
        exact_or_substring = max(
            (
                1.0
                if target == value
                else 0.98
                if target and target in value
                else 0.0
            )
            for value in normalized_values
        ) if normalized_values else 0.0
        candidate_text = " ".join(normalized_values)
        target_tokens = set(re.findall(r"[\w]+", target))
        candidate_tokens = set(re.findall(r"[\w]+", candidate_text))
        token_score = (
            len(target_tokens & candidate_tokens) / len(target_tokens)
            if target_tokens
            else 0.0
        )
        fuzzy = SequenceMatcher(None, target, candidate_text).ratio() if target else 0.0
        score = max(exact_or_substring, token_score, fuzzy)
        return ReminderMutationCandidate(
            candidate_key=reminder_id,
            status=_text(row.get("status")),
            version=int(row.get("version") or 0),
            deterministic_score=float(max(0.0, min(1.0, score))),
            fields=fields,
            next_fire_time=_text(row.get("next_fire_time")),
            parent_recurring_reminder_id=_text(
                row.get("parent_recurring_reminder_id")
            ),
            timing_plan_status=_text(row.get("timing_plan_status")) or "pending",
        )

    def _validated_action(
        self,
        *,
        action: ReminderAction,
        action_payload: dict[str, Any],
        selected: ReminderMutationCandidate | None,
        validation: LLMRetrievalValidationResult,
        finalized: ReminderFinalizationResult | None,
    ) -> ValidatedReminderAction:
        if action is ReminderAction.MODIFY:
            if finalized is None:
                raise ValueError("MODIFY requires model-3 finalization")
            record = finalized.final_reminder
        elif action is ReminderAction.ADD:
            record = {
                field: _text(action_payload.get(field))
                for field in REMINDER_EDITABLE_FIELDS
            }
        else:
            if selected is None:
                raise ValueError("Target reminder action requires one SQL candidate")
            record = dict(selected.fields)
        notification_time = _parse_datetime(record["notification_time"])
        event_time = _parse_datetime(record["event_time"])
        storage_time = notification_time or event_time
        changed_fields = {
            _text(field) for field in action_payload.get("changed_fields") or []
        }
        if action is ReminderAction.ADD:
            timing_plan_required = notification_time is None
        elif action is ReminderAction.MODIFY:
            if "notification_time" in changed_fields:
                timing_plan_required = False
            elif "event_time" in changed_fields:
                timing_plan_required = True
            else:
                timing_plan_required = bool(
                    selected and selected.timing_plan_status != "planned"
                )
        else:
            timing_plan_required = True
        target_ids = (selected.candidate_key,) if selected else ()
        matched_fields = tuple(
            dict.fromkeys(
                field
                for assessment in validation.candidate_assessments
                if assessment.candidate_key in target_ids
                for field in assessment.matched_fields
            )
        )
        confidence_values = [
            float(action_payload.get("confidence", 1.0)),
            validation.confidence,
        ]
        reason_parts = [validation.reason_summary]
        if finalized is not None:
            confidence_values.append(finalized.confidence)
            reason_parts.append(finalized.reason_summary)
        confidence = min(confidence_values)
        common = dict(
            action=action,
            validation_result=ActionValidationResult.EXECUTE,
            target_reminder_ids=target_ids,
            observed_status=selected.status if selected else None,
            observed_version=selected.version if selected else None,
            observed_reminder_time=selected.reminder_time if selected else None,
            subject=record["subject"],
            event_time=event_time,
            reminder_time=storage_time,
            reminder_summary=record["reminder_summary"],
            raw_reminder=record["raw_reminder"],
            user_timezone=record["user_timezone"] or self.config.default_timezone,
            original_time_text=record["original_time_text"] or None,
            recurrence_rule=record["recurrence_rule"] or None,
            recurrence_timezone=record["recurrence_timezone"] or None,
            supporting_question=record["supporting_question"],
            supporting_response=record["supporting_response"],
            timing_plan_required=timing_plan_required,
            confidence=confidence,
            matched_fields=matched_fields,
            reason_summary=" ".join(reason_parts).strip(),
            requires_hitl=False,
            factuality_concern=False,
            hitl_reason=None,
        )
        if action is ReminderAction.MODIFY:
            common.update(
                replacement_subject=record["subject"],
                replacement_time=storage_time,
                replacement_summary=record["reminder_summary"],
                replacement_recurrence_rule=record["recurrence_rule"],
                replacement_recurrence_timezone=record["recurrence_timezone"],
            )
        return ValidatedReminderAction(**common)

    @staticmethod
    def _rejection(
        action: ReminderAction,
        action_payload: dict[str, Any],
        reason: str,
    ) -> ValidatedReminderAction:
        return ValidatedReminderAction(
            action=action,
            validation_result=ActionValidationResult.REJECT_UNSAFE_TRANSITION,
            confidence=0.0,
            reason_summary=reason,
            requires_hitl=False,
            factuality_concern=False,
            hitl_reason="internal_pipeline_failure",
        )

    @staticmethod
    def _technical_failure(
        action: ReminderAction,
        action_payload: dict[str, Any],
        reason: str,
    ) -> ValidatedReminderAction:
        return ValidatedReminderAction(
            action=action,
            validation_result=ActionValidationResult.REJECT_UNSAFE_TRANSITION,
            confidence=0.0,
            reason_summary=reason,
            requires_hitl=False,
            factuality_concern=False,
            hitl_reason="internal_pipeline_failure",
        )
