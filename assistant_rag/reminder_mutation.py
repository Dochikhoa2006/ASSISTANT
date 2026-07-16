"""Three-stage LLM orchestration for reminder mutations only."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from difflib import SequenceMatcher
from hashlib import sha256
import json
from math import isfinite
import re
from typing import Any
import unicodedata

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

_TEXT_FIELDS = {
    "subject",
    "reminder_summary",
    "raw_reminder",
    "original_time_text",
    "recurrence_rule",
    "supporting_question",
    "supporting_response",
}
_TIME_FIELDS = {"notification_time", "event_time"}
_CLEARABLE_FIELDS = {
    "reminder_summary",
    "raw_reminder",
    "original_time_text",
    "recurrence_rule",
    "recurrence_timezone",
    "supporting_question",
    "supporting_response",
}
_TOGGLE_ACTIONS = {ReminderAction.TURN_ON, ReminderAction.TURN_OFF}
_EXTERNAL_REMINDER_ACTIONS = {"add", "delete", "modify", "toggle"}
_TOGGLE_DIRECTIONS = {
    ReminderAction.TURN_ON.value,
    ReminderAction.TURN_OFF.value,
}
_EXTRACTED_ACTION_SOURCE = "extracted_action"
_SELECTED_CANDIDATE_SOURCE = "selected_candidate"
_FIELD_PREVIEW_EDGE_CHARS = 48
_REMINDER_EXTRACTION_FIELDS = {
    "action",
    "toggle_direction",
    "retrieval_text",
    "changed_fields",
    *REMINDER_EDITABLE_FIELDS,
    "time_semantics",
    "confidence",
    "missing_fields",
    "reason_summary",
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
        validated_confirmation_actions = list(
            request.metadata.get("validated_reminder_actions") or []
        )
        is_confirmation_replay = bool(
            request.confirmation_token
            and request.metadata.get("confirmation_approved")
        )
        extraction_metadata = {
            key: value
            for key, value in request.metadata.items()
            if key
            not in {
                "intent",
                "reminder_actions",
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
                                "toggle",
                            ],
                            "toggle_directions": ["turn_on", "turn_off"],
                            "cardinality": "exactly_one",
                            "editable_fields": list(REMINDER_EDITABLE_FIELDS),
                            "action_content_contract": {
                                "add": {
                                    "target": "retrieval_text",
                                    "changed_fields": "empty",
                                    "content": "all explicitly supplied new reminder fields",
                                    "toggle_direction": "empty",
                                },
                                "delete": {
                                    "target": "retrieval_text",
                                    "changed_fields": "empty",
                                    "content": "all editable fields empty",
                                    "toggle_direction": "empty",
                                },
                                "modify": {
                                    "target": "retrieval_text",
                                    "changed_fields": "one or more fields on the same reminder",
                                    "content": "only values or explicit clearings for changed_fields",
                                    "toggle_direction": "empty",
                                },
                                "toggle": {
                                    "target": "retrieval_text",
                                    "changed_fields": "empty",
                                    "content": "all editable fields empty",
                                    "toggle_direction": "turn_on or turn_off",
                                },
                            },
                            "field_semantics": {
                                "subject": "title or name",
                                "reminder_summary": "concise summary",
                                "raw_reminder": "body, content, instructions, or note",
                                "notification_time": "when to notify the user",
                                "event_time": "when the event or deadline occurs",
                                "user_timezone": "user time zone for absolute interpretation",
                                "original_time_text": "verbatim current-turn time expression",
                                "recurrence_rule": "repeat schedule",
                                "recurrence_timezone": "time zone governing recurrence",
                                "supporting_question": "stored follow-up question",
                                "supporting_response": "stored answer to the follow-up question",
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
                            "confirmation_replay": is_confirmation_replay,
                            "trusted_confirmation_action_context": (
                                validated_confirmation_actions
                                if is_confirmation_replay
                                else []
                            ),
                        },
                    )
                ),
                schema=REMINDER_ACTION_EXTRACTION_SCHEMA,
            )
        except Exception:
            return self._failed("reminder_action_extraction_failed", ["action"])

        if not isinstance(payload, dict) or set(payload) != _REMINDER_EXTRACTION_FIELDS:
            return self._failed("invalid_reminder_action_extraction", ["action"])
        string_fields = {
            "action",
            "toggle_direction",
            "retrieval_text",
            *REMINDER_EDITABLE_FIELDS,
            "time_semantics",
            "reason_summary",
        }
        if not all(isinstance(payload[field], str) for field in string_fields):
            return self._failed("invalid_reminder_action_extraction", ["action"])
        if (
            isinstance(payload["confidence"], bool)
            or not isinstance(payload["confidence"], (int, float))
            or not isinstance(payload["changed_fields"], list)
            or not all(isinstance(item, str) for item in payload["changed_fields"])
            or not isinstance(payload["missing_fields"], list)
            or not all(isinstance(item, str) for item in payload["missing_fields"])
        ):
            return self._failed("invalid_reminder_action_extraction", ["action"])

        try:
            extracted_action_name = payload["action"].strip().casefold()
            toggle_direction = payload["toggle_direction"].strip().casefold()
            retrieval_text = payload["retrieval_text"].strip()
            values = {
                field: _text(payload[field]) for field in REMINDER_EDITABLE_FIELDS
            }
            confidence = float(payload["confidence"])
            changed_fields = [item.strip() for item in payload["changed_fields"]]
            missing_fields = [item.strip() for item in payload["missing_fields"]]
            time_semantics = payload["time_semantics"].strip()
        except (KeyError, TypeError, ValueError):
            return self._failed("invalid_reminder_action_extraction", ["action"])

        if (
            extracted_action_name not in _EXTERNAL_REMINDER_ACTIONS
            or not isfinite(confidence)
            or not 0.0 <= confidence <= 1.0
            or confidence < self.min_confidence
            or missing_fields
            or len(changed_fields) != len(set(changed_fields))
            or any(field not in REMINDER_EDITABLE_FIELDS for field in changed_fields)
            or time_semantics
            not in {"notification_time", "event_time", "both", "unchanged"}
        ):
            return ActionDetectionResult(
                intent=Intent.REMINDER,
                confidence=0.0,
                missing_fields=self._normalize_missing_fields(missing_fields)
                or ["action"],
                risk_flags=["low_confidence_or_incomplete_reminder_extraction"],
            )

        if extracted_action_name == "toggle":
            if toggle_direction not in _TOGGLE_DIRECTIONS:
                return self._failed(
                    "toggle_reminder_requires_exact_direction",
                    ["toggle_direction"],
                )
            action_name = toggle_direction
        else:
            if toggle_direction:
                return self._failed(
                    "non_toggle_reminder_forbids_toggle_direction",
                    ["toggle_direction"],
                )
            action_name = extracted_action_name

        shape_error = self._validate_shape(
            action_name=action_name,
            retrieval_text=retrieval_text,
            changed_fields=changed_fields,
            values=values,
            time_semantics=time_semantics,
        )
        if shape_error:
            reason, missing = shape_error
            return self._failed(reason, missing)

        trusted_confirmation_source = ""
        if is_confirmation_replay:
            if len(validated_confirmation_actions) != 1:
                return self._failed(
                    "invalid_reminder_confirmation_action_context",
                    ["single_action"],
                )
            trusted_confirmation_action = validated_confirmation_actions[0]
            stored_action_name = (
                _text(trusted_confirmation_action.get("action")).casefold()
                if isinstance(trusted_confirmation_action, dict)
                else _text(
                    getattr(
                        getattr(trusted_confirmation_action, "action", None),
                        "value",
                        getattr(trusted_confirmation_action, "action", ""),
                    )
                ).casefold()
            )
            if stored_action_name != action_name:
                return self._failed(
                    "reminder_confirmation_action_mismatch",
                    ["single_action"],
                )
            trusted_confirmation_source = json.dumps(
                validated_confirmation_actions,
                ensure_ascii=False,
                default=str,
            )

        current_sources = [rewritten_query]
        history_source = json.dumps(chat_history, ensure_ascii=False, default=str)
        retrieval_sources = (
            list(current_sources)
            if action_name == ReminderAction.ADD.value
            else current_sources + [history_source]
        )
        if trusted_confirmation_source:
            retrieval_sources.append(trusted_confirmation_source)
        if not _is_grounded(retrieval_text, retrieval_sources):
            return self._failed(
                "reminder_target_not_grounded_in_request_context",
                ["target_description"],
            )

        grounded_fields = (
            {field for field in REMINDER_EDITABLE_FIELDS if values[field]}
            if action_name == ReminderAction.ADD.value
            else {field for field in changed_fields if values[field]}
        )
        content_sources = current_sources + (
            [trusted_confirmation_source] if trusted_confirmation_source else []
        )
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

        if any(values[field] for field in _TIME_FIELDS):
            original_time = values["original_time_text"]
            if not original_time or not _is_grounded(original_time, content_sources):
                return self._failed(
                    "reminder_time_not_grounded_in_current_turn",
                    ["reminder_time"],
                )
            if any(
                values[field] and _parse_datetime(values[field]) is None
                for field in _TIME_FIELDS
            ):
                return self._failed(
                    "invalid_reminder_timestamp",
                    ["reminder_time"],
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
    ) -> tuple[str, list[str]] | None:
        if not retrieval_text:
            return "missing_reminder_retrieval_text", ["target_description"]
        nonempty = {field for field, value in values.items() if value}
        if action_name == ReminderAction.ADD.value:
            missing = []
            if not values["subject"]:
                missing.append("subject")
            if not values["raw_reminder"]:
                missing.append("reminder_content")
            if not (values["notification_time"] or values["event_time"]):
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

        if not changed_fields:
            return "modify_reminder_requires_changed_fields", ["reminder_update"]
        unexpected = nonempty - set(changed_fields)
        missing_values = [
            field
            for field in changed_fields
            if not values[field] and field not in _CLEARABLE_FIELDS
        ]
        if unexpected or missing_values:
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
    def _normalize_missing_fields(fields: list[str]) -> list[str]:
        mapping = {
            "retrieval_text": "target_description",
            "notification_time": "reminder_time",
            "event_time": "reminder_time",
            "changed_fields": "reminder_update",
            "raw_reminder": "reminder_content",
        }
        return list(dict.fromkeys(mapping.get(field, field) for field in fields if field))

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
    ) -> LLMRetrievalValidationResult:
        operation = _text(action_payload.get("action")).casefold()
        try:
            raw = self.llm.generate_json(
                task=LLMTask.REMINDER_ACTION_VALIDATION,
                system_prompt=self.prompts.system("reminder_action_validation"),
                user_prompt=self.prompts.user(
                    PromptContext(
                        stage="reminder_action_validation",
                        user_id=context.request.user_id,
                        rewritten_query=context.rewritten_query,
                        intent=Intent.REMINDER.value,
                        metadata=context.request.metadata,
                        platform_context=context.request.platform_context,
                        chat_history=context.chat_history,
                        extra={
                            "operation": operation,
                            "runtime_now_utc": _text(
                                action_payload.get("runtime_now_utc")
                            ),
                            "extracted_action": action_payload,
                            "candidate_reminders": [
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
                            ],
                            "validation_policy": {
                                "min_confidence": self.config.retrieval_validation.reminder_llm_validation_min_confidence,
                                "allowed_statuses": list(
                                    self._allowed_statuses(ReminderAction(operation))
                                ),
                                "no_op_statuses": list(
                                    self._no_op_statuses(ReminderAction(operation))
                                ),
                                "candidate_assessment_cardinality": "exactly_once_each",
                                "runtime_now_utc": _text(
                                    action_payload.get("runtime_now_utc")
                                ),
                            },
                        },
                    )
                ),
                schema=REMINDER_ACTION_VALIDATION_SCHEMA,
                model_override=(
                    self.config.retrieval_validation.reminder_llm_validation_model
                ),
            )
        except Exception:
            return self._clarification(
                operation,
                "Reminder validation model failed safely.",
                "validation_failure",
            )
        return self._parse(
            raw=raw,
            operation=operation,
            action_payload=action_payload,
            candidates=candidates,
        )

    def _parse(
        self,
        *,
        raw: dict[str, Any],
        operation: str,
        action_payload: dict[str, Any],
        candidates: list[ReminderMutationCandidate],
    ) -> LLMRetrievalValidationResult:
        candidate_map = {candidate.candidate_key: candidate for candidate in candidates}
        try:
            assessments = tuple(
                RetrievalCandidateAssessment(
                    candidate_key=_text(item["candidate_key"]),
                    matches_target=bool(item["matches_target"]),
                    action_compatible=bool(item["action_compatible"]),
                    confidence=float(item["confidence"]),
                    matched_fields=tuple(_text(value) for value in item["matched_fields"]),
                    matched_text=str(item["matched_text"]),
                    reason_summary=str(item["reason_summary"]),
                )
                for item in raw["candidate_assessments"]
            )
            result = LLMRetrievalValidationResult(
                operation=_text(raw["operation"]).casefold(),
                validation_result=ActionValidationResult(
                    _text(raw["validation_result"]).casefold()
                ),
                selected_candidate_keys=tuple(
                    _text(value) for value in raw["selected_candidate_keys"]
                ),
                confidence=float(raw["confidence"]),
                ambiguous=bool(raw["ambiguous"]),
                should_execute=bool(raw["should_execute"]),
                requires_hitl=bool(raw["requires_hitl"]),
                factuality_concern=bool(raw["factuality_concern"]),
                hitl_reason=_text(raw["hitl_reason"]) or None,
                reason_summary=str(raw["reason_summary"]),
                candidate_assessments=assessments,
            )
        except (KeyError, TypeError, ValueError):
            return self._clarification(
                operation,
                "Reminder validation returned an invalid contract.",
                "invalid_validation_contract",
            )

        minimum = self.config.retrieval_validation.reminder_llm_validation_min_confidence
        if (
            result.operation != operation
            or not isfinite(result.confidence)
            or not 0.0 <= result.confidence <= 1.0
            or result.confidence < minimum
            or len(set(result.selected_candidate_keys))
            != len(result.selected_candidate_keys)
            or any(key not in candidate_map for key in result.selected_candidate_keys)
        ):
            return self._clarification(
                operation,
                "Reminder validation confidence or selection was unsafe.",
                "unsafe_validation_selection",
            )

        assessment_keys = [item.candidate_key for item in result.candidate_assessments]
        if (
            len(set(assessment_keys)) != len(assessment_keys)
            or set(assessment_keys) != set(candidate_map)
        ):
            return self._clarification(
                operation,
                "Every reminder candidate must be assessed exactly once.",
                "incomplete_candidate_assessments",
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
                    field not in {*REMINDER_EDITABLE_FIELDS, "status"}
                    for field in assessment.matched_fields
                )
            ):
                return self._clarification(
                    operation,
                    "Reminder candidate evidence was invalid.",
                    "invalid_candidate_evidence",
                )
            if assessment.matches_target:
                if (
                    not assessment.matched_fields
                    or not assessment.matched_text
                    or not any(
                        assessment.matched_text
                        in (
                            candidate.status
                            if field == "status"
                            else candidate.fields[field]
                        )
                        for field in assessment.matched_fields
                    )
                ):
                    return self._clarification(
                        operation,
                        "Reminder match evidence was not present in SQL fields.",
                        "ungrounded_candidate_match",
                    )
            elif assessment.matched_fields or assessment.matched_text:
                return self._clarification(
                    operation,
                    "A non-match cannot contain match evidence.",
                    "contradictory_candidate_evidence",
                )

        is_execute = result.validation_result is ActionValidationResult.EXECUTE
        is_clarification = result.validation_result in {
            ActionValidationResult.CLARIFY_AMBIGUOUS_TARGET,
            ActionValidationResult.CLARIFY_MISSING_FIELDS,
        }
        if (
            result.should_execute != is_execute
            or result.requires_hitl != is_clarification
            or (result.factuality_concern and not is_clarification)
        ):
            return self._clarification(
                operation,
                "Reminder validation flags contradicted its result.",
                "contradictory_validation_flags",
            )
        if result.factuality_concern and not result.hitl_reason:
            result = replace(result, hitl_reason="factuality_concern")

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
            allowed = {
                ActionValidationResult.EXECUTE,
                ActionValidationResult.SKIP_ALREADY_EXISTS,
                ActionValidationResult.CLARIFY_AMBIGUOUS_TARGET,
                ActionValidationResult.CLARIFY_MISSING_FIELDS,
            }
            if result.validation_result not in allowed:
                return self._clarification(operation, "Invalid add decision.", "invalid_action_result")
            if is_execute and (result.selected_candidate_keys or strong_matches):
                return self._clarification(operation, "Add duplicate evidence was inconsistent.", "duplicate_conflict")
            if (
                result.validation_result is ActionValidationResult.SKIP_ALREADY_EXISTS
                and (result.selected_candidate_keys or not strong_matches)
            ):
                return self._clarification(operation, "Duplicate reminder evidence was insufficient.", "duplicate_conflict")
        else:
            allowed = {
                ActionValidationResult.EXECUTE,
                ActionValidationResult.SKIP_NOT_FOUND,
                ActionValidationResult.SKIP_ALREADY_EXISTS,
                ActionValidationResult.CLARIFY_AMBIGUOUS_TARGET,
                ActionValidationResult.CLARIFY_MISSING_FIELDS,
            }
            if result.validation_result not in allowed:
                return self._clarification(operation, "Invalid target decision.", "invalid_action_result")
            if is_execute and (
                len(result.selected_candidate_keys) != 1
                or strong_matches != {result.selected_candidate_keys[0]}
                or semantic_matches != {result.selected_candidate_keys[0]}
            ):
                return self._clarification(operation, "Reminder target was not unique.", "ambiguous_target")
            if (
                result.validation_result is ActionValidationResult.SKIP_NOT_FOUND
                and (result.selected_candidate_keys or semantic_matches)
            ):
                return self._clarification(operation, "Not-found result hid matching evidence.", "contradictory_not_found")
            if result.validation_result is ActionValidationResult.SKIP_ALREADY_EXISTS:
                no_op_valid = False
                if (
                    len(result.selected_candidate_keys) == 1
                    and semantic_matches == {result.selected_candidate_keys[0]}
                ):
                    matched_key = next(iter(semantic_matches))
                    matched_candidate = candidate_map[matched_key]
                    if requested_action is ReminderAction.MODIFY:
                        no_op_valid = (
                            matched_key in strong_matches
                            and not self._modify_changes_candidate(
                                action_payload,
                                matched_candidate,
                            )
                        )
                    elif requested_action in _TOGGLE_ACTIONS:
                        no_op_valid = (
                            matched_candidate.status
                            in self._no_op_statuses(requested_action)
                            and matched_key not in strong_matches
                        )
                if not no_op_valid:
                    return self._clarification(
                        operation,
                        "Reminder no-op evidence was inconsistent.",
                        "invalid_no_op_decision",
                    )

        if (
            result.validation_result is ActionValidationResult.CLARIFY_AMBIGUOUS_TARGET
            and not result.ambiguous
        ):
            return self._clarification(operation, "Ambiguity flag was missing.", "ambiguous_target")
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
    def _clarification(
        operation: str,
        reason: str,
        hitl_reason: str,
    ) -> LLMRetrievalValidationResult:
        return LLMRetrievalValidationResult(
            operation=operation,
            validation_result=ActionValidationResult.CLARIFY_AMBIGUOUS_TARGET,
            selected_candidate_keys=(),
            confidence=1.0,
            ambiguous=True,
            reason_summary=reason,
            candidate_assessments=(),
            should_execute=False,
            requires_hitl=True,
            factuality_concern=hitl_reason == "factuality_concern",
            hitl_reason=hitl_reason,
        )


class ReminderContentFinalizationStrategy:
    """Bind reminder fields to trusted sources and merge them deterministically."""

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
        candidates: list[ReminderMutationCandidate],
        selected: ReminderMutationCandidate | None,
        validation: LLMRetrievalValidationResult,
    ) -> ReminderFinalizationResult | None:
        operation = _text(action_payload.get("action")).casefold()
        expected_bindings = self._expected_bindings(
            operation=operation,
            action_payload=action_payload,
            selected=selected,
        )
        if expected_bindings is None:
            return None
        selected_key = selected.candidate_key if selected else ""
        try:
            raw = self.llm.generate_json(
                task=LLMTask.REMINDER_CONTENT_FINALIZATION,
                system_prompt=self.prompts.system("reminder_content_finalization"),
                user_prompt=self.prompts.user(
                    PromptContext(
                        stage="reminder_content_finalization",
                        user_id=context.request.user_id,
                        rewritten_query=context.rewritten_query,
                        intent=Intent.REMINDER.value,
                        metadata=context.request.metadata,
                        platform_context=context.request.platform_context,
                        chat_history=context.chat_history,
                        extra={
                            "operation": operation,
                            "runtime_now_utc": _text(
                                action_payload.get("runtime_now_utc")
                            ),
                            "extracted_action_manifest": {
                                "action": operation,
                                "changed_fields": [
                                    _text(field)
                                    for field in action_payload.get(
                                        "changed_fields", []
                                    )
                                ],
                                "time_semantics": _text(
                                    action_payload.get("time_semantics")
                                ),
                                "field_manifest": self._field_manifest(
                                    action_payload
                                ),
                            },
                            "selected_candidate_manifest": (
                                {
                                    "candidate_key": selected.candidate_key,
                                    "status": selected.status,
                                    "version": selected.version,
                                    "next_fire_time_present": bool(
                                        selected.next_fire_time
                                    ),
                                    "parent_recurring_reminder_present": bool(
                                        selected.parent_recurring_reminder_id
                                    ),
                                    "timing_plan_status": (
                                        selected.timing_plan_status
                                    ),
                                    "field_manifest": self._field_manifest(
                                        selected.fields
                                    ),
                                }
                                if selected is not None
                                else None
                            ),
                            "validation_summary": {
                                "operation": validation.operation,
                                "validation_result": (
                                    validation.validation_result.value
                                ),
                                "selected_candidate_keys": list(
                                    validation.selected_candidate_keys
                                ),
                                "confidence": validation.confidence,
                                "should_execute": validation.should_execute,
                            },
                        },
                    )
                ),
                schema=REMINDER_CONTENT_FINALIZATION_SCHEMA,
            )
            if not isinstance(raw, dict) or set(raw) != {
                "operation",
                "selected_candidate_key",
                "field_bindings",
                "confidence",
                "reason_summary",
            }:
                return None
            returned_operation = _text(raw["operation"]).casefold()
            returned_key = _text(raw["selected_candidate_key"])
            returned_bindings = self._parse_bindings(raw["field_bindings"])
            confidence = float(raw["confidence"])
            reason = str(raw["reason_summary"])
        except Exception:
            return None
        if (
            returned_operation != operation
            or returned_key != selected_key
            or returned_bindings != expected_bindings
            or not isfinite(confidence)
            or not 0.0 <= confidence <= 1.0
            or confidence < self.min_confidence
        ):
            return None
        final_record = self._merge_bound_record(
            bindings=returned_bindings,
            action_payload=action_payload,
            selected=selected,
        )
        if final_record is None:
            return None
        return ReminderFinalizationResult(
            final_reminder=final_record,
            confidence=confidence,
            reason_summary=reason,
        )

    @staticmethod
    def _expected_bindings(
        *,
        operation: str,
        action_payload: dict[str, Any],
        selected: ReminderMutationCandidate | None,
    ) -> dict[str, str] | None:
        if operation == ReminderAction.ADD.value:
            if selected is not None:
                return None
            return {
                field: _EXTRACTED_ACTION_SOURCE
                for field in REMINDER_EDITABLE_FIELDS
            }
        if selected is None:
            return None
        expected = {
            field: _SELECTED_CANDIDATE_SOURCE
            for field in REMINDER_EDITABLE_FIELDS
        }
        if operation == ReminderAction.MODIFY.value:
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
    def _field_manifest(values: dict[str, Any]) -> list[dict[str, Any]]:
        manifest: list[dict[str, Any]] = []
        for field in REMINDER_EDITABLE_FIELDS:
            value = _text(values.get(field))
            bounded_preview = (
                value
                if len(value) <= _FIELD_PREVIEW_EDGE_CHARS * 2
                else (
                    f"{value[:_FIELD_PREVIEW_EDGE_CHARS]}..."
                    f"{value[-_FIELD_PREVIEW_EDGE_CHARS:]}"
                )
            )
            manifest.append(
                {
                    "field": field,
                    "is_empty": not bool(value),
                    "character_count": len(value),
                    "sha256": sha256(value.encode("utf-8")).hexdigest(),
                    "bounded_preview": bounded_preview,
                }
            )
        return manifest

    @staticmethod
    def _parse_bindings(raw_bindings: Any) -> dict[str, str] | None:
        if not isinstance(raw_bindings, list) or len(raw_bindings) != len(
            REMINDER_EDITABLE_FIELDS
        ):
            return None
        parsed: dict[str, str] = {}
        for item in raw_bindings:
            if not isinstance(item, dict) or set(item) != {"field", "source"}:
                return None
            field = _text(item.get("field"))
            source = _text(item.get("source"))
            if (
                field not in REMINDER_EDITABLE_FIELDS
                or field in parsed
                or source
                not in {_EXTRACTED_ACTION_SOURCE, _SELECTED_CANDIDATE_SOURCE}
            ):
                return None
            parsed[field] = source
        if set(parsed) != set(REMINDER_EDITABLE_FIELDS):
            return None
        return parsed

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
    """Retrieve, validate, and finalize one extracted reminder mutation."""

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
            return self._clarification(
                ReminderAction.ADD,
                action_payload,
                "Unsupported extracted reminder action.",
            )
        action = ReminderAction(action_name)
        retrieval_text = _text(action_payload.get("retrieval_text"))
        if not retrieval_text:
            return self._clarification(
                action,
                action_payload,
                "The extracted reminder action has no retrieval text.",
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
            return self._clarification(
                action,
                action_payload,
                "Reminder candidate retrieval failed safely.",
            )

        validation = self.validator.validate(
            context=context,
            action_payload=action_payload,
            candidates=candidates,
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
            )

        candidate_map = {candidate.candidate_key: candidate for candidate in candidates}
        selected = None
        if action is not ReminderAction.ADD:
            if len(validation.selected_candidate_keys) != 1:
                return self._clarification(
                    action,
                    action_payload,
                    "Reminder validation did not select exactly one SQL target.",
                )
            selected = candidate_map.get(validation.selected_candidate_keys[0])
            if selected is None:
                return self._clarification(
                    action,
                    action_payload,
                    "Reminder validation selected an unavailable SQL target.",
                )
        elif validation.selected_candidate_keys:
            return self._clarification(
                action,
                action_payload,
                "Reminder add validation selected an existing target.",
            )

        finalized = self.finalizer.finalize(
            context=context,
            action_payload=action_payload,
            candidates=candidates,
            selected=selected,
            validation=validation,
        )
        if finalized is None:
            return self._clarification(
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
        finalized: ReminderFinalizationResult,
    ) -> ValidatedReminderAction:
        record = finalized.final_reminder
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
        confidence = min(
            float(action_payload.get("confidence", 1.0)),
            validation.confidence,
            finalized.confidence,
        )
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
            reason_summary=(
                f"{validation.reason_summary} {finalized.reason_summary}"
            ).strip(),
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
    def _clarification(
        action: ReminderAction,
        action_payload: dict[str, Any],
        reason: str,
    ) -> ValidatedReminderAction:
        return ValidatedReminderAction(
            action=action,
            validation_result=ActionValidationResult.CLARIFY_MISSING_FIELDS,
            confidence=0.0,
            reason_summary=reason,
            requires_hitl=True,
            factuality_concern=False,
            hitl_reason="reminder_validation_clarification",
        )
