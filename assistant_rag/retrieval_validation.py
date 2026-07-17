"""LLM-Based Retrieval Validation Strategies."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from math import isfinite
from typing import Any

from .config import RetrievalValidationConfig
from .contracts import (
    ActionValidationResult,
    KnowledgeValidationCandidate,
    LLMKnowledgeCandidatePayload,
    LLMReminderCandidatePayload,
    LLMRetrievalValidationResult,
    ReminderValidationCandidate,
    RetrievalCandidateAssessment,
)
from .llm import LLMClient, LLMTask, is_structured_fallback
from .prompts import (
    KNOWLEDGE_RETRIEVAL_VALIDATION_SCHEMA,
    PromptContext,
    PromptRegistry,
)
from .chat_history import inject_chat_history

logger = logging.getLogger(__name__)


class KnowledgeRetrievalValidationStrategy:
    def __init__(self, config: RetrievalValidationConfig, llm: LLMClient, prompts: PromptRegistry):
        self.config = config
        self.llm = llm
        self.prompts = prompts

    def validate(
        self,
        operation: str,
        user_query: str,
        rewritten_query: str,
        target_description: str,
        candidates: list[KnowledgeValidationCandidate],
        *,
        proposed_content: str | None = None,
        replacement_content: str | None = None,
        chat_history: list[dict[str, Any]] | None = None,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        platform_context: dict[str, Any] | None = None,
        first_model_response: dict[str, Any] | None = None,
    ) -> LLMRetrievalValidationResult | None:
        if not self.config.knowledge_llm_validation_enabled:
            return None

        if operation not in {"add", "delete", "modify"}:
            return self._build_internal_failure_result(operation)
        isolated_first_response = self._validate_first_model_response(
            first_model_response,
            requested_operation=operation,
        )
        if isolated_first_response is None:
            return self._build_internal_failure_result(operation)
        first_model_response_ready = self._first_model_response_ready(
            isolated_first_response
        )

        payloads = []
        candidate_map = {}
        for c in candidates:
            candidate_map[c.candidate_key] = c
            payloads.append(
                asdict(
                    LLMKnowledgeCandidatePayload(
                        candidate_key=c.candidate_key,
                        # Knowledge chunks are already bounded by the canonical
                        # chunker. Keep the full SQL text so a short target near
                        # the end of a long chunk cannot be hidden from validation.
                        text_excerpt=c.text,
                        source_title=c.source_title,
                        retrieval_score=c.retrieval_score,
                        rerank_score=c.rerank_score,
                        matched_fields=(),
                    )
                )
            )

        system_prompt = self.prompts.system("knowledge_action_validation")
        user_prompt = self.prompts.user(
            PromptContext(
                stage="knowledge_action_validation",
                extra={
                    "first_model_response": isolated_first_response,
                    "knowledge_retrieval": payloads,
                },
            )
        )

        try:
            raw_response = self.llm.generate_json(
                task=LLMTask.KNOWLEDGE_ACTION_VALIDATION,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema=self._schema(),
                model_override=self.config.knowledge_llm_validation_model,
            )
            if is_structured_fallback(raw_response):
                return self._build_internal_failure_result(operation)
            return self._parse_and_validate_result(
                raw_response,
                candidate_map,
                requested_operation=operation,
                first_model_response_ready=first_model_response_ready,
            )
        except Exception as e:
            logger.error(f"Knowledge LLM validation failed: {e}")
            return self._handle_failure()

    @staticmethod
    def _validate_first_model_response(
        response: dict[str, Any] | None,
        *,
        requested_operation: str,
    ) -> dict[str, Any] | None:
        required_fields = {
            "action",
            "text_content",
            "original_text",
            "replacement_text",
            "confidence",
        }
        legacy_fields = {"missing_fields", "reason_summary"}
        if (
            not isinstance(response, dict)
            or not required_fields.issubset(response)
            or not set(response).issubset(required_fields | legacy_fields)
        ):
            return None
        action = response["action"]
        text_content = response["text_content"]
        original_text = response["original_text"]
        replacement_text = response["replacement_text"]
        confidence = response["confidence"]
        if (
            not isinstance(action, str)
            or action != requested_operation
            or not isinstance(text_content, str)
            or not isinstance(original_text, str)
            or not isinstance(replacement_text, str)
            or isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not isfinite(float(confidence))
            or not 0.0 <= float(confidence) <= 1.0
        ):
            return None
        if "missing_fields" in response and (
            not isinstance(response["missing_fields"], list)
            or not all(
                isinstance(item, str) for item in response["missing_fields"]
            )
        ):
            return None
        if "reason_summary" in response and not isinstance(
            response["reason_summary"], str
        ):
            return None
        canonical_response: dict[str, Any] = {
            "action": action,
            "text_content": text_content.strip(),
            "original_text": original_text.strip(),
            "replacement_text": replacement_text.strip(),
            "confidence": float(confidence),
        }
        for legacy_field in legacy_fields:
            if legacy_field in response:
                canonical_response[legacy_field] = response[legacy_field]
        return canonical_response

    def _first_model_response_ready(
        self,
        response: dict[str, Any],
    ) -> bool:
        action = response["action"]
        text_content = response["text_content"]
        original_text = response["original_text"]
        replacement_text = response["replacement_text"]
        if action in {"add", "delete"}:
            return bool(text_content and not original_text and not replacement_text)
        return bool(
            action == "modify"
            and not text_content
            and original_text
            and replacement_text
        )

    def _parse_and_validate_result(
        self,
        raw_response: dict[str, Any],
        candidate_map: dict[str, KnowledgeValidationCandidate],
        *,
        requested_operation: str,
        first_model_response_ready: bool,
    ) -> LLMRetrievalValidationResult:
        """Guard 2: validate only model 2's contract and grounded evidence.

        The finalizer owns exact MODIFY content construction and the repository
        owns concurrency/database invariants, so this guard does not duplicate
        those later-stage responsibilities.
        """
        required_fields = {
            "decision",
            "selected_candidate_keys",
            "confidence",
            "clarification_question",
            "candidate_assessments",
        }
        required_assessment_fields = {
            "candidate_key",
            "confidence",
            "matched_text",
        }
        legacy_root_fields = {"operation", "reason_summary"}
        if (
            not isinstance(raw_response, dict)
            or not required_fields.issubset(raw_response)
            or not set(raw_response).issubset(
                required_fields | legacy_root_fields
            )
        ):
            return self._build_internal_failure_result(requested_operation)
        try:
            # ``operation`` used to be a required model-generated mirror of
            # model 1.  Model 2 cannot authoritatively change it, so derive it
            # from the already guarded request.  If an older model still emits
            # the mirror, reject only an actual contradiction.
            raw_operation = raw_response.get("operation", requested_operation)
            raw_decision = raw_response["decision"]
            raw_selected = raw_response["selected_candidate_keys"]
            raw_confidence = raw_response["confidence"]
            raw_question = raw_response["clarification_question"]
            raw_assessments = raw_response["candidate_assessments"]
            if (
                not isinstance(raw_operation, str)
                or raw_decision not in {"PASS", "FAIL"}
                or not isinstance(raw_selected, list)
                or not all(isinstance(value, str) for value in raw_selected)
                or isinstance(raw_confidence, bool)
                or not isinstance(raw_confidence, (int, float))
                or not isinstance(raw_question, str)
                or not isinstance(raw_assessments, list)
            ):
                return self._build_internal_failure_result(requested_operation)

            assessments_list: list[RetrievalCandidateAssessment] = []
            for item in raw_assessments:
                legacy_assessment_fields = {
                    "matches_target",
                    "action_compatible",
                    "matched_fields",
                    "reason_summary",
                }
                if (
                    not isinstance(item, dict)
                    or not required_assessment_fields.issubset(item)
                    or not set(item).issubset(
                        required_assessment_fields | legacy_assessment_fields
                    )
                ):
                    return self._build_internal_failure_result(requested_operation)
                item_confidence = item["confidence"]
                matched_text = item["matched_text"]
                if (
                    not isinstance(item["candidate_key"], str)
                    or isinstance(item_confidence, bool)
                    or not isinstance(item_confidence, (int, float))
                    or not isinstance(matched_text, str)
                ):
                    return self._build_internal_failure_result(requested_operation)
                # The exact grounded excerpt already carries the match
                # decision: empty means no match, non-empty means a match.
                # Keep accepting the former boolean during rolling upgrades,
                # but reject it if it contradicts the evidence-bearing field.
                matches_target = bool(matched_text)
                if "matches_target" in item:
                    legacy_matches_target = item["matches_target"]
                    if (
                        not isinstance(legacy_matches_target, bool)
                        or legacy_matches_target != matches_target
                    ):
                        return self._build_internal_failure_result(
                            requested_operation
                        )
                assessments_list.append(
                    RetrievalCandidateAssessment(
                        candidate_key=item["candidate_key"],
                        matches_target=matches_target,
                        # Candidate ownership/status and operation semantics are
                        # enforced before/after model 2.  Asking the model to
                        # repeat an action-compatible boolean added a failure
                        # surface without adding evidence.
                        action_compatible=True,
                        confidence=float(item_confidence),
                        matched_fields=("text",) if matches_target else (),
                        reason_summary=("matched" if matches_target else "not_matched"),
                        matched_text=matched_text,
                    )
                )
            assessments = tuple(assessments_list)
            res = LLMRetrievalValidationResult(
                operation=raw_operation.lower(),
                validation_result=(
                    ActionValidationResult.EXECUTE
                    if raw_decision == "PASS"
                    else ActionValidationResult.CLARIFY_MISSING_FIELDS
                ),
                selected_candidate_keys=tuple(raw_selected),
                confidence=float(raw_confidence),
                ambiguous=False,
                reason_summary=(
                    "Knowledge validation passed."
                    if raw_decision == "PASS"
                    else "Knowledge validation requires clarification."
                ),
                candidate_assessments=assessments,
                should_execute=raw_decision == "PASS",
                requires_hitl=raw_decision == "FAIL",
                factuality_concern=False,
                hitl_reason=(
                    "knowledge_validation_fail" if raw_decision == "FAIL" else None
                ),
                clarification_question=raw_question,
            )
        except (KeyError, TypeError, ValueError):
            return self._build_internal_failure_result(requested_operation)

        if res.operation != requested_operation:
            return self._build_internal_failure_result(requested_operation)
        if not isfinite(res.confidence) or not 0.0 <= res.confidence <= 1.0:
            return self._build_internal_failure_result(requested_operation)
        if len(set(res.selected_candidate_keys)) != len(res.selected_candidate_keys):
            return self._build_internal_failure_result(requested_operation)
        if any(key not in candidate_map for key in res.selected_candidate_keys):
            return self._build_internal_failure_result(requested_operation)
        if any(
            assessment.candidate_key not in candidate_map
            or not isfinite(assessment.confidence)
            or not 0.0 <= assessment.confidence <= 1.0
            for assessment in res.candidate_assessments
        ):
            return self._build_internal_failure_result(requested_operation)
        assessment_keys = [
            assessment.candidate_key for assessment in res.candidate_assessments
        ]
        if len(set(assessment_keys)) != len(assessment_keys):
            return self._build_internal_failure_result(requested_operation)
        for assessment in res.candidate_assessments:
            candidate_text = candidate_map[assessment.candidate_key].text
            if assessment.matches_target:
                if (
                    not assessment.matched_text
                    or assessment.matched_text not in candidate_text
                    or assessment.matched_fields != ("text",)
                ):
                    return self._build_internal_failure_result(
                        requested_operation
                    )
            elif assessment.matched_text or assessment.matched_fields:
                return self._build_internal_failure_result(requested_operation)

        if res.validation_result is ActionValidationResult.CLARIFY_MISSING_FIELDS:
            # FAIL is already non-executable. It needs a usable question and no
            # selected target, but incomplete assessment coverage must not turn
            # a safe refusal into an internal pipeline error.
            if res.selected_candidate_keys or not (
                res.clarification_question or ""
            ).strip():
                return self._build_internal_failure_result(requested_operation)
            return res

        if res.clarification_question != "":
            return self._build_internal_failure_result(requested_operation)
        if not first_model_response_ready:
            return self._build_internal_failure_result(requested_operation)
        if res.confidence < self.config.knowledge_llm_validation_min_confidence:
            return self._build_internal_failure_result(requested_operation)
        if set(assessment_keys) != set(candidate_map):
            return self._build_internal_failure_result(requested_operation)
        if requested_operation == "add":
            if res.selected_candidate_keys or any(
                assessment.matches_target
                and assessment.confidence
                >= self.config.knowledge_llm_validation_min_confidence
                for assessment in res.candidate_assessments
            ):
                return self._build_internal_failure_result(requested_operation)
        else:
            if len(res.selected_candidate_keys) != 1:
                return self._build_internal_failure_result(requested_operation)
            selected_key = res.selected_candidate_keys[0]
            selected_assessment = next(
                (
                    assessment
                    for assessment in res.candidate_assessments
                    if assessment.candidate_key == selected_key
                ),
                None,
            )
            if (
                selected_assessment is None
                or not selected_assessment.matches_target
                or not selected_assessment.action_compatible
                or selected_assessment.confidence
                < self.config.knowledge_llm_validation_min_confidence
            ):
                return self._build_internal_failure_result(requested_operation)
            strong_compatible_matches = {
                assessment.candidate_key
                for assessment in res.candidate_assessments
                if assessment.matches_target
                and assessment.action_compatible
                and assessment.confidence
                >= self.config.knowledge_llm_validation_min_confidence
            }
            if strong_compatible_matches != {selected_key}:
                return self._build_internal_failure_result(requested_operation)
            if (
                requested_operation == "delete"
                and " ".join(selected_assessment.matched_text.split())
                != " ".join(candidate_map[selected_key].text.split())
            ):
                return self._build_internal_failure_result(requested_operation)

        return res

    def _handle_failure(self) -> LLMRetrievalValidationResult | None:
        if self.config.knowledge_llm_validation_failure_policy == "fail_closed":
            return self._build_internal_failure_result("unknown")
        return None

    def _build_internal_failure_result(
        self,
        operation: str,
    ) -> LLMRetrievalValidationResult:
        return LLMRetrievalValidationResult(
            operation=operation,
            validation_result=ActionValidationResult.REJECT_UNSAFE_TRANSITION,
            selected_candidate_keys=(),
            confidence=0.0,
            ambiguous=False,
            reason_summary=(
                "Knowledge validation failed its strict response or evidence contract."
            ),
            candidate_assessments=(),
            should_execute=False,
            requires_hitl=False,
            factuality_concern=False,
            hitl_reason="internal_validation_failure",
            clarification_question=None,
        )

    def _schema(self) -> dict[str, Any]:
        return KNOWLEDGE_RETRIEVAL_VALIDATION_SCHEMA


class ReminderRetrievalValidationStrategy:
    def __init__(self, config: RetrievalValidationConfig, llm: LLMClient, prompts: PromptRegistry):
        self.config = config
        self.llm = llm
        self.prompts = prompts

    def validate(
        self,
        operation: str,
        user_query: str,
        rewritten_query: str,
        target_description: str,
        target_time_signals: Any,
        candidates: list[ReminderValidationCandidate],
    ) -> LLMRetrievalValidationResult | None:
        if not self.config.reminder_llm_validation_enabled:
            return None

        if operation == "add":
            return None

        if len(candidates) > self.config.reminder_llm_validation_max_candidates:
            return self._build_clarification_result()

        payloads = []
        candidate_map = {}
        for c in candidates:
            candidate_map[c.candidate_key] = c
            payloads.append(
                asdict(
                    LLMReminderCandidatePayload(
                        candidate_key=c.candidate_key,
                        subject=c.subject,
                        reminder_summary=c.reminder_summary,
                        reminder_time=c.reminder_time,
                        status=c.status,
                        deterministic_score=c.deterministic_score,
                        matched_fields=(),
                    )
                )
            )

        prompt_input = {
            "operation": operation,
            "rewritten_query": rewritten_query,
            "target_description": target_description,
            "target_time_signals": target_time_signals,
            "candidate_reminders": payloads,
            "validation_policy": {
                "min_confidence": self.config.reminder_llm_validation_min_confidence,
                "destructive_action_requires_unambiguous_target": self.config.destructive_action_requires_unambiguous_target,
            },
        }

        system_prompt = self.prompts.system("reminder_retrieval_validation")
        user_prompt = "Runtime context:\n" + json.dumps(
            inject_chat_history(prompt_input),
            indent=2,
            default=str,
        )

        try:
            raw_response = self.llm.generate_json(
                task=LLMTask.RETRIEVAL_VALIDATION,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema=self._schema(),
            )
            if is_structured_fallback(raw_response):
                return None
            return self._parse_and_validate_result(raw_response, candidate_map)
        except Exception as e:
            logger.error(f"Reminder LLM validation failed: {e}")
            return None  # Fallback to deterministic safely for ReminderBranch

    def _parse_and_validate_result(
        self, raw_response: dict[str, Any], candidate_map: dict[str, ReminderValidationCandidate]
    ) -> LLMRetrievalValidationResult | None:
        res = LLMRetrievalValidationResult(
            operation=raw_response["operation"],
            validation_result=ActionValidationResult(raw_response["validation_result"].lower()),
            selected_candidate_keys=tuple(raw_response["selected_candidate_keys"]),
            confidence=raw_response["confidence"],
            ambiguous=raw_response["ambiguous"],
            reason_summary=raw_response["reason_summary"],
            candidate_assessments=tuple(
                RetrievalCandidateAssessment(**a) for a in raw_response.get("candidate_assessments", [])
            ),
        )

        if res.validation_result == ActionValidationResult.EXECUTE:
            if res.ambiguous:
                return None
            if res.confidence < self.config.reminder_llm_validation_min_confidence:
                return None
            if not res.selected_candidate_keys:
                return None
            if len(res.selected_candidate_keys) != 1:
                return None
            for key in res.selected_candidate_keys:
                if key not in candidate_map:
                    return None

        if res.validation_result == ActionValidationResult.CLARIFY_AMBIGUOUS_TARGET:
            if not res.ambiguous:
                return None

        if res.validation_result == ActionValidationResult.SKIP_NOT_FOUND:
            if res.selected_candidate_keys:
                return None

        return res

    def _build_clarification_result(self) -> LLMRetrievalValidationResult:
        return LLMRetrievalValidationResult(
            operation="unknown",
            validation_result=ActionValidationResult.CLARIFY_AMBIGUOUS_TARGET,
            selected_candidate_keys=(),
            confidence=1.0,
            ambiguous=True,
            reason_summary="Fallback to clarification due to validation limits or failure.",
            candidate_assessments=(),
        )
