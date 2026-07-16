"""LLM-Based Retrieval Validation Strategies."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, replace
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
from .llm import LLMClient, LLMTask
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
    ) -> LLMRetrievalValidationResult | None:
        if not self.config.knowledge_llm_validation_enabled:
            return None

        if operation not in {"add", "delete", "modify"}:
            return self._build_clarification_result(operation=operation)

        candidates = candidates[
            : self.config.knowledge_llm_validation_max_candidates
        ]

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
                raw_query=user_query,
                rewritten_query=rewritten_query,
                intent="knowledge_facts",
                chat_history=chat_history,
                extra={
                    "operation": operation,
                    "target_description": target_description,
                    "proposed_content": proposed_content,
                    "replacement_content": replacement_content,
                    "candidate_chunks": payloads,
                    "validation_policy": {
                        "min_confidence": self.config.knowledge_llm_validation_min_confidence,
                        "destructive_action_requires_unambiguous_target": self.config.destructive_action_requires_unambiguous_target,
                    },
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
            return self._parse_and_validate_result(
                raw_response,
                candidate_map,
                requested_operation=operation,
            )
        except Exception as e:
            logger.error(f"Knowledge LLM validation failed: {e}")
            return self._handle_failure()

    def _parse_and_validate_result(
        self,
        raw_response: dict[str, Any],
        candidate_map: dict[str, KnowledgeValidationCandidate],
        *,
        requested_operation: str,
    ) -> LLMRetrievalValidationResult:
        try:
            assessments = tuple(
                RetrievalCandidateAssessment(
                    candidate_key=str(item["candidate_key"]),
                    matches_target=bool(item["matches_target"]),
                    action_compatible=bool(item["action_compatible"]),
                    confidence=float(item["confidence"]),
                    matched_fields=tuple(str(value) for value in item["matched_fields"]),
                    reason_summary=str(item["reason_summary"]),
                    matched_text=str(item["matched_text"]),
                )
                for item in raw_response.get("candidate_assessments", [])
            )
            res = LLMRetrievalValidationResult(
                operation=str(raw_response["operation"]).lower(),
                validation_result=ActionValidationResult(
                    str(raw_response["validation_result"]).lower()
                ),
                selected_candidate_keys=tuple(
                    str(value) for value in raw_response["selected_candidate_keys"]
                ),
                confidence=float(raw_response["confidence"]),
                ambiguous=bool(raw_response["ambiguous"]),
                reason_summary=str(raw_response["reason_summary"]),
                candidate_assessments=assessments,
                should_execute=bool(raw_response["should_execute"]),
                requires_hitl=bool(raw_response["requires_hitl"]),
                factuality_concern=bool(raw_response["factuality_concern"]),
            )
        except (KeyError, TypeError, ValueError):
            return self._build_clarification_result(operation=requested_operation)

        if res.operation != requested_operation:
            return self._build_clarification_result(operation=requested_operation)
        if not isfinite(res.confidence) or not 0.0 <= res.confidence <= 1.0:
            return self._build_clarification_result(operation=requested_operation)
        if res.confidence < self.config.knowledge_llm_validation_min_confidence:
            return self._build_clarification_result(operation=requested_operation)
        if len(set(res.selected_candidate_keys)) != len(res.selected_candidate_keys):
            return self._build_clarification_result(operation=requested_operation)
        if any(key not in candidate_map for key in res.selected_candidate_keys):
            return self._build_clarification_result(operation=requested_operation)
        if any(
            assessment.candidate_key not in candidate_map
            or not isfinite(assessment.confidence)
            or not 0.0 <= assessment.confidence <= 1.0
            for assessment in res.candidate_assessments
        ):
            return self._build_clarification_result(operation=requested_operation)
        assessment_keys = [
            assessment.candidate_key for assessment in res.candidate_assessments
        ]
        if (
            len(set(assessment_keys)) != len(assessment_keys)
            or set(assessment_keys) != set(candidate_map)
        ):
            return self._build_clarification_result(operation=requested_operation)
        for assessment in res.candidate_assessments:
            candidate_text = candidate_map[assessment.candidate_key].text
            if assessment.matches_target:
                if (
                    not assessment.matched_text
                    or assessment.matched_text not in candidate_text
                ):
                    return self._build_clarification_result(
                        operation=requested_operation
                    )
            elif assessment.matched_text:
                return self._build_clarification_result(
                    operation=requested_operation
                )

        is_execute = res.validation_result == ActionValidationResult.EXECUTE
        is_clarification = res.validation_result in {
            ActionValidationResult.CLARIFY_AMBIGUOUS_TARGET,
            ActionValidationResult.CLARIFY_MISSING_FIELDS,
        }
        if res.should_execute != is_execute or res.requires_hitl != is_clarification:
            return self._build_clarification_result(operation=requested_operation)
        if res.factuality_concern and not is_clarification:
            return self._build_clarification_result(operation=requested_operation)
        if res.factuality_concern:
            res = replace(res, hitl_reason="factuality_concern")

        allowed_results = (
            {
                ActionValidationResult.EXECUTE,
                ActionValidationResult.SKIP_ALREADY_EXISTS,
                ActionValidationResult.CLARIFY_AMBIGUOUS_TARGET,
                ActionValidationResult.CLARIFY_MISSING_FIELDS,
            }
            if requested_operation == "add"
            else {
                ActionValidationResult.EXECUTE,
                ActionValidationResult.SKIP_NOT_FOUND,
                ActionValidationResult.CLARIFY_AMBIGUOUS_TARGET,
                ActionValidationResult.CLARIFY_MISSING_FIELDS,
            }
        )
        if res.validation_result not in allowed_results:
            return self._build_clarification_result(operation=requested_operation)

        if is_execute:
            if res.ambiguous:
                return self._build_clarification_result(operation=requested_operation)
            if requested_operation == "add":
                if res.selected_candidate_keys or any(
                    assessment.matches_target
                    and assessment.confidence
                    >= self.config.knowledge_llm_validation_min_confidence
                    for assessment in res.candidate_assessments
                ):
                    return self._build_clarification_result(operation=requested_operation)
            else:
                if len(res.selected_candidate_keys) != 1:
                    return self._build_clarification_result(operation=requested_operation)
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
                    or candidate_map[selected_key].text.count(
                        selected_assessment.matched_text
                    ) != 1
                ):
                    return self._build_clarification_result(operation=requested_operation)
                strong_compatible_matches = {
                    assessment.candidate_key
                    for assessment in res.candidate_assessments
                    if assessment.matches_target
                    and assessment.action_compatible
                    and assessment.confidence
                    >= self.config.knowledge_llm_validation_min_confidence
                }
                if strong_compatible_matches != {selected_key}:
                    return self._build_clarification_result(
                        operation=requested_operation
                    )
                if (
                    requested_operation == "delete"
                    and " ".join(selected_assessment.matched_text.split())
                    != " ".join(candidate_map[selected_key].text.split())
                ):
                    return self._build_clarification_result(
                        operation=requested_operation,
                        reason_summary=(
                            "A partial-chunk delete would remove unrelated knowledge."
                        ),
                        hitl_reason="partial_chunk_delete",
                    )

        if res.validation_result == ActionValidationResult.CLARIFY_AMBIGUOUS_TARGET:
            if not res.ambiguous:
                return self._build_clarification_result(operation=requested_operation)

        if res.validation_result == ActionValidationResult.SKIP_NOT_FOUND:
            if res.selected_candidate_keys or any(
                assessment.matches_target
                and assessment.confidence
                >= self.config.knowledge_llm_validation_min_confidence
                for assessment in res.candidate_assessments
            ):
                return self._build_clarification_result(operation=requested_operation)

        if res.validation_result == ActionValidationResult.SKIP_ALREADY_EXISTS:
            if requested_operation != "add" or not any(
                assessment.matches_target
                and assessment.confidence
                >= self.config.knowledge_llm_validation_min_confidence
                for assessment in res.candidate_assessments
            ):
                return self._build_clarification_result(operation=requested_operation)

        return res

    def _handle_failure(self) -> LLMRetrievalValidationResult | None:
        if self.config.knowledge_llm_validation_failure_policy == "fail_closed":
            return self._build_clarification_result()
        return None

    def _build_clarification_result(
        self,
        *,
        operation: str = "unknown",
        reason_summary: str = (
            "Fallback to clarification due to validation limits or failure."
        ),
        hitl_reason: str | None = None,
    ) -> LLMRetrievalValidationResult:
        return LLMRetrievalValidationResult(
            operation=operation,
            validation_result=ActionValidationResult.CLARIFY_AMBIGUOUS_TARGET,
            selected_candidate_keys=(),
            confidence=1.0,
            ambiguous=True,
            reason_summary=reason_summary,
            candidate_assessments=(),
            should_execute=False,
            requires_hitl=True,
            factuality_concern=False,
            hitl_reason=hitl_reason,
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
            "user_query": user_query,
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
