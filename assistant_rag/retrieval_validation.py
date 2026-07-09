"""LLM-Based Retrieval Validation Strategies."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
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
from .llm import LLMClient
from .prompts import PromptRegistry

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
    ) -> LLMRetrievalValidationResult | None:
        if not self.config.knowledge_llm_validation_enabled:
            return None

        if operation == "add":
            return None  # Validation never called for add

        if len(candidates) > self.config.knowledge_llm_validation_max_candidates:
            # Enforce max-candidate limits
            return self._build_clarification_result()

        payloads = []
        candidate_map = {}
        for c in candidates:
            candidate_map[c.candidate_key] = c
            payloads.append(
                asdict(
                    LLMKnowledgeCandidatePayload(
                        candidate_key=c.candidate_key,
                        text_excerpt=c.text[:500],  # truncate safely
                        source_title=c.source_title,
                        retrieval_score=c.retrieval_score,
                        rerank_score=c.rerank_score,
                        matched_fields=(),
                    )
                )
            )

        prompt_input = {
            "operation": operation,
            "user_query": user_query,
            "rewritten_query": rewritten_query,
            "target_description": target_description,
            "candidate_chunks": payloads,
            "validation_policy": {
                "min_confidence": self.config.knowledge_llm_validation_min_confidence,
                "destructive_action_requires_unambiguous_target": self.config.destructive_action_requires_unambiguous_target,
            },
        }

        system_prompt = self.prompts.system("knowledge_retrieval_validation")
        user_prompt = "Runtime context:\n" + json.dumps(prompt_input, indent=2)

        try:
            raw_response = self.llm.generate_structured(
                model=self.config.knowledge_llm_validation_model,
                system=system_prompt,
                prompt=user_prompt,
                schema_name="knowledge_retrieval_validation",
                retry_count=self.config.knowledge_llm_validation_json_retry_count,
            )
            return self._parse_and_validate_result(raw_response, candidate_map)
        except Exception as e:
            logger.error(f"Knowledge LLM validation failed: {e}")
            return self._handle_failure()

    def _parse_and_validate_result(
        self, raw_response: dict[str, Any], candidate_map: dict[str, KnowledgeValidationCandidate]
    ) -> LLMRetrievalValidationResult:
        res = LLMRetrievalValidationResult(
            operation=raw_response["operation"],
            validation_result=ActionValidationResult(raw_response["validation_result"].lower()),
            selected_candidate_keys=tuple(raw_response["selected_candidate_keys"]),
            confidence=raw_response["confidence"],
            ambiguous=raw_response["ambiguous"],
            reason_summary=raw_response["reason_summary"],
            candidate_assessments=tuple(
                RetrievalCandidateAssessment(**a, action_compatible=True)
                for a in raw_response.get("candidate_assessments", [])
            ),
        )

        if res.validation_result == ActionValidationResult.EXECUTE:
            if res.ambiguous:
                return self._handle_failure()
            if res.confidence < self.config.knowledge_llm_validation_min_confidence:
                return self._handle_failure()
            if not res.selected_candidate_keys:
                return self._handle_failure()
            if len(res.selected_candidate_keys) != 1:
                return self._handle_failure()
            for key in res.selected_candidate_keys:
                if key not in candidate_map:
                    return self._handle_failure()

        if res.validation_result == ActionValidationResult.CLARIFY_AMBIGUOUS_TARGET:
            if not res.ambiguous:
                return self._handle_failure()

        if res.validation_result == ActionValidationResult.SKIP_NOT_FOUND:
            if res.selected_candidate_keys:
                return self._handle_failure()

        return res

    def _handle_failure(self) -> LLMRetrievalValidationResult | None:
        if self.config.knowledge_llm_validation_failure_policy == "fail_closed":
            return self._build_clarification_result()
        return None

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
        user_prompt = "Runtime context:\n" + json.dumps(prompt_input, indent=2, default=str)

        try:
            raw_response = self.llm.generate_structured(
                model=self.config.reminder_llm_validation_model,
                system=system_prompt,
                prompt=user_prompt,
                schema_name="reminder_retrieval_validation",
                retry_count=self.config.reminder_llm_validation_json_retry_count,
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
