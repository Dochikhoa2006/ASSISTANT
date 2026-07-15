"""General response sub-branch detection and persistence planning."""

from __future__ import annotations

import logging
from typing import Optional

from .config import GeneralPurposeConfig
from .contracts import (
    GeneralResponsePersistencePlan,
    GeneralSubBranch,
    GeneralSubBranchDecision,
    PersistenceMode,
    PipelineContext,
)
from .llm import LLMClient, LLMTask
from .prompts import GENERAL_SUB_BRANCH_DETECTION_SCHEMA, PromptContext, PromptRegistry

logger = logging.getLogger(__name__)


class GeneralSubBranchDetector:
    def __init__(self, llm: LLMClient, prompt_registry: PromptRegistry) -> None:
        self.llm = llm
        self.prompt_registry = prompt_registry

    def detect(
        self,
        context: PipelineContext,
        config: GeneralPurposeConfig,
        merged_supporting_detail: str = "",
    ) -> GeneralSubBranchDecision:
        if not config.general_sub_branch_detector_enabled:
            fallback = GeneralSubBranch(config.general_sub_branch_fallback_mode)
            return GeneralSubBranchDecision(
                sub_branch=fallback,
                confidence=1.0,
                persistence_mode=PersistenceMode.CREATE_NEW_TOPIC,
                reason_summary="Detector disabled by config.",
            )

        approved_context = context.approved_conversation_context
        last_qa_state = context.last_qa_state
        support_question_rule_fired = bool(
            last_qa_state is not None
            and last_qa_state.supporting_questions
            and approved_context is not None
            and approved_context._internal_selected_hop_candidates
        )

        if support_question_rule_fired:
            selected_topic_id = last_qa_state.linked_topic_id
            selected_hop_id = last_qa_state.linked_hop_id
            return GeneralSubBranchDecision(
                sub_branch=GeneralSubBranch.SUPPORT_QUESTION_ANSWER,
                confidence=1.0,
                persistence_mode=PersistenceMode.APPEND_TO_EXISTING_TOPIC,
                selected_topic_id=selected_topic_id,
                selected_hop_id=selected_hop_id,
                selected_parent_hop_id=selected_hop_id,
                reason_summary="Deterministic rule fired: SUPPORT_QUESTION_ANSWER.",
            )

        if approved_context is not None and approved_context.approved_conversation_history:
            selected_topic_id = (
                approved_context._internal_selected_topic_candidates[0]
                if approved_context._internal_selected_topic_candidates
                else None
            )
            selected_hop_id = (
                approved_context._internal_selected_hop_candidates[0]
                if approved_context._internal_selected_hop_candidates
                else None
            )
            return GeneralSubBranchDecision(
                sub_branch=GeneralSubBranch.CONVERSATION_FOLLOW_UP,
                confidence=1.0,
                persistence_mode=PersistenceMode.APPEND_TO_EXISTING_TOPIC,
                selected_topic_id=selected_topic_id,
                selected_hop_id=selected_hop_id,
                selected_parent_hop_id=selected_hop_id,
                reason_summary="Deterministic rule fired: CONVERSATION_FOLLOW_UP.",
            )

        return GeneralSubBranchDecision(
            sub_branch=GeneralSubBranch.NEW_CONVERSATION_TOPIC,
            confidence=1.0,
            persistence_mode=PersistenceMode.CREATE_NEW_TOPIC,
            reason_summary="Deterministic rule fired: NEW_CONVERSATION_TOPIC.",
        )


class GeneralSubBranchValidator:
    def validate(
        self, decision: GeneralSubBranchDecision, context: PipelineContext, config: GeneralPurposeConfig
    ) -> GeneralSubBranchDecision:
        if decision.sub_branch == GeneralSubBranch.CONVERSATION_FOLLOW_UP:
            if not decision.selected_topic_id or not decision.selected_hop_id:
                return GeneralSubBranchDecision(
                    sub_branch=GeneralSubBranch(config.general_sub_branch_fallback_mode),
                    confidence=decision.confidence,
                    persistence_mode=PersistenceMode.CREATE_NEW_TOPIC,
                    reason_summary="Missing required topic_id/hop_id for follow-up.",
                )
        elif decision.sub_branch == GeneralSubBranch.SUPPORT_QUESTION_ANSWER:
            if not context.last_qa_state or not context.last_qa_state.linked_topic_id or not context.last_qa_state.linked_hop_id:
                if not decision.selected_topic_id or not decision.selected_hop_id:
                    return GeneralSubBranchDecision(
                        sub_branch=GeneralSubBranch(config.general_sub_branch_fallback_mode),
                        confidence=decision.confidence,
                        persistence_mode=PersistenceMode.CREATE_NEW_TOPIC,
                        reason_summary="Missing required context for support question answer.",
                    )
        return decision


class GeneralPersistencePlanBuilder:
    def build_plan(
        self, decision: GeneralSubBranchDecision, context: PipelineContext, config: GeneralPurposeConfig
    ) -> GeneralResponsePersistencePlan:
        topic_id: Optional[str] = None
        previous_hop_id: Optional[str] = None
        parent_hop_id: Optional[str] = None
        mode = decision.persistence_mode
        sub_branch = decision.sub_branch

        if mode == PersistenceMode.CREATE_NEW_TOPIC:
            pass
        elif mode == PersistenceMode.APPEND_TO_EXISTING_TOPIC:
            if sub_branch == GeneralSubBranch.SUPPORT_QUESTION_ANSWER and context.last_qa_state and context.last_qa_state.linked_topic_id:
                topic_id = context.last_qa_state.linked_topic_id
                previous_hop_id = context.last_qa_state.linked_hop_id
                parent_hop_id = context.last_qa_state.linked_hop_id
            else:
                topic_id = decision.selected_topic_id
                previous_hop_id = decision.selected_hop_id
                parent_hop_id = decision.selected_parent_hop_id
        elif mode == PersistenceMode.BRANCH_FROM_EXISTING_HOP:
            topic_id = decision.selected_topic_id
            parent_hop_id = decision.selected_parent_hop_id

        return GeneralResponsePersistencePlan(
            persistence_mode=mode,
            sub_branch=sub_branch,
            topic_id=topic_id,
            previous_hop_id=previous_hop_id,
            parent_hop_id=parent_hop_id,
            reason_summary=decision.reason_summary,
        )
