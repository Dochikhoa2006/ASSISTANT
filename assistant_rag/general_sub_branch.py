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

        history_map = {}
        human_qs = []
        reminder_qs = []
        extracted_types = []
        
        if context.chat_history:
            for idx, hop in enumerate(context.chat_history):
                history_map[f"conversation_candidate_{idx}"] = {
                    "role": "conversation_hop",
                    "text": hop.get("text") or (
                        f"User: {hop.get('raw_user_query', '')}\n"
                        f"Assistant: {hop.get('raw_response', '')}"
                    ).strip(),
                    "topic_id": hop.get("topic_id"),
                    "hop_id": hop.get("hop_id"),
                }
            human_qs = [q.text for q in context.approved_conversation_context.human_supporting_questions]
            reminder_qs = [q.text for q in context.approved_conversation_context.reminder_supporting_questions]
            extracted_types = [t.value for t in context.approved_conversation_context.extracted_expected_response_types]

        try:
            payload = self.llm.generate_json(
                task=LLMTask.GENERAL_SUB_BRANCH_DETECTION,
                system_prompt=self.prompt_registry.system("general_sub_branch_detector"),
                user_prompt=self.prompt_registry.user(
                    PromptContext(
                        stage="general_sub_branch_detector",
                        rewritten_query=context.rewritten_query,
                        extra={
                            "approved_conversation_history": history_map,
                            "human_supporting_questions": human_qs,
                            "reminder_supporting_questions": reminder_qs,
                            "extracted_expected_response_types": extracted_types,
                            "merged_supporting_detail": merged_supporting_detail,
                        },
                    )
                ),
                schema=GENERAL_SUB_BRANCH_DETECTION_SCHEMA,
            )
            
            sub_branch_val = str(payload.get("sub_branch", config.general_sub_branch_fallback_mode))
            try:
                sub_branch = GeneralSubBranch(sub_branch_val)
            except ValueError:
                sub_branch = GeneralSubBranch(config.general_sub_branch_fallback_mode)

            try:
                persistence_val = str(payload.get("persistence_mode", "create_new_topic"))
                persistence_mode = PersistenceMode(persistence_val)
            except ValueError:
                persistence_mode = PersistenceMode.CREATE_NEW_TOPIC

            confidence = float(payload.get("confidence", 0.0))
            if confidence < config.general_sub_branch_confidence_threshold:
                sub_branch = GeneralSubBranch(config.general_sub_branch_fallback_mode)
                persistence_mode = PersistenceMode.CREATE_NEW_TOPIC

            selected_ref = payload.get("selected_candidate_ref")
            selected_topic_id = None
            selected_hop_id = None
            
            if selected_ref and selected_ref in history_map:
                selected_topic_id = history_map[selected_ref].get("topic_id")
                selected_hop_id = history_map[selected_ref].get("hop_id")

            return GeneralSubBranchDecision(
                sub_branch=sub_branch,
                confidence=confidence,
                persistence_mode=persistence_mode,
                selected_candidate_ref=selected_ref,
                selected_topic_id=selected_topic_id,
                selected_hop_id=selected_hop_id,
                selected_parent_hop_id=selected_hop_id,
                reason_summary=str(payload.get("reason_summary", "")),
                risk_flags=tuple(payload.get("risk_flags", [])),
                missing_context=tuple(payload.get("missing_context", [])),
            )
        except Exception as e:
            logger.debug("General sub-branch detection failed: %s", e)
            fallback = GeneralSubBranch(config.general_sub_branch_fallback_mode)
            return GeneralSubBranchDecision(
                sub_branch=fallback,
                confidence=0.0,
                persistence_mode=PersistenceMode.CREATE_NEW_TOPIC,
                reason_summary=f"LLM failure: {e}",
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
