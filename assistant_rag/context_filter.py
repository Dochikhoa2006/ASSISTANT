"""Context filtering stages for General Response."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Protocol, Literal

from .contracts import (
    Intent, RetrievalResult, ApprovedConversationContext,
    GeneratedQuestion, QuestionSource, ExpectedResponseType
)
from .config import ContextFilterConfig
from .llm import LLMClient
from .prompts import PromptRegistry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ApprovedContext:
    knowledge_evidence: list[str]
    reminder_context: list[dict[str, Any]]
    approved_conversation_history: list[dict[str, Any]]
    rejected_knowledge_ids: list[str]
    rejected_reminder_ids: list[str]
    rejected_conversation_ids: list[str]


class ContextFilter(Protocol):
    def filter(
        self,
        *,
        user_id: str,
        knowledge_results: list[RetrievalResult],
        reminder_results: list[dict[str, Any]],
        conversation_results: list[RetrievalResult],
        query: str,
        intent: Intent,
        linked_topic_id: str | None = None,
    ) -> ApprovedContext:
        ...

    def filter_conversation_only(
        self,
        *,
        user_id: str,
        conversation_results: list[RetrievalResult],
        query: str,
        config: ContextFilterConfig,
    ) -> ApprovedConversationContext:
        ...


@dataclass
class HardRuleContextFilter:
    allowed_reminder_statuses: tuple[str, ...]
    reminder_approved_max_items: int
    reminder_min_confidence: float

    def filter(
        self,
        *,
        user_id: str,
        knowledge_results: list[RetrievalResult],
        reminder_results: list[dict[str, Any]],
        conversation_results: list[RetrievalResult],
        query: str,
        intent: Intent,
        linked_topic_id: str | None = None,
    ) -> ApprovedContext:
        approved_knowledge: list[str] = []
        rejected_knowledge_ids: list[str] = []
        
        # Knowledge filtering
        seen_knowledge_ids: set[str] = set()
        for result in knowledge_results:
            payload = result.payload
            payload_user_id = payload.get("user_id")
            if payload_user_id != user_id:
                rejected_knowledge_ids.append(result.entity_id)
                continue
                
            if payload.get("is_deleted", False):
                rejected_knowledge_ids.append(result.entity_id)
                continue
                
            if result.entity_id in seen_knowledge_ids:
                rejected_knowledge_ids.append(result.entity_id)
                continue
            seen_knowledge_ids.add(result.entity_id)
            approved_knowledge.append(str(payload.get("text", "")))

        # Reminder filtering
        approved_reminders: list[dict[str, Any]] = []
        rejected_reminder_ids: list[str] = []
        seen_reminder_ids = set()
        
        for reminder in reminder_results:
            reminder_id = str(reminder.get("reminder_id", ""))
            
            if reminder.get("user_id") != user_id:
                rejected_reminder_ids.append(reminder_id)
                continue
                
            status = reminder.get("status")
            if status not in self.allowed_reminder_statuses:
                rejected_reminder_ids.append(reminder_id)
                continue

            confidence = float(reminder.get("confidence", 1.0))
            if confidence < self.reminder_min_confidence:
                rejected_reminder_ids.append(reminder_id)
                continue
                
            subject = str(reminder.get("subject") or "").strip()
            summary = str(reminder.get("reminder_summary") or "").strip()
            if not subject and not summary:
                rejected_reminder_ids.append(reminder_id)
                continue
                
            if reminder_id in seen_reminder_ids:
                rejected_reminder_ids.append(reminder_id)
                continue
                
            seen_reminder_ids.add(reminder_id)
            approved_reminders.append(reminder)
            
            if len(approved_reminders) >= self.reminder_approved_max_items:
                break

        # Conversation filtering
        approved_conversation: list[dict[str, Any]] = []
        rejected_conversation_ids: list[str] = []
        seen_conversation_ids: set[str] = set()
        for result in conversation_results:
            payload = result.payload
            payload_user_id = payload.get("user_id")
            if payload_user_id != user_id:
                rejected_conversation_ids.append(result.entity_id)
                continue
            if payload.get("is_deleted", False):
                rejected_conversation_ids.append(result.entity_id)
                continue
            if result.entity_id in seen_conversation_ids:
                rejected_conversation_ids.append(result.entity_id)
                continue
            seen_conversation_ids.add(result.entity_id)
            approved_conversation.append(payload)

        return ApprovedContext(
            knowledge_evidence=approved_knowledge,
            reminder_context=approved_reminders,
            approved_conversation_history=approved_conversation,
            rejected_knowledge_ids=rejected_knowledge_ids,
            rejected_reminder_ids=rejected_reminder_ids,
            rejected_conversation_ids=rejected_conversation_ids,
        )

    def filter_conversation_only(
        self,
        *,
        user_id: str,
        conversation_results: list[RetrievalResult],
        query: str,
        config: ContextFilterConfig,
    ) -> ApprovedConversationContext:
        approved_conversation: list[dict[str, Any]] = []
        rejected_conversation_ids: list[str] = []
        human_supporting_questions: list[GeneratedQuestion] = []
        reminder_supporting_questions: list[GeneratedQuestion] = []
        clarification_question_context: GeneratedQuestion | None = None
        extracted_expected_response_types: list[ExpectedResponseType] = []
        
        _internal_selected_topic_candidates: list[str] = []
        _internal_selected_hop_candidates: list[str] = []
        approved_rerank_scores: list[float] = []
        
        seen_conversation_ids: set[str] = set()
        for result in conversation_results:
            payload = result.payload
            payload_user_id = payload.get("user_id")
            if payload_user_id != user_id:
                rejected_conversation_ids.append(result.entity_id)
                continue
            if payload.get("is_deleted", False):
                rejected_conversation_ids.append(result.entity_id)
                continue
            if result.entity_id in seen_conversation_ids:
                rejected_conversation_ids.append(result.entity_id)
                continue
            seen_conversation_ids.add(result.entity_id)
            approved_rerank_scores.append(float(result.rerank_score))

            # Extract native expected_response_type from the hop if available
            raw_hop_ert = payload.get("expected_response_type")
            if raw_hop_ert:
                try:
                    hop_ert = ExpectedResponseType(raw_hop_ert)
                    if hop_ert not in extracted_expected_response_types:
                        extracted_expected_response_types.append(hop_ert)
                    payload["expected_response_type"] = hop_ert.value
                except ValueError:
                    payload["expected_response_type"] = ExpectedResponseType.UNKNOWN.value

            # Process supporting questions JSON
            supp_questions_json = payload.get("supporting_questions_json")
            
            if supp_questions_json:
                try:
                    sq_list = json.loads(supp_questions_json) if isinstance(supp_questions_json, str) else supp_questions_json
                    if isinstance(sq_list, list):
                        for sq in sq_list:
                            q_source = sq.get("question_source", "none")
                            raw_ert = sq.get("expected_response_type")
                            
                            ert = ExpectedResponseType.UNKNOWN
                            if raw_ert:
                                try:
                                    ert = ExpectedResponseType(raw_ert)
                                except ValueError:
                                    ert = ExpectedResponseType.UNKNOWN
                            else:
                                ert = ExpectedResponseType.UNKNOWN
                                
                            q_obj = GeneratedQuestion(
                                text=sq.get("question_text", ""),
                                source=QuestionSource(q_source) if q_source in [e.value for e in QuestionSource] else QuestionSource.NONE,
                                purpose=sq.get("purpose", ""),
                                confidence=float(sq.get("confidence", 1.0)),
                                should_ask=sq.get("should_ask", True),
                                expected_response_type=ert
                            )
                            
                            if q_obj.source == QuestionSource.HUMAN_SUPPORTING_QUESTION:
                                human_supporting_questions.append(q_obj)
                            elif q_obj.source == QuestionSource.REMINDER_SUPPORTING_QUESTION:
                                reminder_supporting_questions.append(q_obj)
                            elif q_obj.source == QuestionSource.CLARIFICATION_QUESTION:
                                clarification_question_context = q_obj
                                
                            if ert not in extracted_expected_response_types:
                                extracted_expected_response_types.append(ert)
                                
                            sq["expected_response_type"] = ert.value
                            
                        # Replace stringified JSON with the structured parsed list for LLM visibility
                        payload["supporting_questions"] = sq_list
                        if "supporting_questions_json" in payload:
                            del payload["supporting_questions_json"]
                                
                except Exception as e:
                    logger.warning(f"Failed to parse supporting_questions_json: {e}")
                    
            approved_conversation.append(payload)
            
            topic_id = payload.get("topic_id")
            if topic_id and topic_id not in _internal_selected_topic_candidates:
                _internal_selected_topic_candidates.append(topic_id)
                
            hop_id = payload.get("hop_id")
            if hop_id and hop_id not in _internal_selected_hop_candidates:
                _internal_selected_hop_candidates.append(hop_id)
            
        status: Literal["not_run", "approved", "all_rejected", "empty"] = "approved"
        if not conversation_results:
            status = "empty"
        elif not approved_conversation:
            status = "all_rejected"

        return ApprovedConversationContext(
            approved_conversation_history=approved_conversation,
            human_supporting_questions=human_supporting_questions,
            reminder_supporting_questions=reminder_supporting_questions,
            clarification_question_context=clarification_question_context,
            extracted_expected_response_types=extracted_expected_response_types,
            conversation_retrieval_ran=True,
            conversation_context_status=status,
            approved_conversation_count=len(approved_conversation),
            top_hop_rerank_score=(
                max(approved_rerank_scores) if approved_rerank_scores else None
            ),
            _internal_selected_topic_candidates=_internal_selected_topic_candidates,
            _internal_selected_hop_candidates=_internal_selected_hop_candidates,
            _rejected_conversation_ids=tuple(rejected_conversation_ids),
        )


@dataclass
class LLMSemanticContextJudge:
    llm: LLMClient
    prompt_registry: PromptRegistry

    def filter(
        self,
        hard_rule_context: ApprovedContext,
        query: str,
        conversation_results: list[RetrievalResult] = [],
    ) -> ApprovedContext:
        # Evaluates the hard-rule approved context for semantic usefulness/relevance
        # to the specific query. 
        # MUST NEVER override deterministic safety rules.
        # MUST NEVER mutate facts or invent new context.
        # For this version, it's a passthrough.
        return hard_rule_context


@dataclass
class TwoLayerContextFilter:
    hard_rule_filter: HardRuleContextFilter
    llm_judge: LLMSemanticContextJudge | None = None

    def filter(
        self,
        *,
        user_id: str,
        knowledge_results: list[RetrievalResult],
        reminder_results: list[dict[str, Any]],
        conversation_results: list[RetrievalResult],
        query: str,
        intent: Intent,
        linked_topic_id: str | None = None,
    ) -> ApprovedContext:
        # Layer 1: Strict deterministic filtering
        safe_context = self.hard_rule_filter.filter(
            user_id=user_id,
            knowledge_results=knowledge_results,
            reminder_results=reminder_results,
            conversation_results=conversation_results,
            query=query,
            intent=intent,
            linked_topic_id=linked_topic_id,
        )
        
        # Layer 2: Optional semantic judgement (receives only safe context)
        if self.llm_judge:
            return self.llm_judge.filter(safe_context, query, conversation_results=conversation_results)
            
        return safe_context

    def filter_conversation_only(
        self,
        *,
        user_id: str,
        conversation_results: list[RetrievalResult],
        query: str,
        config: ContextFilterConfig,
    ) -> ApprovedConversationContext:
        return self.hard_rule_filter.filter_conversation_only(
            user_id=user_id,
            conversation_results=conversation_results,
            query=query,
            config=config,
        )
