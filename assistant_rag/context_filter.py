"""Context filtering stages for General Response."""

from __future__ import annotations

import json
import logging
import re
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


def _normalize_text(text: str) -> str:
    return re.sub(r'\s+', ' ', text.lower().strip())


def _text_similarity(t1: str, t2: str) -> float:
    # A simple token overlap similarity for near-duplicate detection
    tokens1 = set(t1.split())
    tokens2 = set(t2.split())
    if not tokens1 or not tokens2:
        return 0.0
    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    return intersection / union


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
    knowledge_min_confidence: float
    allowed_reminder_statuses: tuple[str, ...]
    conversation_min_confidence: float
    conversation_approved_max_items: int
    conversation_duplicate_threshold: float
    knowledge_approved_max_items: int
    knowledge_duplicate_threshold: float
    low_information_text_patterns: tuple[str, ...]
    reminder_approved_max_items: int
    reminder_min_confidence: float
    low_information_min_chars: int

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
        for result in knowledge_results:
            payload = result.payload
            
            if result.entity_type != "knowledge_chunk":
                rejected_knowledge_ids.append(result.entity_id)
                continue
                
            payload_user_id = payload.get("user_id")
            if payload_user_id and payload_user_id != user_id:
                rejected_knowledge_ids.append(result.entity_id)
                continue
                
            if payload.get("is_deleted", False):
                rejected_knowledge_ids.append(result.entity_id)
                continue
                
            if result.confidence < self.knowledge_min_confidence:
                rejected_knowledge_ids.append(result.entity_id)
                continue
                
            text = payload.get("text", "").strip()
            if not text:
                rejected_knowledge_ids.append(result.entity_id)
                continue
                
            if len(text) < self.low_information_min_chars:
                rejected_knowledge_ids.append(result.entity_id)
                continue

            norm_text = _normalize_text(text)
            is_duplicate = False
            for existing in approved_knowledge:
                if _text_similarity(norm_text, _normalize_text(existing)) >= self.knowledge_duplicate_threshold:
                    is_duplicate = True
                    break
            
            if is_duplicate:
                rejected_knowledge_ids.append(result.entity_id)
                continue
                
            approved_knowledge.append(str(text))
            
            if len(approved_knowledge) >= self.knowledge_approved_max_items:
                break

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
        
        norm_query = _normalize_text(query)
        low_info_patterns = {p.casefold() for p in self.low_information_text_patterns}
        
        for result in conversation_results:
            payload = result.payload
            
            if result.entity_type != "conversation_hop":
                rejected_conversation_ids.append(result.entity_id)
                continue
                
            payload_user_id = payload.get("user_id")
            if payload_user_id and payload_user_id != user_id:
                rejected_conversation_ids.append(result.entity_id)
                continue
                
            if result.confidence < self.conversation_min_confidence:
                rejected_conversation_ids.append(result.entity_id)
                continue
                
            text = payload.get("text", "").strip()
            if not text:
                rejected_conversation_ids.append(result.entity_id)
                continue
                
            norm_text = _normalize_text(text)
            if norm_text == norm_query:
                rejected_conversation_ids.append(result.entity_id)
                continue
                
            if norm_text in low_info_patterns:
                rejected_conversation_ids.append(result.entity_id)
                continue
                
            is_duplicate = False
            for existing in approved_conversation:
                existing_text = existing.get("text", "")
                if _text_similarity(norm_text, _normalize_text(existing_text)) >= self.conversation_duplicate_threshold:
                    is_duplicate = True
                    break
                    
            if is_duplicate:
                rejected_conversation_ids.append(result.entity_id)
                continue
                
            approved_conversation.append(payload)
            
            if len(approved_conversation) >= self.conversation_approved_max_items:
                break

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
        
        norm_query = _normalize_text(query)
        low_info_patterns = {p.casefold() for p in self.low_information_text_patterns}
        
        for result in conversation_results:
            payload = result.payload
            
            if result.entity_type != "conversation_hop":
                rejected_conversation_ids.append(result.entity_id)
                continue
                
            payload_user_id = payload.get("user_id")
            if payload_user_id and payload_user_id != user_id:
                rejected_conversation_ids.append(result.entity_id)
                continue
                
            if result.confidence < config.conversation_min_confidence:
                rejected_conversation_ids.append(result.entity_id)
                continue
                
            text = payload.get("text", "").strip()
            if not text:
                rejected_conversation_ids.append(result.entity_id)
                continue
                
            norm_text = _normalize_text(text)
            if norm_text == norm_query:
                rejected_conversation_ids.append(result.entity_id)
                continue
                
            if norm_text in low_info_patterns:
                rejected_conversation_ids.append(result.entity_id)
                continue
                
            is_duplicate = False
            for existing in approved_conversation:
                existing_text = existing.get("text", "")
                if _text_similarity(norm_text, _normalize_text(existing_text)) >= config.conversation_duplicate_threshold:
                    is_duplicate = True
                    break
                    
            if is_duplicate:
                rejected_conversation_ids.append(result.entity_id)
                continue

            # Extract native expected_response_type from the hop if available
            raw_hop_ert = payload.get("expected_response_type")
            if raw_hop_ert:
                try:
                    hop_ert = ExpectedResponseType(raw_hop_ert)
                    if hop_ert not in extracted_expected_response_types:
                        extracted_expected_response_types.append(hop_ert)
                    payload["expected_response_type"] = hop_ert.value
                except ValueError:
                    if config.expected_response_type_fallback_policy == "reject":
                        rejected_conversation_ids.append(result.entity_id)
                        continue
                    payload["expected_response_type"] = ExpectedResponseType.UNKNOWN.value

            # Process supporting questions JSON
            supp_questions_json = payload.get("supporting_questions_json")
            payload_rejected = False
            
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
                                    if config.expected_response_type_fallback_policy == "reject":
                                        payload_rejected = True
                                        break
                                    ert = ExpectedResponseType.UNKNOWN
                            else:
                                if config.expected_response_type_fallback_policy == "reject":
                                    payload_rejected = True
                                    break
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
                            
                        if not payload_rejected:
                            # Replace stringified JSON with the structured parsed list for LLM visibility
                            payload["supporting_questions"] = sq_list
                            if "supporting_questions_json" in payload:
                                del payload["supporting_questions_json"]
                                
                except Exception as e:
                    logger.warning(f"Failed to parse supporting_questions_json: {e}")
                    
            if payload_rejected:
                rejected_conversation_ids.append(result.entity_id)
                continue
                
            approved_conversation.append(payload)
            
            topic_id = payload.get("topic_id")
            if topic_id and topic_id not in _internal_selected_topic_candidates:
                _internal_selected_topic_candidates.append(topic_id)
                
            hop_id = payload.get("hop_id")
            if hop_id and hop_id not in _internal_selected_hop_candidates:
                _internal_selected_hop_candidates.append(hop_id)
            
            if len(approved_conversation) >= config.conversation_approved_max_items:
                break
                
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
