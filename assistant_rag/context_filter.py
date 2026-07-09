"""Context filtering stages for General Response."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .contracts import Intent, RetrievalResult
from .llm import LLMClient
from .prompts import PromptRegistry


@dataclass(frozen=True)
class ApprovedContext:
    knowledge_evidence: list[str]
    reminder_context: list[dict[str, Any]]
    rejected_knowledge_ids: list[str]
    rejected_reminder_ids: list[str]


class ContextFilter(Protocol):
    def filter(
        self,
        *,
        user_id: str,
        knowledge_results: list[RetrievalResult],
        reminder_results: list[dict[str, Any]],
        query: str,
        intent: Intent,
    ) -> ApprovedContext:
        ...


@dataclass
class HardRuleContextFilter:
    knowledge_min_confidence: float
    allowed_reminder_statuses: tuple[str, ...]

    def filter(
        self,
        *,
        user_id: str,
        knowledge_results: list[RetrievalResult],
        reminder_results: list[dict[str, Any]],
        query: str,
        intent: Intent,
    ) -> ApprovedContext:
        approved_knowledge: list[str] = []
        rejected_knowledge_ids: list[str] = []
        
        for result in knowledge_results:
            payload = result.payload
            
            # Validation 1: Entity type
            if result.entity_type != "knowledge_chunk":
                rejected_knowledge_ids.append(result.entity_id)
                continue
                
            # Validation 2: Ownership
            payload_user_id = payload.get("user_id")
            if payload_user_id and payload_user_id != user_id:
                rejected_knowledge_ids.append(result.entity_id)
                continue
                
            # Validation 3: Deletion status
            if payload.get("is_deleted", False):
                rejected_knowledge_ids.append(result.entity_id)
                continue
                
            # Validation 4: Confidence threshold
            if result.confidence < self.knowledge_min_confidence:
                rejected_knowledge_ids.append(result.entity_id)
                continue
                
            text = payload.get("text")
            if not text:
                rejected_knowledge_ids.append(result.entity_id)
                continue
                
            approved_knowledge.append(str(text))

        approved_reminders: list[dict[str, Any]] = []
        rejected_reminder_ids: list[str] = []
        
        for reminder in reminder_results:
            reminder_id = str(reminder.get("reminder_id", ""))
            
            # Validation 1: Ownership
            if reminder.get("user_id") != user_id:
                rejected_reminder_ids.append(reminder_id)
                continue
                
            # Validation 2: Allowed status
            status = reminder.get("status")
            if status not in self.allowed_reminder_statuses:
                rejected_reminder_ids.append(reminder_id)
                continue
                
            # Must not mutate reminders here, just pass them as context
            approved_reminders.append(reminder)

        return ApprovedContext(
            knowledge_evidence=approved_knowledge,
            reminder_context=approved_reminders,
            rejected_knowledge_ids=rejected_knowledge_ids,
            rejected_reminder_ids=rejected_reminder_ids,
        )


@dataclass
class LLMSemanticContextJudge:
    llm: LLMClient
    prompt_registry: PromptRegistry

    def filter(
        self,
        hard_rule_context: ApprovedContext,
        query: str,
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
        query: str,
        intent: Intent,
    ) -> ApprovedContext:
        # Layer 1: Strict deterministic filtering
        safe_context = self.hard_rule_filter.filter(
            user_id=user_id,
            knowledge_results=knowledge_results,
            reminder_results=reminder_results,
            query=query,
            intent=intent,
        )
        
        # Layer 2: Optional semantic judgement (receives only safe context)
        if self.llm_judge:
            return self.llm_judge.filter(safe_context, query)
            
        return safe_context
