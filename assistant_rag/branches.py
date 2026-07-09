"""Intent branch implementations."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from .config import AssistantConfig
from .contracts import AnswerMode, BranchResult, ChatRequest, Intent, PipelineContext, ResponseType, GeneratedQuestion, QuestionSource, ActionValidationResult
from .database import SQLRepository
from .llm import LLMClient, LLMTask
from .prompts import DEFAULT_PROMPT_REGISTRY, PromptContext, PromptRegistry
from .retrieval import HybridRetriever
from .context_filter import ContextFilter
from .action_detection import ActionDetector
from .generation import QuestionGenerationStrategy
from .branch_orchestration import ValidatedActionBuilder
from .settings import MutationPartialExecutionPolicy


class HumanInTheLoopStrategy(Protocol):
    def evaluate(
        self,
        context: PipelineContext,
        response_text: str,
        confidence: float,
    ) -> tuple[list[GeneratedQuestion], dict[str, Any] | None]:
        ...


class PassthroughHITL:
    def evaluate(
        self, context: PipelineContext, response_text: str, confidence: float
    ) -> tuple[list[GeneratedQuestion], dict[str, Any] | None]:
        return [], None


class Branch(Protocol):
    def execute(self, context: PipelineContext) -> BranchResult:
        ...


class ClarificationBranch:
    def __init__(self, prompt_registry: PromptRegistry = DEFAULT_PROMPT_REGISTRY, config: AssistantConfig | None = None, clarification_strategy: QuestionGenerationStrategy | None = None) -> None:
        self.prompt_registry = prompt_registry
        self.config = config
        self.clarification_strategy = clarification_strategy

    def execute(self, context: PipelineContext) -> BranchResult:
        question: GeneratedQuestion | None = context.request.metadata.get("clarification_question")
        if not question:
            if self.clarification_strategy:
                question = self.clarification_strategy.generate(context)
        if not question:
            fallback = self.config.question_generation.fallback_policy if self.config else "fallback_message"
            question = GeneratedQuestion(
                text=self.prompt_registry.message(fallback),
                source=QuestionSource.CLARIFICATION_QUESTION,
                purpose="resolve_missing_info",
                confidence=1.0,
            )
            
        return BranchResult(
            response_type=ResponseType.CLARIFICATION,
            clarification_question=question,
        )


@dataclass
class GeneralResponseBranch:
    repository: SQLRepository
    retriever: HybridRetriever
    config: AssistantConfig
    context_filter: ContextFilter
    llm: LLMClient | None = None
    prompt_registry: PromptRegistry = field(default_factory=lambda: DEFAULT_PROMPT_REGISTRY)
    hitl_strategy: HumanInTheLoopStrategy = field(default_factory=PassthroughHITL)

    def execute(self, context: PipelineContext) -> BranchResult:
        knowledge_results = self.retriever.retrieve_knowledge(
            user_id=context.request.user_id,
            query=context.rewritten_query,
            limit=self.config.retrieval.max_results,
            min_confidence=self.config.retrieval.knowledge_min_confidence,
        )

        reminder_context = []
        if self._should_retrieve_reminder_context(context):
            reminder_context = self.repository.list_reminders(
                user_id=context.request.user_id,
                # In actual implementation, we might filter by time here, but for now we fetch recent/scheduled
            )

        approved_context = self.context_filter.filter(
            user_id=context.request.user_id,
            knowledge_results=knowledge_results,
            reminder_results=reminder_context,
            query=context.rewritten_query,
            intent=context.intent,
        )

        answer_mode = self._detect_answer_mode(context, approved_context)

        evidence = approved_context.knowledge_evidence
        
        response = context.request.metadata.get("normal_response_text")
        if not response:
            response = self._generate_response(context, evidence, answer_mode)
        
        metadata_questions = list(context.request.metadata.get("supporting_questions", []))
        hitl_questions, hitl_result = self.hitl_strategy.evaluate(
            context, response, confidence=1.0
        )
        supporting_questions = metadata_questions + hitl_questions
        topic_title = context.request.metadata.get("topic_title", "General")
        try:
            with self.repository.transaction() as cursor:
                topic_id = self.repository.ensure_topic(
                    cursor, user_id=context.request.user_id, title=topic_title
                )
                hop = self.repository.append_conversation_hop(
                    cursor,
                    topic_id=topic_id,
                    user_id=context.request.user_id,
                    intent=Intent.GENERAL_RESPONSE.value,
                    raw_user_query=context.request.raw_query,
                    rewritten_user_query=context.rewritten_query,
                    raw_response=response,
                    response_type=ResponseType.NORMAL.value,
                    supporting_questions=[{
                        "question_text": q.text,
                        "question_source": q.source.value,
                        "purpose": q.purpose,
                        "confidence": q.confidence
                    } for q in supporting_questions],
                )
                # Future enhancement: we should use `supporting_questions_json` on SQL DB.
                # But the requirement is to use existing fields if schema hasn't changed.
        except Exception:
            return BranchResult(
                response_type=ResponseType.ERROR,
                fallback_or_error_message="The database transaction failed and was rolled back.",
            )
            
        return BranchResult(
            response_type=ResponseType.NORMAL,
            normal_response_text=response,
            human_supporting_questions=supporting_questions,
            human_in_the_loop_result=hitl_result,
            linked_topic_id=hop.topic_id,
            linked_hop_id=hop.hop_id,
            database_write_result={"conversation_hop_id": hop.hop_id},
            indexing_job_result={"conversation_hop_job_id": hop.outbox_job_id},
        )

    def _should_retrieve_reminder_context(self, context: PipelineContext) -> bool:
        query_lower = context.rewritten_query.lower()
        reminder_keywords = ["remind", "schedule", "plan", "upcoming"]
        if any(kw in query_lower for kw in reminder_keywords):
            return True
        return False

    def _detect_answer_mode(self, context: PipelineContext, approved_context: Any) -> AnswerMode:
        if context.last_qa_state and getattr(context.last_qa_state, "supporting_questions", []):
            return AnswerMode.SUPPORT_QUESTION_ANSWER
        if not context.conversation_results and not context.last_qa_state:
            return AnswerMode.NEW_CONVERSATION
        return AnswerMode.FOLLOW_UP_CONVERSATION

    def _generate_response(self, context: PipelineContext, evidence: list[str], answer_mode: AnswerMode) -> str:
        fallback = " ".join(evidence) if evidence else self.prompt_registry.message("answer_model_unavailable")
        if self.llm is None:
            return fallback
        try:
            response = self.llm.chat(
                task=LLMTask.ANSWER,
                system_prompt=self.prompt_registry.system("answer_generation"),
                user_prompt=self.prompt_registry.user(
                    PromptContext(
                        stage="answer_generation",
                        user_id=context.request.user_id,
                        raw_query=context.request.raw_query,
                        rewritten_query=context.rewritten_query,
                        intent=context.intent.value,
                        metadata=context.request.metadata,
                        platform_context=context.request.platform_context,
                        extra={
                            "knowledge_evidence": evidence,
                            "conversation_result_count": len(context.conversation_results),
                            "answer_mode": answer_mode.value,
                        },
                    )
                ),
            ).strip()
        except Exception:
            return fallback
        return response or fallback


@dataclass
class KnowledgeFactsBranch:
    repository: SQLRepository
    config: AssistantConfig
    action_detector: ActionDetector | None = None
    clarification_strategy: QuestionGenerationStrategy | None = None
    prompt_registry: PromptRegistry = field(default_factory=lambda: DEFAULT_PROMPT_REGISTRY)
    validated_action_builder: ValidatedActionBuilder | None = None

    def _generate_clarification(self, context: PipelineContext, missing_fields: list[str], ambiguity_reason: str) -> BranchResult:
        question = None
        if self.clarification_strategy:
            question = self.clarification_strategy.generate(context, missing_fields=missing_fields, ambiguity_reason=ambiguity_reason)
        if not question:
            fallback = self.config.question_generation.fallback_policy if self.config else "fallback_message"
            question = GeneratedQuestion(
                text=self.prompt_registry.message(fallback),
                source=QuestionSource.CLARIFICATION_QUESTION,
                purpose="resolve_missing_info",
                confidence=1.0,
            )
        return BranchResult(
            response_type=ResponseType.CLARIFICATION,
            clarification_question=question,
        )

    def execute(self, context: PipelineContext) -> BranchResult:
        actions = list(context.request.metadata.get("knowledge_actions", []))
        if not actions:
            if self.action_detector:
                detection = self.action_detector.detect(
                    context.request, context.rewritten_query, Intent.KNOWLEDGE_FACTS
                )
                if detection.requires_clarification:
                    return self._generate_clarification(context, getattr(detection, "missing_fields", []), "Missing fields for knowledge action.")
                if detection.metadata:
                    actions = detection.metadata.get("knowledge_actions", [])
                    context.request.metadata.update(detection.metadata)
                    
        if not actions:
            return self._generate_clarification(context, [], "No valid knowledge action detected.")
            
        executable_actions = []
        pre_repo_results = []
        if self.validated_action_builder:
            validated_actions = self.validated_action_builder.build_knowledge_actions(
                user_id=context.request.user_id, 
                actions=actions,
                user_query=context.request.raw_query,
                rewritten_query=context.rewritten_query
            )
            clarification_needed = False
            for v_act in validated_actions:
                if v_act.validation_result in (ActionValidationResult.CLARIFY_AMBIGUOUS_TARGET, ActionValidationResult.CLARIFY_MISSING_FIELDS):
                    clarification_needed = True
                    break
            
            if clarification_needed:
                if self.config.mutation_policy.partial_execution_policy == MutationPartialExecutionPolicy.ALL_OR_NOTHING:
                    return self._generate_clarification(context, [], "Action requires clarification.")
                else:
                    return self._generate_clarification(context, [], "Ambiguous destructive actions block partial execution.")
                    
            from .contracts import RepositoryActionResult
            for v_act in validated_actions:
                if v_act.validation_result == ActionValidationResult.EXECUTE:
                    executable_actions.append(v_act)
                else:
                    status = "validation_failed"
                    if v_act.validation_result == ActionValidationResult.SKIP_NOT_FOUND:
                        status = "not_found"
                    pre_repo_results.append(RepositoryActionResult(
                        action_id="pre-repo",
                        action_type=v_act.action.value,
                        status=status,
                        domain_entity_type="knowledge_chunk",
                        user_safe_summary=f"Action skipped due to {v_act.validation_result.value}.",
                        reason_summary=v_act.reason_summary or f"Validation resulted in {v_act.validation_result.value}",
                    ))
                    
            if not executable_actions and not clarification_needed:
                response_text = self.prompt_registry.message("knowledge_no_op") if hasattr(self.prompt_registry, "message") else "No matching knowledge item was found."
                result = self.repository.record_action_audit_noop(
                    user_id=context.request.user_id,
                    topic_title=context.request.metadata.get("topic_title", "Knowledge"),
                    raw_user_query=context.request.raw_query,
                    rewritten_user_query=context.rewritten_query,
                    response_text=response_text,
                    intent=Intent.KNOWLEDGE_FACTS.value,
                    response_type="safe_noop",
                )
                return BranchResult(
                    response_type=ResponseType.KNOWLEDGE_ACTION,
                    normal_response_text=response_text,
                    knowledge_operation_results=pre_repo_results + list(result.results),
                    linked_hop_id=result.audit_hop_id,
                )
        else:
            executable_actions = actions

        success_text = context.request.metadata.get(
            "operation_response", self.prompt_registry.message("knowledge_updated") if hasattr(self.prompt_registry, "message") else "Knowledge updated successfully."
        )
        
        result = self.repository.transactional_knowledge_actions(
            user_id=context.request.user_id,
            topic_title=context.request.metadata.get("topic_title", "Knowledge"),
            raw_user_query=context.request.raw_query,
            rewritten_user_query=context.rewritten_query,
            response_text=success_text,
            actions=executable_actions,
        )
        
        if result.committed:
            return BranchResult(
                response_type=ResponseType.KNOWLEDGE_ACTION,
                normal_response_text=success_text,
                supporting_questions=[],
                knowledge_operation_results=pre_repo_results + list(result.results),
                linked_hop_id=result.audit_hop_id,
            )
            
        return BranchResult(
            response_type=ResponseType.ERROR,
            fallback_or_error_message=result.reason_summary,
        )


@dataclass
class ReminderBranch:
    repository: SQLRepository
    config: AssistantConfig
    action_detector: ActionDetector | None = None
    clarification_strategy: QuestionGenerationStrategy | None = None
    reminder_supporting_strategy: QuestionGenerationStrategy | None = None
    prompt_registry: PromptRegistry = field(default_factory=lambda: DEFAULT_PROMPT_REGISTRY)
    validated_action_builder: ValidatedActionBuilder | None = None

    def _generate_clarification(self, context: PipelineContext, missing_fields: list[str], ambiguity_reason: str) -> BranchResult:
        question = None
        if self.clarification_strategy:
            question = self.clarification_strategy.generate(context, missing_fields=missing_fields, ambiguity_reason=ambiguity_reason)
        if not question:
            fallback = self.config.question_generation.fallback_policy if self.config else "fallback_message"
            question = GeneratedQuestion(
                text=self.prompt_registry.message(fallback),
                source=QuestionSource.CLARIFICATION_QUESTION,
                purpose="resolve_missing_info",
                confidence=1.0,
            )
        return BranchResult(
            response_type=ResponseType.CLARIFICATION,
            clarification_question=question,
        )

    def execute(self, context: PipelineContext) -> BranchResult:
        actions = list(context.request.metadata.get("reminder_actions", []))
        if not actions:
            if self.action_detector:
                detection = self.action_detector.detect(
                    context.request, context.rewritten_query, Intent.REMINDER
                )
                if detection.requires_clarification:
                    return self._generate_clarification(context, getattr(detection, "missing_fields", []), "Missing fields for reminder action.")
                if detection.metadata:
                    actions = detection.metadata.get("reminder_actions", [])
                    context.request.metadata.update(detection.metadata)
                    
        if not actions:
            return self._generate_clarification(context, [], "No valid reminder action detected.")
            
        executable_actions = []
        pre_repo_results = []
        if self.validated_action_builder:
            validated_actions = self.validated_action_builder.build_reminder_actions(
                user_id=context.request.user_id, 
                actions=actions,
                user_query=context.request.raw_query,
                rewritten_query=context.rewritten_query
            )
            clarification_needed = False
            for v_act in validated_actions:
                if v_act.validation_result in (ActionValidationResult.CLARIFY_AMBIGUOUS_TARGET, ActionValidationResult.CLARIFY_MISSING_FIELDS):
                    clarification_needed = True
                    break
            
            if clarification_needed:
                if self.config.mutation_policy.partial_execution_policy == MutationPartialExecutionPolicy.ALL_OR_NOTHING:
                    return self._generate_clarification(context, [], "Action requires clarification.")
                else:
                    return self._generate_clarification(context, [], "Ambiguous actions block execution.")
                    
            from .contracts import RepositoryActionResult
            for v_act in validated_actions:
                if v_act.validation_result == ActionValidationResult.EXECUTE:
                    executable_actions.append(v_act)
                else:
                    status = "validation_failed"
                    if v_act.validation_result == ActionValidationResult.SKIP_NOT_FOUND:
                        status = "not_found"
                    pre_repo_results.append(RepositoryActionResult(
                        action_id="pre-repo",
                        action_type=v_act.action.value,
                        status=status,
                        domain_entity_type="reminder",
                        user_safe_summary=f"Action skipped due to {v_act.validation_result.value}.",
                        reason_summary=v_act.reason_summary or f"Validation resulted in {v_act.validation_result.value}",
                    ))
                    
            if not executable_actions and not clarification_needed:
                response_text = self.prompt_registry.message("reminder_no_op") if hasattr(self.prompt_registry, "message") else "No reminder needed to be changed."
                result = self.repository.record_action_audit_noop(
                    user_id=context.request.user_id,
                    topic_title=context.request.metadata.get("topic_title", "Reminders"),
                    raw_user_query=context.request.raw_query,
                    rewritten_user_query=context.rewritten_query,
                    response_text=response_text,
                    intent=Intent.REMINDER.value,
                    response_type="safe_noop",
                )
                return BranchResult(
                    response_type=ResponseType.REMINDER_ACTION,
                    normal_response_text=response_text,
                    reminder_operation_results=pre_repo_results + list(result.results),
                    linked_hop_id=result.audit_hop_id,
                )
        else:
            executable_actions = actions

        success_text = context.request.metadata.get(
            "operation_response", self.prompt_registry.message("reminder_updated") if hasattr(self.prompt_registry, "message") else "Reminder updated successfully."
        )
        
        reminder_supporting_question = None
        if self.reminder_supporting_strategy:
            reminder_supporting_question = self.reminder_supporting_strategy.generate(
                context, 
                action_summary=success_text, 
                reminder_metadata={"actions": actions}
            )
            if reminder_supporting_question and executable_actions:
                if isinstance(executable_actions[0], dict):
                    executable_actions[0]["supporting_question"] = json.dumps({
                        "question_text": reminder_supporting_question.text,
                        "question_source": reminder_supporting_question.source.value,
                        "purpose": reminder_supporting_question.purpose,
                        "confidence": reminder_supporting_question.confidence
                    })

        result = self.repository.transactional_reminder_actions(
            user_id=context.request.user_id,
            topic_title=context.request.metadata.get("topic_title", "Reminders"),
            raw_user_query=context.request.raw_query,
            rewritten_user_query=context.rewritten_query,
            response_text=success_text,
            actions=executable_actions,
        )
        
        if result.committed:
            return BranchResult(
                response_type=ResponseType.REMINDER_ACTION,
                normal_response_text=success_text,
                human_supporting_questions=[],
                reminder_supporting_question=reminder_supporting_question,
                reminder_operation_results=pre_repo_results + list(result.results),
                linked_hop_id=result.audit_hop_id,
            )

        return BranchResult(
            response_type=ResponseType.ERROR,
            fallback_or_error_message=result.reason_summary,
        )


class BranchRouter:
    def __init__(self, branches: dict[Intent, Branch]) -> None:
        self.branches = branches

    def route(self, context: PipelineContext) -> BranchResult:
        if context.intent not in self.branches:
            raise ValueError(f"No branch registered for intent: {context.intent}")
        return self.branches[context.intent].execute(context)



