"""Intent branch implementations."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Protocol

from .config import AssistantConfig
from .contracts import (
    AnswerMode,
    ActionValidationResult,
    BranchResult,
    ChatRequest,
    ContentComposerInput,
    ExpectedResponseType,
    GeneratedQuestion,
    GeneralSubBranch,
    GeneralSubBranchDecision,
    Intent,
    LastQAInteractionType,
    PersistenceMode,
    PipelineContext,
    QuestionSource,
    ReminderAction,
    KnowledgeAction,
    RepositoryActionResult,
    ResponseType,
    SubBranchPromptContext,
    ValidatedReminderAction,
    ValidatedKnowledgeAction,
)

SUB_BRANCH_PROMPT_POLICIES = {
    GeneralSubBranch.SUPPORT_QUESTION_ANSWER: {
        "chat_history_role": "The user is answering a previous human supporting question.",
        "response_goal": "Use the prior supporting question, expected_response_type, previous assistant response, and current user answer to enhance, refine, continue, or personalize the prior answer.",
        "database_update_mode": "append_to_existing_topic_or_branch_from_existing_hop",
        "allowed_database_updates": ("conversation_hop_append",),
        "prohibited_database_updates": ("new_topic_creation_unless_no_valid_topic", "knowledge_mutation", "reminder_mutation", "notification_write"),
    },
    GeneralSubBranch.CONVERSATION_FOLLOW_UP: {
        "chat_history_role": "The user is continuing an approved existing conversation.",
        "response_goal": "Use approved chat_history to continue the same topic while keeping the current user query primary.",
        "database_update_mode": "append_to_existing_topic",
        "allowed_database_updates": ("conversation_hop_append",),
        "prohibited_database_updates": ("new_topic_creation", "knowledge_mutation", "reminder_mutation", "notification_write"),
    },
    GeneralSubBranch.NEW_CONVERSATION_TOPIC: {
        "chat_history_role": "The user is starting a new conversation topic.",
        "response_goal": "Answer the current query directly. Do not force continuity from unrelated old history.",
        "database_update_mode": "create_or_ensure_new_topic_then_append_hop",
        "allowed_database_updates": ("conversation_topic_create_or_ensure", "conversation_hop_append"),
        "prohibited_database_updates": ("knowledge_mutation", "reminder_mutation", "notification_write"),
    },
}
from .database import AssistantRepository
from .llm import LLMClient, LLMTask
from .prompts import DEFAULT_PROMPT_REGISTRY, PromptContext, PromptRegistry
from .retrieval import HybridRetriever
from .canonical_retrieval import retrieve_knowledge
from .reminder_retrieval import retrieve_reminder_candidates, reminder_candidate_to_context
from .semantic_chunking import split_semantic_chunks
from .context_filter import ContextFilter
from .action_detection import ActionDetector
from .generation import QuestionGenerationStrategy
from .branch_orchestration import ValidatedActionBuilder
from .settings import MutationPartialExecutionPolicy
from .reminder_safety import ReminderTimeNormalizer, ReminderTimeNormalizationError


def _mutation_clarification_message_key(domain: str, missing_fields: list[str]) -> str:
    """Choose a fast, user-facing question for predictable mutation gaps."""
    missing = {str(field).casefold() for field in missing_fields}
    if domain == "knowledge":
        if {"replacement_text", "replacement", "new_text"} & missing:
            return "knowledge_missing_replacement"
        if {"target_description", "target", "target_entities"} & missing:
            return "knowledge_missing_target"
        if {"text", "knowledge_text", "content"} & missing:
            return "knowledge_missing_content"
        return "knowledge_missing_action"

    if {"reminder_time", "new_reminder_time", "time", "date", "timezone"} & missing:
        return "reminder_missing_time"
    if {"subject", "new_subject", "reminder_subject"} & missing:
        return "reminder_missing_subject"
    if {"target_description", "target", "reminder_id"} & missing:
        return "reminder_missing_target"
    if {"new_reminder_time", "new_subject", "replacement"} & missing:
        return "reminder_missing_update"
    return "reminder_missing_action"


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _json_safe(val) for key, val in asdict(value).items()}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _expires_at(config: AssistantConfig) -> str:
    return (datetime.now(timezone.utc) + timedelta(minutes=config.confirmation_expiry_minutes)).isoformat()


def _pending_confirmation_result(
    *,
    response_type: ResponseType,
    text: str,
    confirmation: dict[str, Any],
) -> BranchResult:
    return BranchResult(
        response_type=response_type,
        normal_response_text=text,
        actions_pending_confirmation=[
            {
                "confirmation_token": confirmation["confirmation_token"],
                "action_type": confirmation["action_type"],
                "target_entity_type": confirmation["target_entity_type"],
                "target_entity_id": confirmation.get("target_entity_id"),
                "expires_at": confirmation["expires_at"],
                "status": confirmation["status"],
            }
        ],
    )


def _parse_dt(value: Any) -> datetime | None:
    if value is None or isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def _validated_reminder_from_dict(payload: dict[str, Any]) -> ValidatedReminderAction:
    return ValidatedReminderAction(
        action=ReminderAction(payload["action"]),
        validation_result=ActionValidationResult(payload.get("validation_result", ActionValidationResult.EXECUTE.value)),
        target_reminder_ids=tuple(payload.get("target_reminder_ids") or ()),
        observed_status=payload.get("observed_status"),
        observed_version=payload.get("observed_version"),
        observed_reminder_time=_parse_dt(payload.get("observed_reminder_time")),
        subject=payload.get("subject"),
        event_time=_parse_dt(payload.get("event_time")),
        reminder_time=_parse_dt(payload.get("reminder_time")),
        reminder_summary=payload.get("reminder_summary"),
        raw_reminder=payload.get("raw_reminder"),
        user_timezone=payload.get("user_timezone"),
        original_time_text=payload.get("original_time_text"),
        replacement_subject=payload.get("replacement_subject"),
        replacement_time=_parse_dt(payload.get("replacement_time")),
        replacement_summary=payload.get("replacement_summary"),
        confidence=float(payload.get("confidence", 1.0)),
        matched_fields=tuple(payload.get("matched_fields") or ()),
        reason_summary=payload.get("reason_summary"),
    )


def _validated_knowledge_from_dict(payload: dict[str, Any]) -> ValidatedKnowledgeAction:
    return ValidatedKnowledgeAction(
        action=KnowledgeAction(payload["action"]),
        validation_result=ActionValidationResult(payload.get("validation_result", ActionValidationResult.EXECUTE.value)),
        target_chunk_ids=tuple(payload.get("target_chunk_ids") or ()),
        target_topic_ids=tuple(payload.get("target_topic_ids") or ()),
        observed_versions=dict(payload.get("observed_versions") or {}),
        observed_is_deleted=dict(payload.get("observed_is_deleted") or {}),
        knowledge_text=payload.get("knowledge_text"),
        replacement_text=payload.get("replacement_text"),
        new_text=payload.get("new_text"),
        target_status=payload.get("target_status"),
        topic_title=payload.get("topic_title"),
        target_description=payload.get("target_description"),
        confidence=float(payload.get("confidence", 1.0)),
        matched_fields=tuple(payload.get("matched_fields") or ()),
        reason_summary=payload.get("reason_summary"),
    )


def _simple_fact_pair(text: str | None) -> tuple[str, str] | None:
    if not text:
        return None
    value = " ".join(str(text).strip().split())
    match = re.match(r"(?i)^(?:my|the|our)?\s*(.+?)\s+(?:is|are|=)\s+(.+)$", value)
    if not match:
        return None
    subject = re.sub(r"[^a-z0-9 ]+", " ", match.group(1).casefold())
    subject = " ".join(subject.split())
    fact_value = re.sub(r"\s+", " ", match.group(2).strip())
    if not subject or not fact_value:
        return None
    return subject, fact_value.casefold()


def _find_simple_knowledge_conflict(
    repository: AssistantRepository,
    *,
    user_id: str,
    new_text: str,
) -> dict[str, Any] | None:
    pair = _simple_fact_pair(new_text)
    if not pair:
        return None
    new_subject, new_value = pair
    for fact in repository.list_knowledge_facts(user_id=user_id, include_deleted=False):
        old_pair = _simple_fact_pair(fact.get("normalized_text") or fact.get("raw_text"))
        if old_pair and old_pair[0] == new_subject and old_pair[1] != new_value:
            return fact
    return None


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
    def execute(self, context: PipelineContext, repository: AssistantRepository) -> BranchResult:
        ...


class ClarificationBranch:
    def __init__(self, prompt_registry: PromptRegistry = DEFAULT_PROMPT_REGISTRY, config: AssistantConfig | None = None, clarification_strategy: QuestionGenerationStrategy | None = None) -> None:
        self.prompt_registry = prompt_registry
        self.config = config
        self.clarification_strategy = clarification_strategy

    def execute(self, context: PipelineContext, repository: AssistantRepository) -> BranchResult:
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

    retriever: HybridRetriever
    config: AssistantConfig
    context_filter: ContextFilter
    prompt_registry: PromptRegistry = field(default_factory=lambda: DEFAULT_PROMPT_REGISTRY)
    hitl_strategy: HumanInTheLoopStrategy = field(default_factory=PassthroughHITL)
    llm: LLMClient | None = None
    sub_branch_detector: Any | None = None
    content_composer: Any | None = None
    general_hitl_strategy: Any | None = None
    general_purpose_config: Any | None = None

    def execute(self, context: PipelineContext, repository: AssistantRepository) -> BranchResult:
        knowledge_results = retrieve_knowledge(
            retriever=self.retriever,
            repository=repository,
            user_id=context.request.user_id,
            query=context.rewritten_query,
        )

        reminder_raw = [
            reminder_candidate_to_context(
                user_id=context.request.user_id,
                candidate=candidate,
            )
            for candidate in retrieve_reminder_candidates(
                repository=repository,
                user_id=context.request.user_id,
                statuses=self.config.retrieval.general_response_reminder_statuses,
                candidate_limit=self.config.retrieval.general_response_reminder_limit,
            )
        ]

        approved_context = self.context_filter.filter(
            user_id=context.request.user_id,
            knowledge_results=knowledge_results,
            reminder_results=reminder_raw,
            conversation_results=[],
            query=context.rewritten_query,
            intent=context.intent,
        )
        approved_context = replace(
            approved_context,
            approved_conversation_history=(
                context.approved_conversation_context.approved_conversation_history
                if context.approved_conversation_context
                else []
            ),
        )
        merged_supporting_detail = self._merge_supporting_detail(approved_context)

        from .general_sub_branch import GeneralSubBranchValidator, GeneralPersistencePlanBuilder
        
        decision = self._deterministic_general_decision(context, approved_context)
        if decision and self.general_purpose_config:
            plan = GeneralPersistencePlanBuilder().build_plan(decision, context, self.general_purpose_config)
            answer_mode = self._general_sub_branch_to_answer_mode(decision.sub_branch)
        elif self.sub_branch_detector and self.general_purpose_config:
            decision = self.sub_branch_detector.detect(
                context,
                self.general_purpose_config,
                merged_supporting_detail,
            )
            decision = GeneralSubBranchValidator().validate(decision, context, self.general_purpose_config)
            plan = GeneralPersistencePlanBuilder().build_plan(decision, context, self.general_purpose_config)
            answer_mode = self._general_sub_branch_to_answer_mode(decision.sub_branch)
        else:
            decision = None
            plan = None
            answer_mode = self._detect_answer_mode(context, approved_context)
            
        resolved_sub_branch = decision.sub_branch if decision else GeneralSubBranch.NEW_CONVERSATION_TOPIC
        if not decision:
            if answer_mode == AnswerMode.SUPPORT_QUESTION_ANSWER:
                resolved_sub_branch = GeneralSubBranch.SUPPORT_QUESTION_ANSWER
            elif answer_mode == AnswerMode.FOLLOW_UP_CONVERSATION:
                resolved_sub_branch = GeneralSubBranch.CONVERSATION_FOLLOW_UP
            
        resolved_persistence_mode = plan.persistence_mode if plan else PersistenceMode.CREATE_NEW_TOPIC
        
        policy_dict = SUB_BRANCH_PROMPT_POLICIES.get(resolved_sub_branch, SUB_BRANCH_PROMPT_POLICIES[GeneralSubBranch.NEW_CONVERSATION_TOPIC])
        prompt_ctx = SubBranchPromptContext(
            sub_branch=resolved_sub_branch,
            persistence_mode=resolved_persistence_mode,
            chat_history_role=policy_dict["chat_history_role"],
            response_goal=policy_dict["response_goal"],
            database_update_mode=policy_dict["database_update_mode"],
            allowed_database_updates=policy_dict["allowed_database_updates"],
            prohibited_database_updates=policy_dict["prohibited_database_updates"],
            expected_response_type=context.last_qa_state.expected_response_type if context.last_qa_state and context.last_qa_state.expected_response_type else ExpectedResponseType.UNKNOWN
        )
        
        sub_branch_supporting_prompt = self.prompt_registry.message(
            "sub_branch_supporting_prompt",
            sub_branch=prompt_ctx.sub_branch.value,
            persistence_mode=prompt_ctx.persistence_mode.value,
            chat_history_role=prompt_ctx.chat_history_role,
            response_goal=prompt_ctx.response_goal,
            database_update_mode=prompt_ctx.database_update_mode,
            allowed_database_updates=", ".join(prompt_ctx.allowed_database_updates),
            prohibited_database_updates=", ".join(prompt_ctx.prohibited_database_updates),
            expected_response_type=prompt_ctx.expected_response_type.value,
        ) + "\nRetrieved supporting detail:\n" + merged_supporting_detail

        if self.content_composer and self.general_purpose_config:
            composer_input = ContentComposerInput(
                user_id=context.request.user_id,
                raw_user_query=context.request.raw_query,
                rewritten_query=context.rewritten_query,
                sub_branch=resolved_sub_branch,
                persistence_mode=resolved_persistence_mode,
                approved_conversation_history=approved_context.approved_conversation_history,
                human_supporting_questions=context.approved_conversation_context.human_supporting_questions if context.approved_conversation_context else [],
                reminder_supporting_questions=context.approved_conversation_context.reminder_supporting_questions if context.approved_conversation_context else [],
                extracted_expected_response_types=context.approved_conversation_context.extracted_expected_response_types if context.approved_conversation_context else [],
                approved_knowledge_evidence=approved_context.knowledge_evidence,
                approved_reminder_context=approved_context.reminder_context,
                metadata=context.request.metadata,
                platform_context=context.request.platform_context,
                sub_branch_prompt_context=prompt_ctx,
                sub_branch_supporting_prompt=sub_branch_supporting_prompt,
                repository=repository,
                merged_supporting_detail=merged_supporting_detail,
            )
            composer_result = self.content_composer.compose(composer_input, self.general_purpose_config)
            response = composer_result.final_response_text
        else:
            composer_result = None
            response = context.request.metadata.get("normal_response_text")
            if not response:
                response = self._generate_response(
                    context,
                    approved_context,
                    prompt_ctx,
                    sub_branch_supporting_prompt,
                    merged_supporting_detail,
                )

        if self.general_purpose_config is not None:
            if self.general_hitl_strategy:
                hitl_decision = self.general_hitl_strategy.evaluate(
                    context=context,
                    response_text=response,
                    approved_conversation_history=approved_context.approved_conversation_history,
                    merged_supporting_detail=merged_supporting_detail,
                )
                hitl_questions = [GeneratedQuestion(
                    text=hitl_decision.question,
                    source=hitl_decision.question_source,
                    purpose="optional_context",
                    confidence=hitl_decision.confidence,
                    should_ask=True,
                    expected_response_type=hitl_decision.expected_response_type,
                )] if hitl_decision.should_ask else []
                hitl_result = {"triggered": hitl_decision.should_ask, "confidence": hitl_decision.confidence, "question_count": len(hitl_questions)}
            else:
                hitl_questions = []
                hitl_result = {"triggered": False, "confidence": 1.0, "question_count": 0, "reason": "disabled_by_general_purpose_config"}
        else:
            hitl_questions, hitl_result = self.hitl_strategy.evaluate(
                context, response, confidence=1.0
            )

        supporting_questions = hitl_questions
        
        topic_title = context.request.metadata.get("topic_title", self.general_purpose_config.general_response_default_topic_title if self.general_purpose_config else "General Conversation")
        
        try:
            with repository.transaction() as cursor:
                if plan and plan.persistence_mode == PersistenceMode.APPEND_TO_EXISTING_TOPIC and plan.topic_id:
                    topic_id = plan.topic_id
                elif plan and plan.persistence_mode == PersistenceMode.BRANCH_FROM_EXISTING_HOP and plan.topic_id:
                    topic_id = plan.topic_id
                else:
                    topic_id = repository.ensure_topic(
                        cursor, user_id=context.request.user_id, title=topic_title
                    )
                
                parent_hop_id = plan.parent_hop_id if plan else context.request.parent_hop_id

                entities = {}
                if decision:
                    entities["sub_branch"] = decision.sub_branch.value
                if composer_result:
                    entities["used_tools"] = list(composer_result.used_tool_names)
                    entities["tool_trace_summary"] = composer_result.tool_trace_summary

                hop = repository.append_conversation_hop(
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
                    parent_hop_id=parent_hop_id,
                    entities=entities if entities else None
                )
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
            warnings=list(composer_result.content_warnings) if composer_result else [],
            database_write_result={"conversation_hop_id": hop.hop_id},
            indexing_job_result={"conversation_hop_job_id": hop.outbox_job_id},
            platform_payload={"artifacts": list(composer_result.artifacts)} if composer_result and composer_result.artifacts else {},
        )

    def _deterministic_general_decision(
        self, context: PipelineContext, approved_context: Any
    ) -> GeneralSubBranchDecision | None:
        authoritative = self._authoritative_last_qa_decision(context)
        if authoritative:
            return authoritative
        if context.last_qa_state:
            return None
        if getattr(approved_context, "approved_conversation_history", []):
            return None
        return GeneralSubBranchDecision(
            sub_branch=GeneralSubBranch.NEW_CONVERSATION_TOPIC,
            confidence=1.0,
            persistence_mode=PersistenceMode.CREATE_NEW_TOPIC,
            reason_summary="No approved prior context; deterministic new conversation topic.",
        )

    def _authoritative_last_qa_decision(self, context: PipelineContext) -> GeneralSubBranchDecision | None:
        trace = context.last_qa_trace or {}
        state = context.last_qa_state
        if not state:
            return None
        if not trace.get("skip_broad_retrieval") or not trace.get("is_authoritative_state"):
            return None
        if not state.linked_topic_id or not state.linked_hop_id:
            return None

        interaction_type = trace.get("interaction_type")
        if interaction_type == LastQAInteractionType.SUPPORTING_QUESTION_ANSWER.value:
            sub_branch = GeneralSubBranch.SUPPORT_QUESTION_ANSWER
        else:
            sub_branch = GeneralSubBranch.CONVERSATION_FOLLOW_UP

        return GeneralSubBranchDecision(
            sub_branch=sub_branch,
            confidence=1.0,
            persistence_mode=PersistenceMode.APPEND_TO_EXISTING_TOPIC,
            selected_topic_id=state.linked_topic_id,
            selected_hop_id=state.linked_hop_id,
            selected_parent_hop_id=state.linked_hop_id,
            reason_summary="Reused authoritative Last-QA latest-context resolution.",
        )

    def _general_sub_branch_to_answer_mode(self, sub_branch: Any) -> AnswerMode:
        from .contracts import GeneralSubBranch
        if sub_branch == GeneralSubBranch.SUPPORT_QUESTION_ANSWER:
            return AnswerMode.SUPPORT_QUESTION_ANSWER
        if sub_branch == GeneralSubBranch.CONVERSATION_FOLLOW_UP:
            return AnswerMode.FOLLOW_UP_CONVERSATION
        return AnswerMode.NEW_CONVERSATION

    @staticmethod
    def _merge_supporting_detail(approved_context: Any) -> str:
        return json.dumps(
            {
                "knowledge_evidence": approved_context.knowledge_evidence,
                "reminder_context": approved_context.reminder_context,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )

    def _detect_answer_mode(self, context: PipelineContext, approved_context: Any) -> AnswerMode:
        if context.last_qa_state and getattr(context.last_qa_state, "supporting_questions", []):
            return AnswerMode.SUPPORT_QUESTION_ANSWER
        if not approved_context.approved_conversation_history and not context.last_qa_state:
            return AnswerMode.NEW_CONVERSATION
        return AnswerMode.FOLLOW_UP_CONVERSATION

    def _generate_response(
        self,
        context: PipelineContext,
        approved_context: Any,
        prompt_ctx: SubBranchPromptContext,
        sub_branch_supporting_prompt: str,
        merged_supporting_detail: str,
    ) -> str:
        evidence = approved_context.knowledge_evidence
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
                            "approved_conversation_history": approved_context.approved_conversation_history,
                            "approved_knowledge_evidence": approved_context.knowledge_evidence,
                            "approved_reminder_context": approved_context.reminder_context,
                            "merged_supporting_detail": merged_supporting_detail,
                            "sub_branch_supporting_prompt": sub_branch_supporting_prompt,
                            "human_supporting_questions": [q.text for q in context.approved_conversation_context.human_supporting_questions] if context.approved_conversation_context else [],
                            "reminder_supporting_questions": [q.text for q in context.approved_conversation_context.reminder_supporting_questions] if context.approved_conversation_context else [],
                            "extracted_expected_response_types": [t.value for t in context.approved_conversation_context.extracted_expected_response_types] if context.approved_conversation_context else [],
                        },
                    )
                ),
            ).strip()
        except Exception:
            return fallback
        return response or fallback


@dataclass
class KnowledgeFactsBranch:

    config: AssistantConfig
    action_detector: ActionDetector | None = None
    clarification_strategy: QuestionGenerationStrategy | None = None
    prompt_registry: PromptRegistry = field(default_factory=lambda: DEFAULT_PROMPT_REGISTRY)
    validated_action_builder: ValidatedActionBuilder | None = None
    retriever: HybridRetriever | None = None
    context_filter: ContextFilter | None = None
    llm: LLMClient | None = None

    def _generate_clarification(self, context: PipelineContext, missing_fields: list[str], ambiguity_reason: str) -> BranchResult:
        question = GeneratedQuestion(
            text=self.prompt_registry.message(
                _mutation_clarification_message_key("knowledge", missing_fields)
            ),
            source=QuestionSource.CLARIFICATION_QUESTION,
            purpose="resolve_missing_info",
            confidence=1.0,
        )
        return BranchResult(
            response_type=ResponseType.CLARIFICATION,
            clarification_question=question,
        )

    def _expand_semantic_knowledge_actions(
        self, actions: list[ValidatedKnowledgeAction]
    ) -> list[ValidatedKnowledgeAction]:
        expanded: list[ValidatedKnowledgeAction] = []
        for action in actions:
            text = action.knowledge_text or action.replacement_text or action.new_text
            if action.action not in {KnowledgeAction.ADD, KnowledgeAction.MODIFY} or not text:
                expanded.append(action)
                continue
            chunks = split_semantic_chunks(
                text,
                settings=self.config.knowledge_chunk_settings,
            )
            if action.action is KnowledgeAction.MODIFY:
                expanded.append(
                    replace(
                        action,
                        action=KnowledgeAction.DELETE,
                        knowledge_text=None,
                        replacement_text=None,
                        new_text=None,
                    )
                )
            expanded.extend(
                replace(
                    action,
                    action=KnowledgeAction.ADD,
                    target_chunk_ids=(),
                    target_topic_ids=(),
                    observed_versions={},
                    observed_is_deleted={},
                    knowledge_text=chunk,
                    replacement_text=None,
                    new_text=chunk,
                )
                for chunk in chunks
            )
        return expanded

    def execute(self, context: PipelineContext, repository: AssistantRepository) -> BranchResult:
        actions = list(context.request.metadata.get("knowledge_actions", []))
        if not actions:
            if self.action_detector:
                detection = self.action_detector.detect(
                    context.request, context.rewritten_query, Intent.KNOWLEDGE_FACTS
                )
                if detection.metadata.get("knowledge_lookup"):
                    if self.retriever is None:
                        return self._generate_clarification(
                            context,
                            ["target_description"],
                            "Knowledge retrieval is unavailable.",
                        )
                    knowledge_results = retrieve_knowledge(
                        retriever=self.retriever,
                        repository=repository,
                        user_id=context.request.user_id,
                        query=context.rewritten_query,
                    )
                    if knowledge_results:
                        return BranchResult(
                            response_type=ResponseType.NORMAL,
                            normal_response_text=str(
                                knowledge_results[0].payload.get("text", "")
                            ),
                        )
                    return self._generate_clarification(
                        context,
                        ["target_description"],
                        "No single stored knowledge item matched the lookup.",
                    )
                if detection.requires_clarification:
                    return self._generate_clarification(context, getattr(detection, "missing_fields", []), "Missing fields for knowledge action.")
                if detection.metadata:
                    actions = detection.metadata.get("knowledge_actions", [])
                    context.request.metadata.update(detection.metadata)
                    
        if not actions:
            return self._generate_clarification(context, [], "No valid knowledge action detected.")
            
        approved_conv_history: list[dict[str, Any]] = []
        if context.approved_conversation_context:
            approved_conv_history = context.approved_conversation_context.approved_conversation_history

        executable_actions = []
        pre_repo_results = []
        prevalidated_actions = [
            _validated_knowledge_from_dict(action)
            for action in context.request.metadata.get("validated_knowledge_actions", [])
        ] if context.request.metadata.get("confirmation_approved") else []
        if self.validated_action_builder:
            if prevalidated_actions:
                validated_actions = prevalidated_actions
            else:
                validated_actions = self.validated_action_builder.build_knowledge_actions(
                    context.request.user_id,
                    actions,
                    context.request.raw_query,
                    context.rewritten_query,
                    repository,
                )
            clarification_needed = False
            clarification_missing_fields: list[str] = []
            for v_act in validated_actions:
                if v_act.validation_result in (ActionValidationResult.CLARIFY_AMBIGUOUS_TARGET, ActionValidationResult.CLARIFY_MISSING_FIELDS):
                    clarification_needed = True
                    if v_act.action is KnowledgeAction.MODIFY and not v_act.new_text:
                        clarification_missing_fields.append("replacement_text")
                    elif v_act.action in {KnowledgeAction.MODIFY, KnowledgeAction.DELETE}:
                        clarification_missing_fields.append("target_description")
            
            if clarification_needed:
                if self.config.mutation_policy.partial_execution_policy == MutationPartialExecutionPolicy.ALL_OR_NOTHING:
                    return self._generate_clarification(context, clarification_missing_fields, "Action requires clarification.")
                else:
                    return self._generate_clarification(context, clarification_missing_fields, "Ambiguous destructive actions block partial execution.")
                    
            from .contracts import RepositoryActionResult
            for v_act in validated_actions:
                if v_act.validation_result == ActionValidationResult.EXECUTE:
                    executable_actions.append(v_act)
                else:
                    status = "validation_failed"
                    if v_act.validation_result == ActionValidationResult.SKIP_NOT_FOUND:
                        status = "not_found"
                    user_safe_summary = f"Action skipped due to {v_act.validation_result.value}."
                    if v_act.validation_result == ActionValidationResult.SKIP_NOT_FOUND:
                        user_safe_summary = "No matching knowledge item was found."
                    pre_repo_results.append(RepositoryActionResult(
                        action_id="pre-repo",
                        action_type=v_act.action.value,
                        status=status,
                        domain_entity_type="knowledge_chunk",
                        user_safe_summary=user_safe_summary,
                        reason_summary=v_act.reason_summary or f"Validation resulted in {v_act.validation_result.value}",
                    ))
                    
            if not executable_actions and not clarification_needed:
                response_text = self.prompt_registry.message("knowledge_no_op") if hasattr(self.prompt_registry, "message") else "No matching knowledge item was found."
                result = repository.record_action_audit_noop(
                    user_id=context.request.user_id,
                    topic_title=context.request.metadata.get("topic_title", "Knowledge"),
                    raw_user_query=context.request.raw_query,
                    rewritten_user_query=context.rewritten_query,
                    response_text=response_text,
                    intent=Intent.KNOWLEDGE_FACTS.value,
                    response_type=ResponseType.SAFE_NOOP.value,
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
        executable_actions = self._expand_semantic_knowledge_actions(executable_actions)

        if executable_actions and not context.request.metadata.get("confirmation_approved"):
            for action in executable_actions:
                if not isinstance(action, ValidatedKnowledgeAction) or action.action is not KnowledgeAction.ADD:
                    continue
                new_text = action.knowledge_text or action.new_text or ""
                conflict = _find_simple_knowledge_conflict(
                    repository,
                    user_id=context.request.user_id,
                    new_text=new_text,
                )
                if not conflict:
                    continue
                replacement = replace(
                    action,
                    action=KnowledgeAction.MODIFY,
                    target_chunk_ids=(conflict["chunk_id"],),
                    observed_versions={conflict["chunk_id"]: conflict["version"]},
                    replacement_text=new_text,
                    new_text=new_text,
                    target_description=conflict.get("summary") or conflict.get("normalized_text"),
                    reason_summary="conflicting_fact_replace",
                )
                confirmation = repository.create_pending_confirmation(
                    user_id=context.request.user_id,
                    action_type="knowledge_conflict_replace",
                    target_entity_type="knowledge_chunk",
                    target_entity_id=conflict["chunk_id"],
                    proposed_action={
                        "domain": "knowledge",
                        "actions": _json_safe([replacement]),
                        "operation_response": "Updated the conflicting knowledge.",
                        "topic_title": action.topic_title or context.request.metadata.get("topic_title", "Knowledge"),
                    },
                    target_snapshot={
                        "chunk_id": conflict["chunk_id"],
                        "version": conflict["version"],
                        "old_text": conflict.get("normalized_text"),
                        "new_text": new_text,
                    },
                    expires_at=_expires_at(self.config),
                )
                return _pending_confirmation_result(
                    response_type=ResponseType.KNOWLEDGE_ACTION,
                    text="I found an existing fact that appears to conflict. Please confirm whether I should replace the old fact.",
                    confirmation=confirmation,
                )

        if (
            executable_actions
            and not context.request.metadata.get("confirmation_approved")
            and any(
                getattr(action, "action", None) in {KnowledgeAction.MODIFY, KnowledgeAction.DELETE}
                for action in executable_actions
            )
        ) or (
            executable_actions
            and not context.request.metadata.get("confirmation_approved")
            and len(executable_actions) > 1
        ):
            first_action = executable_actions[0]
            target_id = None
            if hasattr(first_action, "target_chunk_ids"):
                target_ids = getattr(first_action, "target_chunk_ids") or ()
                target_id = target_ids[0] if target_ids else None
            confirmation = repository.create_pending_confirmation(
                user_id=context.request.user_id,
                action_type="knowledge_mutation",
                target_entity_type="knowledge_chunk",
                target_entity_id=target_id,
                proposed_action={
                    "domain": "knowledge",
                    "actions": _json_safe(executable_actions),
                    "operation_response": success_text,
                    "topic_title": context.request.metadata.get("topic_title", "Knowledge"),
                },
                target_snapshot={"actions": _json_safe(executable_actions)},
                expires_at=_expires_at(self.config),
            )
            return _pending_confirmation_result(
                response_type=ResponseType.KNOWLEDGE_ACTION,
                text="Please confirm this knowledge change before I apply it.",
                confirmation=confirmation,
            )
        
        result = repository.transactional_knowledge_actions(
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
                knowledge_operation_results=pre_repo_results + list(result.results),
                linked_hop_id=result.audit_hop_id,
            )
            
        return BranchResult(
            response_type=ResponseType.ERROR,
            fallback_or_error_message=result.reason_summary,
        )


@dataclass
class ReminderBranch:

    config: AssistantConfig
    action_detector: ActionDetector | None = None
    clarification_strategy: QuestionGenerationStrategy | None = None
    reminder_supporting_strategy: QuestionGenerationStrategy | None = None
    prompt_registry: PromptRegistry = field(default_factory=lambda: DEFAULT_PROMPT_REGISTRY)
    validated_action_builder: ValidatedActionBuilder | None = None
    retriever: HybridRetriever | None = None
    context_filter: ContextFilter | None = None
    llm: LLMClient | None = None

    def _generate_clarification(self, context: PipelineContext, missing_fields: list[str], ambiguity_reason: str) -> BranchResult:
        question = GeneratedQuestion(
            text=self.prompt_registry.message(
                _mutation_clarification_message_key("reminder", missing_fields)
            ),
            source=QuestionSource.CLARIFICATION_QUESTION,
            purpose="resolve_missing_info",
            confidence=1.0,
        )
        return BranchResult(
            response_type=ResponseType.CLARIFICATION,
            clarification_question=question,
        )

    def execute(self, context: PipelineContext, repository: AssistantRepository) -> BranchResult:
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
            
        approved_conv_history: list[dict[str, Any]] = []
        if context.approved_conversation_context:
            approved_conv_history = context.approved_conversation_context.approved_conversation_history

        executable_actions = []
        pre_repo_results = []
        prevalidated_actions = [
            _validated_reminder_from_dict(action)
            for action in context.request.metadata.get("validated_reminder_actions", [])
        ] if context.request.metadata.get("confirmation_approved") else []
        if self.validated_action_builder:
            if prevalidated_actions:
                validated_actions = prevalidated_actions
            else:
                validated_actions = self.validated_action_builder.build_reminder_actions(
                    context.request.user_id, 
                    actions, 
                    context.request.raw_query, 
                    context.rewritten_query,
                    repository,
                )
            clarification_needed = False
            clarification_missing_fields: list[str] = []
            for v_act in validated_actions:
                if v_act.validation_result in (ActionValidationResult.CLARIFY_AMBIGUOUS_TARGET, ActionValidationResult.CLARIFY_MISSING_FIELDS):
                    clarification_needed = True
                    if v_act.action is ReminderAction.ADD:
                        if not v_act.reminder_time:
                            clarification_missing_fields.append("reminder_time")
                        if not v_act.subject:
                            clarification_missing_fields.append("subject")
                    elif v_act.action is ReminderAction.MODIFY:
                        if not v_act.replacement_time and not v_act.replacement_subject:
                            clarification_missing_fields.append("new_reminder_time")
                        else:
                            clarification_missing_fields.append("target_description")
                    else:
                        clarification_missing_fields.append("target_description")
            
            if clarification_needed:
                if self.config.mutation_policy.partial_execution_policy == MutationPartialExecutionPolicy.ALL_OR_NOTHING:
                    return self._generate_clarification(context, clarification_missing_fields, "Action requires clarification.")
                else:
                    return self._generate_clarification(context, clarification_missing_fields, "Ambiguous actions block execution.")
                    
            from .contracts import RepositoryActionResult
            for v_act in validated_actions:
                if v_act.validation_result == ActionValidationResult.EXECUTE:
                    executable_actions.append(v_act)
                else:
                    status = "validation_failed"
                    if v_act.validation_result == ActionValidationResult.SKIP_NOT_FOUND:
                        status = "not_found"
                    user_safe_summary = f"Action skipped due to {v_act.validation_result.value}."
                    if v_act.validation_result == ActionValidationResult.SKIP_NOT_FOUND:
                        user_safe_summary = "No matching reminder was found."
                    pre_repo_results.append(RepositoryActionResult(
                        action_id="pre-repo",
                        action_type=v_act.action.value,
                        status=status,
                        domain_entity_type="reminder",
                        user_safe_summary=user_safe_summary,
                        reason_summary=v_act.reason_summary or f"Validation resulted in {v_act.validation_result.value}",
                    ))
                    
            if not executable_actions and not clarification_needed:
                response_text = self.prompt_registry.message("reminder_no_op") if hasattr(self.prompt_registry, "message") else "No reminder needed to be changed."
                result = repository.record_action_audit_noop(
                    user_id=context.request.user_id,
                    topic_title=context.request.metadata.get("topic_title", "Reminders"),
                    raw_user_query=context.request.raw_query,
                    rewritten_user_query=context.rewritten_query,
                    response_text=response_text,
                    intent=Intent.REMINDER.value,
                    response_type=ResponseType.SAFE_NOOP.value,
                    parent_hop_id=context.request.parent_hop_id,
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
        warnings: list[str] = []
        normalized_actions: list[ValidatedReminderAction] = []
        normalizer = ReminderTimeNormalizer(default_timezone=self.config.default_timezone)
        request_timezone = context.request.platform_context.get("timezone") if context.request.platform_context else None
        for action in executable_actions:
            if not isinstance(action, ValidatedReminderAction):
                normalized_actions.append(action)
                continue
            updated_action = action
            try:
                if action.action is ReminderAction.ADD and action.reminder_time:
                    # Store the user's source timestamp.  Autoscan, rather
                    # than this online request, derives the notification time.
                    source_time = action.event_time or action.reminder_time
                    normalized = normalizer.normalize(
                        source_time,
                        platform_timezone=action.user_timezone or request_timezone,
                        original_time_text=action.original_time_text
                        or context.request.metadata.get("original_time_text")
                        or context.request.raw_query,
                    )
                    updated_action = replace(
                        action,
                        reminder_time=normalized.reminder_time_utc,
                        event_time=normalized.reminder_time_utc,
                        user_timezone=normalized.user_timezone,
                        original_time_text=normalized.original_time_text,
                    )
                    if normalized.warning:
                        warnings.append(normalized.warning)
                elif action.action is ReminderAction.MODIFY:
                    target_time = action.replacement_time or action.reminder_time
                    if target_time:
                        normalized = normalizer.normalize(
                            target_time,
                            platform_timezone=action.user_timezone or request_timezone,
                            original_time_text=action.original_time_text
                            or context.request.metadata.get("original_time_text")
                            or context.request.raw_query,
                        )
                        updated_action = replace(
                            action,
                            replacement_time=normalized.reminder_time_utc if action.replacement_time else action.replacement_time,
                            reminder_time=normalized.reminder_time_utc if not action.replacement_time else action.reminder_time,
                            event_time=normalized.reminder_time_utc,
                            user_timezone=normalized.user_timezone,
                            original_time_text=normalized.original_time_text,
                        )
                        if normalized.warning:
                            warnings.append(normalized.warning)
            except ReminderTimeNormalizationError as exc:
                return self._generate_clarification(context, ["timezone"], str(exc))
            normalized_actions.append(updated_action)
        executable_actions = normalized_actions

        if not context.request.metadata.get("confirmation_approved"):
            for action in executable_actions:
                if not isinstance(action, ValidatedReminderAction):
                    continue
                if action.action is ReminderAction.ADD and action.reminder_time:
                    duplicate = repository.find_active_reminder_duplicates(
                        user_id=context.request.user_id,
                        subject=action.subject or action.reminder_summary or action.raw_reminder or "",
                        reminder_time=action.reminder_time,
                    )
                    if duplicate.get("type") == "exact":
                        response_text = "That reminder already exists, so I did not create a duplicate."
                        result = repository.record_action_audit_noop(
                            user_id=context.request.user_id,
                            topic_title=context.request.metadata.get("topic_title", "Reminders"),
                            raw_user_query=context.request.raw_query,
                            rewritten_user_query=context.rewritten_query,
                            response_text=response_text,
                            intent=Intent.REMINDER.value,
                            response_type=ResponseType.SAFE_NOOP.value,
                            parent_hop_id=context.request.parent_hop_id,
                        )
                        return BranchResult(
                            response_type=ResponseType.SAFE_NOOP,
                            normal_response_text=response_text,
                            reminder_operation_results=list(result.results),
                            linked_hop_id=result.audit_hop_id,
                            warnings=warnings,
                        )
                    if duplicate.get("type") == "similar":
                        candidate = duplicate.get("candidate", {})
                        subject = candidate.get("subject") or "an existing reminder"
                        question = GeneratedQuestion(
                            text=f"I found a similar reminder, '{subject}'. Should I keep both, replace the old one, or cancel this new reminder?",
                            source=QuestionSource.CLARIFICATION_QUESTION,
                            purpose="resolve_duplicate_reminder",
                            confidence=1.0,
                        )
                        return BranchResult(
                            response_type=ResponseType.CLARIFICATION,
                            clarification_question=question,
                            warnings=warnings,
                        )

        destructive_turn_off_count = sum(
            1
            for action in executable_actions
            if isinstance(action, ValidatedReminderAction) and action.action is ReminderAction.TURN_OFF
        )
        dangerous_reminder_action = any(
            isinstance(action, ValidatedReminderAction)
            and (
                action.action is ReminderAction.DELETE
                or (
                    action.action is ReminderAction.MODIFY
                    and action.confidence < self.config.confirmation_high_confidence_threshold
                )
            )
            for action in executable_actions
        ) or destructive_turn_off_count > 1
        if executable_actions and dangerous_reminder_action and not context.request.metadata.get("confirmation_approved"):
            first_action = executable_actions[0]
            target_id = None
            if isinstance(first_action, ValidatedReminderAction) and first_action.target_reminder_ids:
                target_id = first_action.target_reminder_ids[0]
            confirmation = repository.create_pending_confirmation(
                user_id=context.request.user_id,
                action_type="reminder_mutation",
                target_entity_type="reminder",
                target_entity_id=target_id,
                proposed_action={
                    "domain": "reminder",
                    "actions": _json_safe(executable_actions),
                    "operation_response": success_text,
                    "topic_title": context.request.metadata.get("topic_title", "Reminders"),
                },
                target_snapshot={"actions": _json_safe(executable_actions)},
                expires_at=_expires_at(self.config),
            )
            return _pending_confirmation_result(
                response_type=ResponseType.REMINDER_ACTION,
                text="Please confirm this reminder change before I apply it.",
                confirmation=confirmation,
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

        result = repository.transactional_reminder_actions(
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
                warnings=warnings,
            )

        return BranchResult(
            response_type=ResponseType.ERROR,
            fallback_or_error_message=result.reason_summary,
        )


class BranchRouter:
    def __init__(self, branches: dict[Intent, Branch]) -> None:
        self.branches = branches

    def route(self, context: PipelineContext, repository: AssistantRepository) -> BranchResult:
        if context.intent not in self.branches:
            raise ValueError(f"No branch registered for intent: {context.intent}")
        return self.branches[context.intent].execute(context, repository)
