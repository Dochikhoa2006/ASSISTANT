"""Intent branch implementations."""

from __future__ import annotations

import json
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
    PersistenceMode,
    PipelineContext,
    QuestionSource,
    ReminderAction,
    KnowledgeAction,
    OutboxEntityType,
    OutboxOperation,
    RepositoryActionResult,
    ResponseType,
    SubBranchPromptContext,
    ValidatedReminderAction,
    ValidatedKnowledgeAction,
)

SUB_BRANCH_PROMPT_POLICIES = {
    GeneralSubBranch.SUPPORT_QUESTION_ANSWER: {
        "chat_history_role": "User is answering a prior supporting question.",
        "response_goal": "Enhance/refine prior answer using current reply.",
        "database_update_mode": "append_to_existing_topic",
        "allowed_database_updates": ("conversation_hop_append",),
        "prohibited_database_updates": ("new_topic_creation", "knowledge_mutation", "reminder_mutation", "notification_write"),
    },
    GeneralSubBranch.CONVERSATION_FOLLOW_UP: {
        "chat_history_role": "User is continuing an approved existing conversation.",
        "response_goal": "Use chat history to stay on topic.",
        "database_update_mode": "append_to_existing_topic",
        "allowed_database_updates": ("conversation_hop_append",),
        "prohibited_database_updates": ("new_topic_creation", "knowledge_mutation", "reminder_mutation", "notification_write"),
    },
    GeneralSubBranch.NEW_CONVERSATION_TOPIC: {
        "chat_history_role": "User is starting fresh.",
        "response_goal": "Answer directly without forcing old context.",
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
from .context_filter import ContextFilter
from .action_detection import (
    ActionDetector,
)
from .generation import QuestionGenerationStrategy
from .branch_orchestration import ValidatedActionBuilder
from .settings import MutationPartialExecutionPolicy
from .reminder_safety import ReminderTimeNormalizer, ReminderTimeNormalizationError
from .chat_history import canonical_chat_history_scope, current_chat_history


def _mutation_clarification_message_key(domain: str, missing_fields: list[str]) -> str:
    """Choose a fast, user-facing question for predictable mutation gaps."""
    missing = {str(field).casefold() for field in missing_fields}
    if domain == "knowledge":
        if "factuality_confirmation" in missing:
            return "knowledge_factuality_confirmation"
        if "partial_chunk_delete" in missing:
            return "knowledge_partial_chunk_delete"
        if {"replacement_text", "replacement", "new_text"} & missing:
            return "knowledge_missing_replacement"
        if {"target_description", "target", "target_entities"} & missing:
            return "knowledge_missing_target"
        if {"text", "knowledge_text", "content"} & missing:
            return "knowledge_missing_content"
        return "knowledge_missing_action"

    if "factuality_confirmation" in missing:
        return "reminder_factuality_confirmation"
    if "reminder_validation_clarification" in missing:
        return "reminder_validation_clarification"
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


def _action_authorization(
    context: PipelineContext,
    intent: Intent,
    action_name: str,
) -> dict[str, Any]:
    existing = context.request.metadata.get("action_authorization") or {}
    if (
        existing.get("intent") == intent.value
        and str(existing.get("action") or "").casefold() == action_name.casefold()
    ):
        return dict(existing)
    return {
        "intent": intent.value,
        "action": action_name.casefold(),
        "source": "llm_action_extraction",
        "reason_summary": "Bound to the dedicated extraction LLM output and validated action.",
    }


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
        supporting_question=payload.get("supporting_question"),
        supporting_response=payload.get("supporting_response"),
        user_timezone=payload.get("user_timezone"),
        original_time_text=payload.get("original_time_text"),
        recurrence_rule=payload.get("recurrence_rule"),
        recurrence_timezone=payload.get("recurrence_timezone"),
        next_fire_time=_parse_dt(payload.get("next_fire_time")),
        parent_recurring_reminder_id=payload.get("parent_recurring_reminder_id"),
        timing_plan_required=bool(payload.get("timing_plan_required", True)),
        replacement_subject=payload.get("replacement_subject"),
        replacement_time=_parse_dt(payload.get("replacement_time")),
        replacement_summary=payload.get("replacement_summary"),
        replacement_recurrence_rule=payload.get("replacement_recurrence_rule"),
        replacement_recurrence_timezone=payload.get("replacement_recurrence_timezone"),
        confidence=float(payload.get("confidence", 1.0)),
        matched_fields=tuple(payload.get("matched_fields") or ()),
        reason_summary=payload.get("reason_summary"),
        requires_hitl=bool(payload.get("requires_hitl", False)),
        factuality_concern=bool(payload.get("factuality_concern", False)),
        hitl_reason=payload.get("hitl_reason"),
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
        requires_hitl=bool(payload.get("requires_hitl", False)),
        factuality_concern=bool(payload.get("factuality_concern", False)),
        hitl_reason=payload.get("hitl_reason"),
        clarification_question=payload.get("clarification_question"),
    )


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
        # A clarification decision belongs to the current turn. Always invoke
        # the configured generator instead of reusing a question carried in
        # request metadata from an earlier turn.
        question: GeneratedQuestion | None = None
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
            approved_conversation_history=list(context.chat_history),
        )
        merged_supporting_detail = self._merge_supporting_detail(approved_context)

        from .general_sub_branch import GeneralSubBranchValidator, GeneralPersistencePlanBuilder
        
        if self.general_purpose_config:
            if self.sub_branch_detector:
                decision = self.sub_branch_detector.detect(
                    context,
                    self.general_purpose_config,
                    merged_supporting_detail,
                )
            else:
                decision = GeneralSubBranchDecision(
                    sub_branch=GeneralSubBranch.NEW_CONVERSATION_TOPIC,
                    confidence=1.0,
                    persistence_mode=PersistenceMode.CREATE_NEW_TOPIC,
                    reason_summary=(
                        "Deterministic rule fired: NEW_CONVERSATION_TOPIC "
                        "because no detector is configured."
                    ),
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
                raw_user_query=context.rewritten_query,
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
                warnings=list(composer_result.content_warnings) if composer_result else [],
                # File creation completed before conversation persistence. Keep
                # every successful artifact visible in the chatbot even when
                # the hop transaction fails; platform delivery is blocked for
                # error responses by PlatformSelector.
                platform_payload={"artifacts": list(composer_result.artifacts)}
                if composer_result and composer_result.artifacts
                else {},
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
            indexing_job_result={
                "conversation_hop_job_id": hop.outbox_job_id,
                "outbox_job_ids": [hop.outbox_job_id],
            },
            platform_payload={"artifacts": list(composer_result.artifacts)} if composer_result and composer_result.artifacts else {},
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
    action_detector: ActionDetector
    clarification_strategy: QuestionGenerationStrategy | None = None
    prompt_registry: PromptRegistry = field(default_factory=lambda: DEFAULT_PROMPT_REGISTRY)
    validated_action_builder: ValidatedActionBuilder | None = None
    retriever: HybridRetriever | None = None
    context_filter: ContextFilter | None = None
    llm: LLMClient | None = None
    knowledge_mutation_pipeline: Any | None = None

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

    def execute(self, context: PipelineContext, repository: AssistantRepository) -> BranchResult:
        # Intent dispatch is authoritative. The dedicated extraction model is
        # deliberately the first executable component inside this branch.
        context_history = getattr(context, "chat_history", None)
        branch_chat_history = (
            current_chat_history()
            if context_history is None
            else list(context_history)
        )
        with canonical_chat_history_scope(branch_chat_history):
            detection = self.action_detector.detect(
                context.request, context.rewritten_query, Intent.KNOWLEDGE_FACTS
            )
        if detection.requires_clarification:
            return self._generate_clarification(
                context,
                getattr(detection, "missing_fields", []),
                "Missing fields for knowledge action.",
            )
        actions = list(detection.metadata.get("knowledge_actions") or [])
        context.request.metadata.update(detection.metadata)
                    
        if not actions:
            return self._generate_clarification(context, [], "No valid knowledge action detected.")

        executable_actions = []
        pre_repo_results = []
        prevalidated_actions = [
            _validated_knowledge_from_dict(action)
            for action in context.request.metadata.get("validated_knowledge_actions", [])
        ] if context.request.metadata.get("confirmation_approved") else []
        if prevalidated_actions and (
            len(actions) != 1
            or len(prevalidated_actions) != 1
            or str(actions[0].get("action") or "").casefold()
            != prevalidated_actions[0].action.value
        ):
            return self._generate_clarification(
                context,
                ["single_action"],
                "The confirmation extraction did not match the validated action.",
            )
        if self.knowledge_mutation_pipeline:
            # Even legacy pending confirmations are re-retrieved and passed
            # through LLM2 (and LLM3 for MODIFY). No knowledge execution may
            # bypass the validation model.
            validated_actions = [
                self.knowledge_mutation_pipeline.build_action(
                    context=context,
                    action_payload=actions[0],
                    repository=repository,
                )
            ]
            if len(validated_actions) != 1:
                return self._generate_clarification(
                    context,
                    ["single_action"],
                    "Exactly one knowledge action is required.",
                )
            clarification_needed = False
            clarification_missing_fields: list[str] = []
            for v_act in validated_actions:
                if v_act.hitl_reason == "internal_pipeline_failure":
                    return BranchResult(
                        response_type=ResponseType.ERROR,
                        fallback_or_error_message=self.prompt_registry.message(
                            "knowledge_pipeline_unavailable"
                        ),
                    )
                if v_act.hitl_reason == "knowledge_validation_fail":
                    if not (v_act.clarification_question or "").strip():
                        return BranchResult(
                            response_type=ResponseType.ERROR,
                            fallback_or_error_message=self.prompt_registry.message(
                                "knowledge_pipeline_unavailable"
                            ),
                        )
                    return BranchResult(
                        response_type=ResponseType.CLARIFICATION,
                        clarification_question=GeneratedQuestion(
                            text=v_act.clarification_question,
                            source=QuestionSource.CLARIFICATION_QUESTION,
                            purpose="knowledge_validation_fail",
                            confidence=v_act.confidence,
                        ),
                    )
                if v_act.validation_result in (ActionValidationResult.CLARIFY_AMBIGUOUS_TARGET, ActionValidationResult.CLARIFY_MISSING_FIELDS):
                    clarification_needed = True
                    if v_act.factuality_concern:
                        clarification_missing_fields.append(
                            "factuality_confirmation"
                        )
                    elif v_act.hitl_reason == "partial_chunk_delete":
                        clarification_missing_fields.append(
                            "partial_chunk_delete"
                        )
                    elif v_act.action is KnowledgeAction.MODIFY and not v_act.new_text:
                        clarification_missing_fields.append("replacement_text")
                    elif v_act.action in {KnowledgeAction.MODIFY, KnowledgeAction.DELETE}:
                        clarification_missing_fields.append("target_description")
                    elif v_act.action is KnowledgeAction.ADD:
                        clarification_missing_fields.append("text")
            
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
                    elif v_act.validation_result == ActionValidationResult.SKIP_ALREADY_EXISTS:
                        status = "skipped"
                    user_safe_summary = f"Action skipped due to {v_act.validation_result.value}."
                    if v_act.validation_result == ActionValidationResult.SKIP_NOT_FOUND:
                        user_safe_summary = "No matching knowledge item was found."
                    elif v_act.validation_result == ActionValidationResult.SKIP_ALREADY_EXISTS:
                        user_safe_summary = "That knowledge is already stored."
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
                    parent_hop_id=context.request.parent_hop_id,
                )
                return BranchResult(
                    response_type=ResponseType.KNOWLEDGE_ACTION,
                    normal_response_text=response_text,
                    knowledge_operation_results=pre_repo_results + list(result.results),
                    linked_topic_id=getattr(result, "audit_topic_id", None),
                    linked_hop_id=result.audit_hop_id,
                    database_write_result={
                        "conversation_hop_id": result.audit_hop_id,
                    },
                    indexing_job_result={
                        "outbox_job_ids": list(
                            getattr(result, "indexing_outbox_ids", ())
                        ),
                    },
                )
        else:
            # Keep the legacy constructor field for compatibility, but never
            # allow a newly extracted action to bypass retrieval, the second
            # validation model, or final content generation.
            return BranchResult(
                response_type=ResponseType.ERROR,
                fallback_or_error_message=self.prompt_registry.message(
                    "knowledge_pipeline_unavailable"
                ),
            )

        success_text = context.request.metadata.get(
            "operation_response", self.prompt_registry.message("knowledge_updated") if hasattr(self.prompt_registry, "message") else "Knowledge updated successfully."
        )
        if len(executable_actions) != 1:
            return self._generate_clarification(
                context,
                ["single_action"],
                "Exactly one executable knowledge action is required.",
            )
        # LLM2 PASS is final authorization for knowledge mutations. ADD and
        # DELETE reach SQL immediately; MODIFY reaches SQL immediately after
        # its mandatory LLM3 finalization. Any needed user confirmation must
        # have been returned by LLM2 as FAIL plus a clarification question.
        result = repository.transactional_knowledge_actions(
            user_id=context.request.user_id,
            topic_title=context.request.metadata.get("topic_title", "Knowledge"),
            raw_user_query=context.request.raw_query,
            rewritten_user_query=context.rewritten_query,
            response_text=success_text,
            actions=executable_actions,
            parent_hop_id=context.request.parent_hop_id,
        )
        
        if result.committed:
            return BranchResult(
                response_type=ResponseType.KNOWLEDGE_ACTION,
                normal_response_text=success_text,
                knowledge_operation_results=pre_repo_results + list(result.results),
                linked_topic_id=getattr(result, "audit_topic_id", None),
                linked_hop_id=result.audit_hop_id,
                database_write_result={
                    "conversation_hop_id": result.audit_hop_id,
                },
                indexing_job_result={
                    "outbox_job_ids": list(
                        getattr(result, "indexing_outbox_ids", ())
                    ),
                },
            )
            
        return BranchResult(
            response_type=ResponseType.ERROR,
            fallback_or_error_message=result.reason_summary,
        )


@dataclass
class ReminderBranch:

    config: AssistantConfig
    action_detector: ActionDetector
    clarification_strategy: QuestionGenerationStrategy | None = None
    reminder_supporting_strategy: QuestionGenerationStrategy | None = None
    prompt_registry: PromptRegistry = field(default_factory=lambda: DEFAULT_PROMPT_REGISTRY)
    validated_action_builder: ValidatedActionBuilder | None = None
    retriever: HybridRetriever | None = None
    context_filter: ContextFilter | None = None
    llm: LLMClient | None = None
    reminder_mutation_pipeline: Any | None = None

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
        # Intent dispatch is authoritative. The dedicated extraction model is
        # deliberately the first executable component inside this branch.
        context_history = getattr(context, "chat_history", None)
        branch_chat_history = (
            current_chat_history()
            if context_history is None
            else list(context_history)
        )
        with canonical_chat_history_scope(branch_chat_history):
            detection = self.action_detector.detect(
                context.request, context.rewritten_query, Intent.REMINDER
            )
        if detection.requires_clarification:
            return self._generate_clarification(
                context,
                getattr(detection, "missing_fields", []),
                "Missing fields for reminder action.",
            )
        actions = list(detection.metadata.get("reminder_actions") or [])
        context.request.metadata.update(detection.metadata)
                    
        if not actions:
            return self._generate_clarification(context, [], "No valid reminder action detected.")

        executable_actions = []
        pre_repo_results = []
        prevalidated_actions = [
            _validated_reminder_from_dict(action)
            for action in context.request.metadata.get("validated_reminder_actions", [])
        ] if context.request.metadata.get("confirmation_approved") else []
        if prevalidated_actions and (
            len(actions) != 1
            or len(prevalidated_actions) != 1
            or str(actions[0].get("action") or "").casefold()
            != prevalidated_actions[0].action.value
        ):
            return self._generate_clarification(
                context,
                ["single_action"],
                "The confirmation extraction did not match the validated action.",
            )
        if self.reminder_mutation_pipeline or self.validated_action_builder:
            if prevalidated_actions:
                validated_actions = prevalidated_actions
            elif self.reminder_mutation_pipeline:
                validated_actions = [
                    self.reminder_mutation_pipeline.build_action(
                        context=context,
                        action_payload=actions[0],
                        repository=repository,
                    )
                ]
            else:
                assert self.validated_action_builder is not None
                validated_actions = self.validated_action_builder.build_reminder_actions(
                    context.request.user_id, 
                    actions, 
                    context.rewritten_query,
                    context.rewritten_query,
                    repository,
                )
            if len(validated_actions) != 1:
                return self._generate_clarification(
                    context,
                    ["single_action"],
                    "Exactly one reminder action is required.",
                )
            clarification_needed = False
            clarification_missing_fields: list[str] = []
            for v_act in validated_actions:
                if v_act.hitl_reason == "internal_pipeline_failure":
                    return BranchResult(
                        response_type=ResponseType.ERROR,
                        fallback_or_error_message=self.prompt_registry.message(
                            # TODO: add reminder_pipeline_unavailable prompt key.
                            "knowledge_pipeline_unavailable"
                        ),
                    )
                if v_act.hitl_reason == "reminder_validation_clarification":
                    clarification_q = getattr(v_act, "clarification_question", None)
                    if clarification_q and clarification_q.strip():
                        return BranchResult(
                            response_type=ResponseType.CLARIFICATION,
                            clarification_question=GeneratedQuestion(
                                text=clarification_q,
                                source=QuestionSource.CLARIFICATION_QUESTION,
                                purpose="reminder_validation_fail",
                                confidence=v_act.confidence,
                            ),
                        )
                if v_act.validation_result in (ActionValidationResult.CLARIFY_AMBIGUOUS_TARGET, ActionValidationResult.CLARIFY_MISSING_FIELDS):
                    clarification_needed = True
                    if v_act.factuality_concern:
                        clarification_missing_fields.append(
                            "factuality_confirmation"
                        )
                    elif v_act.hitl_reason:
                        clarification_missing_fields.append(
                            "reminder_validation_clarification"
                        )
                    elif v_act.action is ReminderAction.ADD:
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
                    elif v_act.validation_result == ActionValidationResult.SKIP_ALREADY_EXISTS:
                        status = "skipped"
                    user_safe_summary = f"Action skipped due to {v_act.validation_result.value}."
                    if v_act.validation_result == ActionValidationResult.SKIP_NOT_FOUND:
                        user_safe_summary = "No matching reminder was found."
                    elif v_act.validation_result == ActionValidationResult.SKIP_ALREADY_EXISTS:
                        if v_act.action is ReminderAction.ADD:
                            user_safe_summary = "That reminder already exists."
                        elif v_act.action is ReminderAction.MODIFY:
                            user_safe_summary = "That reminder already has the requested details."
                        else:
                            user_safe_summary = "That reminder is already in the requested state."
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
                    linked_topic_id=getattr(result, "audit_topic_id", None),
                    linked_hop_id=result.audit_hop_id,
                    database_write_result={
                        "conversation_hop_id": result.audit_hop_id,
                    },
                    indexing_job_result={
                        "outbox_job_ids": list(
                            getattr(result, "indexing_outbox_ids", ())
                        ),
                    },
                )
        else:
            # No mutation pipeline is configured. Never allow a newly extracted
            # action to bypass retrieval, the validation model, or finalization.
            return BranchResult(
                response_type=ResponseType.ERROR,
                fallback_or_error_message=self.prompt_registry.message(
                    # TODO: add reminder_pipeline_unavailable prompt key.
                    "knowledge_pipeline_unavailable"
                ),
            )

        success_text = context.request.metadata.get(
            "operation_response", self.prompt_registry.message("reminder_updated") if hasattr(self.prompt_registry, "message") else "Reminder updated successfully."
        )
        if len(executable_actions) != 1:
            return self._generate_clarification(
                context,
                ["single_action"],
                "Exactly one executable reminder action is required.",
            )
        warnings: list[str] = []
        normalized_actions: list[ValidatedReminderAction] = []
        normalizer = ReminderTimeNormalizer(default_timezone=self.config.default_timezone)
        request_timezone = context.request.platform_context.get("timezone") if context.request.platform_context else None
        for action in executable_actions:
            if not isinstance(action, ValidatedReminderAction):
                normalized_actions.append(action)
                continue
            if action.action not in {ReminderAction.ADD, ReminderAction.MODIFY}:
                normalized_actions.append(action)
                continue
            updated_action = action
            try:
                source_time_text = (
                    action.original_time_text
                    or context.request.metadata.get("original_time_text")
                    or context.rewritten_query
                )
                normalized_notification = (
                    normalizer.normalize(
                        action.reminder_time,
                        platform_timezone=action.user_timezone or request_timezone,
                        original_time_text=source_time_text,
                    )
                    if action.reminder_time
                    else None
                )
                normalized_event = (
                    normalizer.normalize(
                        action.event_time,
                        platform_timezone=action.user_timezone or request_timezone,
                        original_time_text=source_time_text,
                    )
                    if action.event_time
                    else None
                )
                for normalized in (normalized_notification, normalized_event):
                    if normalized and normalized.warning and normalized.warning not in warnings:
                        warnings.append(normalized.warning)

                if action.action is ReminderAction.ADD:
                    normalized_source = normalized_notification or normalized_event
                    updated_action = replace(
                        action,
                        reminder_time=(
                            normalized_notification.reminder_time_utc
                            if normalized_notification
                            else None
                        ),
                        event_time=(
                            normalized_event.reminder_time_utc
                            if normalized_event
                            else None
                        ),
                        user_timezone=(
                            normalized_source.user_timezone
                            if normalized_source
                            else action.user_timezone
                        ),
                        original_time_text=(
                            normalized_source.original_time_text
                            if normalized_source
                            else action.original_time_text
                        ),
                    )
                elif action.action is ReminderAction.MODIFY:
                    normalized_replacement = (
                        normalizer.normalize(
                            action.replacement_time,
                            platform_timezone=action.user_timezone or request_timezone,
                            original_time_text=source_time_text,
                        )
                        if action.replacement_time
                        else None
                    )
                    if (
                        normalized_replacement
                        and normalized_replacement.warning
                        and normalized_replacement.warning not in warnings
                    ):
                        warnings.append(normalized_replacement.warning)
                    normalized_source = (
                        normalized_replacement
                        or normalized_notification
                        or normalized_event
                    )
                    updated_action = replace(
                        action,
                        replacement_time=(
                            normalized_replacement.reminder_time_utc
                            if normalized_replacement
                            else action.replacement_time
                        ),
                        reminder_time=(
                            normalized_notification.reminder_time_utc
                            if normalized_notification
                            else action.reminder_time
                        ),
                        event_time=(
                            normalized_event.reminder_time_utc
                            if normalized_event
                            else action.event_time
                        ),
                        user_timezone=(
                            normalized_source.user_timezone
                            if normalized_source
                            else action.user_timezone
                        ),
                        original_time_text=(
                            normalized_source.original_time_text
                            if normalized_source
                            else action.original_time_text
                        ),
                    )
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
                            linked_topic_id=getattr(
                                result,
                                "audit_topic_id",
                                None,
                            ),
                            linked_hop_id=result.audit_hop_id,
                            database_write_result={
                                "conversation_hop_id": result.audit_hop_id,
                            },
                            indexing_job_result={
                                "outbox_job_ids": list(
                                    getattr(
                                        result,
                                        "indexing_outbox_ids",
                                        (),
                                    )
                                ),
                            },
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
                    "action_authorization": _action_authorization(
                        context,
                        Intent.REMINDER,
                        getattr(getattr(first_action, "action", None), "value", ""),
                    ),
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
                supporting_payload = json.dumps({
                    "question_text": reminder_supporting_question.text,
                    "question_source": reminder_supporting_question.source.value,
                    "purpose": reminder_supporting_question.purpose,
                    "confidence": reminder_supporting_question.confidence,
                })
                if isinstance(executable_actions[0], dict):
                    executable_actions[0]["supporting_question"] = supporting_payload

        result = repository.transactional_reminder_actions(
            user_id=context.request.user_id,
            topic_title=context.request.metadata.get("topic_title", "Reminders"),
            raw_user_query=context.request.raw_query,
            rewritten_user_query=context.rewritten_query,
            response_text=success_text,
            actions=executable_actions,
            parent_hop_id=context.request.parent_hop_id,
        )
        
        if result.committed:
            return BranchResult(
                response_type=ResponseType.REMINDER_ACTION,
                normal_response_text=success_text,
                human_supporting_questions=[],
                reminder_supporting_question=reminder_supporting_question,
                reminder_operation_results=pre_repo_results + list(result.results),
                linked_topic_id=getattr(result, "audit_topic_id", None),
                linked_hop_id=result.audit_hop_id,
                database_write_result={
                    "conversation_hop_id": result.audit_hop_id,
                },
                indexing_job_result={
                    "outbox_job_ids": list(
                        getattr(result, "indexing_outbox_ids", ())
                    ),
                },
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
        branch = self.branches[context.intent]
        if context.intent is Intent.CLARIFICATION:
            return branch.execute(context, repository)

        try:
            result = branch.execute(context, repository)
        except Exception:
            # A reached main branch must still produce a durable, indexable
            # outcome hop. Do not expose the internal exception to the user.
            result = BranchResult(
                response_type=ResponseType.ERROR,
                fallback_or_error_message=DEFAULT_PROMPT_REGISTRY.message(
                    "bundler_empty"
                ),
            )

        # Successful mutation/general transactions normally create their hop
        # atomically. Verify that identity and ensure its durable outbox job
        # before allowing the branch to bypass fallback persistence.
        if result.linked_hop_id:
            try:
                indexed_hop = repository.load_outbox_entity(
                    entity_type=OutboxEntityType.CONVERSATION_HOP.value,
                    entity_id=result.linked_hop_id,
                )
                if indexed_hop.user_id != context.request.user_id:
                    raise ValueError("Conversation hop belongs to another user")
                with repository.transaction() as cursor:
                    ensured_hop_job_id = repository.insert_outbox_job(
                        cursor,
                        entity_type=OutboxEntityType.CONVERSATION_HOP,
                        entity_id=result.linked_hop_id,
                        operation=OutboxOperation.UPSERT,
                    )
                existing_job_ids = list(
                    result.indexing_job_result.get("outbox_job_ids") or []
                )
                normalized_job_ids = list(
                    dict.fromkeys([*existing_job_ids, ensured_hop_job_id])
                )
                return replace(
                    result,
                    linked_topic_id=str(indexed_hop.metadata["topic_id"]),
                    database_write_result={
                        **result.database_write_result,
                        "conversation_hop_id": result.linked_hop_id,
                    },
                    indexing_job_result={
                        **result.indexing_job_result,
                        "outbox_job_ids": normalized_job_ids,
                        "conversation_hop_job_id": ensured_hop_job_id,
                    },
                )
            except Exception:
                # A branch-provided identity that cannot be proven from SQL is
                # not a persisted outcome. Clear it and use the normal atomic
                # fallback write below.
                result = replace(
                    result,
                    linked_topic_id=None,
                    linked_hop_id=None,
                    database_write_result={},
                    indexing_job_result={},
                )

        from .bundler import render_branch_result_text

        supporting_questions: list[dict[str, Any]] = []
        if result.clarification_question is not None:
            supporting_questions.append(
                {
                    "question_text": result.clarification_question.text,
                    "question_source": result.clarification_question.source.value,
                    "purpose": result.clarification_question.purpose,
                    "confidence": result.clarification_question.confidence,
                }
            )
        supporting_questions.extend(
            {
                "question_text": question.text,
                "question_source": question.source.value,
                "purpose": question.purpose,
                "confidence": question.confidence,
            }
            for question in result.human_supporting_questions
        )
        if result.reminder_supporting_question is not None:
            supporting_questions.append(
                {
                    "question_text": result.reminder_supporting_question.text,
                    "question_source": (
                        result.reminder_supporting_question.source.value
                    ),
                    "purpose": result.reminder_supporting_question.purpose,
                    "confidence": result.reminder_supporting_question.confidence,
                }
            )

        default_titles = {
            Intent.GENERAL_RESPONSE: getattr(
                getattr(branch, "general_purpose_config", None),
                "general_response_default_topic_title",
                "General Conversation",
            ),
            Intent.KNOWLEDGE_FACTS: "Knowledge",
            Intent.REMINDER: "Reminders",
        }
        audit = repository.record_branch_outcome(
            user_id=context.request.user_id,
            topic_title=str(
                context.request.metadata.get("topic_title")
                or default_titles[context.intent]
            ),
            raw_user_query=context.request.raw_query,
            rewritten_user_query=context.rewritten_query,
            response_text=render_branch_result_text(
                result,
                getattr(branch, "prompt_registry", DEFAULT_PROMPT_REGISTRY),
            ),
            intent=context.intent.value,
            response_type=result.response_type.value,
            supporting_questions=supporting_questions,
            parent_hop_id=context.request.parent_hop_id,
            entities={
                "branch_outcome": {
                    "intent": context.intent.value,
                    "response_type": result.response_type.value,
                }
            },
        )
        if not audit.committed or not audit.audit_hop_id:
            raise RuntimeError(
                audit.reason_summary
                or "The branch outcome could not be persisted."
            )

        outbox_ids = list(audit.indexing_outbox_ids)
        return replace(
            result,
            linked_topic_id=audit.audit_topic_id,
            linked_hop_id=audit.audit_hop_id,
            database_write_result={
                **result.database_write_result,
                "conversation_hop_id": audit.audit_hop_id,
            },
            indexing_job_result={
                **result.indexing_job_result,
                "outbox_job_ids": outbox_ids,
                "conversation_hop_job_id": (
                    outbox_ids[0] if outbox_ids else None
                ),
            },
        )
