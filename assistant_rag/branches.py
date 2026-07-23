"""Intent branch implementations."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
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
from .llm import (
    LLMClient,
    LLMTask,
    StructuredOutputInvariantError,
    is_structured_fallback,
)
from .prompts import DEFAULT_PROMPT_REGISTRY, PromptContext, PromptRegistry
from .retrieval import HybridRetriever
from .canonical_retrieval import retrieve_knowledge
from .reminder_retrieval import retrieve_reminder_candidates, reminder_candidate_to_context
from .context_filter import ApprovedContext, ContextFilter
from .action_detection import (
    ActionDetector,
)
from .generation import QuestionGenerationStrategy
from .branch_orchestration import ValidatedActionBuilder
from .settings import MutationPartialExecutionPolicy
from .chat_history import canonical_chat_history_scope, current_chat_history
from .reminder_reply import (
    reminder_reply_hop_entities,
    verified_reminder_state,
)
from .semantic_actions import semantic_action_from_internal_payload
from .answer_grounding import (
    AnswerGenerationOutcome,
    generate_answer,
    is_grounded_answer_text,
)


_PERSONAL_CONTEXT_DEPENDENCY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "requires_unavailable_context": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "reason_summary": {"type": "string"},
    },
    "required": [
        "requires_unavailable_context",
        "confidence",
        "reason_summary",
    ],
}

_PERSONAL_CONTEXT_DEGRADED_ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "answerable_without_unavailable_context": {"type": "boolean"},
        "answer_text": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "reason_summary": {"type": "string"},
    },
    "required": [
        "answerable_without_unavailable_context",
        "answer_text",
        "confidence",
        "reason_summary",
    ],
}

_PERSONAL_CONTEXT_CERTIFICATE_MIN_CONFIDENCE = 0.9


@dataclass(frozen=True)
class _DegradedContextAuthorization:
    certified_answer_text: str = ""
    answer_stage_attempted: bool = False

    @property
    def authorized(self) -> bool:
        return bool(self.certified_answer_text.strip())


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

    raise ValueError(f"Clarification messages are not supported for domain: {domain}")


def _normalized_question_text(value: str) -> str:
    return " ".join(value.casefold().split()).strip(" \t\r\n.,;:!?")


def _suppress_repeated_clarification(
    result: BranchResult,
    context: PipelineContext,
    prompt_registry: PromptRegistry,
) -> BranchResult:
    """Stop an identical consecutive clarification from becoming a loop."""

    current = result.clarification_question
    previous_state = getattr(context, "previous_last_qa_state", None)
    previous = (
        previous_state.clarification_question
        if previous_state is not None
        and previous_state.response_type is ResponseType.CLARIFICATION
        else None
    )
    if (
        result.response_type is not ResponseType.CLARIFICATION
        or current is None
        or previous is None
        or not _normalized_question_text(current.text)
        or _normalized_question_text(current.text)
        != _normalized_question_text(previous.text)
    ):
        return result

    return replace(
        result,
        response_type=ResponseType.SAFE_NOOP,
        normal_response_text=prompt_registry.message(
            "clarification_repeat_suppressed"
        ),
        clarification_question=None,
        fallback_or_error_message=None,
        warnings=list(result.warnings) + ["clarification_repeat_suppressed"],
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
                return BranchResult(
                    response_type=ResponseType.ERROR,
                    fallback_or_error_message=self.prompt_registry.message(
                        "clarification_generation_unavailable"
                    ),
                )
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

    def __post_init__(self) -> None:
        composer_path_available = (
            self.content_composer is not None
            and self.general_purpose_config is not None
        )
        if not composer_path_available and self.llm is None:
            raise ValueError(
                "GeneralResponseBranch requires either a configured content composer "
                "or an answer LLM; general-purpose responses may not bypass answer "
                "generation."
            )

    def execute(self, context: PipelineContext, repository: AssistantRepository) -> BranchResult:
        general_warnings: list[str] = []
        personal_context_unavailable = False
        unavailable_context_domains: set[str] = set()
        try:
            knowledge_results = retrieve_knowledge(
                retriever=self.retriever,
                repository=repository,
                user_id=context.request.user_id,
                query=context.rewritten_query,
            )
        except Exception as exc:
            knowledge_results = []
            personal_context_unavailable = True
            unavailable_context_domains.add("knowledge")
            general_warnings.append(
                f"general_knowledge_retrieval_unavailable:{type(exc).__name__}"
            )
        try:
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
        except Exception as exc:
            reminder_raw = []
            personal_context_unavailable = True
            unavailable_context_domains.add("reminder")
            general_warnings.append(
                f"general_reminder_retrieval_unavailable:{type(exc).__name__}"
            )

        try:
            approved_context = self.context_filter.filter(
                user_id=context.request.user_id,
                knowledge_results=knowledge_results,
                reminder_results=reminder_raw,
                conversation_results=[],
                query=context.rewritten_query,
                intent=context.intent,
            )
        except Exception as exc:
            personal_context_unavailable = True
            unavailable_context_domains.add("context_filter")
            approved_context = ApprovedContext(
                knowledge_evidence=[],
                reminder_context=[],
                approved_conversation_history=[],
                rejected_knowledge_ids=[],
                rejected_reminder_ids=[],
                rejected_conversation_ids=[],
            )
            general_warnings.append(
                f"general_context_filter_unavailable:{type(exc).__name__}"
            )
        approved_context = replace(
            approved_context,
            approved_conversation_history=list(context.chat_history),
        )
        try:
            merged_supporting_detail = self._merge_supporting_detail(approved_context)
        except Exception as exc:
            merged_supporting_detail = (
                '{"knowledge_evidence":[],"reminder_context":[]}'
            )
            general_warnings.append(
                f"general_supporting_context_unavailable:{type(exc).__name__}"
            )

        from .general_sub_branch import GeneralSubBranchValidator, GeneralPersistencePlanBuilder

        decision = None
        plan = None
        if self.general_purpose_config is not None:
            try:
                if self.sub_branch_detector is not None:
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
                decision = GeneralSubBranchValidator().validate(
                    decision,
                    context,
                    self.general_purpose_config,
                )
                plan = GeneralPersistencePlanBuilder().build_plan(
                    decision,
                    context,
                    self.general_purpose_config,
                )
            except Exception as exc:
                decision = None
                plan = None
                general_warnings.append(
                    f"general_sub_branch_unavailable:{type(exc).__name__}"
                )
            answer_mode = (
                self._general_sub_branch_to_answer_mode(decision.sub_branch)
                if decision is not None
                else self._detect_answer_mode(context, approved_context)
            )
        else:
            answer_mode = self._detect_answer_mode(context, approved_context)
            
        resolved_sub_branch = decision.sub_branch if decision else GeneralSubBranch.NEW_CONVERSATION_TOPIC
        if not decision:
            if answer_mode == AnswerMode.SUPPORT_QUESTION_ANSWER:
                resolved_sub_branch = GeneralSubBranch.SUPPORT_QUESTION_ANSWER
            elif answer_mode == AnswerMode.FOLLOW_UP_CONVERSATION:
                resolved_sub_branch = GeneralSubBranch.CONVERSATION_FOLLOW_UP
            
        resolved_persistence_mode = plan.persistence_mode if plan else PersistenceMode.CREATE_NEW_TOPIC
        explicit_conversation_resume = bool(
            context.request.conversation_id
            and context.request.parent_hop_id
            and context.last_qa_state is not None
            and context.last_qa_state.linked_topic_id
        )
        if (
            explicit_conversation_resume
            and resolved_sub_branch is not GeneralSubBranch.SUPPORT_QUESTION_ANSWER
        ):
            # An ownership-validated explicit hop cursor outranks a heuristic
            # new-topic classification. It is the deterministic process variant
            # for reopening or continuing a saved conversation.
            resolved_sub_branch = GeneralSubBranch.CONVERSATION_FOLLOW_UP
            resolved_persistence_mode = PersistenceMode.APPEND_TO_EXISTING_TOPIC
        
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

        composer_result = None
        degraded_authorization = (
            self._authorize_degraded_personal_context(
                context=context,
                unavailable_domains=unavailable_context_domains,
            )
            if personal_context_unavailable
            else _DegradedContextAuthorization()
        )
        degraded_certified_answer = (
            degraded_authorization.certified_answer_text.strip()
        )
        blocked_by_context_dependency = bool(
            personal_context_unavailable
            and not degraded_authorization.authorized
        )
        answer_generation_attempted = False
        answer_generation_succeeded = False
        writer_generation_attempted = False
        writer_generation_succeeded = False
        answer_delivery_text = ""
        answer_recovery_scope = context.rewritten_query
        if blocked_by_context_dependency:
            # The mandatory answer stage still executes, but its ungrounded
            # prose cannot authorize a guess about unavailable durable state.
            answer_generation_attempted = (
                degraded_authorization.answer_stage_attempted
            )
            if not answer_generation_attempted:
                answer_generation_attempted = True
                self._generate_response(
                    context,
                    approved_context,
                    prompt_ctx,
                    sub_branch_supporting_prompt,
                    merged_supporting_detail,
                )
            response = self.prompt_registry.message(
                "personal_context_temporarily_unavailable"
            )
            answer_generation_succeeded = False
            general_warnings.append("personal_context_safe_retry")
        elif (
            self.content_composer is not None
            and self.general_purpose_config is not None
        ):
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
                approved_knowledge_records=approved_context.knowledge_records,
            )
            try:
                composer_result = self.content_composer.compose(
                    composer_input, self.general_purpose_config
                )
            except Exception as exc:
                composer_result = None
                general_warnings.append(
                    f"content_composer_unavailable:{type(exc).__name__}"
                )
            if composer_result is not None:
                answer_generation_attempted = (
                    "answer_generation" in composer_result.used_tool_names
                )
                trace_payload: dict[str, Any] = {}
                try:
                    decoded_trace = json.loads(composer_result.tool_trace_summary)
                    if isinstance(decoded_trace, dict):
                        trace_payload = decoded_trace
                except (TypeError, ValueError):
                    trace_payload = {}
                outcomes = trace_payload.get("stage_outcomes")
                outcomes = outcomes if isinstance(outcomes, dict) else {}
                traced_answer_scope = str(
                    trace_payload.get("answer_request_scope") or ""
                ).strip()
                if traced_answer_scope:
                    answer_recovery_scope = traced_answer_scope
                answer_outcome = outcomes.get("answer_generation")
                if isinstance(answer_outcome, dict):
                    answer_generation_succeeded = bool(
                        answer_outcome.get("succeeded")
                    )
                else:
                    # Compatibility for injected composers predating detailed
                    # stage outcomes. They must still identify the mandatory
                    # answer stage and report that no fallback was used.
                    answer_generation_succeeded = bool(
                        answer_generation_attempted
                        and not composer_result.fallback_used
                        and str(composer_result.final_response_text or "").strip()
                    )
                if (
                    answer_generation_succeeded
                    and approved_context.knowledge_records
                ):
                    # A composer cannot self-certify grounding. Recheck its
                    # separately-delimited answer at the branch boundary.
                    answer_generation_succeeded = is_grounded_answer_text(
                        str(composer_result.answer_response_text or ""),
                        approved_context.knowledge_records,
                    )
                if answer_generation_succeeded:
                    answer_delivery_text = str(
                        composer_result.answer_response_text or ""
                    ).strip()
                writer_outcomes = [
                    outcome
                    for name, outcome in outcomes.items()
                    if name != "answer_generation" and isinstance(outcome, dict)
                ]
                writer_generation_attempted = bool(writer_outcomes) or any(
                    name != "answer_generation"
                    for name in composer_result.used_tool_names
                )
                writer_generation_succeeded = any(
                    bool(outcome.get("succeeded")) for outcome in writer_outcomes
                )
                if not writer_outcomes and writer_generation_attempted:
                    writer_generation_succeeded = bool(
                        composer_result.artifacts and not composer_result.fallback_used
                    )
                response = str(composer_result.final_response_text or "").strip()
                if (
                    approved_context.knowledge_records
                    and answer_generation_succeeded
                ):
                    # Reconstruct the user-visible prose from the separately
                    # validated answer boundary. Arbitrary combined composer
                    # prose cannot smuggle unsupported personal claims.
                    response = answer_delivery_text
                    if composer_result.artifacts:
                        response = "\n\n".join(
                            [
                                response,
                                self.prompt_registry.message(
                                    "generated_artifacts_available"
                                ),
                            ]
                        )
                if (
                    approved_context.knowledge_records
                    and not answer_generation_succeeded
                ):
                    response = ""
                elif not answer_generation_succeeded:
                    # Once the mandatory answer stage is known to have failed,
                    # none of the composer's combined prose has an attributable
                    # trust boundary.  Discard it before direct recovery.  A
                    # successful artifact remains available through the typed
                    # artifact payload and a code-owned status message below.
                    response = ""
            else:
                response = ""

            if degraded_certified_answer:
                # The capable ANSWER task already produced the only prose that
                # is certified not to depend on the unavailable durable store.
                # Composer output is retained only for typed artifacts/actions.
                answer_generation_attempted = True
                answer_generation_succeeded = True
                answer_delivery_text = degraded_certified_answer
                response = degraded_certified_answer
                if composer_result is not None and composer_result.artifacts:
                    response = "\n\n".join(
                        (
                            response,
                            self.prompt_registry.message(
                                "generated_artifacts_available"
                            ),
                        )
                    )

            # The branch, rather than an arbitrary injected composer, owns the
            # unconditional answer-stage invariant. Missing, failed, or blank
            # answer output gets one direct ANSWER-model recovery attempt.
            if not answer_generation_succeeded or not response:
                answer_generation_attempted = True
                recovery = self._generate_response(
                    context,
                    approved_context,
                    prompt_ctx,
                    sub_branch_supporting_prompt,
                    merged_supporting_detail,
                    rewritten_query=answer_recovery_scope,
                )
                recovered_answer = recovery.text.strip()
                unavailable = self.prompt_registry.message(
                    "answer_model_unavailable"
                ).strip()
                recovery_succeeded = recovery.model_succeeded
                answer_generation_succeeded = recovery_succeeded
                if recovery_succeeded:
                    answer_delivery_text = recovered_answer
                    response = recovered_answer
                    if composer_result is not None and composer_result.artifacts:
                        response = "\n\n".join(
                            (
                                response,
                                self.prompt_registry.message(
                                    "generated_artifacts_available"
                                ),
                            )
                        )
                    general_warnings.append("answer_generation_recovered")
                elif recovered_answer and approved_context.knowledge_evidence:
                    # SQL-approved evidence is a safer terminal response than a
                    # generic model-unavailable string emitted by the composer.
                    response = recovered_answer
                    general_warnings.append("answer_grounded_evidence_fallback")
                elif not response:
                    response = recovered_answer or unavailable
                    general_warnings.append("answer_model_unavailable")
        else:
            answer_generation_attempted = True
            if degraded_certified_answer:
                response = degraded_certified_answer
                answer_generation_succeeded = True
                answer_delivery_text = degraded_certified_answer
            else:
                generation = self._generate_response(
                    context,
                    approved_context,
                    prompt_ctx,
                    sub_branch_supporting_prompt,
                    merged_supporting_detail,
                )
                response = generation.text
                answer_generation_succeeded = generation.model_succeeded
                if answer_generation_succeeded:
                    answer_delivery_text = generation.text.strip()
        response_warnings = list(
            dict.fromkeys(
                [
                    *general_warnings,
                    *(
                        composer_result.content_warnings
                        if composer_result is not None
                        else ()
                    ),
                ]
            )
        )

        completed_explicit_request_reason = ""
        semantic_action = semantic_action_from_internal_payload(
            (
                composer_result.semantic_action_decision
                if composer_result is not None
                else None
            ),
            canonical_query=context.rewritten_query,
        )
        if (
            semantic_action.message.requests_message
            and answer_generation_succeeded
        ):
            completed_explicit_request_reason = "explicit_delivery_request_is_actionable"
        elif composer_result and composer_result.artifacts:
            completed_explicit_request_reason = "requested_artifact_was_created"

        if self.general_purpose_config is not None:
            if blocked_by_context_dependency:
                hitl_questions = []
                hitl_result = {
                    "triggered": False,
                    "confidence": 1.0,
                    "question_count": 0,
                    "reason": "personal_context_safe_retry",
                }
            elif completed_explicit_request_reason:
                hitl_questions = []
                hitl_result = {
                    "triggered": False,
                    "confidence": 1.0,
                    "question_count": 0,
                    "reason": completed_explicit_request_reason,
                }
            elif self.general_hitl_strategy:
                hitl_decision = self.general_hitl_strategy.evaluate(
                    context=context,
                    response_text=response,
                    approved_conversation_history=approved_context.approved_conversation_history,
                    merged_supporting_detail=merged_supporting_detail,
                )
                hitl_questions = [GeneratedQuestion(
                    text=hitl_decision.question,
                    source=hitl_decision.question_source,
                    purpose="resolve_required_context",
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

        composition_payload = {
            "answer_attempted": answer_generation_attempted,
            "answer_succeeded": answer_generation_succeeded,
            "writer_attempted": writer_generation_attempted,
            "writer_succeeded": writer_generation_succeeded,
        }
        
        try:
            with repository.transaction() as cursor:
                explicit_parent_topic_id = (
                    context.last_qa_state.linked_topic_id
                    if context.request.conversation_id
                    and context.request.parent_hop_id
                    and context.last_qa_state is not None
                    else None
                )
                if explicit_parent_topic_id:
                    topic_id = explicit_parent_topic_id
                elif plan and plan.persistence_mode == PersistenceMode.APPEND_TO_EXISTING_TOPIC and plan.topic_id:
                    topic_id = plan.topic_id
                elif plan and plan.persistence_mode == PersistenceMode.BRANCH_FROM_EXISTING_HOP and plan.topic_id:
                    topic_id = plan.topic_id
                else:
                    fresh_topic_creator = getattr(repository, "create_topic", None)
                    if callable(fresh_topic_creator):
                        topic_id = fresh_topic_creator(
                            cursor,
                            user_id=context.request.user_id,
                            title=topic_title,
                        )
                    else:
                        topic_id = repository.ensure_topic(
                            cursor, user_id=context.request.user_id, title=topic_title
                        )
                
                parent_hop_id = (
                    context.request.parent_hop_id
                    if explicit_parent_topic_id
                    else (
                        plan.parent_hop_id
                        if plan
                        else context.request.parent_hop_id
                    )
                )

                entities = {
                    "process_variant": resolved_sub_branch.value,
                    "persistence_mode": resolved_persistence_mode.value,
                }
                if context.request.conversation_id:
                    entities["conversation_id"] = context.request.conversation_id
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
                warnings=response_warnings,
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
            warnings=response_warnings,
            database_write_result={"conversation_hop_id": hop.hop_id},
            indexing_job_result={
                "conversation_hop_job_id": hop.outbox_job_id,
                "outbox_job_ids": [hop.outbox_job_id],
            },
            platform_payload={
                **(
                    {"artifacts": list(composer_result.artifacts)}
                    if composer_result and composer_result.artifacts
                    else {}
                ),
                **(
                    {
                        "semantic_action_decision": (
                            composer_result.semantic_action_decision
                        )
                    }
                    if composer_result
                    and composer_result.semantic_action_decision
                    else {}
                ),
                **(
                    {"answer_generation_text": answer_delivery_text}
                    if answer_delivery_text
                    else {}
                ),
                "content_composition": composition_payload,
            },
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
                "knowledge_records": approved_context.knowledge_records,
                "reminder_context": approved_context.reminder_context,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )

    def _detect_answer_mode(self, context: PipelineContext, approved_context: Any) -> AnswerMode:
        if (
            (context.last_qa_trace or {}).get("interaction_type")
            == "supporting_question_answer"
        ):
            return AnswerMode.SUPPORT_QUESTION_ANSWER
        if not approved_context.approved_conversation_history and not context.last_qa_state:
            return AnswerMode.NEW_CONVERSATION
        return AnswerMode.FOLLOW_UP_CONVERSATION

    def _authorize_degraded_personal_context(
        self,
        *,
        context: PipelineContext,
        unavailable_domains: set[str],
    ) -> _DegradedContextAuthorization:
        """Require a dependency denial plus an independent answer certificate.

        RETRIEVAL_VALIDATION can only provisionally deny dependency.  The
        independently routed ANSWER task must then certify and supply the exact
        context-free prose that may cross the response boundary.  Any missing,
        malformed, low-confidence, or contradictory certificate fails closed.
        No vocabulary or phrase table is used.
        """

        if self.llm is None:
            return _DegradedContextAuthorization()

        def validate(
            payload: dict[str, Any],
            *,
            boolean_field: str,
            answer_envelope: bool = False,
        ) -> None:
            if not isinstance(payload.get(boolean_field), bool):
                raise StructuredOutputInvariantError(
                    f"{boolean_field} must be boolean"
                )
            try:
                confidence = float(payload.get("confidence"))
            except (TypeError, ValueError) as exc:
                raise StructuredOutputInvariantError(
                    "confidence must be numeric"
                ) from exc
            if not 0.0 <= confidence <= 1.0:
                raise StructuredOutputInvariantError(
                    "confidence must be between zero and one"
                )
            if not str(payload.get("reason_summary") or "").strip():
                raise StructuredOutputInvariantError(
                    "reason_summary must be non-empty"
                )
            if answer_envelope:
                answer_text = payload.get("answer_text")
                if not isinstance(answer_text, str):
                    raise StructuredOutputInvariantError(
                        "answer_text must be a string"
                    )
                if bool(payload[boolean_field]) != bool(answer_text.strip()):
                    raise StructuredOutputInvariantError(
                        "answer_text must be non-empty exactly when the request "
                        "is certified context-independent"
                    )

        def decide(
            *,
            stage: str,
            boolean_field: str,
            schema: dict[str, Any],
            decision_contract: str,
            task: LLMTask = LLMTask.RETRIEVAL_VALIDATION,
            answer_envelope: bool = False,
        ) -> dict[str, Any] | None:
            try:
                payload = self.llm.generate_json(
                    task=task,
                    system_prompt=self.prompt_registry.system(stage),
                    user_prompt=self.prompt_registry.user(
                        PromptContext(
                            stage=stage,
                            user_id=context.request.user_id,
                            rewritten_query=context.rewritten_query,
                            intent=context.intent.value,
                            metadata=context.request.metadata,
                            platform_context=context.request.platform_context,
                            extra={
                                "unavailable_context_domains": sorted(
                                    unavailable_domains
                                ),
                                "decision_contract": decision_contract,
                            },
                            # These gates classify only the current request.
                            # Earlier conversation is neither available context
                            # nor evidence that the current request is safe.
                            chat_history=[],
                        )
                    ),
                    schema=schema,
                    invariant_validator=lambda candidate: validate(
                        candidate,
                        boolean_field=boolean_field,
                        answer_envelope=answer_envelope,
                    ),
                )
                if is_structured_fallback(payload):
                    return None
                validate(
                    payload,
                    boolean_field=boolean_field,
                    answer_envelope=answer_envelope,
                )
                return payload
            except Exception:
                return None

        dependency = decide(
            stage="personal_context_dependency",
            boolean_field="requires_unavailable_context",
            schema=_PERSONAL_CONTEXT_DEPENDENCY_SCHEMA,
            decision_contract=(
                "Determine whether completing the current request requires "
                "information from any unavailable durable context domain. "
                "Evaluate semantic dependency, not surface wording."
            ),
        )
        if (
            dependency is None
            or bool(dependency["requires_unavailable_context"])
            or float(dependency["confidence"])
            < _PERSONAL_CONTEXT_CERTIFICATE_MIN_CONFIDENCE
        ):
            return _DegradedContextAuthorization()

        certificate = decide(
            stage="personal_context_degraded_answer",
            boolean_field="answerable_without_unavailable_context",
            schema=_PERSONAL_CONTEXT_DEGRADED_ANSWER_SCHEMA,
            decision_contract=(
                "Return the exact proposed user-facing answer only when the "
                "complete current request can be answered using its supplied "
                "content and general knowledge, with no fact, state, history, "
                "target, or record from an unavailable durable context domain."
            ),
            task=LLMTask.ANSWER,
            answer_envelope=True,
        )
        if (
            certificate is None
            or not bool(certificate["answerable_without_unavailable_context"])
            or float(certificate["confidence"])
            < _PERSONAL_CONTEXT_CERTIFICATE_MIN_CONFIDENCE
        ):
            return _DegradedContextAuthorization(answer_stage_attempted=True)
        return _DegradedContextAuthorization(
            certified_answer_text=str(certificate["answer_text"]).strip(),
            answer_stage_attempted=True,
        )

    def _generate_response(
        self,
        context: PipelineContext,
        approved_context: Any,
        prompt_ctx: SubBranchPromptContext,
        sub_branch_supporting_prompt: str,
        merged_supporting_detail: str,
        *,
        rewritten_query: str | None = None,
    ) -> AnswerGenerationOutcome:
        answer_query = str(rewritten_query or context.rewritten_query).strip()
        return generate_answer(
            llm=self.llm,
            prompt_registry=self.prompt_registry,
            prompt_context=PromptContext(
                stage="answer_generation",
                user_id=context.request.user_id,
                rewritten_query=answer_query,
                intent=context.intent.value,
                metadata=context.request.metadata,
                platform_context=context.request.platform_context,
                extra={
                    "approved_conversation_history": approved_context.approved_conversation_history,
                    "approved_knowledge_evidence": approved_context.knowledge_evidence,
                    "approved_knowledge_records": approved_context.knowledge_records,
                    "approved_reminder_context": approved_context.reminder_context,
                    "merged_supporting_detail": merged_supporting_detail,
                    "sub_branch_supporting_prompt": sub_branch_supporting_prompt,
                    "human_supporting_questions": [q.text for q in context.approved_conversation_context.human_supporting_questions] if context.approved_conversation_context else [],
                    "reminder_supporting_questions": [q.text for q in context.approved_conversation_context.reminder_supporting_questions] if context.approved_conversation_context else [],
                    "extracted_expected_response_types": [t.value for t in context.approved_conversation_context.extracted_expected_response_types] if context.approved_conversation_context else [],
                },
            ),
            approved_knowledge_records=approved_context.knowledge_records,
            approved_knowledge_evidence=approved_context.knowledge_evidence,
        )


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
        actions = list(detection.metadata.get("knowledge_actions") or [])
        # A schema-valid best-effort extraction must reach model 2, which owns
        # semantic completeness and confidence. Only a failed response guard
        # with no usable action state clarifies before retrieval validation.
        if detection.requires_clarification and not actions:
            if detection.failure_kind == "semantic_gap":
                return self._generate_clarification(
                    context,
                    getattr(detection, "missing_fields", []),
                    "The extracted knowledge content was not grounded in the current request.",
                )
            return BranchResult(
                response_type=ResponseType.ERROR,
                fallback_or_error_message=self.prompt_registry.message(
                    "knowledge_pipeline_unavailable"
                ),
            )
        context.request.metadata.update(detection.metadata)
                    
        if not actions:
            return BranchResult(
                response_type=ResponseType.ERROR,
                fallback_or_error_message=self.prompt_registry.message(
                    "knowledge_pipeline_unavailable"
                ),
            )
        if len(actions) != 1:
            # Guard 1 hands exactly one canonical model-1 action state to the
            # validation pipeline. Never truncate a malformed multi-action
            # detector result by silently taking actions[0].
            return self._generate_clarification(
                context,
                ["single_action"],
                "Exactly one knowledge action is required.",
            )

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
                    conversation_id=context.request.conversation_id,
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
            conversation_id=context.request.conversation_id,
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
    prompt_registry: PromptRegistry = field(default_factory=lambda: DEFAULT_PROMPT_REGISTRY)
    retriever: HybridRetriever | None = None
    context_filter: ContextFilter | None = None
    llm: LLMClient | None = None
    reminder_mutation_pipeline: Any | None = None

    def _safe_rejection(self) -> BranchResult:
        """Stop a reminder mutation without generating a clarification question."""
        return BranchResult(
            response_type=ResponseType.SAFE_NOOP,
            normal_response_text=self.prompt_registry.message(
                "reminder_not_confident"
            ),
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
        actions = list(detection.metadata.get("reminder_actions") or [])
        context.request.metadata.update(detection.metadata)
                    
        if not actions:
            return self._safe_rejection()
        if len(actions) != 1:
            # Keep the same guard-1 cardinality boundary as knowledge while
            # preserving the reminder branch's non-generative safe rejection.
            return self._safe_rejection()

        executable_actions = []
        pre_repo_results = []
        if self.reminder_mutation_pipeline:
            validated_actions = [
                self.reminder_mutation_pipeline.build_action(
                    context=context,
                    action_payload=actions[0],
                    repository=repository,
                )
            ]
            for v_act in validated_actions:
                if v_act.hitl_reason == "internal_pipeline_failure":
                    return BranchResult(
                        response_type=ResponseType.ERROR,
                        fallback_or_error_message=self.prompt_registry.message(
                            "reminder_pipeline_unavailable"
                        ),
                    )
                if v_act.hitl_reason == "reminder_validation_fail":
                    if not (v_act.clarification_question or "").strip():
                        return BranchResult(
                            response_type=ResponseType.ERROR,
                            fallback_or_error_message=self.prompt_registry.message(
                                "reminder_pipeline_unavailable"
                            ),
                        )
                    return BranchResult(
                        response_type=ResponseType.CLARIFICATION,
                        clarification_question=GeneratedQuestion(
                            text=v_act.clarification_question,
                            source=QuestionSource.CLARIFICATION_QUESTION,
                            purpose="reminder_validation_fail",
                            confidence=v_act.confidence,
                        ),
                    )
                    
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
                    
            if not executable_actions:
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
                    conversation_id=context.request.conversation_id,
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
                    "reminder_pipeline_unavailable"
                ),
            )

        success_text = context.request.metadata.get(
            "operation_response", self.prompt_registry.message("reminder_updated") if hasattr(self.prompt_registry, "message") else "Reminder updated successfully."
        )
        if len(executable_actions) != 1:
            return self._safe_rejection()

        result = repository.transactional_reminder_actions(
            user_id=context.request.user_id,
            topic_title=context.request.metadata.get("topic_title", "Reminders"),
            raw_user_query=context.request.raw_query,
            rewritten_user_query=context.rewritten_query,
            response_text=success_text,
            actions=executable_actions,
            parent_hop_id=context.request.parent_hop_id,
            conversation_id=context.request.conversation_id,
        )
        
        if result.committed:
            return BranchResult(
                response_type=ResponseType.REMINDER_ACTION,
                normal_response_text=success_text,
                human_supporting_questions=[],
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
                warnings=[],
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
        reminder_reply_state: dict[str, Any] | None = None
        if (
            context.last_qa_trace.get("interaction_type")
            == "reminder_notification_reply"
            and context.last_qa_state is not None
        ):
            reminder_reply_state = verified_reminder_state(
                context.last_qa_state.reminder_state,
                context.last_qa_state.reminder_state_hash,
            )
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

        result = _suppress_repeated_clarification(
            result,
            context,
            getattr(branch, "prompt_registry", DEFAULT_PROMPT_REGISTRY),
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
                    if reminder_reply_state is not None:
                        repository.merge_conversation_hop_entities(
                            cursor,
                            user_id=context.request.user_id,
                            hop_id=result.linked_hop_id,
                            entities=reminder_reply_hop_entities(
                                reminder_reply_state,
                                reply_hop_id=result.linked_hop_id,
                            ),
                        )
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
                    "should_ask": result.clarification_question.should_ask,
                    "expected_response_type": (
                        result.clarification_question.expected_response_type.value
                    ),
                }
            )
        supporting_questions.extend(
            {
                "question_text": question.text,
                "question_source": question.source.value,
                "purpose": question.purpose,
                "confidence": question.confidence,
                "should_ask": question.should_ask,
                "expected_response_type": question.expected_response_type.value,
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
                    "should_ask": result.reminder_supporting_question.should_ask,
                    "expected_response_type": (
                        result.reminder_supporting_question.expected_response_type.value
                    ),
                }
            )

        default_titles = {
            Intent.CLARIFICATION: "General Conversation",
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
            conversation_id=context.request.conversation_id,
            entities={
                "branch_outcome": {
                    "intent": context.intent.value,
                    "response_type": result.response_type.value,
                },
                **(
                    reminder_reply_hop_entities(reminder_reply_state)
                    if reminder_reply_state is not None
                    else {}
                ),
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
