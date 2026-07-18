"""Query rewrite, Last-QA resolution, and intent classification."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
from typing import Any

from .config import ClassificationConfig, LastQAConfig
from .contracts import (
    ChatRequest, GeneratedQuestion, Intent, LastQAState, ResponseType, LastQAPath,
    LastQAInteractionType, OutboundFollowUpAction, QuestionSource, LastQAResolution,
    validate_last_qa_resolution
)
from .llm import (
    LLMClient,
    LLMTask,
    _has_pending_clarification,
    _is_authoritative_outbound_action,
    build_intent_conversation_extra,
    intent_classification_schema,
    is_structured_fallback,
    validate_json_schema,
)
from .prompts import DEFAULT_PROMPT_REGISTRY, PromptContext, PromptRegistry
from .reminder_reply import verified_reminder_state
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

logger = logging.getLogger(__name__)


def _log_llm_fallback(stage: str, exc: Exception) -> None:
    logger.debug("%s LLM fallback engaged: %s", stage, exc)


def _unit_confidence(value: Any) -> float | None:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        return None
    return confidence


def _confidence_meets(value: Any, threshold: float) -> bool:
    confidence = _unit_confidence(value)
    return confidence is not None and confidence >= threshold



class QueryRewriter:
    def rewrite(self, query: str) -> str:
        return " ".join(query.split())


@dataclass
class LLMQueryRewriter:
    llm: LLMClient
    prompt_registry: PromptRegistry = field(default_factory=lambda: DEFAULT_PROMPT_REGISTRY)
    fallback: QueryRewriter = field(default_factory=QueryRewriter)

    def rewrite(self, query: str) -> str:
        normalized = self.fallback.rewrite(query)
        if not normalized:
            return normalized
        schema = {
            "type": "object",
            "properties": {
                "rewritten_query": {"type": "string"},
            },
            "required": ["rewritten_query"],
        }
        try:
            payload = self.llm.generate_json(
                task=LLMTask.QUERY_REWRITE,
                system_prompt=self.prompt_registry.system("query_rewrite"),
                user_prompt=self.prompt_registry.user(
                    PromptContext(stage="query_rewrite", raw_query=query)
                ),
                schema=schema,
            )
            if is_structured_fallback(payload):
                return normalized
            validate_json_schema(payload, schema)
        except Exception as e:
            _log_llm_fallback("Query rewrite", e)
            return normalized
        rewritten = str(payload.get("rewritten_query") or normalized).strip()
        return rewritten or normalized


class T5CanardQueryRewriter:
    def __init__(self, model_name: str = "castorini/t5-base-canard", fallback: QueryRewriter | None = None):
        self.fallback = fallback or QueryRewriter()
        logger.info("Loading HuggingFace model for query rewriting: %s", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def rewrite(self, query: str) -> str:
        normalized = self.fallback.rewrite(query)
        if not normalized:
            return normalized
        try:
            input_ids = self.tokenizer(query, return_tensors="pt").input_ids
            outputs = self.model.generate(input_ids, max_length=128)
            rewritten = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return rewritten.strip() or normalized
        except Exception as e:
            _log_llm_fallback("Query rewrite (T5)", e)
            return normalized


# LastQAResolution moved to contracts.py


def _resolve_exact_reminder_notification_reply(
    request: ChatRequest,
    rewritten_query: str,
    state: LastQAState | None,
) -> LastQAResolution | None:
    """Resolve a source-matched notification reply inside the mandatory resolver."""
    metadata = request.metadata or {}
    if not metadata.get("reminder_reply_context") or state is None:
        return None

    metadata_state = verified_reminder_state(
        metadata.get("reminder_state"),
        metadata.get("reminder_state_hash"),
    )
    last_qa_reminder_state = verified_reminder_state(
        state.reminder_state,
        state.reminder_state_hash,
    )
    if (
        metadata_state is None
        or last_qa_reminder_state is None
        or metadata_state != last_qa_reminder_state
    ):
        return None

    source_topic_id = metadata.get("source_topic_id")
    source_hop_id = metadata.get("source_hop_id")
    reminder_id = str(metadata.get("reminder_id") or "")
    notification_id = str(metadata.get("notification_id") or "")
    if (
        not source_topic_id
        or not source_hop_id
        or not reminder_id
        or not notification_id
        or source_topic_id != state.linked_topic_id
        or source_hop_id != state.linked_hop_id
        or source_topic_id != metadata_state.get("source_topic_id")
        or source_hop_id != metadata_state.get("source_hop_id")
        or reminder_id != metadata_state.get("reminder_id")
        or notification_id != metadata_state.get("notification_id")
        or (request.reminder_id is not None and request.reminder_id != reminder_id)
        or (
            request.notification_id is not None
            and request.notification_id != notification_id
        )
    ):
        return None

    resolution = LastQAResolution(
        path=LastQAPath.LATEST_CONTEXT_INTERACTION,
        rewritten_query=rewritten_query,
        state=state,
        did_merge_query=False,
        skip_broad_retrieval=True,
        confidence=1.0,
        interaction_type=LastQAInteractionType.REMINDER_NOTIFICATION_REPLY,
        question_source=QuestionSource.NONE,
        linked_topic_id=state.linked_topic_id,
        linked_hop_id=state.linked_hop_id,
        reminder_id=reminder_id,
        notification_id=notification_id,
        source_topic_id=source_topic_id,
        source_hop_id=source_hop_id,
        merge_reason="exact_reminder_notification_context",
        skip_reason="reminder_reply_context_hash_matched_source_hop",
        is_authoritative_state=True,
    )
    validate_last_qa_resolution(resolution)
    return resolution


class SupportingQuestionMatcher:
    def __init__(self, match_threshold: float = 0.7) -> None:
        self.match_threshold = match_threshold

    def matches(self, query: str, supporting_questions: list[Any]) -> bool:
        return self.confidence(query, supporting_questions) >= self.match_threshold

    def confidence(self, query: str, supporting_questions: list[Any]) -> float:
        if not query or not supporting_questions:
            return 0.0
        
        query_terms = set(query.casefold().split())
        if not query_terms:
            return 0.0

        best_score = 0.0
        for question in supporting_questions:
            question_terms = set(question.text.casefold().split())
            if not question_terms:
                continue
            intersection = query_terms.intersection(question_terms)
            overlap_ratio = len(intersection) / len(query_terms)
            best_score = max(best_score, overlap_ratio)
        return best_score


def can_skip_broad_retrieval(
    payload: dict[str, Any], state: LastQAState, interaction_type: LastQAInteractionType, config: LastQAConfig, request: ChatRequest
) -> bool:
    if not _confidence_meets(
        payload.get("confidence", 0.0),
        config.skip_broad_retrieval_min_confidence,
    ):
        return False
        
        
    if interaction_type.value not in config.skip_allowed_interaction_types:
        return False

    question_source = str(payload.get("question_source") or QuestionSource.NONE.value)
    matched_question = " ".join(str(payload.get("matched_question") or "").casefold().split())

    if interaction_type == LastQAInteractionType.NORMAL_FOLLOW_UP:
        return bool(
            state.linked_topic_id
            and state.linked_hop_id
            and state.last_user_query.strip()
            and state.last_response.strip()
            and question_source == QuestionSource.NONE.value
            and not matched_question
            and str(payload.get("relationship") or "")
            == LastQAInteractionType.NORMAL_FOLLOW_UP.value
        )

    if interaction_type == LastQAInteractionType.SUPPORTING_QUESTION_ANSWER:
        human_questions = {
            " ".join(question.text.casefold().split())
            for question in state.supporting_questions
            if question.text
        }
        reminder_question = state.reminder_supporting_question
        reminder_questions = {
            " ".join(reminder_question.text.casefold().split())
        } if reminder_question and reminder_question.text else set()
        source_matches_question = (
            question_source == QuestionSource.HUMAN_SUPPORTING_QUESTION.value
            and matched_question in human_questions
        ) or (
            question_source == QuestionSource.REMINDER_SUPPORTING_QUESTION.value
            and matched_question in reminder_questions
        )
        return bool(
            state.linked_topic_id
            and state.linked_hop_id
            and state.last_response
            and source_matches_question
        )

    if interaction_type == LastQAInteractionType.REMINDER_NOTIFICATION_REPLY:
        if not config.enable_reminder_metadata_reply:
            return False
        md = request.metadata or {}
        pc = request.platform_context or {}
        return bool(
            question_source == QuestionSource.NONE.value and
            (md.get("reminder_id") or pc.get("reminder_id")) and
            (md.get("notification_id") or pc.get("notification_id")) and
            (md.get("source_topic_id") or pc.get("source_topic_id")) and
            (md.get("source_hop_id") or pc.get("source_hop_id"))
        )
        
    return False


class LastQAResolver:
    def __init__(self, matcher: SupportingQuestionMatcher | None = None) -> None:
        self.matcher = matcher or SupportingQuestionMatcher()

    def resolve(self, request: ChatRequest, rewritten_query: str, state: LastQAState | None) -> LastQAResolution:
        reminder_resolution = _resolve_exact_reminder_notification_reply(
            request, rewritten_query, state
        )
        if reminder_resolution is not None:
            return reminder_resolution

        if state is None:
            resolution = LastQAResolution(
                rewritten_query=rewritten_query, 
                state=None, 
                did_merge_query=False,
                skip_broad_retrieval=False,
                confidence=1.0,
                path=LastQAPath.NO_LAST_QA,
                merge_reason=None,
                skip_reason="no_last_qa_exists",
                is_authoritative_state=False
            )
            validate_last_qa_resolution(resolution)
            return resolution
            
        if state.response_type is ResponseType.CLARIFICATION or state.clarification_question:
            resolution = LastQAResolution(
                rewritten_query=rewritten_query,
                state=None,
                did_merge_query=False,
                skip_broad_retrieval=False,
                confidence=0.0,
                path=LastQAPath.BROAD_RETRIEVAL_REQUIRED,
                merge_reason="clarification_not_answered_or_low_confidence",
                skip_reason="broad_retrieval_required",
                is_authoritative_state=False,
                missing_context=["semantic clarification merge unavailable"],
                diagnostic_context={"previous_last_qa_state": "safe_summary_only"}
            )
            validate_last_qa_resolution(resolution)
            return resolution

        if state.response_type in (
            ResponseType.NORMAL,
            ResponseType.KNOWLEDGE_ACTION,
            ResponseType.REMINDER_ACTION,
            ResponseType.REMINDER_REPLY,
            ResponseType.ERROR,
            ResponseType.SAFE_NOOP,
        ):
            support_confidence = self.matcher.confidence(
                rewritten_query, state.supporting_questions
            )
            if (
                state.supporting_questions
                and support_confidence >= self.matcher.match_threshold
            ):
                skip = bool(state.linked_topic_id and state.linked_hop_id)
                resolution = LastQAResolution(
                    rewritten_query=rewritten_query, 
                    state=state, 
                    did_merge_query=False,
                    skip_broad_retrieval=skip, 
                    confidence=support_confidence,
                    path=LastQAPath.LATEST_CONTEXT_INTERACTION if skip else LastQAPath.BROAD_RETRIEVAL_REQUIRED,
                    interaction_type=LastQAInteractionType.SUPPORTING_QUESTION_ANSWER if skip else None,
                    question_source=QuestionSource.HUMAN_SUPPORTING_QUESTION if skip else QuestionSource.NONE,
                    linked_topic_id=state.linked_topic_id if skip else None,
                    linked_hop_id=state.linked_hop_id if skip else None,
                    merge_reason="case_3_never_merges",
                    skip_reason="strong_latest_context_match_and_guard_passed" if skip else "weak_match_or_context_incomplete",
                    is_authoritative_state=skip
                )
                validate_last_qa_resolution(resolution)
                return resolution

        resolution = LastQAResolution(
            rewritten_query=rewritten_query,
            state=None,
            did_merge_query=False,
            skip_broad_retrieval=False,
            confidence=0.0,
            path=LastQAPath.BROAD_RETRIEVAL_REQUIRED,
            merge_reason="no_safe_last_qa_merge",
            skip_reason="broad_retrieval_required",
            is_authoritative_state=False
        )
        validate_last_qa_resolution(resolution)
        return resolution


@dataclass
class LLMLastQAResolver:
    llm: LLMClient
    config: LastQAConfig
    prompt_registry: PromptRegistry = field(default_factory=lambda: DEFAULT_PROMPT_REGISTRY)

    @staticmethod
    def _require_broad_retrieval(
        rewritten_query: str,
        *,
        reason: str,
        diagnostic_context: dict[str, Any] | None = None,
    ) -> LastQAResolution:
        resolution = LastQAResolution(
            path=LastQAPath.BROAD_RETRIEVAL_REQUIRED,
            rewritten_query=rewritten_query,
            state=None,
            did_merge_query=False,
            skip_broad_retrieval=False,
            confidence=0.0,
            is_authoritative_state=False,
            diagnostic_context=dict(diagnostic_context or {}),
            merge_reason=reason,
            skip_reason="broad_retrieval_required",
        )
        validate_last_qa_resolution(resolution)
        return resolution

    def resolve_outbound_follow_up(
        self,
        request: ChatRequest,
        state: LastQAState,
        rewritten_query: str,
    ) -> LastQAResolution | None:
        """Classify a reference to the active outbound envelope semantically.

        The model may authorize only the relationship/action. Recipients,
        message content, credentials, and attachment paths remain outside this
        decision and are validated by the platform stage.
        """
        outbound = state.outbound_state
        if outbound is None:
            return None
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "outbound_action": {
                    "type": "string",
                    "enum": [
                        "none",
                        OutboundFollowUpAction.SEND.value,
                        OutboundFollowUpAction.REVISE.value,
                        OutboundFollowUpAction.REVISE_AND_SEND.value,
                    ],
                },
                "confidence": {"type": "number"},
            },
            "required": ["outbound_action", "confidence"],
        }
        try:
            payload = self.llm.generate_json(
                task=LLMTask.LAST_QA,
                system_prompt=self.prompt_registry.system("outbound_follow_up"),
                user_prompt=self.prompt_registry.user(
                    PromptContext(
                        stage="outbound_follow_up",
                        user_id=request.user_id,
                        rewritten_query=rewritten_query,
                        extra={
                            "active_outbound_state": {
                                "channel": outbound.channel,
                                "status": outbound.status,
                                "recipients": list(outbound.recipients),
                                "subject": outbound.subject,
                                "body_present": bool(outbound.body),
                                "attachment_filenames": list(
                                    outbound.attachment_filenames
                                ),
                            }
                        },
                    )
                ),
                schema=schema,
            )
            if is_structured_fallback(payload):
                return None
            validate_json_schema(payload, schema)
            action_value = str(payload.get("outbound_action") or "none")
            confidence = float(payload.get("confidence", 0.0))
            if action_value == "none" or not _confidence_meets(
                confidence, self.config.min_confidence
            ):
                return None
            action = OutboundFollowUpAction(action_value)
        except Exception as exc:
            _log_llm_fallback("Outbound Last-QA resolver", exc)
            return None

        resolution = LastQAResolution(
            path=LastQAPath.LATEST_CONTEXT_INTERACTION,
            interaction_type=LastQAInteractionType.OUTBOUND_MESSAGE_ACTION,
            question_source=QuestionSource.NONE,
            rewritten_query=rewritten_query,
            state=state,
            did_merge_query=False,
            skip_broad_retrieval=True,
            confidence=confidence,
            linked_topic_id=outbound.source_topic_id or state.linked_topic_id,
            linked_hop_id=outbound.source_hop_id or state.linked_hop_id,
            source_topic_id=outbound.source_topic_id or state.linked_topic_id,
            source_hop_id=outbound.source_hop_id or state.linked_hop_id,
            is_authoritative_state=True,
            merge_reason="active_outbound_message_action",
            skip_reason="semantic_action_bound_to_active_outbound_state",
            outbound_action=action,
        )
        validate_last_qa_resolution(resolution)
        return resolution

    def merge_clarification_answer(
        self, request: ChatRequest, state: LastQAState, rewritten_query: str
    ) -> dict[str, Any] | None:
        schema = {
            "type": "object",
            "properties": {
                "answered_clarification": {"type": "boolean"},
                "merged_query": {"type": "string"},
                "confidence": {"type": "number"},
            },
            "required": ["answered_clarification", "merged_query", "confidence"],
        }
        try:
            payload = self.llm.generate_json(
                task=LLMTask.CLARIFICATION_MERGE,
                system_prompt=self.prompt_registry.system("clarification_merge"),
                user_prompt=self.prompt_registry.user(
                    PromptContext(
                        stage="clarification_merge",
                        user_id=request.user_id,
                        rewritten_query=rewritten_query,
                        extra={"last_qa_state": state.__dict__},
                    )
                ),
                schema=schema,
            )
            if is_structured_fallback(payload):
                return None
            validate_json_schema(payload, schema)
            
            if not payload.get("answered_clarification"):
                return None
            if not _confidence_meets(
                payload.get("confidence", 0.0),
                self.config.clarification_merge_min_confidence,
            ):
                return None
            if not str(payload.get("merged_query") or "").strip():
                return None
                
            return payload
        except Exception as e:
            _log_llm_fallback("Clarification merge", e)
            return None

    def resolve(
        self, request: ChatRequest, rewritten_query: str, state: LastQAState | None
    ) -> LastQAResolution:
        reminder_resolution = _resolve_exact_reminder_notification_reply(
            request, rewritten_query, state
        )
        if reminder_resolution is not None:
            return reminder_resolution

        if state is None:
            res = LastQAResolution(
                path=LastQAPath.NO_LAST_QA,
                rewritten_query=rewritten_query,
                state=None,
                did_merge_query=False,
                skip_broad_retrieval=False,
                is_authoritative_state=False,
                merge_reason=None,
                skip_reason="no_last_qa_exists",
            )
            validate_last_qa_resolution(res)
            return res

        outbound_resolution = self.resolve_outbound_follow_up(
            request,
            state,
            rewritten_query,
        )
        if outbound_resolution is not None:
            return outbound_resolution

        is_clarification = (
            state.response_type == ResponseType.CLARIFICATION or
            (state.clarification_question is not None)
        )
        
        if is_clarification:
            if not self.config.clarification_merge_enabled:
                res = LastQAResolution(
                    path=LastQAPath.BROAD_RETRIEVAL_REQUIRED,
                    rewritten_query=rewritten_query,
                    state=None,
                    did_merge_query=False,
                    skip_broad_retrieval=False,
                    confidence=0.0,
                    is_authoritative_state=False,
                    diagnostic_context={"previous_last_qa_state": "safe_summary_only"},
                    merge_reason="clarification_merge_disabled",
                    skip_reason="broad_retrieval_required",
                )
                validate_last_qa_resolution(res)
                return res

            merge_payload = self.merge_clarification_answer(request, state, rewritten_query)
            if merge_payload:
                merged = str(merge_payload.get("merged_query") or rewritten_query).strip() or rewritten_query
                res = LastQAResolution(
                    path=LastQAPath.CLARIFICATION_CHECK,
                    interaction_type=LastQAInteractionType.CLARIFICATION_ANSWER,
                    question_source=QuestionSource.CLARIFICATION_QUESTION,
                    rewritten_query=merged,
                    state=state,
                    did_merge_query=True,
                    skip_broad_retrieval=False,
                    confidence=float(merge_payload.get("confidence", 0.0)),
                    is_authoritative_state=False,
                    missing_context=[],
                    merge_reason="current_query_answered_previous_clarification",
                    skip_reason="clarification_merge_must_continue_to_broad_retrieval",
                )
                validate_last_qa_resolution(res)
                return res
            
            res = LastQAResolution(
                path=LastQAPath.BROAD_RETRIEVAL_REQUIRED,
                rewritten_query=rewritten_query,
                state=None,
                did_merge_query=False,
                skip_broad_retrieval=False,
                confidence=0.0,
                is_authoritative_state=False,
                diagnostic_context={"previous_last_qa_state": "safe_summary_only"},
                merge_reason="clarification_not_answered_or_low_confidence",
                skip_reason="broad_retrieval_required",
            )
            validate_last_qa_resolution(res)
            return res

        if state.response_type in (
            ResponseType.NORMAL,
            ResponseType.KNOWLEDGE_ACTION,
            ResponseType.REMINDER_ACTION,
            ResponseType.REMINDER_REPLY,
            ResponseType.ERROR,
            ResponseType.SAFE_NOOP,
        ):
            active_questions: list[tuple[QuestionSource, GeneratedQuestion]] = [
                (QuestionSource.HUMAN_SUPPORTING_QUESTION, question)
                for question in state.supporting_questions
                if question.text.strip()
            ]
            if (
                state.reminder_supporting_question is not None
                and state.reminder_supporting_question.text.strip()
            ):
                active_questions.append(
                    (
                        QuestionSource.REMINDER_SUPPORTING_QUESTION,
                        state.reminder_supporting_question,
                    )
                )

            relationship_values = [
                "unrelated_or_uncertain",
                LastQAInteractionType.NORMAL_FOLLOW_UP.value,
            ]
            properties: dict[str, Any] = {
                "relationship": {
                    "type": "string",
                    "enum": relationship_values,
                },
                "confidence": {"type": "number"},
            }
            required = ["relationship", "confidence"]
            if active_questions:
                relationship_values.append(
                    LastQAInteractionType.SUPPORTING_QUESTION_ANSWER.value
                )
                properties["matched_question_index"] = {
                    "type": "integer",
                    "default": -1,
                }
                required.insert(1, "matched_question_index")

            schema = {
                "type": "object",
                "additionalProperties": False,
                "properties": properties,
                "required": required,
            }
            try:
                payload = self.llm.generate_json(
                    task=LLMTask.LAST_QA,
                    system_prompt=self.prompt_registry.system("last_qa"),
                    user_prompt=self.prompt_registry.user(
                        PromptContext(
                            stage="last_qa",
                            user_id=request.user_id,
                            rewritten_query=rewritten_query,
                            metadata=request.metadata,
                            platform_context=request.platform_context,
                            extra={
                                "latest_exchange": {
                                    "previous_user_query": state.last_user_query,
                                    "previous_assistant_response": state.last_response,
                                    "response_type": state.response_type.value,
                                    "expected_response_type": (
                                        state.expected_response_type.value
                                        if state.expected_response_type is not None
                                        else None
                                    ),
                                    "linked_context_available": bool(
                                        state.linked_topic_id
                                        and state.linked_hop_id
                                    ),
                                },
                                "active_supporting_questions": [
                                    {
                                        "index": index,
                                        "source": source.value,
                                        "question": question.text,
                                        "expected_response_type": (
                                            question.expected_response_type.value
                                        ),
                                    }
                                    for index, (source, question) in enumerate(
                                        active_questions
                                    )
                                ]
                            },
                        )
                    ),
                    schema=schema,
                )
                if is_structured_fallback(payload):
                    return self._require_broad_retrieval(
                        rewritten_query,
                        reason="last_qa_structured_generation_failed",
                        diagnostic_context={
                            "model_failure_policy": "conservative_broad_retrieval"
                        },
                    )
                validate_json_schema(payload, schema)

                relationship = str(payload.get("relationship") or "")
                question_index = int(payload.get("matched_question_index", -1))
                confidence = _unit_confidence(payload.get("confidence", 0.0))
                if confidence is None:
                    confidence = 0.0
                if (
                    relationship
                    == LastQAInteractionType.SUPPORTING_QUESTION_ANSWER.value
                    and 0 <= question_index < len(active_questions)
                    and confidence >= self.config.min_confidence
                ):
                    question_source, matched_question = active_questions[question_index]
                    interaction_type = LastQAInteractionType.SUPPORTING_QUESTION_ANSWER
                    bound_payload = {
                        "relationship": relationship,
                        "question_source": question_source.value,
                        "matched_question": matched_question.text,
                        "confidence": confidence,
                    }
                    if can_skip_broad_retrieval(bound_payload, state, interaction_type, self.config, request):
                        md = request.metadata or {}
                        pc = request.platform_context or {}
                        res = LastQAResolution(
                            path=LastQAPath.LATEST_CONTEXT_INTERACTION,
                            interaction_type=interaction_type,
                            question_source=question_source,
                            rewritten_query=rewritten_query,
                            state=state,
                            did_merge_query=False,
                            skip_broad_retrieval=True,
                            confidence=confidence,
                            linked_topic_id=state.linked_topic_id,
                            linked_hop_id=state.linked_hop_id,
                            matched_question=matched_question.text,
                            reminder_id=md.get("reminder_id") or pc.get("reminder_id"),
                            notification_id=md.get("notification_id") or pc.get("notification_id"),
                            source_topic_id=md.get("source_topic_id") or pc.get("source_topic_id"),
                            source_hop_id=md.get("source_hop_id") or pc.get("source_hop_id"),
                            is_authoritative_state=True,
                            missing_context=[],
                            merge_reason="case_3_never_merges",
                            skip_reason="strong_latest_context_match_and_guard_passed",
                        )
                        validate_last_qa_resolution(res)
                        return res

                if (
                    relationship == LastQAInteractionType.NORMAL_FOLLOW_UP.value
                    and question_index == -1
                    and confidence >= self.config.min_confidence
                ):
                    interaction_type = LastQAInteractionType.NORMAL_FOLLOW_UP
                    bound_payload = {
                        "relationship": relationship,
                        "question_source": QuestionSource.NONE.value,
                        "matched_question": "",
                        "confidence": confidence,
                    }
                    if can_skip_broad_retrieval(
                        bound_payload,
                        state,
                        interaction_type,
                        self.config,
                        request,
                    ):
                        res = LastQAResolution(
                            path=LastQAPath.LATEST_CONTEXT_INTERACTION,
                            interaction_type=interaction_type,
                            question_source=QuestionSource.NONE,
                            rewritten_query=rewritten_query,
                            state=state,
                            did_merge_query=False,
                            skip_broad_retrieval=True,
                            confidence=confidence,
                            linked_topic_id=state.linked_topic_id,
                            linked_hop_id=state.linked_hop_id,
                            is_authoritative_state=True,
                            missing_context=[],
                            diagnostic_context={
                                "model_relationship": relationship,
                            },
                            merge_reason=(
                                "current_message_depends_on_latest_exchange"
                            ),
                            skip_reason=(
                                "high_confidence_latest_context_relationship"
                            ),
                        )
                        validate_last_qa_resolution(res)
                        return res

                res = LastQAResolution(
                    path=LastQAPath.BROAD_RETRIEVAL_REQUIRED,
                    rewritten_query=rewritten_query,
                    state=None,
                    did_merge_query=False,
                    skip_broad_retrieval=False,
                    confidence=confidence,
                    is_authoritative_state=False,
                    diagnostic_context={
                        "model_relationship": relationship,
                        "matched_question_index": question_index,
                    },
                    merge_reason="model_did_not_authorize_latest_context",
                    skip_reason=(
                        "relationship_uncertain_or_latest_context_guard_failed"
                    ),
                )
                validate_last_qa_resolution(res)
                return res
            except Exception as e:
                _log_llm_fallback("Last-QA resolver", e)
                return self._require_broad_retrieval(
                    rewritten_query,
                    reason="last_qa_model_error",
                    diagnostic_context={
                        "model_failure_policy": "conservative_broad_retrieval"
                    },
                )

        return self._require_broad_retrieval(
            rewritten_query,
            reason="last_qa_response_type_not_eligible",
        )




from .contracts import ApprovedConversationContext

class IntentClassifierProtocol:
    def classify(
        self, 
        request: ChatRequest, 
        rewritten_query: str, 
        last_qa_resolution: LastQAResolution | None = None,
        approved_conversation_context: ApprovedConversationContext | None = None
    ) -> Intent:
        ...

@dataclass
class LLMIntentClassifier(IntentClassifierProtocol):
    llm: LLMClient
    config: ClassificationConfig
    prompt_registry: PromptRegistry = field(default_factory=lambda: DEFAULT_PROMPT_REGISTRY)

    def classify(
        self, 
        request: ChatRequest, 
        rewritten_query: str, 
        last_qa_resolution: LastQAResolution | None = None,
        approved_conversation_context: ApprovedConversationContext | None = None
    ) -> Intent:
        explicit_intent = request.metadata.get("intent")
        if explicit_intent:
            try:
                return Intent(explicit_intent)
            except ValueError:
                pass

        if _is_authoritative_outbound_action(last_qa_resolution):
            return Intent.GENERAL_RESPONSE

        schema = intent_classification_schema()
        
        # Include compact approved chat history so intent can resolve safe follow-ups.
        context_extra = build_intent_conversation_extra(
            approved_conversation_context=approved_conversation_context,
            last_qa_resolution=last_qa_resolution,
        )

        try:
            payload = self.llm.generate_json(
                task=LLMTask.INTENT,
                system_prompt=self.prompt_registry.system("intent_classifier"),
                user_prompt=self.prompt_registry.user(
                    PromptContext(
                        stage="intent_classifier",
                        user_id=request.user_id,
                        rewritten_query=rewritten_query,
                        metadata=request.metadata,
                        platform_context=request.platform_context,
                        extra=context_extra,
                    )
                ),
                schema=schema,
            )
            if is_structured_fallback(payload):
                return Intent.GENERAL_RESPONSE
            validate_json_schema(payload, schema)
            
            if float(payload.get("confidence", 0.0)) >= self.config.min_confidence:
                intent = Intent(str(payload["intent"]))
                if (
                    intent is Intent.CLARIFICATION
                    and not _has_pending_clarification(last_qa_resolution)
                ):
                    return Intent.GENERAL_RESPONSE
                return intent
        except Exception as e:
            _log_llm_fallback("Intent classifier", e)

        return Intent.GENERAL_RESPONSE

class KeywordIntentClassifier(IntentClassifierProtocol):
    def __init__(self, config: ClassificationConfig) -> None:
        self.config = config

    def classify(
        self, 
        request: ChatRequest, 
        rewritten_query: str, 
        last_qa_resolution: LastQAResolution | None = None,
        approved_conversation_context: ApprovedConversationContext | None = None
    ) -> Intent:
        explicit_intent = request.metadata.get("intent")
        if explicit_intent:
            return Intent(explicit_intent)
        query = rewritten_query.casefold()
        for intent_name, keywords in self.config.intent_keywords.items():
            if any(keyword.casefold() in query for keyword in keywords):
                return Intent(intent_name)
        return Intent.GENERAL_RESPONSE

# Expose IntentClassifier as the base protocol for type hints
IntentClassifier = IntentClassifierProtocol
