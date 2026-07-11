"""Query rewrite, Last-QA resolution, and intent classification."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any

from .config import ClassificationConfig, LastQAConfig
from .contracts import (
    ChatRequest, Intent, LastQAState, ResponseType, LastQAPath,
    LastQAInteractionType, QuestionSource, LastQAResolution,
    validate_last_qa_resolution
)
from .llm import LLMClient, LLMTask, build_intent_conversation_extra, validate_json_schema
from .prompts import DEFAULT_PROMPT_REGISTRY, PromptContext, PromptRegistry
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

logger = logging.getLogger(__name__)


def _log_llm_fallback(stage: str, exc: Exception) -> None:
    logger.debug("%s LLM fallback engaged: %s", stage, exc)



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


class SupportingQuestionMatcher:
    def __init__(self, match_threshold: float = 0.7) -> None:
        self.match_threshold = match_threshold

    def matches(self, query: str, supporting_questions: list[Any]) -> bool:
        if not query or not supporting_questions:
            return False
        
        query_terms = set(query.casefold().split())
        if not query_terms:
            return False

        for question in supporting_questions:
            question_terms = set(question.text.casefold().split())
            if not question_terms:
                continue
            intersection = query_terms.intersection(question_terms)
            overlap_ratio = len(intersection) / len(query_terms)
            if overlap_ratio >= self.match_threshold:
                return True
        return False


def can_skip_broad_retrieval(
    payload: dict[str, Any], state: LastQAState, interaction_type: LastQAInteractionType, config: LastQAConfig, request: ChatRequest
) -> bool:
    confidence = float(payload.get("confidence", 0.0))
    if confidence < config.skip_broad_retrieval_min_confidence:
        return False
    if not payload.get("llm_suggested_skip_broad_retrieval") and not payload.get("skip_broad_retrieval"):
        return False
        
        
    if interaction_type.value not in config.skip_allowed_interaction_types:
        return False

    question_source = str(payload.get("question_source") or QuestionSource.NONE.value)
    matched_question = " ".join(str(payload.get("matched_question") or "").casefold().split())

    if interaction_type == LastQAInteractionType.NORMAL_FOLLOW_UP:
        return bool(
            state.linked_topic_id
            and state.linked_hop_id
            and state.last_response
            and question_source == QuestionSource.NONE.value
            and not matched_question
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
        if state is None:
            resolution = LastQAResolution(
                rewritten_query=rewritten_query, 
                state=None, 
                did_merge_query=False,
                skip_broad_retrieval=False,
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
                path=LastQAPath.BROAD_RETRIEVAL_REQUIRED,
                merge_reason="clarification_not_answered_or_low_confidence",
                skip_reason="broad_retrieval_required",
                is_authoritative_state=False,
                missing_context=["semantic clarification merge unavailable"],
                diagnostic_context={"previous_last_qa_state": "safe_summary_only"}
            )
            validate_last_qa_resolution(resolution)
            return resolution

        if state.response_type in (ResponseType.NORMAL, ResponseType.KNOWLEDGE_ACTION, ResponseType.REMINDER_ACTION, ResponseType.REMINDER_REPLY, ResponseType.ERROR):
            if state.supporting_questions and self.matcher.matches(rewritten_query, state.supporting_questions):
                skip = bool(state.linked_topic_id and state.linked_hop_id)
                resolution = LastQAResolution(
                    rewritten_query=rewritten_query, 
                    state=state, 
                    did_merge_query=False,
                    skip_broad_retrieval=skip, 
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

    def __post_init__(self) -> None:
        self.fallback = LastQAResolver()

    def merge_clarification_answer(
        self, request: ChatRequest, state: LastQAState, rewritten_query: str
    ) -> dict[str, Any] | None:
        schema = {
            "type": "object",
            "properties": {
                "answered_clarification": {"type": "boolean"},
                "merged_query": {"type": "string"},
                "confidence": {"type": "number"},
                "missing_context": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["answered_clarification", "merged_query", "confidence", "missing_context"],
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
            validate_json_schema(payload, schema)
            
            if not payload.get("answered_clarification"):
                return None
            if float(payload.get("confidence", 0.0)) < self.config.clarification_merge_min_confidence:
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
                    is_authoritative_state=False,
                    missing_context=merge_payload.get("missing_context", []),
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
                is_authoritative_state=False,
                diagnostic_context={"previous_last_qa_state": "safe_summary_only"},
                merge_reason="clarification_not_answered_or_low_confidence",
                skip_reason="broad_retrieval_required",
            )
            validate_last_qa_resolution(res)
            return res

        if state.response_type in (ResponseType.NORMAL, ResponseType.KNOWLEDGE_ACTION, ResponseType.REMINDER_ACTION, ResponseType.REMINDER_REPLY, ResponseType.ERROR):
            schema = {
                "type": "object",
                "properties": {
                "interaction_detected": {"type": "boolean"},
                    "interaction_type": {
                        "type": "string",
                        "enum": [
                            LastQAInteractionType.CLARIFICATION_ANSWER.value,
                            LastQAInteractionType.SUPPORTING_QUESTION_ANSWER.value,
                            LastQAInteractionType.NORMAL_FOLLOW_UP.value,
                            LastQAInteractionType.REMINDER_NOTIFICATION_REPLY.value,
                            "unrelated",
                            "ambiguous",
                        ],
                    },
                    "question_source": {
                        "type": "string",
                        "enum": [source.value for source in QuestionSource],
                    },
                    "matched_question": {"type": "string"},
                    "confidence": {"type": "number"},
                    "llm_suggested_skip_broad_retrieval": {"type": "boolean"},
                },
                "required": [
                    "interaction_detected",
                    "interaction_type",
                    "question_source",
                    "confidence",
                    "llm_suggested_skip_broad_retrieval",
                ],
            }
            try:
                payload = self.llm.generate_json(
                    task=LLMTask.LAST_QA,
                    system_prompt=self.prompt_registry.system("last_qa"),
                    user_prompt=self.prompt_registry.user(
                        PromptContext(
                            stage="last_qa",
                            user_id=request.user_id,
                            raw_query=request.raw_query,
                            rewritten_query=rewritten_query,
                            metadata=request.metadata,
                            platform_context=request.platform_context,
                            extra={"last_qa_state": state.__dict__},
                        )
                    ),
                    schema=schema,
                )
                validate_json_schema(payload, schema)

                it_str = payload.get("interaction_type")
                qs_str = payload.get("question_source")
                try:
                    interaction_type = LastQAInteractionType(it_str) if it_str else None
                except ValueError:
                    interaction_type = None
                try:
                    question_source = QuestionSource(qs_str) if qs_str else QuestionSource.NONE
                except ValueError:
                    question_source = QuestionSource.NONE

                if payload.get("interaction_detected") and interaction_type and float(payload.get("confidence", 0.0)) >= self.config.min_confidence:
                    if can_skip_broad_retrieval(payload, state, interaction_type, self.config, request):
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
                            linked_topic_id=state.linked_topic_id,
                            linked_hop_id=state.linked_hop_id,
                            matched_question=payload.get("matched_question"),
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
            except Exception as e:
                _log_llm_fallback("Last-QA resolver", e)

        return self.fallback.resolve(request, rewritten_query, state)




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

        schema = {
            "type": "object",
            "properties": {
                "intent": {"type": "string", "enum": [e.value for e in Intent]},
                "confidence": {"type": "number"},
            },
            "required": ["intent", "confidence"],
        }
        
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
                        raw_query=request.raw_query,
                        rewritten_query=rewritten_query,
                        metadata=request.metadata,
                        platform_context=request.platform_context,
                        extra=context_extra,
                    )
                ),
                schema=schema,
            )
            validate_json_schema(payload, schema)
            
            intent_str = payload.get("intent")
            if intent_str and float(payload.get("confidence", 0.0)) >= self.config.min_confidence:
                return Intent(intent_str)
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
