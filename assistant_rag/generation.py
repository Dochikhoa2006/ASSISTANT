"""Dynamic Question Generation Strategies."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Protocol

from .config import GeneralPurposeConfig, QuestionGenerationConfig
from .contracts import ContentComposerResult, GeneratedQuestion, HumanSupportingDecision, LastQAInteractionType, PipelineContext, QuestionSource, ExpectedResponseType
from .llm import LLMClient, LLMTask, is_structured_fallback
from .platform import is_explicit_email_message_request
from .prompts import PromptContext, PromptRegistry

logger = logging.getLogger(__name__)


def _question_item_schema() -> dict[str, Any]:
    """Return only values the question model genuinely has to decide."""
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "question_text": {"type": "string"},
            "confidence": {"type": "number"},
            "should_ask": {"type": "boolean"},
            "expected_response_type": {
                "type": "string",
                "enum": [item.value for item in ExpectedResponseType],
            },
        },
        "required": [
            "question_text",
            "confidence",
            "should_ask",
            "expected_response_type",
        ],
    }


def _required_question_schema() -> dict[str, Any]:
    """Represent one included question without a redundant ask/no-ask mirror."""
    schema = _question_item_schema()
    schema["properties"].pop("should_ask")
    schema["required"].remove("should_ask")
    return schema


def _question_list_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "questions": {
                "type": "array",
                # Inclusion in this array already means "ask"; an empty array
                # represents the no-question decision.
                "items": _required_question_schema(),
            }
        },
        "required": ["questions"],
    }


class QuestionGenerationStrategy(Protocol):
    def generate(self, context: PipelineContext, **kwargs: Any) -> list[GeneratedQuestion] | GeneratedQuestion | None:
        ...


@dataclass
class LLMClarificationStrategy:
    llm: LLMClient
    prompt_registry: PromptRegistry
    config: QuestionGenerationConfig

    def generate(self, context: PipelineContext, **kwargs: Any) -> GeneratedQuestion | None:
        missing_fields = kwargs.get("missing_fields", [])
        ambiguity_reason = kwargs.get("ambiguity_reason", "Missing required fields for safe action.")
        
        # The branch already owns the ask decision and needs only the question.
        schema = _required_question_schema()
        
        try:
            payload = self.llm.generate_json(
                task=LLMTask.GENERATE_CLARIFICATION,
                system_prompt=self.prompt_registry.system("question_generation"),
                user_prompt=self.prompt_registry.user(
                    PromptContext(
                        stage="generate_clarification",
                        user_id=context.request.user_id,
                        rewritten_query=context.rewritten_query,
                        intent=context.intent.value,
                        extra={
                            "task_type": "clarification",
                            "missing_fields": missing_fields,
                            "ambiguity_reason": ambiguity_reason,
                            "generation_instruction": (
                                "Generate a new clarification question for the current turn. "
                                "Do not reuse a precomputed clarification_question from request metadata."
                            ),
                        },
                        chat_history=context.chat_history,
                    )
                ),
                schema=schema,
            )
            if is_structured_fallback(payload):
                return None

            question_text = str(payload["question_text"]).strip()
            confidence = float(payload.get("confidence", 0.0))
            if (
                not question_text
                or confidence
                < self.config.question_generation_confidence_threshold
            ):
                return None

            return GeneratedQuestion(
                text=question_text,
                source=QuestionSource.CLARIFICATION_QUESTION,
                purpose="resolve_missing_info",
                confidence=confidence,
                should_ask=True,
                expected_response_type=ExpectedResponseType(payload.get("expected_response_type", ExpectedResponseType.UNKNOWN.value)),
            )
        except Exception:
            return None


@dataclass
class LLMHumanInTheLoopStrategy:
    llm: LLMClient
    prompt_registry: PromptRegistry
    config: QuestionGenerationConfig

    def evaluate(self, context: PipelineContext, response_text: str, confidence: float) -> tuple[list[GeneratedQuestion], dict[str, Any] | None]:
        if is_explicit_email_message_request(context.rewritten_query):
            return [], {
                "triggered": False,
                "confidence": 1.0,
                "question_count": 0,
                "reason": "explicit_delivery_request_is_actionable",
            }
        if confidence >= self.config.question_generation_confidence_threshold:
            return [], None
            
        if not self.config.enabled:
            return [], None

        schema = _question_list_schema()

        try:
            payload = self.llm.generate_json(
                task=LLMTask.GENERATE_HUMAN_SUPPORTING,
                system_prompt=self.prompt_registry.system("question_generation"),
                user_prompt=self.prompt_registry.user(
                    PromptContext(
                        stage="generate_human_supporting",
                        user_id=context.request.user_id,
                        rewritten_query=context.rewritten_query,
                        intent=context.intent.value,
                        extra={
                            "response_text": response_text,
                            "confidence": confidence,
                            "task_type": "required_general_context_decision",
                            "decision_rule": (
                                "Include one question only when a required user-provided fact "
                                "is missing and the current request cannot otherwise be fulfilled."
                            ),
                        },
                    )
                ),
                schema=schema,
            )
            if is_structured_fallback(payload):
                return [], None
            
            questions = []
            for q in payload.get("questions", []):
                questions.append(
                    GeneratedQuestion(
                        text=str(q["question_text"]),
                        source=QuestionSource.HUMAN_SUPPORTING_QUESTION,
                        purpose="resolve_required_context",
                        confidence=float(q.get("confidence", 1.0)),
                        should_ask=True,
                        expected_response_type=ExpectedResponseType(q.get("expected_response_type", ExpectedResponseType.UNKNOWN.value)),
                    )
                )
            
            questions = questions[:self.config.human_supporting_max_count]
            
        except Exception:
            return [], None
            
        hitl_result = {
            "triggered": bool(questions),
            "confidence": confidence,
            "question_count": len(questions),
        }
        return questions, hitl_result


@dataclass
class LLMGeneralHITLStrategy:
    llm: LLMClient
    prompt_registry: PromptRegistry
    config: GeneralPurposeConfig

    def evaluate(
        self,
        *,
        context: PipelineContext,
        response_text: str,
        approved_conversation_history: list[dict[str, Any]],
        merged_supporting_detail: str,
    ) -> HumanSupportingDecision:
        if not self.config.hitl_supporting_question_enabled:
            return HumanSupportingDecision(
                should_ask=False, question="", confidence=1.0,
                question_source=QuestionSource.HUMAN_SUPPORTING_QUESTION,
                expected_response_type=ExpectedResponseType.UNKNOWN,
                reason_summary="Disabled by config", risk_flags=()
            )

        if (
            context.last_qa_trace.get("interaction_type")
            == LastQAInteractionType.OUTBOUND_MESSAGE_ACTION.value
        ):
            return HumanSupportingDecision(
                should_ask=False,
                question="",
                confidence=1.0,
                question_source=QuestionSource.HUMAN_SUPPORTING_QUESTION,
                expected_response_type=ExpectedResponseType.UNKNOWN,
                reason_summary=(
                    "An authoritative outbound action needs no additional context question."
                ),
                risk_flags=(),
            )

        if is_explicit_email_message_request(context.rewritten_query):
            return HumanSupportingDecision(
                should_ask=False,
                question="",
                confidence=1.0,
                question_source=QuestionSource.HUMAN_SUPPORTING_QUESTION,
                expected_response_type=ExpectedResponseType.UNKNOWN,
                reason_summary=(
                    "The explicit delivery request already supplies an actionable recipient."
                ),
                risk_flags=(),
            )
            
        # General HITL is contractually limited to one required-context question, so an
        # array wrapper adds output tokens and invalid-shape opportunities.
        schema = _question_item_schema()

        try:
            payload = self.llm.generate_json(
                task=LLMTask.GENERATE_HUMAN_SUPPORTING,
                system_prompt=self.prompt_registry.system("question_generation"),
                user_prompt=self.prompt_registry.user(
                    PromptContext(
                        stage="general_hitl_evaluation",
                        user_id=context.request.user_id,
                        rewritten_query=context.rewritten_query,
                        extra={
                            "draft_response": response_text,
                            "approved_conversation_history": approved_conversation_history,
                            "merged_supporting_detail": merged_supporting_detail,
                            "task_type": "required_general_context_decision",
                            "decision_rule": (
                                "Ask only when the current request cannot be fulfilled because one "
                                "required user-provided fact is missing. Never ask for delivery "
                                "credentials, information already present, preferences that merely "
                                "improve an already complete response, or a next-step suggestion. "
                                "Otherwise set should_ask=false."
                            ),
                        },
                    )
                ),
                schema=schema,
            )
            if is_structured_fallback(payload):
                return HumanSupportingDecision(
                    should_ask=False,
                    question="",
                    confidence=0.0,
                    question_source=QuestionSource.HUMAN_SUPPORTING_QUESTION,
                    expected_response_type=ExpectedResponseType.UNKNOWN,
                    reason_summary="Supporting-question generation was unavailable.",
                    risk_flags=("structured_output_fallback",),
                )

            question = str(payload.get("question_text") or "").strip()
            confidence = float(payload.get("confidence", 0.0))
            if (
                payload.get("should_ask")
                and question
                and len(question) <= self.config.hitl_supporting_question_max_length
                and confidence >= self.config.hitl_supporting_question_confidence_threshold
                and self._is_novel_question(
                    question,
                    response_text=response_text,
                    approved_conversation_history=approved_conversation_history,
                )
            ):
                try:
                    expected_response_type = ExpectedResponseType(
                        payload.get(
                            "expected_response_type",
                            ExpectedResponseType.UNKNOWN.value,
                        )
                    )
                except (TypeError, ValueError):
                    expected_response_type = ExpectedResponseType.UNKNOWN
                return HumanSupportingDecision(
                    should_ask=True,
                    question=question,
                    confidence=confidence,
                    question_source=QuestionSource.HUMAN_SUPPORTING_QUESTION,
                    expected_response_type=expected_response_type,
                    reason_summary="One required missing-context question was selected.",
                    risk_flags=(),
                )
        except Exception as e:
            logger.debug("General HITL evaluation failed: %s", e)

        return HumanSupportingDecision(
            should_ask=False, question="", confidence=0.0,
            question_source=QuestionSource.HUMAN_SUPPORTING_QUESTION,
            expected_response_type=ExpectedResponseType.UNKNOWN,
            reason_summary="HITL generation failed", risk_flags=()
        )

    @staticmethod
    def _is_novel_question(
        question: str,
        *,
        response_text: str,
        approved_conversation_history: list[dict[str, Any]],
    ) -> bool:
        normalized_question = " ".join(question.casefold().split())
        if normalized_question in " ".join(response_text.casefold().split()):
            return False
        for hop in approved_conversation_history:
            prior_text = " ".join(str(hop.get("text") or "").casefold().split())
            if normalized_question and normalized_question in prior_text:
                return False
        return True
