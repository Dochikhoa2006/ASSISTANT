"""Dynamic Question Generation Strategies."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Protocol

from .config import GeneralPurposeConfig, QuestionGenerationConfig
from .contracts import ContentComposerResult, GeneratedQuestion, HumanSupportingDecision, PipelineContext, QuestionSource, ExpectedResponseType
from .llm import LLMClient, LLMTask
from .prompts import PromptContext, PromptRegistry

logger = logging.getLogger(__name__)


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
        
        schema = {
            "type": "object",
            "properties": {
                "question_text": {"type": "string"},
                "question_source": {"type": "string"},
                "purpose": {"type": "string"},
                "confidence": {"type": "number"},
                "should_ask": {"type": "boolean"},
                "expected_response_type": {
                    "type": "string",
                    "enum": [e.value for e in ExpectedResponseType]
                },
                "reason_summary": {"type": "string"}
            },
            "required": ["question_text", "question_source", "purpose", "confidence", "should_ask", "expected_response_type", "reason_summary"]
        }
        
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
            
            should_ask = payload.get("should_ask", True)
            if not should_ask:
                return None
                
            return GeneratedQuestion(
                text=str(payload["question_text"]),
                source=QuestionSource.CLARIFICATION_QUESTION,
                purpose=str(payload.get("purpose", "resolve_missing_info")),
                confidence=float(payload.get("confidence", 1.0)),
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
        if confidence >= self.config.question_generation_confidence_threshold:
            return [], None
            
        if not self.config.enabled:
            return [], None

        schema = {
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question_text": {"type": "string"},
                            "question_source": {"type": "string"},
                            "purpose": {"type": "string"},
                            "confidence": {"type": "number"},
                            "should_ask": {"type": "boolean"},
                            "expected_response_type": {
                                "type": "string",
                                "enum": [e.value for e in ExpectedResponseType]
                            },
                            "reason_summary": {"type": "string"}
                        },
                        "required": ["question_text", "question_source", "purpose", "confidence", "should_ask", "expected_response_type", "reason_summary"]
                    }
                }
            },
            "required": ["questions"]
        }

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
                        },
                    )
                ),
                schema=schema,
            )
            
            questions = []
            for q in payload.get("questions", []):
                if not q.get("should_ask", True):
                    continue
                questions.append(
                    GeneratedQuestion(
                        text=str(q["question_text"]),
                        source=QuestionSource.HUMAN_SUPPORTING_QUESTION,
                        purpose=str(q.get("purpose", "optional_context")),
                        confidence=float(q.get("confidence", 1.0)),
                        should_ask=True,
                        expected_response_type=ExpectedResponseType(q.get("expected_response_type", ExpectedResponseType.UNKNOWN.value)),
                    )
                )
            
            questions = questions[:self.config.human_supporting_max_count]
            
        except Exception:
            return [], None
            
        hitl_result = {
            "triggered": True,
            "confidence": confidence,
            "question_count": len(questions),
        }
        return questions, hitl_result


@dataclass
class LLMReminderSupportingStrategy:
    llm: LLMClient
    prompt_registry: PromptRegistry
    config: QuestionGenerationConfig

    def generate(self, context: PipelineContext, **kwargs: Any) -> GeneratedQuestion | None:
        if not self.config.enabled or not self.config.reminder_supporting_enabled:
            return None

        schema = {
            "type": "object",
            "properties": {
                "question_text": {"type": "string"},
                "question_source": {"type": "string"},
                "purpose": {"type": "string"},
                "confidence": {"type": "number"},
                "should_ask": {"type": "boolean"},
                "expected_response_type": {
                    "type": "string",
                    "enum": [e.value for e in ExpectedResponseType]
                },
                "reason_summary": {"type": "string"}
            },
            "required": ["question_text", "question_source", "purpose", "confidence", "should_ask", "expected_response_type", "reason_summary"]
        }

        try:
            payload = self.llm.generate_json(
                task=LLMTask.GENERATE_REMINDER_SUPPORTING,
                system_prompt=self.prompt_registry.system("question_generation"),
                user_prompt=self.prompt_registry.user(
                    PromptContext(
                        stage="generate_reminder_supporting",
                        user_id=context.request.user_id,
                        rewritten_query=context.rewritten_query,
                        intent=context.intent.value,
                        extra={
                            "reminder_metadata": kwargs.get("reminder_metadata", {}),
                            "action_summary": kwargs.get("action_summary", ""),
                        }
                    )
                ),
                schema=schema,
            )
            
            should_ask = payload.get("should_ask", True)
            confidence = float(payload.get("confidence", 1.0))
            
            if not should_ask or confidence < self.config.reminder_supporting_min_confidence:
                return None
                
            return GeneratedQuestion(
                text=str(payload["question_text"]),
                source=QuestionSource.REMINDER_SUPPORTING_QUESTION,
                purpose=str(payload.get("purpose", "reminder_followup")),
                confidence=confidence,
                should_ask=True,
                expected_response_type=ExpectedResponseType(payload.get("expected_response_type", ExpectedResponseType.UNKNOWN.value)),
            )
        except Exception:
            return None


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
            
        schema = {
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question_text": {"type": "string"},
                            "question_source": {"type": "string"},
                            "purpose": {"type": "string"},
                            "confidence": {"type": "number"},
                            "should_ask": {"type": "boolean"},
                            "expected_response_type": {
                                "type": "string",
                                "enum": [e.value for e in ExpectedResponseType]
                            },
                            "reason_summary": {"type": "string"}
                        },
                        "required": ["question_text", "question_source", "purpose", "confidence", "should_ask", "expected_response_type", "reason_summary"]
                    }
                }
            },
            "required": ["questions"]
        }

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
                            "task_type": "general_supporting_question_decision",
                            "decision_rule": (
                                "Ask only when one specific next-turn answer would materially improve "
                                "the drafted response. Otherwise set should_ask=false."
                            ),
                        },
                    )
                ),
                schema=schema,
            )

            questions = payload.get("questions", [])[:1]
            for q in questions:
                question = str(q.get("question_text") or "").strip()
                confidence = float(q.get("confidence", 0.0))
                if (
                    q.get("should_ask")
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
                            q.get(
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
                        reason_summary=str(q.get("reason_summary", "")),
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
