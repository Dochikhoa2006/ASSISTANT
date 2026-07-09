"""Dynamic Question Generation Strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .config import QuestionGenerationConfig
from .contracts import GeneratedQuestion, PipelineContext, QuestionSource
from .llm import LLMClient, LLMTask
from .prompts import PromptContext, PromptRegistry


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
                "reason_summary": {"type": "string"}
            },
            "required": ["question_text", "question_source", "purpose", "confidence", "should_ask", "reason_summary"]
        }
        
        try:
            payload = self.llm.generate_json(
                task=LLMTask.GENERATE_CLARIFICATION,
                system_prompt=self.prompt_registry.system("question_generation"),
                user_prompt=self.prompt_registry.user(
                    PromptContext(
                        stage="generate_clarification",
                        user_id=context.request.user_id,
                        raw_query=context.request.raw_query,
                        rewritten_query=context.rewritten_query,
                        intent=context.intent.value,
                        extra={
                            "missing_fields": missing_fields,
                            "ambiguity_reason": ambiguity_reason,
                        }
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
                            "reason_summary": {"type": "string"}
                        },
                        "required": ["question_text", "question_source", "purpose", "confidence", "should_ask", "reason_summary"]
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
                        raw_query=context.request.raw_query,
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
                "reason_summary": {"type": "string"}
            },
            "required": ["question_text", "question_source", "purpose", "confidence", "should_ask", "reason_summary"]
        }

        try:
            payload = self.llm.generate_json(
                task=LLMTask.GENERATE_REMINDER_SUPPORTING,
                system_prompt=self.prompt_registry.system("question_generation"),
                user_prompt=self.prompt_registry.user(
                    PromptContext(
                        stage="generate_reminder_supporting",
                        user_id=context.request.user_id,
                        raw_query=context.request.raw_query,
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
            )
        except Exception:
            return None
