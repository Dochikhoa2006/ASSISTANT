"""Schema-validated action detection before branch execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .contracts import ChatRequest, Intent
from .llm import LLMClient, LLMTask, validate_json_schema
from .prompts import DEFAULT_PROMPT_REGISTRY, PromptContext, PromptRegistry
from .settings import PromptPolicySettings


@dataclass(frozen=True)
class ActionDetectionResult:
    intent: Intent
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)
    missing_fields: list[str] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)

    @property
    def requires_clarification(self) -> bool:
        return self.confidence < 0.5 or bool(self.missing_fields)


class ActionDetector(Protocol):
    def detect(self, request: ChatRequest, rewritten_query: str, intent: Intent) -> ActionDetectionResult:
        ...


class NoOpActionDetector:
    def detect(self, request: ChatRequest, rewritten_query: str, intent: Intent) -> ActionDetectionResult:
        return ActionDetectionResult(intent=intent, confidence=1.0, metadata={})


@dataclass
class LLMActionDetector:
    llm: LLMClient
    prompt_registry: PromptRegistry = field(default_factory=lambda: DEFAULT_PROMPT_REGISTRY)
    min_confidence: float = PromptPolicySettings.action_min_confidence
    risky_action_validation_enabled: bool = field(
        default_factory=lambda: PromptPolicySettings.risky_action_validation_enabled
    )
    risky_action_operations: tuple[str, ...] = field(
        default_factory=lambda: PromptPolicySettings.risky_action_operations
    )
    risky_action_confidence_threshold: float = field(
        default_factory=lambda: PromptPolicySettings.risky_action_confidence_threshold
    )

    def detect(self, request: ChatRequest, rewritten_query: str, intent: Intent) -> ActionDetectionResult:
        if request.metadata.get("knowledge_actions") or request.metadata.get("reminder_actions"):
            return ActionDetectionResult(intent=intent, confidence=1.0, metadata={})
        if intent not in {Intent.KNOWLEDGE_FACTS, Intent.REMINDER}:
            return ActionDetectionResult(intent=intent, confidence=1.0, metadata={})

        schema = self._schema()
        try:
            payload = self.llm.generate_json(
                task=LLMTask.ACTION_EXTRACTION,
                system_prompt=self.prompt_registry.system("action_detection"),
                user_prompt=self.prompt_registry.user(
                    PromptContext(
                        stage="action_detection",
                        user_id=request.user_id,
                        raw_query=request.raw_query,
                        rewritten_query=rewritten_query,
                        intent=intent.value,
                        metadata=request.metadata,
                        platform_context=request.platform_context,
                    )
                ),
                schema=schema,
            )
            validate_json_schema(payload, schema)
        except Exception:
            return ActionDetectionResult(
                intent=intent,
                confidence=0.0,
                metadata={},
                missing_fields=["llm_action_detection_unavailable"],
                risk_flags=["safe_fallback_to_clarification"],
            )
        result = self._parse_payload(payload, fallback_intent=intent)
        
        if self.risky_action_validation_enabled and result.metadata:
            result = self._validate_risky_actions(request, rewritten_query, result)
            
        return result

    def _parse_payload(self, payload: dict[str, Any], *, fallback_intent: Intent) -> ActionDetectionResult:
        intent_value = str(payload.get("intent") or fallback_intent.value)
        intent = Intent(intent_value) if intent_value in {item.value for item in Intent} else fallback_intent
        confidence = float(payload.get("confidence", 0.0))
        metadata = {}
        if intent is Intent.KNOWLEDGE_FACTS:
            metadata["knowledge_actions"] = list(payload.get("knowledge_actions") or [])
        if intent is Intent.REMINDER:
            metadata["reminder_actions"] = list(payload.get("reminder_actions") or [])
        missing_fields = [str(item) for item in payload.get("missing_fields") or []]
        risk_flags = [str(item) for item in payload.get("risk_flags") or []]
        if confidence < self.min_confidence:
            missing_fields = missing_fields or ["low_confidence_action_detection"]
        return ActionDetectionResult(
            intent=intent,
            confidence=confidence,
            metadata=metadata,
            missing_fields=missing_fields,
            risk_flags=risk_flags,
        )

    def _validate_risky_actions(self, request: ChatRequest, rewritten_query: str, result: ActionDetectionResult) -> ActionDetectionResult:
        risky_actions = []
        for i, action in enumerate(result.metadata.get("knowledge_actions", [])):
            if action.get("action") in self.risky_action_operations:
                risky_actions.append({"type": "knowledge", "index": i, "action": action})
        for i, action in enumerate(result.metadata.get("reminder_actions", [])):
            if action.get("action") in self.risky_action_operations:
                risky_actions.append({"type": "reminder", "index": i, "action": action})
                
        if not risky_actions:
            return result
            
        schema = self._risky_schema()
        try:
            payload = self.llm.generate_json(
                task=LLMTask.RISKY_ACTION,
                system_prompt=self.prompt_registry.system("risky_action_validation"),
                user_prompt=self.prompt_registry.user(
                    PromptContext(
                        stage="risky_action_validation",
                        user_id=request.user_id,
                        raw_query=request.raw_query,
                        rewritten_query=rewritten_query,
                        intent=result.intent.value,
                        metadata={"risky_actions": [ra["action"] for ra in risky_actions]},
                        platform_context=request.platform_context,
                    )
                ),
                schema=schema,
            )
            validate_json_schema(payload, schema)
        except Exception:
            result.missing_fields.append("risky_action_validation_failed")
            return result
            
        validation_results = payload.get("results", [])
        
        if len(validation_results) != len(risky_actions):
            result.missing_fields.append("risky_validation_count_mismatch")
            return result
            
        for ra, val_res in zip(risky_actions, validation_results):
            action = ra["action"]
            
            action["risk_approved"] = val_res.get("approved", False)
            action["risk_level"] = val_res.get("risk_level", "high")
            
            if val_res.get("requires_clarification") or val_res.get("confidence", 0.0) < self.risky_action_confidence_threshold:
                result.missing_fields.append("risky_action_requires_clarification")
            
        return result

    def _risky_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "action_index": {"type": "integer"},
                            "approved": {"type": "boolean"},
                            "risk_level": {"type": "string"},
                            "reason_summary": {"type": "string"},
                            "missing_fields": {"type": "array", "items": {"type": "string"}},
                            "requires_clarification": {"type": "boolean"},
                            "confidence": {"type": "number"}
                        },
                        "required": ["action_index", "approved", "risk_level", "reason_summary", "missing_fields", "requires_clarification", "confidence"]
                    }
                }
            },
            "required": ["results"]
        }

    def _schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "intent": {
                    "type": "string",
                    "enum": [
                        "clarification",
                        "general_response",
                        "knowledge_facts",
                        "reminder",
                    ],
                },
                "confidence": {"type": "number"},
                "knowledge_actions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string"},
                            "topic_title": {"type": "string"},
                            "text": {"type": "string"},
                            "target_description": {"type": "string"},
                            "target_entities": {"type": "array", "items": {"type": "string"}},
                            "replacement_text": {"type": "string"},
                            "raw_user_instruction": {"type": "string"},
                            "confidence": {"type": "number"},
                            "missing_fields": {"type": "array", "items": {"type": "string"}}
                        },
                        "required": ["action", "raw_user_instruction"]
                    },
                },
                "reminder_actions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string"},
                            "subject": {"type": "string"},
                            "reminder_time": {"type": "string"},
                            "raw_reminder": {"type": "string"},
                            "target_description": {"type": "string"},
                            "target_entities": {"type": "array", "items": {"type": "string"}},
                            "new_reminder_time": {"type": "string"},
                            "new_subject": {"type": "string"},
                            "raw_user_instruction": {"type": "string"},
                            "confidence": {"type": "number"},
                            "missing_fields": {"type": "array", "items": {"type": "string"}}
                        },
                        "required": ["action", "raw_user_instruction"]
                    },
                },
                "missing_fields": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "risk_flags": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "normalized_entities": {
                    "type": "object",
                },
                "reason_summary": {"type": "string"},
            },
            "required": [
                "intent",
                "confidence",
                "knowledge_actions",
                "reminder_actions",
                "missing_fields",
                "risk_flags",
                "normalized_entities",
                "reason_summary",
            ],
        }
