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

        schema = self._schema(intent)
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
        result = self._parse_payload(payload, selected_intent=intent)
        if self._needs_knowledge_recovery(result):
            recovered = self._recover_knowledge_action(request, rewritten_query)
            if recovered is not None:
                result = recovered
        
        if self.risky_action_validation_enabled and result.metadata:
            result = self._validate_risky_actions(request, rewritten_query, result)
            
        return result

    def _parse_payload(self, payload: dict[str, Any], *, selected_intent: Intent) -> ActionDetectionResult:
        # Action extraction is downstream of intent routing. The extractor may
        # describe fields for that branch, but it must never reroute the request.
        intent = selected_intent
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

    @staticmethod
    def _needs_knowledge_recovery(result: ActionDetectionResult) -> bool:
        return bool(
            result.intent is Intent.KNOWLEDGE_FACTS
            and not result.metadata.get("knowledge_actions")
        )

    def _recover_knowledge_action(
        self, request: ChatRequest, rewritten_query: str
    ) -> ActionDetectionResult | None:
        operation_schema = {
            "type": "object",
            "properties": {
                "operation": {"type": "string", "enum": ["add", "modify", "delete", "lookup"]},
            },
            "required": ["operation"],
        }
        try:
            operation_payload = self.llm.generate_json(
                task=LLMTask.ACTION_EXTRACTION,
                system_prompt=self.prompt_registry.system("knowledge_operation_recovery"),
                user_prompt=self.prompt_registry.user(
                    PromptContext(
                        stage="action_detection",
                        user_id=request.user_id,
                        raw_query=request.raw_query,
                        rewritten_query=rewritten_query,
                        intent=Intent.KNOWLEDGE_FACTS.value,
                        metadata=request.metadata,
                        platform_context=request.platform_context,
                        extra={"recovery_reason": "initial knowledge extraction returned no executable action"},
                    )
                ),
                schema=operation_schema,
            )
            validate_json_schema(operation_payload, operation_schema)
        except Exception:
            return None

        operation = operation_payload.get("operation")
        if operation == "lookup":
            return ActionDetectionResult(
                intent=Intent.KNOWLEDGE_FACTS,
                confidence=1.0,
                metadata={"knowledge_lookup": True},
            )
        if operation != "add":
            return None

        content_schema = {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        }
        try:
            content_payload = self.llm.generate_json(
                task=LLMTask.ACTION_EXTRACTION,
                system_prompt=self.prompt_registry.system("knowledge_add_content_recovery"),
                user_prompt=self.prompt_registry.user(
                    PromptContext(
                        stage="action_detection",
                        user_id=request.user_id,
                        raw_query=request.raw_query,
                        rewritten_query=rewritten_query,
                        intent=Intent.KNOWLEDGE_FACTS.value,
                        metadata=request.metadata,
                        platform_context=request.platform_context,
                    )
                ),
                schema=content_schema,
            )
            validate_json_schema(content_payload, content_schema)
        except Exception:
            return None
        text = str(content_payload.get("text") or "").strip()
        if not text:
            return None
        return ActionDetectionResult(
            intent=Intent.KNOWLEDGE_FACTS,
            confidence=1.0,
            metadata={"knowledge_actions": [{"action": "add", "text": text}]},
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
                            "missing_fields": {"type": "array", "items": {"type": "string"}},
                            "requires_clarification": {"type": "boolean"},
                            "confidence": {"type": "number"}
                        },
                        "required": ["action_index", "approved", "risk_level", "missing_fields", "requires_clarification", "confidence"]
                    }
                }
            },
            "required": ["results"]
        }

    def _schema(self, selected_intent: Intent) -> dict[str, Any]:
        action_key = "knowledge_actions" if selected_intent is Intent.KNOWLEDGE_FACTS else "reminder_actions"
        action_schema = self._knowledge_action_schema() if selected_intent is Intent.KNOWLEDGE_FACTS else self._reminder_action_schema()
        return {
            "type": "object",
            "properties": {
                "confidence": {"type": "number"},
                action_key: {"type": "array", "items": action_schema},
                "missing_fields": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "risk_flags": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": [
                "confidence",
                action_key,
                "missing_fields",
                "risk_flags",
            ],
        }

    @staticmethod
    def _knowledge_action_schema() -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["add", "delete", "modify"]},
                "text": {"type": "string"},
                "target_description": {"type": "string"},
                "replacement_text": {"type": "string"},
            },
            "required": ["action"],
        }

    @staticmethod
    def _reminder_action_schema() -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["add", "delete", "modify", "turn_on", "turn_off"]},
                "subject": {"type": "string"},
                "reminder_time": {"type": "string"},
                "target_description": {"type": "string"},
                "new_reminder_time": {"type": "string"},
                "new_subject": {"type": "string"},
            },
            "required": ["action"],
        }
