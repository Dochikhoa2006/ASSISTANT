"""Three-stage LLM orchestration for knowledge mutations only."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import json
from math import isfinite
from typing import Any
import unicodedata

from .action_detection import ActionDetectionResult
from .canonical_retrieval import retrieve_knowledge
from .chat_history import current_chat_history, supporting_question_context
from .config import AssistantConfig
from .contracts import (
    ActionValidationResult,
    ChatRequest,
    Intent,
    KnowledgeAction,
    KnowledgeValidationCandidate,
    PipelineContext,
    ValidatedKnowledgeAction,
)
from .database import AssistantRepository
from .llm import LLMClient, LLMTask
from .prompts import (
    KNOWLEDGE_ACTION_EXTRACTION_SCHEMA,
    KNOWLEDGE_CONTENT_FINALIZATION_SCHEMA,
    PromptContext,
    PromptRegistry,
)
from .retrieval import HybridRetriever
from .retrieval_validation import KnowledgeRetrievalValidationStrategy


_KNOWLEDGE_MUTATION_CONTROL_METADATA_KEYS = frozenset(
    {
        "intent",
        "knowledge_actions",
        "knowledge_action_extraction_response",
        "validated_knowledge_actions",
        "action_authorization",
        "confirmation_approved",
    }
)


def _knowledge_prompt_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Keep metadata as context without letting it compete for action authority."""

    return {
        key: value
        for key, value in metadata.items()
        if key not in _KNOWLEDGE_MUTATION_CONTROL_METADATA_KEYS
    }


@dataclass(frozen=True)
class _KnowledgeActionExtraction:
    """One schema-complete first-stage knowledge extraction."""

    action: str
    text_content: str
    original_text: str
    replacement_text: str
    confidence: float
    missing_fields: tuple[str, ...]
    reason_summary: str

    @classmethod
    def from_payload(cls, payload: Any) -> _KnowledgeActionExtraction | None:
        expected_fields = {
            "action",
            "text_content",
            "original_text",
            "replacement_text",
            "confidence",
            "missing_fields",
            "reason_summary",
        }
        if not isinstance(payload, dict) or set(payload) != expected_fields:
            return None
        if not all(
            isinstance(payload[field], str)
            for field in (
                "action",
                "text_content",
                "original_text",
                "replacement_text",
                "reason_summary",
            )
        ):
            return None
        raw_confidence = payload["confidence"]
        if isinstance(raw_confidence, bool) or not isinstance(
            raw_confidence,
            (int, float),
        ):
            return None
        raw_missing_fields = payload["missing_fields"]
        if not isinstance(raw_missing_fields, list) or not all(
            isinstance(value, str) for value in raw_missing_fields
        ):
            return None
        return cls(
            action=payload["action"].casefold(),
            text_content=payload["text_content"].strip(),
            original_text=payload["original_text"].strip(),
            replacement_text=payload["replacement_text"].strip(),
            confidence=float(raw_confidence),
            missing_fields=tuple(raw_missing_fields),
            reason_summary=payload["reason_summary"].strip(),
        )


class LLMKnowledgeActionDetector:
    """Extract exactly one knowledge action from an intent-selected request."""

    def __init__(
        self,
        *,
        llm: LLMClient,
        prompts: PromptRegistry,
        min_confidence: float = 0.76,
    ) -> None:
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("Knowledge action extraction confidence must be in [0, 1]")
        self.llm = llm
        self.prompts = prompts
        self.min_confidence = min_confidence

    def detect(
        self,
        request: ChatRequest,
        rewritten_query: str,
        intent: Intent,
    ) -> ActionDetectionResult:
        confirmation_replay = bool(
            request.confirmation_token
            and request.metadata.get("confirmation_approved")
        )
        validated_confirmation_actions = list(
            request.metadata.get("validated_knowledge_actions") or []
        )
        extraction_metadata = _knowledge_prompt_metadata(request.metadata)
        chat_history = current_chat_history()
        try:
            payload = self.llm.generate_json(
                task=LLMTask.KNOWLEDGE_ACTION_EXTRACTION,
                system_prompt=self.prompts.system("knowledge_action_extraction"),
                user_prompt=self.prompts.user(
                    PromptContext(
                        stage="knowledge_action_extraction",
                        user_id=request.user_id,
                        rewritten_query=rewritten_query,
                        intent=intent.value,
                        metadata=extraction_metadata,
                        platform_context=request.platform_context,
                        chat_history=chat_history,
                        extra={
                            "allowed_actions": ["add", "delete", "modify"],
                            "cardinality": "exactly_one",
                            "action_content_contract": {
                                "add": {
                                    "required_non_empty": ["text_content"],
                                    "required_empty": [
                                        "original_text",
                                        "replacement_text",
                                    ],
                                },
                                "delete": {
                                    "required_non_empty": ["text_content"],
                                    "required_empty": [
                                        "original_text",
                                        "replacement_text",
                                    ],
                                },
                                "modify": {
                                    "required_non_empty": [
                                        "original_text",
                                        "replacement_text",
                                    ],
                                    "required_empty": ["text_content"],
                                },
                            },
                            "context_policy": {
                                "current_turn_defines_action": True,
                                "history_resolves_existing_fact_references": True,
                                "history_cannot_invent_add_or_replacement_content": True,
                                "platform_context_is_supporting_context_only": True,
                                "non_action_metadata_is_supporting_context_only": True,
                            },
                            "supporting_question_context": supporting_question_context(
                                chat_history
                            ),
                            "confirmation_replay": confirmation_replay,
                            # This is prompt context only. Trust and consistency
                            # are evaluated after the extraction call so no
                            # semantic guard can precede the first model stage.
                            "trusted_confirmation_action_context": (
                                validated_confirmation_actions
                                if confirmation_replay
                                else []
                            ),
                        },
                    )
                ),
                schema=KNOWLEDGE_ACTION_EXTRACTION_SCHEMA,
            )
        except Exception:
            return self._failed(
                intent,
                "knowledge_action_extraction_failed",
                ["action"],
            )

        extraction = _KnowledgeActionExtraction.from_payload(payload)
        if extraction is None:
            return self._failed(
                intent,
                "invalid_knowledge_action_extraction",
                ["action"],
            )
        action_name = extraction.action
        text_content = extraction.text_content
        original_text = extraction.original_text
        replacement_text = extraction.replacement_text
        confidence = extraction.confidence
        missing_fields = list(extraction.missing_fields)

        if intent is not Intent.KNOWLEDGE_FACTS:
            return self._failed(
                intent,
                "knowledge_extraction_called_for_non_knowledge_intent",
                ["knowledge_intent"],
            )

        trusted_confirmation_action = self._trusted_confirmation_action(request)
        if confirmation_replay and trusted_confirmation_action is None:
            return self._failed(
                intent,
                "invalid_knowledge_confirmation_context",
                ["confirmed_action"],
            )

        normalized_missing = self._normalize_missing_fields(
            action_name,
            missing_fields,
        )

        if (
            action_name not in {"add", "delete", "modify"}
            or not isfinite(confidence)
            or not 0.0 <= confidence <= 1.0
            or confidence < self.min_confidence
            or missing_fields
        ):
            return ActionDetectionResult(
                intent=intent,
                confidence=0.0,
                missing_fields=normalized_missing or ["action"],
                risk_flags=["low_confidence_or_incomplete_knowledge_extraction"],
            )

        if action_name in {"add", "delete"}:
            if not text_content or original_text or replacement_text:
                return self._failed(
                    intent,
                    f"invalid_{action_name}_content_contract",
                    ["text" if action_name == "add" else "target_description"],
                )
            action_payload: dict[str, Any] = {
                "action": action_name,
                "confidence": confidence,
            }
            if action_name == "add":
                action_payload["text"] = text_content
            else:
                action_payload["target_description"] = text_content
        else:
            if text_content or not original_text or not replacement_text:
                missing = []
                if not original_text:
                    missing.append("target_description")
                if not replacement_text:
                    missing.append("replacement_text")
                return self._failed(
                    intent,
                    "invalid_modify_content_contract",
                    missing or ["target_description", "replacement_text"],
                )
            action_payload = {
                "action": action_name,
                "target_description": original_text,
                "replacement_text": replacement_text,
                "confidence": confidence,
            }

        current_turn_sources = [rewritten_query]
        trusted_confirmation_source = (
            json.dumps(
                trusted_confirmation_action,
                default=str,
                ensure_ascii=False,
            )
            if trusted_confirmation_action
            else ""
        )
        if trusted_confirmation_source:
            # A verified confirmation message normally contains only "confirm".
            # Its immutable action content was already validated and hydrated by
            # the request lifecycle, so it is the only additional grounding
            # source permitted for this replay extraction.
            current_turn_sources.append(trusted_confirmation_source)
        history_source = json.dumps(
            chat_history,
            default=str,
            ensure_ascii=False,
        )
        grounded_values = (
            [("text", text_content, current_turn_sources)]
            if action_name == "add"
            else [
                (
                    "target_description",
                    text_content,
                    current_turn_sources + [history_source],
                )
            ]
            if action_name == "delete"
            else [
                (
                    "target_description",
                    original_text,
                    current_turn_sources + [history_source],
                ),
                (
                    "replacement_text",
                    replacement_text,
                    current_turn_sources,
                ),
            ]
        )
        ungrounded_fields = [
            field_name
            for field_name, value, sources in grounded_values
            if not self._is_grounded(value, sources)
        ]
        if ungrounded_fields:
            return self._failed(
                intent,
                "knowledge_content_not_grounded_in_request_context",
                ungrounded_fields,
            )

        if trusted_confirmation_action and not self._matches_confirmation_action(
            action_name=action_name,
            text_content=text_content,
            original_text=original_text,
            replacement_text=replacement_text,
            trusted_action=trusted_confirmation_action,
        ):
            return self._failed(
                intent,
                "knowledge_confirmation_extraction_mismatch",
                ["confirmed_action"],
            )

        trusted_topic_title = request.metadata.get("topic_title")
        if trusted_topic_title:
            action_payload["topic_title"] = str(trusted_topic_title)

        return ActionDetectionResult(
            intent=intent,
            confidence=confidence,
            metadata={
                "knowledge_actions": [action_payload],
                "knowledge_action_extraction_response": {
                    "action": action_name,
                    "text_content": text_content,
                    "original_text": original_text,
                    "replacement_text": replacement_text,
                    "confidence": confidence,
                    "missing_fields": list(extraction.missing_fields),
                    "reason_summary": extraction.reason_summary,
                },
                "action_authorization": {
                    "intent": intent.value,
                    "action": action_name,
                    "reason_summary": "selected_by_knowledge_action_extraction_llm",
                    "source": "knowledge_action_extraction_llm",
                },
            },
        )

    @staticmethod
    def _trusted_confirmation_action(request: ChatRequest) -> dict[str, Any] | None:
        if not (
            request.confirmation_token
            and request.metadata.get("confirmation_approved")
        ):
            return None
        actions = list(request.metadata.get("validated_knowledge_actions") or [])
        if len(actions) != 1 or not isinstance(actions[0], dict):
            return None
        action = dict(actions[0])
        action_name = str(action.get("action") or "").casefold()
        authorization = request.metadata.get("action_authorization") or {}
        if (
            action_name not in {"add", "delete", "modify"}
            or authorization.get("intent") != Intent.KNOWLEDGE_FACTS.value
            or str(authorization.get("action") or "").casefold() != action_name
        ):
            return None
        return action

    @classmethod
    def _matches_confirmation_action(
        cls,
        *,
        action_name: str,
        text_content: str,
        original_text: str,
        replacement_text: str,
        trusted_action: dict[str, Any],
    ) -> bool:
        trusted_name = str(trusted_action.get("action") or "").casefold()
        if action_name != trusted_name:
            return False

        if action_name == "add":
            expected_text = str(
                trusted_action.get("knowledge_text")
                or trusted_action.get("new_text")
                or ""
            )
            extracted_values = (text_content, original_text, replacement_text)
            expected_values = (expected_text, "", "")
        elif action_name == "delete":
            expected_target = str(trusted_action.get("target_description") or "")
            extracted_values = (text_content, original_text, replacement_text)
            expected_values = (expected_target, "", "")
        else:
            expected_target = str(trusted_action.get("target_description") or "")
            expected_replacement = str(
                trusted_action.get("replacement_text")
                or trusted_action.get("new_text")
                or ""
            )
            return (
                not text_content
                and cls._grounding_text(original_text)
                == cls._grounding_text(expected_target)
                and bool(cls._grounding_text(replacement_text))
                and cls._grounding_text(replacement_text)
                in cls._grounding_text(expected_replacement)
            )

        return all(
            cls._grounding_text(extracted) == cls._grounding_text(expected)
            for extracted, expected in zip(extracted_values, expected_values)
        )

    @staticmethod
    def _normalize_missing_fields(
        action_name: str,
        fields: list[str],
    ) -> list[str]:
        normalized: list[str] = []
        for field in fields:
            key = field.casefold()
            if action_name == "add" and key in {
                "text_content",
                "knowledge_content",
                "content",
            }:
                key = "text"
            elif action_name == "delete" and key in {
                "text_content",
                "original_text",
                "knowledge_content",
                "content",
            }:
                key = "target_description"
            elif action_name == "modify" and key in {
                "original_text",
                "text_content",
            }:
                key = "target_description"
            normalized.append(key)
        return list(dict.fromkeys(normalized))

    @staticmethod
    def _grounding_text(value: str) -> str:
        return " ".join(
            unicodedata.normalize("NFKC", value).casefold().split()
        ).strip(" \t\r\n\"'`.,;:!?")

    @classmethod
    def _is_grounded(cls, value: str, sources: list[str]) -> bool:
        needle = cls._grounding_text(value)
        return bool(needle) and any(
            needle in cls._grounding_text(source) for source in sources
        )

    @staticmethod
    def _failed(
        intent: Intent,
        reason: str,
        missing_fields: list[str],
    ) -> ActionDetectionResult:
        return ActionDetectionResult(
            intent=intent,
            confidence=0.0,
            missing_fields=missing_fields,
            risk_flags=[reason],
        )


@dataclass(frozen=True)
class KnowledgeFinalizationResult:
    final_content: str
    confidence: float
    reason_summary: str


class KnowledgeContentFinalizationStrategy:
    """Construct one final chunk only after high-confidence validation."""

    def __init__(
        self,
        *,
        llm: LLMClient,
        prompts: PromptRegistry,
        min_confidence: float,
    ) -> None:
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("Knowledge finalization confidence must be in [0, 1]")
        self.llm = llm
        self.prompts = prompts
        self.min_confidence = min_confidence

    def finalize(
        self,
        *,
        context: PipelineContext,
        action: KnowledgeAction,
        action_payload: dict[str, Any],
        all_candidates: list[KnowledgeValidationCandidate],
        selected_candidates: list[KnowledgeValidationCandidate],
        validation_result: Any,
    ) -> KnowledgeFinalizationResult | None:
        # LLM3 has exactly one responsibility: rewrite one validated existing
        # chunk for MODIFY. ADD and DELETE are structurally unable to call it.
        if action is not KnowledgeAction.MODIFY or len(selected_candidates) != 1:
            return None
        selected_candidate = selected_candidates[0]
        if (
            validation_result.validation_result is not ActionValidationResult.EXECUTE
            or tuple(validation_result.selected_candidate_keys)
            != (selected_candidate.candidate_key,)
            or sum(
                candidate.candidate_key == selected_candidate.candidate_key
                for candidate in all_candidates
            )
            != 1
        ):
            return None
        assessment = next(
            (
                item
                for item in validation_result.candidate_assessments
                if item.candidate_key == selected_candidate.candidate_key
            ),
            None,
        )
        matched_text = str(getattr(assessment, "matched_text", "") or "")
        replacement = str(action_payload.get("replacement_text") or "").strip()
        candidate_text = selected_candidate.text
        if (
            assessment is None
            or not matched_text
            or not replacement
            or candidate_text.count(matched_text) != 1
        ):
            return None

        try:
            payload = self.llm.generate_json(
                task=LLMTask.KNOWLEDGE_CONTENT_FINALIZATION,
                system_prompt=self.prompts.system("knowledge_content_finalization"),
                user_prompt=self.prompts.user(
                    PromptContext(
                        stage="knowledge_content_finalization",
                        user_id=context.request.user_id,
                        rewritten_query=context.rewritten_query,
                        intent=Intent.KNOWLEDGE_FACTS.value,
                        metadata=_knowledge_prompt_metadata(
                            context.request.metadata
                        ),
                        platform_context=context.request.platform_context,
                        chat_history=context.chat_history,
                        extra={
                            "operation": KnowledgeAction.MODIFY.value,
                            "extracted_action_content": {
                                "text_content": None,
                                "original_text": action_payload.get(
                                    "target_description"
                                ),
                                "replacement_text": replacement,
                            },
                            "validated_candidate_context": [
                                {
                                    "candidate_key": selected_candidate.candidate_key,
                                    "text": selected_candidate.text,
                                    "source_title": selected_candidate.source_title,
                                    "selected": True,
                                    "matched_text": matched_text,
                                }
                            ],
                            "validation_result": {
                                "operation": validation_result.operation,
                                "decision": "PASS",
                                "selected_candidate_keys": list(
                                    validation_result.selected_candidate_keys
                                ),
                                "confidence": validation_result.confidence,
                                "clarification_question": "",
                                "reason_summary": validation_result.reason_summary,
                                "candidate_assessments": [
                                    asdict(item)
                                    for item in validation_result.candidate_assessments
                                ],
                            },
                            "supporting_question_context": supporting_question_context(
                                context.chat_history
                            ),
                        },
                    )
                ),
                schema=KNOWLEDGE_CONTENT_FINALIZATION_SCHEMA,
            )
            final_content = str(payload["final_content"]).strip()
            confidence = float(payload["confidence"])
            reason_summary = str(payload["reason_summary"])
        except Exception:
            return None

        if (
            not final_content
            or not isfinite(confidence)
            or not 0.0 <= confidence <= 1.0
            or confidence < self.min_confidence
        ):
            return None

        expected_content = candidate_text.replace(
            matched_text,
            replacement,
            1,
        )
        if self._compact(final_content) != self._compact(expected_content):
            return None

        return KnowledgeFinalizationResult(
            final_content=final_content,
            confidence=confidence,
            reason_summary=reason_summary,
        )

    @staticmethod
    def _compact(value: str) -> str:
        return " ".join(value.split())


class KnowledgeMutationPipeline:
    """Retrieve, validate, and finalize one extracted knowledge mutation."""

    def __init__(
        self,
        *,
        retriever: HybridRetriever,
        config: AssistantConfig,
        validator: KnowledgeRetrievalValidationStrategy,
        finalizer: KnowledgeContentFinalizationStrategy,
    ) -> None:
        if not config.retrieval_validation.knowledge_llm_validation_enabled:
            raise ValueError(
                "Knowledge mutation validation must remain LLM-enabled"
            )
        self.retriever = retriever
        self.config = config
        self.validator = validator
        self.finalizer = finalizer

    def build_action(
        self,
        *,
        context: PipelineContext,
        action_payload: dict[str, Any],
        repository: AssistantRepository,
    ) -> ValidatedKnowledgeAction:
        action_name = str(action_payload.get("action") or "").casefold()
        if action_name not in {item.value for item in KnowledgeAction}:
            return self._clarification_action(
                KnowledgeAction.ADD,
                action_payload,
                "Unsupported extracted knowledge action.",
            )
        action = KnowledgeAction(action_name)
        proposed_content = str(action_payload.get("text") or "").strip()
        target_description = str(
            action_payload.get("target_description") or ""
        ).strip()
        replacement_content = str(
            action_payload.get("replacement_text") or ""
        ).strip()

        retrieval_query = proposed_content if action is KnowledgeAction.ADD else target_description
        if not retrieval_query or (
            action is KnowledgeAction.MODIFY and not replacement_content
        ):
            return self._clarification_action(
                action,
                action_payload,
                "The extracted action is missing required content.",
            )

        try:
            results = retrieve_knowledge(
                retriever=self.retriever,
                repository=repository,
                user_id=context.request.user_id,
                query=retrieval_query,
                # This bypass is deliberately scoped to this mutation pipeline.
                # Cross-encoder scores remain evidence for the validator, but no
                # score may hide a small matching fact inside a longer chunk.
                enforce_min_score=False,
            )
        except Exception:
            return self._technical_failure_action(
                action,
                action_payload,
                "Knowledge mutation retrieval failed safely.",
            )

        try:
            chunk_ids = [result.entity_id for result in results]
            sql_chunks = repository.get_knowledge_chunks_by_ids(
                context.request.user_id,
                chunk_ids,
                include_deleted=False,
            )
            chunk_map = {str(chunk["chunk_id"]): chunk for chunk in sql_chunks}
            active_results = [
                result for result in results if result.entity_id in chunk_map
            ]
            candidates = [
                KnowledgeValidationCandidate(
                    candidate_key=result.entity_id,
                    knowledge_chunk_id=result.entity_id,
                    knowledge_topic_id=str(
                        chunk_map[result.entity_id].get("knowledge_topic_id") or ""
                    ),
                    text=str(
                        chunk_map[result.entity_id].get("raw_text")
                        or chunk_map[result.entity_id].get("normalized_text")
                        or ""
                    ),
                    source_title=result.payload.get("source_title"),
                    retrieval_score=float(result.confidence),
                    rerank_score=float(result.rerank_score),
                    is_deleted=bool(
                        chunk_map[result.entity_id].get("is_deleted", False)
                    ),
                    user_id=context.request.user_id,
                )
                for result in active_results
            ]
        except Exception:
            return self._technical_failure_action(
                action,
                action_payload,
                "Knowledge candidate hydration failed safely.",
            )
        candidate_map = {candidate.candidate_key: candidate for candidate in candidates}
        first_model_response = context.request.metadata.get(
            "knowledge_action_extraction_response"
        )

        validation = self.validator.validate(
            operation=action.value,
            # The validator's retained compatibility parameters are
            # intentionally blank. LLM2 receives only first_model_response and
            # knowledge_retrieval in its isolated prompt.
            user_query="",
            rewritten_query="",
            target_description="",
            candidates=candidates,
            first_model_response=(
                dict(first_model_response)
                if isinstance(first_model_response, dict)
                else None
            ),
        )
        if validation is None:
            return self._technical_failure_action(
                action,
                action_payload,
                "Knowledge validation model did not return a safe decision.",
            )
        if validation.hitl_reason == "internal_validation_failure":
            return self._technical_failure_action(
                action,
                action_payload,
                "Knowledge validation model failed safely.",
            )
        if validation.validation_result is not ActionValidationResult.EXECUTE:
            if (
                validation.hitl_reason != "knowledge_validation_fail"
                or not (validation.clarification_question or "").strip()
            ):
                return self._technical_failure_action(
                    action,
                    action_payload,
                    "Knowledge validation returned an unusable FAIL decision.",
                )
            return ValidatedKnowledgeAction(
                action=action,
                validation_result=ActionValidationResult.CLARIFY_MISSING_FIELDS,
                target_chunk_ids=(),
                knowledge_text=proposed_content or None,
                replacement_text=replacement_content or None,
                new_text=(proposed_content or replacement_content or None),
                target_status="active",
                topic_title=action_payload.get("topic_title"),
                target_description=target_description or None,
                confidence=validation.confidence,
                reason_summary=validation.reason_summary,
                requires_hitl=validation.requires_hitl,
                factuality_concern=False,
                hitl_reason="knowledge_validation_fail",
                clarification_question=validation.clarification_question,
            )

        selected_candidates = [
            candidate_map[key]
            for key in validation.selected_candidate_keys
            if key in candidate_map
        ]
        if action is KnowledgeAction.ADD:
            if selected_candidates:
                return self._technical_failure_action(
                    action,
                    action_payload,
                    "Add validation unexpectedly selected an existing target.",
                )
        elif len(selected_candidates) != 1:
            return self._technical_failure_action(
                action,
                action_payload,
                "Destructive validation did not select exactly one SQL target.",
            )

        # Confirmation gate: destructive MODIFY and DELETE actions require
        # explicit user approval before LLM3 may rewrite content or before
        # the SQL transaction executes. ADD is always pre-authorized by LLM2
        # EXECUTE and must never be blocked here.
        if action in {KnowledgeAction.MODIFY, KnowledgeAction.DELETE}:
            if not (context.request.metadata.get("confirmation_approved")):
                selected = selected_candidates[0]
                target_id = selected.candidate_key
                # Inline _expires_at because branches.py cannot be imported here.
                _confirmation_expires_at = (
                    datetime.now(timezone.utc)
                    + timedelta(minutes=self.config.confirmation_expiry_minutes)
                ).isoformat()
                confirmation = repository.create_pending_confirmation(
                    user_id=context.request.user_id,
                    action_type="knowledge_mutation",
                    target_entity_type="knowledge_chunk",
                    target_entity_id=target_id,
                    proposed_action={
                        "domain": "knowledge",
                        "action": action.value,
                        "target_chunk_id": target_id,
                        "target_description": target_description or None,
                        "replacement_text": replacement_content or None,
                        "action_authorization": {
                            "intent": "knowledge_facts",
                            "action": action.value,
                            "source": "llm2_validated",
                            "reason_summary": validation.reason_summary,
                        },
                        "validated_knowledge_actions": [
                            {
                                "action": action.value,
                                "target_chunk_ids": [target_id],
                                "target_description": target_description or None,
                                "replacement_text": replacement_content or None,
                                "confidence": float(validation.confidence),
                                "reason_summary": validation.reason_summary,
                            }
                        ],
                    },
                    target_snapshot={
                        "chunk_id": target_id,
                        "text": selected.text,
                        "version": chunk_map[target_id].get("version"),
                    },
                    expires_at=_confirmation_expires_at,
                )
                return ValidatedKnowledgeAction(
                    action=action,
                    validation_result=ActionValidationResult.CLARIFY_MISSING_FIELDS,
                    target_chunk_ids=(target_id,),
                    confidence=float(validation.confidence),
                    reason_summary="Awaiting user confirmation before applying destructive mutation.",
                    requires_hitl=True,
                    factuality_concern=False,
                    hitl_reason="knowledge_pending_confirmation",
                    clarification_question=None,
                    pending_confirmation_token=confirmation.get("confirmation_token"),
                )

        finalized: KnowledgeFinalizationResult | None = None
        if action is KnowledgeAction.MODIFY:
            finalized = self.finalizer.finalize(
                context=context,
                action=action,
                action_payload=action_payload,
                all_candidates=candidates,
                selected_candidates=selected_candidates,
                validation_result=validation,
            )
            if finalized is None:
                return self._technical_failure_action(
                    action,
                    action_payload,
                    "Knowledge content finalization failed safely.",
                )

        target_ids = tuple(candidate.candidate_key for candidate in selected_candidates)
        observed_versions = {
            candidate_id: int(chunk_map[candidate_id]["version"])
            for candidate_id in target_ids
            if chunk_map[candidate_id].get("version") is not None
        }
        observed_is_deleted = {
            candidate_id: bool(chunk_map[candidate_id].get("is_deleted", False))
            for candidate_id in target_ids
        }
        matched_fields = tuple(
            dict.fromkeys(
                field
                for assessment in validation.candidate_assessments
                if assessment.candidate_key in target_ids
                for field in assessment.matched_fields
            )
        )
        final_text = (
            proposed_content
            if action is KnowledgeAction.ADD
            else finalized.final_content
            if finalized is not None
            else None
        )
        confidence_values = [
            float(action_payload.get("confidence", 1.0)),
            validation.confidence,
        ]
        reason_parts = [validation.reason_summary]
        if finalized is not None:
            confidence_values.append(finalized.confidence)
            reason_parts.append(finalized.reason_summary)
        return ValidatedKnowledgeAction(
            action=action,
            validation_result=ActionValidationResult.EXECUTE,
            target_chunk_ids=target_ids,
            target_topic_ids=tuple(
                dict.fromkeys(
                    candidate.knowledge_topic_id
                    for candidate in selected_candidates
                    if candidate.knowledge_topic_id
                )
            ),
            observed_versions=observed_versions,
            observed_is_deleted=observed_is_deleted,
            knowledge_text=final_text if action is KnowledgeAction.ADD else None,
            replacement_text=final_text if action is KnowledgeAction.MODIFY else None,
            new_text=(
                final_text
                if action in {KnowledgeAction.ADD, KnowledgeAction.MODIFY}
                else None
            ),
            target_status="active",
            topic_title=action_payload.get("topic_title"),
            target_description=target_description or None,
            confidence=min(confidence_values),
            matched_fields=matched_fields,
            reason_summary=" ".join(reason_parts).strip(),
            requires_hitl=False,
            factuality_concern=False,
            hitl_reason=None,
        )

    @staticmethod
    def _clarification_action(
        action: KnowledgeAction,
        action_payload: dict[str, Any],
        reason: str,
    ) -> ValidatedKnowledgeAction:
        return ValidatedKnowledgeAction(
            action=action,
            validation_result=ActionValidationResult.CLARIFY_MISSING_FIELDS,
            knowledge_text=action_payload.get("text"),
            replacement_text=action_payload.get("replacement_text"),
            new_text=action_payload.get("text") or action_payload.get("replacement_text"),
            target_description=action_payload.get("target_description"),
            confidence=0.0,
            reason_summary=reason,
            requires_hitl=True,
            factuality_concern=False,
            hitl_reason=None,
        )

    @staticmethod
    def _technical_failure_action(
        action: KnowledgeAction,
        action_payload: dict[str, Any],
        reason: str,
    ) -> ValidatedKnowledgeAction:
        return ValidatedKnowledgeAction(
            action=action,
            validation_result=ActionValidationResult.REJECT_UNSAFE_TRANSITION,
            knowledge_text=action_payload.get("text"),
            replacement_text=action_payload.get("replacement_text"),
            new_text=action_payload.get("text") or action_payload.get("replacement_text"),
            target_description=action_payload.get("target_description"),
            confidence=0.0,
            reason_summary=reason,
            requires_hitl=False,
            factuality_concern=False,
            hitl_reason="internal_pipeline_failure",
        )
