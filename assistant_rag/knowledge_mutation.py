"""Three-stage LLM orchestration for knowledge mutations only."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from math import isfinite
from typing import Any
import unicodedata

from .action_detection import (
    ActionDetectionResult,
    DeterministicActionDetector,
    classify_action_request,
    request_has_explicit_mutation,
)
from .canonical_retrieval import retrieve_knowledge
from .chat_history import current_chat_history
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


class LLMKnowledgeActionDetector:
    """Extract one knowledge action while retaining deterministic authorization."""

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
        self._confirmation_detector = DeterministicActionDetector()

    def detect(
        self,
        request: ChatRequest,
        rewritten_query: str,
        intent: Intent,
    ) -> ActionDetectionResult:
        if intent is not Intent.KNOWLEDGE_FACTS:
            return ActionDetectionResult(intent=intent, confidence=0.0)

        # Confirmation replay is bound to a previously validated immutable
        # action. Re-extracting from a message such as "confirm" would discard
        # the stored target and violate exactly-once confirmation semantics.
        if (
            request.confirmation_token
            and request.metadata.get("confirmation_approved")
            and request.metadata.get("validated_knowledge_actions")
        ):
            return self._confirmation_detector.detect(request, rewritten_query, intent)

        try:
            payload = self.llm.generate_json(
                task=LLMTask.KNOWLEDGE_ACTION_EXTRACTION,
                system_prompt=self.prompts.system("knowledge_action_extraction"),
                user_prompt=self.prompts.user(
                    PromptContext(
                        stage="knowledge_action_extraction",
                        user_id=request.user_id,
                        raw_query=request.raw_query,
                        rewritten_query=rewritten_query,
                        intent=intent.value,
                        metadata=request.metadata,
                        platform_context=request.platform_context,
                        extra={
                            "allowed_actions": ["add", "delete", "modify"],
                            "cardinality": "exactly_one",
                        },
                    )
                ),
                schema=KNOWLEDGE_ACTION_EXTRACTION_SCHEMA,
            )
        except Exception:
            return self._failed(
                intent,
                "knowledge_action_extraction_failed",
                ["action_keyword"],
            )

        try:
            action_name = str(payload["action"]).casefold()
            text_content = str(payload["text_content"]).strip()
            original_text = str(payload["original_text"]).strip()
            replacement_text = str(payload["replacement_text"]).strip()
            confidence = float(payload["confidence"])
            missing_fields = [str(value) for value in payload["missing_fields"]]
        except (KeyError, TypeError, ValueError):
            return self._failed(
                intent,
                "invalid_knowledge_action_extraction",
                ["action_keyword"],
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
                missing_fields=normalized_missing or ["action_keyword"],
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

        current_turn_sources = [
            request.raw_query,
            rewritten_query,
        ]
        history_source = json.dumps(
            current_chat_history(),
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

        # The model extracts semantics but never authorizes a mutation. The raw
        # current query must independently contain exactly the same one action.
        raw_decision = classify_action_request(request.raw_query, intent)
        if (
            not request_has_explicit_mutation(request, intent)
            or raw_decision.selected_action != action_name
        ):
            return self._failed(
                intent,
                "llm_action_not_authorized_by_raw_query",
                ["action_keyword"],
            )

        trusted_topic_title = request.metadata.get("topic_title")
        if trusted_topic_title:
            action_payload["topic_title"] = str(trusted_topic_title)

        return ActionDetectionResult(
            intent=intent,
            confidence=confidence,
            metadata={
                "knowledge_actions": [action_payload],
                "action_authorization": {
                    "intent": intent.value,
                    "action": action_name,
                    "matched_keywords": list(raw_decision.matched_keywords),
                    "reason_summary": raw_decision.reason_summary,
                },
            },
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
        selected_keys = {
            candidate.candidate_key for candidate in selected_candidates
        }
        try:
            payload = self.llm.generate_json(
                task=LLMTask.KNOWLEDGE_CONTENT_FINALIZATION,
                system_prompt=self.prompts.system("knowledge_content_finalization"),
                user_prompt=self.prompts.user(
                    PromptContext(
                        stage="knowledge_content_finalization",
                        user_id=context.request.user_id,
                        raw_query=context.request.raw_query,
                        rewritten_query=context.rewritten_query,
                        intent=Intent.KNOWLEDGE_FACTS.value,
                        metadata=context.request.metadata,
                        platform_context=context.request.platform_context,
                        chat_history=context.chat_history,
                        extra={
                            "operation": action.value,
                            "extracted_action_content": {
                                "text_content": (
                                    action_payload.get("text")
                                    if action is KnowledgeAction.ADD
                                    else action_payload.get("target_description")
                                    if action is KnowledgeAction.DELETE
                                    else None
                                ),
                                "original_text": (
                                    action_payload.get("target_description")
                                    if action is KnowledgeAction.MODIFY
                                    else None
                                ),
                                "replacement_text": action_payload.get("replacement_text"),
                            },
                            "validated_candidate_context": [
                                {
                                    "candidate_key": candidate.candidate_key,
                                    "text": candidate.text,
                                    "source_title": candidate.source_title,
                                    "selected": candidate.candidate_key in selected_keys,
                                }
                                for candidate in all_candidates
                            ],
                            "validation_result": asdict(validation_result),
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

        if action is KnowledgeAction.DELETE:
            if len(selected_candidates) != 1:
                return None
            # Delete does not index new content. Requiring an exact copy here
            # makes the third model an integrity check without changing the
            # repository's established soft-delete/outbox behavior.
            if final_content != selected_candidates[0].text.strip():
                return None
        elif action is KnowledgeAction.ADD:
            proposed = str(action_payload.get("text") or "").strip()
            if self._compact(proposed) != self._compact(final_content):
                return None
        elif action is KnowledgeAction.MODIFY:
            if len(selected_candidates) != 1:
                return None
            selected_key = selected_candidates[0].candidate_key
            assessment = next(
                (
                    item
                    for item in validation_result.candidate_assessments
                    if item.candidate_key == selected_key
                ),
                None,
            )
            matched_text = str(
                getattr(assessment, "matched_text", "") or ""
            )
            replacement = str(
                action_payload.get("replacement_text") or ""
            ).strip()
            candidate_text = selected_candidates[0].text
            if (
                not matched_text
                or not replacement
                or candidate_text.count(matched_text) != 1
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
            return self._clarification_action(
                action,
                action_payload,
                "Knowledge mutation retrieval failed safely.",
            )

        try:
            results = results[
                : self.config.retrieval_validation.knowledge_llm_validation_max_candidates
            ]
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
                        chunk_map[result.entity_id].get("normalized_text")
                        or chunk_map[result.entity_id].get("raw_text")
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
            return self._clarification_action(
                action,
                action_payload,
                "Knowledge candidate hydration failed safely.",
            )
        candidate_map = {candidate.candidate_key: candidate for candidate in candidates}

        validation = self.validator.validate(
            operation=action.value,
            user_query=context.request.raw_query,
            rewritten_query=context.rewritten_query,
            target_description=retrieval_query,
            proposed_content=proposed_content or None,
            replacement_content=replacement_content or None,
            candidates=candidates,
            chat_history=context.chat_history,
        )
        if validation is None:
            return self._clarification_action(
                action,
                action_payload,
                "Knowledge validation model did not return a safe decision.",
            )
        if validation.validation_result is not ActionValidationResult.EXECUTE:
            return ValidatedKnowledgeAction(
                action=action,
                validation_result=validation.validation_result,
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
                factuality_concern=validation.factuality_concern,
                hitl_reason=validation.hitl_reason,
            )

        selected_candidates = [
            candidate_map[key]
            for key in validation.selected_candidate_keys
            if key in candidate_map
        ]
        if action is KnowledgeAction.ADD:
            if selected_candidates:
                return self._clarification_action(
                    action,
                    action_payload,
                    "Add validation unexpectedly selected an existing target.",
                )
        elif len(selected_candidates) != 1:
            return self._clarification_action(
                action,
                action_payload,
                "Destructive validation did not select exactly one SQL target.",
            )

        finalized = self.finalizer.finalize(
            context=context,
            action=action,
            action_payload=action_payload,
            all_candidates=candidates,
            selected_candidates=selected_candidates,
            validation_result=validation,
        )
        if finalized is None:
            return self._clarification_action(
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
        final_text = finalized.final_content
        return ValidatedKnowledgeAction(
            action=action,
            validation_result=ActionValidationResult.EXECUTE,
            target_chunk_ids=target_ids,
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
            confidence=min(
                float(action_payload.get("confidence", 1.0)),
                validation.confidence,
                finalized.confidence,
            ),
            matched_fields=matched_fields,
            reason_summary=(
                f"{validation.reason_summary} {finalized.reason_summary}"
            ).strip(),
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
