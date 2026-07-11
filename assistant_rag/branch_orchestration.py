"""Modular orchestration helpers for Knowledge and Reminder branches."""

from __future__ import annotations

import string
import unicodedata
from datetime import datetime
from typing import Any, Protocol

from .config import AssistantConfig
from .contracts import (
    ActionValidationResult,
    KnowledgeAction,
    ReminderAction,
    ReminderCandidateScore,
    ReminderCandidateSummary,
    ReminderTargetResolution,
    ValidatedKnowledgeAction,
    ValidatedReminderAction,
    KnowledgeValidationCandidate,
    ReminderValidationCandidate,
)
from .database import AssistantRepository
from .retrieval import HybridRetriever
from .settings import TargetNotFoundPolicy, UnsupportedActionPolicy
from .retrieval_validation import KnowledgeRetrievalValidationStrategy, ReminderRetrievalValidationStrategy

class KnowledgeTargetResolver:
    def __init__(
        self, 
        retriever: HybridRetriever, 
        config: AssistantConfig, 
        llm_validator: KnowledgeRetrievalValidationStrategy | None = None,
    ):
        self.retriever = retriever
        self.config = config
        self.llm_validator = llm_validator

    def resolve(
        self, 
        user_id: str, 
        action_dict: dict[str, Any],
        user_query: str,
        rewritten_query: str,
        repository: AssistantRepository,
    ) -> tuple[tuple[str, ...], ActionValidationResult]:
        target_description = action_dict.get("target_description")
        if not target_description:
            return (), ActionValidationResult.CLARIFY_MISSING_FIELDS

        results = self.retriever.retrieve_knowledge(
            user_id=user_id,
            query=target_description,
            limit=self.config.retrieval.max_results,
            min_confidence=self.config.mutation_policy.knowledge_relevance_threshold,
        )
        
        if not results:
            if self.config.mutation_policy.knowledge_not_found_policy == TargetNotFoundPolicy.CLARIFY_ON_NOT_FOUND:
                return (), ActionValidationResult.CLARIFY_AMBIGUOUS_TARGET
            return (), ActionValidationResult.SKIP_NOT_FOUND
            
        chunk_ids = [r.entity_id for r in results]
        sql_chunks = repository.get_knowledge_chunks_by_ids(user_id, chunk_ids, include_deleted=False)
        chunk_map = {str(c["chunk_id"]): c for c in sql_chunks}
        
        active_results = []
        for r in results:
            if r.entity_id in chunk_map:
                active_results.append(r)
        
        if not active_results:
            if self.config.mutation_policy.knowledge_not_found_policy == TargetNotFoundPolicy.CLARIFY_ON_NOT_FOUND:
                return (), ActionValidationResult.CLARIFY_AMBIGUOUS_TARGET
            return (), ActionValidationResult.SKIP_NOT_FOUND
        
        # A single candidate has already passed retrieval's configured relevance
        # threshold and SQL rehydration/user-ownership checks. Mutating knowledge
        # still requires an explicit confirmation later in the branch, so a second
        # model pass adds latency and GPU pressure without improving target choice.
        if len(active_results) == 1:
            return (active_results[0].entity_id,), ActionValidationResult.EXECUTE
            
        # Optional LLM validation
        if self.llm_validator and self.config.retrieval_validation.knowledge_llm_validation_enabled:
            candidates = []
            for r in active_results:
                chunk = chunk_map[r.entity_id]
                candidates.append(
                    KnowledgeValidationCandidate(
                        candidate_key=r.entity_id,
                        knowledge_chunk_id=r.entity_id,
                        knowledge_topic_id=chunk.get("knowledge_topic_id", ""),
                        text=chunk.get("normalized_text") or chunk.get("raw_text", ""),
                        source_title=chunk.get("source_title"),
                        retrieval_score=r.confidence,
                        rerank_score=r.rerank_score,
                        is_deleted=chunk.get("is_deleted", False),
                        user_id=user_id,
                    )
                )
            
            val_res = self.llm_validator.validate(
                operation=action_dict.get("action", "").lower(),
                user_query=user_query,
                rewritten_query=rewritten_query,
                target_description=target_description,
                candidates=candidates,
            )
            
            if val_res:
                return val_res.selected_candidate_keys, val_res.validation_result
                
        # Deterministic ambiguity check
        if len(active_results) > 1:
            top_score = active_results[0].confidence
            second_score = active_results[1].confidence
            if (top_score - second_score) < self.config.mutation_policy.knowledge_ambiguity_margin:
                return (), ActionValidationResult.CLARIFY_AMBIGUOUS_TARGET
            
        return (active_results[0].entity_id,), ActionValidationResult.EXECUTE


class ReminderTextNormalizer:
    def normalize(self, text: str | None) -> str:
        if not text:
            return ""
        text = unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("utf-8")
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        return " ".join(text.split())

    def tokenize(self, text: str | None) -> tuple[str, ...]:
        return tuple(self.normalize(text).split())


class FuzzyMatcher(Protocol):
    def score(self, query: str, candidate: str) -> float:
        ...

class RapidFuzzMatcher:
    def score(self, query: str, candidate: str) -> float:
        try:
            from rapidfuzz import fuzz
            return max(0.0, min(1.0, fuzz.ratio(query, candidate) / 100.0))
        except ImportError:
            return 0.0

class DifflibMatcher:
    def score(self, query: str, candidate: str) -> float:
        import difflib
        return max(0.0, min(1.0, difflib.SequenceMatcher(None, query, candidate).ratio()))

class LLMClientProtocol(Protocol):
    def generate_json(self, prompt: str) -> dict[str, Any]:
        ...

class ReminderTargetResolver:
    def __init__(
        self,
        config: AssistantConfig,
        fuzzy_matcher: FuzzyMatcher | None = None,
        llm_client: LLMClientProtocol | None = None,
        llm_validator: ReminderRetrievalValidationStrategy | None = None,
    ):
        self.config = config.reminder_resolver
        self.app_config = config
        self.fuzzy_matcher = fuzzy_matcher or (RapidFuzzMatcher() if self.config.reminder_fuzzy_matcher == "rapidfuzz" else DifflibMatcher())
        self.llm_client = llm_client
        self.normalizer = ReminderTextNormalizer()
        self.llm_validator = llm_validator

    def candidate_statuses_for_action(self, action: ReminderAction) -> tuple[str, ...]:
        if action == ReminderAction.MODIFY:
            return self.config.allowed_reminder_modify_statuses
        if action == ReminderAction.DELETE:
            return self.config.allowed_reminder_delete_statuses
        if action == ReminderAction.TURN_ON:
            return self.config.allowed_reminder_turn_on_statuses
        if action == ReminderAction.TURN_OFF:
            return self.config.allowed_reminder_turn_off_statuses
        return ()

    def resolve(
        self, 
        user_id: str, 
        action: ReminderAction, 
        target_description: str,
        user_query: str,
        rewritten_query: str,
        repository: AssistantRepository,
    ) -> ReminderTargetResolution:
        if not target_description:
            return ReminderTargetResolution(validation_result=ActionValidationResult.CLARIFY_MISSING_FIELDS)
            
        statuses = self.candidate_statuses_for_action(action)
        candidates = repository.list_reminder_candidates(
            user_id=user_id,
            statuses=statuses,
            time_window=None,
            limit=self.config.reminder_target_candidate_limit,
        )
        
        if not candidates:
            if self.config.reminder_target_not_found_policy == TargetNotFoundPolicy.CLARIFY_ON_NOT_FOUND:
                return ReminderTargetResolution(validation_result=ActionValidationResult.CLARIFY_AMBIGUOUS_TARGET)
            return ReminderTargetResolution(validation_result=ActionValidationResult.SKIP_NOT_FOUND)
            
        target_norm = self.normalizer.normalize(target_description)
        target_tokens = set(self.normalizer.tokenize(target_description))
        
        scores: list[ReminderCandidateScore] = []
        for c in candidates:
            c_subj_norm = self.normalizer.normalize(c.subject)
            c_sum_norm = self.normalizer.normalize(c.reminder_summary)
            
            exact = 1.0 if target_norm == c_subj_norm else 0.0
            
            c_tokens = set(self.normalizer.tokenize(c.subject + " " + c.reminder_summary))
            overlap = len(target_tokens & c_tokens)
            token = overlap / max(len(target_tokens), 1)
            
            fuzzy = max(
                self.fuzzy_matcher.score(target_norm, c_subj_norm),
                self.fuzzy_matcher.score(target_norm, c_sum_norm)
            )
            
            w_sum = self.config.reminder_subject_weight + self.config.reminder_summary_weight
            final = (exact * self.config.reminder_subject_weight + fuzzy * self.config.reminder_summary_weight) / max(w_sum, 1.0)
            final = max(final, exact)
            final = max(0.0, min(1.0, final))
            
            # Tie break with recency
            final += 0.0001
            
            scores.append(ReminderCandidateScore(
                candidate_id=c.reminder_id,
                exact_score=exact,
                token_score=token,
                fuzzy_score=fuzzy,
                time_score=0.0,
                entity_score=0.0,
                status_score=0.0,
                recency_score=0.0,
                final_score=final,
                matched_fields=("subject",) if exact > 0 else (),
                reason_summary="Scored deterministically"
            ))
            
        scores.sort(key=lambda x: x.final_score, reverse=True)
        top_score = scores[0]
        
        if top_score.exact_score < 1.0 and top_score.fuzzy_score < self.config.reminder_fuzzy_match_threshold:
            if self.config.reminder_target_not_found_policy == TargetNotFoundPolicy.CLARIFY_ON_NOT_FOUND:
                return ReminderTargetResolution(validation_result=ActionValidationResult.CLARIFY_AMBIGUOUS_TARGET)
            return ReminderTargetResolution(validation_result=ActionValidationResult.SKIP_NOT_FOUND)

        if top_score.final_score < self.config.reminder_target_relevance_threshold:
            if self.config.reminder_target_not_found_policy == TargetNotFoundPolicy.CLARIFY_ON_NOT_FOUND:
                return ReminderTargetResolution(validation_result=ActionValidationResult.CLARIFY_AMBIGUOUS_TARGET)
            return ReminderTargetResolution(validation_result=ActionValidationResult.SKIP_NOT_FOUND)
            
        if len(scores) > 1:
            margin = top_score.final_score - scores[1].final_score
            if margin < self.config.reminder_target_ambiguity_margin:
                clarification_candidates = tuple(c for c in candidates if c.reminder_id in {s.candidate_id for s in scores[:2]})
                
                if self.llm_validator and self.app_config.retrieval_validation.reminder_llm_validation_enabled:
                    val_candidates = []
                    for c in candidates:
                        s = next((s for s in scores if s.candidate_id == c.reminder_id), None)
                        if s:
                            val_candidates.append(
                                ReminderValidationCandidate(
                                    candidate_key=c.reminder_id,
                                    reminder_id=c.reminder_id,
                                    subject=c.subject,
                                    reminder_summary=c.reminder_summary,
                                    raw_reminder=c.raw_reminder,
                                    reminder_time=c.reminder_time,
                                    status=c.status,
                                    deterministic_score=s.final_score,
                                    observed_version=c.version,
                                    observed_status=c.status,
                                )
                            )
                    val_res = self.llm_validator.validate(
                        operation=action.value.lower(),
                        user_query=user_query,
                        rewritten_query=rewritten_query,
                        target_description=target_description,
                        target_time_signals=None, # TBD from parsing if any
                        candidates=val_candidates,
                    )
                    if val_res and val_res.validation_result == ActionValidationResult.EXECUTE:
                        chosen = next(c for c in candidates if c.reminder_id == val_res.selected_candidate_keys[0])
                        s = next(s for s in scores if s.candidate_id == chosen.reminder_id)
                        return ReminderTargetResolution(
                            validation_result=ActionValidationResult.EXECUTE,
                            target_reminder_ids=(chosen.reminder_id,),
                            top_score=s,
                            chosen_candidate=chosen
                        )
                
                return ReminderTargetResolution(
                    validation_result=ActionValidationResult.CLARIFY_AMBIGUOUS_TARGET,
                    top_score=top_score,
                    clarification_candidates=clarification_candidates
                )
        
        chosen = next(c for c in candidates if c.reminder_id == top_score.candidate_id)
        return ReminderTargetResolution(
            validation_result=ActionValidationResult.EXECUTE,
            target_reminder_ids=(top_score.candidate_id,),
            top_score=top_score,
            chosen_candidate=chosen
        )


class ValidatedActionBuilder:
    def __init__(self, config: AssistantConfig, knowledge_resolver: KnowledgeTargetResolver, reminder_resolver: ReminderTargetResolver):
        self.config = config
        self.knowledge_resolver = knowledge_resolver
        self.reminder_resolver = reminder_resolver

    def build_knowledge_actions(self, user_id: str, actions: list[dict[str, Any]], user_query: str, rewritten_query: str, repository: AssistantRepository) -> list[ValidatedKnowledgeAction]:
        validated = []
        for action_dict in actions:
            action_type_str = action_dict.get("action", "").lower()
            
            if action_dict.get("risk_approved") is False:
                res = ActionValidationResult.REJECT_UNSUPPORTED_OPERATION
                if self.config.mutation_policy.unsupported_action_policy == UnsupportedActionPolicy.CLARIFY:
                    res = ActionValidationResult.CLARIFY_MISSING_FIELDS
                validated.append(ValidatedKnowledgeAction(
                    action=KnowledgeAction.ADD, target_chunk_ids=(), new_text=None,
                    target_status="", confidence=0.0, validation_result=res
                ))
                continue
                
            if action_type_str not in {a.value for a in KnowledgeAction}:
                res = ActionValidationResult.REJECT_UNSUPPORTED_OPERATION
                if self.config.mutation_policy.unsupported_action_policy == UnsupportedActionPolicy.CLARIFY:
                    res = ActionValidationResult.CLARIFY_MISSING_FIELDS
                validated.append(ValidatedKnowledgeAction(
                    action=KnowledgeAction.ADD, target_chunk_ids=(), new_text=None,
                    target_status="", confidence=0.0, validation_result=res
                ))
                continue
                
            action_type = KnowledgeAction(action_type_str)
            confidence = float(action_dict.get("confidence", 1.0))
            
            if action_type == KnowledgeAction.ADD:
                text = action_dict.get("text")
                if not text:
                    res = ActionValidationResult.CLARIFY_MISSING_FIELDS
                else:
                    res = ActionValidationResult.EXECUTE
                validated.append(ValidatedKnowledgeAction(
                    action=action_type, target_chunk_ids=(), new_text=text,
                    target_status="active", confidence=confidence, validation_result=res,
                    topic_title=action_dict.get("topic_title")
                ))
            elif action_type in {KnowledgeAction.DELETE, KnowledgeAction.MODIFY}:
                target_ids, res = self.knowledge_resolver.resolve(
                    user_id=user_id, 
                    action_dict=action_dict, 
                    user_query=user_query, 
                    rewritten_query=rewritten_query,
                    repository=repository,
                )
                new_text = action_dict.get("replacement_text")
                if action_type == KnowledgeAction.MODIFY and not new_text and res == ActionValidationResult.EXECUTE:
                    res = ActionValidationResult.CLARIFY_MISSING_FIELDS
                validated.append(ValidatedKnowledgeAction(
                    action=action_type, target_chunk_ids=target_ids, new_text=new_text,
                    target_status="active", confidence=confidence, validation_result=res,
                    target_description=action_dict.get("target_description")
                ))
        return validated

    def build_reminder_actions(self, user_id: str, actions: list[dict[str, Any]], user_query: str, rewritten_query: str, repository: AssistantRepository) -> list[ValidatedReminderAction]:
        validated = []
        for action_dict in actions:
            action_type_str = action_dict.get("action", "").lower()
            if action_type_str not in {a.value for a in ReminderAction}:
                res = ActionValidationResult.REJECT_UNSUPPORTED_OPERATION
                if self.config.mutation_policy.unsupported_action_policy == UnsupportedActionPolicy.CLARIFY:
                    res = ActionValidationResult.CLARIFY_MISSING_FIELDS
                validated.append(ValidatedReminderAction(
                    action=ReminderAction.ADD, validation_result=res
                ))
                continue
                
            action_type = ReminderAction(action_type_str)
            confidence = float(action_dict.get("confidence", 1.0))
            
            if action_type == ReminderAction.ADD:
                # Notification time is authoritative only when the user stated
                # it explicitly.  An event/deadline time is passed through to
                # the timing planner later; it must not silently be treated as
                # the notification timestamp.
                event_value = action_dict.get("event_time")
                notification_value = action_dict.get("notification_time")
                legacy_value = action_dict.get("reminder_time")
                semantics = str(action_dict.get("time_semantics") or "unspecified")
                time = notification_value or event_value or legacy_value
                subject = action_dict.get("subject")
                if not time or not subject:
                    res = ActionValidationResult.CLARIFY_MISSING_FIELDS
                else:
                    res = ActionValidationResult.EXECUTE
                try:
                    dt = datetime.fromisoformat(str(time)) if time else None
                    event_dt = datetime.fromisoformat(str(event_value)) if event_value else None
                except (TypeError, ValueError):
                    dt = None
                    event_dt = None
                    res = ActionValidationResult.CLARIFY_MISSING_FIELDS
                # Legacy reminder_time has always meant notification time, so
                # leave it untouched unless the extractor explicitly marked it
                # as an event time.
                if event_dt is None and semantics == "event_time" and dt is not None:
                    event_dt = dt
                validated.append(ValidatedReminderAction(
                    action=action_type, 
                    event_time=event_dt,
                    reminder_time=dt, 
                    subject=subject,
                    reminder_summary=action_dict.get("reminder_summary"), 
                    confidence=confidence, 
                    validation_result=res
                ))
            else:
                target_desc = action_dict.get("target_description")
                resolution = self.reminder_resolver.resolve(
                    user_id=user_id, 
                    action=action_type, 
                    target_description=target_desc or "",
                    user_query=user_query,
                    rewritten_query=rewritten_query,
                    repository=repository,
                )
                res = resolution.validation_result
                
                if action_type == ReminderAction.MODIFY and res == ActionValidationResult.EXECUTE:
                    if not action_dict.get("new_reminder_time") and not action_dict.get("new_subject"):
                        res = ActionValidationResult.CLARIFY_MISSING_FIELDS
                
                dt = None
                if action_dict.get("new_reminder_time"):
                    dt = datetime.fromisoformat(action_dict.get("new_reminder_time"))

                observed_status = None
                observed_version = None
                observed_time = None
                if res == ActionValidationResult.EXECUTE and resolution.chosen_candidate:
                    observed_status = resolution.chosen_candidate.status
                    observed_version = resolution.chosen_candidate.version
                    observed_time = resolution.chosen_candidate.reminder_time

                validated.append(ValidatedReminderAction(
                    action=action_type, 
                    target_reminder_ids=resolution.target_reminder_ids,
                    observed_status=observed_status,
                    observed_version=observed_version,
                    observed_reminder_time=observed_time,
                    replacement_time=dt,
                    replacement_subject=action_dict.get("new_subject"),
                    replacement_summary=None,
                    confidence=confidence, 
                    validation_result=res
                ))
        return validated
