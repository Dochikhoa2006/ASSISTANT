"""Ordered assistant pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal
from .bundler import ChatOutput, ResponseBundler
from .classification import IntentClassifier, LastQAResolver, QueryRewriter
from .config import AssistantConfig
from .contracts import (
    BundledResponse,
    ChatRequest,
    PipelineContext,
    ApprovedConversationContext,
    LastQAInteractionType,
    LastQAPath,
    LastQAResolution,
    LastQAState,
    RetrievalResult,
)
from .last_qa import InMemoryLastQAStore
from .platform import PlatformSelector
from .prompts import DEFAULT_PROMPT_REGISTRY, PromptRegistry
from .retrieval import HybridRetriever
from .branches import BranchRouter
from .database import AssistantRepository
from .context_filter import TwoLayerContextFilter
from .metrics import GLOBAL_METRICS
from .observability import StageTimer, current_trace
from .chat_history import canonical_chat_history_scope, select_chat_history
from .indexing import BackgroundIndexer
from .reminder_reply import (
    mark_reminder_state_replied,
    reminder_state_hash,
    verified_reminder_state,
)


def _branch_outbox_job_ids(branch_result: Any) -> list[str]:
    payload = dict(getattr(branch_result, "indexing_job_result", {}) or {})
    values = payload.get("outbox_job_ids") or []
    if isinstance(values, str):
        values = [values]
    fallback = payload.get("conversation_hop_job_id")
    return list(
        dict.fromkeys(
            str(job_id)
            for job_id in [*values, fallback]
            if job_id
        )
    )

@dataclass
class AssistantPipeline:
    config: AssistantConfig
    last_qa_store: InMemoryLastQAStore
    query_rewriter: QueryRewriter
    last_qa_resolver: LastQAResolver
    retriever: HybridRetriever
    context_filter: TwoLayerContextFilter
    classifier: IntentClassifier
    router: BranchRouter
    bundler: ResponseBundler
    platform_selector: PlatformSelector
    chat_output: ChatOutput
    prompt_registry: PromptRegistry = field(default_factory=lambda: DEFAULT_PROMPT_REGISTRY)

    def handle(self, request: ChatRequest, repository: AssistantRepository) -> BundledResponse:
        with StageTimer("rewrite"):
            rewritten = self.query_rewriter.rewrite(request.raw_query)

        # Last-QA is a compulsory stage for every request. Intent classification
        # happens only after the resolver has accepted, merged, or rejected the
        # latest state.
        with StageTimer("last_qa_resolution") as last_qa_stage:
            last_state = self.last_qa_store.get(request.user_id)
            resolution = self.last_qa_resolver.resolve(request, rewritten, last_state)
            last_qa_stage.metadata["path"] = resolution.path.value
        
        # Semantic authority boundary: after Last-QA resolution, every
        # downstream consumer must use this rewritten value. The request's
        # original text remains available only to quarantined audit/control
        # paths and must not influence retrieval, routing, prompts, or tools.
        retrieval_query = resolution.rewritten_query
        intent_classifier_query = resolution.rewritten_query
        
        with StageTimer("conversation_retrieval_gate", {
            "last_qa_requested_skip": resolution.skip_broad_retrieval,
            "after_last_qa_enabled": self.config.context_filter.conversation_retrieval_after_last_qa_enabled,
            "before_intent_enabled": self.config.context_filter.conversation_retrieval_before_intent_enabled,
        }) as retrieval_gate:
            should_run_broad_retrieval = (
                not resolution.skip_broad_retrieval
                and self.config.context_filter.conversation_retrieval_after_last_qa_enabled
                and self.config.context_filter.conversation_retrieval_before_intent_enabled
            )
            retrieval_gate.metadata["will_run"] = should_run_broad_retrieval
        
        approved_conversation_context: ApprovedConversationContext | None = None
        conversation_results = []
        
        if should_run_broad_retrieval:
            with StageTimer("retrieval", {"entity_type": "conversation_hop"}):
                raw_conversation_results = self.retriever.retrieve_conversation(
                    user_id=request.user_id,
                    query=retrieval_query,
                )
            with StageTimer("sql_validation", {"entity_type": "conversation_hop", "candidate_count": len(raw_conversation_results)}):
                conversation_results = repository.hydrate_conversation_retrieval_results(
                    user_id=request.user_id,
                    results=raw_conversation_results,
                )
            if not conversation_results:
                GLOBAL_METRICS.increment("retrieval_empty_total", entity_type="conversation_hop")
            from dataclasses import replace
            resolution = replace(resolution, state=None)
            if conversation_results:
                approved_conversation_context = self.context_filter.filter_conversation_only(
                    user_id=request.user_id,
                    conversation_results=conversation_results,
                    query=retrieval_query,
                    config=self.config.context_filter,
                )
            else:
                context_status = (
                    "all_rejected" if raw_conversation_results else "empty"
                )
                approved_conversation_context = ApprovedConversationContext(
                    approved_conversation_history=[],
                    human_supporting_questions=[],
                    reminder_supporting_questions=[],
                    clarification_question_context=None,
                    extracted_expected_response_types=[],
                    conversation_retrieval_ran=True,
                    conversation_context_status=context_status,
                    approved_conversation_count=0,
                    _rejected_conversation_ids=tuple(
                        result.entity_id for result in raw_conversation_results
                    ),
                )

        # Exact supporting-question answers intentionally skip broad retrieval.
        # Preserve the resolver's authoritative topic/hop identity as approved
        # selection candidates so the deterministic sub-branch detector can
        # apply the same candidate contract without introducing unrelated
        # conversation text into canonical chat history.
        if (
            not should_run_broad_retrieval
            and resolution.path == LastQAPath.LATEST_CONTEXT_INTERACTION
            and resolution.interaction_type
            == LastQAInteractionType.SUPPORTING_QUESTION_ANSWER
            and resolution.is_authoritative_state
            and resolution.state is not None
            and resolution.state.linked_topic_id
            and resolution.state.linked_hop_id
        ):
            state = resolution.state
            approved_conversation_context = ApprovedConversationContext(
                approved_conversation_history=[],
                human_supporting_questions=list(state.supporting_questions),
                reminder_supporting_questions=(
                    [state.reminder_supporting_question]
                    if state.reminder_supporting_question is not None
                    else []
                ),
                clarification_question_context=state.clarification_question,
                extracted_expected_response_types=(
                    [state.expected_response_type]
                    if state.expected_response_type is not None
                    else []
                ),
                conversation_retrieval_ran=False,
                conversation_context_status="not_run",
                approved_conversation_count=0,
                top_hop_rerank_score=None,
                _internal_selected_topic_candidates=[state.linked_topic_id],
                _internal_selected_hop_candidates=[state.linked_hop_id],
                _validation_summary=(
                    "Authoritative Last-QA supporting-question identity."
                ),
            )
        
        chat_history = select_chat_history(
            conversation_retrieval=should_run_broad_retrieval,
            approved_conversation_context=approved_conversation_context,
            last_qa_state=resolution.state,
        )
        chat_history_source = (
            "conversation_retrieval" if should_run_broad_retrieval else "last_qa"
        )

        # Classification is the first consumer. The same immutable request-scoped
        # value remains active through branches, bundling, and platform selection.
        with canonical_chat_history_scope(chat_history):
            return self._handle_with_chat_history(
                request=request,
                repository=repository,
                initially_rewritten_query=rewritten,
                retrieval_query=retrieval_query,
                intent_classifier_query=intent_classifier_query,
                resolution=resolution,
                conversation_results=conversation_results,
                approved_conversation_context=approved_conversation_context,
                conversation_retrieval=should_run_broad_retrieval,
                chat_history=chat_history,
                chat_history_source=chat_history_source,
                previous_last_qa_state=last_state,
            )

    def _handle_with_chat_history(
        self,
        *,
        request: ChatRequest,
        repository: AssistantRepository,
        initially_rewritten_query: str,
        retrieval_query: str,
        intent_classifier_query: str,
        resolution: LastQAResolution,
        conversation_results: list[RetrievalResult],
        approved_conversation_context: ApprovedConversationContext | None,
        conversation_retrieval: bool,
        chat_history: list[dict[str, Any]],
        chat_history_source: Literal["conversation_retrieval", "last_qa"],
        previous_last_qa_state: LastQAState | None,
    ) -> BundledResponse:
        with StageTimer("classification"):
            intent = self.classifier.classify(
                request,
                intent_classifier_query,
                last_qa_resolution=resolution,
                approved_conversation_context=approved_conversation_context,
            )

        last_qa_trace = {
            "input_rewritten_query": initially_rewritten_query,
            "output_rewritten_query": resolution.rewritten_query,
            "did_merge_query": resolution.did_merge_query,
            "skip_broad_retrieval": resolution.skip_broad_retrieval,
            "resolution_confidence": resolution.confidence,
            "path": resolution.path.value,
            "interaction_type": resolution.interaction_type.value if resolution.interaction_type else None,
            "question_source": resolution.question_source.value if resolution.question_source else None,
            "matched_question": resolution.matched_question,
            "is_authoritative_state": resolution.is_authoritative_state,
            "merge_reason": resolution.merge_reason,
            "skip_reason": resolution.skip_reason,
            "broad_retrieval_ran": conversation_retrieval,
            "conversation_retrieval": conversation_retrieval,
            "chat_history_source": chat_history_source,
            "chat_history_count": len(chat_history),
            "query_sent_to_retrieval": retrieval_query if conversation_retrieval else None,
            "query_sent_to_intent_classifier": intent_classifier_query,
            "approved_conversation_history_count": len(approved_conversation_context.approved_conversation_history) if approved_conversation_context else 0,
        }

        context = PipelineContext(
            request=request,
            rewritten_query=intent_classifier_query,
            last_qa_state=resolution.state,
            conversation_results=conversation_results,
            intent=intent,
            chat_history=chat_history,
            conversation_retrieval=conversation_retrieval,
            chat_history_source=chat_history_source,
            last_qa_trace=last_qa_trace,
            approved_conversation_context=approved_conversation_context,
            previous_last_qa_state=previous_last_qa_state,
        )
        with StageTimer("branch_execution", {"intent": intent.value}):
            branch_result = self.router.route(context, repository)
        outbox_job_ids = _branch_outbox_job_ids(branch_result)
        with StageTimer(
            "branch_index_sync",
            {"requested_job_count": len(outbox_job_ids)},
        ) as index_stage:
            processed_jobs = 0
            bm25 = getattr(self.retriever, "bm25", None)
            chroma = getattr(self.retriever, "chroma", None)
            outbox_config = getattr(self.config, "outbox", None)
            if outbox_job_ids and bm25 is not None and chroma is not None and outbox_config is not None:
                processed_jobs = BackgroundIndexer(
                    repository=repository,
                    bm25=bm25,
                    chroma=chroma,
                    config=outbox_config,
                ).process_job_ids(outbox_job_ids)
                index_stage.metadata["sync_mode"] = "request_scoped"
            elif outbox_job_ids:
                # Deterministic/unit-injected retrievers may intentionally have
                # no derived stores. Keep the durable jobs pending for the
                # normal background worker instead of losing the branch result.
                index_stage.metadata["sync_mode"] = "durable_outbox_deferred"
            index_stage.metadata["claimed_job_count"] = processed_jobs
        with StageTimer("bundling"):
            bundled = self.bundler.bundle(
                request=request,
                rewritten_query=intent_classifier_query,
                branch_result=branch_result,
            )
        if (
            resolution.interaction_type
            is LastQAInteractionType.REMINDER_NOTIFICATION_REPLY
            and resolution.is_authoritative_state
            and resolution.state is not None
        ):
            reminder_state = verified_reminder_state(
                resolution.state.reminder_state,
                resolution.state.reminder_state_hash,
            )
            if reminder_state is not None:
                replied_state = mark_reminder_state_replied(
                    reminder_state,
                    reply_hop_id=bundled.conversation_hop_id,
                )
                bundled = replace(
                    bundled,
                    last_qa_state=replace(
                        bundled.last_qa_state,
                        reminder_state=replied_state,
                        reminder_state_hash=reminder_state_hash(replied_state),
                    ),
                )
        with StageTimer("platform_selector"):
            platform_payload = self.platform_selector.select(bundled, request)
        with StageTimer("response_finalize"):
            delivery = platform_payload.get("delivery", {})
            if delivery.get("status") in {"needs_input", "pending_review", "failed", "partial_failure"}:
                final_chat_text = str(delivery.get("question") or bundled.final_chat_text)
            elif delivery.get("status") == "sent":
                final_chat_text = f"Sent via {delivery.get('provider', delivery.get('channel', 'platform')).title()} to {delivery.get('recipient', 'the recipient')}."
            elif delivery.get("status") == "draft_ready":
                final_chat_text = (
                    f"{delivery.get('channel', 'Message').title()} draft is ready for review "
                    f"for {delivery.get('recipient', 'the recipient')}. It has not been sent. "
                    "To deliver it, explicitly ask to send the email."
                )
            elif delivery.get("status") == "draft_saved":
                final_chat_text = (
                    f"Draft saved in {delivery.get('provider', delivery.get('channel', 'the platform')).title()} "
                    f"for {delivery.get('recipient', 'the recipient')}."
                )
            else:
                final_chat_text = bundled.final_chat_text
        bundled = replace(
            bundled,
            final_chat_text=final_chat_text,
            last_qa_state=replace(bundled.last_qa_state, last_response=final_chat_text),
            platform_payload=platform_payload,
        )
        # Record delivery metadata separately from the conversation answer. The
        # record intentionally contains no credential, token, or attachment path.
        recorder = getattr(repository, "record_platform_delivery", None)
        with StageTimer("platform_delivery_audit"):
            if callable(recorder) and delivery.get("channel") not in (None, "none"):
                try:
                    recorder(
                        user_id=request.user_id,
                        conversation_hop_id=bundled.conversation_hop_id,
                        channel=str(delivery.get("channel")),
                        status=str(delivery.get("status", "unknown")),
                        recipient=str(delivery.get("recipient") or bundled.platform_payload.get("draft", {}).get("recipient") or ""),
                        message=bundled.platform_payload.get("draft", {}),
                        error_message=str(delivery.get("question") or "") if delivery.get("status") in {"failed", "partial_failure", "needs_input"} else None,
                    )
                except Exception:
                    # Delivery history must not hide a completed send or response.
                    pass
        with StageTimer("last_qa_persistence"):
            self.last_qa_store.save(request.user_id, bundled.last_qa_state)
        with StageTimer("chat_output"):
            self.chat_output.emit(bundled)
        trace = current_trace()
        return replace(bundled, trace_summary=trace.summary() if trace else None)
