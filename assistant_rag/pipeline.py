"""Ordered assistant pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from .bundler import ChatOutput, ResponseBundler
from .classification import IntentClassifier, LastQAResolver, QueryRewriter
from .config import AssistantConfig
from .contracts import (
    BundledResponse,
    ChatRequest,
    PipelineContext,
    ApprovedConversationContext,
    LastQAPath,
    LastQAResolution,
    LastQAInteractionType,
    QuestionSource,
    validate_last_qa_resolution,
)
from .contracts import Intent
from .last_qa import InMemoryLastQAStore
from .platform import PlatformSelector
from .prompts import DEFAULT_PROMPT_REGISTRY, PromptRegistry
from .retrieval import HybridRetriever
from .branches import BranchRouter
from .database import AssistantRepository
from .context_filter import TwoLayerContextFilter
from .metrics import GLOBAL_METRICS
from .observability import StageTimer, current_trace

from .settings import PromptPolicySettings

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

        # Current-state mutations own their request. Establish that ownership
        # before consulting temporary conversational state so an unrelated stale
        # turn cannot add model latency, trigger retrieval, or redirect it.
        with StageTimer("intent_ownership"):
            current_request_intent = self.classifier.classify(request, rewritten)

        reminder_reply = bool((request.metadata or {}).get("reminder_reply_context"))
        # A notification reply carries an exact reminder-to-source-hop mapping.
        # It must restore that Last-QA state before normal routing, even when
        # its text contains a reminder action such as "turn it off".
        state_mutation = (
            current_request_intent in {Intent.KNOWLEDGE_FACTS, Intent.REMINDER}
            and not reminder_reply
        )
        with StageTimer("last_qa_resolution", {"state_mutation": state_mutation}) as last_qa_stage:
            if reminder_reply:
                last_state = self.last_qa_store.get(request.user_id)
                metadata = request.metadata or {}
                source_topic_id = metadata.get("source_topic_id")
                source_hop_id = metadata.get("source_hop_id")
                if (
                    last_state is not None
                    and source_topic_id == last_state.linked_topic_id
                    and source_hop_id == last_state.linked_hop_id
                ):
                    resolution = LastQAResolution(
                        path=LastQAPath.LATEST_CONTEXT_INTERACTION,
                        rewritten_query=rewritten,
                        state=last_state,
                        did_merge_query=False,
                        skip_broad_retrieval=True,
                        interaction_type=LastQAInteractionType.REMINDER_NOTIFICATION_REPLY,
                        question_source=QuestionSource.NONE,
                        linked_topic_id=last_state.linked_topic_id,
                        linked_hop_id=last_state.linked_hop_id,
                        reminder_id=metadata.get("reminder_id"),
                        notification_id=metadata.get("notification_id"),
                        source_topic_id=source_topic_id,
                        source_hop_id=source_hop_id,
                        merge_reason="exact_reminder_notification_context",
                        skip_reason="reminder_reply_context_hash_matched_source_hop",
                        is_authoritative_state=True,
                    )
                    validate_last_qa_resolution(resolution)
                else:
                    resolution = self.last_qa_resolver.resolve(request, rewritten, last_state)
            elif state_mutation:
                resolution = LastQAResolution(
                    path=LastQAPath.CURRENT_STATE_MUTATION,
                    rewritten_query=rewritten,
                    state=None,
                    did_merge_query=False,
                    skip_broad_retrieval=True,
                    diagnostic_context={"current_request_intent": current_request_intent.value},
                    merge_reason="current_state_mutation_bypassed_last_qa",
                    skip_reason="current_state_mutation_bypassed_broad_retrieval",
                    is_authoritative_state=False,
                )
                validate_last_qa_resolution(resolution)
                intent = current_request_intent
            else:
                last_state = self.last_qa_store.get(request.user_id)
                resolution = self.last_qa_resolver.resolve(request, rewritten, last_state)
            last_qa_stage.metadata["path"] = resolution.path.value
        
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
                conversation_results = self.retriever.retrieve_conversation(
                    user_id=request.user_id,
                    query=retrieval_query,
                )
            with StageTimer("sql_validation", {"entity_type": "conversation_hop", "candidate_count": len(conversation_results)}):
                conversation_results = repository.hydrate_conversation_retrieval_results(
                    user_id=request.user_id,
                    results=conversation_results,
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
                approved_conversation_context = ApprovedConversationContext(
                    approved_conversation_history=[],
                    human_supporting_questions=[],
                    reminder_supporting_questions=[],
                    clarification_question_context=None,
                    extracted_expected_response_types=[],
                    conversation_retrieval_ran=True,
                    conversation_context_status="empty",
                    approved_conversation_count=0,
                )
        
        with StageTimer("classification_final", {"intent_ownership_owned": state_mutation}) as final_classification_stage:
            if not state_mutation:
                intent = self.classifier.classify(
                    request,
                    intent_classifier_query,
                    last_qa_resolution=resolution,
                    approved_conversation_context=approved_conversation_context,
                )
            final_classification_stage.metadata["ran"] = not state_mutation

        last_qa_trace = {
            "input_rewritten_query": rewritten,
            "output_rewritten_query": resolution.rewritten_query,
            "did_merge_query": resolution.did_merge_query,
            "skip_broad_retrieval": resolution.skip_broad_retrieval,
            "path": resolution.path.value,
            "interaction_type": resolution.interaction_type.value if resolution.interaction_type else None,
            "question_source": resolution.question_source.value if resolution.question_source else None,
            "matched_question": resolution.matched_question,
            "is_authoritative_state": resolution.is_authoritative_state,
            "merge_reason": resolution.merge_reason,
            "skip_reason": resolution.skip_reason,
            "broad_retrieval_ran": should_run_broad_retrieval,
            "query_sent_to_retrieval": retrieval_query if should_run_broad_retrieval else None,
            "query_sent_to_intent_classifier": intent_classifier_query,
            "approved_conversation_history_count": len(approved_conversation_context.approved_conversation_history) if approved_conversation_context else 0,
        }

        context = PipelineContext(
            request=request,
            rewritten_query=intent_classifier_query,
            last_qa_state=resolution.state,
            conversation_results=conversation_results,
            intent=intent,
            last_qa_trace=last_qa_trace,
            approved_conversation_context=approved_conversation_context,
        )
        with StageTimer("branch_execution", {"intent": intent.value}):
            branch_result = self.router.route(context, repository)
        with StageTimer("bundling"):
            bundled = self.bundler.bundle(
                request=request,
                rewritten_query=intent_classifier_query,
                branch_result=branch_result,
            )
        with StageTimer("platform_selector"):
            platform_payload = self.platform_selector.select(bundled, request)
        from dataclasses import replace
        with StageTimer("response_finalize"):
            delivery = platform_payload.get("delivery", {})
            if delivery.get("status") in {"needs_input", "pending_review", "failed", "partial_failure"}:
                final_chat_text = str(delivery.get("question") or bundled.final_chat_text)
            elif delivery.get("status") == "sent":
                final_chat_text = f"Sent via {delivery.get('provider', delivery.get('channel', 'platform')).title()} to {delivery.get('recipient', 'the recipient')}."
            elif delivery.get("status") == "draft_ready":
                final_chat_text = (
                    f"{delivery.get('channel', 'Message').title()} draft is ready for review "
                    f"for {delivery.get('recipient', 'the recipient')}."
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
