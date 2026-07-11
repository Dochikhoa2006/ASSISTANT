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
    validate_last_qa_resolution,
)
from .contracts import Intent
from .last_qa import InMemoryLastQAStore
from .platform import PlatformSelector, PostSelectorHITL
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
    post_selector_hitl: PostSelectorHITL = field(default_factory=PostSelectorHITL)
    prompt_registry: PromptRegistry = field(default_factory=lambda: DEFAULT_PROMPT_REGISTRY)

    def handle(self, request: ChatRequest, repository: AssistantRepository) -> BundledResponse:
        with StageTimer("rewrite"):
            rewritten = self.query_rewriter.rewrite(request.raw_query)

        # State mutations own their current request. Resolve their branch before
        # consulting temporary conversational state so an unrelated stale turn
        # cannot add model latency, trigger retrieval, or redirect the mutation.
        with StageTimer("classification_preflight"):
            preflight_intent = self.classifier.classify(request, rewritten)

        state_mutation = preflight_intent in {Intent.KNOWLEDGE_FACTS, Intent.REMINDER}
        with StageTimer("last_qa_resolution", {"state_mutation": state_mutation}) as last_qa_stage:
            if state_mutation:
                resolution = LastQAResolution(
                    path=LastQAPath.CURRENT_STATE_MUTATION,
                    rewritten_query=rewritten,
                    state=None,
                    did_merge_query=False,
                    skip_broad_retrieval=True,
                    diagnostic_context={"preflight_intent": preflight_intent.value},
                    merge_reason="current_state_mutation_bypassed_last_qa",
                    skip_reason="current_state_mutation_bypassed_broad_retrieval",
                    is_authoritative_state=False,
                )
                validate_last_qa_resolution(resolution)
                intent = preflight_intent
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
                    limit=self.config.retrieval.max_results,
                    min_confidence=self.config.retrieval.conversation_min_confidence,
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
        
        with StageTimer("classification_final", {"preflight_intent_owned": state_mutation}) as final_classification_stage:
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
            platform_state = self.platform_selector.select(bundled, request)
        # This is the sole user-facing HITL decision point.  The selector only
        # supplies channel/draft state; it cannot replace a normal HITL question.
        with StageTimer("post_selector_hitl"):
            platform_payload = self.post_selector_hitl.apply(bundled, platform_state)
        from dataclasses import replace
        with StageTimer("response_finalize"):
            delivery = platform_payload.get("delivery", {})
            hitl = platform_payload.get("hitl", {})
            if hitl.get("required"):
                question = str(hitl["question"])
                if hitl.get("scope") == "general":
                    final_chat_text = (
                        f"{bundled.final_chat_text}\n\n{question}".strip()
                        if hitl.get("append_to_answer", True)
                        else bundled.final_chat_text
                    )
                else:
                    final_chat_text = question
            elif delivery.get("status") == "sent":
                final_chat_text = f"Sent via {delivery.get('provider', delivery.get('channel', 'platform')).title()} to {delivery.get('recipient', 'the recipient')}."
            elif delivery.get("status") == "draft_ready":
                final_chat_text = f"{delivery.get('channel', 'Message').title()} draft is ready for review."
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
                        error_message=str(bundled.platform_payload.get("hitl", {}).get("question") or "") if delivery.get("status") in {"failed", "needs_input"} else None,
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
