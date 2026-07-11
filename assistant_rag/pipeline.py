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

        # State mutations own their current request. Resolve their branch before
        # consulting temporary conversational state so an unrelated stale turn
        # cannot add model latency, trigger retrieval, or redirect the mutation.
        with StageTimer("classification_preflight"):
            preflight_intent = self.classifier.classify(request, rewritten)

        state_mutation = preflight_intent in {Intent.KNOWLEDGE_FACTS, Intent.REMINDER}
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
            with StageTimer("last_qa"):
                last_state = self.last_qa_store.get(request.user_id)
                resolution = self.last_qa_resolver.resolve(request, rewritten, last_state)
        
        retrieval_query = resolution.rewritten_query
        intent_classifier_query = resolution.rewritten_query
        
        should_run_broad_retrieval = (
            not resolution.skip_broad_retrieval
            and self.config.context_filter.conversation_retrieval_after_last_qa_enabled
            and self.config.context_filter.conversation_retrieval_before_intent_enabled
        )
        
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
        
        if not state_mutation:
            with StageTimer("classification"):
                intent = self.classifier.classify(
                    request,
                    intent_classifier_query,
                    last_qa_resolution=resolution,
                    approved_conversation_context=approved_conversation_context,
                )

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
        platform_payload = self.platform_selector.select(bundled, request)
        from dataclasses import replace
        trace = current_trace()
        bundled = replace(
            bundled,
            platform_payload=platform_payload,
            trace_summary=trace.summary() if trace else None,
        )
        self.last_qa_store.save(request.user_id, bundled.last_qa_state)
        self.chat_output.emit(bundled)
        return bundled
