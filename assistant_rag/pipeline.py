"""Ordered assistant pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from .bundler import ChatOutput, ResponseBundler
from .classification import IntentClassifier, LastQAResolver, QueryRewriter
from .config import AssistantConfig
from .contracts import BundledResponse, ChatRequest, PipelineContext
from .contracts import Intent
from .last_qa import InMemoryLastQAStore
from .platform import PlatformSelector
from .prompts import DEFAULT_PROMPT_REGISTRY, PromptRegistry
from .retrieval import HybridRetriever
from .branches import BranchRouter

from .settings import PromptPolicySettings

@dataclass
class AssistantPipeline:
    config: AssistantConfig
    last_qa_store: InMemoryLastQAStore
    query_rewriter: QueryRewriter
    last_qa_resolver: LastQAResolver
    retriever: HybridRetriever
    classifier: IntentClassifier
    router: BranchRouter
    bundler: ResponseBundler
    platform_selector: PlatformSelector
    chat_output: ChatOutput
    prompt_registry: PromptRegistry = field(default_factory=lambda: DEFAULT_PROMPT_REGISTRY)

    def handle(self, request: ChatRequest) -> BundledResponse:
        rewritten = self.query_rewriter.rewrite(request.raw_query)
        last_state = self.last_qa_store.get(request.user_id)
        resolution = self.last_qa_resolver.resolve(request, rewritten, last_state)
        
        should_run_broad_retrieval = not resolution.skip_broad_retrieval
        
        # In our implementation, resolution.rewritten_query is the merged query if did_merge_query=True
        # and the original rewritten query otherwise.
        retrieval_query = resolution.rewritten_query
        intent_classifier_query = resolution.rewritten_query
        
        conversation_results = []
        if should_run_broad_retrieval:
            conversation_results = self.retriever.retrieve_conversation(
                user_id=request.user_id,
                query=retrieval_query,
                limit=self.config.retrieval.max_results,
                min_confidence=self.config.retrieval.conversation_min_confidence,
            )
            
        intent = self.classifier.classify(request, intent_classifier_query)

        last_qa_trace = {
            "input_rewritten_query": rewritten,
            "output_rewritten_query": resolution.rewritten_query,
            "did_merge_query": resolution.did_merge_query,
            "skip_broad_retrieval": resolution.skip_broad_retrieval,
            "path": resolution.path.value,
            "merge_reason": resolution.merge_reason,
            "skip_reason": resolution.skip_reason,
            "broad_retrieval_ran": should_run_broad_retrieval,
            "query_sent_to_retrieval": retrieval_query if should_run_broad_retrieval else None,
            "query_sent_to_intent_classifier": intent_classifier_query,
        }

        context = PipelineContext(
            request=request,
            rewritten_query=intent_classifier_query,
            last_qa_state=resolution.state,
            conversation_results=conversation_results,
            intent=intent,
            last_qa_trace=last_qa_trace,
        )
        branch_result = self.router.route(context)
        bundled = self.bundler.bundle(
            request=request,
            rewritten_query=intent_classifier_query,
            branch_result=branch_result,
        )
        self.platform_selector.select(bundled, request)
        self.last_qa_store.save(request.user_id, bundled.last_qa_state)
        self.chat_output.emit(bundled)
        return bundled
