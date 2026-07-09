"""Production assembly for the SQL-first assistant."""

from __future__ import annotations

import os

from .bm25_opensearch import OpenSearchBM25Index
from .branches import (
    BranchRouter,
    ClarificationBranch,
    GeneralResponseBranch,
    KnowledgeFactsBranch,
    ReminderBranch,
)
from .branch_orchestration import KnowledgeTargetResolver, ReminderTargetResolver, ValidatedActionBuilder
from .bundler import ChatOutput, ResponseBundler
from .chroma_index import ChromaPersistentVectorIndex
from .classification import LLMLastQAResolver, LLMQueryRewriter
from .config import AssistantConfig, AutoscanConfig, ClassificationConfig, OutboxConfig, RetrievalConfig, QuestionGenerationConfig, MutationPolicyConfig
from .contracts import Intent
from .database import SQLRepository
from .embeddings import SentenceTransformerEmbeddingClient
from .last_qa import DiskCacheLastQAStore
from .llm import OllamaIntentClassifier, OllamaLLMClient, OllamaModelRouter
from .pipeline import AssistantPipeline
from .platform import PlatformSelector
from .prompts import DEFAULT_PROMPT_REGISTRY
from .retrieval import HybridRetriever
from .reranking import SentenceTransformerCrossEncoderReranker
from .action_detection import LLMActionDetector
from .context_filter import HardRuleContextFilter, TwoLayerContextFilter
from .generation import LLMClarificationStrategy, LLMHumanInTheLoopStrategy, LLMReminderSupportingStrategy
from .retrieval_validation import KnowledgeRetrievalValidationStrategy, ReminderRetrievalValidationStrategy
from .settings import ProductionSettings


def build_assistant_config(settings: ProductionSettings) -> AssistantConfig:
    return AssistantConfig(
        retrieval=RetrievalConfig(
            conversation_min_confidence=settings.retrieval.conversation_min_confidence,
            knowledge_min_confidence=settings.retrieval.knowledge_min_confidence,
            max_results=settings.retrieval.max_results,
            rrf_k=settings.retrieval.rrf_k,
            lexical_weight=settings.retrieval.lexical_weight,
            semantic_weight=settings.retrieval.semantic_weight,
            rerank_candidate_limit=settings.retrieval.rerank_candidate_limit,
        ),
        outbox=OutboxConfig(
            max_attempts=settings.worker.outbox_max_attempts,
            batch_size=settings.worker.outbox_batch_size,
            retry_backoff_seconds=settings.worker.outbox_retry_backoff_seconds,
            processing_timeout_seconds=settings.worker.outbox_processing_timeout_seconds,
        ),
        autoscan=AutoscanConfig(interval_seconds=settings.worker.autoscan_interval_seconds),
        classification=ClassificationConfig(intent_keywords={}),
        mutation_policy=MutationPolicyConfig(
            partial_execution_policy=settings.prompt_policy.mutation_partial_execution_policy,
            knowledge_relevance_threshold=settings.prompt_policy.knowledge_target_relevance_threshold,
            knowledge_ambiguity_margin=settings.prompt_policy.knowledge_target_ambiguity_margin,
            knowledge_not_found_policy=settings.prompt_policy.knowledge_target_not_found_policy,
            unsupported_action_policy=settings.prompt_policy.unsupported_action_policy,
        ),
        reminder_resolver=settings.reminder_resolver,
        retrieval_validation=settings.retrieval_validation,
        question_generation=QuestionGenerationConfig(
            enabled=settings.prompt_policy.question_generation_enabled,
            clarification_model=settings.ollama.clarification_question_model,
            human_supporting_model=settings.ollama.human_supporting_question_model,
            reminder_supporting_model=settings.ollama.reminder_supporting_question_model,
            clarification_temperature=settings.ollama.clarification_question_temperature,
            human_supporting_temperature=settings.ollama.human_supporting_question_temperature,
            reminder_supporting_temperature=settings.ollama.reminder_supporting_question_temperature,
            timeout_seconds=settings.ollama.question_generation_timeout,
            clarification_max_tokens=settings.ollama.clarification_question_max_tokens,
            human_supporting_max_tokens=settings.ollama.human_supporting_question_max_tokens,
            reminder_supporting_max_tokens=settings.ollama.reminder_supporting_question_max_tokens,
            clarification_retry_count=settings.ollama.clarification_question_json_retry_count,
            human_supporting_retry_count=settings.ollama.human_supporting_question_json_retry_count,
            reminder_supporting_retry_count=settings.ollama.reminder_supporting_question_json_retry_count,
            question_generation_confidence_threshold=settings.prompt_policy.question_generation_confidence_threshold,
            reminder_supporting_enabled=settings.prompt_policy.reminder_supporting_question_enabled,
            reminder_supporting_min_confidence=settings.prompt_policy.reminder_supporting_question_min_confidence,
            human_supporting_max_count=settings.prompt_policy.human_supporting_question_max_count,
            fallback_policy=settings.prompt_policy.question_generation_fallback_policy,
        ),
        last_qa=LastQAConfig(
            min_confidence=settings.prompt_policy.last_qa_min_confidence,
            clarification_merge_min_confidence=settings.prompt_policy.clarification_merge_min_confidence,
            skip_allowed_interaction_types=settings.prompt_policy.skip_broad_retrieval_allowed_relationships,
            semantic_match_model=settings.ollama.last_qa_model,
            clarification_merge_model=settings.ollama.clarification_merge_model,
            json_retry_count=settings.ollama.last_qa_json_retry_count,
            clarification_merge_json_retry_count=settings.ollama.clarification_merge_json_retry_count,
            enable_reminder_metadata_reply=settings.prompt_policy.last_qa_enable_reminder_metadata_reply,
            clarification_merge_enabled=settings.prompt_policy.clarification_merge_enabled,
        ),
    )


def build_production_repository(settings: ProductionSettings) -> SQLRepository:
    directory = os.path.dirname(settings.database.path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    repository = SQLRepository.persistent(
        settings.database.path,
        enable_wal=settings.database.enable_wal,
        busy_timeout_ms=settings.database.busy_timeout_ms,
    )
    repository.initialize_schema()
    return repository


def build_production_pipeline(settings: ProductionSettings) -> AssistantPipeline:
    repository = build_production_repository(settings)
    assistant_config = build_assistant_config(settings)
    bm25 = OpenSearchBM25Index(settings.opensearch)
    bm25.initialize()
    embeddings = SentenceTransformerEmbeddingClient(settings.embeddings)
    chroma = ChromaPersistentVectorIndex(settings.chroma, embeddings)
    prompt_registry = DEFAULT_PROMPT_REGISTRY
    model_router = OllamaModelRouter(settings.ollama)
    llm = OllamaLLMClient(settings.ollama, model_router)
    reranker = SentenceTransformerCrossEncoderReranker(settings.reranker)
    retriever = HybridRetriever(
        bm25=bm25,
        chroma=chroma,
        reranker=reranker,
        rrf_k=assistant_config.retrieval.rrf_k,
        lexical_weight=assistant_config.retrieval.lexical_weight,
        semantic_weight=assistant_config.retrieval.semantic_weight,
        rerank_candidate_limit=assistant_config.retrieval.rerank_candidate_limit,
    )
    action_detector = LLMActionDetector(
        llm,
        prompt_registry,
        min_confidence=settings.prompt_policy.action_min_confidence,
        risky_action_terms=settings.prompt_policy.risky_action_terms,
    )
    
    hard_rule_filter = HardRuleContextFilter(
        knowledge_min_confidence=settings.prompt_policy.context_filter_knowledge_min_confidence,
        allowed_reminder_statuses=settings.prompt_policy.context_filter_allowed_reminder_statuses,
    )
    context_filter = TwoLayerContextFilter(
        hard_rule_filter=hard_rule_filter,
        llm_judge=None,  # Optional semantic judge can be added here
    )

    clarification_strategy = LLMClarificationStrategy(
        llm=llm,
        prompt_registry=prompt_registry,
        config=assistant_config.question_generation,
    )
    hitl_strategy = LLMHumanInTheLoopStrategy(
        llm=llm,
        prompt_registry=prompt_registry,
        config=assistant_config.question_generation,
    )
    reminder_supporting_strategy = LLMReminderSupportingStrategy(
        llm=llm,
        prompt_registry=prompt_registry,
        config=assistant_config.question_generation,
    )

    knowledge_llm_validator = KnowledgeRetrievalValidationStrategy(
        config=assistant_config.retrieval_validation,
        llm=llm,
        prompts=prompt_registry,
    )
    reminder_llm_validator = ReminderRetrievalValidationStrategy(
        config=assistant_config.retrieval_validation,
        llm=llm,
        prompts=prompt_registry,
    )

    knowledge_target_resolver = KnowledgeTargetResolver(
        retriever=retriever,
        config=assistant_config,
        repository=repository,
        llm_validator=knowledge_llm_validator,
    )
    reminder_target_resolver = ReminderTargetResolver(
        repository=repository,
        config=assistant_config,
        llm_validator=reminder_llm_validator,
    )
    validated_action_builder = ValidatedActionBuilder(
        config=assistant_config,
        knowledge_resolver=knowledge_target_resolver,
        reminder_resolver=reminder_target_resolver,
    )

    router = BranchRouter(
        {
            Intent.CLARIFICATION: ClarificationBranch(
                prompt_registry=prompt_registry,
                config=assistant_config,
                clarification_strategy=clarification_strategy,
            ),
            Intent.GENERAL_RESPONSE: GeneralResponseBranch(
                repository,
                retriever,
                assistant_config,
                context_filter=context_filter,
                llm=llm,
                prompt_registry=prompt_registry,
                hitl_strategy=hitl_strategy,
            ),
            Intent.KNOWLEDGE_FACTS: KnowledgeFactsBranch(
                repository=repository,
                config=assistant_config,
                action_detector=action_detector,
                clarification_strategy=clarification_strategy,
                prompt_registry=prompt_registry,
                validated_action_builder=validated_action_builder,
            ),
            Intent.REMINDER: ReminderBranch(
                repository=repository,
                config=assistant_config,
                action_detector=action_detector,
                clarification_strategy=clarification_strategy,
                reminder_supporting_strategy=reminder_supporting_strategy,
                prompt_registry=prompt_registry,
                validated_action_builder=validated_action_builder,
            ),
        }
    )
    return AssistantPipeline(
        config=assistant_config,
        last_qa_store=DiskCacheLastQAStore(
            settings.last_qa.path,
            ttl_seconds=settings.last_qa.ttl_seconds,
        ),
        query_rewriter=LLMQueryRewriter(llm, prompt_registry),
        last_qa_resolver=LLMLastQAResolver(
            llm,
            prompt_registry,
            min_confidence=settings.prompt_policy.last_qa_min_confidence,
        ),
        retriever=retriever,
        classifier=OllamaIntentClassifier(
            llm,
            prompt_registry,
            min_confidence=settings.prompt_policy.intent_min_confidence,
        ),
        router=router,
        bundler=ResponseBundler(prompt_registry),
        platform_selector=PlatformSelector(),
        chat_output=ChatOutput(),
        prompt_registry=prompt_registry,
    )
