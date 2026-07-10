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
from .classification import LLMLastQAResolver, LLMQueryRewriter, T5CanardQueryRewriter
from .config import AssistantConfig, AutoscanConfig, ClassificationConfig, OutboxConfig, RetrievalConfig, QuestionGenerationConfig, MutationPolicyConfig, ContextFilterConfig, GeneralPurposeConfig, LastQAConfig
from .contracts import Intent
from .database import AssistantRepository
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
from .generation import LLMClarificationStrategy, LLMHumanInTheLoopStrategy, LLMReminderSupportingStrategy, LLMGeneralHITLStrategy
from .retrieval_validation import KnowledgeRetrievalValidationStrategy, ReminderRetrievalValidationStrategy
from .settings import ProductionSettings
from .general_sub_branch import GeneralSubBranchDetector
from .content_composer import (
    AnswerGenerationTool,
    GenerateExcelTool,
    GeneratePDFTool,
    GeneratePPTXTool,
    ContentToolRegistry,
    ReActContentComposer,
)


def build_assistant_config(settings: ProductionSettings) -> AssistantConfig:
    return AssistantConfig(
        retrieval=RetrievalConfig(
            conversation_min_confidence=settings.retrieval.conversation_min_confidence,
            knowledge_min_confidence=settings.retrieval.knowledge_min_confidence,
            max_results=settings.retrieval.max_results,
            bm25_top_k=settings.retrieval.bm25_top_k,
            chroma_top_k=settings.retrieval.chroma_top_k,
            min_confidence=settings.retrieval.min_confidence,
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
            clarification_model=settings.ollama.model_generate_clarification,
            human_supporting_model=settings.ollama.model_generate_human_supporting,
            reminder_supporting_model=settings.ollama.model_generate_reminder_supporting,
            clarification_temperature=settings.ollama.temperature_generate_clarification,
            human_supporting_temperature=settings.ollama.temperature_generate_human_supporting,
            reminder_supporting_temperature=settings.ollama.temperature_generate_reminder_supporting,
            timeout_seconds=settings.ollama.timeout_generate_clarification,
            clarification_max_tokens=settings.ollama.num_predict_generate_clarification,
            human_supporting_max_tokens=settings.ollama.num_predict_generate_human_supporting,
            reminder_supporting_max_tokens=settings.ollama.num_predict_generate_reminder_supporting,
            clarification_retry_count=settings.ollama.json_retry_count_generate_clarification,
            human_supporting_retry_count=settings.ollama.json_retry_count_generate_human_supporting,
            reminder_supporting_retry_count=settings.ollama.json_retry_count_generate_reminder_supporting,
            question_generation_confidence_threshold=settings.prompt_policy.question_generation_confidence_threshold,
            reminder_supporting_enabled=settings.prompt_policy.reminder_supporting_question_enabled,
            reminder_supporting_min_confidence=settings.prompt_policy.reminder_supporting_question_min_confidence,
            human_supporting_max_count=settings.prompt_policy.human_supporting_question_max_count,
            fallback_policy=settings.prompt_policy.question_generation_fallback_policy,
        ),
        last_qa=LastQAConfig(
            min_confidence=settings.prompt_policy.last_qa_min_confidence,
            skip_broad_retrieval_min_confidence=settings.prompt_policy.last_qa_skip_broad_retrieval_min_confidence,
            clarification_merge_min_confidence=settings.prompt_policy.clarification_merge_min_confidence,
            skip_allowed_interaction_types=settings.prompt_policy.skip_broad_retrieval_allowed_relationships,
            semantic_match_model=settings.ollama.model_last_qa,
            clarification_merge_model=settings.ollama.model_clarification_merge,
            json_retry_count=settings.ollama.json_retry_count_last_qa,
            clarification_merge_json_retry_count=settings.ollama.json_retry_count_clarification_merge,
            enable_reminder_metadata_reply=settings.prompt_policy.last_qa_enable_reminder_metadata_reply,
            clarification_merge_enabled=settings.prompt_policy.clarification_merge_enabled,
        ),
        context_filter=ContextFilterConfig(
            conversation_min_confidence=settings.context_filter.conversation_min_confidence,
            conversation_approved_max_items=settings.context_filter.conversation_approved_max_items,
            conversation_duplicate_threshold=settings.context_filter.conversation_duplicate_threshold,
            knowledge_approved_max_items=settings.context_filter.knowledge_approved_max_items,
            knowledge_duplicate_threshold=settings.context_filter.knowledge_duplicate_threshold,
            low_information_text_patterns=settings.context_filter.low_information_text_patterns,
            reminder_approved_max_items=settings.context_filter.reminder_approved_max_items,
            semantic_context_judge_enabled=settings.context_filter.semantic_context_judge_enabled,
            context_filter_debug_diagnostics_enabled=settings.context_filter.context_filter_debug_diagnostics_enabled,
            low_information_min_chars=settings.context_filter.low_information_min_chars,
            conversation_retrieval_after_last_qa_enabled=settings.prompt_policy.conversation_retrieval_after_last_qa_enabled,
            conversation_retrieval_before_intent_enabled=settings.prompt_policy.conversation_retrieval_before_intent_enabled,
            expected_response_type_required=settings.prompt_policy.expected_response_type_required,
            expected_response_type_fallback_policy=settings.prompt_policy.expected_response_type_fallback_policy,
            clarification_expected_response_type_required=settings.prompt_policy.clarification_expected_response_type_required,
        ),
        default_timezone=settings.safety.default_timezone,
        reminder_duplicate_similarity_threshold=settings.safety.reminder_duplicate_similarity_threshold,
        reminder_duplicate_time_window_minutes=settings.safety.reminder_duplicate_time_window_minutes,
        confirmation_expiry_minutes=settings.safety.confirmation_expiry_minutes,
        confirmation_high_confidence_threshold=settings.safety.confirmation_high_confidence_threshold,
        human_in_the_loop_min_confidence=settings.prompt_policy.human_in_the_loop_min_confidence,
    )


def build_production_repository(settings: ProductionSettings) -> AssistantRepository:
    if os.environ.get("ASSISTANT_DATABASE_URL"):
        from .postgres_repository import PostgresRepository
        url = os.environ.get("ASSISTANT_DATABASE_URL")
        # Ensure it's not a sqlite url if it's meant to be postgres, 
        # but Alembic dummy run uses sqlite URL. We can just pass it directly.
        repository = PostgresRepository.create(
            url,
            pool_size=settings.database.pool_size if hasattr(settings.database, 'pool_size') else 10,
            max_overflow=settings.database.max_overflow if hasattr(settings.database, 'max_overflow') else 20
        )
        return repository
    
    from .database import SQLiteRepository
    directory = os.path.dirname(settings.database.path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    repository = SQLiteRepository.persistent(
        settings.database.path,
        enable_wal=settings.database.enable_wal,
        busy_timeout_ms=settings.database.busy_timeout_ms,
    )
    repository.initialize_schema()
    return repository


def build_production_pipeline(settings: ProductionSettings) -> AssistantPipeline:
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
        bm25_top_k=assistant_config.retrieval.bm25_top_k,
        chroma_top_k=assistant_config.retrieval.chroma_top_k,
    )
    action_detector = LLMActionDetector(
        llm,
        prompt_registry,
        min_confidence=settings.prompt_policy.action_min_confidence,
        risky_action_validation_enabled=settings.prompt_policy.risky_action_validation_enabled,
        risky_action_operations=settings.prompt_policy.risky_action_operations,
        risky_action_confidence_threshold=settings.prompt_policy.risky_action_confidence_threshold,
    )
    
    hard_rule_filter = HardRuleContextFilter(
        knowledge_min_confidence=settings.prompt_policy.context_filter_knowledge_min_confidence,
        allowed_reminder_statuses=settings.prompt_policy.context_filter_allowed_reminder_statuses,
        conversation_min_confidence=settings.context_filter.conversation_min_confidence,
        conversation_approved_max_items=settings.context_filter.conversation_approved_max_items,
        conversation_duplicate_threshold=settings.context_filter.conversation_duplicate_threshold,
        knowledge_approved_max_items=settings.context_filter.knowledge_approved_max_items,
        knowledge_duplicate_threshold=settings.context_filter.knowledge_duplicate_threshold,
        low_information_text_patterns=settings.context_filter.low_information_text_patterns,
        reminder_approved_max_items=settings.context_filter.reminder_approved_max_items,
        reminder_min_confidence=settings.context_filter.reminder_min_confidence,
        low_information_min_chars=settings.context_filter.low_information_min_chars,
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
        llm_validator=knowledge_llm_validator,
    )
    reminder_target_resolver = ReminderTargetResolver(
        config=assistant_config,
        llm_validator=reminder_llm_validator,
    )
    validated_action_builder = ValidatedActionBuilder(
        config=assistant_config,
        knowledge_resolver=knowledge_target_resolver,
        reminder_resolver=reminder_target_resolver,
    )

    import dataclasses
    gp_config = GeneralPurposeConfig(**dataclasses.asdict(settings.general_purpose))

    sub_branch_detector = GeneralSubBranchDetector(
        llm=llm, prompt_registry=prompt_registry,
    ) if gp_config.general_sub_branch_detector_enabled else None

    answer_gen_tool = AnswerGenerationTool(llm=llm, prompt_registry=prompt_registry)
    excel_tool = GenerateExcelTool(llm=llm, prompt_registry=prompt_registry, config=gp_config)
    pdf_tool = GeneratePDFTool(llm=llm, prompt_registry=prompt_registry, config=gp_config)
    pptx_tool = GeneratePPTXTool(llm=llm, prompt_registry=prompt_registry, config=gp_config)

    tool_registry = ContentToolRegistry(
        tools=[answer_gen_tool, excel_tool, pdf_tool, pptx_tool],
        config=gp_config,
    )

    content_composer = ReActContentComposer(
        registry=tool_registry, llm=llm,
        prompt_registry=prompt_registry, config=gp_config,
    ) if gp_config.content_composer_enabled else None

    general_hitl = LLMGeneralHITLStrategy(
        llm=llm, prompt_registry=prompt_registry,
        config=gp_config,
    ) if gp_config.hitl_supporting_question_enabled else None

    router = BranchRouter(
        {
            Intent.CLARIFICATION: ClarificationBranch(
                prompt_registry=prompt_registry,
                config=assistant_config,
                clarification_strategy=clarification_strategy,
            ),
            Intent.GENERAL_RESPONSE: GeneralResponseBranch(
                retriever=retriever,
                config=assistant_config,
                context_filter=context_filter,
                prompt_registry=prompt_registry,
                hitl_strategy=hitl_strategy,
                llm=llm,
                sub_branch_detector=sub_branch_detector,
                content_composer=content_composer,
                general_hitl_strategy=general_hitl,
                general_purpose_config=gp_config,
            ),
            Intent.KNOWLEDGE_FACTS: KnowledgeFactsBranch(
                config=assistant_config,
                action_detector=action_detector,
                clarification_strategy=clarification_strategy,
                prompt_registry=prompt_registry,
                validated_action_builder=validated_action_builder,
                retriever=retriever,
                context_filter=context_filter,
                llm=llm,
            ),
            Intent.REMINDER: ReminderBranch(
                config=assistant_config,
                action_detector=action_detector,
                clarification_strategy=clarification_strategy,
                prompt_registry=prompt_registry,
                reminder_supporting_strategy=reminder_supporting_strategy,
                validated_action_builder=validated_action_builder,
                retriever=retriever,
                context_filter=context_filter,
                llm=llm,
            ),
        }
    )
    return AssistantPipeline(
        config=assistant_config,
        last_qa_store=DiskCacheLastQAStore(
            settings.last_qa.path,
            ttl_seconds=settings.last_qa.ttl_seconds,
        ),
        query_rewriter=LLMQueryRewriter(
            llm=llm,
            prompt_registry=prompt_registry,
        ),
        last_qa_resolver=LLMLastQAResolver(
            llm=llm,
            config=assistant_config.last_qa,
            prompt_registry=prompt_registry,
        ),
        retriever=retriever,
        classifier=OllamaIntentClassifier(
            llm,
            prompt_registry,
            min_confidence=settings.prompt_policy.intent_min_confidence,
        ),
        router=router,
        context_filter=context_filter,
        bundler=ResponseBundler(prompt_registry),
        platform_selector=PlatformSelector(),
        chat_output=ChatOutput(),
        prompt_registry=prompt_registry,
    )
