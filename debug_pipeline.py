from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
import json
import os
import shlex
from typing import Any, Iterable

from assistant_rag.bm25_opensearch import OpenSearchBM25Index
from assistant_rag.autoscan import ReminderAutoscan
from assistant_rag.action_detection import LLMActionDetector
from assistant_rag.context_filter import HardRuleContextFilter, TwoLayerContextFilter
from assistant_rag.branches import (
    BranchRouter,
    ClarificationBranch,
    GeneralResponseBranch,
    KnowledgeFactsBranch,
    ReminderBranch,
)
from assistant_rag.generation import LLMHumanInTheLoopStrategy, LLMClarificationStrategy, LLMReminderSupportingStrategy
from assistant_rag.branch_orchestration import KnowledgeTargetResolver, ReminderTargetResolver, ValidatedActionBuilder
from assistant_rag.retrieval_validation import KnowledgeRetrievalValidationStrategy, ReminderRetrievalValidationStrategy
from assistant_rag.bundler import ChatOutput, ResponseBundler
from assistant_rag.classification import LLMLastQAResolver, LLMQueryRewriter
from assistant_rag.config import (
    AssistantConfig,
    AutoscanConfig,
    ClassificationConfig,
    OutboxConfig,
    RetrievalConfig,
    MutationPolicyConfig,
    QuestionGenerationConfig,
    LastQAConfig,
)
from assistant_rag.contracts import (
    BranchResult,
    BundledResponse,
    ChatRequest,
    Intent,
    PipelineContext,
    RetrievalResult,
)
from assistant_rag.chroma_index import ChromaPersistentVectorIndex
from assistant_rag.database import SQLRepository
from assistant_rag.embeddings import HashEmbeddingClient
from assistant_rag.indexing import BackgroundIndexer
from assistant_rag.last_qa import InMemoryLastQAStore
from assistant_rag.llm import LLMTask, OllamaIntentClassifier, OllamaLLMClient, OllamaModelRouter
from assistant_rag.pipeline import AssistantPipeline
from assistant_rag.platform import PlatformSelector
from assistant_rag.prompts import DEFAULT_PROMPT_REGISTRY
from assistant_rag.retrieval import HybridRetriever
from assistant_rag.settings import ChromaSettings, OpenSearchSettings, ProductionSettings


class DebugReranker:
    def rerank(self, query: str, results: Iterable[RetrievalResult]) -> list[RetrievalResult]:
        return sorted(results, key=lambda item: item.rerank_score, reverse=True)


@dataclass
class DebugRuntime:
    repository: SQLRepository
    pipeline: AssistantPipeline
    bm25: OpenSearchBM25Index
    chroma: ChromaPersistentVectorIndex
    config: AssistantConfig
    settings: ProductionSettings
    model_router: OllamaModelRouter
    llm: OllamaLLMClient


def build_config(settings: ProductionSettings) -> AssistantConfig:
    return AssistantConfig(
        retrieval=RetrievalConfig(
            conversation_min_confidence=settings.debug.conversation_min_confidence,
            knowledge_min_confidence=settings.debug.knowledge_min_confidence,
            max_results=settings.debug.max_results,
            rrf_k=settings.debug.rrf_k,
            lexical_weight=settings.debug.lexical_weight,
            semantic_weight=settings.debug.semantic_weight,
            rerank_candidate_limit=settings.debug.rerank_candidate_limit,
        ),
        outbox=OutboxConfig(
            max_attempts=settings.debug.outbox_max_attempts,
            batch_size=settings.debug.outbox_batch_size,
            retry_backoff_seconds=settings.debug.outbox_retry_backoff_seconds,
            processing_timeout_seconds=settings.debug.outbox_processing_timeout_seconds,
        ),
        autoscan=AutoscanConfig(interval_seconds=settings.debug.autoscan_interval_seconds),
        classification=ClassificationConfig(
            intent_keywords={
                Intent.KNOWLEDGE_FACTS.value: ("remember", "knowledge"),
                Intent.REMINDER.value: ("remind", "reminder"),
            }
        ),
        mutation_policy=MutationPolicyConfig(
            partial_execution_policy=settings.prompt_policy.mutation_partial_execution_policy,
            knowledge_relevance_threshold=settings.prompt_policy.knowledge_target_relevance_threshold,
            knowledge_ambiguity_margin=settings.prompt_policy.knowledge_target_ambiguity_margin,
            knowledge_not_found_policy=settings.prompt_policy.knowledge_target_not_found_policy,
            unsupported_action_policy=settings.prompt_policy.unsupported_action_policy,
        ),
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
        reminder_resolver=settings.reminder_resolver,
        retrieval_validation=settings.retrieval_validation,
    )


def build_runtime() -> DebugRuntime:
    settings = debug_settings()
    os.makedirs(os.path.dirname(settings.debug.db_path), exist_ok=True)
    repository = SQLRepository.persistent(
        settings.debug.db_path,
        enable_wal=settings.database.enable_wal,
        busy_timeout_ms=settings.database.busy_timeout_ms,
    )
    repository.initialize_schema()
    config = build_config(settings)
    bm25 = OpenSearchBM25Index(settings.opensearch)
    bm25.initialize()
    chroma = ChromaPersistentVectorIndex(settings.chroma, HashEmbeddingClient())
    prompt_registry = DEFAULT_PROMPT_REGISTRY
    model_router = OllamaModelRouter(settings.ollama)
    llm = OllamaLLMClient(settings.ollama, model_router)
    retriever = HybridRetriever(
        bm25=bm25,
        chroma=chroma,
        reranker=DebugReranker(),
        rrf_k=config.retrieval.rrf_k,
        lexical_weight=config.retrieval.lexical_weight,
        semantic_weight=config.retrieval.semantic_weight,
        rerank_candidate_limit=config.retrieval.rerank_candidate_limit,
    )
    action_detector = LLMActionDetector(
        llm=llm,
        prompt_registry=prompt_registry,
        min_confidence=settings.prompt_policy.action_min_confidence,
        risky_action_validation_enabled=settings.prompt_policy.risky_action_validation_enabled,
        risky_action_operations=settings.prompt_policy.risky_action_operations,
        risky_action_confidence_threshold=settings.prompt_policy.risky_action_confidence_threshold,
    )
    
    hard_rule_filter = HardRuleContextFilter(
        knowledge_min_confidence=settings.prompt_policy.context_filter_knowledge_min_confidence,
        allowed_reminder_statuses=settings.prompt_policy.context_filter_allowed_reminder_statuses,
    )
    context_filter = TwoLayerContextFilter(
        hard_rule_filter=hard_rule_filter,
        llm_judge=None,
    )

    clarification_strategy = LLMClarificationStrategy(
        llm=llm,
        prompt_registry=prompt_registry,
        config=config.question_generation,
    )
    hitl_strategy = LLMHumanInTheLoopStrategy(
        llm=llm,
        prompt_registry=prompt_registry,
        config=config.question_generation,
    )
    reminder_supporting_strategy = LLMReminderSupportingStrategy(
        llm=llm,
        prompt_registry=prompt_registry,
        config=config.question_generation,
    )

    knowledge_llm_validator = KnowledgeRetrievalValidationStrategy(
        config=config.retrieval_validation,
        llm=llm,
        prompts=prompt_registry,
    )
    reminder_llm_validator = ReminderRetrievalValidationStrategy(
        config=config.retrieval_validation,
        llm=llm,
        prompts=prompt_registry,
    )

    knowledge_target_resolver = KnowledgeTargetResolver(
        retriever=retriever,
        config=config,
        repository=repository,
        llm_validator=knowledge_llm_validator,
    )
    reminder_target_resolver = ReminderTargetResolver(
        repository=repository,
        config=config,
        llm_validator=reminder_llm_validator,
    )
    validated_action_builder = ValidatedActionBuilder(
        config=config,
        knowledge_resolver=knowledge_target_resolver,
        reminder_resolver=reminder_target_resolver,
    )

    router = BranchRouter(
        {
            Intent.CLARIFICATION: ClarificationBranch(
                prompt_registry=prompt_registry,
                config=config,
                clarification_strategy=clarification_strategy,
            ),
            Intent.GENERAL_RESPONSE: GeneralResponseBranch(
                repository,
                retriever,
                config,
                context_filter=context_filter,
                llm=llm,
                prompt_registry=prompt_registry,
                hitl_strategy=hitl_strategy,
            ),
            Intent.KNOWLEDGE_FACTS: KnowledgeFactsBranch(
                repository=repository,
                config=config,
                action_detector=action_detector,
                clarification_strategy=clarification_strategy,
                prompt_registry=prompt_registry,
                validated_action_builder=validated_action_builder,
            ),
            Intent.REMINDER: ReminderBranch(
                repository=repository,
                config=config,
                action_detector=action_detector,
                clarification_strategy=clarification_strategy,
                reminder_supporting_strategy=reminder_supporting_strategy,
                prompt_registry=prompt_registry,
                validated_action_builder=validated_action_builder,
            ),
        }
    )
    pipeline = AssistantPipeline(
        config=config,
        last_qa_store=InMemoryLastQAStore(),
        query_rewriter=LLMQueryRewriter(llm, prompt_registry),
        last_qa_resolver=LLMLastQAResolver(
            llm=llm,
            config=config.last_qa,
            prompt_registry=prompt_registry,
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
    runtime = DebugRuntime(
        repository=repository,
        pipeline=pipeline,
        bm25=bm25,
        chroma=chroma,
        config=config,
        settings=settings,
        model_router=model_router,
        llm=llm,
    )
    rebuild_indexes(runtime)
    return runtime


def debug_settings() -> ProductionSettings:
    base = ProductionSettings.from_env()
    return replace(
        base,
        opensearch=OpenSearchSettings(
            url=base.opensearch.url,
            username=base.opensearch.username,
            password=base.opensearch.password,
            verify_certs=base.opensearch.verify_certs,
            conversation_index=base.debug.opensearch_conversation_index,
            knowledge_index=base.debug.opensearch_knowledge_index,
            conversation_write_alias=base.debug.opensearch_conversation_write_alias,
            knowledge_write_alias=base.debug.opensearch_knowledge_write_alias,
            reminder_context_alias=base.debug.opensearch_reminder_context_alias,
            analyzer_name=base.opensearch.analyzer_name,
            timeout_seconds=base.opensearch.timeout_seconds,
            max_retries=base.opensearch.max_retries,
        ),
        chroma=ChromaSettings(
            path=base.debug.chroma_path,
            host=base.chroma.host,
            port=base.chroma.port,
            conversation_collection=base.debug.chroma_conversation_collection,
            knowledge_collection=base.debug.chroma_knowledge_collection,
        ),
    )


def log_step(name: str, payload: Any) -> None:
    print(f"\n--- {name} ---")
    print(to_jsonable(payload))


def log_model_decision(runtime: DebugRuntime, stage: str, task: Any) -> None:
    decision = runtime.model_router.decision_for_task(task)
    log_step(
        f"{stage} Model Decision",
        {
            "task": decision.task.value,
            "model": decision.model,
            "temperature": decision.temperature,
            "timeout_seconds": decision.timeout_seconds,
            "reason_summary": decision.reason_summary,
        },
    )


def log_llm_stage_status(runtime: DebugRuntime, stage: str, task: LLMTask) -> None:
    error = runtime.llm.last_error_by_task.get(task)
    if error:
        log_step(
            f"{stage} LLM Fallback",
            {
                "task": task.value,
                "fallback_used": True,
                "error": error,
            },
        )


def to_jsonable(value: Any) -> str:
    try:
        return json.dumps(value, indent=2, default=str)
    except TypeError:
        return str(value)


def summarize_results(results: list[RetrievalResult]) -> list[dict[str, Any]]:
    return [
        {
            "entity_type": item.entity_type,
            "entity_id": item.entity_id,
            "confidence": item.confidence,
            "rerank_score": item.rerank_score,
            "text": item.payload.get("text"),
        }
        for item in results
    ]


def summarize_branch(branch_result: BranchResult) -> dict[str, Any]:
    return {
        "response_type": branch_result.response_type.value,
        "normal_response_text": branch_result.normal_response_text,
        "clarification_question": branch_result.clarification_question,
        "human_supporting_questions": branch_result.human_supporting_questions,
        "reminder_supporting_question": branch_result.reminder_supporting_question,
        "knowledge_operation_results": [
            result.__dict__ for result in branch_result.knowledge_operation_results
        ],
        "reminder_operation_results": [
            result.__dict__ for result in branch_result.reminder_operation_results
        ],
        "database_write_result": branch_result.database_write_result,
        "indexing_job_result": branch_result.indexing_job_result,
        "linked_topic_id": branch_result.linked_topic_id,
        "linked_hop_id": branch_result.linked_hop_id,
        "human_in_the_loop_result": getattr(branch_result, "human_in_the_loop_result", None),
    }


def summarize_bundled(response: BundledResponse) -> dict[str, Any]:
    return {
        "final_chat_text": response.final_chat_text,
        "response_type": response.response_type.value,
        "last_qa_state": response.last_qa_state.__dict__,
        "persistence_instructions": response.persistence_instructions,
    }


def handle_with_logs(runtime: DebugRuntime, request: ChatRequest) -> BundledResponse:
    pipeline = runtime.pipeline
    log_step("0. Incoming ChatRequest", request.__dict__)

    log_model_decision(runtime, "1. Query Rewrite", LLMTask.QUERY_REWRITE)
    rewritten = pipeline.query_rewriter.rewrite(request.raw_query)
    log_llm_stage_status(runtime, "1. Query Rewrite", LLMTask.QUERY_REWRITE)
    log_step("1. Query Rewrite", {"raw_query": request.raw_query, "rewritten_query": rewritten})

    last_state = pipeline.last_qa_store.get(request.user_id)
    log_step("2. Load Last-QA State", None if last_state is None else last_state.__dict__)

    if last_state is not None:
        log_model_decision(runtime, "3. Last-QA Resolver", LLMTask.LAST_QA)
    resolution = pipeline.last_qa_resolver.resolve(request, rewritten, last_state)
    if last_state is not None:
        log_llm_stage_status(runtime, "3. Last-QA Resolver", LLMTask.LAST_QA)
    log_step(
        "3. Last-QA Resolver",
        {
            "rewritten_query": resolution.rewritten_query,
            "skip_broad_retrieval": resolution.skip_broad_retrieval,
            "state": None if resolution.state is None else resolution.state.__dict__,
        },
    )

    conversation_results: list[RetrievalResult] = []
    if not resolution.skip_broad_retrieval:
        conversation_results = pipeline.retriever.retrieve_conversation(
            user_id=request.user_id,
            query=resolution.rewritten_query,
            limit=pipeline.config.retrieval.max_results,
            min_confidence=pipeline.config.retrieval.conversation_min_confidence,
        )
    log_step("4. Optional Conversation Retrieval", summarize_results(conversation_results))

    log_model_decision(runtime, "5. Intent Classifier", LLMTask.INTENT)
    intent = pipeline.classifier.classify(request, resolution.rewritten_query)
    log_llm_stage_status(runtime, "5. Intent Classifier", LLMTask.INTENT)
    log_step("5. Intent Classifier", {"intent": intent.value})


    context = PipelineContext(
        request=request,
        rewritten_query=resolution.rewritten_query,
        last_qa_state=resolution.state,
        conversation_results=conversation_results,
        intent=intent,
    )
    log_step(
        "7. PipelineContext",
        {
            "rewritten_query": context.rewritten_query,
            "intent": context.intent.value,
            "conversation_result_count": len(context.conversation_results),
            "metadata": context.request.metadata,
        },
    )

    if context.intent is Intent.GENERAL_RESPONSE:
        log_model_decision(runtime, "8. General Answer Generation", LLMTask.ANSWER)
    branch_result = pipeline.router.route(context)
    if context.intent is Intent.GENERAL_RESPONSE:
        log_llm_stage_status(runtime, "8. General Answer Generation", LLMTask.ANSWER)
    log_step("8. Branch Router Result", summarize_branch(branch_result))

    bundled = pipeline.bundler.bundle(
        request=request,
        rewritten_query=resolution.rewritten_query,
        branch_result=branch_result,
    )
    log_step("9. Response Bundler", summarize_bundled(bundled))

    platform_payload = pipeline.platform_selector.select(bundled, request)
    log_step("10. Platform Selector", platform_payload)

    pipeline.last_qa_store.save(request.user_id, bundled.last_qa_state)
    log_step("11. Save Last-QA", bundled.last_qa_state.__dict__)

    emitted = pipeline.chat_output.emit(bundled)
    log_step("12. Chat Output", {"emitted_text": emitted})

    indexed = process_outbox(runtime)
    log_step("13. indexing_outbox Processing", {"indexed_jobs": indexed})

    return bundled


def make_request(line: str, *, user_id: str) -> ChatRequest:
    if line.startswith("ask "):
        return ChatRequest(user_id=user_id, raw_query=line.removeprefix("ask ").strip())
    if line.startswith("remember "):
        fact = line.removeprefix("remember ").strip()
        return ChatRequest(
            user_id=user_id,
            raw_query=line,
            metadata={
                "intent": Intent.KNOWLEDGE_FACTS.value,
                "knowledge_actions": [
                    {
                        "action": "add",
                        "topic_title": "Debug Facts",
                        "text": fact,
                        "summary": DEFAULT_PROMPT_REGISTRY.message("knowledge_added"),
                    }
                ],
            },
        )
    if line.startswith("remind "):
        subject, reminder_time = parse_reminder(line.removeprefix("remind ").strip())
        return ChatRequest(
            user_id=user_id,
            raw_query=line,
            metadata={
                "intent": Intent.REMINDER.value,
                "reminder_actions": [
                    {
                        "action": "add",
                        "reminder_time": reminder_time,
                        "raw_reminder": line,
                        "reminder_summary": subject,
                        "subject": subject,
                        "summary": DEFAULT_PROMPT_REGISTRY.message(
                            "reminder_changed", action="add"
                        ),
                    }
                ],
            },
        )
    return ChatRequest(user_id=user_id, raw_query=line)


def parse_reminder(text: str) -> tuple[str, str]:
    if " at " not in text:
        return text, (datetime.now(UTC) + timedelta(days=1)).isoformat()
    subject, reminder_time = text.rsplit(" at ", 1)
    return subject.strip(), reminder_time.strip()


def process_outbox(runtime: DebugRuntime) -> int:
    indexer = BackgroundIndexer(
        connection=runtime.repository.connection,
        bm25=runtime.bm25,
        chroma=runtime.chroma,
        config=runtime.config.outbox,
    )
    return indexer.process_pending()


def rebuild_indexes(runtime: DebugRuntime) -> None:
    indexer = BackgroundIndexer(
        connection=runtime.repository.connection,
        bm25=runtime.bm25,
        chroma=runtime.chroma,
        config=runtime.config.outbox,
    )
    indexer.rebuild_from_sql()


def show_tables(runtime: DebugRuntime) -> None:
    log_step(
        "SQL Table Counts",
        {
            table_name: runtime.repository.table_count(table_name)
            for table_name in (
                "conversation_topics",
                "conversation_hops",
                "knowledge_topics",
                "knowledge_chunks",
                "reminders",
                "reminder_notifications",
                "indexing_outbox",
            )
        },
    )


def show_notifications(runtime: DebugRuntime) -> None:
    log_step(
        "Reminder Notifications",
        runtime.repository.list_notifications(user_id=runtime.settings.debug.user_id),
    )


def reset_debug_database() -> DebugRuntime:
    settings = debug_settings()
    if os.path.exists(settings.debug.db_path):
        os.remove(settings.debug.db_path)
    return build_runtime()


def print_help() -> None:
    print(
        """
Commands:
  ask <question>              Run a normal question through every pipeline stage.
  remember <fact>             Add a knowledge fact and index it through the outbox.
  remind <subject> at <time>  Add a reminder. Time should be ISO text.
  autoscan [time]             Run SQL-only reminder autoscan and print notifications.
  notifications               Print reminder notification rows.
  tables                      Print SQL table counts.
  rebuild                     Rebuild local derived indexes from SQL.
  reset                       Delete only the debug SQLite DB and start fresh.
  help                        Show this help.
  quit                        Exit.

Free text without a command is treated like: ask <text>
""".strip()
    )


def main() -> None:
    try:
        runtime = build_runtime()
    except Exception as exc:
        print("Debug pipeline storage bootstrap failed.")
        print("SQL files and Chroma collections can be created by this project.")
        print("OpenSearch indexes can also be created automatically, but the OpenSearch server must already be running.")
        print(f"Default expected URL: {ProductionSettings.from_env().opensearch.url}")
        print(f"Error: {exc}")
        return
    print("Debug pipeline runner")
    print(f"Storage: {runtime.settings.debug.db_path}")
    print("This does not start Streamlit or FastAPI.")
    print(f"OpenSearch URL: {runtime.settings.opensearch.url}")
    print(f"OpenSearch indexes: {runtime.settings.opensearch.conversation_index}, {runtime.settings.opensearch.knowledge_index}")
    print(f"Chroma storage: {runtime.settings.chroma.path}")
    print_help()
    while True:
        try:
            line = input("\nYou> ").strip()
        except EOFError:
            print()
            break
        if not line:
            continue
        parts = shlex.split(line)
        command = parts[0].casefold() if parts else ""
        if command in {"quit", "exit"}:
            break
        if command == "help":
            print_help()
            continue
        if command == "tables":
            show_tables(runtime)
            continue
        if command == "notifications":
            show_notifications(runtime)
            continue
        if command == "rebuild":
            rebuild_indexes(runtime)
            log_step("Rebuild", "Rebuilt local derived indexes from authoritative SQL.")
            continue
        if command == "reset":
            runtime = reset_debug_database()
            log_step("Reset", f"Deleted and recreated {runtime.settings.debug.db_path}.")
            continue
        if command == "autoscan":
            now_value = parts[1] if len(parts) > 1 else datetime.now(UTC).isoformat()
            notified = ReminderAutoscan(runtime.repository).scan_due(now_value=now_value)
            log_step("Reminder Autoscan", {"now": now_value, "notified_reminders": notified})
            show_notifications(runtime)
            continue

        response = handle_with_logs(
            runtime,
            make_request(line, user_id=runtime.settings.debug.user_id),
        )
        print(f"\nAssistant> {response.final_chat_text}")


if __name__ == "__main__":
    main()
