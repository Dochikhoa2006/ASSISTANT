import sys

def replace_in_file():
    with open("debug_pipeline.py", "r") as f:
        lines = f.readlines()
        
    start_idx = -1
    for i, line in enumerate(lines):
        if line.startswith("def scenario_model_routing_policy("):
            start_idx = i
            break
            
    end_idx = -1
    for i in range(start_idx + 1, len(lines)):
        if line.startswith("def scenario_"):
            end_idx = i - 2
            break
            
    if start_idx == -1 or end_idx == -1:
        print("Could not find bounds")
        sys.exit(1)
        
    new_def = """def scenario_model_routing_policy(settings: ProductionSettings) -> ScenarioResult:
    default_settings = ProductionSettings()
    router = OllamaModelRouter(default_settings.ollama)
    expected_models = {
        LLMTask.QUERY_REWRITE: "qwen2.5:1.5b",
        LLMTask.LAST_QA: "qwen2.5:3b",
        LLMTask.INTENT: "qwen2.5:3b",
        LLMTask.ACTION_EXTRACTION: "qwen3.5:4b",
        LLMTask.GENERATE_CLARIFICATION: "qwen3.5:4b",
        LLMTask.GENERATE_HUMAN_SUPPORTING: "qwen3.5:4b",
        LLMTask.GENERATE_REMINDER_SUPPORTING: "qwen3.5:4b",
        LLMTask.CLARIFICATION_MERGE: "qwen3.5:4b",
        LLMTask.ANSWER: "qwen2.5:7b",
        LLMTask.WRITING: "qwen3.5:4b",
        LLMTask.RISKY_ACTION: "qwen2.5:3b",
        LLMTask.RETRIEVAL_VALIDATION: "qwen2.5:7b",
        LLMTask.GENERAL_SUB_BRANCH_DETECTION: "qwen2.5:1.5b",
        LLMTask.CONTENT_COMPOSER_REACT: "qwen2.5:1.5b",
        LLMTask.ACTION_PLANNING: "qwen2.5:7b",
    }
    actual_models = {task: router.model_for_task(task) for task in expected_models}
    mismatches = {
        task.value: {"expected": expected_model, "actual": actual_models[task]}
        for task, expected_model in expected_models.items()
        if actual_models[task] != expected_model
    }
    if mismatches:
        raise AssertionError(f"model routing mismatches: {mismatches}")
    
    # Check some basic policy settings
    policy_values = {
        "bm25_top_k": default_settings.retrieval.bm25_top_k,
        "chroma_top_k": default_settings.retrieval.chroma_top_k,
        "rrf_k": default_settings.retrieval.rrf_k,
        "reranker_top_k": default_settings.retrieval.rerank_candidate_limit,
        "reranker_min_score": default_settings.reranker.min_score,
        "reranker_batch_size": default_settings.reranker.batch_size,
        "final_context_top_k": default_settings.retrieval.max_results,
        "retrieval_min_confidence": default_settings.retrieval.min_confidence,
        "knowledge_context_min_confidence": default_settings.retrieval.knowledge_min_confidence,
        "conversation_context_min_confidence": default_settings.retrieval.conversation_min_confidence,
        "last_qa_min_confidence": default_settings.prompt_policy.last_qa_min_confidence,
        "last_qa_clarification_merge_min_confidence": default_settings.prompt_policy.clarification_merge_min_confidence,
        "last_qa_skip_broad_retrieval_min_confidence": default_settings.prompt_policy.last_qa_skip_broad_retrieval_min_confidence,
        "action_min_confidence": default_settings.prompt_policy.action_min_confidence,
        "risky_action_confidence_threshold": default_settings.prompt_policy.risky_action_confidence_threshold,
        "reminder_candidate_limit": default_settings.reminder_resolver.reminder_target_candidate_limit,
        "reminder_target_min_score": default_settings.reminder_resolver.reminder_target_relevance_threshold,
        "reminder_target_ambiguity_margin": default_settings.reminder_resolver.reminder_target_ambiguity_margin,
        "reminder_fuzzy_match_threshold": default_settings.reminder_resolver.reminder_fuzzy_match_threshold,
        "reminder_context_min_confidence": default_settings.context_filter.reminder_min_confidence,
        "knowledge_chunk_size_tokens": default_settings.knowledge_chunks.chunk_size_tokens,
        "knowledge_chunk_overlap_tokens": default_settings.knowledge_chunks.chunk_overlap_tokens,
        "knowledge_min_chunk_tokens": default_settings.knowledge_chunks.min_chunk_tokens,
        "knowledge_max_chunk_tokens": default_settings.knowledge_chunks.max_chunk_tokens,
        "embedding_batch_size": default_settings.embeddings.batch_size,
        "embedding_max_length": default_settings.embeddings.max_length,
        "outbox_batch_size": default_settings.worker.outbox_batch_size,
        "outbox_max_retries": default_settings.worker.outbox_max_attempts,
        "outbox_retry_backoff_seconds": default_settings.worker.outbox_retry_backoff_seconds,
        "outbox_stale_processing_after_seconds": default_settings.worker.outbox_processing_timeout_seconds,
        "outbox_worker_interval_seconds": default_settings.worker.outbox_worker_interval_seconds,
        "reminder_autoscan_interval_seconds": default_settings.worker.autoscan_interval_seconds,
    }
    expected_policy_values = {
        "bm25_top_k": 30,
        "chroma_top_k": 30,
        "rrf_k": 60,
        "reranker_top_k": 20,
        "reranker_min_score": 0.35,
        "reranker_batch_size": 16,
        "final_context_top_k": 8,
        "retrieval_min_confidence": 0.25,
        "knowledge_context_min_confidence": 0.35,
        "conversation_context_min_confidence": 0.40,
        "last_qa_min_confidence": 0.75,
        "last_qa_clarification_merge_min_confidence": 0.80,
        "last_qa_skip_broad_retrieval_min_confidence": 0.85,
        "action_min_confidence": 0.70,
        "risky_action_confidence_threshold": 0.85,
        "reminder_candidate_limit": 20,
        "reminder_target_min_score": 0.72,
        "reminder_target_ambiguity_margin": 0.12,
        "reminder_fuzzy_match_threshold": 0.78,
        "reminder_context_min_confidence": 0.50,
        "knowledge_chunk_size_tokens": 700,
        "knowledge_chunk_overlap_tokens": 100,
        "knowledge_min_chunk_tokens": 80,
        "knowledge_max_chunk_tokens": 1000,
        "embedding_batch_size": 32,
        "embedding_max_length": 8192,
        "outbox_batch_size": 50,
        "outbox_max_retries": 5,
        "outbox_retry_backoff_seconds": 30,
        "outbox_stale_processing_after_seconds": 300,
        "outbox_worker_interval_seconds": 5,
        "reminder_autoscan_interval_seconds": 60,
    }
    policy_mismatches = {
        key: {"expected": expected_policy_values[key], "actual": policy_values[key]}
        for key in expected_policy_values
        if policy_values[key] != expected_policy_values[key]
    }
    if policy_mismatches:
        raise AssertionError(f"policy default mismatches: {policy_mismatches}")
    if default_settings.prompt_policy.risky_action_operations != ("delete", "modify", "turn_off"):
        raise AssertionError(f"risky action operations mismatch: {default_settings.prompt_policy.risky_action_operations}")
    if not default_settings.retrieval_validation.reminder_llm_validation_enabled:
        raise AssertionError("reminder LLM validation should be enabled by default")
    if not default_settings.embeddings.normalize_embeddings:
        raise AssertionError("embedding normalization should be enabled")
    if default_settings.embeddings.model_name != "BAAI/bge-m3":
        raise AssertionError(f"embedding model mismatch: {default_settings.embeddings.model_name}")
    if router.decision_for_task(LLMTask.ANSWER).num_ctx != 4096:
        raise AssertionError("answer task should use writing context window")
    
    expected_timeouts = {
        LLMTask.QUERY_REWRITE: 120.0,
        LLMTask.LAST_QA: 60.0,
        LLMTask.INTENT: 120.0,
        LLMTask.ACTION_EXTRACTION: 120.0,
        LLMTask.RISKY_ACTION: 120.0,
        LLMTask.RETRIEVAL_VALIDATION: 120.0,
        LLMTask.ANSWER: 120.0,
        LLMTask.WRITING: 120.0,
    }
    timeout_mismatches = {
        task.value: {"expected": expected_timeout, "actual": router.decision_for_task(task).timeout_seconds}
        for task, expected_timeout in expected_timeouts.items()
        if router.decision_for_task(task).timeout_seconds != expected_timeout
    }
    if timeout_mismatches:
        raise AssertionError(f"LLM timeout mismatches: {timeout_mismatches}")
    return ScenarioResult("model_routing_policy", True, "LLM task routing, retrieval, action, context, and worker defaults match policy")
"""
    final_lines = lines[:start_idx] + [new_def + "\n"] + lines[end_idx:]
    with open("debug_pipeline.py", "w") as f:
        f.write("".join(final_lines))

replace_in_file()
