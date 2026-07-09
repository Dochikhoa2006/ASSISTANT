# Runtime File Guide

This guide explains the scripts and modules that affect how the assistant runs.
Test files are intentionally excluded unless they are useful as examples.

## Main Entry Points

- `debug_pipeline.py`
  - Fast terminal debug runner for asking your own questions without starting the whole app.
  - Automatically creates/uses debug SQL, debug OpenSearch indexes, and debug ChromaDB collections.
  - Prints every runtime stage: rewrite, Last-QA, model decisions, retrieval, intent/action detection, branch routing, bundling, platform output, Last-QA save, chat output, and outbox indexing.
  - Best first command: `./run`.

- `streamlit_app.py`
  - Streamlit chat UI.
  - Builds the production pipeline from `ProductionSettings`.
  - Use this when you want a browser UI instead of terminal.

- `run`
  - Shell wrapper for `debug_pipeline.py`.
  - Starts OpenSearch, ChromaDB, and Ollama through `docker-compose.runtime.yml`.
  - Uses `.venv/bin/python` when present, otherwise falls back to `python3`.
  - Use `./run` for terminal debugging and `./run streamlit` for the Streamlit UI.

## API and Worker Runtime

- `assistant_rag/api.py`
  - FastAPI application factory.
  - Exposes `/chat`, `/health`, notification APIs, reminder APIs, and notification WebSocket support.

- `assistant_rag/worker.py`
  - Production worker loop.
  - Runs the SQL outbox indexer and reminder autoscan on a configured interval.

- `assistant_rag/indexing.py`
  - Background indexing engine.
  - Reads `indexing_outbox`, loads authoritative SQL text, and syncs OpenSearch/ChromaDB derived caches.
  - Handles retries, stale processing jobs, and rebuild-from-SQL.

- `assistant_rag/autoscan.py`
  - SQL-only reminder autoscan and reminder reply context loading.
  - Finds due scheduled reminders, creates UI notifications, and marks reminders as notified.

## Core Chat Pipeline

- `assistant_rag/pipeline.py`
  - Ordered assistant pipeline.
  - Runs query rewrite, Last-QA resolution, optional retrieval, intent/action detection, branch routing, bundling, platform selection, Last-QA save, and chat output.

- `assistant_rag/branches.py`
  - Intent branch implementations.
  - Handles clarification, general response, knowledge actions, and reminder actions.
  - Knowledge and reminder mutations go through repository transactions.

- `assistant_rag/bundler.py`
  - Final response assembly layer.
  - Collects branch results and creates the final user-facing text plus Last-QA metadata.

- `assistant_rag/classification.py`
  - Query rewrite, Last-QA resolver, LLM-backed rewrite/resolution variants, and local keyword fallback classifier.
  - Production and debug wiring use the LLM-backed variants with deterministic fallback.

- `assistant_rag/action_detection.py`
  - Schema-validated LLM action detector.
  - Converts natural language into structured knowledge/reminder action metadata before branch execution.

- `assistant_rag/llm.py`
  - Ollama-backed LLM client, model-decision router, JSON parsing/validation, and LLM intent classifier.
  - Routes fast/balanced/accurate/writing tasks to configured models.

- `assistant_rag/prompts.py`
  - Central prompt registry and user-facing message catalog.
  - Provides stage-specific system prompts, safe runtime context rendering, Gmail/platform policy, and shared SQL/retrieval/mutation safety rules.

- `assistant_rag/platform.py`
  - Platform output selection.
  - Lets final bundled responses be formatted for a channel such as plain text or future integrations.

## Data and Persistence

- `assistant_rag/database.py`
  - SQL source-of-truth repository.
  - Defines SQLite-backed persistence, transactions, table schema initialization, knowledge mutations, reminder mutations, conversation hops, notification updates, and reminder replies.

- `assistant_rag/contracts.py`
  - Shared dataclasses and enums used across the runtime.
  - Defines requests, responses, intents, operation results, retrieval results, and pipeline context.

- `assistant_rag/last_qa.py`
  - Last-QA state stores.
  - Provides in-memory and disk-backed storage for the latest temporary conversation context.

- `assistant_rag/settings.py`
  - Production runtime settings.
  - Reads database, retrieval, OpenSearch, ChromaDB, embedding, reranker, LLM, worker, UI, and API configuration from environment variables.

- `assistant_rag/config.py`
  - Internal configuration contracts passed into runtime components.
  - Keeps thresholds, outbox limits, classification keywords, and retrieval tuning structured.

## Retrieval and ML Adapters

- `assistant_rag/retrieval.py`
  - Hybrid retrieval orchestration.
  - Runs BM25/OpenSearch search, ChromaDB semantic search, RRF merge, reranking, entity filtering, and confidence gating.

- `assistant_rag/bm25_opensearch.py`
  - OpenSearch BM25 derived-cache adapter.
  - Creates indexes/aliases, searches by user, upserts conversation/knowledge documents, and rejects reminder entity indexing.

- `assistant_rag/chroma_index.py`
  - ChromaDB vector derived-cache adapter.
  - Stores embeddings for conversation hops and knowledge chunks only.

- `assistant_rag/embeddings.py`
  - SentenceTransformer embedding adapter.
  - Converts text into vectors for ChromaDB.

- `assistant_rag/reranking.py`
  - Cross-encoder reranker adapter.
  - Supports local SentenceTransformer reranking or a configured HTTP reranker service.

## Operations and Validation

- `assistant_rag/cli.py`
  - Command-line maintenance entrypoint.
  - Supports DB init, service checks, DB/index inspection, and derived-index rebuild.

- `assistant_rag/consistency.py`
  - Post-migration and rollout consistency checks.
  - Detects broken topic/hop/chunk/reminder/outbox relationships.

- `assistant_rag/validation.py`
  - Deterministic validation rule registry.
  - Provides reusable checks such as ownership, entity type, active status, relevance threshold, and action compatibility.

- `assistant_rag/health.py`
  - Health probe registry.
  - Lets runtime dependencies expose structured health reports.

- `assistant_rag/evaluation.py`
  - Retrieval evaluation helpers.
  - Computes precision, recall, MRR, and nDCG for threshold/RRF tuning.

## Database Migration Runtime

- `alembic/env.py`
  - Alembic migration environment.
  - Reads `ASSISTANT_DATABASE_URL` and runs migrations against the configured database.

- `alembic/versions/0001_postgresql_baseline.py`
  - PostgreSQL production baseline schema migration.
  - Creates the production tables, constraints, and indexes.

- `alembic.ini`
  - Alembic configuration file.
  - Points Alembic at the migration folder; runtime DB URL should come from `ASSISTANT_DATABASE_URL`.
