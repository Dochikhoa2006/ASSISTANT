# SQL-First RAG Assistant Architecture

This workspace contains a greenfield implementation scaffold for the requested
assistant architecture.

The package enforces the main invariants:

- SQL is the source of truth.
- BM25/OpenSearch and ChromaDB are derived caches.
- Cross-store updates happen through `indexing_outbox`.
- Reminder retrieval stays SQL-only.
- Knowledge actions mutate knowledge tables only.
- Reminder actions mutate reminder tables only.
- Action branches return results to the Response Bundler before Chat Output.
- Last-QA state is volatile and separate from permanent conversation history.

## Package Map

- `assistant_rag/contracts.py` defines shared request/result contracts.
- `assistant_rag/database.py` defines the authoritative SQL schema and atomic writes.
- `assistant_rag/retrieval.py` defines BM25/Chroma-style derived cache adapters.
- `assistant_rag/bm25_opensearch.py` defines the production OpenSearch BM25 cache.
- `assistant_rag/validation.py` defines deterministic validation registries.
- `assistant_rag/branches.py` defines the four branch executors.
- `assistant_rag/bundler.py` defines the single response assembly layer.
- `assistant_rag/pipeline.py` wires the exact runtime order.
- `assistant_rag/indexing.py` processes `indexing_outbox`.
- `assistant_rag/autoscan.py` implements SQL-only reminder autoscan and reply context loading.
- `assistant_rag/platform.py` performs post-bundling platform formatting.
- `assistant_rag/consistency.py` provides rollout and migration integrity checks.

## Fast Debug Runner

Use this when you want to test your own questions quickly in the terminal
without starting Streamlit or FastAPI. The runner creates/uses SQL,
OpenSearch indexes, and ChromaDB collections so retrieval sync can be debugged.
The `run` wrapper starts OpenSearch, ChromaDB, and Ollama with Docker Compose
before launching the debug runner.

```bash
./run
```

Equivalent direct command:

```bash
python3 debug_pipeline.py
```

To start the same infrastructure and then run the Streamlit UI:

```bash
./run streamlit
```

To skip Docker startup when services are already running:

```bash
ASSISTANT_SKIP_INFRA=1 ./run
```

To automatically pull configured Ollama models before running:

```bash
ASSISTANT_AUTO_PULL_OLLAMA_MODELS=1 ./run
```

Useful commands:

```text
ask hello
remember my laptop is silver
remind team meeting at 2030-01-01T09:00:00+00:00
rebuild
tables
notifications
autoscan 2030-01-01T09:01:00+00:00
reset
quit
```

Default production settings:

```text
ASSISTANT_DB_PATH=assistant_data/assistant.sqlite3
OPENSEARCH_URL=http://localhost:9200
OPENSEARCH_CONVERSATION_INDEX=assistant_conversation_hops
OPENSEARCH_KNOWLEDGE_INDEX=assistant_knowledge_chunks
DEBUG_OPENSEARCH_CONVERSATION_INDEX=debug_assistant_conversation_hops
DEBUG_OPENSEARCH_KNOWLEDGE_INDEX=debug_assistant_knowledge_chunks
DEBUG_CHROMA_PATH=assistant_data/debug_chroma
ASSISTANT_CHROMA_HOST=localhost
ASSISTANT_CHROMA_PORT=8000
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_FAST_MODEL=llama3.2:3b
OLLAMA_BALANCED_MODEL=llama3.2:3b
```

## Production-Local Commands

Ollama defaults to `http://localhost:11434`. Configure runtime values with
environment variables such as `OLLAMA_BASE_URL`, `OLLAMA_FAST_MODEL`,
`OLLAMA_BALANCED_MODEL`, `ASSISTANT_DB_PATH`, and `ASSISTANT_CHROMA_PATH`.

```bash
python3 -m assistant_rag.cli init-db
python3 -m assistant_rag.cli check-ollama
python3 -m assistant_rag.cli check-opensearch
python3 -m assistant_rag.cli inspect-db
python3 -m assistant_rag.cli inspect-indexes
python3 -m assistant_rag.cli rebuild-indexes
streamlit run streamlit_app.py
```

Using the local virtualenv:

```bash
.venv/bin/python -m assistant_rag.cli init-db
.venv/bin/python -m assistant_rag.cli check-ollama
.venv/bin/python -m assistant_rag.cli check-opensearch
.venv/bin/python -m assistant_rag.cli inspect-db
.venv/bin/python -m assistant_rag.cli inspect-indexes
.venv/bin/python -m assistant_rag.cli rebuild-indexes
.venv/bin/streamlit run streamlit_app.py
```
