# SQL-First RAG Assistant

This project is a sophisticated Retrieval-Augmented Generation (RAG) assistant pipeline built on a SQL-first foundation. It is designed to intelligently handle multi-turn conversations, retrieve past knowledge and reminders, route queries across intent branches, and construct dynamic responses using an LLM.

## Directory Structure

* **`assistant_rag/`**  
  The core backend application logic. Contains the production pipeline, RAG models, intent classification, orchestration branches, database wrappers, context filtering, and prompt registries.
  
* **`alembic/` & `alembic.ini`**  
  Database migration files and configuration. Used to maintain the SQL schema across versions.

* **`assistant_data/`**  
  A directory holding runtime data, such as the production `assistant.sqlite3` database file.

* **`runner/`**  
  Contains infrastructure and bootstrapping scripts.
  * **`run`**: A shell script used to bring up required Docker containers (OpenSearch, ChromaDB, Ollama) and launch the pipeline in either debug or UI mode.
  * **`docker-compose.runtime.yml`**: Defines the backend service dependencies required by the application.

* **`streamlit_app.py`**  
  The primary web-based User Interface (UI) for interacting with the assistant using Streamlit.
  
* **`debug_pipeline.py`**  
  A Command Line Interface (CLI) testing utility. It executes the exact same RAG operations as the Streamlit app but inside the terminal without loading a web framework.
  
* **`requirements-production.txt`**  
  The list of Python packages required to run the production system.

## Running the Application

To start the infrastructure services and the interactive terminal UI, use the runner script:
```bash
./runner/run
```

To run the Streamlit web interface:
```bash
./runner/run streamlit
```

## Default Model Policy

Normal runtime uses these central defaults from `assistant_rag/settings.py`:

* Fast structured work uses `qwen3.5:4b`: query rewrite, intent classification, action extraction, clarification, supporting-question generation, merge, risk checks, and sub-branch/composer detection.
* Strong semantic and generative work uses `qwen3.5:9b`: Last-QA, final answers, long-form writing, mutation validation/finalization, retrieval validation, and action planning. These routes recover through `qwen3.5:4b` where supported.
* Production validates every primary, fallback, and call-time override against this two-model pool. The hybrid Ollama/ONNX architecture remains available, but no ONNX LLM is routed or loaded by default. Before consolidation, these strong routes used `microsoft/Phi-4-mini-instruct-onnx`, which created a third resident LLM allocation.
* Embeddings: `BAAI/bge-m3`
* Cross-encoder reranking: `BAAI/bge-reranker-v2-m3` (the embedding and reranker are retrieval models, not generative LLMs)

Runtime policy defaults:

* Deterministic routing/extraction temperatures: `0.0`
* Final answer temperature: `0.22`
* Long-form writing temperature: `0.38`
* Ollama idle retention: `5m`; override with `OLLAMA_KEEP_ALIVE` when a deployment deliberately trades RAM for fewer cold starts.
* Timeouts: routing `12-30s`, extraction `24s`, validation/planning `35-45s`, answer `75s`, finalization/writing `90s`
* Context windows: compact routing `768-2048`, extraction `1536-8192`, candidate validation `4096-12288`, answer/writing `4096`
* Output caps (`num_predict`): compact routing `64-256`, validation `128-1024`, answer/writing `1024`, lossless knowledge content `2048`
* Retrieval: BM25 `20`, Chroma `20`, RRF constant `40`, RRF candidates `15`, reranker min score `0.30`, final context `5`
* Confidence floors: reranker `0.30`, conversation retrieval `0.50`
* Last-QA: minimum `0.80`, clarification merge `0.84`, broad-retrieval skip `0.90`
* Actions: action minimum `0.76`, risky action threshold `0.90`, risky ops `delete,turn_off`
* Reminder context: minimum confidence `0.58`
* Reminder resolver: candidates `12`, target score `0.78`, ambiguity margin `0.08`, fuzzy threshold `0.82`, LLM validation enabled
* Knowledge chunks: size `560`, overlap `80`, minimum `80`, maximum `800`
* Embeddings: normalized, batch size `48`, max length `4096`
* Worker: outbox batch `64`, retries `4`, retry backoff `15s`, stale processing timeout `180s`, outbox interval `3s`, reminder autoscan `30s`
