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

* Last-QA latest-context relationship resolution: `qwen3.5:9b`, with `qwen3.5:4b` recovery
* Query rewrite, intent classification, action extraction, clarification, supporting-question generation, and sub-branch detection: `qwen3.5:4b`
* Final answers, long-form writing, mutation validation/finalization, retrieval validation, and action planning: `microsoft/Phi-4-mini-instruct-onnx`, with configured Ollama recovery models where required
* Embeddings: `BAAI/bge-m3`

Runtime policy defaults:

* Deterministic routing/extraction temperatures: `0.0`
* Final answer temperature: `0.22`
* Long-form writing temperature: `0.38`
* Timeouts: tiny routing `12s`, short JSON `15-18s`, extraction `30s`, validation/planning `35s`, answer `75s`, writing `90s`
* Context windows: routing `1024-2048`, extraction/merge `3072`, validation/planning `4096`, answer/writing `8192`
* Retrieval: BM25 `24`, Chroma `24`, RRF `40`, reranker top-k `16`, reranker min score `0.30`, final context `6`
* Confidence floors: retrieval `0.30`, knowledge context `0.38`, conversation context `0.42`
* Last-QA: minimum `0.80`, clarification merge `0.84`, broad-retrieval skip `0.90`
* Actions: action minimum `0.76`, risky action threshold `0.90`, risky ops `delete,modify,turn_off`
* Reminder context: minimum confidence `0.58`
* Reminder resolver: candidates `12`, target score `0.78`, ambiguity margin `0.08`, fuzzy threshold `0.82`, LLM validation enabled
* Knowledge chunks: size `560`, overlap `80`, minimum `80`, maximum `800`
* Embeddings: normalized, batch size `48`, max length `4096`
* Worker: outbox batch `64`, retries `4`, retry backoff `15s`, stale processing timeout `180s`, outbox interval `3s`, reminder autoscan `30s`
