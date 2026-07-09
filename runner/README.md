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

* Fast routing, query rewrite, Last-QA, and lightweight tool routing: `qwen3:4b`
* Intent classification and action extraction: `qwen3:8b`
* Risky delete/modify validation: `qwen3:14b`
* Final answer generation: `llama3.1:8b`
* Embeddings: `BAAI/bge-m3`

Runtime policy defaults:

* Deterministic routing/extraction temperatures: `0.0`
* Final answer temperature: `0.25`
* Long-form writing temperature: `0.45`
* Timeouts: fast `20s`, balanced `45s`, accurate/risky `90s`, writing `120s`
* Context windows: fast `8192`, balanced `16384`, accurate/writing `32768`
* Retrieval: BM25 `30`, Chroma `30`, RRF `60`, reranker top-k `20`, reranker min score `0.35`, final context `8`
* Confidence floors: retrieval `0.25`, knowledge context `0.35`, conversation context `0.40`
* Last-QA: minimum `0.75`, clarification merge `0.80`, broad-retrieval skip `0.85`
* Actions: action minimum `0.70`, risky action threshold `0.85`, risky ops `delete,modify,turn_off`
* Reminder context: minimum confidence `0.50`
* Reminder resolver: candidates `20`, target score `0.72`, ambiguity margin `0.12`, fuzzy threshold `0.78`, LLM validation enabled
* Knowledge chunks: size `700`, overlap `100`, minimum `80`, maximum `1000`
* Embeddings: normalized, batch size `32`, max length `8192`
* Worker: outbox batch `50`, retries `5`, retry backoff `30s`, stale processing timeout `300s`, outbox interval `5s`, reminder autoscan `60s`

Heavy production mode is opt-in so normal laptop runs do not auto-pull a 30B model:
```bash
OLLAMA_HEAVY_PRODUCTION_ENABLED=1 ./runner/run
```

When enabled, the configured heavy model default is `qwen3:30b`.
