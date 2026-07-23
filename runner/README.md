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

* Every generative task uses the exact Ollama tag `qwen3.5:2b`, including routing, extraction, Last-QA, final answers, writing, mutation validation/finalization, retrieval validation, and action planning.
* Production rejects any different primary, fallback, environment, or call-time model override. Optional fallback fields are empty by default because pointing them to the same tag would not provide a real fallback.
* Structured stages recover through one bounded, format-diversified retry on `qwen3.5:2b`; unstructured generation retries one transient failure on that same model. The hybrid Ollama/ONNX implementation remains available for compatibility tests, but production neither routes nor preloads an ONNX generative model.
* Thinking is explicitly disabled. Qwen3.5-2B defaults to non-thinking mode, and keeping that mode explicit avoids hidden reasoning loops and makes schema-constrained output more predictable.
* Embeddings: `BAAI/bge-m3`
* Cross-encoder reranking: `BAAI/bge-reranker-v2-m3` (the embedding and reranker are retrieval models, not generative LLMs)

Runtime policy defaults:

* Deterministic routing/extraction temperatures: `0.0`
* Final answer temperature: `0.22`
* Long-form writing temperature: `0.38`
* Ollama idle retention: `5m`; override with `OLLAMA_KEEP_ALIVE` when a deployment deliberately trades RAM for fewer cold starts.
* Timeouts: routing `12-30s`, extraction `24-36s`, validation/planning `35-45s`, answer `75s`, finalization `90s`, writing `120s`
* Context windows: compact routing `2048-4096`, extraction `4096-8192`, candidate validation `8192-16384`, answer/writing `8192`. These stay well below the model's native 256K window to bound KV-cache use while leaving room for system prompts, schemas, and runtime evidence.
* Output caps (`num_predict`): compact routing `64-256`, validation `128-1024`, answer `1536`, writing/lossless knowledge content `2048`
* Retrieval: BM25 `20`, Chroma `20`, RRF constant `40`, RRF candidates `15`, reranker min score `0.30`, final context `5`
* Confidence floors: reranker `0.30`, conversation retrieval `0.50`
* Last-QA: minimum `0.80`, clarification merge `0.84`, broad-retrieval skip `0.90`
* Actions: action minimum `0.76`, risky action threshold `0.90`, risky ops `delete,turn_off`
* Reminder context: minimum confidence `0.58`
* Reminder resolver: candidates `12`, target score `0.78`, ambiguity margin `0.08`, fuzzy threshold `0.82`, LLM validation enabled
* Knowledge chunks: size `560`, overlap `80`, minimum `80`, maximum `800`
* Embeddings: normalized, batch size `48`, max length `4096`
* Worker: outbox batch `64`, retries `4`, retry backoff `15s`, stale processing timeout `180s`, outbox interval `3s`, reminder autoscan `30s`
