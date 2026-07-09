# Release 4 Operations

## Lifecycle

SQL is the source of truth. Derived indexes contain only active `conversation_hop` rows whose topic is active and active `knowledge_chunk` rows where `is_deleted = 0`. Live reminder rows are never indexed. Archived conversation topics remain SQL-readable, but normal retrieval and rebuilds skip them. Deleted or expired artifacts are not downloadable through normal artifact reads. Expired confirmations cannot execute.

## Health

`GET /health` returns JSON:

```json
{
  "status": "ok",
  "dependencies": [],
  "outbox_pending_count": 0,
  "outbox_failed_count": 0
}
```

SQL is critical. SQL failures return `unhealthy`. OpenSearch, Chroma, Ollama, Redis, and storage are optional by default and become degraded on failure unless the matching `ASSISTANT_HEALTH_STRICT_*` setting promotes them to unhealthy. Health payloads must not include credentials, tokens, raw paths, or user content.

## Metrics

`GET /metrics` returns JSON counters, gauges, latency summaries, and `generated_at`. Metrics are process-local for Release 4. They intentionally exclude raw prompts, document text, tokens, storage paths, and artifact paths.

Important counters include chat requests/responses/errors, LLM failures and JSON parse failures, retrieval empty results, mutation success/conflict/confirmation-required, idempotency replay, outbox success/failure, reminder scans, notification creation and delivery failures, artifact generation, and ingestion success/failure.

## Logging And Tracing

Structured logs use stdlib `logging` plus `JsonLogFormatter`. Raw content logging is disabled by default with `ASSISTANT_LOG_RAW_CONTENT=false`. Even when raw content logging is enabled for development, tokens, secrets, authorization headers, file contents, artifact paths, and SQL credentials remain redacted.

`/chat` includes `trace_summary` only when `ASSISTANT_DEBUG_TRACE_RESPONSES=true`. Trace stages include rewrite, Last-QA, retrieval, SQL validation, classification, branch execution, bundling, and API total timing.

## Recurring Reminders

Recurring reminders support `daily`, `weekly`, and `monthly` rules natively. `next_fire_time` and `last_fire_time` are stored as UTC ISO strings. Recurrence calculation happens in `recurrence_timezone`, defaulting to the request timezone and then `ASSISTANT_RECURRENCE_DEFAULT_TIMEZONE`.

Autoscan uses `COALESCE(next_fire_time, reminder_time)`. One-time reminders move from `scheduled` to `notified` as before. Recurring reminders remain `scheduled`, create one notification for each fire time, update `last_fire_time`, and advance `next_fire_time`. When a recurrence has no next fire, the reminder becomes `completed`.

## Index Rebuild And Drift

CLI commands:

```bash
python -m assistant_rag.ops_cli rebuild-all-indexes
python -m assistant_rag.ops_cli rebuild-user-indexes --user-id USER_ID
python -m assistant_rag.ops_cli rebuild-knowledge-index --user-id USER_ID
python -m assistant_rag.ops_cli rebuild-conversation-index --user-id USER_ID
python -m assistant_rag.ops_cli check-index-drift --user-id USER_ID
python -m assistant_rag.ops_cli check-index-drift --user-id USER_ID --repair
```

Rebuilds read SQL truth only and write conversation hops plus active knowledge chunks. Drift checks compare SQL counts with BM25/Chroma counts, failed outbox count, missing documents, deleted indexed chunks, forbidden reminder documents, and metadata mismatches. Repair is opt-in.

## Evaluation

Run:

```bash
python -m assistant_rag.ops_cli run-retrieval-eval --cases tests/fixtures/eval/release4_cases.json
python -m assistant_rag.ops_cli run-retrieval-eval --cases tests/fixtures/eval/release4_cases.json --strict
```

The runner writes a machine-readable JSON report with top-1/top-3 accuracy, wrong-target rate, clarification rate, false mutation rate, retrieval-empty rate, and average latency. Strict mode exits non-zero when configured thresholds fail.
