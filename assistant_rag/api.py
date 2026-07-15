"""FastAPI application factory."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from fastapi import HTTPException

from .contracts import AuthContext, ChatRequest, ResponseType
from .database import AssistantRepository
from .health import build_health_report
from .ingestion import KnowledgeIngestionService
from .metrics import GLOBAL_METRICS
from .observability import StageTimer, new_request_id, start_trace, trace_summary_asdict
from .pipeline import AssistantPipeline
from .auth import authenticate_token, get_auth_context
from .rate_limit import RateLimiter, build_rate_limiter
from .settings import ProductionSettings
from .reminder_reply import build_reminder_reply_last_qa, build_reminder_reply_metadata
from .request_lifecycle import (
    ChatRequestLifecycleExecutor,
    RequestLifecycleConflict,
    looks_destructive as _looks_destructive,
    looks_like_mutation as _looks_like_mutation,
    stable_payload_hash as _stable_payload_hash,
)


@dataclass
class NotificationWebSocketHub:
    connections: dict[str, set[Any]] = field(default_factory=dict)
    repository_factory: Callable[[], AssistantRepository] | None = None

    async def connect(self, *, user_id: str, websocket: Any) -> None:
        await websocket.accept()
        self.connections.setdefault(user_id, set()).add(websocket)

    def disconnect(self, *, user_id: str, websocket: Any) -> None:
        self.connections.get(user_id, set()).discard(websocket)

    async def publish(self, *, user_id: str, payload: dict[str, Any]) -> None:
        stale = []
        for websocket in self.connections.get(user_id, set()):
            try:
                await websocket.send_json(payload)
                self._mark_delivery(user_id=user_id, payload=payload, sent=True)
            except Exception:
                self._mark_delivery(user_id=user_id, payload=payload, sent=False)
                stale.append(websocket)
        for websocket in stale:
            self.disconnect(user_id=user_id, websocket=websocket)

    def _mark_delivery(self, *, user_id: str, payload: dict[str, Any], sent: bool) -> None:
        data = payload.get("data") if isinstance(payload, dict) else None
        notification_id = data.get("notification_id") if isinstance(data, dict) else None
        if not notification_id or self.repository_factory is None:
            return
        repo = self.repository_factory()
        try:
            if sent:
                repo.mark_notification_delivery_sent(user_id=user_id, notification_id=notification_id)
            else:
                GLOBAL_METRICS.increment("notification_delivery_failure_total")
                repo.mark_notification_delivery_failed(
                    user_id=user_id,
                    notification_id=notification_id,
                    error_message="websocket delivery failed",
                    retrying=True,
                )
        finally:
            if hasattr(repo, "close"):
                repo.close()


def _response_payload(response: Any, *, request_id: str, latency_ms: int, include_trace: bool = False) -> dict[str, object]:
    payload: dict[str, object] = {
        "request_id": request_id,
        "final_chat_text": response.final_chat_text,
        "response_type": response.response_type.value,
        "conversation_topic_id": response.conversation_topic_id,
        "conversation_hop_id": response.conversation_hop_id,
        "actions_committed": response.actions_committed,
        "actions_pending_confirmation": response.actions_pending_confirmation,
        "warnings": response.warnings,
        "latency_ms": latency_ms,
        "persistence_instructions": response.persistence_instructions,
        "platform_payload": response.platform_payload,
    }
    if include_trace:
        payload["trace_summary"] = trace_summary_asdict(getattr(response, "trace_summary", None))
    return payload


def build_api_app(
    pipeline: AssistantPipeline,
    repository_factory: Callable[[], AssistantRepository] | None = None,
    notification_hub: NotificationWebSocketHub | None = None,
    rate_limiter: RateLimiter | None = None,
):
    try:
        from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect, Depends
    except ImportError as exc:
        raise RuntimeError("Install fastapi and uvicorn to run the production API") from exc

    app = FastAPI(title="SQL-First RAG Assistant")
    hub = notification_hub or NotificationWebSocketHub()
    hub.repository_factory = repository_factory
    limiter = rate_limiter or build_rate_limiter()
    settings = ProductionSettings.from_env()

    def enforce_idempotency_policy(
        request: ChatRequest, is_mutation: bool, _is_destructive: bool
    ) -> None:
        if (
            is_mutation
            and not str(request.idempotency_key or "").strip()
            and not settings.safety.allow_missing_idempotency_key
        ):
            raise HTTPException(
                status_code=400,
                detail="idempotency_key is required for mutation requests",
            )

    def get_repository() -> Iterator[AssistantRepository]:
        if repository_factory is None:
            raise HTTPException(status_code=503, detail="Repository is not configured")
        repo = repository_factory()
        try:
            yield repo
        finally:
            if hasattr(repo, "close"):
                repo.close()

    @app.get("/health")
    def health(repository: AssistantRepository = Depends(get_repository)) -> dict[str, object]:
        return build_health_report(repository=repository, settings=settings)

    @app.get("/metrics")
    def metrics() -> dict[str, object]:
        return GLOBAL_METRICS.snapshot_dict()

    @app.post("/chat")
    def chat(
        request: ChatRequest,
        repository: AssistantRepository = Depends(get_repository),
        auth_context: AuthContext = Depends(get_auth_context),
    ) -> dict[str, object]:
        if request.user_id != auth_context.user_id:
            raise HTTPException(status_code=403, detail="Not authorized for this user_id")
        request_id = new_request_id(request.idempotency_key)
        start_trace(request_id)
        start = time.perf_counter()
        GLOBAL_METRICS.increment("chat_requests_total")
        with StageTimer("rate_limit_checked", {"endpoint": "chat"}):
            rate_key = f"chat:{auth_context.user_id}"
            rate_result = limiter.allow(rate_key, 60, 60)
        if not rate_result.allowed:
            GLOBAL_METRICS.increment("rate_limit_blocked_total", endpoint="chat")
            raise HTTPException(
                status_code=429,
                detail={"error": "rate_limited", "retry_after_seconds": rate_result.retry_after_seconds},
                headers={"Retry-After": str(rate_result.retry_after_seconds)},
            )

        executor = ChatRequestLifecycleExecutor(
            pipeline=pipeline,
            repository=repository,
        )

        def enforce_mutation_limits(
            _request: ChatRequest, is_mutation: bool, is_destructive: bool
        ) -> None:
            enforce_idempotency_policy(_request, is_mutation, is_destructive)
            if not is_mutation:
                return
            with StageTimer("rate_limit_checked", {"endpoint": "mutation"}):
                mutation_result = limiter.allow(
                    f"mutation:{auth_context.user_id}", 20, 60
                )
            if not mutation_result.allowed:
                GLOBAL_METRICS.increment(
                    "rate_limit_blocked_total", endpoint="mutation"
                )
                raise HTTPException(
                    status_code=429,
                    detail={
                        "error": "rate_limited",
                        "retry_after_seconds": mutation_result.retry_after_seconds,
                    },
                    headers={
                        "Retry-After": str(mutation_result.retry_after_seconds)
                    },
                )
            if not is_destructive:
                return
            with StageTimer("rate_limit_checked", {"endpoint": "destructive"}):
                destructive_result = limiter.allow(
                    f"destructive:{auth_context.user_id}", 5, 60
                )
            if not destructive_result.allowed:
                GLOBAL_METRICS.increment(
                    "rate_limit_blocked_total", endpoint="destructive"
                )
                raise HTTPException(
                    status_code=429,
                    detail={
                        "error": "rate_limited",
                        "retry_after_seconds": destructive_result.retry_after_seconds,
                    },
                    headers={
                        "Retry-After": str(destructive_result.retry_after_seconds)
                    },
                )

        def serialize_response(response: Any, lifecycle_request_id: str) -> dict[str, Any]:
            latency_ms = int((time.perf_counter() - start) * 1000)
            GLOBAL_METRICS.observe_latency("chat_latency_ms", latency_ms)
            GLOBAL_METRICS.increment(
                "chat_responses_total", response_type=response.response_type.value
            )
            payload = _response_payload(
                response,
                request_id=lifecycle_request_id,
                latency_ms=latency_ms,
                include_trace=settings.operations.debug_trace_responses,
            )
            if payload["actions_committed"]:
                GLOBAL_METRICS.increment(
                    "mutation_success_total", len(payload["actions_committed"])
                )
            if payload["actions_pending_confirmation"]:
                GLOBAL_METRICS.increment(
                    "mutation_confirmation_required_total",
                    len(payload["actions_pending_confirmation"]),
                )
            if response.response_type is ResponseType.CLARIFICATION:
                GLOBAL_METRICS.increment("clarification_total")
            return payload

        try:
            with StageTimer("api_total"):
                execution = executor.execute(
                    request,
                    fallback_request_id=request_id,
                    serialize_response=serialize_response,
                    before_claim=enforce_mutation_limits,
                )
            if execution.replayed:
                GLOBAL_METRICS.increment("idempotency_replay_total")
            return execution.payload
        except RequestLifecycleConflict as exc:
            GLOBAL_METRICS.increment(
                "mutation_conflict_total", reason=str(exc) or "request_conflict"
            )
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except HTTPException:
            raise
        except Exception as exc:
            GLOBAL_METRICS.increment("chat_errors_total", error_type=type(exc).__name__)
            raise

    @app.get("/notifications")
    def notifications(
        user_id: str,
        include_deleted: bool = Query(default=False),
        repository: AssistantRepository = Depends(get_repository),
        auth_context: AuthContext = Depends(get_auth_context),
    ) -> list[dict[str, object]]:
        if user_id != auth_context.user_id:
            raise HTTPException(status_code=403, detail="Not authorized for this user_id")
        rate_result = limiter.allow(f"notifications:{auth_context.user_id}", 120, 60)
        if not rate_result.allowed:
            raise HTTPException(
                status_code=429,
                detail={"error": "rate_limited", "retry_after_seconds": rate_result.retry_after_seconds},
                headers={"Retry-After": str(rate_result.retry_after_seconds)},
            )
        return repository.list_notifications(user_id=user_id, include_deleted=include_deleted)

    @app.patch("/notifications/{notification_id}")
    async def patch_notification(
        notification_id: str,
        payload: dict[str, str],
        user_id: str,
        repository: AssistantRepository = Depends(get_repository),
        auth_context: AuthContext = Depends(get_auth_context),
    ) -> dict[str, object]:
        if user_id != auth_context.user_id:
            raise HTTPException(status_code=403, detail="Not authorized for this user_id")
        try:
            updated = repository.update_notification_ui_status(
                user_id=user_id,
                notification_id=notification_id,
                ui_status=payload["ui_status"],
            )
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        await hub.publish(user_id=user_id, payload={"type": "notification_updated", "data": updated})
        return updated

    @app.get("/reminders")
    def reminders(
        user_id: str,
        status: Optional[str] = None,
        from_time: Optional[str] = None,
        to_time: Optional[str] = None,
        repository: AssistantRepository = Depends(get_repository),
        auth_context: AuthContext = Depends(get_auth_context),
    ) -> list[dict[str, object]]:
        if user_id != auth_context.user_id:
            raise HTTPException(status_code=403, detail="Not authorized for this user_id")
        return repository.list_reminders(
            user_id=user_id,
            status=status,
            from_time=from_time,
            to_time=to_time,
        )

    @app.get("/knowledge/sources")
    def knowledge_sources(
        include_deleted: bool = Query(default=False),
        repository: AssistantRepository = Depends(get_repository),
        auth_context: AuthContext = Depends(get_auth_context),
    ) -> list[dict[str, object]]:
        return repository.list_knowledge_sources(
            user_id=auth_context.user_id,
            include_deleted=include_deleted,
        )

    @app.get("/knowledge/sources/{source_id}")
    def knowledge_source(
        source_id: str,
        repository: AssistantRepository = Depends(get_repository),
        auth_context: AuthContext = Depends(get_auth_context),
    ) -> dict[str, object]:
        try:
            return repository.get_knowledge_source(user_id=auth_context.user_id, source_id=source_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/knowledge/sources/ingest-text")
    def ingest_text_source(
        payload: dict[str, Any],
        repository: AssistantRepository = Depends(get_repository),
        auth_context: AuthContext = Depends(get_auth_context),
    ) -> dict[str, object]:
        text = str(payload.get("text") or "")
        filename = str(payload.get("filename") or "manual.txt")
        topic_title = str(payload.get("topic_title") or "Knowledge")
        if not text.strip():
            raise HTTPException(status_code=400, detail="text is required")
        result = KnowledgeIngestionService(repository).ingest_text(
            user_id=auth_context.user_id,
            filename=filename,
            text=text,
            topic_title=topic_title,
            metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else None,
        )
        return {
            "source_id": result.source_id,
            "status": result.status.value,
            "chunk_ids": list(result.chunk_ids),
            "outbox_job_ids": list(result.outbox_job_ids),
            "error_message": result.error_message,
        }

    @app.delete("/knowledge/sources/{source_id}")
    def delete_knowledge_source(
        source_id: str,
        repository: AssistantRepository = Depends(get_repository),
        auth_context: AuthContext = Depends(get_auth_context),
    ) -> dict[str, object]:
        try:
            return repository.soft_delete_knowledge_source(user_id=auth_context.user_id, source_id=source_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/knowledge/sources/{source_id}/reindex")
    def reindex_knowledge_source(
        source_id: str,
        repository: AssistantRepository = Depends(get_repository),
        auth_context: AuthContext = Depends(get_auth_context),
    ) -> dict[str, object]:
        try:
            job_ids = repository.reindex_knowledge_source(user_id=auth_context.user_id, source_id=source_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"source_id": source_id, "outbox_job_ids": job_ids}

    @app.get("/knowledge/facts")
    def knowledge_facts(
        include_deleted: bool = Query(default=False),
        source_id: Optional[str] = None,
        repository: AssistantRepository = Depends(get_repository),
        auth_context: AuthContext = Depends(get_auth_context),
    ) -> list[dict[str, object]]:
        return repository.list_knowledge_facts(
            user_id=auth_context.user_id,
            include_deleted=include_deleted,
            source_id=source_id,
        )

    @app.patch("/knowledge/facts/{chunk_id}")
    def patch_knowledge_fact(
        chunk_id: str,
        payload: dict[str, Any],
        repository: AssistantRepository = Depends(get_repository),
        auth_context: AuthContext = Depends(get_auth_context),
    ) -> dict[str, object]:
        text = str(payload.get("text") or "")
        if not text.strip():
            raise HTTPException(status_code=400, detail="text is required")
        try:
            return repository.update_knowledge_chunk_text(
                user_id=auth_context.user_id,
                chunk_id=chunk_id,
                text=text,
                change_reason=str(payload.get("change_reason") or "api_update"),
                modified_by_user_query=str(payload.get("modified_by_user_query") or ""),
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.delete("/knowledge/facts/{chunk_id}")
    def delete_knowledge_fact(
        chunk_id: str,
        repository: AssistantRepository = Depends(get_repository),
        auth_context: AuthContext = Depends(get_auth_context),
    ) -> dict[str, object]:
        try:
            with repository.transaction() as cursor:
                job_id = repository.soft_delete_knowledge_chunk(
                    cursor,
                    user_id=auth_context.user_id,
                    chunk_id=chunk_id,
                    change_reason="api_delete",
                )
            return {"chunk_id": chunk_id, "outbox_job_id": job_id}
        except (ValueError, Exception) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/knowledge/facts/{chunk_id}/restore")
    def restore_knowledge_fact(
        chunk_id: str,
        repository: AssistantRepository = Depends(get_repository),
        auth_context: AuthContext = Depends(get_auth_context),
    ) -> dict[str, object]:
        try:
            job_id = repository.restore_knowledge_chunk(user_id=auth_context.user_id, chunk_id=chunk_id)
            return {"chunk_id": chunk_id, "outbox_job_id": job_id}
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/artifacts")
    def artifacts(
        include_deleted: bool = Query(default=False),
        repository: AssistantRepository = Depends(get_repository),
        auth_context: AuthContext = Depends(get_auth_context),
    ) -> list[dict[str, object]]:
        return repository.list_generated_artifacts(
            user_id=auth_context.user_id,
            include_deleted=include_deleted,
        )

    @app.get("/artifacts/{artifact_id}")
    def artifact(
        artifact_id: str,
        repository: AssistantRepository = Depends(get_repository),
        auth_context: AuthContext = Depends(get_auth_context),
    ) -> dict[str, object]:
        try:
            payload = repository.get_generated_artifact(user_id=auth_context.user_id, artifact_id=artifact_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        payload = dict(payload)
        payload.pop("storage_path", None)
        return payload

    @app.get("/artifacts/{artifact_id}/download")
    def download_artifact(
        artifact_id: str,
        repository: AssistantRepository = Depends(get_repository),
        auth_context: AuthContext = Depends(get_auth_context),
    ):
        from fastapi.responses import FileResponse

        try:
            payload = repository.get_generated_artifact(user_id=auth_context.user_id, artifact_id=artifact_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        path = Path(str(payload["storage_path"])).resolve()
        if not path.exists() or not path.is_file():
            raise HTTPException(status_code=404, detail="Artifact file is missing")
        return FileResponse(path, filename=str(payload["filename"]))

    @app.delete("/artifacts/{artifact_id}")
    def delete_artifact(
        artifact_id: str,
        repository: AssistantRepository = Depends(get_repository),
        auth_context: AuthContext = Depends(get_auth_context),
    ) -> dict[str, object]:
        try:
            payload = repository.delete_generated_artifact(user_id=auth_context.user_id, artifact_id=artifact_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        payload = dict(payload)
        payload.pop("storage_path", None)
        return payload

    @app.post("/reminders/{reminder_id}/reply")
    def reply_to_reminder(
        reminder_id: str,
        payload: dict[str, str],
        repository: AssistantRepository = Depends(get_repository),
        auth_context: AuthContext = Depends(get_auth_context),
    ) -> dict[str, object]:
        user_id = payload.get("user_id")
        if not user_id or user_id != auth_context.user_id:
            raise HTTPException(status_code=403, detail="Not authorized for this user_id")
        try:
            reply_text = str(payload.get("reply_text") or "").strip()
            if not reply_text:
                raise HTTPException(status_code=422, detail="reply_text is required")
            notification_id = str(payload.get("notification_id") or "")
            context = repository.load_reminder_reply_context(
                user_id=user_id,
                reminder_id=reminder_id,
                notification_id=notification_id,
            )
            if not context:
                raise ValueError("Reminder reply context not found for user")
            if not context.get("source_hop_id"):
                raise ValueError("Reminder source conversation is unavailable")
            reply_last_qa = build_reminder_reply_last_qa(context)
            idempotency_key = str(payload.get("idempotency_key") or "").strip() or None
            request_id = new_request_id(idempotency_key)
            start_trace(request_id)
            started_at = time.perf_counter()
            request = ChatRequest(
                user_id=user_id,
                raw_query=reply_text,
                reminder_id=reminder_id,
                notification_id=notification_id,
                reply_text=reply_text,
                parent_hop_id=context.get("source_hop_id"),
                idempotency_key=idempotency_key,
                metadata=build_reminder_reply_metadata(
                    reminder_id=reminder_id,
                    notification_id=notification_id,
                    context=context,
                ),
            )

            def serialize_response(response: Any, lifecycle_request_id: str) -> dict[str, Any]:
                # A successful response and its acknowledgement are one logical
                # lifecycle result.  Perform the acknowledgement before the
                # executor stores a completed replay payload; ERROR stays unread
                # and the same key remains retryable.
                if response.response_type is not ResponseType.ERROR:
                    repository.update_notification_ui_status(
                        user_id=user_id,
                        notification_id=notification_id,
                        ui_status="read",
                    )
                return _response_payload(
                    response,
                    request_id=lifecycle_request_id,
                    latency_ms=int((time.perf_counter() - started_at) * 1000),
                    include_trace=True,
                )

            def hydrate_reply_last_qa(_prepared_request: ChatRequest) -> None:
                # The lifecycle hook runs only after a fresh/failed-retry claim.
                # Replays and conflicts therefore never rewind Last-QA state.
                pipeline.last_qa_store.save(user_id, reply_last_qa)

            execution = ChatRequestLifecycleExecutor(
                pipeline=pipeline,
                repository=repository,
            ).execute(
                request,
                fallback_request_id=request_id,
                serialize_response=serialize_response,
                before_claim=enforce_idempotency_policy,
                before_pipeline=hydrate_reply_last_qa,
            )
            return execution.payload
        except RequestLifecycleConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.websocket("/ws/notifications")
    async def notification_socket(websocket: WebSocket) -> None:
        authorization = websocket.headers.get("authorization")
        token = None
        if authorization and authorization.startswith("Bearer "):
            token = authorization[len("Bearer ") :].strip()
        if not token:
            token = websocket.query_params.get("access_token")
        try:
            auth_context = authenticate_token(token or "")
        except HTTPException:
            await websocket.close(code=1008)
            return
        await hub.connect(user_id=auth_context.user_id, websocket=websocket)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            hub.disconnect(user_id=auth_context.user_id, websocket=websocket)

    return app
