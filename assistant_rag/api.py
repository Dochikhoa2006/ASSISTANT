"""FastAPI application factory."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from hashlib import sha256
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


def _stable_payload_hash(request: ChatRequest) -> str:
    payload = asdict(request)
    payload.pop("idempotency_key", None)
    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return sha256(encoded.encode("utf-8")).hexdigest()


def _looks_like_mutation(request: ChatRequest) -> bool:
    metadata = request.metadata or {}
    if request.confirmation_token:
        return True
    if metadata.get("knowledge_actions") or metadata.get("reminder_actions"):
        return True
    lowered = request.raw_query.casefold()
    mutation_words = ("remind", "reminder", "delete", "remove", "modify", "change", "turn off", "turn on", "remember that")
    return any(word in lowered for word in mutation_words)


def _looks_destructive(request: ChatRequest) -> bool:
    lowered = request.raw_query.casefold()
    return any(word in lowered for word in ("delete", "remove", "modify", "change", "turn off"))


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

        metadata = dict(request.metadata or {})
        confirmation_to_mark: str | None = None
        if request.confirmation_token:
            try:
                confirmation = repository.load_pending_confirmation(
                    user_id=request.user_id,
                    confirmation_token=request.confirmation_token,
                    now_value=datetime.now(timezone.utc).isoformat(),
                )
            except ValueError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            proposed = confirmation["proposed_action"]
            domain = proposed.get("domain")
            if domain == "reminder":
                metadata["validated_reminder_actions"] = proposed.get("actions", [])
                metadata["reminder_actions"] = proposed.get("actions", [])
            elif domain == "knowledge":
                metadata["validated_knowledge_actions"] = proposed.get("actions", [])
                metadata["knowledge_actions"] = proposed.get("actions", [])
            metadata["operation_response"] = proposed.get("operation_response", metadata.get("operation_response"))
            metadata["topic_title"] = proposed.get("topic_title", metadata.get("topic_title"))
            metadata["confirmation_approved"] = True
            confirmation_to_mark = request.confirmation_token
            request = replace(request, metadata=metadata)

        is_mutation = _looks_like_mutation(request)
        if is_mutation:
            with StageTimer("rate_limit_checked", {"endpoint": "mutation"}):
                mutation_result = limiter.allow(f"mutation:{auth_context.user_id}", 20, 60)
            if not mutation_result.allowed:
                GLOBAL_METRICS.increment("rate_limit_blocked_total", endpoint="mutation")
                raise HTTPException(
                    status_code=429,
                    detail={"error": "rate_limited", "retry_after_seconds": mutation_result.retry_after_seconds},
                    headers={"Retry-After": str(mutation_result.retry_after_seconds)},
                )
            if _looks_destructive(request):
                with StageTimer("rate_limit_checked", {"endpoint": "destructive"}):
                    destructive_result = limiter.allow(f"destructive:{auth_context.user_id}", 5, 60)
                if not destructive_result.allowed:
                    GLOBAL_METRICS.increment("rate_limit_blocked_total", endpoint="destructive")
                    raise HTTPException(
                        status_code=429,
                        detail={"error": "rate_limited", "retry_after_seconds": destructive_result.retry_after_seconds},
                        headers={"Retry-After": str(destructive_result.retry_after_seconds)},
                    )

        idempotency_request_id: str | None = None
        if is_mutation and request.idempotency_key:
            claim = repository.claim_idempotency_key(
                user_id=request.user_id,
                idempotency_key=request.idempotency_key,
                payload_hash=_stable_payload_hash(request),
            )
            idempotency_request_id = claim.request_id
            if claim.status == "replay" and claim.stored_response_json:
                GLOBAL_METRICS.increment("idempotency_replay_total")
                return json.loads(claim.stored_response_json)
            if claim.status in {"in_progress", "conflict"}:
                GLOBAL_METRICS.increment("mutation_conflict_total", reason=claim.reason or claim.status)
                raise HTTPException(status_code=409, detail=claim.reason or claim.status)
        elif is_mutation:
            metadata.setdefault("warnings", [])
            metadata["warnings"] = list(metadata["warnings"]) + ["Mutation request had no idempotency_key; processed in compatibility mode."]
            request = replace(request, metadata=metadata)

        try:
            with StageTimer("api_total"):
                response = pipeline.handle(request, repository)
            latency_ms = int((time.perf_counter() - start) * 1000)
            GLOBAL_METRICS.observe_latency("chat_latency_ms", latency_ms)
            GLOBAL_METRICS.increment("chat_responses_total", response_type=response.response_type.value)
            payload = _response_payload(
                response,
                request_id=idempotency_request_id or request_id,
                latency_ms=latency_ms,
                include_trace=settings.operations.debug_trace_responses,
            )
            if payload["actions_committed"]:
                GLOBAL_METRICS.increment("mutation_success_total", len(payload["actions_committed"]))  # type: ignore[arg-type]
            if payload["actions_pending_confirmation"]:
                GLOBAL_METRICS.increment("mutation_confirmation_required_total", len(payload["actions_pending_confirmation"]))  # type: ignore[arg-type]
            if response.response_type is ResponseType.CLARIFICATION:
                GLOBAL_METRICS.increment("clarification_total")
            if idempotency_request_id:
                repository.complete_idempotency_request(
                    request_id=idempotency_request_id,
                    stored_response_json=json.dumps(payload, default=str),
                )
            if confirmation_to_mark and response.response_type is not ResponseType.ERROR:
                repository.mark_confirmation_confirmed(
                    user_id=request.user_id,
                    confirmation_token=confirmation_to_mark,
                )
            return payload
        except Exception as exc:
            GLOBAL_METRICS.increment("chat_errors_total", error_type=type(exc).__name__)
            if idempotency_request_id:
                repository.fail_idempotency_request(
                    request_id=idempotency_request_id,
                    error_message=str(exc),
                )
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
            write = repository.append_reminder_reply(
                user_id=payload["user_id"],
                reminder_id=reminder_id,
                notification_id=payload["notification_id"],
                reply_text=payload["reply_text"],
                response_text="Noted. (Context saved)",
            )
            return {"conversation_hop_id": write.hop_id, "indexing_outbox_job_id": write.outbox_job_id}
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
