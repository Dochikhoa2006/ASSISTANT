"""FastAPI application factory."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .contracts import ChatRequest
from .database import SQLRepository
from .pipeline import AssistantPipeline


@dataclass
class NotificationWebSocketHub:
    connections: dict[str, set[Any]] = field(default_factory=dict)

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
            except Exception:
                stale.append(websocket)
        for websocket in stale:
            self.disconnect(user_id=user_id, websocket=websocket)


def build_api_app(
    pipeline: AssistantPipeline,
    repository: SQLRepository | None = None,
    notification_hub: NotificationWebSocketHub | None = None,
):
    try:
        from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
    except ImportError as exc:
        raise RuntimeError("Install fastapi and uvicorn to run the production API") from exc

    app = FastAPI(title="SQL-First RAG Assistant")
    hub = notification_hub or NotificationWebSocketHub()

    @app.get("/health")
    def health() -> dict[str, object]:
        payload: dict[str, object] = {"status": "ok"}
        if repository is not None:
            pending = repository.connection.execute(
                "SELECT COUNT(*) AS total FROM indexing_outbox WHERE status IN ('pending', 'failed')"
            ).fetchone()
            payload["outbox_pending_or_failed"] = int(pending["total"])
        return payload

    @app.post("/chat")
    def chat(request: ChatRequest) -> dict[str, object]:
        response = pipeline.handle(request)
        return {
            "final_chat_text": response.final_chat_text,
            "response_type": response.response_type.value,
            "persistence_instructions": response.persistence_instructions,
        }

    @app.get("/notifications")
    def notifications(
        user_id: str,
        include_deleted: bool = Query(default=False),
    ) -> list[dict[str, object]]:
        if repository is None:
            raise HTTPException(status_code=503, detail="Repository is not configured")
        return repository.list_notifications(user_id=user_id, include_deleted=include_deleted)

    @app.patch("/notifications/{notification_id}")
    async def patch_notification(
        notification_id: str,
        payload: dict[str, str],
        user_id: str,
    ) -> dict[str, object]:
        if repository is None:
            raise HTTPException(status_code=503, detail="Repository is not configured")
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
        status: str | None = None,
        from_time: str | None = None,
        to_time: str | None = None,
    ) -> list[dict[str, object]]:
        if repository is None:
            raise HTTPException(status_code=503, detail="Repository is not configured")
        return repository.list_reminders(
            user_id=user_id, status=status, from_time=from_time, to_time=to_time
        )

    @app.post("/reminders/{reminder_id}/reply")
    def reminder_reply(reminder_id: str, payload: dict[str, str], user_id: str) -> dict[str, object]:
        if repository is None:
            raise HTTPException(status_code=503, detail="Repository is not configured")
        try:
            hop = repository.append_reminder_reply(
                user_id=user_id,
                reminder_id=reminder_id,
                notification_id=payload["notification_id"],
                reply_text=payload["reply_text"],
                response_text=payload.get("response_text", "I saved your reminder reply."),
            )
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"conversation_hop_id": hop.hop_id, "indexing_outbox_job_id": hop.outbox_job_id}

    @app.websocket("/ws/notifications")
    async def notification_socket(websocket: WebSocket, user_id: str) -> None:
        await hub.connect(user_id=user_id, websocket=websocket)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            hub.disconnect(user_id=user_id, websocket=websocket)

    return app
