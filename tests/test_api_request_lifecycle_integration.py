from __future__ import annotations

from dataclasses import dataclass, field

from fastapi.testclient import TestClient

from assistant_rag.api import build_api_app
from assistant_rag.contracts import (
    BundledResponse,
    ChatRequest,
    LastQAState,
    RateLimitResult,
    ResponseType,
)
from assistant_rag.database import SQLiteRepository


class RecordingPipeline:
    def __init__(self) -> None:
        self.requests: list[ChatRequest] = []

    def handle(
        self,
        request: ChatRequest,
        _repository: SQLiteRepository,
    ) -> BundledResponse:
        self.requests.append(request)
        text = "Stored exactly once."
        return BundledResponse(
            final_chat_text=text,
            response_type=ResponseType.KNOWLEDGE_ACTION,
            last_qa_state=LastQAState(
                last_user_query=request.raw_query,
                last_response=text,
                response_type=ResponseType.KNOWLEDGE_ACTION,
            ),
            actions_committed=[{"action_type": "add"}],
        )


@dataclass
class RecordingRateLimiter:
    keys: list[str] = field(default_factory=list)

    def allow(self, key: str, _limit: int, _window_seconds: int) -> RateLimitResult:
        self.keys.append(key)
        return RateLimitResult(allowed=True)


def test_chat_endpoint_preserves_shared_idempotency_replay_and_conflict_contract() -> None:
    repository = SQLiteRepository.in_memory()
    repository.initialize_schema()
    pipeline = RecordingPipeline()
    limiter = RecordingRateLimiter()
    app = build_api_app(
        pipeline=pipeline,  # type: ignore[arg-type]
        repository_factory=lambda: repository,
        rate_limiter=limiter,
    )
    client = TestClient(app)
    headers = {"Authorization": "Bearer lifecycle-user"}
    request = {
        "user_id": "lifecycle-user",
        "raw_query": "Remember that Project Atlas uses PostgreSQL.",
        "idempotency_key": "api-lifecycle-key",
    }

    first = client.post("/chat", json=request, headers=headers)
    replay = client.post("/chat", json=request, headers=headers)
    conflict = client.post(
        "/chat",
        json={
            **request,
            "raw_query": "Remember that Project Atlas uses SQLite.",
        },
        headers=headers,
    )

    assert first.status_code == 200
    assert first.json()["final_chat_text"] == "Stored exactly once."
    assert first.json()["actions_committed"] == [{"action_type": "add"}]
    assert replay.status_code == 200
    assert replay.json() == first.json()
    assert conflict.status_code == 409
    assert "idempotency_conflict" in conflict.json()["detail"]
    assert len(pipeline.requests) == 1
    assert limiter.keys == [
        "chat:lifecycle-user",
        "chat:lifecycle-user",
        "chat:lifecycle-user",
    ]
