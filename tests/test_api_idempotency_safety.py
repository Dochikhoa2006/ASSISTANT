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
        return BundledResponse(
            final_chat_text="Handled.",
            response_type=ResponseType.NORMAL,
            last_qa_state=LastQAState(
                last_user_query=request.raw_query,
                last_response="Handled.",
                response_type=ResponseType.NORMAL,
            ),
        )


@dataclass
class RecordingRateLimiter:
    keys: list[str] = field(default_factory=list)

    def allow(self, key: str, _limit: int, _window_seconds: int) -> RateLimitResult:
        self.keys.append(key)
        return RateLimitResult(allowed=True)


def _client(
    monkeypatch,
    *,
    allow_missing: bool,
) -> tuple[TestClient, RecordingPipeline, RecordingRateLimiter]:
    monkeypatch.setenv(
        "ASSISTANT_ALLOW_MISSING_IDEMPOTENCY_KEY",
        "true" if allow_missing else "false",
    )
    repository = SQLiteRepository.in_memory()
    repository.initialize_schema()
    pipeline = RecordingPipeline()
    limiter = RecordingRateLimiter()
    app = build_api_app(
        pipeline=pipeline,  # type: ignore[arg-type]
        repository_factory=lambda: repository,
        rate_limiter=limiter,
    )
    return TestClient(app), pipeline, limiter


def test_allow_missing_idempotency_key_true_keeps_compatibility_mode(
    monkeypatch,
) -> None:
    client, pipeline, limiter = _client(monkeypatch, allow_missing=True)

    response = client.post(
        "/chat",
        json={
            "user_id": "safety-user",
            "raw_query": "Remember that Project Atlas uses PostgreSQL.",
        },
        headers={"Authorization": "Bearer safety-user"},
    )

    assert response.status_code == 200
    assert len(pipeline.requests) == 1
    assert pipeline.requests[0].metadata["warnings"] == [
        "Mutation request had no idempotency_key; processed in compatibility mode."
    ]
    assert limiter.keys == ["chat:safety-user", "mutation:safety-user"]


def test_allow_missing_idempotency_key_false_rejects_only_unkeyed_mutations(
    monkeypatch,
) -> None:
    client, pipeline, limiter = _client(monkeypatch, allow_missing=False)

    rejected = client.post(
        "/chat",
        json={
            "user_id": "safety-user",
            "raw_query": "Remember that Project Atlas uses PostgreSQL.",
        },
        headers={"Authorization": "Bearer safety-user"},
    )
    blank_key_rejected = client.post(
        "/chat",
        json={
            "user_id": "safety-user",
            "raw_query": "Remember that Project Atlas uses PostgreSQL.",
            "idempotency_key": "   ",
        },
        headers={"Authorization": "Bearer safety-user"},
    )
    ordinary_chat = client.post(
        "/chat",
        json={
            "user_id": "safety-user",
            "raw_query": "What color is the daytime sky?",
        },
        headers={"Authorization": "Bearer safety-user"},
    )
    keyed_mutation = client.post(
        "/chat",
        json={
            "user_id": "safety-user",
            "raw_query": "Remember that Project Atlas uses PostgreSQL.",
            "idempotency_key": "strict-mode-key",
        },
        headers={"Authorization": "Bearer safety-user"},
    )

    assert rejected.status_code == 400
    assert rejected.json()["detail"] == (
        "idempotency_key is required for mutation requests"
    )
    assert blank_key_rejected.status_code == 400
    assert blank_key_rejected.json() == rejected.json()
    assert ordinary_chat.status_code == 200
    assert keyed_mutation.status_code == 200
    assert [request.raw_query for request in pipeline.requests] == [
        "What color is the daytime sky?",
        "Remember that Project Atlas uses PostgreSQL.",
    ]
    assert limiter.keys == [
        "chat:safety-user",
        "chat:safety-user",
        "chat:safety-user",
        "chat:safety-user",
        "mutation:safety-user",
    ]
