from __future__ import annotations

import ast
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from assistant_rag.contracts import (
    AuthContext,
    BundledResponse,
    ChatRequest,
    LastQAState,
    ResponseType,
)
from assistant_rag.api import build_api_app
from assistant_rag.auth import get_auth_context
from assistant_rag.database import SQLiteRepository
from assistant_rag.request_lifecycle import (
    ChatRequestLifecycleExecutor,
    RequestLifecycleConflict,
    looks_like_mutation,
)


def _response(
    text: str = "Request completed.",
    response_type: ResponseType = ResponseType.KNOWLEDGE_ACTION,
) -> BundledResponse:
    return BundledResponse(
        final_chat_text=text,
        response_type=response_type,
        last_qa_state=LastQAState(
            last_user_query="request",
            last_response=text,
            response_type=response_type,
        ),
        actions_committed=[{"action_type": "delete"}],
    )


class RecordingPipeline:
    def __init__(self, outcomes: list[BundledResponse | Exception]) -> None:
        self.outcomes = list(outcomes)
        self.requests: list[ChatRequest] = []

    def handle(self, request: ChatRequest, _repository: object) -> BundledResponse:
        self.requests.append(request)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture
def repository() -> SQLiteRepository:
    repository = SQLiteRepository.in_memory()
    repository.initialize_schema()
    return repository


def _pending_confirmation(
    repository: SQLiteRepository,
    *,
    user_id: str = "user-1",
    expires_delta: timedelta = timedelta(minutes=10),
) -> str:
    confirmation = repository.create_pending_confirmation(
        user_id=user_id,
        action_type="knowledge_mutation",
        target_entity_type="knowledge_chunk",
        target_entity_id="chunk-1",
        proposed_action={
            "domain": "knowledge",
            "actions": [
                {
                    "action": "delete",
                    "validation_result": "execute",
                    "target_chunk_ids": ["chunk-1"],
                }
            ],
            "action_authorization": {
                "intent": "knowledge_facts",
                "action": "delete",
                "matched_keywords": ["delete"],
            },
            "operation_response": "Deleted the knowledge item.",
            "topic_title": "Knowledge",
        },
        target_snapshot={"chunk_id": "chunk-1"},
        expires_at=(datetime.now(timezone.utc) + expires_delta).isoformat(),
    )
    return str(confirmation["confirmation_token"])


def _confirmation_status(repository: SQLiteRepository, token: str) -> str:
    row = repository.connection.execute(
        "SELECT status FROM pending_action_confirmations WHERE confirmation_token = ?",
        (token,),
    ).fetchone()
    return str(row["status"])


def _mutation_status(repository: SQLiteRepository, idempotency_key: str) -> str:
    row = repository.connection.execute(
        "SELECT status FROM mutation_requests WHERE idempotency_key = ?",
        (idempotency_key,),
    ).fetchone()
    return str(row["status"])


def test_confirmation_hydrates_exact_action_marks_confirmed_and_replays_once(
    repository: SQLiteRepository,
) -> None:
    token = _pending_confirmation(repository)
    pipeline = RecordingPipeline([_response("Deleted exactly once.")])
    executor = ChatRequestLifecycleExecutor(
        pipeline=pipeline,  # type: ignore[arg-type]
        repository=repository,
    )
    request = ChatRequest(
        user_id="user-1",
        raw_query="Confirm the pending action.",
        confirmation_token=token,
        idempotency_key=f"confirm:{token}",
    )

    first = executor.execute(request, fallback_request_id="fallback-first")

    assert not first.replayed
    assert first.response is not None
    assert first.request.metadata["intent"] == "knowledge_facts"
    assert first.request.metadata["confirmation_approved"] is True
    assert first.request.metadata["validated_knowledge_actions"] == [
        {
            "action": "delete",
            "validation_result": "execute",
            "target_chunk_ids": ["chunk-1"],
        }
    ]
    assert first.request.metadata["action_authorization"]["action"] == "delete"
    assert _confirmation_status(repository, token) == "confirmed"
    assert _mutation_status(repository, f"confirm:{token}") == "completed"

    second = executor.execute(request, fallback_request_id="fallback-second")

    assert second.replayed
    assert second.response is None
    assert second.payload["final_chat_text"] == "Deleted exactly once."
    assert len(pipeline.requests) == 1
    assert _confirmation_status(repository, token) == "confirmed"


def test_idempotency_conflict_does_not_corrupt_completed_request(
    repository: SQLiteRepository,
) -> None:
    pipeline = RecordingPipeline([_response("Created once.")])
    executor = ChatRequestLifecycleExecutor(
        pipeline=pipeline,  # type: ignore[arg-type]
        repository=repository,
    )
    original = ChatRequest(
        user_id="user-1",
        raw_query="Remember that Atlas uses PostgreSQL.",
        idempotency_key="same-key",
    )
    executor.execute(original, fallback_request_id="first")

    with pytest.raises(RequestLifecycleConflict, match="idempotency_conflict"):
        executor.execute(
            ChatRequest(
                user_id="user-1",
                raw_query="Remember that Atlas uses SQLite.",
                idempotency_key="same-key",
            ),
            fallback_request_id="second",
        )

    assert _mutation_status(repository, "same-key") == "completed"
    assert len(pipeline.requests) == 1


def test_keyed_general_delivery_artifact_and_conversation_side_effects_replay_once(
    repository: SQLiteRepository,
    tmp_path: Path,
) -> None:
    class GeneralSideEffectPipeline:
        def __init__(self) -> None:
            self.requests: list[ChatRequest] = []
            self.generated_files: list[Path] = []
            self.delivery_count = 0
            self.conversation_write_count = 0

        def handle(
            self, request: ChatRequest, _repository: object
        ) -> BundledResponse:
            self.requests.append(request)
            artifact_path = tmp_path / f"quarterly-report-{len(self.requests)}.pdf"
            artifact_path.write_bytes(b"%PDF-1.4\n% lifecycle test\n")
            self.generated_files.append(artifact_path)
            self.delivery_count += 1
            self.conversation_write_count += 1
            text = "Sent the generated quarterly report."
            return BundledResponse(
                final_chat_text=text,
                response_type=ResponseType.NORMAL,
                last_qa_state=LastQAState(
                    last_user_query=request.raw_query,
                    last_response=text,
                    response_type=ResponseType.NORMAL,
                ),
                conversation_topic_id="topic-1",
                conversation_hop_id="hop-1",
                platform_payload={
                    "artifacts": [
                        {
                            "artifact_id": "artifact-1",
                            "filename": "quarterly-report.pdf",
                            "file_type": "pdf",
                            "storage_path": str(artifact_path),
                        }
                    ],
                    "delivery": {
                        "channel": "gmail",
                        "status": "sent",
                        "recipient": "finance@example.com",
                    },
                },
            )

    pipeline = GeneralSideEffectPipeline()
    executor = ChatRequestLifecycleExecutor(
        pipeline=pipeline,  # type: ignore[arg-type]
        repository=repository,
    )
    request = ChatRequest(
        user_id="user-1",
        raw_query=(
            "Generate a PDF quarterly report and email it to finance@example.com."
        ),
        idempotency_key="general-file-delivery-key",
    )
    rate_limit_signals: list[tuple[bool, bool]] = []

    assert looks_like_mutation(request) is False
    first = executor.execute(
        request,
        fallback_request_id="fallback-first",
        before_claim=lambda _request, is_mutation, is_destructive: (
            rate_limit_signals.append((is_mutation, is_destructive))
        ),
    )
    replay = executor.execute(
        request,
        fallback_request_id="fallback-replay",
        before_claim=lambda _request, is_mutation, is_destructive: (
            rate_limit_signals.append((is_mutation, is_destructive))
        ),
    )

    assert first.response is not None
    assert first.is_mutation is False
    assert replay.response is None
    assert replay.replayed is True
    assert replay.is_mutation is False
    assert replay.request_id == first.request_id
    assert replay.payload == first.payload
    assert len(pipeline.requests) == 1
    assert pipeline.delivery_count == 1
    assert pipeline.conversation_write_count == 1
    assert pipeline.generated_files == [tmp_path / "quarterly-report-1.pdf"]
    assert pipeline.generated_files[0].exists()
    assert not (tmp_path / "quarterly-report-2.pdf").exists()
    assert _mutation_status(repository, "general-file-delivery-key") == "completed"
    # API mutation/destructive rate limits continue to use semantic mutation
    # classification even though the general request now owns an idempotency
    # claim.
    assert rate_limit_signals == [(False, False), (False, False)]


def test_reminder_confirmation_hydrates_reminder_branch_metadata(
    repository: SQLiteRepository,
) -> None:
    confirmation = repository.create_pending_confirmation(
        user_id="user-1",
        action_type="reminder_mutation",
        target_entity_type="reminder",
        target_entity_id="reminder-1",
        proposed_action={
            "domain": "reminder",
            "actions": [
                {
                    "action": "turn_off",
                    "validation_result": "execute",
                    "target_reminder_ids": ["reminder-1"],
                    "observed_status": "scheduled",
                    "observed_version": 1,
                }
            ],
            "action_authorization": {
                "intent": "reminder",
                "action": "turn_off",
                "matched_keywords": ["turn off"],
            },
            "operation_response": "Turned off the reminder.",
            "topic_title": "Reminders",
        },
        target_snapshot={"reminder_id": "reminder-1"},
        expires_at=(datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat(),
    )
    token = str(confirmation["confirmation_token"])
    pipeline = RecordingPipeline(
        [_response("Turned off the reminder.", ResponseType.REMINDER_ACTION)]
    )
    executor = ChatRequestLifecycleExecutor(
        pipeline=pipeline,  # type: ignore[arg-type]
        repository=repository,
    )

    result = executor.execute(
        ChatRequest(
            user_id="user-1",
            raw_query="Confirm the pending reminder change.",
            confirmation_token=token,
            idempotency_key=f"reminder:{token}",
        ),
        fallback_request_id="reminder",
    )

    expected_actions = [
        {
            "action": "turn_off",
            "validation_result": "execute",
            "target_reminder_ids": ["reminder-1"],
            "observed_status": "scheduled",
            "observed_version": 1,
        }
    ]
    assert result.request.metadata["intent"] == "reminder"
    assert result.request.metadata["validated_reminder_actions"] == expected_actions
    assert result.request.metadata["reminder_actions"] == expected_actions
    assert result.request.metadata["action_authorization"]["action"] == "turn_off"
    assert _confirmation_status(repository, token) == "confirmed"


def test_pipeline_failure_marks_idempotency_failed_and_retry_can_complete(
    repository: SQLiteRepository,
) -> None:
    pipeline = RecordingPipeline([RuntimeError("temporary failure"), _response()])
    executor = ChatRequestLifecycleExecutor(
        pipeline=pipeline,  # type: ignore[arg-type]
        repository=repository,
    )
    request = ChatRequest(
        user_id="user-1",
        raw_query="Remember that Atlas uses PostgreSQL.",
        idempotency_key="retry-key",
    )
    before_pipeline_queries: list[str] = []

    with pytest.raises(RuntimeError, match="temporary failure"):
        executor.execute(
            request,
            fallback_request_id="first",
            before_pipeline=lambda prepared: before_pipeline_queries.append(
                prepared.raw_query
            ),
        )
    assert _mutation_status(repository, "retry-key") == "failed"

    completed = executor.execute(
        request,
        fallback_request_id="second",
        before_pipeline=lambda prepared: before_pipeline_queries.append(
            prepared.raw_query
        ),
    )
    replayed = executor.execute(
        request,
        fallback_request_id="replay",
        before_pipeline=lambda prepared: before_pipeline_queries.append(
            prepared.raw_query
        ),
    )
    with pytest.raises(RequestLifecycleConflict, match="idempotency_conflict"):
        executor.execute(
            ChatRequest(
                user_id="user-1",
                raw_query="Remember that Atlas uses SQLite.",
                idempotency_key="retry-key",
            ),
            fallback_request_id="conflict",
            before_pipeline=lambda prepared: before_pipeline_queries.append(
                prepared.raw_query
            ),
        )

    assert completed.response is not None
    assert replayed.replayed
    assert _mutation_status(repository, "retry-key") == "completed"
    assert len(pipeline.requests) == 2
    assert before_pipeline_queries == [request.raw_query, request.raw_query]


def test_error_response_does_not_consume_pending_confirmation(
    repository: SQLiteRepository,
) -> None:
    token = _pending_confirmation(repository)
    pipeline = RecordingPipeline(
        [
            _response("The transaction failed.", ResponseType.ERROR),
            _response("Deleted after retry."),
        ]
    )
    executor = ChatRequestLifecycleExecutor(
        pipeline=pipeline,  # type: ignore[arg-type]
        repository=repository,
    )

    result = executor.execute(
        ChatRequest(
            user_id="user-1",
            raw_query="Confirm the pending action.",
            confirmation_token=token,
            idempotency_key=f"error:{token}",
        ),
        fallback_request_id="error",
    )

    assert result.response is not None
    assert result.response.response_type is ResponseType.ERROR
    assert _confirmation_status(repository, token) == "pending"
    assert _mutation_status(repository, f"error:{token}") == "failed"

    retried = executor.execute(
        ChatRequest(
            user_id="user-1",
            raw_query="Confirm the pending action.",
            confirmation_token=token,
            idempotency_key=f"error:{token}",
        ),
        fallback_request_id="retry",
    )

    assert retried.response is not None
    assert _confirmation_status(repository, token) == "confirmed"
    assert _mutation_status(repository, f"error:{token}") == "completed"


def test_expired_confirmation_fails_closed_and_marks_request_failed(
    repository: SQLiteRepository,
) -> None:
    token = _pending_confirmation(repository, expires_delta=timedelta(seconds=-1))
    pipeline = RecordingPipeline([_response()])
    executor = ChatRequestLifecycleExecutor(
        pipeline=pipeline,  # type: ignore[arg-type]
        repository=repository,
    )

    with pytest.raises(RequestLifecycleConflict, match="expired"):
        executor.execute(
            ChatRequest(
                user_id="user-1",
                raw_query="Confirm the pending action.",
                confirmation_token=token,
                idempotency_key=f"expired:{token}",
            ),
            fallback_request_id="expired",
        )

    assert _confirmation_status(repository, token) == "expired"
    assert _mutation_status(repository, f"expired:{token}") == "failed"
    assert not pipeline.requests


def test_mutation_without_idempotency_key_preserves_compatibility_warning(
    repository: SQLiteRepository,
) -> None:
    pipeline = RecordingPipeline([_response()])
    executor = ChatRequestLifecycleExecutor(
        pipeline=pipeline,  # type: ignore[arg-type]
        repository=repository,
    )

    result = executor.execute(
        ChatRequest(
            user_id="user-1",
            raw_query="Remember that Atlas uses PostgreSQL.",
        ),
        fallback_request_id="compatibility",
    )

    assert result.response is not None
    assert result.request.metadata["warnings"] == [
        "Mutation request had no idempotency_key; processed in compatibility mode."
    ]


def test_every_user_entrypoint_uses_the_shared_lifecycle_executor() -> None:
    root = Path(__file__).resolve().parents[1]
    entrypoints = (
        root / "assistant_rag" / "api.py",
        root / "debug_pipeline.py",
        root / "streamlit_app.py",
    )

    for entrypoint in entrypoints:
        tree = ast.parse(entrypoint.read_text(encoding="utf-8"))
        constructed_names = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "ChatRequestLifecycleExecutor" in constructed_names, entrypoint.name

    streamlit_tree = ast.parse(
        (root / "streamlit_app.py").read_text(encoding="utf-8")
    )
    confirmation_render_count = sum(
        1
        for node in ast.walk(streamlit_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_render_confirmation_controls"
    )
    assert confirmation_render_count >= 2


def test_api_confirmation_uses_shared_exactly_once_lifecycle(
    repository: SQLiteRepository,
) -> None:
    from fastapi.testclient import TestClient

    token = _pending_confirmation(repository)
    pipeline = RecordingPipeline([_response("API confirmation committed.")])
    app = build_api_app(
        pipeline,  # type: ignore[arg-type]
        repository_factory=lambda: repository,
    )
    app.dependency_overrides[get_auth_context] = lambda: AuthContext(user_id="user-1")
    client = TestClient(app)
    payload = {
        "user_id": "user-1",
        "raw_query": "Confirm the pending action.",
        "confirmation_token": token,
        "idempotency_key": f"api-confirm:{token}",
    }

    first = client.post("/chat", json=payload)
    second = client.post("/chat", json=payload)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["final_chat_text"] == "API confirmation committed."
    assert second.json()["final_chat_text"] == "API confirmation committed."
    assert len(pipeline.requests) == 1
    assert pipeline.requests[0].metadata["confirmation_approved"] is True
    assert pipeline.requests[0].metadata["intent"] == "knowledge_facts"
    assert _confirmation_status(repository, token) == "confirmed"
