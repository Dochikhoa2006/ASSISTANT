from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from assistant_rag.contracts import (
    ChatRequest,
    LastQAState,
    OutboundMessageState,
    ResponseType,
)
from assistant_rag.last_qa import DiskCacheLastQAStore, InMemoryLastQAStore
from assistant_rag.database import SQLiteRepository
from assistant_rag.pipeline import (
    AssistantPipeline,
    _last_qa_state_from_owned_hop,
    _outbound_state_from_platform,
    _rehydrate_outbound_artifacts,
)


def _state(label: str) -> LastQAState:
    return LastQAState(
        last_user_query=f"query-{label}",
        last_response=f"response-{label}",
        response_type=ResponseType.NORMAL,
        outbound_state=OutboundMessageState(
            channel="gmail",
            status="draft_ready",
            recipients=(f"{label}@example.com",),
            subject=f"subject-{label}",
            body=f"body-{label}",
        ),
    )


def test_in_memory_last_qa_is_scoped_by_user_and_conversation() -> None:
    store = InMemoryLastQAStore()
    conversation_a = _state("a")
    conversation_b = _state("b")

    store.save("same-user", conversation_a, conversation_id="conversation-a")
    store.save("same-user", conversation_b, conversation_id="conversation-b")

    assert store.get("same-user", conversation_id="conversation-a") is conversation_a
    assert store.get("same-user", conversation_id="conversation-b") is conversation_b
    assert store.get("same-user") is None


def test_disk_last_qa_is_scoped_without_overwriting_legacy_state(
    tmp_path: Path,
) -> None:
    store = DiskCacheLastQAStore(
        str(tmp_path / "last-qa.sqlite3"), ttl_seconds=60
    )
    legacy = _state("legacy")
    conversation_a = _state("a")
    conversation_b = _state("b")

    store.save("same-user", legacy)
    store.save("same-user", conversation_a, conversation_id="conversation-a")
    store.save("same-user", conversation_b, conversation_id="conversation-b")

    assert store.get("same-user") == legacy
    assert store.get("same-user", conversation_id="conversation-a") == conversation_a
    assert store.get("same-user", conversation_id="conversation-b") == conversation_b


def test_sent_outbound_envelope_remains_reusable_for_explicit_resend() -> None:
    previous = _state("team").outbound_state
    assert previous is not None

    restored = _outbound_state_from_platform(
        platform_payload={
            "delivery": {
                "channel": "gmail",
                "status": "sent",
                "recipient": "team@example.com",
            }
        },
        previous_state=previous,
        source_topic_id="topic-send",
        source_hop_id="hop-send",
    )

    assert restored is not None
    assert restored.status == "sent"
    assert restored.recipients == ("team@example.com",)
    assert restored.subject == "subject-team"
    assert restored.body == "body-team"
    assert restored.source_hop_id == "hop-send"


@dataclass
class _RestorationRepository:
    artifact: dict[str, Any]
    delivery: dict[str, Any] | None = None
    hop_artifact_calls: list[tuple[str, str]] = field(default_factory=list)

    def get_latest_platform_delivery_for_hop(
        self, *, user_id: str, hop_id: str
    ) -> dict[str, Any] | None:
        assert user_id == "owned-user"
        assert hop_id == "hop-owned"
        return self.delivery

    def list_generated_artifacts_for_hop(
        self, *, user_id: str, hop_id: str, include_deleted: bool = False
    ) -> list[dict[str, Any]]:
        assert include_deleted is False
        self.hop_artifact_calls.append((user_id, hop_id))
        return [self.artifact]

    def get_generated_artifact(
        self, *, user_id: str, artifact_id: str
    ) -> dict[str, Any]:
        assert user_id == "owned-user"
        assert artifact_id == self.artifact["artifact_id"]
        return self.artifact


def test_owned_hop_restores_durable_delivery_and_exact_artifact() -> None:
    artifact = {
        "artifact_id": "artifact-beach",
        "filename": "beach-plan.pptx",
        "storage_path": "/private/tmp/beach-plan.pptx",
        "status": "created",
    }
    repository = _RestorationRepository(
        artifact=artifact,
        delivery={
            "channel": "gmail",
            "status": "sent",
            "recipient": "vaiojjr@gmail.com, koffdo75@gmail.com",
                "message": {
                    "subject": "Day Off Notification",
                    "body": "I will be taking a day off next week.",
                    "recipients": [
                        "vaiojjr@gmail.com",
                        "koffdo75@gmail.com",
                    ],
                    "excluded_recipients": [],
                    "attachments": [artifact],
                },
        },
    )
    hop = {
        "hop_id": "hop-owned",
        "topic_id": "topic-owned",
        "rewritten_user_query": "send and attach that pptx",
        "raw_response": "Sent.",
        "response_type": "normal",
        "supporting_questions_json": "[]",
        "entities_json": '{"conversation_id":"conversation-a"}',
    }

    state = _last_qa_state_from_owned_hop(
        repository=repository, user_id="owned-user", hop=hop
    )
    available = _rehydrate_outbound_artifacts(
        repository=repository,
        user_id="owned-user",
        outbound_state=state.outbound_state,
        current_artifacts=[],
        source_hop_id="hop-owned",
    )

    assert state.linked_topic_id == "topic-owned"
    assert state.linked_hop_id == "hop-owned"
    assert state.outbound_state is not None
    assert state.outbound_state.status == "sent"
    assert state.outbound_state.recipients == (
        "vaiojjr@gmail.com",
        "koffdo75@gmail.com",
    )
    assert state.outbound_state.subject == "Day Off Notification"
    assert state.outbound_state.artifact_ids == ("artifact-beach",)
    assert [item["artifact_id"] for item in available] == ["artifact-beach"]


class _LineageRestorationRepository:
    def __init__(self, hops: dict[str, dict[str, Any]]) -> None:
        self.hops = hops

    def get_conversation_hop(
        self, *, user_id: str, hop_id: str
    ) -> dict[str, Any]:
        assert user_id == "owned-user"
        return dict(self.hops[hop_id])

    def get_latest_platform_delivery_for_hop(
        self, *, user_id: str, hop_id: str
    ) -> dict[str, Any] | None:
        assert user_id == "owned-user"
        return self.hops[hop_id].get("delivery")


def test_legacy_overbroad_audit_envelope_fails_closed_without_structured_polarity() -> None:
    first = {
        "hop_id": "hop-email",
        "topic_id": "topic-owned",
        "previous_hop_id": None,
        "rewritten_user_query": (
            "Write an email to alice@example.com and bob@example.com, but do not "
            "send it to minh@example.com."
        ),
        "raw_response": "Draft ready.",
        "response_type": "normal",
        "supporting_questions_json": "[]",
        "entities_json": '{"conversation_id":"conversation-a"}',
        # Legacy audit rows could contain an over-broad envelope. Restoration
        # must reapply the source turn's durable recipient constraints.
        "delivery": {
            "channel": "gmail",
            "status": "draft_ready",
            "recipient": "alice@example.com, bob@example.com, minh@example.com",
            "message": {
                "subject": "Day off",
                "body": "I will be away.",
                "recipients": [
                    "alice@example.com",
                    "bob@example.com",
                    "minh@example.com",
                ],
            },
        },
    }
    later = {
        "hop_id": "hop-hello",
        "topic_id": "topic-owned",
        "previous_hop_id": "hop-email",
        "rewritten_user_query": "hello",
        "raw_response": "Hello.",
        "response_type": "normal",
        "supporting_questions_json": "[]",
        "entities_json": '{"conversation_id":"conversation-a"}',
    }
    repository = _LineageRestorationRepository(
        {"hop-email": first, "hop-hello": later}
    )

    state = _last_qa_state_from_owned_hop(
        repository=repository,  # type: ignore[arg-type]
        user_id="owned-user",
        hop=later,
    )

    assert state.linked_hop_id == "hop-hello"
    assert state.outbound_state is None


class _Rewriter:
    def rewrite(self, query: str) -> str:
        return query


class _MustNotRun:
    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"unsafe stage ran after rejected parent: {name}")


@dataclass
class _ChatOutput:
    emitted: list[Any] = field(default_factory=list)

    def emit(self, response: Any) -> None:
        self.emitted.append(response)


class _RejectingRepository:
    def get_conversation_hop(self, *, user_id: str, hop_id: str) -> dict[str, Any]:
        raise ValueError("hop does not belong to user")


def test_foreign_explicit_parent_fails_closed_before_routing_or_side_effects() -> None:
    output = _ChatOutput()
    never = _MustNotRun()
    pipeline = AssistantPipeline(
        config=never,  # type: ignore[arg-type]
        last_qa_store=InMemoryLastQAStore(),
        query_rewriter=_Rewriter(),  # type: ignore[arg-type]
        last_qa_resolver=never,  # type: ignore[arg-type]
        retriever=never,  # type: ignore[arg-type]
        context_filter=never,  # type: ignore[arg-type]
        classifier=never,  # type: ignore[arg-type]
        router=never,  # type: ignore[arg-type]
        bundler=never,  # type: ignore[arg-type]
        platform_selector=never,  # type: ignore[arg-type]
        chat_output=output,  # type: ignore[arg-type]
    )

    response = pipeline.handle(
        ChatRequest(
            user_id="owned-user",
            raw_query="send it",
            conversation_id="conversation-a",
            parent_hop_id="foreign-hop",
        ),
        _RejectingRepository(),  # type: ignore[arg-type]
    )

    assert response.response_type is ResponseType.ERROR
    assert response.conversation_id == "conversation-a"
    assert response.warnings == ["selected_conversation_hop_rejected"]
    assert "No operation was executed" in response.final_chat_text
    assert output.emitted == [response]

    legacy_response = pipeline.handle(
        ChatRequest(
            user_id="owned-user",
            raw_query="send it",
            parent_hop_id="foreign-hop",
        ),
        _RejectingRepository(),  # type: ignore[arg-type]
    )

    assert legacy_response.response_type is ResponseType.ERROR
    assert legacy_response.conversation_id is None
    assert legacy_response.warnings == ["selected_conversation_hop_rejected"]
    assert output.emitted == [response, legacy_response]


def test_operation_audit_stays_in_selected_conversation_across_intent_titles() -> None:
    repository = SQLiteRepository.in_memory()
    repository.initialize_schema()
    with repository.transaction() as cursor:
        topic_id = repository.create_topic(
            cursor,
            user_id="owned-user",
            title="General Conversation",
            entities={"conversation_id": "conversation-a"},
        )
        root = repository.append_conversation_hop(
            cursor,
            topic_id=topic_id,
            user_id="owned-user",
            intent="general_response",
            raw_user_query="hello",
            rewritten_user_query="hello",
            raw_response="Hello.",
            response_type="normal",
            entities={"conversation_id": "conversation-a"},
        )

    continued = repository.record_action_audit_noop(
        user_id="owned-user",
        topic_title="Knowledge",
        raw_user_query="remember this",
        rewritten_user_query="remember this",
        response_text="No change required.",
        intent="knowledge_facts",
        response_type="safe_noop",
        parent_hop_id=root.hop_id,
        conversation_id="conversation-a",
    )
    separate = repository.record_action_audit_noop(
        user_id="owned-user",
        topic_title="Knowledge",
        raw_user_query="new chat operation",
        rewritten_user_query="new chat operation",
        response_text="No change required.",
        intent="knowledge_facts",
        response_type="safe_noop",
        conversation_id="conversation-b",
    )

    assert continued.committed is True
    assert continued.audit_topic_id == topic_id
    assert separate.committed is True
    assert separate.audit_topic_id != topic_id
    continued_hop = repository.get_conversation_hop(
        user_id="owned-user",
        hop_id=str(continued.audit_hop_id),
    )
    assert continued_hop["parent_hop_id"] == root.hop_id
    assert '"conversation_id": "conversation-a"' in continued_hop["entities_json"]

    mismatch = repository.record_action_audit_noop(
        user_id="owned-user",
        topic_title="Knowledge",
        raw_user_query="wrong cursor",
        rewritten_user_query="wrong cursor",
        response_text="No change required.",
        intent="knowledge_facts",
        response_type="safe_noop",
        parent_hop_id=root.hop_id,
        conversation_id="conversation-b",
    )
    assert mismatch.committed is False
