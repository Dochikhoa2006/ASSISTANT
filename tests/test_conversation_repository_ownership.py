from __future__ import annotations

import inspect
from typing import Any

import pytest

from assistant_rag.database import SQLiteRepository
from assistant_rag.postgres_repository import PostgresRepository
from assistant_rag.repository import AssistantRepository


USER_ID = "conversation-owner"


def _sqlite_repository() -> SQLiteRepository:
    repository = SQLiteRepository.in_memory()
    repository.initialize_schema()
    return repository


def _append(
    repository: Any,
    cursor: Any,
    *,
    topic_id: str,
    user_id: str = USER_ID,
    parent_hop_id: str | None = None,
    branch_id: str | None = None,
    query: str = "Continue",
) -> Any:
    return repository.append_conversation_hop(
        cursor,
        topic_id=topic_id,
        user_id=user_id,
        intent="general_response",
        raw_user_query=query,
        rewritten_user_query=query,
        raw_response=f"Response to {query}",
        response_type="normal",
        parent_hop_id=parent_hop_id,
        branch_id=branch_id,
    )


def test_repository_protocol_requires_owned_conversation_persistence_apis() -> None:
    required = {
        "create_topic",
        "get_conversation_hop",
        "list_generated_artifacts_for_hop",
        "get_latest_platform_delivery_for_hop",
    }

    assert required <= AssistantRepository.__abstractmethods__
    assert not inspect.isabstract(SQLiteRepository)
    assert not inspect.isabstract(PostgresRepository)
    for implementation in (SQLiteRepository, PostgresRepository):
        for method_name in required:
            assert callable(getattr(implementation, method_name))


def test_create_topic_is_fresh_while_ensure_topic_still_reuses() -> None:
    repository = _sqlite_repository()

    with repository.transaction() as cursor:
        first = repository.create_topic(
            cursor,
            user_id=USER_ID,
            title="Repeated title",
        )
    with repository.transaction() as cursor:
        reused = repository.ensure_topic(
            cursor,
            user_id=USER_ID,
            title="Repeated title",
        )
        second = repository.create_topic(
            cursor,
            user_id=USER_ID,
            title="Repeated title",
        )

    assert reused == first
    assert second != first
    rows = repository.connection.execute(
        """
        SELECT topic_id FROM conversation_topics
        WHERE user_id = ? AND title = ? AND status = 'active'
        """,
        (USER_ID, "Repeated title"),
    ).fetchall()
    assert {str(row["topic_id"]) for row in rows} == {first, second}


def test_append_uses_branch_local_lineage_and_forks_an_older_parent() -> None:
    repository = _sqlite_repository()

    with repository.transaction() as cursor:
        topic_id = repository.create_topic(
            cursor,
            user_id=USER_ID,
            title="Branching",
        )
        root = _append(repository, cursor, topic_id=topic_id, query="Root")
        implicit_continuation = _append(
            repository,
            cursor,
            topic_id=topic_id,
            query="Continue current head",
        )
        explicit_continuation = _append(
            repository,
            cursor,
            topic_id=topic_id,
            parent_hop_id=implicit_continuation.hop_id,
            query="Explicit current head",
        )
        resumed = _append(
            repository,
            cursor,
            topic_id=topic_id,
            parent_hop_id=root.hop_id,
            query="Resume root",
        )
        resumed_continuation = _append(
            repository,
            cursor,
            topic_id=topic_id,
            query="Continue resumed branch",
        )

    rows = {
        hop_id: repository.get_conversation_hop(user_id=USER_ID, hop_id=hop_id)
        for hop_id in (
            root.hop_id,
            implicit_continuation.hop_id,
            explicit_continuation.hop_id,
            resumed.hop_id,
            resumed_continuation.hop_id,
        )
    }
    root_row = rows[root.hop_id]
    implicit_row = rows[implicit_continuation.hop_id]
    explicit_row = rows[explicit_continuation.hop_id]
    resumed_row = rows[resumed.hop_id]
    resumed_continuation_row = rows[resumed_continuation.hop_id]

    assert root_row["branch_id"] == root.hop_id
    assert root_row["previous_hop_id"] is None
    assert root_row["root_hop_id"] is None
    assert root_row["depth_from_root"] == 0

    assert implicit_row["branch_id"] == root.hop_id
    assert implicit_row["previous_hop_id"] == root.hop_id
    assert implicit_row["root_hop_id"] == root.hop_id
    assert implicit_row["depth_from_root"] == 1

    assert explicit_row["branch_id"] == root.hop_id
    assert explicit_row["parent_hop_id"] == implicit_continuation.hop_id
    assert explicit_row["previous_hop_id"] == implicit_continuation.hop_id
    assert explicit_row["depth_from_root"] == 2

    assert resumed_row["branch_id"] == resumed.hop_id
    assert resumed_row["branch_id"] != root_row["branch_id"]
    assert resumed_row["parent_hop_id"] == root.hop_id
    assert resumed_row["previous_hop_id"] == root.hop_id
    assert resumed_row["root_hop_id"] == root.hop_id
    assert resumed_row["depth_from_root"] == 1

    assert resumed_continuation_row["branch_id"] == resumed.hop_id
    assert resumed_continuation_row["previous_hop_id"] == resumed.hop_id
    assert resumed_continuation_row["root_hop_id"] == root.hop_id
    assert resumed_continuation_row["depth_from_root"] == 2


@pytest.mark.parametrize("backend", ("sqlite", "sqlalchemy"))
def test_reopening_a_branch_head_continues_it_after_other_branch_activity(
    backend: str,
) -> None:
    repository: Any
    if backend == "sqlite":
        repository = _sqlite_repository()
    else:
        repository = PostgresRepository.create("sqlite+pysqlite:///:memory:")
        repository.initialize_schema()

    try:
        with repository.transaction() as cursor:
            topic_id = repository.create_topic(
                cursor,
                user_id=USER_ID,
                title="Parallel branch heads",
            )
            root = _append(repository, cursor, topic_id=topic_id, query="Root")
            branch_a_head = _append(
                repository,
                cursor,
                topic_id=topic_id,
                query="Branch A head",
            )
            branch_b_head = _append(
                repository,
                cursor,
                topic_id=topic_id,
                parent_hop_id=root.hop_id,
                query="Branch B head",
            )
            branch_a_continuation = _append(
                repository,
                cursor,
                topic_id=topic_id,
                parent_hop_id=branch_a_head.hop_id,
                query="Continue A after B",
            )
            branch_b_continuation = _append(
                repository,
                cursor,
                topic_id=topic_id,
                parent_hop_id=branch_b_head.hop_id,
                query="Continue B after A",
            )

        rows = {
            hop_id: repository.get_conversation_hop(
                user_id=USER_ID,
                hop_id=hop_id,
            )
            for hop_id in (
                root.hop_id,
                branch_a_head.hop_id,
                branch_b_head.hop_id,
                branch_a_continuation.hop_id,
                branch_b_continuation.hop_id,
            )
        }
        assert rows[branch_a_continuation.hop_id]["branch_id"] == rows[
            branch_a_head.hop_id
        ]["branch_id"]
        assert rows[branch_b_continuation.hop_id]["branch_id"] == rows[
            branch_b_head.hop_id
        ]["branch_id"]
        assert rows[branch_a_continuation.hop_id]["previous_hop_id"] == (
            branch_a_head.hop_id
        )
        assert rows[branch_b_continuation.hop_id]["previous_hop_id"] == (
            branch_b_head.hop_id
        )
        assert repository.is_active_conversation_link(
            user_id=USER_ID,
            topic_id=topic_id,
            hop_id=branch_a_continuation.hop_id,
        )
        assert repository.is_active_conversation_link(
            user_id=USER_ID,
            topic_id=topic_id,
            hop_id=branch_b_continuation.hop_id,
        )
        assert not repository.is_active_conversation_link(
            user_id=USER_ID,
            topic_id=topic_id,
            hop_id=branch_a_head.hop_id,
        )

        with pytest.raises(ValueError, match="branch does not match"):
            with repository.transaction() as cursor:
                _append(
                    repository,
                    cursor,
                    topic_id=topic_id,
                    parent_hop_id=branch_a_continuation.hop_id,
                    branch_id="invented-branch-switch",
                )

        with repository.transaction() as cursor:
            fork_again = _append(
                repository,
                cursor,
                topic_id=topic_id,
                parent_hop_id=root.hop_id,
                query="Fork root again",
            )
        assert repository.get_conversation_hop(
            user_id=USER_ID,
            hop_id=fork_again.hop_id,
        )["branch_id"] == fork_again.hop_id
    finally:
        closer = getattr(repository, "close", None)
        if callable(closer):
            closer()


def test_append_rejects_foreign_cross_topic_archived_and_corrupt_head_links() -> None:
    repository = _sqlite_repository()
    other_user = "different-owner"

    with repository.transaction() as cursor:
        target_topic = repository.create_topic(
            cursor,
            user_id=USER_ID,
            title="Target",
        )
        target_root = _append(repository, cursor, topic_id=target_topic, query="Target root")
        sibling_topic = repository.create_topic(
            cursor,
            user_id=USER_ID,
            title="Sibling",
        )
        sibling_root = _append(repository, cursor, topic_id=sibling_topic, query="Sibling root")
        foreign_topic = repository.create_topic(
            cursor,
            user_id=other_user,
            title="Foreign",
        )
        foreign_root = _append(
            repository,
            cursor,
            topic_id=foreign_topic,
            user_id=other_user,
            query="Foreign root",
        )

    for invalid_parent in (sibling_root.hop_id, foreign_root.hop_id, "missing-hop"):
        with pytest.raises(ValueError, match="Active conversation hop"):
            with repository.transaction() as cursor:
                _append(
                    repository,
                    cursor,
                    topic_id=target_topic,
                    parent_hop_id=invalid_parent,
                )

    assert repository.get_conversation_hop(
        user_id=USER_ID,
        hop_id=target_root.hop_id,
    )["topic_id"] == target_topic

    with repository.transaction() as cursor:
        cursor.execute(
            "UPDATE conversation_topics SET status = 'archived' WHERE topic_id = ?",
            (target_topic,),
        )
    with pytest.raises(ValueError, match="Active conversation topic"):
        with repository.transaction() as cursor:
            _append(
                repository,
                cursor,
                topic_id=target_topic,
                parent_hop_id=target_root.hop_id,
            )

    with repository.transaction() as cursor:
        fresh_target = repository.create_topic(
            cursor,
            user_id=USER_ID,
            title="Corrupt head target",
        )
        cursor.execute(
            "UPDATE conversation_topics SET last_hop_id = ? WHERE topic_id = ?",
            (foreign_root.hop_id, fresh_target),
        )
    with pytest.raises(ValueError, match="Active conversation hop"):
        with repository.transaction() as cursor:
            _append(repository, cursor, topic_id=fresh_target)

    with repository.transaction() as cursor:
        root_scope_topic = repository.create_topic(
            cursor,
            user_id=USER_ID,
            title="Corrupt root target",
        )
        _append(repository, cursor, topic_id=root_scope_topic, query="Root")
        root_scope_head = _append(
            repository,
            cursor,
            topic_id=root_scope_topic,
            query="Head",
        )
        cursor.execute(
            "UPDATE conversation_hops SET root_hop_id = ? WHERE hop_id = ?",
            (foreign_root.hop_id, root_scope_head.hop_id),
        )
    with pytest.raises(ValueError, match="Active conversation hop"):
        with repository.transaction() as cursor:
            _append(repository, cursor, topic_id=root_scope_topic)


def test_hop_scoped_reads_and_writes_enforce_owner_and_restore_exact_state() -> None:
    repository = _sqlite_repository()
    other_user = "artifact-foreign-owner"

    with repository.transaction() as cursor:
        topic_id = repository.create_topic(
            cursor,
            user_id=USER_ID,
            title="Owned state",
        )
        hop = _append(repository, cursor, topic_id=topic_id, query="Create report")
        other_topic = repository.create_topic(
            cursor,
            user_id=other_user,
            title="Other state",
        )
        other_hop = _append(
            repository,
            cursor,
            topic_id=other_topic,
            user_id=other_user,
            query="Other report",
        )

    owned_hop = repository.get_conversation_hop(user_id=USER_ID, hop_id=hop.hop_id)
    assert owned_hop["topic_id"] == topic_id
    assert owned_hop["topic_status"] == "active"
    with pytest.raises(ValueError, match="Conversation hop"):
        repository.get_conversation_hop(user_id=other_user, hop_id=hop.hop_id)

    artifact = repository.create_generated_artifact(
        user_id=USER_ID,
        conversation_hop_id=hop.hop_id,
        file_type="pptx",
        filename="beach-plan.pptx",
        storage_path="/private/tmp/beach-plan.pptx",
        storage_url="/artifacts/beach-plan.pptx",
        metadata={"source": "test"},
    )
    assert [
        row["artifact_id"]
        for row in repository.list_generated_artifacts_for_hop(
            user_id=USER_ID,
            hop_id=hop.hop_id,
        )
    ] == [artifact["artifact_id"]]

    artifact_count = repository.table_count("generated_artifacts")
    with pytest.raises(ValueError, match="Conversation hop"):
        repository.create_generated_artifact(
            user_id=USER_ID,
            conversation_hop_id=other_hop.hop_id,
            file_type="pptx",
            filename="foreign.pptx",
            storage_path="/private/tmp/foreign.pptx",
            storage_url="/artifacts/foreign.pptx",
        )
    assert repository.table_count("generated_artifacts") == artifact_count
    with pytest.raises(ValueError, match="Conversation hop"):
        repository.list_generated_artifacts_for_hop(
            user_id=other_user,
            hop_id=hop.hop_id,
        )

    assert repository.get_latest_platform_delivery_for_hop(
        user_id=USER_ID,
        hop_id=hop.hop_id,
    ) is None
    repository.record_platform_delivery(
        user_id=USER_ID,
        conversation_hop_id=hop.hop_id,
        channel="gmail",
        status="drafted",
        recipient="first@example.com",
        message={"subject": "Draft", "body": "First"},
    )
    repository.record_platform_delivery(
        user_id=USER_ID,
        conversation_hop_id=hop.hop_id,
        channel="gmail",
        status="sent",
        recipient="final@example.com",
        message={
            "subject": "Day off",
            "body": "Final body",
            "attachments": [artifact["artifact_id"]],
        },
    )
    latest = repository.get_latest_platform_delivery_for_hop(
        user_id=USER_ID,
        hop_id=hop.hop_id,
    )
    assert latest is not None
    assert latest["status"] == "sent"
    assert latest["recipient"] == "final@example.com"
    assert latest["message"] == {
        "subject": "Day off",
        "body": "Final body",
        "attachments": [artifact["artifact_id"]],
    }

    delivery_count = repository.connection.execute(
        "SELECT COUNT(*) FROM platform_deliveries"
    ).fetchone()[0]
    with pytest.raises(ValueError, match="Conversation hop"):
        repository.record_platform_delivery(
            user_id=USER_ID,
            conversation_hop_id=other_hop.hop_id,
            channel="gmail",
            status="sent",
            recipient="wrong@example.com",
            message={"body": "Wrong owner"},
        )
    assert repository.connection.execute(
        "SELECT COUNT(*) FROM platform_deliveries"
    ).fetchone()[0] == delivery_count
    with pytest.raises(ValueError, match="Conversation hop"):
        repository.get_latest_platform_delivery_for_hop(
            user_id=other_user,
            hop_id=hop.hop_id,
        )


def test_sqlalchemy_repository_new_contract_round_trip_and_ownership() -> None:
    repository = PostgresRepository.create("sqlite+pysqlite:///:memory:")
    repository.initialize_schema()
    try:
        with repository.transaction() as cursor:
            first_topic = repository.create_topic(
                cursor,
                user_id=USER_ID,
                title="SQLAlchemy repeated",
            )
            second_topic = repository.create_topic(
                cursor,
                user_id=USER_ID,
                title="SQLAlchemy repeated",
            )
            hop = _append(repository, cursor, topic_id=second_topic, query="Create")

        assert first_topic != second_topic
        assert repository.get_conversation_hop(
            user_id=USER_ID,
            hop_id=hop.hop_id,
        )["branch_id"] == hop.hop_id

        artifact = repository.create_generated_artifact(
            user_id=USER_ID,
            conversation_hop_id=hop.hop_id,
            file_type="pdf",
            filename="owned.pdf",
            storage_path="/private/tmp/owned.pdf",
            storage_url="/artifacts/owned.pdf",
        )
        assert repository.list_generated_artifacts_for_hop(
            user_id=USER_ID,
            hop_id=hop.hop_id,
        )[0]["artifact_id"] == artifact["artifact_id"]

        repository.record_platform_delivery(
            user_id=USER_ID,
            conversation_hop_id=hop.hop_id,
            channel="gmail",
            status="sent",
            recipient="owner@example.com",
            message={"body": "Owned"},
        )
        assert repository.get_latest_platform_delivery_for_hop(
            user_id=USER_ID,
            hop_id=hop.hop_id,
        )["message"] == {"body": "Owned"}

        with pytest.raises(ValueError, match="Conversation hop"):
            repository.get_conversation_hop(
                user_id="different-owner",
                hop_id=hop.hop_id,
            )
    finally:
        repository.close()


@pytest.mark.parametrize("backend", ("sqlite", "sqlalchemy"))
def test_durable_conversation_loading_uses_exact_latest_branch(
    backend: str,
) -> None:
    repository: Any = (
        _sqlite_repository()
        if backend == "sqlite"
        else PostgresRepository.create("sqlite+pysqlite:///:memory:")
    )
    if backend == "sqlalchemy":
        repository.initialize_schema()
    try:
        with repository.transaction() as cursor:
            topic_id = repository.create_topic(
                cursor,
                user_id=USER_ID,
                title="Beach trip",
                entities={"conversation_id": "conversation-beach"},
            )
            root = repository.append_conversation_hop(
                cursor,
                topic_id=topic_id,
                user_id=USER_ID,
                intent="general_response",
                raw_user_query="Plan a beach trip.",
                rewritten_user_query="Plan a beach trip.",
                raw_response="Initial plan.",
                response_type="normal",
                entities={
                    "conversation_id": "conversation-beach",
                    "final_chat_text": "Initial plan.",
                },
            )
            abandoned = repository.append_conversation_hop(
                cursor,
                topic_id=topic_id,
                user_id=USER_ID,
                intent="general_response",
                raw_user_query="Use the mountain instead.",
                rewritten_user_query="Use the mountain instead.",
                raw_response="Mountain branch.",
                response_type="normal",
                entities={
                    "conversation_id": "conversation-beach",
                    "final_chat_text": "Mountain branch.",
                },
            )
            resumed = repository.append_conversation_hop(
                cursor,
                topic_id=topic_id,
                user_id=USER_ID,
                parent_hop_id=root.hop_id,
                intent="general_response",
                raw_user_query="Keep the beach plan.",
                rewritten_user_query="Keep the beach plan.",
                raw_response="Beach branch restored.",
                response_type="normal",
                entities={
                    "conversation_id": "conversation-beach",
                    "final_chat_text": "Beach branch restored.",
                },
            )

        listed = repository.list_conversations(user_id=USER_ID)
        loaded = repository.load_conversation(
            user_id=USER_ID,
            conversation_id="conversation-beach",
        )

        assert [item["conversation_id"] for item in listed] == [
            "conversation-beach"
        ]
        assert listed[0]["latest_hop_id"] == resumed.hop_id
        assert [message["content"] for message in loaded["messages"]] == [
            "Plan a beach trip.",
            "Initial plan.",
            "Keep the beach plan.",
            "Beach branch restored.",
        ]
        assert abandoned.hop_id not in {
            message.get("conversation_hop_id")
            for message in loaded["messages"]
        }
        assert repository.list_conversations(user_id="different-user") == []
        assert repository.load_conversation(
            user_id="different-user",
            conversation_id="conversation-beach",
        ) == {}
    finally:
        if backend == "sqlalchemy":
            repository.close()


@pytest.mark.parametrize("backend", ("sqlite", "sqlalchemy"))
def test_legacy_conversation_scope_can_be_claimed_only_once(backend: str) -> None:
    repository: Any = (
        _sqlite_repository()
        if backend == "sqlite"
        else PostgresRepository.create("sqlite+pysqlite:///:memory:")
    )
    if backend == "sqlalchemy":
        repository.initialize_schema()
    try:
        with repository.transaction() as cursor:
            topic_id = repository.create_topic(
                cursor,
                user_id=USER_ID,
                title="Legacy unscoped",
            )
            hop = _append(repository, cursor, topic_id=topic_id)

        assert repository.claim_conversation_scope(
            user_id=USER_ID,
            hop_id=hop.hop_id,
            conversation_id="conversation-a",
        ) is True
        assert repository.claim_conversation_scope(
            user_id=USER_ID,
            hop_id=hop.hop_id,
            conversation_id="conversation-a",
        ) is True
        assert repository.claim_conversation_scope(
            user_id=USER_ID,
            hop_id=hop.hop_id,
            conversation_id="conversation-b",
        ) is False
        owned = repository.get_conversation_hop(
            user_id=USER_ID,
            hop_id=hop.hop_id,
        )
        assert "conversation-a" in str(owned["entities_json"])
        assert "conversation-a" in str(owned["topic_entities_json"])
    finally:
        if backend == "sqlalchemy":
            repository.close()
