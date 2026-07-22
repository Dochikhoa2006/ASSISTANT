from __future__ import annotations

from typing import Any, Iterator

import pytest

from assistant_rag.contracts import ChatRequest
from assistant_rag.database import SQLiteRepository
from assistant_rag.errors import KnowledgeConflictError, RepositoryValidationError
from assistant_rag.postgres_repository import PostgresRepository


@pytest.fixture(params=("sqlite", "sqlalchemy"))
def repository(request: pytest.FixtureRequest) -> Iterator[Any]:
    if request.param == "sqlite":
        value = SQLiteRepository.in_memory()
    else:
        value = PostgresRepository.create("sqlite+pysqlite:///:memory:")
    value.initialize_schema()
    try:
        yield value
    finally:
        if hasattr(value, "close"):
            value.close()
        else:
            value.connection.close()


@pytest.mark.parametrize("user_id", ("", "   ", "\n\t", None))
def test_chat_request_rejects_blank_or_missing_stable_identity(user_id: Any) -> None:
    with pytest.raises(ValueError, match="non-empty stable identity"):
        ChatRequest(user_id=user_id, raw_query="Recall my saved preference.")


def test_repository_rejects_blank_identity_before_knowledge_write(
    repository: Any,
) -> None:
    with pytest.raises(RepositoryValidationError, match="non-empty stable identity"):
        with repository.transaction() as cursor:
            repository.add_knowledge_chunk(
                cursor,
                user_id=" \t ",
                title="Invalid owner",
                text="This row must never be persisted.",
            )
    assert repository.table_count("knowledge_chunks") == 0


@pytest.mark.parametrize("text", ("", "   ", "\n\t"))
def test_repository_rejects_empty_normalized_knowledge_before_any_write(
    repository: Any,
    text: str,
) -> None:
    with pytest.raises(
        RepositoryValidationError,
        match="non-whitespace content",
    ):
        with repository.transaction() as cursor:
            repository.add_knowledge_chunk(
                cursor,
                user_id="empty-content-owner",
                title="Must not be created",
                text=text,
            )

    assert repository.table_count("knowledge_chunks") == 0
    assert repository.table_count("knowledge_topics") == 0
    assert repository.table_count("indexing_outbox") == 0


def test_knowledge_source_cannot_be_linked_across_owners(repository: Any) -> None:
    with repository.transaction() as cursor:
        source_id = repository.create_knowledge_source(
            cursor,
            user_id="source-owner",
            filename="notes.txt",
            file_type="text/plain",
            content_hash="source-hash",
        )

    with pytest.raises(KnowledgeConflictError, match="not found for user"):
        with repository.transaction() as cursor:
            repository.add_knowledge_chunk(
                cursor,
                user_id="different-owner",
                title="Imported notes",
                text="A fact that belongs to the source owner.",
                source_id=source_id,
            )

    assert repository.list_knowledge_facts(user_id="different-owner") == []


def test_knowledge_source_can_be_linked_by_its_owner(repository: Any) -> None:
    with repository.transaction() as cursor:
        source_id = repository.create_knowledge_source(
            cursor,
            user_id="source-owner",
            filename="notes.txt",
            file_type="text/plain",
            content_hash="source-hash",
        )
        _topic_id, chunk_id, outbox_job_id = repository.add_knowledge_chunk(
            cursor,
            user_id="source-owner",
            title="Imported notes",
            text="A fact that belongs to the source owner.",
            source_id=source_id,
        )

    facts = repository.list_knowledge_facts(user_id="source-owner")
    assert [fact["chunk_id"] for fact in facts] == [chunk_id]
    assert facts[0]["source_id"] == source_id
    assert outbox_job_id


def test_recovery_candidate_page_is_independent_of_index_fanout_state(
    repository: Any,
) -> None:
    with repository.transaction() as cursor:
        _topic_id, chunk_id, outbox_job_id = repository.add_knowledge_chunk(
            cursor,
            user_id="repair-owner",
            title="Repair state",
            text="A durable row awaiting derived index fanout.",
        )
    assert outbox_job_id

    pending = repository.list_knowledge_recovery_candidates(
        user_id="repair-owner",
        limit=5,
    )
    assert [row["chunk_id"] for row in pending] == [chunk_id]

    repository.mark_outbox_job_completed(job_id=outbox_job_id)
    completed = repository.list_knowledge_recovery_candidates(
        user_id="repair-owner",
        limit=5,
    )
    assert [row["chunk_id"] for row in completed] == [chunk_id]

    repository.mark_outbox_job_failed(
        job_id=outbox_job_id,
        error_message="derived cache drift",
    )
    failed = repository.list_knowledge_recovery_candidates(
        user_id="repair-owner",
        limit=5,
    )
    assert [row["chunk_id"] for row in failed] == [chunk_id]
