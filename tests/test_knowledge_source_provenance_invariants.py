from __future__ import annotations

from typing import Any, Iterator

import pytest
from sqlalchemy import and_, update

from assistant_rag.database import SQLiteRepository, content_hash
from assistant_rag.postgres_repository import PostgresRepository
from assistant_rag.postgres_schema import knowledge_chunks


USER_ID = "source-provenance-user"
TOPIC_TITLE = "Imported observations"
SOURCE_A_TEXT = "The recorded observation has   ordinal 41."
SOURCE_B_TEXT = "  The recorded observation has ordinal 41.  "


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


def _create_distinct_owned_sources(repository: Any) -> tuple[str, str]:
    with repository.transaction() as cursor:
        source_a = repository.create_knowledge_source(
            cursor,
            user_id=USER_ID,
            filename="source-a.txt",
            file_type="text/plain",
            content_hash="whole-source-a-hash",
            processing_status="indexed",
        )
        source_b = repository.create_knowledge_source(
            cursor,
            user_id=USER_ID,
            filename="source-b.txt",
            file_type="text/plain",
            content_hash="whole-source-b-hash",
            processing_status="indexed",
        )
    return source_a, source_b


def _add_fact(repository: Any, *, source_id: str, text: str) -> str:
    with repository.transaction() as cursor:
        _topic_id, chunk_id, _outbox_job_id = repository.add_knowledge_chunk(
            cursor,
            user_id=USER_ID,
            title=TOPIC_TITLE,
            text=text,
            source_id=source_id,
        )
    return chunk_id


def test_equal_normalized_text_from_distinct_sources_keeps_distinct_provenance(
    repository: Any,
) -> None:
    source_a, source_b = _create_distinct_owned_sources(repository)

    chunk_a = _add_fact(repository, source_id=source_a, text=SOURCE_A_TEXT)
    chunk_b = _add_fact(repository, source_id=source_b, text=SOURCE_B_TEXT)

    assert chunk_b != chunk_a
    assert [
        fact["chunk_id"]
        for fact in repository.list_knowledge_facts(
            user_id=USER_ID,
            source_id=source_a,
        )
    ] == [chunk_a]
    assert [
        fact["chunk_id"]
        for fact in repository.list_knowledge_facts(
            user_id=USER_ID,
            source_id=source_b,
        )
    ] == [chunk_b]


def test_deleting_source_a_preserves_equal_durable_fact_from_source_b(
    repository: Any,
) -> None:
    source_a, source_b = _create_distinct_owned_sources(repository)
    chunk_a = _add_fact(repository, source_id=source_a, text=SOURCE_A_TEXT)
    chunk_b = _add_fact(repository, source_id=source_b, text=SOURCE_B_TEXT)

    repository.soft_delete_knowledge_source(
        user_id=USER_ID,
        source_id=source_a,
    )

    assert repository.list_knowledge_facts(
        user_id=USER_ID,
        source_id=source_a,
    ) == []
    source_b_facts = repository.list_knowledge_facts(
        user_id=USER_ID,
        source_id=source_b,
    )
    assert [fact["chunk_id"] for fact in source_b_facts] == [chunk_b]
    assert source_b_facts[0]["source_id"] == source_b
    assert chunk_b != chunk_a


def test_readding_through_source_b_does_not_reactivate_source_a_chunk(
    repository: Any,
) -> None:
    source_a, source_b = _create_distinct_owned_sources(repository)
    chunk_a = _add_fact(repository, source_id=source_a, text=SOURCE_A_TEXT)
    repository.soft_delete_knowledge_source(
        user_id=USER_ID,
        source_id=source_a,
    )

    chunk_b = _add_fact(repository, source_id=source_b, text=SOURCE_B_TEXT)

    assert chunk_b != chunk_a
    assert repository.list_knowledge_facts(
        user_id=USER_ID,
        source_id=source_a,
    ) == []
    source_a_history = repository.list_knowledge_facts(
        user_id=USER_ID,
        source_id=source_a,
        include_deleted=True,
    )
    assert [fact["chunk_id"] for fact in source_a_history] == [chunk_a]
    assert source_a_history[0]["is_deleted"] == 1

    source_b_facts = repository.list_knowledge_facts(
        user_id=USER_ID,
        source_id=source_b,
    )
    assert [fact["chunk_id"] for fact in source_b_facts] == [chunk_b]
    assert source_b_facts[0]["source_id"] == source_b


def test_legacy_source_chunk_hash_is_adopted_without_duplication(
    repository: Any,
) -> None:
    source_a, _source_b = _create_distinct_owned_sources(repository)
    chunk_id = _add_fact(
        repository,
        source_id=source_a,
        text=SOURCE_A_TEXT,
    )
    normalized = " ".join(SOURCE_A_TEXT.split())
    legacy_hash = content_hash(USER_ID, normalized)
    with repository.transaction() as cursor:
        if isinstance(repository, SQLiteRepository):
            cursor.execute(
                "UPDATE knowledge_chunks SET content_hash = ? WHERE chunk_id = ?",
                (legacy_hash, chunk_id),
            )
        else:
            cursor.execute(
                update(knowledge_chunks)
                .where(
                    and_(
                        knowledge_chunks.c.user_id == USER_ID,
                        knowledge_chunks.c.chunk_id == chunk_id,
                    )
                )
                .values(content_hash=legacy_hash)
            )

    repeated_chunk_id = _add_fact(
        repository,
        source_id=source_a,
        text=SOURCE_B_TEXT,
    )

    assert repeated_chunk_id == chunk_id
    facts = repository.list_knowledge_facts(
        user_id=USER_ID,
        source_id=source_a,
    )
    assert [fact["chunk_id"] for fact in facts] == [chunk_id]
    assert facts[0]["content_hash"] != legacy_hash
