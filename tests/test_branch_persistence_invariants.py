from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy import func, select

from assistant_rag.branches import BranchRouter
from assistant_rag.bundler import ChatOutput, ResponseBundler
from assistant_rag.config import OutboxConfig
from assistant_rag.contracts import (
    ActionValidationResult,
    BranchResult,
    ChatRequest,
    GeneratedQuestion,
    Intent,
    KnowledgeAction,
    LastQAPath,
    LastQAResolution,
    PipelineContext,
    QuestionSource,
    ResponseType,
    ValidatedKnowledgeAction,
)
from assistant_rag.database import SQLiteRepository
from assistant_rag.indexing import BackgroundIndexer
from assistant_rag.errors import RepositoryValidationError
from assistant_rag.pipeline import AssistantPipeline
from assistant_rag.postgres_repository import PostgresRepository
from assistant_rag.postgres_schema import (
    conversation_hops,
    indexing_outbox,
    knowledge_chunks,
    knowledge_topics,
)


USER_ID = "branch-persistence-user"


class _MemoryIndex:
    def __init__(self) -> None:
        self.documents: dict[str, dict[str, Any]] = {}

    def search(self, **_kwargs: Any) -> list[Any]:
        return []

    def upsert(self, **kwargs: Any) -> None:
        self.documents[str(kwargs["entity_id"])] = dict(kwargs)

    def delete(self, *, entity_id: str) -> None:
        self.documents.pop(entity_id, None)

    def clear(self) -> None:
        self.documents.clear()


class _FlakyMemoryIndex(_MemoryIndex):
    def __init__(self, failures: int) -> None:
        super().__init__()
        self.failures = failures

    def upsert(self, **kwargs: Any) -> None:
        if self.failures > 0:
            self.failures -= 1
            raise RuntimeError("transient derived-index failure")
        super().upsert(**kwargs)


@dataclass
class _StaticBranch:
    result: BranchResult | None = None
    error: Exception | None = None

    def execute(self, _context: PipelineContext, _repository: Any) -> BranchResult:
        if self.error is not None:
            raise self.error
        assert self.result is not None
        return self.result


def _repository() -> SQLiteRepository:
    repository = SQLiteRepository.in_memory()
    repository.initialize_schema()
    return repository


def _context(intent: Intent, query: str = "Handle this turn") -> PipelineContext:
    return PipelineContext(
        request=ChatRequest(user_id=USER_ID, raw_query=query),
        rewritten_query=query,
        last_qa_state=None,
        conversation_results=[],
        intent=intent,
    )


def _outbox_config() -> OutboxConfig:
    return OutboxConfig(
        max_attempts=4,
        batch_size=64,
        retry_backoff_seconds=0,
        processing_timeout_seconds=180,
    )


def _sync(
    repository: SQLiteRepository,
    bm25: _MemoryIndex,
    chroma: _MemoryIndex,
    job_ids: list[str],
) -> int:
    return BackgroundIndexer(
        repository=repository,
        bm25=bm25,
        chroma=chroma,
        config=_outbox_config(),
    ).process_job_ids(job_ids)


def _result_variant(name: str, intent: Intent) -> tuple[BranchResult | None, Exception | None]:
    if name == "normal":
        response_type = {
            Intent.GENERAL_RESPONSE: ResponseType.NORMAL,
            Intent.KNOWLEDGE_FACTS: ResponseType.KNOWLEDGE_ACTION,
            Intent.REMINDER: ResponseType.REMINDER_ACTION,
        }[intent]
        return (
            BranchResult(
                response_type=response_type,
                normal_response_text=f"{intent.value} completed",
            ),
            None,
        )
    if name == "clarification":
        return (
            BranchResult(
                response_type=ResponseType.CLARIFICATION,
                clarification_question=GeneratedQuestion(
                    text="Which exact item should I use?",
                    source=QuestionSource.CLARIFICATION_QUESTION,
                    purpose="resolve_target",
                    confidence=1.0,
                ),
            ),
            None,
        )
    if name == "error":
        return (
            BranchResult(
                response_type=ResponseType.ERROR,
                fallback_or_error_message="The branch failed safely.",
            ),
            None,
        )
    if name == "pending_confirmation":
        return (
            BranchResult(
                response_type=(
                    ResponseType.REMINDER_ACTION
                    if intent is Intent.REMINDER
                    else ResponseType.KNOWLEDGE_ACTION
                    if intent is Intent.KNOWLEDGE_FACTS
                    else ResponseType.NORMAL
                ),
                normal_response_text="Please confirm this operation.",
                actions_pending_confirmation=[{"confirmation_token": "token"}],
            ),
            None,
        )
    if name == "exception":
        return None, RuntimeError("internal branch failure")
    raise AssertionError(name)


@pytest.mark.parametrize(
    "intent",
    (Intent.GENERAL_RESPONSE, Intent.KNOWLEDGE_FACTS, Intent.REMINDER),
)
@pytest.mark.parametrize(
    "variant",
    ("normal", "clarification", "error", "pending_confirmation", "exception"),
)
def test_every_reached_main_branch_outcome_persists_and_indexes_one_hop(
    intent: Intent,
    variant: str,
) -> None:
    repository = _repository()
    branch_result, branch_error = _result_variant(variant, intent)
    router = BranchRouter(
        {intent: _StaticBranch(result=branch_result, error=branch_error)}
    )

    result = router.route(_context(intent), repository)

    assert result.linked_topic_id
    assert result.linked_hop_id
    rows = repository.connection.execute(
        "SELECT * FROM conversation_hops"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["hop_id"] == result.linked_hop_id
    assert rows[0]["intent"] == intent.value
    assert rows[0]["response_type"] == result.response_type.value
    assert json.loads(rows[0]["entities_json"])["branch_outcome"] == {
        "intent": intent.value,
        "response_type": result.response_type.value,
    }

    job_ids = result.indexing_job_result["outbox_job_ids"]
    assert len(job_ids) == 1
    bm25 = _MemoryIndex()
    chroma = _MemoryIndex()
    assert _sync(repository, bm25, chroma, job_ids) == 1
    assert result.linked_hop_id in bm25.documents
    assert result.linked_hop_id in chroma.documents
    status = repository.connection.execute(
        "SELECT status FROM indexing_outbox WHERE job_id = ?",
        (job_ids[0],),
    ).fetchone()["status"]
    assert status == "completed"


def test_top_level_clarification_branch_remains_exempt_from_hop_persistence() -> None:
    repository = _repository()
    router = BranchRouter(
        {
            Intent.CLARIFICATION: _StaticBranch(
                result=BranchResult(
                    response_type=ResponseType.CLARIFICATION,
                    clarification_question=GeneratedQuestion(
                        text="What should I clarify?",
                        source=QuestionSource.CLARIFICATION_QUESTION,
                        purpose="resolve_missing_info",
                        confidence=1.0,
                    ),
                )
            )
        }
    )

    result = router.route(_context(Intent.CLARIFICATION), repository)

    assert result.linked_hop_id is None
    assert repository.table_count("conversation_hops") == 0
    assert repository.table_count("indexing_outbox") == 0


class _Store:
    def __init__(self) -> None:
        self.saved: list[Any] = []

    def get(self, _user_id: str) -> None:
        return None

    def save(self, _user_id: str, state: Any) -> None:
        self.saved.append(state)


class _PipelineRetriever:
    def __init__(self, bm25: _MemoryIndex, chroma: _MemoryIndex) -> None:
        self.bm25 = bm25
        self.chroma = chroma

    def retrieve_conversation(self, **_kwargs: Any) -> list[Any]:
        return []


class _Classifier:
    def __init__(self, intent: Intent) -> None:
        self.intent = intent

    def classify(self, *_args: Any, **_kwargs: Any) -> Intent:
        return self.intent


class _PlatformSelector:
    def select(self, _bundled: Any, _request: ChatRequest) -> dict[str, Any]:
        return {"delivery": {"channel": "none", "status": "not_requested"}}


@pytest.mark.parametrize(
    ("intent", "branch_result"),
    (
        (
            Intent.GENERAL_RESPONSE,
            BranchResult(
                response_type=ResponseType.NORMAL,
                normal_response_text="General response",
            ),
        ),
        (
            Intent.KNOWLEDGE_FACTS,
            BranchResult(
                response_type=ResponseType.ERROR,
                fallback_or_error_message="Knowledge failed safely",
            ),
        ),
        (
            Intent.REMINDER,
            BranchResult(
                response_type=ResponseType.CLARIFICATION,
                clarification_question=GeneratedQuestion(
                    text="Which reminder?",
                    source=QuestionSource.CLARIFICATION_QUESTION,
                    purpose="resolve_target",
                    confidence=1.0,
                ),
            ),
        ),
    ),
)
def test_pipeline_synchronously_fans_current_branch_hop_to_both_indexes(
    intent: Intent,
    branch_result: BranchResult,
) -> None:
    repository = _repository()
    bm25 = _MemoryIndex()
    chroma = _MemoryIndex()
    pipeline = AssistantPipeline(
        config=SimpleNamespace(
            context_filter=SimpleNamespace(
                conversation_retrieval_after_last_qa_enabled=True,
                conversation_retrieval_before_intent_enabled=False,
            ),
            outbox=_outbox_config(),
        ),
        last_qa_store=_Store(),
        query_rewriter=SimpleNamespace(rewrite=lambda query: query),
        last_qa_resolver=SimpleNamespace(
            resolve=lambda *_args: LastQAResolution(
                path=LastQAPath.NO_LAST_QA,
                rewritten_query="Handle this turn",
                state=None,
                did_merge_query=False,
                skip_broad_retrieval=True,
            )
        ),
        retriever=_PipelineRetriever(bm25, chroma),  # type: ignore[arg-type]
        context_filter=SimpleNamespace(),
        classifier=_Classifier(intent),
        router=BranchRouter({intent: _StaticBranch(result=branch_result)}),
        bundler=ResponseBundler(),
        platform_selector=_PlatformSelector(),
        chat_output=ChatOutput(),
    )

    response = pipeline.handle(
        ChatRequest(user_id=USER_ID, raw_query="Handle this turn"),
        repository,
    )

    assert response.conversation_hop_id in bm25.documents
    assert response.conversation_hop_id in chroma.documents
    assert repository.connection.execute(
        "SELECT status FROM indexing_outbox"
    ).fetchone()["status"] == "completed"


def _knowledge_action(
    action: KnowledgeAction,
    *,
    text: str | None = None,
    target_chunk_id: str | None = None,
    target_topic_id: str | None = None,
    expected_version: int | None = None,
) -> ValidatedKnowledgeAction:
    target_ids = (target_chunk_id,) if target_chunk_id else ()
    topic_ids = (target_topic_id,) if target_topic_id else ()
    observed = (
        {target_chunk_id: expected_version}
        if target_chunk_id and expected_version is not None
        else {}
    )
    return ValidatedKnowledgeAction(
        action=action,
        validation_result=ActionValidationResult.EXECUTE,
        target_chunk_ids=target_ids,
        target_topic_ids=topic_ids,
        observed_versions=observed,
        observed_is_deleted={target_chunk_id: False} if target_chunk_id else {},
        knowledge_text=text if action is KnowledgeAction.ADD else None,
        replacement_text=text if action is KnowledgeAction.MODIFY else None,
        new_text=text if action in {KnowledgeAction.ADD, KnowledgeAction.MODIFY} else None,
        topic_title="Atlas Facts" if action is KnowledgeAction.ADD else None,
        target_description="Atlas retention",
        confidence=1.0,
        reason_summary=f"Validated {action.value}.",
    )


def _knowledge_transaction(
    repository: SQLiteRepository,
    action: ValidatedKnowledgeAction,
    query: str,
) -> Any:
    result = repository.transactional_knowledge_actions(
        user_id=USER_ID,
        topic_title="Knowledge Audit",
        raw_user_query=query,
        rewritten_user_query=query,
        response_text=f"Completed {action.action.value}",
        actions=[action],
    )
    assert result.committed
    assert result.audit_topic_id
    assert result.audit_hop_id
    return result


def _assert_knowledge_hop(
    repository: SQLiteRepository,
    *,
    hop_id: str,
    action: KnowledgeAction,
    topic_id: str,
    chunk_id: str,
    target_chunk_ids: list[str],
) -> None:
    row = repository.connection.execute(
        "SELECT intent, response_type, entities_json FROM conversation_hops WHERE hop_id = ?",
        (hop_id,),
    ).fetchone()
    assert row["intent"] == Intent.KNOWLEDGE_FACTS.value
    assert row["response_type"] == ResponseType.KNOWLEDGE_ACTION.value
    entity = json.loads(row["entities_json"])["knowledge"]
    assert entity == [
        {
            "entity_type": "knowledge_chunk",
            "action": action.value,
            "knowledge_topic_id": topic_id,
            "knowledge_chunk_id": chunk_id,
            "target_chunk_ids": target_chunk_ids,
            "status": "committed",
        }
    ]


def test_add_modify_delete_preserve_topic_hops_chunks_and_both_indexes() -> None:
    repository = _repository()
    bm25 = _MemoryIndex()
    chroma = _MemoryIndex()

    add_result = _knowledge_transaction(
        repository,
        _knowledge_action(
            KnowledgeAction.ADD,
            text="Atlas retention is 30 days.",
        ),
        "Remember Atlas retention is 30 days.",
    )
    added_chunk_id = add_result.results[0].domain_entity_id
    assert added_chunk_id
    added = repository.connection.execute(
        "SELECT * FROM knowledge_chunks WHERE chunk_id = ?",
        (added_chunk_id,),
    ).fetchone()
    topic_id = added["knowledge_topic_id"]
    topic = repository.connection.execute(
        "SELECT * FROM knowledge_topics WHERE knowledge_topic_id = ?",
        (topic_id,),
    ).fetchone()
    assert topic["title"] == "Atlas Facts"
    assert topic["version"] == 2
    _assert_knowledge_hop(
        repository,
        hop_id=add_result.audit_hop_id,
        action=KnowledgeAction.ADD,
        topic_id=topic_id,
        chunk_id=added_chunk_id,
        target_chunk_ids=[],
    )
    assert _sync(
        repository,
        bm25,
        chroma,
        list(add_result.indexing_outbox_ids),
    ) == 2
    for index in (bm25, chroma):
        assert added_chunk_id in index.documents
        assert add_result.audit_hop_id in index.documents

    modify_result = _knowledge_transaction(
        repository,
        _knowledge_action(
            KnowledgeAction.MODIFY,
            text="Atlas retention is 45 days.",
            target_chunk_id=added_chunk_id,
            target_topic_id=topic_id,
            expected_version=int(added["version"]),
        ),
        "Change Atlas retention to 45 days.",
    )
    modified_chunk_id = modify_result.results[0].domain_entity_id
    assert modified_chunk_id and modified_chunk_id != added_chunk_id
    rows = repository.connection.execute(
        "SELECT * FROM knowledge_chunks ORDER BY created_at"
    ).fetchall()
    old = next(row for row in rows if row["chunk_id"] == added_chunk_id)
    new = next(row for row in rows if row["chunk_id"] == modified_chunk_id)
    assert old["is_deleted"] == 1
    assert old["replaced_by_chunk_id"] == modified_chunk_id
    assert new["is_deleted"] == 0
    assert new["replaces_chunk_id"] == added_chunk_id
    assert new["knowledge_topic_id"] == topic_id
    assert repository.table_count("knowledge_topics") == 1
    assert repository.connection.execute(
        "SELECT version FROM knowledge_topics WHERE knowledge_topic_id = ?",
        (topic_id,),
    ).fetchone()["version"] == 3
    _assert_knowledge_hop(
        repository,
        hop_id=modify_result.audit_hop_id,
        action=KnowledgeAction.MODIFY,
        topic_id=topic_id,
        chunk_id=modified_chunk_id,
        target_chunk_ids=[added_chunk_id],
    )
    assert _sync(
        repository,
        bm25,
        chroma,
        list(modify_result.indexing_outbox_ids),
    ) == 3
    for index in (bm25, chroma):
        assert added_chunk_id not in index.documents
        assert modified_chunk_id in index.documents
        assert modify_result.audit_hop_id in index.documents

    delete_result = _knowledge_transaction(
        repository,
        _knowledge_action(
            KnowledgeAction.DELETE,
            target_chunk_id=modified_chunk_id,
            target_topic_id=topic_id,
            expected_version=int(new["version"]),
        ),
        "Delete the Atlas retention fact.",
    )
    deleted = repository.connection.execute(
        "SELECT * FROM knowledge_chunks WHERE chunk_id = ?",
        (modified_chunk_id,),
    ).fetchone()
    assert deleted["is_deleted"] == 1
    assert repository.connection.execute(
        "SELECT COUNT(*) FROM knowledge_chunks WHERE is_deleted = 0"
    ).fetchone()[0] == 0
    assert repository.connection.execute(
        "SELECT version FROM knowledge_topics WHERE knowledge_topic_id = ?",
        (topic_id,),
    ).fetchone()["version"] == 4
    _assert_knowledge_hop(
        repository,
        hop_id=delete_result.audit_hop_id,
        action=KnowledgeAction.DELETE,
        topic_id=topic_id,
        chunk_id=modified_chunk_id,
        target_chunk_ids=[modified_chunk_id],
    )
    assert _sync(
        repository,
        bm25,
        chroma,
        list(delete_result.indexing_outbox_ids),
    ) == 2
    for index in (bm25, chroma):
        assert modified_chunk_id not in index.documents
        assert delete_result.audit_hop_id in index.documents

    assert repository.connection.execute(
        "SELECT COUNT(*) FROM conversation_hops WHERE intent = ?",
        (Intent.KNOWLEDGE_FACTS.value,),
    ).fetchone()[0] == 3
    assert repository.connection.execute(
        "SELECT COUNT(*) FROM indexing_outbox WHERE status != 'completed'"
    ).fetchone()[0] == 0


def test_sqlalchemy_repository_preserves_the_same_knowledge_invariants() -> None:
    repository = PostgresRepository.create("sqlite+pysqlite:///:memory:")
    repository.initialize_schema()
    bm25 = _MemoryIndex()
    chroma = _MemoryIndex()
    try:
        add_result = _knowledge_transaction(
            repository,  # type: ignore[arg-type]
            _knowledge_action(
                KnowledgeAction.ADD,
                text="Atlas retention is 30 days.",
            ),
            "Remember Atlas retention is 30 days.",
        )
        added_chunk_id = add_result.results[0].domain_entity_id
        assert added_chunk_id
        with repository.engine.connect() as connection:
            added = connection.execute(
                select(knowledge_chunks).where(
                    knowledge_chunks.c.chunk_id == added_chunk_id
                )
            ).mappings().one()
            topic_id = str(added["knowledge_topic_id"])
            assert connection.execute(
                select(knowledge_topics.c.version).where(
                    knowledge_topics.c.knowledge_topic_id == topic_id
                )
            ).scalar_one() == 2
        assert _sync(
            repository,  # type: ignore[arg-type]
            bm25,
            chroma,
            list(add_result.indexing_outbox_ids),
        ) == 2

        modify_result = _knowledge_transaction(
            repository,  # type: ignore[arg-type]
            _knowledge_action(
                KnowledgeAction.MODIFY,
                text="Atlas retention is 45 days.",
                target_chunk_id=added_chunk_id,
                target_topic_id=topic_id,
                expected_version=int(added["version"]),
            ),
            "Change Atlas retention to 45 days.",
        )
        modified_chunk_id = modify_result.results[0].domain_entity_id
        assert modified_chunk_id
        with repository.engine.connect() as connection:
            old = connection.execute(
                select(knowledge_chunks).where(
                    knowledge_chunks.c.chunk_id == added_chunk_id
                )
            ).mappings().one()
            new = connection.execute(
                select(knowledge_chunks).where(
                    knowledge_chunks.c.chunk_id == modified_chunk_id
                )
            ).mappings().one()
            assert old["is_deleted"] == 1
            assert old["replaced_by_chunk_id"] == modified_chunk_id
            assert new["knowledge_topic_id"] == topic_id
            assert new["replaces_chunk_id"] == added_chunk_id
            assert new["chunk_index"] == 1
            assert connection.execute(
                select(knowledge_topics.c.version).where(
                    knowledge_topics.c.knowledge_topic_id == topic_id
                )
            ).scalar_one() == 3
        assert _sync(
            repository,  # type: ignore[arg-type]
            bm25,
            chroma,
            list(modify_result.indexing_outbox_ids),
        ) == 3
        for index in (bm25, chroma):
            assert added_chunk_id not in index.documents
            assert modified_chunk_id in index.documents

        delete_result = _knowledge_transaction(
            repository,  # type: ignore[arg-type]
            _knowledge_action(
                KnowledgeAction.DELETE,
                target_chunk_id=modified_chunk_id,
                target_topic_id=topic_id,
                expected_version=int(new["version"]),
            ),
            "Delete the Atlas retention fact.",
        )
        assert _sync(
            repository,  # type: ignore[arg-type]
            bm25,
            chroma,
            list(delete_result.indexing_outbox_ids),
        ) == 2
        with repository.engine.connect() as connection:
            assert connection.execute(
                select(func.count()).select_from(knowledge_topics)
            ).scalar_one() == 1
            assert connection.execute(
                select(knowledge_topics.c.version).where(
                    knowledge_topics.c.knowledge_topic_id == topic_id
                )
            ).scalar_one() == 4
            assert connection.execute(
                select(func.count())
                .select_from(conversation_hops)
                .where(
                    conversation_hops.c.intent
                    == Intent.KNOWLEDGE_FACTS.value
                )
            ).scalar_one() == 3
            assert connection.execute(
                select(func.count())
                .select_from(knowledge_chunks)
                .where(knowledge_chunks.c.is_deleted == 0)
            ).scalar_one() == 0
            assert connection.execute(
                select(func.count())
                .select_from(indexing_outbox)
                .where(indexing_outbox.c.status != "completed")
            ).scalar_one() == 0
        for index in (bm25, chroma):
            assert modified_chunk_id not in index.documents
            assert add_result.audit_hop_id in index.documents
            assert modify_result.audit_hop_id in index.documents
            assert delete_result.audit_hop_id in index.documents
    finally:
        repository.close()


def test_same_text_modify_rolls_back_without_self_referential_lineage() -> None:
    repository = _repository()
    add_result = _knowledge_transaction(
        repository,
        _knowledge_action(KnowledgeAction.ADD, text="Atlas retention is 30 days."),
        "Remember Atlas retention is 30 days.",
    )
    chunk_id = str(add_result.results[0].domain_entity_id)
    chunk = repository.connection.execute(
        "SELECT * FROM knowledge_chunks WHERE chunk_id = ?",
        (chunk_id,),
    ).fetchone()

    result = repository.transactional_knowledge_actions(
        user_id=USER_ID,
        topic_title="Knowledge Audit",
        raw_user_query="Keep the same Atlas fact.",
        rewritten_user_query="Keep the same Atlas fact.",
        response_text="No change",
        actions=[
            _knowledge_action(
                KnowledgeAction.MODIFY,
                text="Atlas retention is 30 days.",
                target_chunk_id=chunk_id,
                target_topic_id=str(chunk["knowledge_topic_id"]),
                expected_version=int(chunk["version"]),
            )
        ],
    )

    assert not result.committed
    persisted = repository.connection.execute(
        "SELECT * FROM knowledge_chunks WHERE chunk_id = ?",
        (chunk_id,),
    ).fetchone()
    assert persisted["is_deleted"] == 0
    assert persisted["replaces_chunk_id"] is None
    assert persisted["replaced_by_chunk_id"] is None
    assert repository.table_count("knowledge_chunks") == 1
    assert repository.connection.execute(
        "SELECT COUNT(*) FROM conversation_hops WHERE intent = ?",
        (Intent.KNOWLEDGE_FACTS.value,),
    ).fetchone()[0] == 1


def test_sqlalchemy_same_text_modify_rolls_back_without_self_lineage() -> None:
    repository = PostgresRepository.create("sqlite+pysqlite:///:memory:")
    repository.initialize_schema()
    try:
        add_result = _knowledge_transaction(
            repository,  # type: ignore[arg-type]
            _knowledge_action(
                KnowledgeAction.ADD,
                text="Atlas retention is 30 days.",
            ),
            "Remember Atlas retention is 30 days.",
        )
        chunk_id = str(add_result.results[0].domain_entity_id)
        with repository.engine.connect() as connection:
            chunk = connection.execute(
                select(knowledge_chunks).where(
                    knowledge_chunks.c.chunk_id == chunk_id
                )
            ).mappings().one()
        result = repository.transactional_knowledge_actions(
            user_id=USER_ID,
            topic_title="Knowledge Audit",
            raw_user_query="Keep the same Atlas fact.",
            rewritten_user_query="Keep the same Atlas fact.",
            response_text="No change",
            actions=[
                _knowledge_action(
                    KnowledgeAction.MODIFY,
                    text="Atlas retention is 30 days.",
                    target_chunk_id=chunk_id,
                    target_topic_id=str(chunk["knowledge_topic_id"]),
                    expected_version=int(chunk["version"]),
                )
            ],
        )
        assert not result.committed
        with repository.engine.connect() as connection:
            persisted = connection.execute(
                select(knowledge_chunks).where(
                    knowledge_chunks.c.chunk_id == chunk_id
                )
            ).mappings().one()
            assert persisted["is_deleted"] == 0
            assert persisted["replaces_chunk_id"] is None
            assert persisted["replaced_by_chunk_id"] is None
            assert connection.execute(
                select(func.count()).select_from(knowledge_chunks)
            ).scalar_one() == 1
    finally:
        repository.close()


def test_delayed_delete_job_converges_to_reactivated_sql_knowledge() -> None:
    repository = _repository()
    bm25 = _MemoryIndex()
    chroma = _MemoryIndex()
    indexer = BackgroundIndexer(
        repository=repository,
        bm25=bm25,
        chroma=chroma,
        config=_outbox_config(),
    )

    add_result = _knowledge_transaction(
        repository,
        _knowledge_action(KnowledgeAction.ADD, text="Atlas retention is 30 days."),
        "Remember Atlas retention is 30 days.",
    )
    chunk_id = str(add_result.results[0].domain_entity_id)
    assert indexer.process_job_ids(list(add_result.indexing_outbox_ids)) == 2
    current = repository.connection.execute(
        "SELECT * FROM knowledge_chunks WHERE chunk_id = ?",
        (chunk_id,),
    ).fetchone()

    delete_result = _knowledge_transaction(
        repository,
        _knowledge_action(
            KnowledgeAction.DELETE,
            target_chunk_id=chunk_id,
            target_topic_id=str(current["knowledge_topic_id"]),
            expected_version=int(current["version"]),
        ),
        "Delete Atlas retention.",
    )
    delete_job_id = next(
        job_id
        for job_id in delete_result.indexing_outbox_ids
        if repository.connection.execute(
            "SELECT entity_type FROM indexing_outbox WHERE job_id = ?",
            (job_id,),
        ).fetchone()["entity_type"]
        == "knowledge_chunk"
    )

    readd_result = _knowledge_transaction(
        repository,
        _knowledge_action(KnowledgeAction.ADD, text="Atlas retention is 30 days."),
        "Remember Atlas retention again.",
    )
    assert readd_result.results[0].domain_entity_id == chunk_id
    assert indexer.process_job_ids(list(readd_result.indexing_outbox_ids)) == 2
    for index in (bm25, chroma):
        assert chunk_id in index.documents

    # The older DELETE is processed after reactivation. It must converge to
    # current SQL state and therefore upsert, never remove, the active chunk.
    assert indexer.process_job_ids([delete_job_id]) == 1
    for index in (bm25, chroma):
        assert chunk_id in index.documents
    assert repository.connection.execute(
        "SELECT is_deleted FROM knowledge_chunks WHERE chunk_id = ?",
        (chunk_id,),
    ).fetchone()["is_deleted"] == 0


def test_sqlalchemy_delayed_delete_converges_to_reactivated_knowledge() -> None:
    repository = PostgresRepository.create("sqlite+pysqlite:///:memory:")
    repository.initialize_schema()
    bm25 = _MemoryIndex()
    chroma = _MemoryIndex()
    indexer = BackgroundIndexer(
        repository=repository,
        bm25=bm25,
        chroma=chroma,
        config=_outbox_config(),
    )
    try:
        add_result = _knowledge_transaction(
            repository,  # type: ignore[arg-type]
            _knowledge_action(
                KnowledgeAction.ADD,
                text="Atlas retention is 30 days.",
            ),
            "Remember Atlas retention is 30 days.",
        )
        chunk_id = str(add_result.results[0].domain_entity_id)
        assert indexer.process_job_ids(list(add_result.indexing_outbox_ids)) == 2
        with repository.engine.connect() as connection:
            current = connection.execute(
                select(knowledge_chunks).where(
                    knowledge_chunks.c.chunk_id == chunk_id
                )
            ).mappings().one()
        delete_result = _knowledge_transaction(
            repository,  # type: ignore[arg-type]
            _knowledge_action(
                KnowledgeAction.DELETE,
                target_chunk_id=chunk_id,
                target_topic_id=str(current["knowledge_topic_id"]),
                expected_version=int(current["version"]),
            ),
            "Delete Atlas retention.",
        )
        with repository.engine.connect() as connection:
            delete_job_id = next(
                job_id
                for job_id in delete_result.indexing_outbox_ids
                if connection.execute(
                    select(indexing_outbox.c.entity_type).where(
                        indexing_outbox.c.job_id == job_id
                    )
                ).scalar_one()
                == "knowledge_chunk"
            )
        readd_result = _knowledge_transaction(
            repository,  # type: ignore[arg-type]
            _knowledge_action(
                KnowledgeAction.ADD,
                text="Atlas retention is 30 days.",
            ),
            "Remember Atlas retention again.",
        )
        assert readd_result.results[0].domain_entity_id == chunk_id
        assert indexer.process_job_ids(list(readd_result.indexing_outbox_ids)) == 2
        assert indexer.process_job_ids([delete_job_id]) == 1
        for index in (bm25, chroma):
            assert chunk_id in index.documents
        with repository.engine.connect() as connection:
            assert connection.execute(
                select(knowledge_chunks.c.is_deleted).where(
                    knowledge_chunks.c.chunk_id == chunk_id
                )
            ).scalar_one() == 0
    finally:
        repository.close()


def test_request_scoped_indexing_retries_until_both_indexes_succeed() -> None:
    repository = _repository()
    result = repository.record_branch_outcome(
        user_id=USER_ID,
        topic_title="General",
        raw_user_query="Hello",
        rewritten_user_query="Hello",
        response_text="Hi",
        intent=Intent.GENERAL_RESPONSE.value,
        response_type=ResponseType.NORMAL.value,
    )
    bm25 = _MemoryIndex()
    chroma = _FlakyMemoryIndex(failures=1)

    assert _sync(
        repository,
        bm25,
        chroma,
        list(result.indexing_outbox_ids),
    ) == 1
    assert result.audit_hop_id in bm25.documents
    assert result.audit_hop_id in chroma.documents
    job = repository.connection.execute(
        "SELECT status, retry_count FROM indexing_outbox WHERE job_id = ?",
        (result.indexing_outbox_ids[0],),
    ).fetchone()
    assert job["status"] == "completed"
    assert job["retry_count"] == 1


def test_request_scoped_indexing_never_reports_partial_store_success() -> None:
    repository = _repository()
    result = repository.record_branch_outcome(
        user_id=USER_ID,
        topic_title="General",
        raw_user_query="Hello",
        rewritten_user_query="Hello",
        response_text="Hi",
        intent=Intent.GENERAL_RESPONSE.value,
        response_type=ResponseType.NORMAL.value,
    )
    bm25 = _MemoryIndex()
    chroma = _FlakyMemoryIndex(failures=100)

    with pytest.raises(RuntimeError, match="did not complete"):
        _sync(
            repository,
            bm25,
            chroma,
            list(result.indexing_outbox_ids),
        )
    job = repository.connection.execute(
        "SELECT status, retry_count FROM indexing_outbox WHERE job_id = ?",
        (result.indexing_outbox_ids[0],),
    ).fetchone()
    assert job["status"] == "failed"
    assert job["retry_count"] == _outbox_config().max_attempts
    assert result.audit_hop_id in bm25.documents
    assert result.audit_hop_id not in chroma.documents


def test_sqlite_knowledge_topic_identity_is_unique_per_user_and_title() -> None:
    repository = _repository()
    with repository.transaction() as cursor:
        first = repository.ensure_knowledge_topic(
            cursor,
            user_id=USER_ID,
            title="Atlas Facts",
        )
    with repository.transaction() as cursor:
        repeated = repository.ensure_knowledge_topic(
            cursor,
            user_id=USER_ID,
            title="Atlas Facts",
        )
        other_user = repository.ensure_knowledge_topic(
            cursor,
            user_id="other-user",
            title="Atlas Facts",
        )

    assert repeated == first
    assert other_user != first
    assert repository.table_count("knowledge_topics") == 2
    indexes = repository.connection.execute(
        "PRAGMA index_list(knowledge_topics)"
    ).fetchall()
    assert any(
        row["name"] == "uq_knowledge_topics_user_title" and row["unique"] == 1
        for row in indexes
    )


def test_sqlalchemy_knowledge_topic_identity_is_unique_per_user_and_title() -> None:
    repository = PostgresRepository.create("sqlite+pysqlite:///:memory:")
    repository.initialize_schema()
    try:
        with repository.transaction() as cursor:
            first = repository.ensure_knowledge_topic(
                cursor,
                user_id=USER_ID,
                title="Atlas Facts",
            )
        with repository.transaction() as cursor:
            repeated = repository.ensure_knowledge_topic(
                cursor,
                user_id=USER_ID,
                title="Atlas Facts",
            )
            other_user = repository.ensure_knowledge_topic(
                cursor,
                user_id="other-user",
                title="Atlas Facts",
            )
        assert repeated == first
        assert other_user != first
        assert repository.table_count("knowledge_topics") == 2
    finally:
        repository.close()


def test_sqlite_schema_upgrade_fails_closed_on_legacy_duplicate_topics() -> None:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.execute(
        """
        CREATE TABLE knowledge_topics (
            knowledge_topic_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            entities_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            version INTEGER NOT NULL
        )
        """
    )
    connection.executemany(
        """
        INSERT INTO knowledge_topics VALUES (?, ?, ?, '', '{}', ?, ?, 1)
        """,
        (
            ("topic-one", USER_ID, "Atlas Facts", "now", "now"),
            ("topic-two", USER_ID, "Atlas Facts", "now", "now"),
        ),
    )
    connection.commit()
    repository = SQLiteRepository(connection)

    with pytest.raises(RepositoryValidationError, match="legacy duplicate"):
        repository.initialize_schema()
    assert repository.connection.execute(
        "SELECT COUNT(*) FROM knowledge_topics"
    ).fetchone()[0] == 2
