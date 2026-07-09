from __future__ import annotations

from pathlib import Path

import pytest

from assistant_rag.artifacts import ArtifactGenerator
from assistant_rag.contracts import KnowledgeSourceStatus
from assistant_rag.database import SQLiteRepository
from assistant_rag.ingestion import KnowledgeIngestionService
from assistant_rag.settings import KnowledgeChunkSettings


def make_repo() -> SQLiteRepository:
    repo = SQLiteRepository.in_memory()
    repo.initialize_schema()
    return repo


def test_txt_ingestion_creates_source_chunks_and_outbox_jobs() -> None:
    repo = make_repo()
    service = KnowledgeIngestionService(
        repo,
        chunk_settings=KnowledgeChunkSettings(
            chunk_size_tokens=12,
            chunk_overlap_tokens=2,
            min_chunk_tokens=2,
            max_chunk_tokens=20,
        ),
    )

    result = service.ingest_text(
        user_id="u1",
        filename="facts.txt",
        text="My company address is 123 Main Street. Project Atlas retention is 30 days.",
        topic_title="Company",
    )

    assert result.status is KnowledgeSourceStatus.INDEXED
    assert result.source_id
    assert result.chunk_ids
    assert result.outbox_job_ids
    source = repo.get_knowledge_source(user_id="u1", source_id=result.source_id)
    assert source["processing_status"] == "indexed"
    facts = repo.list_knowledge_facts(user_id="u1", source_id=result.source_id)
    assert {fact["source_id"] for fact in facts} == {result.source_id}


def test_duplicate_source_upload_reuses_indexed_source() -> None:
    repo = make_repo()
    service = KnowledgeIngestionService(
        repo,
        chunk_settings=KnowledgeChunkSettings(12, 2, 2, 20),
    )
    first = service.ingest_text(
        user_id="u1",
        filename="facts.txt",
        text="Project Atlas retention is 30 days.",
        topic_title="Project Atlas",
    )
    second = service.ingest_text(
        user_id="u1",
        filename="facts-copy.txt",
        text="Project Atlas retention is 30 days.",
        topic_title="Project Atlas",
    )
    assert second.source_id == first.source_id
    assert len(repo.list_knowledge_sources(user_id="u1")) == 1


def test_source_delete_soft_deletes_child_chunks_and_reindex_uses_sql_truth() -> None:
    repo = make_repo()
    service = KnowledgeIngestionService(
        repo,
        chunk_settings=KnowledgeChunkSettings(12, 2, 2, 20),
    )
    result = service.ingest_text(
        user_id="u1",
        filename="facts.txt",
        text="One useful fact. Another useful fact. A third useful fact.",
        topic_title="Facts",
    )

    reindex_jobs = repo.reindex_knowledge_source(user_id="u1", source_id=result.source_id)
    assert reindex_jobs

    deleted = repo.soft_delete_knowledge_source(user_id="u1", source_id=result.source_id)
    assert deleted["processing_status"] == "deleted"
    assert deleted["outbox_job_ids"]
    assert repo.list_knowledge_facts(user_id="u1", source_id=result.source_id) == []
    assert repo.list_knowledge_facts(user_id="u1", source_id=result.source_id, include_deleted=True)

    with pytest.raises(ValueError):
        repo.restore_knowledge_chunk(user_id="u1", chunk_id=result.chunk_ids[0])


def test_knowledge_version_history_links_old_and_replacement_chunks() -> None:
    repo = make_repo()
    with repo.transaction() as cursor:
        _, chunk_id, _ = repo.add_knowledge_chunk(
            cursor,
            user_id="u1",
            title="Company",
            text="My company address is 123 Main Street",
        )

    update = repo.update_knowledge_chunk_text(
        user_id="u1",
        chunk_id=chunk_id,
        text="My company address is 456 Market Street",
        change_reason="address_changed",
        modified_by_user_query="change my company address",
    )
    old = repo.get_knowledge_chunks_by_ids("u1", [chunk_id], include_deleted=True)[0]
    new = repo.get_knowledge_chunks_by_ids("u1", [update["new_chunk_id"]], include_deleted=False)[0]

    assert old["is_deleted"] == 1
    assert old["replaced_by_chunk_id"] == update["new_chunk_id"]
    assert new["replaces_chunk_id"] == chunk_id
    assert new["is_deleted"] == 0
    assert len(update["outbox_job_ids"]) == 2


def test_artifact_generation_creates_real_files_and_sql_rows(tmp_path: Path) -> None:
    repo = make_repo()
    generator = ArtifactGenerator(repo, storage_dir=str(tmp_path), download_base_url="/artifacts")

    for file_type in ("xlsx", "pdf", "pptx"):
        artifact = generator.generate(
            user_id="u1",
            file_type=file_type,
            filename=f"report.{file_type}",
            content="Title\nFirst point\nSecond point",
        )
        assert artifact["status"] == "created"
        assert artifact["file_type"] == file_type
        assert Path(artifact["storage_path"]).exists()
        assert Path(artifact["storage_path"]).is_relative_to(tmp_path)

    assert len(repo.list_generated_artifacts(user_id="u1")) == 3
    first_artifact = repo.list_generated_artifacts(user_id="u1")[0]
    deleted = repo.delete_generated_artifact(user_id="u1", artifact_id=first_artifact["artifact_id"])
    assert deleted["status"] == "deleted"
    with pytest.raises(ValueError):
        repo.get_generated_artifact(user_id="u1", artifact_id=first_artifact["artifact_id"])


def test_pdf_ingestion_failure_marks_source_failed_when_extractor_unavailable() -> None:
    repo = make_repo()
    service = KnowledgeIngestionService(repo)
    result = service.ingest_bytes(
        user_id="u1",
        filename="broken.pdf",
        data=b"%PDF-1.4 not really a pdf",
        topic_title="Docs",
        file_type="pdf",
    )
    assert result.status is KnowledgeSourceStatus.FAILED
    if result.source_id:
        source = repo.get_knowledge_source(user_id="u1", source_id=result.source_id, include_deleted=True)
        assert source["processing_status"] == "failed"
        assert repo.list_knowledge_facts(user_id="u1", source_id=result.source_id, include_deleted=True) == []
