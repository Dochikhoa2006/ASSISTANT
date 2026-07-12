"""Batch knowledge ingestion service.

The service writes only SQL rows and outbox jobs. BM25/Chroma remain derived
stores handled by the background indexer.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from .contracts import IngestionResult, KnowledgeSourceStatus
from .database import now_iso
from .metrics import GLOBAL_METRICS
from .repository import AssistantRepository
from .settings import KnowledgeChunkSettings
from .semantic_chunking import split_semantic_chunks


def _content_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _chunk_text(text: str, *, chunk_size: int, overlap: int, min_size: int, max_size: int) -> list[str]:
    return split_semantic_chunks(
        text,
        settings=KnowledgeChunkSettings(
            chunk_size_tokens=chunk_size,
            chunk_overlap_tokens=overlap,
            min_chunk_tokens=min_size,
            max_chunk_tokens=max_size,
        ),
    )


def _extract_csv(data: bytes) -> str:
    text = data.decode("utf-8-sig")
    rows = list(csv.reader(io.StringIO(text)))
    lines = []
    for index, row in enumerate(rows):
        cells = [cell.strip() for cell in row]
        if index == 0:
            lines.append("Headers: " + ", ".join(cells))
        else:
            lines.append("Row: " + "; ".join(cells))
    return "\n".join(lines)


def _xml_texts(raw: bytes) -> list[str]:
    try:
        root = ElementTree.fromstring(raw)
    except ElementTree.ParseError:
        return []
    values: list[str] = []
    for elem in root.iter():
        if elem.text and elem.text.strip():
            values.append(elem.text.strip())
    return values


def _extract_xlsx(data: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared_strings = _xml_texts(archive.read("xl/sharedStrings.xml"))
        lines: list[str] = []
        for name in sorted(n for n in archive.namelist() if n.startswith("xl/worksheets/sheet") and n.endswith(".xml")):
            sheet_texts = _xml_texts(archive.read(name))
            cells = []
            for value in sheet_texts:
                if value.isdigit() and shared_strings:
                    idx = int(value)
                    cells.append(shared_strings[idx] if 0 <= idx < len(shared_strings) else value)
                else:
                    cells.append(value)
            if cells:
                lines.append(f"{Path(name).stem}: " + " | ".join(cells))
        return "\n".join(lines)


def _extract_pptx(data: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        lines: list[str] = []
        for name in sorted(n for n in archive.namelist() if n.startswith("ppt/slides/slide") and n.endswith(".xml")):
            texts = _xml_texts(archive.read(name))
            if texts:
                lines.append(f"{Path(name).stem}: " + " ".join(texts))
        return "\n".join(lines)


def _extract_pdf(data: bytes) -> str:
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError as exc:
        raise RuntimeError("PDF extraction requires pypdf") from exc
    reader = PdfReader(io.BytesIO(data))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


@dataclass
class KnowledgeIngestionService:
    repository: AssistantRepository
    chunk_settings: KnowledgeChunkSettings = KnowledgeChunkSettings()

    def ingest_bytes(
        self,
        *,
        user_id: str,
        filename: str,
        data: bytes,
        topic_title: str = "Knowledge",
        metadata: dict[str, Any] | None = None,
        file_type: str | None = None,
    ) -> IngestionResult:
        resolved_type = (file_type or Path(filename).suffix.lstrip(".") or "txt").casefold()
        source_hash = _content_sha256(data)
        source_id: str | None = None
        try:
            for existing in self.repository.list_knowledge_sources(user_id=user_id, include_deleted=False):
                if existing.get("content_hash") == source_hash and existing.get("processing_status") == KnowledgeSourceStatus.INDEXED.value:
                    chunk_ids = tuple(
                        fact["chunk_id"]
                        for fact in self.repository.list_knowledge_facts(
                            user_id=user_id,
                            include_deleted=False,
                            source_id=existing["source_id"],
                        )
                    )
                    GLOBAL_METRICS.increment("ingestion_success_total", file_type=resolved_type, deduped="true")
                    return IngestionResult(
                        source_id=existing["source_id"],
                        status=KnowledgeSourceStatus.INDEXED,
                        chunk_ids=chunk_ids,
                    )
            with self.repository.transaction() as cursor:
                source_id = self.repository.create_knowledge_source(
                    cursor,
                    user_id=user_id,
                    filename=filename,
                    file_type=resolved_type,
                    content_hash=source_hash,
                    metadata=metadata or {},
                    processing_status=KnowledgeSourceStatus.PENDING.value,
                )
                self.repository.update_knowledge_source_status(
                    cursor,
                    user_id=user_id,
                    source_id=source_id,
                    processing_status=KnowledgeSourceStatus.PROCESSING.value,
                )

            extracted = self.extract_text(data, resolved_type)
            chunks = _chunk_text(
                extracted,
                chunk_size=self.chunk_settings.chunk_size_tokens,
                overlap=self.chunk_settings.chunk_overlap_tokens,
                min_size=self.chunk_settings.min_chunk_tokens,
                max_size=self.chunk_settings.max_chunk_tokens,
            )
            if not chunks:
                raise RuntimeError("No indexable text was extracted")

            chunk_ids: list[str] = []
            outbox_ids: list[str] = []
            with self.repository.transaction() as cursor:
                for index, chunk in enumerate(chunks):
                    _, chunk_id, outbox_job_id = self.repository.add_knowledge_chunk(
                        cursor,
                        user_id=user_id,
                        title=topic_title,
                        text=chunk,
                        source_id=source_id,
                        metadata={
                            **(metadata or {}),
                            "source_filename": filename,
                            "source_file_type": resolved_type,
                            "chunk_index": index,
                            "chunking_strategy": "semantic_sentence_paragraph",
                            "semantic_chunk": True,
                        },
                    )
                    chunk_ids.append(chunk_id)
                    if outbox_job_id:
                        outbox_ids.append(outbox_job_id)
                self.repository.update_knowledge_source_status(
                    cursor,
                    user_id=user_id,
                    source_id=source_id,
                    processing_status=KnowledgeSourceStatus.INDEXED.value,
                    metadata={**(metadata or {}), "chunk_count": len(chunk_ids), "indexed_at": now_iso()},
                )
            GLOBAL_METRICS.increment("ingestion_success_total", file_type=resolved_type, deduped="false")
            return IngestionResult(
                source_id=source_id,
                status=KnowledgeSourceStatus.INDEXED,
                chunk_ids=tuple(chunk_ids),
                outbox_job_ids=tuple(outbox_ids),
            )
        except Exception as exc:
            if source_id:
                with self.repository.transaction() as cursor:
                    self.repository.update_knowledge_source_status(
                        cursor,
                        user_id=user_id,
                        source_id=source_id,
                        processing_status=KnowledgeSourceStatus.FAILED.value,
                        metadata={**(metadata or {}), "error": str(exc)},
                    )
            GLOBAL_METRICS.increment("ingestion_failure_total", file_type=resolved_type)
            return IngestionResult(
                source_id=source_id or "",
                status=KnowledgeSourceStatus.FAILED,
                error_message=str(exc),
            )

    def ingest_text(
        self,
        *,
        user_id: str,
        filename: str,
        text: str,
        topic_title: str = "Knowledge",
        metadata: dict[str, Any] | None = None,
    ) -> IngestionResult:
        return self.ingest_bytes(
            user_id=user_id,
            filename=filename,
            data=text.encode("utf-8"),
            topic_title=topic_title,
            metadata=metadata,
            file_type="txt",
        )

    def extract_text(self, data: bytes, file_type: str) -> str:
        if file_type in {"txt", "text"}:
            return data.decode("utf-8-sig")
        if file_type == "csv":
            return _extract_csv(data)
        if file_type == "xlsx":
            return _extract_xlsx(data)
        if file_type == "pptx":
            return _extract_pptx(data)
        if file_type == "pdf":
            return _extract_pdf(data)
        raise RuntimeError(f"Unsupported file type: {file_type}")
