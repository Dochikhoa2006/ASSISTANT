"""Semantic, configuration-driven chunking for indexable knowledge text."""

from __future__ import annotations

import re

from .settings import KnowledgeChunkSettings


_PARAGRAPH_BOUNDARY = re.compile(r"\n\s*\n+")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def split_semantic_chunks(text: str, *, settings: KnowledgeChunkSettings) -> list[str]:
    """Keep paragraphs/sentences intact whenever they fit within policy limits."""
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []

    target_size = min(settings.chunk_size_tokens, settings.max_chunk_tokens)
    overlap_size = min(settings.chunk_overlap_tokens, max(target_size - 1, 0))
    units = _semantic_units(text, max_tokens=settings.max_chunk_tokens)
    chunks: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0

    for unit in units:
        unit_tokens = _token_count(unit)
        if current and current_tokens + unit_tokens > target_size:
            chunks.append(current)
            current = _overlap_units(current, overlap_size)
            current_tokens = sum(_token_count(item) for item in current)
        current.append(unit)
        current_tokens += unit_tokens
    if current:
        chunks.append(current)

    rendered = [" ".join(chunk).strip() for chunk in chunks if chunk]
    if len(rendered) > 1 and _token_count(rendered[-1]) < settings.min_chunk_tokens:
        rendered[-2] = f"{rendered[-2]} {rendered[-1]}".strip()
        rendered.pop()
    return rendered


def _semantic_units(text: str, *, max_tokens: int) -> list[str]:
    units: list[str] = []
    for paragraph in _PARAGRAPH_BOUNDARY.split(text):
        normalized = re.sub(r"\s+", " ", paragraph).strip()
        if not normalized:
            continue
        if _token_count(normalized) <= max_tokens:
            units.append(normalized)
            continue
        for sentence in _SENTENCE_BOUNDARY.split(normalized):
            sentence = sentence.strip()
            if not sentence:
                continue
            if _token_count(sentence) <= max_tokens:
                units.append(sentence)
                continue
            words = sentence.split()
            for start in range(0, len(words), max_tokens):
                units.append(" ".join(words[start : start + max_tokens]))
    return units


def _overlap_units(units: list[str], overlap_tokens: int) -> list[str]:
    overlap: list[str] = []
    covered = 0
    for unit in reversed(units):
        overlap.insert(0, unit)
        covered += _token_count(unit)
        if covered >= overlap_tokens:
            break
    return overlap


def _token_count(text: str) -> int:
    return len(text.split())
