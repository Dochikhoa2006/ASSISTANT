"""Embedding adapters."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Protocol

from .settings import EmbeddingSettings


class EmbeddingClient(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


@dataclass
class SentenceTransformerEmbeddingClient:
    settings: EmbeddingSettings

    def __post_init__(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Install sentence-transformers for production embeddings") from exc
        kwargs = {}
        if self.settings.device:
            kwargs["device"] = self.settings.device
        self.model = SentenceTransformer(self.settings.model_name, **kwargs)
        if self.settings.max_length:
            self.model.max_seq_length = self.settings.max_length

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            batch_size=self.settings.batch_size,
            normalize_embeddings=self.settings.normalize_embeddings,
        )
        return [list(map(float, vector)) for vector in vectors]

    def warmup(self) -> None:
        """Execute one encode so runtime kernels are ready before a user query."""
        self.embed(["assistant model warmup"])


@dataclass(frozen=True)
class HashEmbeddingClient:
    dimensions: int = 32

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        digest = sha256(text.encode("utf-8")).digest()
        values: list[float] = []
        while len(values) < self.dimensions:
            for byte in digest:
                values.append((float(byte) / 127.5) - 1.0)
                if len(values) == self.dimensions:
                    break
            digest = sha256(digest).digest()
        return values
