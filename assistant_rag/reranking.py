"""Production cross-encoder reranker."""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import isfinite
import time
from typing import Iterable
from urllib import error, request

from .contracts import RetrievalResult
from .settings import RerankerSettings


@dataclass
class SentenceTransformerCrossEncoderReranker:
    settings: RerankerSettings

    def __post_init__(self) -> None:
        if self.settings.endpoint_url:
            self.remote = HTTPReranker(self.settings)
            return
        self.remote = None
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError("Install sentence-transformers for production reranking") from exc
        kwargs = {}
        if self.settings.device:
            kwargs["device"] = self.settings.device
        self.model = CrossEncoder(self.settings.model_name, **kwargs)

    def rerank(
        self,
        query: str,
        results: Iterable[RetrievalResult],
        *,
        enforce_min_score: bool = True,
    ) -> list[RetrievalResult]:
        if self.remote is not None:
            return self.remote.rerank(
                query,
                results,
                enforce_min_score=enforce_min_score,
            )
        candidates = list(results)[: self.settings.max_candidates]
        pairs = [(query, str(item.payload.get("text", ""))) for item in candidates]
        if not pairs:
            return []
        scores = self.model.predict(pairs, batch_size=self.settings.batch_size)
        reranked = []
        for result, score in zip(candidates, scores, strict=False):
            value = float(score)
            reranked.append(
                RetrievalResult(
                    entity_type=result.entity_type,
                    entity_id=result.entity_id,
                    source_store_evidence=result.source_store_evidence,
                    rerank_score=value,
                    confidence=value,
                    validation_status=result.validation_status,
                    payload=result.payload,
                )
            )
        return [
            item
            for item in sorted(reranked, key=lambda item: item.rerank_score, reverse=True)
            if isfinite(item.rerank_score)
            and (
                not enforce_min_score
                or item.rerank_score >= self.settings.min_score
            )
        ]

    def warmup(self) -> None:
        """Execute the smallest valid scoring operation at process startup."""
        if self.remote is not None:
            # A remote reranker owns its own model lifecycle. There is no
            # empty request contract we can safely assume here.
            return
        self.model.predict([("assistant model warmup", "assistant model warmup")], batch_size=1)


@dataclass
class HTTPReranker:
    settings: RerankerSettings

    def rerank(
        self,
        query: str,
        results: Iterable[RetrievalResult],
        *,
        enforce_min_score: bool = True,
    ) -> list[RetrievalResult]:
        candidates = list(results)[: self.settings.max_candidates]
        if not candidates:
            return []
        started = time.perf_counter()
        body = {
            "model": self.settings.model_name,
            "query": query,
            "documents": [str(item.payload.get("text", "")) for item in candidates],
        }
        try:
            scores = self._score(body)
        except Exception:
            # A lexical/vector ordering is not a cross-encoder result.  Do not
            # let it bypass the canonical final-ranking stage when the remote
            # cross-encoder is unavailable.
            return []
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        reranked = []
        for result, score in zip(candidates, scores, strict=False):
            value = float(score)
            payload = dict(result.payload)
            payload["reranker"] = {
                "model": self.settings.model_name,
                "latency_ms": latency_ms,
                "backend": "http",
            }
            reranked.append(
                RetrievalResult(
                    entity_type=result.entity_type,
                    entity_id=result.entity_id,
                    source_store_evidence=result.source_store_evidence,
                    rerank_score=value,
                    confidence=value,
                    validation_status=result.validation_status,
                    payload=payload,
                )
            )
        return [
            item
            for item in sorted(reranked, key=lambda item: item.rerank_score, reverse=True)
            if isfinite(item.rerank_score)
            and (
                not enforce_min_score
                or item.rerank_score >= self.settings.min_score
            )
        ]

    def _score(self, body: dict[str, object]) -> list[float]:
        if not self.settings.endpoint_url:
            raise ValueError("Reranker endpoint URL is not configured")
        data = json.dumps(body).encode("utf-8")
        req = request.Request(
            self.settings.endpoint_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.settings.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except error.URLError as exc:
            raise ConnectionError(f"Reranker request failed: {exc}") from exc
        scores = payload.get("scores")
        if not isinstance(scores, list):
            raise ValueError("Reranker response must include a scores list")
        return [float(score) for score in scores]
