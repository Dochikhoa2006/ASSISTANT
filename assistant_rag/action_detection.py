"""Neutral contracts for model-backed branch action extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from .contracts import ChatRequest, Intent


@dataclass(frozen=True)
class ActionDetectionResult:
    """Structured output produced by a knowledge or reminder extraction model."""

    intent: Intent
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)
    missing_fields: list[str] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)
    failure_kind: Literal["semantic_gap", "technical_failure"] | None = None

    @property
    def requires_clarification(self) -> bool:
        return self.confidence < 0.5 or bool(self.missing_fields)


class ActionDetector(Protocol):
    """Contract implemented by the branch-specific LLM extractors."""

    def detect(
        self,
        request: ChatRequest,
        rewritten_query: str,
        intent: Intent,
    ) -> ActionDetectionResult:
        ...
