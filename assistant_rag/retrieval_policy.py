"""Single policy contract for knowledge and conversation-hop retrieval."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalPipelinePolicy:
    source_top_k: int = 20
    rrf_top_k: int = 15
    final_top_k: int = 5

    def validate(self) -> None:
        if self.source_top_k != 20:
            raise ValueError("Each retrieval backend must return exactly 20 candidates")
        if self.rrf_top_k != 15:
            raise ValueError("Reciprocal-rank fusion must pass exactly 15 candidates")
        if self.final_top_k != 5:
            raise ValueError("Cross-encoder retrieval must return exactly 5 candidates")


RETRIEVAL_PIPELINE_POLICY = RetrievalPipelinePolicy()
