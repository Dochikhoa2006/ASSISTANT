"""SQL-first RAG assistant architecture package."""

from .contracts import ChatRequest, BundledResponse
from .pipeline import AssistantPipeline

__all__ = ["AssistantPipeline", "ChatRequest", "BundledResponse"]
