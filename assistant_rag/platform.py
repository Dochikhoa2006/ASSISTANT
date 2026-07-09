"""Platform selection after response bundling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .contracts import BundledResponse, ChatRequest


class PlatformFormatter(Protocol):
    def format(self, response: BundledResponse, request: ChatRequest) -> dict[str, str]:
        ...


@dataclass
class PlatformSelector:
    formatters: dict[str, PlatformFormatter] = field(default_factory=dict)

    def register(self, channel: str, formatter: PlatformFormatter) -> None:
        self.formatters[channel] = formatter

    def select(self, response: BundledResponse, request: ChatRequest) -> dict[str, str]:
        channel = request.platform_context.get("channel")
        if not channel:
            return {"text": response.final_chat_text}
        formatter = self.formatters.get(str(channel))
        if formatter is None:
            return {"text": response.final_chat_text}
        return formatter.format(response, request)


class PlainTextFormatter:
    def format(self, response: BundledResponse, request: ChatRequest) -> dict[str, str]:
        return {"text": response.final_chat_text}

