"""Request scoped tracing, structured logs, and redaction helpers."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
import re
import time
from typing import Any
from uuid import uuid4

from .contracts import TraceStage, TraceStageSummary, TraceSummary


_request_id: ContextVar[str | None] = ContextVar("assistant_request_id", default=None)
_trace: ContextVar["TraceRecorder | None"] = ContextVar("assistant_trace", default=None)

SENSITIVE_KEY_PATTERNS = (
    "authorization",
    "access_token",
    "jwt",
    "token",
    "secret",
    "password",
    "api_key",
    "storage_path",
    "artifact_path",
    "file_contents",
    "raw_prompt",
    "prompt",
)


def new_request_id(existing: str | None = None) -> str:
    return existing or uuid4().hex


def set_request_id(request_id: str) -> None:
    _request_id.set(request_id)


def get_request_id() -> str | None:
    return _request_id.get()


def redact_value(key: str, value: Any, *, log_raw_content: bool = False) -> Any:
    lowered = key.casefold()
    if any(pattern in lowered for pattern in SENSITIVE_KEY_PATTERNS):
        return "[REDACTED]"
    if not log_raw_content and lowered in {
        "raw_query",
        "query",
        "user_query",
        "message",
        "text",
        "document",
        "content",
        "raw_response",
    }:
        return "[REDACTED]"
    if isinstance(value, str):
        value = re.sub(r"Bearer\s+[A-Za-z0-9._~+/=-]+", "Bearer [REDACTED]", value)
        value = re.sub(r"//([^:/\s]+):([^@\s]+)@", "//[REDACTED]:[REDACTED]@", value)
    return value


def redact_payload(payload: Any, *, log_raw_content: bool = False) -> Any:
    if isinstance(payload, dict):
        return {
            str(key): redact_payload(
                redact_value(str(key), value, log_raw_content=log_raw_content),
                log_raw_content=log_raw_content,
            )
            for key, value in payload.items()
        }
    if isinstance(payload, list):
        return [redact_payload(item, log_raw_content=log_raw_content) for item in payload]
    if isinstance(payload, tuple):
        return tuple(redact_payload(item, log_raw_content=log_raw_content) for item in payload)
    return payload


@dataclass
class TraceRecorder:
    request_id: str
    started_at: float = field(default_factory=time.perf_counter)
    stages: list[TraceStageSummary] = field(default_factory=list)
    _active_timers: list["StageTimer"] = field(default_factory=list)

    def add_stage(self, stage: str | TraceStage, latency_ms: float, metadata: dict[str, Any] | None = None) -> None:
        stage_name = stage.value if isinstance(stage, TraceStage) else str(stage)
        self.stages.append(
            TraceStageSummary(
                stage=stage_name,
                latency_ms=round(float(latency_ms), 3),
                metadata=redact_payload(metadata or {}),
            )
        )

    def summary(self) -> TraceSummary:
        return TraceSummary(
            request_id=self.request_id,
            total_latency_ms=round((time.perf_counter() - self.started_at) * 1000, 3),
            stages=list(self.stages),
        )


def start_trace(request_id: str) -> TraceRecorder:
    recorder = TraceRecorder(request_id=request_id)
    _trace.set(recorder)
    set_request_id(request_id)
    return recorder


def current_trace() -> TraceRecorder | None:
    return _trace.get()


class StageTimer:
    def __init__(self, stage: str | TraceStage, metadata: dict[str, Any] | None = None):
        self.stage = stage
        self.metadata = metadata or {}
        self.started_at = 0.0

    def __enter__(self) -> "StageTimer":
        self.started_at = time.perf_counter()
        recorder = current_trace()
        if recorder:
            recorder._active_timers.append(self)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        latency_ms = (time.perf_counter() - self.started_at) * 1000
        recorder = current_trace()
        if recorder:
            metadata = dict(self.metadata)
            if exc is not None:
                metadata["error_type"] = type(exc).__name__
            
            if recorder._active_timers and recorder._active_timers[-1] is self:
                recorder._active_timers.pop()
                
            if recorder._active_timers:
                parent = recorder._active_timers[-1]
                if "operations" not in parent.metadata:
                    parent.metadata["operations"] = []
                parent.metadata["operations"].append({
                    "operation": self.stage.value if isinstance(self.stage, TraceStage) else str(self.stage),
                    "latency_ms": round(latency_ms, 3),
                    **metadata
                })
            else:
                recorder.add_stage(self.stage, latency_ms, metadata)


class JsonLogFormatter(logging.Formatter):
    def __init__(self, *, log_raw_content: bool = False):
        super().__init__()
        self.log_raw_content = log_raw_content

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", None) or get_request_id(),
        }
        extra = getattr(record, "payload", None)
        if isinstance(extra, dict):
            payload.update(redact_payload(extra, log_raw_content=self.log_raw_content))
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(redact_payload(payload, log_raw_content=self.log_raw_content), default=str, separators=(",", ":"))


def trace_summary_asdict(summary: TraceSummary | None) -> dict[str, Any] | None:
    return asdict(summary) if summary else None
