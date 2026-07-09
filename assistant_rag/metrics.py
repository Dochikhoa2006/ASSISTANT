"""Small in-process JSON metrics registry."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from threading import Lock
from typing import Any

from .contracts import MetricSnapshot


class MetricsRegistry:
    def __init__(self) -> None:
        self._lock = Lock()
        self._counters: defaultdict[str, int] = defaultdict(int)
        self._gauges: dict[str, float] = {}
        self._latencies: defaultdict[str, list[float]] = defaultdict(list)

    def increment(self, name: str, amount: int = 1, **labels: str) -> None:
        key = self._key(name, labels)
        with self._lock:
            self._counters[key] += amount

    def set_gauge(self, name: str, value: float, **labels: str) -> None:
        key = self._key(name, labels)
        with self._lock:
            self._gauges[key] = float(value)

    def observe_latency(self, name: str, latency_ms: float, **labels: str) -> None:
        key = self._key(name, labels)
        with self._lock:
            self._latencies[key].append(float(latency_ms))

    def snapshot(self) -> MetricSnapshot:
        with self._lock:
            latencies: dict[str, dict[str, float]] = {}
            for key, values in self._latencies.items():
                if not values:
                    continue
                sorted_values = sorted(values)
                count = len(sorted_values)
                p95_idx = min(count - 1, int(count * 0.95))
                latencies[key] = {
                    "count": float(count),
                    "avg": sum(sorted_values) / count,
                    "min": sorted_values[0],
                    "max": sorted_values[-1],
                    "p95": sorted_values[p95_idx],
                }
            return MetricSnapshot(
                counters=dict(self._counters),
                gauges=dict(self._gauges),
                latency_ms=latencies,
                generated_at=datetime.now(timezone.utc).isoformat(),
            )

    def snapshot_dict(self) -> dict[str, Any]:
        return asdict(self.snapshot())

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._latencies.clear()

    def _key(self, name: str, labels: dict[str, str]) -> str:
        if not labels:
            return name
        label_text = ",".join(f"{key}={value}" for key, value in sorted(labels.items()))
        return f"{name}{{{label_text}}}"


GLOBAL_METRICS = MetricsRegistry()
