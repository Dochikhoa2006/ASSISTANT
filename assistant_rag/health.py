"""Production health report assembly."""

from __future__ import annotations

from dataclasses import asdict
from time import perf_counter
from typing import Any

from .contracts import DependencyHealth, DependencyStatus, HealthPayload, HealthStatus
from .settings import OperationsSettings, ProductionSettings


def build_health_report(
    *,
    repository: Any,
    settings: ProductionSettings | None = None,
    checks: dict[str, Any] | None = None,
) -> dict[str, Any]:
    operations = settings.operations if settings else OperationsSettings()
    checks = checks or {}
    dependencies: list[DependencyHealth] = []
    outbox_pending = 0
    outbox_failed = 0

    sql_status = DependencyStatus.OK
    sql_detail = "ok"
    sql_latency = None
    try:
        started = perf_counter()
        repository.table_count("conversation_topics")
        sql_latency = (perf_counter() - started) * 1000
        outbox_pending = _outbox_count(repository, "pending")
        outbox_failed = _outbox_count(repository, "failed")
    except Exception as exc:
        sql_status = DependencyStatus.UNHEALTHY
        sql_detail = type(exc).__name__
    dependencies.append(DependencyHealth("sql", sql_status, sql_detail, sql_latency))

    optional_specs = (
        ("opensearch", operations.health_strict_opensearch),
        ("chroma", operations.health_strict_chroma),
        ("ollama", operations.health_strict_ollama),
        ("redis", operations.health_strict_redis),
        ("storage", operations.health_strict_storage),
    )
    for name, strict in optional_specs:
        dependencies.append(_check_optional(name, strict=strict, checker=checks.get(name)))

    overall = HealthStatus.OK
    if any(dep.status is DependencyStatus.UNHEALTHY for dep in dependencies):
        overall = HealthStatus.UNHEALTHY
    elif any(dep.status is DependencyStatus.DEGRADED for dep in dependencies):
        overall = HealthStatus.DEGRADED

    payload = HealthPayload(
        status=overall,
        dependencies=dependencies,
        outbox_pending_count=outbox_pending,
        outbox_failed_count=outbox_failed,
    )
    return _enum_values(asdict(payload))


def _outbox_count(repository: Any, status: str) -> int:
    connection = getattr(repository, "connection", None)
    if connection is not None:
        row = connection.execute(
            "SELECT COUNT(*) AS total FROM indexing_outbox WHERE status = ?",
            (status,),
        ).fetchone()
        return int(row["total"])
    engine = getattr(repository, "engine", None)
    if engine is not None:
        from sqlalchemy import text

        with engine.connect() as conn:
            return int(
                conn.execute(
                    text("SELECT COUNT(*) FROM indexing_outbox WHERE status = :status"),
                    {"status": status},
                ).scalar()
                or 0
            )
    return 0


def _check_optional(name: str, *, strict: bool, checker: Any | None) -> DependencyHealth:
    if checker is None:
        return DependencyHealth(name, DependencyStatus.NOT_CONFIGURED, "not configured")
    started = perf_counter()
    try:
        result = checker()
        latency_ms = (perf_counter() - started) * 1000
        if isinstance(result, dict) and result.get("ok") is False:
            return DependencyHealth(
                name,
                DependencyStatus.UNHEALTHY if strict else DependencyStatus.DEGRADED,
                str(result.get("detail") or "failed"),
                latency_ms,
            )
        return DependencyHealth(name, DependencyStatus.OK, "ok", latency_ms)
    except Exception as exc:
        return DependencyHealth(
            name,
            DependencyStatus.UNHEALTHY if strict else DependencyStatus.DEGRADED,
            type(exc).__name__,
            (perf_counter() - started) * 1000,
        )


def _enum_values(value: Any) -> Any:
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, dict):
        return {key: _enum_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_enum_values(item) for item in value]
    return value
