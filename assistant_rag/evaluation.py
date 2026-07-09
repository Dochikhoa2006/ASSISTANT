"""Retrieval and mutation-safety evaluation runner."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from .contracts import EvaluationReport
from .settings import OperationsSettings


def load_eval_cases(path: str | Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return list(data.get("cases", []))
    if isinstance(data, list):
        return data
    raise ValueError("Evaluation cases must be a JSON list or an object with a cases list")


class EvaluationRunner:
    def __init__(self, *, retriever: Any | None = None, repository: Any | None = None, settings: OperationsSettings | None = None):
        self.retriever = retriever
        self.repository = repository
        self.settings = settings or OperationsSettings()

    def run_cases(self, cases: list[dict[str, Any]], *, strict: bool = False) -> dict[str, Any]:
        results: list[dict[str, Any]] = []
        latencies: list[float] = []
        for case in cases:
            started = perf_counter()
            result = self._run_case(case)
            latencies.append((perf_counter() - started) * 1000)
            results.append(result)
        report = self._report(results, latencies)
        if strict and not report.passed:
            payload = asdict(report)
            payload["strict_failed"] = True
            return payload
        return asdict(report)

    def run_file(self, path: str | Path, *, strict: bool = False) -> dict[str, Any]:
        return self.run_cases(load_eval_cases(path), strict=strict)

    def _run_case(self, case: dict[str, Any]) -> dict[str, Any]:
        expected_id = case.get("expected_entity_id")
        expected_ids = [expected_id] if expected_id else list(case.get("expected_entity_ids", []))
        query = str(case.get("query", ""))
        entity_type = str(case.get("entity_type", "knowledge_chunk"))
        results = []
        if self.retriever is not None and query:
            method = self.retriever.retrieve_conversation if entity_type == "conversation_hop" else self.retriever.retrieve_knowledge
            try:
                results = method(
                    user_id=str(case.get("user_id", "eval-user")),
                    query=query,
                    limit=3,
                    min_confidence=float(case.get("min_confidence", 0.0)),
                )
            except Exception as exc:
                return {
                    "case_id": case.get("case_id"),
                    "error": type(exc).__name__,
                    "retrieval_empty": True,
                    "top_1": False,
                    "top_3": False,
                    "wrong_target": False,
                    "clarification": True,
                    "false_mutation": False,
                }
        ids = [getattr(result, "entity_id", "") for result in results]
        action = case.get("expected_action")
        false_mutation = bool(case.get("mutation_committed")) and case.get("should_mutate") is False
        return {
            "case_id": case.get("case_id"),
            "category": case.get("category"),
            "retrieval_empty": bool(query and not ids),
            "top_1": bool(expected_ids and ids[:1] and ids[0] in expected_ids),
            "top_3": bool(expected_ids and any(item in expected_ids for item in ids[:3])),
            "wrong_target": bool(expected_ids and ids and ids[0] not in expected_ids),
            "clarification": action == "clarify" or bool(case.get("clarification_required")),
            "false_mutation": false_mutation,
            "ids": ids,
        }

    def _report(self, results: list[dict[str, Any]], latencies: list[float]) -> EvaluationReport:
        count = len(results) or 1
        top_1 = sum(1 for item in results if item.get("top_1")) / count
        top_3 = sum(1 for item in results if item.get("top_3")) / count
        wrong_target = sum(1 for item in results if item.get("wrong_target")) / count
        clarification = sum(1 for item in results if item.get("clarification")) / count
        false_mutation = sum(1 for item in results if item.get("false_mutation")) / count
        empty = sum(1 for item in results if item.get("retrieval_empty")) / count
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        passed = (
            top_1 >= self.settings.eval_top_1_threshold
            and top_3 >= self.settings.eval_top_3_threshold
            and wrong_target <= self.settings.eval_wrong_target_rate_max
            and false_mutation <= self.settings.eval_false_mutation_rate_max
        )
        return EvaluationReport(
            case_count=len(results),
            top_1_accuracy=top_1,
            top_3_accuracy=top_3,
            wrong_target_rate=wrong_target,
            clarification_rate=clarification,
            false_mutation_rate=false_mutation,
            retrieval_empty_rate=empty,
            average_latency_ms=avg_latency,
            passed=passed,
            cases=results,
        )
