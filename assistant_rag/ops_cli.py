"""Operational CLI for index rebuild, drift checks, and evaluation."""

from __future__ import annotations

import argparse
import json
from typing import Any

from .database import SQLiteRepository
from .evaluation import EvaluationRunner
from .index_ops import IndexOperations
from .settings import ProductionSettings


class LocalMemoryIndex:
    def __init__(self) -> None:
        self.docs: dict[str, dict[str, Any]] = {}

    def search(self, *, user_id: str, query: str, limit: int):
        return []

    def upsert(self, *, user_id: str, entity_type: str, entity_id: str, text: str, metadata: dict[str, str | int | float | bool] | None = None) -> None:
        self.docs[entity_id] = {"user_id": user_id, "entity_type": entity_type, "entity_id": entity_id, "text": text, **(metadata or {})}

    def delete(self, *, entity_id: str) -> None:
        self.docs.pop(entity_id, None)

    def clear(self) -> None:
        self.docs.clear()

    def documents(self) -> list[dict[str, Any]]:
        return list(self.docs.values())

    def get_document(self, *, entity_id: str) -> dict[str, Any] | None:
        return self.docs.get(entity_id)

    def count_by_entity_type(self, *, user_id: str | None = None) -> dict[str, int]:
        counts: dict[str, int] = {}
        for doc in self.docs.values():
            if user_id and doc.get("user_id") != user_id:
                continue
            entity_type = str(doc.get("entity_type"))
            counts[entity_type] = counts.get(entity_type, 0) + 1
        return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="assistant-rag-ops")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("rebuild-all-indexes")
    user_rebuild = sub.add_parser("rebuild-user-indexes")
    user_rebuild.add_argument("--user-id", required=True)
    knowledge = sub.add_parser("rebuild-knowledge-index")
    knowledge.add_argument("--user-id")
    conversation = sub.add_parser("rebuild-conversation-index")
    conversation.add_argument("--user-id")
    drift = sub.add_parser("check-index-drift")
    drift.add_argument("--user-id")
    drift.add_argument("--repair", action="store_true")
    eval_parser = sub.add_parser("run-retrieval-eval")
    eval_parser.add_argument("--cases", required=True)
    eval_parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    settings = ProductionSettings.from_env()
    repo = SQLiteRepository.persistent(
        settings.database.path,
        enable_wal=settings.database.enable_wal,
        busy_timeout_ms=settings.database.busy_timeout_ms,
    )
    repo.initialize_schema()
    bm25 = LocalMemoryIndex()
    chroma = LocalMemoryIndex()
    ops = IndexOperations(repository=repo, bm25=bm25, chroma=chroma)

    if args.command == "rebuild-all-indexes":
        output = {"rebuilt": ops.rebuild_all_indexes()}
    elif args.command == "rebuild-user-indexes":
        output = {"rebuilt": ops.rebuild_user_indexes(user_id=args.user_id)}
    elif args.command == "rebuild-knowledge-index":
        output = {"rebuilt": ops.rebuild_knowledge_index(user_id=args.user_id)}
    elif args.command == "rebuild-conversation-index":
        output = {"rebuilt": ops.rebuild_conversation_index(user_id=args.user_id)}
    elif args.command == "check-index-drift":
        output = ops.check_index_drift(user_id=args.user_id, repair=args.repair)
    elif args.command == "run-retrieval-eval":
        output = EvaluationRunner(settings=settings.operations).run_file(args.cases, strict=args.strict)
        print(json.dumps(output, default=str, indent=2))
        return 1 if args.strict and not output.get("passed") else 0
    else:
        parser.error("unsupported command")
        return 2

    print(json.dumps(output, default=str, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
