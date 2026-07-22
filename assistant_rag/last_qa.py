"""Volatile Last-QA state.

The specification treats Last-QA as temporary context, so this module does not
add extra SQL schema beyond the authoritative tables.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
import sqlite3
import time

from .contracts import (
    ExpectedResponseType,
    LastQAState,
    OutboundMessageState,
    ResponseType,
)


@dataclass
class InMemoryLastQAStore:
    _states: dict[tuple[str, str], LastQAState] = field(default_factory=dict)

    @staticmethod
    def _key(user_id: str, conversation_id: str | None) -> tuple[str, str]:
        return (user_id, str(conversation_id or "").strip())

    def get(
        self, user_id: str, conversation_id: str | None = None
    ) -> LastQAState | None:
        return self._states.get(self._key(user_id, conversation_id))

    def save(
        self,
        user_id: str,
        state: LastQAState,
        conversation_id: str | None = None,
    ) -> None:
        self._states[self._key(user_id, conversation_id)] = state


class DiskCacheLastQAStore:
    def __init__(self, path: str, *, ttl_seconds: int) -> None:
        self.path = path
        self.ttl_seconds = ttl_seconds
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        # A Streamlit session keeps its production runtime in session_state,
        # while a rerun may be executed by a different script-runner thread.
        # Let the retained Last-QA cache connection follow that session across
        # reruns, matching the production repository connection.
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS last_qa_state (
                user_id TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL,
                expires_at REAL NOT NULL
            )
            """
        )
        # Keep the legacy user-only cache readable for callers which do not yet
        # provide a conversation cursor.  Scoped clients use this composite-key
        # table and can therefore keep independent drafts in multiple tabs.
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS last_qa_state_by_conversation (
                user_id TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                expires_at REAL NOT NULL,
                PRIMARY KEY (user_id, conversation_id)
            )
            """
        )
        self.connection.commit()

    def get(
        self, user_id: str, conversation_id: str | None = None
    ) -> LastQAState | None:
        scope = str(conversation_id or "").strip()
        if scope:
            row = self.connection.execute(
                """
                SELECT payload_json, expires_at
                FROM last_qa_state_by_conversation
                WHERE user_id = ? AND conversation_id = ?
                """,
                (user_id, scope),
            ).fetchone()
        else:
            row = self.connection.execute(
                "SELECT payload_json, expires_at FROM last_qa_state WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        if not row:
            return None
        if float(row[1]) < time.time():
            if scope:
                self.connection.execute(
                    """
                    DELETE FROM last_qa_state_by_conversation
                    WHERE user_id = ? AND conversation_id = ?
                    """,
                    (user_id, scope),
                )
            else:
                self.connection.execute(
                    "DELETE FROM last_qa_state WHERE user_id = ?", (user_id,)
                )
            self.connection.commit()
            return None
        payload = json.loads(str(row[0]))
        from .contracts import GeneratedQuestion, QuestionSource
        
        clarif = payload.get("clarification_question")
        clarif_q = GeneratedQuestion(
            text=clarif["text"], source=QuestionSource(clarif["source"]), purpose=clarif["purpose"], 
            confidence=clarif["confidence"], should_ask=clarif.get("should_ask", True),
            expected_response_type=ExpectedResponseType(clarif.get("expected_response_type", ExpectedResponseType.UNKNOWN.value))
        ) if clarif else None
        
        remind = payload.get("reminder_supporting_question")
        remind_q = GeneratedQuestion(
            text=remind["text"], source=QuestionSource(remind["source"]), purpose=remind["purpose"], 
            confidence=remind["confidence"], should_ask=remind.get("should_ask", True),
            expected_response_type=ExpectedResponseType(remind.get("expected_response_type", ExpectedResponseType.UNKNOWN.value))
        ) if remind else None
        
        supp = payload.get("supporting_questions", [])
        supp_qs = [
            GeneratedQuestion(
                text=q["text"], source=QuestionSource(q["source"]), purpose=q["purpose"], 
                confidence=q["confidence"], should_ask=q.get("should_ask", True),
                expected_response_type=ExpectedResponseType(q.get("expected_response_type", ExpectedResponseType.UNKNOWN.value))
            ) for q in supp
        ]

        outbound = payload.get("outbound_state")
        outbound_state = None
        if isinstance(outbound, dict):
            recipients = outbound.get("recipients")
            artifact_ids = outbound.get("artifact_ids")
            attachment_filenames = outbound.get("attachment_filenames")
            outbound_state = OutboundMessageState(
                channel=str(outbound.get("channel") or ""),
                status=str(outbound.get("status") or ""),
                recipients=tuple(
                    str(value)
                    for value in (recipients if isinstance(recipients, list) else [])
                    if str(value).strip()
                ),
                subject=str(outbound.get("subject") or ""),
                body=str(outbound.get("body") or ""),
                artifact_ids=tuple(
                    str(value)
                    for value in (artifact_ids if isinstance(artifact_ids, list) else [])
                    if str(value).strip()
                ),
                attachment_filenames=tuple(
                    str(value)
                    for value in (
                        attachment_filenames
                        if isinstance(attachment_filenames, list)
                        else []
                    )
                    if str(value).strip()
                ),
                source_topic_id=(
                    str(outbound["source_topic_id"])
                    if outbound.get("source_topic_id")
                    else None
                ),
                source_hop_id=(
                    str(outbound["source_hop_id"])
                    if outbound.get("source_hop_id")
                    else None
                ),
                excluded_recipients=tuple(
                    str(value)
                    for value in (
                        outbound.get("excluded_recipients")
                        if isinstance(outbound.get("excluded_recipients"), list)
                        else []
                    )
                    if str(value).strip()
                ),
                delivered_recipients=tuple(
                    str(value)
                    for value in (
                        outbound.get("delivered_recipients")
                        if isinstance(outbound.get("delivered_recipients"), list)
                        else []
                    )
                    if str(value).strip()
                ),
                refused_recipients=tuple(
                    str(value)
                    for value in (
                        outbound.get("refused_recipients")
                        if isinstance(outbound.get("refused_recipients"), list)
                        else []
                    )
                    if str(value).strip()
                ),
            )

        return LastQAState(
            last_user_query=payload["last_user_query"],
            last_response=payload["last_response"],
            response_type=ResponseType(payload["response_type"]),
            supporting_questions=supp_qs,
            clarification_question=clarif_q,
            reminder_supporting_question=remind_q,
            linked_topic_id=payload.get("linked_topic_id"),
            linked_hop_id=payload.get("linked_hop_id"),
            expected_response_type=(
                ExpectedResponseType(payload["expected_response_type"])
                if payload.get("expected_response_type")
                else None
            ),
            reminder_state=(
                payload.get("reminder_state")
                if isinstance(payload.get("reminder_state"), dict)
                else None
            ),
            reminder_state_hash=(
                str(payload["reminder_state_hash"])
                if payload.get("reminder_state_hash")
                else None
            ),
            outbound_state=outbound_state,
        )

    def save(
        self,
        user_id: str,
        state: LastQAState,
        conversation_id: str | None = None,
    ) -> None:
        from dataclasses import asdict
        
        payload = {
            "last_user_query": state.last_user_query,
            "last_response": state.last_response,
            "response_type": state.response_type.value,
            "supporting_questions": [asdict(q) for q in state.supporting_questions],
            "clarification_question": asdict(state.clarification_question) if state.clarification_question else None,
            "reminder_supporting_question": asdict(state.reminder_supporting_question) if state.reminder_supporting_question else None,
            "linked_topic_id": state.linked_topic_id,
            "linked_hop_id": state.linked_hop_id,
            "expected_response_type": state.expected_response_type.value if state.expected_response_type else None,
            "reminder_state": state.reminder_state,
            "reminder_state_hash": state.reminder_state_hash,
            "outbound_state": (
                asdict(state.outbound_state) if state.outbound_state else None
            ),
        }
        scope = str(conversation_id or "").strip()
        serialized = json.dumps(payload)
        expires_at = time.time() + self.ttl_seconds
        if scope:
            self.connection.execute(
                """
                INSERT INTO last_qa_state_by_conversation (
                    user_id, conversation_id, payload_json, expires_at
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id, conversation_id) DO UPDATE SET
                    payload_json = excluded.payload_json,
                    expires_at = excluded.expires_at
                """,
                (user_id, scope, serialized, expires_at),
            )
        else:
            self.connection.execute(
                """
                INSERT INTO last_qa_state (user_id, payload_json, expires_at)
                VALUES (?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    payload_json = excluded.payload_json,
                    expires_at = excluded.expires_at
                """,
                (user_id, serialized, expires_at),
            )
        self.connection.commit()
