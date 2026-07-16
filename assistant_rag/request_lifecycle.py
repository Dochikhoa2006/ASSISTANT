"""Shared request lifecycle for every user-facing entrypoint.

The pipeline deliberately owns domain reasoning and persistence, while this
module owns request-scoped idempotency, confirmation hydration, and terminal
lifecycle bookkeeping.  Mutation classification remains a separate semantic
signal used by entrypoints for mutation-specific controls such as rate limits.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
import json
from typing import Any, Callable

from .contracts import BundledResponse, ChatRequest, Intent, ResponseType
from .database import AssistantRepository
from .pipeline import AssistantPipeline
from .request_policy import (
    has_destructive_mutation_policy_signal,
    has_explicit_mutation_policy_signal,
)


class RequestLifecycleConflict(ValueError):
    """A safe request conflict that callers may present without a traceback."""


@dataclass(frozen=True)
class ChatRequestExecution:
    """Result of one fresh or idempotently replayed pipeline request."""

    request: ChatRequest
    request_id: str
    response: BundledResponse | None
    payload: dict[str, Any]
    replayed: bool
    is_mutation: bool


def stable_payload_hash(request: ChatRequest) -> str:
    """Hash the caller-authored request, excluding its idempotency key."""

    payload = asdict(request)
    payload.pop("idempotency_key", None)
    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return sha256(encoded.encode("utf-8")).hexdigest()


def looks_like_mutation(request: ChatRequest) -> bool:
    metadata = request.metadata or {}
    if request.confirmation_token:
        return True
    # A reminder-notification reply writes a conversation hop and acknowledges
    # UI state even when the reply itself contains no action keyword.  Treat the
    # explicit, internally-built route marker as a non-destructive mutation so
    # every keyed API/Streamlit reply receives the same exactly-once lifecycle.
    if metadata.get("reminder_reply_context"):
        return True
    return any(
        has_explicit_mutation_policy_signal(request, intent)
        for intent in (Intent.KNOWLEDGE_FACTS, Intent.REMINDER)
    )


def looks_destructive(request: ChatRequest) -> bool:
    # A pending knowledge confirmation protects a destructive action. The
    # plain confirmation query does not repeat the stored action keyword, so
    # it must remain destructive for request limits.
    if request.confirmation_token:
        return True
    return any(
        has_destructive_mutation_policy_signal(request, intent)
        for intent in (Intent.KNOWLEDGE_FACTS, Intent.REMINDER)
    )


def bundled_response_payload(response: BundledResponse) -> dict[str, Any]:
    """Serialize the public response fields needed by local entrypoints/replay."""

    return {
        "final_chat_text": response.final_chat_text,
        "response_type": response.response_type.value,
        "conversation_topic_id": response.conversation_topic_id,
        "conversation_hop_id": response.conversation_hop_id,
        "actions_committed": list(response.actions_committed),
        "actions_pending_confirmation": list(response.actions_pending_confirmation),
        "warnings": list(response.warnings),
        "persistence_instructions": dict(response.persistence_instructions),
        "platform_payload": dict(response.platform_payload),
    }


def _confirmation_action_was_committed(
    request: ChatRequest,
    response: BundledResponse,
) -> bool:
    """Consume a confirmation only after its one validated action committed."""

    expected_actions = list(
        request.metadata.get("validated_knowledge_actions") or []
    )
    committed_actions = list(response.actions_committed)
    if len(expected_actions) != 1 or len(committed_actions) != 1:
        return False
    expected_action = str(expected_actions[0].get("action") or "").casefold()
    committed_action = str(
        committed_actions[0].get("action_type") or ""
    ).casefold()
    return bool(expected_action) and committed_action == expected_action


class ChatRequestLifecycleExecutor:
    """Execute a pipeline request with durable confirmation and idempotency.

    Authentication, rate limiting, tracing, and presentation remain entrypoint
    responsibilities. ``before_claim`` lets the API retain its mutation limits
    before any idempotency state is written. ``before_pipeline`` performs
    request preparation that must happen only for a fresh claim, after replay
    and conflict resolution, and immediately before pipeline execution.
    """

    def __init__(
        self,
        *,
        pipeline: AssistantPipeline,
        repository: AssistantRepository,
    ) -> None:
        self.pipeline = pipeline
        self.repository = repository

    def execute(
        self,
        request: ChatRequest,
        *,
        fallback_request_id: str,
        serialize_response: Callable[[BundledResponse, str], dict[str, Any]] = (
            lambda response, _request_id: bundled_response_payload(response)
        ),
        before_claim: Callable[[ChatRequest, bool, bool], None] | None = None,
        before_pipeline: Callable[[ChatRequest], None] | None = None,
    ) -> ChatRequestExecution:
        is_mutation = looks_like_mutation(request)
        is_destructive = looks_destructive(request)
        if before_claim is not None:
            before_claim(request, is_mutation, is_destructive)

        idempotency_request_id: str | None = None
        owns_idempotency_claim = False
        confirmation_to_mark: str | None = None
        prepared_request = request

        try:
            # An idempotency key protects the whole user-visible request, not
            # only domain mutations.  General requests can also create files,
            # send messages, and append conversation hops; replaying any of
            # those through the pipeline would duplicate externally visible
            # side effects.  ``is_mutation`` remains the independent semantic
            # signal supplied to ``before_claim`` for mutation rate limits.
            if request.idempotency_key:
                claim = self.repository.claim_idempotency_key(
                    user_id=request.user_id,
                    idempotency_key=request.idempotency_key,
                    payload_hash=stable_payload_hash(request),
                )
                idempotency_request_id = claim.request_id
                if claim.status == "replay" and claim.stored_response_json:
                    return ChatRequestExecution(
                        request=request,
                        request_id=claim.request_id,
                        response=None,
                        payload=json.loads(claim.stored_response_json),
                        replayed=True,
                        is_mutation=is_mutation,
                    )
                if claim.status in {"in_progress", "conflict"}:
                    raise RequestLifecycleConflict(claim.reason or claim.status)
                owns_idempotency_claim = claim.status in {"started", "failed_retry"}

            prepared_request, confirmation_to_mark = self._hydrate_confirmation(request)
            if is_mutation and not request.idempotency_key:
                metadata = dict(prepared_request.metadata or {})
                metadata["warnings"] = list(metadata.get("warnings") or []) + [
                    "Mutation request had no idempotency_key; processed in compatibility mode."
                ]
                prepared_request = replace(prepared_request, metadata=metadata)

            request_id = idempotency_request_id or fallback_request_id
            if before_pipeline is not None:
                before_pipeline(prepared_request)
            response = self.pipeline.handle(prepared_request, self.repository)
            payload = serialize_response(response, request_id)
            if idempotency_request_id and owns_idempotency_claim:
                if response.response_type is ResponseType.ERROR:
                    self.repository.fail_idempotency_request(
                        request_id=idempotency_request_id,
                        error_message=response.final_chat_text,
                    )
                else:
                    self.repository.complete_idempotency_request(
                        request_id=idempotency_request_id,
                        stored_response_json=json.dumps(payload, default=str),
                    )
            if (
                confirmation_to_mark
                and response.response_type is not ResponseType.ERROR
                and _confirmation_action_was_committed(prepared_request, response)
            ):
                self.repository.mark_confirmation_confirmed(
                    user_id=prepared_request.user_id,
                    confirmation_token=confirmation_to_mark,
                )
            return ChatRequestExecution(
                request=prepared_request,
                request_id=request_id,
                response=response,
                payload=payload,
                replayed=False,
                is_mutation=is_mutation,
            )
        except Exception as exc:
            if idempotency_request_id and owns_idempotency_claim:
                self.repository.fail_idempotency_request(
                    request_id=idempotency_request_id,
                    error_message=str(exc),
                )
            raise

    def _hydrate_confirmation(
        self, request: ChatRequest
    ) -> tuple[ChatRequest, str | None]:
        if not request.confirmation_token:
            return request, None

        try:
            confirmation = self.repository.load_pending_confirmation(
                user_id=request.user_id,
                confirmation_token=request.confirmation_token,
                now_value=datetime.now(timezone.utc).isoformat(),
            )
        except ValueError as exc:
            raise RequestLifecycleConflict(str(exc)) from exc

        proposed = confirmation.get("proposed_action") or {}
        domain = str(proposed.get("domain") or "")
        metadata = dict(request.metadata or {})
        if domain == "knowledge":
            actions = list(proposed.get("actions") or [])
            metadata["validated_knowledge_actions"] = actions
            metadata["knowledge_actions"] = actions
            metadata["intent"] = Intent.KNOWLEDGE_FACTS.value
        else:
            raise RequestLifecycleConflict("Confirmation has an unsupported action domain")

        metadata["action_authorization"] = dict(
            proposed.get("action_authorization") or {}
        )
        metadata["operation_response"] = proposed.get(
            "operation_response", metadata.get("operation_response")
        )
        metadata["topic_title"] = proposed.get(
            "topic_title", metadata.get("topic_title")
        )
        metadata["confirmation_approved"] = True
        return replace(request, metadata=metadata), request.confirmation_token
