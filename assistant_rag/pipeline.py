"""Ordered assistant pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
from typing import Any, Literal
from .bundler import ChatOutput, ResponseBundler
from .classification import IntentClassifier, LastQAResolver, QueryRewriter
from .config import AssistantConfig
from .contracts import (
    BundledResponse,
    ChatRequest,
    PipelineContext,
    ApprovedConversationContext,
    ExpectedResponseType,
    GeneratedQuestion,
    LastQAInteractionType,
    LastQAPath,
    LastQAResolution,
    LastQAState,
    OutboundMessageState,
    QuestionSource,
    RetrievalResult,
    ResponseType,
)
from .last_qa import InMemoryLastQAStore
from .platform import PlatformSelector
from .prompts import DEFAULT_PROMPT_REGISTRY, PromptRegistry
from .retrieval import HybridRetriever
from .branches import BranchRouter
from .database import AssistantRepository
from .context_filter import TwoLayerContextFilter
from .metrics import GLOBAL_METRICS
from .observability import StageTimer, current_trace
from .chat_history import (
    canonical_chat_history_scope,
    last_qa_chat_history,
    select_chat_history,
)
from .indexing import BackgroundIndexer
from .reminder_reply import (
    mark_reminder_state_replied,
    reminder_state_hash,
    verified_reminder_state,
)


def _branch_outbox_job_ids(branch_result: Any) -> list[str]:
    payload = dict(getattr(branch_result, "indexing_job_result", {}) or {})
    values = payload.get("outbox_job_ids") or []
    if isinstance(values, str):
        values = [values]
    fallback = payload.get("conversation_hop_job_id")
    return list(
        dict.fromkeys(
            str(job_id)
            for job_id in [*values, fallback]
            if job_id
        )
    )


def _artifact_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        values = [payload]
    elif isinstance(payload, (list, tuple)):
        values = list(payload)
    else:
        values = []
    return [dict(value) for value in values if isinstance(value, dict)]


def _artifact_ids(payload: Any) -> list[str]:
    return list(
        dict.fromkeys(
            str(item.get("artifact_id") or "")
            for item in _artifact_records(payload)
            if item.get("artifact_id")
        )
    )


def _has_active_branch_question(response: BundledResponse) -> bool:
    state = response.last_qa_state
    return bool(
        state.clarification_question
        or state.reminder_supporting_question
        or any(
            question.should_ask and question.text.strip()
            for question in state.supporting_questions
        )
    )


def _platform_deferred_for_branch_question(
    response: BundledResponse,
) -> dict[str, Any]:
    payload = dict(response.platform_payload)
    payload.update(
        {
            "text": response.final_chat_text,
            "platform_selection": {
                "channel": "none",
                "confidence": 1.0,
                "source": "bypassed_active_branch_question",
            },
            "delivery": {
                "channel": "none",
                "status": "deferred_by_active_question",
            },
        }
    )
    return payload


def _notice_only_platform_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Prevent platform integrations from becoming a second question owner."""

    normalized = dict(payload)
    delivery = normalized.get("delivery")
    if not isinstance(delivery, dict):
        return normalized
    delivery = dict(delivery)
    legacy_question = delivery.pop("question", None)
    if legacy_question and not delivery.get("notice"):
        delivery["notice"] = (
            "Platform delivery could not continue with the currently available "
            "chatbot delivery configuration."
        )
    normalized["delivery"] = delivery
    return normalized


def _platform_response_for_supporting_answer(
    response: BundledResponse,
    resolution: LastQAResolution,
) -> BundledResponse:
    """Carry an approved missing-context answer into platform action parsing.

    The Last-QA resolver must first bind the current turn to exactly one active
    question. Only then may the platform stage combine the two rewritten turns;
    neither raw ingress text nor unrelated retrieved history is admitted.
    """

    state = resolution.state
    if not (
        resolution.interaction_type
        is LastQAInteractionType.SUPPORTING_QUESTION_ANSWER
        and resolution.is_authoritative_state
        and state is not None
    ):
        return response
    prior_query = state.last_user_query.strip()
    current_answer = response.last_qa_state.last_user_query.strip()
    if not prior_query or not current_answer:
        return response
    platform_action_query = (
        f"{prior_query}\nResolved required context: {current_answer}"
    )
    platform_payload = dict(response.platform_payload)
    # A semantic decision produced for the short answer alone cannot decide
    # the now-resolved compound action. Force one fresh structured analysis of
    # the resolver-approved prior request plus its bound answer.
    platform_payload.pop("semantic_action_decision", None)
    return replace(
        response,
        last_qa_state=replace(
            response.last_qa_state,
            last_user_query=platform_action_query,
        ),
        platform_payload={
            **platform_payload,
            # This field is created only after the authoritative Last-QA
            # resolver binds the answer to one active supporting question.
            # The platform parser may trust it without treating arbitrary
            # addresses in message prose as delivery recipients.
            "approved_resolved_recipients": [current_answer],
        },
    )


def _rehydrate_outbound_artifacts(
    *,
    repository: AssistantRepository,
    user_id: str,
    outbound_state: OutboundMessageState | None,
    current_artifacts: list[dict[str, Any]],
    source_hop_id: str | None = None,
) -> list[dict[str, Any]]:
    by_id = {
        str(item.get("artifact_id") or ""): item
        for item in current_artifacts
        if item.get("artifact_id")
    }
    getter = getattr(repository, "get_generated_artifact", None)
    if outbound_state is not None and callable(getter):
        for artifact_id in outbound_state.artifact_ids:
            if artifact_id in by_id:
                continue
            try:
                row = getter(user_id=user_id, artifact_id=artifact_id)
            except Exception:
                continue
            if isinstance(row, dict):
                by_id[artifact_id] = dict(row)
    hop_artifact_loader = getattr(
        repository, "list_generated_artifacts_for_hop", None
    )
    if source_hop_id and callable(hop_artifact_loader):
        try:
            hop_artifacts = hop_artifact_loader(
                user_id=user_id,
                hop_id=source_hop_id,
                include_deleted=False,
            )
        except TypeError:
            hop_artifacts = hop_artifact_loader(
                user_id=user_id,
                hop_id=source_hop_id,
            )
        except Exception:
            hop_artifacts = []
        for item in _artifact_records(hop_artifacts):
            artifact_id = str(item.get("artifact_id") or "")
            if artifact_id:
                by_id.setdefault(artifact_id, item)
    return list(by_id.values())


def _last_qa_get(store: Any, request: ChatRequest) -> LastQAState | None:
    """Read scoped state while retaining compatibility with injected stores."""

    try:
        return store.get(
            request.user_id,
            conversation_id=request.conversation_id,
        )
    except TypeError:
        return store.get(request.user_id)


def _last_qa_save(store: Any, request: ChatRequest, state: LastQAState) -> None:
    """Persist scoped state while retaining compatibility with injected stores."""

    try:
        store.save(
            request.user_id,
            state,
            conversation_id=request.conversation_id,
        )
    except TypeError:
        store.save(request.user_id, state)


def _json_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            decoded = json.loads(value)
        except (TypeError, ValueError):
            return {}
        return dict(decoded) if isinstance(decoded, dict) else {}
    return {}


def _json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return list(value)
    if isinstance(value, str) and value.strip():
        try:
            decoded = json.loads(value)
        except (TypeError, ValueError):
            return []
        return list(decoded) if isinstance(decoded, list) else []
    return []


def _outbound_state_from_mapping(
    value: Any,
    *,
    source_topic_id: str | None,
    source_hop_id: str | None,
) -> OutboundMessageState | None:
    payload = _json_mapping(value)
    recipients_value = payload.get("recipients")
    recipients = tuple(
        dict.fromkeys(
            str(item).strip()
            for item in (
                recipients_value
                if isinstance(recipients_value, (list, tuple))
                else [payload.get("recipient")]
            )
            if str(item or "").strip()
        )
    )
    subject = str(payload.get("subject") or "").strip()
    body = str(payload.get("body") or "").strip()
    channel = str(payload.get("channel") or "").strip()
    if not channel or not recipients or not subject or not body:
        return None
    raw_attachments = payload.get("attachments")
    raw_attachments = (
        list(raw_attachments)
        if isinstance(raw_attachments, (list, tuple))
        else []
    )
    attachments = _artifact_records(raw_attachments)
    artifact_values = payload.get("artifact_ids")
    artifact_ids = tuple(
        dict.fromkeys(
            [
                str(item).strip()
                for item in (
                    artifact_values
                    if isinstance(artifact_values, (list, tuple))
                    else []
                )
                if str(item).strip()
            ]
            + [
                str(item.get("artifact_id") or "").strip()
                for item in attachments
                if str(item.get("artifact_id") or "").strip()
            ]
            + [
                str(item).strip()
                for item in raw_attachments
                if not isinstance(item, dict) and str(item).strip()
            ]
        )
    )
    filenames_value = payload.get("attachment_filenames")
    filenames_by_id = {
        str(item.get("artifact_id") or ""): str(item.get("filename") or "")
        for item in attachments
        if item.get("artifact_id")
    }
    explicit_filenames = [
        str(item)
        for item in (
            filenames_value if isinstance(filenames_value, (list, tuple)) else []
        )
    ]
    return OutboundMessageState(
        channel=channel,
        status=str(payload.get("status") or "draft_ready"),
        recipients=recipients,
        subject=subject,
        body=body,
        artifact_ids=artifact_ids,
        attachment_filenames=tuple(
            explicit_filenames[index]
            if index < len(explicit_filenames)
            else filenames_by_id.get(artifact_id, "")
            for index, artifact_id in enumerate(artifact_ids)
        ),
        source_topic_id=(
            str(payload.get("source_topic_id") or "").strip()
            or source_topic_id
        ),
        source_hop_id=(
            str(payload.get("source_hop_id") or "").strip()
            or source_hop_id
        ),
        excluded_recipients=tuple(
            str(item).strip()
            for item in (
                payload.get("excluded_recipients")
                if isinstance(payload.get("excluded_recipients"), (list, tuple))
                else []
            )
            if str(item).strip()
        ),
        delivered_recipients=tuple(
            str(item).strip()
            for item in (
                payload.get("delivered_recipients")
                if isinstance(payload.get("delivered_recipients"), (list, tuple))
                else []
            )
            if str(item).strip()
        ),
        refused_recipients=tuple(
            str(item).strip()
            for item in (
                payload.get("refused_recipients")
                if isinstance(payload.get("refused_recipients"), (list, tuple))
                else []
            )
            if str(item).strip()
        ),
    )


def _generated_questions_from_hop(value: Any) -> list[GeneratedQuestion]:
    questions: list[GeneratedQuestion] = []
    for item in _json_list(value):
        payload = item if isinstance(item, dict) else {"question_text": item}
        text = str(
            payload.get("text") or payload.get("question_text") or ""
        ).strip()
        if not text:
            continue
        try:
            source = QuestionSource(
                str(
                    payload.get("source")
                    or payload.get("question_source")
                    or QuestionSource.HUMAN_SUPPORTING_QUESTION.value
                )
            )
        except ValueError:
            source = QuestionSource.HUMAN_SUPPORTING_QUESTION
        try:
            expected = ExpectedResponseType(
                str(
                    payload.get("expected_response_type")
                    or ExpectedResponseType.UNKNOWN.value
                )
            )
        except ValueError:
            expected = ExpectedResponseType.UNKNOWN
        questions.append(
            GeneratedQuestion(
                text=text,
                source=source,
                purpose=str(payload.get("purpose") or "continue_selected_conversation"),
                confidence=float(payload.get("confidence", 1.0)),
                should_ask=bool(payload.get("should_ask", True)),
                expected_response_type=expected,
            )
        )
    return questions


def _outbound_state_from_owned_hop_lineage(
    *, repository: AssistantRepository, user_id: str, hop: dict[str, Any]
) -> OutboundMessageState | None:
    """Restore the nearest durable envelope on the selected branch lineage."""

    loader = getattr(repository, "get_conversation_hop", None)
    delivery_loader = getattr(
        repository, "get_latest_platform_delivery_for_hop", None
    )
    current = dict(hop)
    selected_topic_id = str(current.get("topic_id") or "").strip()
    seen: set[str] = set()
    for _ in range(100):
        hop_id = str(current.get("hop_id") or "").strip()
        topic_id = str(current.get("topic_id") or "").strip()
        if not hop_id or hop_id in seen or topic_id != selected_topic_id:
            break
        seen.add(hop_id)
        entities = _json_mapping(current.get("entities_json"))
        outbound_state = _outbound_state_from_mapping(
            entities.get("outbound_state"),
            source_topic_id=topic_id or None,
            source_hop_id=hop_id or None,
        )
        if outbound_state is None and callable(delivery_loader):
            try:
                delivery = delivery_loader(user_id=user_id, hop_id=hop_id)
            except Exception:
                delivery = None
            if isinstance(delivery, dict):
                message = delivery.get("message")
                if not isinstance(message, dict):
                    message = _json_mapping(delivery.get("message_json"))
                message = dict(message or {})
                # Historical audit rows did not persist recipient polarity.
                # Reinterpreting their source prose would reintroduce language
                # hardcoding, while trusting an over-broad envelope could send
                # to an excluded address. Only the structured envelope contract
                # is eligible for restoration; legacy rows fail closed.
                if "excluded_recipients" not in message:
                    message = {}
                message.setdefault("channel", delivery.get("channel"))
                message.setdefault("status", delivery.get("status"))
                if not message.get("recipients") and delivery.get("recipient"):
                    message["recipients"] = [
                        item.strip()
                        for item in str(delivery.get("recipient") or "").split(",")
                        if item.strip()
                    ]
                outbound_state = _outbound_state_from_mapping(
                    message,
                    source_topic_id=topic_id or None,
                    source_hop_id=hop_id or None,
                )
        if outbound_state is not None:
            return outbound_state
        previous_hop_id = str(current.get("previous_hop_id") or "").strip()
        if not previous_hop_id or not callable(loader):
            break
        try:
            previous = loader(user_id=user_id, hop_id=previous_hop_id)
        except Exception:
            break
        if not isinstance(previous, dict):
            break
        current = dict(previous)
    return None


def _last_qa_state_from_owned_hop(
    *, repository: AssistantRepository, user_id: str, hop: dict[str, Any]
) -> LastQAState:
    hop_id = str(hop.get("hop_id") or "").strip()
    topic_id = str(hop.get("topic_id") or "").strip()
    entities = _json_mapping(hop.get("entities_json"))
    outbound_state = _outbound_state_from_owned_hop_lineage(
        repository=repository,
        user_id=user_id,
        hop=hop,
    )
    try:
        response_type = ResponseType(str(hop.get("response_type") or "normal"))
    except ValueError:
        response_type = ResponseType.NORMAL
    questions = _generated_questions_from_hop(
        hop.get("supporting_questions_json")
    )
    clarification_question = next(
        (
            question
            for question in questions
            if question.source is QuestionSource.CLARIFICATION_QUESTION
        ),
        None,
    )
    reminder_question = next(
        (
            question
            for question in questions
            if question.source is QuestionSource.REMINDER_SUPPORTING_QUESTION
        ),
        None,
    )
    human_questions = [
        question
        for question in questions
        if question.source is QuestionSource.HUMAN_SUPPORTING_QUESTION
    ]
    expected_response_type = next(
        (
            question.expected_response_type
            for question in questions
            if question.expected_response_type is not ExpectedResponseType.UNKNOWN
        ),
        None,
    )
    return LastQAState(
        last_user_query=str(
            hop.get("rewritten_user_query") or hop.get("summarized_user_query") or ""
        ),
        last_response=str(
            entities.get("final_chat_text")
            or hop.get("raw_response")
            or hop.get("summarized_response")
            or ""
        ),
        response_type=response_type,
        supporting_questions=human_questions,
        clarification_question=clarification_question,
        reminder_supporting_question=reminder_question,
        linked_topic_id=topic_id or None,
        linked_hop_id=hop_id or None,
        expected_response_type=expected_response_type,
        outbound_state=outbound_state,
    )


def _merge_owned_hop_with_scoped_cache(
    owned: LastQAState,
    cached: LastQAState | None,
) -> LastQAState:
    """Restore volatile typed state only from the exact SQL-owned hop cache."""

    if not (
        cached is not None
        and cached.linked_topic_id == owned.linked_topic_id
        and cached.linked_hop_id == owned.linked_hop_id
    ):
        return owned
    return replace(
        owned,
        supporting_questions=(
            owned.supporting_questions or cached.supporting_questions
        ),
        clarification_question=(
            owned.clarification_question or cached.clarification_question
        ),
        reminder_supporting_question=(
            owned.reminder_supporting_question
            or cached.reminder_supporting_question
        ),
        expected_response_type=(
            owned.expected_response_type or cached.expected_response_type
        ),
        reminder_state=cached.reminder_state,
        reminder_state_hash=cached.reminder_state_hash,
        outbound_state=owned.outbound_state or cached.outbound_state,
    )


def _conversation_id_from_result(result: RetrievalResult) -> str:
    entities = _json_mapping(result.payload.get("entities_json"))
    return str(entities.get("conversation_id") or "").strip()


def _outbound_state_payload(state: OutboundMessageState | None) -> dict[str, Any] | None:
    if state is None:
        return None
    return {
        "channel": state.channel,
        "status": state.status,
        "recipients": list(state.recipients),
        "subject": state.subject,
        "body": state.body,
        "artifact_ids": list(state.artifact_ids),
        "attachment_filenames": list(state.attachment_filenames),
        "source_topic_id": state.source_topic_id,
        "source_hop_id": state.source_hop_id,
        "excluded_recipients": list(state.excluded_recipients),
        "delivered_recipients": list(state.delivered_recipients),
        "refused_recipients": list(state.refused_recipients),
    }


def _bundled_has_irreversible_effects(response: BundledResponse) -> bool:
    payload = response.platform_payload or {}
    delivery = payload.get("delivery")
    status = str(
        delivery.get("status") if isinstance(delivery, dict) else ""
    ).strip()
    return bool(
        response.actions_committed
        or response.conversation_hop_id
        or payload.get("artifacts")
        or status
        in {
            "sent",
            "partial_failure",
            "draft_saved",
            "delivery_unknown",
        }
    )


def _outbound_state_from_platform(
    *,
    platform_payload: dict[str, Any],
    previous_state: OutboundMessageState | None,
    source_topic_id: str | None,
    source_hop_id: str | None,
) -> OutboundMessageState | None:
    delivery = platform_payload.get("delivery")
    delivery = delivery if isinstance(delivery, dict) else {}
    status = str(delivery.get("status") or "")
    if status == "not_requested" or delivery.get("channel") in (None, "none"):
        return previous_state

    draft = platform_payload.get("draft")
    if not isinstance(draft, dict):
        if status == "sent" and previous_state is not None:
            return replace(
                previous_state,
                status="sent",
                source_topic_id=source_topic_id or previous_state.source_topic_id,
                source_hop_id=source_hop_id or previous_state.source_hop_id,
            )
        return previous_state
    channel = str(delivery.get("channel") or draft.get("channel") or "").strip()
    recipients_value = draft.get("recipients")
    recipients = [
        str(value).strip()
        for value in (
            recipients_value if isinstance(recipients_value, (list, tuple)) else []
        )
        if str(value).strip()
    ]
    if status == "partial_failure":
        refused = delivery.get("refused_recipients")
        refused_recipients = [
            str(value).strip()
            for value in (refused if isinstance(refused, (list, tuple)) else [])
            if str(value).strip()
        ]
        if refused_recipients:
            recipients = refused_recipients
    delivered = delivery.get("delivered_recipients")
    delivered_recipients = tuple(
        str(value).strip()
        for value in (
            delivered if isinstance(delivered, (list, tuple)) else []
        )
        if str(value).strip()
    )
    refused = delivery.get("refused_recipients")
    refused_recipients_state = tuple(
        str(value).strip()
        for value in (
            refused if isinstance(refused, (list, tuple)) else []
        )
        if str(value).strip()
    )
    subject = str(draft.get("subject") or "").strip()
    body = str(draft.get("body") or "").strip()
    if not channel or not recipients or not subject or not body:
        return previous_state

    attachments = _artifact_records(draft.get("attachments"))
    artifact_ids = tuple(
        dict.fromkeys(
            str(item.get("artifact_id") or "")
            for item in attachments
            if item.get("artifact_id")
        )
    )
    filenames_by_id = {
        str(item.get("artifact_id") or ""): str(item.get("filename") or "")
        for item in attachments
        if item.get("artifact_id")
    }
    return OutboundMessageState(
        channel=channel,
        status=status,
        recipients=tuple(recipients),
        subject=subject,
        body=body,
        artifact_ids=artifact_ids,
        attachment_filenames=tuple(
            filenames_by_id.get(artifact_id, "") for artifact_id in artifact_ids
        ),
        source_topic_id=source_topic_id,
        source_hop_id=source_hop_id,
        excluded_recipients=tuple(
            str(value).strip()
            for value in (
                draft.get("excluded_recipients")
                if isinstance(draft.get("excluded_recipients"), (list, tuple))
                else []
            )
            if str(value).strip()
        ),
        delivered_recipients=delivered_recipients,
        refused_recipients=refused_recipients_state,
    )

@dataclass
class AssistantPipeline:
    config: AssistantConfig
    last_qa_store: InMemoryLastQAStore
    query_rewriter: QueryRewriter
    last_qa_resolver: LastQAResolver
    retriever: HybridRetriever
    context_filter: TwoLayerContextFilter
    classifier: IntentClassifier
    router: BranchRouter
    bundler: ResponseBundler
    platform_selector: PlatformSelector
    chat_output: ChatOutput
    prompt_registry: PromptRegistry = field(default_factory=lambda: DEFAULT_PROMPT_REGISTRY)

    def handle(self, request: ChatRequest, repository: AssistantRepository) -> BundledResponse:
        with StageTimer("rewrite"):
            rewritten = self.query_rewriter.rewrite(request.raw_query)

        explicit_parent_state: LastQAState | None = None
        explicit_parent_validated = False
        if request.parent_hop_id:
            owned_hop_loader = getattr(repository, "get_conversation_hop", None)
            if callable(owned_hop_loader):
                try:
                    owned_hop = owned_hop_loader(
                        user_id=request.user_id,
                        hop_id=request.parent_hop_id,
                    )
                except Exception:
                    owned_hop = None
                if isinstance(owned_hop, dict):
                    hop_entities = _json_mapping(owned_hop.get("entities_json"))
                    stored_conversation_id = str(
                        hop_entities.get("conversation_id") or ""
                    ).strip()
                    if not stored_conversation_id:
                        topic_entities = _json_mapping(
                            owned_hop.get("topic_entities_json")
                        )
                        stored_conversation_id = str(
                            topic_entities.get("conversation_id") or ""
                        ).strip()
                    if str(owned_hop.get("topic_status") or "active") != "active":
                        owned_hop = None
                    elif (
                        request.conversation_id
                        and stored_conversation_id
                        and stored_conversation_id != request.conversation_id
                    ):
                        owned_hop = None
                    elif request.conversation_id and not stored_conversation_id:
                        scope_claimer = getattr(
                            repository, "claim_conversation_scope", None
                        )
                        try:
                            claimed = bool(
                                callable(scope_claimer)
                                and scope_claimer(
                                    user_id=request.user_id,
                                    hop_id=request.parent_hop_id,
                                    conversation_id=request.conversation_id,
                                )
                            )
                        except Exception:
                            claimed = False
                        if claimed:
                            stored_conversation_id = request.conversation_id
                        else:
                            owned_hop = None
                if not isinstance(owned_hop, dict):
                    response = BundledResponse(
                        final_chat_text=(
                            "The selected conversation could not be resumed because "
                            "that conversation hop is unavailable for this user. No "
                            "operation was executed."
                        ),
                        response_type=ResponseType.ERROR,
                        last_qa_state=LastQAState(
                            last_user_query=rewritten,
                            last_response=(
                                "The selected conversation hop is unavailable for this user."
                            ),
                            response_type=ResponseType.ERROR,
                        ),
                        conversation_id=request.conversation_id,
                        warnings=["selected_conversation_hop_rejected"],
                    )
                    self.chat_output.emit(response)
                    trace = current_trace()
                    return replace(
                        response,
                        trace_summary=trace.summary() if trace else None,
                    )
                # Reminder replies and legacy clients may provide only the
                # durable hop cursor.  Once ownership is proven, inherit that
                # hop's stored conversation scope so cache, retrieval, writes,
                # and the response all stay on the same UI conversation.
                if not request.conversation_id and stored_conversation_id:
                    request = replace(
                        request,
                        conversation_id=stored_conversation_id,
                    )
                explicit_parent_state = _last_qa_state_from_owned_hop(
                    repository=repository,
                    user_id=request.user_id,
                    hop=owned_hop,
                )
                explicit_parent_validated = True

        # Last-QA is a compulsory stage for every request. Intent classification
        # happens only after the resolver has accepted, merged, or rejected the
        # latest state.
        with StageTimer("last_qa_resolution") as last_qa_stage:
            cached_last_state = _last_qa_get(self.last_qa_store, request)
            if explicit_parent_state is not None:
                explicit_parent_state = _merge_owned_hop_with_scoped_cache(
                    explicit_parent_state,
                    cached_last_state,
                )
            last_state = explicit_parent_state or cached_last_state
            resolution = self.last_qa_resolver.resolve(request, rewritten, last_state)
            if (
                explicit_parent_validated
                and explicit_parent_state is not None
                and not resolution.is_authoritative_state
                and resolution.interaction_type is None
            ):
                # Selecting a concrete, ownership-validated past hop is a
                # stronger relationship signal than semantic similarity. It
                # binds the turn to that hop without merging unrelated text.
                resolution = LastQAResolution(
                    path=LastQAPath.LATEST_CONTEXT_INTERACTION,
                    rewritten_query=rewritten,
                    state=explicit_parent_state,
                    did_merge_query=False,
                    skip_broad_retrieval=True,
                    confidence=1.0,
                    interaction_type=LastQAInteractionType.NORMAL_FOLLOW_UP,
                    question_source=QuestionSource.NONE,
                    linked_topic_id=explicit_parent_state.linked_topic_id,
                    linked_hop_id=explicit_parent_state.linked_hop_id,
                    source_topic_id=explicit_parent_state.linked_topic_id,
                    source_hop_id=explicit_parent_state.linked_hop_id,
                    is_authoritative_state=True,
                    diagnostic_context={
                        "relationship_source": "owned_explicit_parent_hop"
                    },
                    merge_reason="explicit_conversation_hop_selected",
                    skip_reason="ownership_validated_explicit_parent",
                )
            requires_active_link = bool(
                resolution.skip_broad_retrieval
                and resolution.interaction_type
                in {
                    LastQAInteractionType.NORMAL_FOLLOW_UP,
                    LastQAInteractionType.SUPPORTING_QUESTION_ANSWER,
                }
            )
            if requires_active_link and not explicit_parent_validated:
                active_link_validator = getattr(
                    repository, "is_active_conversation_link", None
                )
                if callable(active_link_validator):
                    try:
                        active_link_valid = bool(
                            resolution.linked_topic_id
                            and resolution.linked_hop_id
                            and active_link_validator(
                                user_id=request.user_id,
                                topic_id=resolution.linked_topic_id,
                                hop_id=resolution.linked_hop_id,
                            )
                        )
                    except Exception as exc:
                        active_link_valid = False
                        last_qa_stage.metadata["link_validation_error"] = (
                            type(exc).__name__
                        )
                    last_qa_stage.metadata["active_link_valid"] = active_link_valid
                    if not active_link_valid:
                        resolution = replace(
                            resolution,
                            path=LastQAPath.BROAD_RETRIEVAL_REQUIRED,
                            interaction_type=None,
                            question_source=QuestionSource.NONE,
                            state=None,
                            did_merge_query=False,
                            skip_broad_retrieval=False,
                            linked_topic_id=None,
                            linked_hop_id=None,
                            matched_question=None,
                            is_authoritative_state=False,
                            diagnostic_context={
                                **resolution.diagnostic_context,
                                "cached_link_validation": "rejected",
                            },
                            merge_reason="cached_last_qa_link_not_active",
                            skip_reason="broad_retrieval_required",
                        )
                else:
                    # Lightweight injected repositories used by deterministic
                    # tests may omit this capability. The production repository
                    # always implements the user/topic/hop lifecycle check.
                    last_qa_stage.metadata["active_link_valid"] = "not_available"
            last_qa_stage.metadata["path"] = resolution.path.value
            last_qa_stage.metadata["confidence"] = resolution.confidence
            last_qa_stage.metadata["skip_broad_retrieval"] = (
                resolution.skip_broad_retrieval
            )
            last_qa_stage.metadata["interaction_type"] = (
                resolution.interaction_type.value
                if resolution.interaction_type is not None
                else None
            )
            model_relationship = resolution.diagnostic_context.get(
                "model_relationship"
            )
            if model_relationship:
                last_qa_stage.metadata["model_relationship"] = model_relationship
            if resolution.skip_reason:
                last_qa_stage.metadata["decision_reason"] = (
                    resolution.skip_reason
                )
        
        # Semantic authority boundary: after Last-QA resolution, every
        # downstream consumer must use this rewritten value. The request's
        # original text remains available only to quarantined audit/control
        # paths and must not influence retrieval, routing, prompts, or tools.
        retrieval_query = resolution.rewritten_query
        intent_classifier_query = resolution.rewritten_query
        
        with StageTimer("conversation_retrieval_gate", {
            "last_qa_requested_skip": resolution.skip_broad_retrieval,
            "after_last_qa_enabled": self.config.context_filter.conversation_retrieval_after_last_qa_enabled,
            "before_intent_enabled": self.config.context_filter.conversation_retrieval_before_intent_enabled,
        }) as retrieval_gate:
            should_run_broad_retrieval = (
                not resolution.skip_broad_retrieval
                and self.config.context_filter.conversation_retrieval_after_last_qa_enabled
                and self.config.context_filter.conversation_retrieval_before_intent_enabled
            )
            retrieval_gate.metadata["will_run"] = should_run_broad_retrieval
        
        approved_conversation_context: ApprovedConversationContext | None = None
        conversation_results = []
        
        if should_run_broad_retrieval:
            with StageTimer("retrieval", {"entity_type": "conversation_hop"}):
                raw_conversation_results = self.retriever.retrieve_conversation(
                    user_id=request.user_id,
                    query=retrieval_query,
                )
            with StageTimer("sql_validation", {"entity_type": "conversation_hop", "candidate_count": len(raw_conversation_results)}):
                conversation_results = repository.hydrate_conversation_retrieval_results(
                    user_id=request.user_id,
                    results=raw_conversation_results,
                )
            if not conversation_results:
                GLOBAL_METRICS.increment("retrieval_empty_total", entity_type="conversation_hop")
            if request.conversation_id:
                conversation_results = [
                    result
                    for result in conversation_results
                    if _conversation_id_from_result(result)
                    == request.conversation_id
                ]
            resolution = replace(resolution, state=None)
            if conversation_results:
                approved_conversation_context = self.context_filter.filter_conversation_only(
                    user_id=request.user_id,
                    conversation_results=conversation_results,
                    query=retrieval_query,
                    config=self.config.context_filter,
                )
            else:
                context_status = (
                    "all_rejected" if raw_conversation_results else "empty"
                )
                approved_conversation_context = ApprovedConversationContext(
                    approved_conversation_history=[],
                    human_supporting_questions=[],
                    reminder_supporting_questions=[],
                    clarification_question_context=None,
                    extracted_expected_response_types=[],
                    conversation_retrieval_ran=True,
                    conversation_context_status=context_status,
                    approved_conversation_count=0,
                    _rejected_conversation_ids=tuple(
                        result.entity_id for result in raw_conversation_results
                    ),
                )

        # Exact supporting-question answers intentionally skip broad retrieval.
        # Preserve the resolver's authoritative topic/hop identity as approved
        # selection candidates so the deterministic sub-branch detector can
        # apply the same candidate contract without introducing unrelated
        # conversation text into canonical chat history.
        if (
            not should_run_broad_retrieval
            and resolution.path == LastQAPath.LATEST_CONTEXT_INTERACTION
            and resolution.interaction_type
            == LastQAInteractionType.SUPPORTING_QUESTION_ANSWER
            and resolution.is_authoritative_state
            and resolution.state is not None
            and resolution.state.linked_topic_id
            and resolution.state.linked_hop_id
        ):
            state = resolution.state
            approved_conversation_context = ApprovedConversationContext(
                approved_conversation_history=[],
                human_supporting_questions=list(state.supporting_questions),
                reminder_supporting_questions=(
                    [state.reminder_supporting_question]
                    if state.reminder_supporting_question is not None
                    else []
                ),
                clarification_question_context=state.clarification_question,
                extracted_expected_response_types=(
                    [state.expected_response_type]
                    if state.expected_response_type is not None
                    else []
                ),
                conversation_retrieval_ran=False,
                conversation_context_status="not_run",
                approved_conversation_count=0,
                top_hop_rerank_score=None,
                _internal_selected_topic_candidates=[state.linked_topic_id],
                _internal_selected_hop_candidates=[state.linked_hop_id],
                _validation_summary=(
                    "Authoritative Last-QA supporting-question identity."
                ),
            )

        # A high-confidence normal follow-up is bound to the immediately
        # preceding SQL-linked hop by the Last-QA resolver. Preserve that one
        # hop as approved context so the existing sub-branch detector appends
        # to it instead of incorrectly creating a new conversation topic.
        if (
            not should_run_broad_retrieval
            and resolution.path == LastQAPath.LATEST_CONTEXT_INTERACTION
            and resolution.interaction_type
            == LastQAInteractionType.NORMAL_FOLLOW_UP
            and resolution.is_authoritative_state
            and resolution.state is not None
            and resolution.state.linked_topic_id
            and resolution.state.linked_hop_id
        ):
            state = resolution.state
            approved_conversation_context = ApprovedConversationContext(
                approved_conversation_history=last_qa_chat_history(state),
                human_supporting_questions=list(state.supporting_questions),
                reminder_supporting_questions=(
                    [state.reminder_supporting_question]
                    if state.reminder_supporting_question is not None
                    else []
                ),
                clarification_question_context=state.clarification_question,
                extracted_expected_response_types=(
                    [state.expected_response_type]
                    if state.expected_response_type is not None
                    else []
                ),
                conversation_retrieval_ran=False,
                conversation_context_status="approved",
                approved_conversation_count=1,
                top_hop_rerank_score=None,
                _internal_selected_topic_candidates=[state.linked_topic_id],
                _internal_selected_hop_candidates=[state.linked_hop_id],
                _validation_summary=(
                    "Authoritative high-confidence latest-exchange relationship."
                ),
            )
        
        chat_history = select_chat_history(
            conversation_retrieval=should_run_broad_retrieval,
            approved_conversation_context=approved_conversation_context,
            last_qa_state=resolution.state,
        )
        chat_history_source = (
            "conversation_retrieval" if should_run_broad_retrieval else "last_qa"
        )

        # Classification is the first consumer. The same immutable request-scoped
        # value remains active through branches, bundling, and platform selection.
        with canonical_chat_history_scope(chat_history):
            return self._handle_with_chat_history(
                request=request,
                repository=repository,
                initially_rewritten_query=rewritten,
                retrieval_query=retrieval_query,
                intent_classifier_query=intent_classifier_query,
                resolution=resolution,
                conversation_results=conversation_results,
                approved_conversation_context=approved_conversation_context,
                conversation_retrieval=should_run_broad_retrieval,
                chat_history=chat_history,
                chat_history_source=chat_history_source,
                previous_last_qa_state=last_state,
                explicit_parent_hop_id=(
                    request.parent_hop_id if explicit_parent_validated else None
                ),
            )

    def _handle_with_chat_history(
        self,
        *,
        request: ChatRequest,
        repository: AssistantRepository,
        initially_rewritten_query: str,
        retrieval_query: str,
        intent_classifier_query: str,
        resolution: LastQAResolution,
        conversation_results: list[RetrievalResult],
        approved_conversation_context: ApprovedConversationContext | None,
        conversation_retrieval: bool,
        chat_history: list[dict[str, Any]],
        chat_history_source: Literal["conversation_retrieval", "last_qa"],
        previous_last_qa_state: LastQAState | None,
        explicit_parent_hop_id: str | None = None,
    ) -> BundledResponse:
        with StageTimer("classification"):
            intent = self.classifier.classify(
                request,
                intent_classifier_query,
                last_qa_resolution=resolution,
                approved_conversation_context=approved_conversation_context,
            )

        last_qa_trace = {
            "input_rewritten_query": initially_rewritten_query,
            "output_rewritten_query": resolution.rewritten_query,
            "did_merge_query": resolution.did_merge_query,
            "skip_broad_retrieval": resolution.skip_broad_retrieval,
            "resolution_confidence": resolution.confidence,
            "path": resolution.path.value,
            "interaction_type": resolution.interaction_type.value if resolution.interaction_type else None,
            "question_source": resolution.question_source.value if resolution.question_source else None,
            "matched_question": resolution.matched_question,
            "is_authoritative_state": resolution.is_authoritative_state,
            "merge_reason": resolution.merge_reason,
            "skip_reason": resolution.skip_reason,
            "broad_retrieval_ran": conversation_retrieval,
            "conversation_retrieval": conversation_retrieval,
            "chat_history_source": chat_history_source,
            "chat_history_count": len(chat_history),
            "query_sent_to_retrieval": retrieval_query if conversation_retrieval else None,
            "query_sent_to_intent_classifier": intent_classifier_query,
            "approved_conversation_history_count": len(approved_conversation_context.approved_conversation_history) if approved_conversation_context else 0,
        }

        context = PipelineContext(
            request=request,
            rewritten_query=intent_classifier_query,
            last_qa_state=resolution.state,
            conversation_results=conversation_results,
            intent=intent,
            chat_history=chat_history,
            conversation_retrieval=conversation_retrieval,
            chat_history_source=chat_history_source,
            last_qa_trace=last_qa_trace,
            approved_conversation_context=approved_conversation_context,
            previous_last_qa_state=previous_last_qa_state,
        )
        with StageTimer("branch_execution", {"intent": intent.value}):
            branch_result = self.router.route(context, repository)
        outbox_job_ids = _branch_outbox_job_ids(branch_result)
        with StageTimer(
            "branch_index_sync",
            {"requested_job_count": len(outbox_job_ids)},
        ) as index_stage:
            processed_jobs = 0
            bm25 = getattr(self.retriever, "bm25", None)
            chroma = getattr(self.retriever, "chroma", None)
            outbox_config = getattr(self.config, "outbox", None)
            if outbox_job_ids and bm25 is not None and chroma is not None and outbox_config is not None:
                try:
                    processed_jobs = BackgroundIndexer(
                        repository=repository,
                        bm25=bm25,
                        chroma=chroma,
                        config=outbox_config,
                    ).process_job_ids(outbox_job_ids)
                    index_stage.metadata["sync_mode"] = "request_scoped"
                except Exception as exc:
                    # The branch transaction is already committed. Derived
                    # cache failure must remain retryable/observable without
                    # falsely reporting that the durable user operation failed.
                    branch_result = replace(
                        branch_result,
                        warnings=list(
                            dict.fromkeys(
                                [
                                    *branch_result.warnings,
                                    "request_scoped_index_sync_failed",
                                ]
                            )
                        ),
                    )
                    index_stage.metadata["sync_mode"] = (
                        "request_scoped_degraded"
                    )
                    index_stage.metadata["sync_error"] = type(exc).__name__
            elif outbox_job_ids:
                # Deterministic/unit-injected retrievers may intentionally have
                # no derived stores. Keep the durable jobs pending for the
                # normal background worker instead of losing the branch result.
                index_stage.metadata["sync_mode"] = "durable_outbox_deferred"
            index_stage.metadata["claimed_job_count"] = processed_jobs
        with StageTimer("bundling"):
            bundled = self.bundler.bundle(
                request=request,
                rewritten_query=intent_classifier_query,
                branch_result=branch_result,
            )
        if (
            resolution.interaction_type
            is LastQAInteractionType.REMINDER_NOTIFICATION_REPLY
            and resolution.is_authoritative_state
            and resolution.state is not None
        ):
            reminder_state = verified_reminder_state(
                resolution.state.reminder_state,
                resolution.state.reminder_state_hash,
            )
            if reminder_state is not None:
                replied_state = mark_reminder_state_replied(
                    reminder_state,
                    reply_hop_id=bundled.conversation_hop_id,
                )
                bundled = replace(
                    bundled,
                    last_qa_state=replace(
                        bundled.last_qa_state,
                        reminder_state=replied_state,
                        reminder_state_hash=reminder_state_hash(replied_state),
                    ),
                )
        current_artifacts = _artifact_records(
            bundled.platform_payload.get("artifacts")
        )
        new_artifact_ids = _artifact_ids(current_artifacts)
        binder = getattr(repository, "bind_generated_artifacts", None)
        if new_artifact_ids and bundled.conversation_hop_id and callable(binder):
            with StageTimer(
                "artifact_hop_binding",
                {"artifact_count": len(new_artifact_ids)},
            ):
                try:
                    binder(
                        user_id=request.user_id,
                        artifact_ids=new_artifact_ids,
                        conversation_hop_id=bundled.conversation_hop_id,
                    )
                except Exception:
                    # Same-turn downloads/delivery still use the validated
                    # in-memory artifact records. A binding failure must not
                    # discard a successfully generated file or response.
                    GLOBAL_METRICS.increment("artifact_hop_binding_failures_total")

        active_outbound = (
            resolution.state.outbound_state
            if resolution.state is not None
            and resolution.interaction_type
            is LastQAInteractionType.OUTBOUND_MESSAGE_ACTION
            else None
        )
        available_artifacts = _rehydrate_outbound_artifacts(
            repository=repository,
            user_id=request.user_id,
            outbound_state=active_outbound,
            current_artifacts=current_artifacts,
            source_hop_id=(
                active_outbound.source_hop_id
                if active_outbound is not None
                else explicit_parent_hop_id
            ),
        )
        active_branch_question = _has_active_branch_question(bundled)
        platform_response = _platform_response_for_supporting_answer(
            bundled,
            resolution,
        )
        with StageTimer(
            "platform_selector",
            {"bypassed_for_active_question": active_branch_question},
        ):
            contextual_selector = getattr(
                self.platform_selector,
                "select_with_outbound_context",
                None,
            )
            if active_branch_question:
                platform_payload = _platform_deferred_for_branch_question(bundled)
            elif callable(contextual_selector):
                platform_payload = contextual_selector(
                    platform_response,
                    request,
                    outbound_state=active_outbound,
                    outbound_action=resolution.outbound_action,
                    available_artifacts=available_artifacts,
                    new_artifact_ids=new_artifact_ids,
                )
            else:
                platform_payload = self.platform_selector.select(platform_response, request)
        platform_payload = _notice_only_platform_payload(platform_payload)
        with StageTimer("response_finalize"):
            delivery = platform_payload.get("delivery", {})
            if delivery.get("status") in {"needs_input", "pending_review", "failed", "partial_failure"}:
                final_chat_text = str(
                    delivery.get("notice")
                    or bundled.final_chat_text
                )
            elif delivery.get("status") == "sent":
                final_chat_text = f"Sent via {delivery.get('provider', delivery.get('channel', 'platform')).title()} to {delivery.get('recipient', 'the recipient')}."
            elif delivery.get("status") == "draft_ready":
                final_chat_text = (
                    f"{delivery.get('channel', 'Message').title()} draft is ready for review "
                    f"for {delivery.get('recipient', 'the recipient')}. It has not been sent. "
                    "To deliver it, explicitly ask to send the email."
                )
            elif delivery.get("status") == "draft_saved":
                final_chat_text = (
                    f"Draft saved in {delivery.get('provider', delivery.get('channel', 'the platform')).title()} "
                    f"for {delivery.get('recipient', 'the recipient')}."
                )
            else:
                final_chat_text = bundled.final_chat_text
        previous_outbound_state = (
            resolution.state.outbound_state
            if resolution.state is not None
            and resolution.state.outbound_state is not None
            else (
                previous_last_qa_state.outbound_state
                if previous_last_qa_state is not None
                else None
            )
        )
        outbound_state = _outbound_state_from_platform(
            platform_payload=platform_payload,
            previous_state=previous_outbound_state,
            source_topic_id=bundled.conversation_topic_id,
            source_hop_id=bundled.conversation_hop_id,
        )
        bundled = replace(
            bundled,
            final_chat_text=final_chat_text,
            last_qa_state=replace(
                bundled.last_qa_state,
                last_response=final_chat_text,
                outbound_state=outbound_state,
            ),
            platform_payload=platform_payload,
        )
        # SQL remains the durable restoration source.  Cache the safe execution
        # cursor on the owned hop so reopening a conversation survives Last-QA
        # expiry or a process restart without storing credentials or file paths.
        state_merger = getattr(repository, "merge_conversation_hop_entities", None)
        if bundled.conversation_hop_id and callable(state_merger):
            with StageTimer("conversation_execution_state_persistence"):
                try:
                    with repository.transaction() as cursor:
                        state_merger(
                            cursor,
                            user_id=request.user_id,
                            hop_id=bundled.conversation_hop_id,
                            entities={
                                **(
                                    {"conversation_id": request.conversation_id}
                                    if request.conversation_id
                                    else {}
                                ),
                                "final_chat_text": final_chat_text,
                                "outbound_state": _outbound_state_payload(
                                    outbound_state
                                ),
                            },
                        )
                except Exception:
                    GLOBAL_METRICS.increment(
                        "conversation_execution_state_persistence_failures_total"
                    )
        # Record delivery metadata separately from the conversation answer. The
        # record intentionally contains no credential, token, or attachment path.
        recorder = getattr(repository, "record_platform_delivery", None)
        with StageTimer("platform_delivery_audit"):
            if callable(recorder) and delivery.get("channel") not in (None, "none"):
                try:
                    recorder(
                        user_id=request.user_id,
                        conversation_hop_id=bundled.conversation_hop_id,
                        channel=str(delivery.get("channel")),
                        status=str(delivery.get("status", "unknown")),
                        recipient=str(delivery.get("recipient") or bundled.platform_payload.get("draft", {}).get("recipient") or ""),
                        message=bundled.platform_payload.get("draft", {}),
                        error_message=str(delivery.get("notice") or "") if delivery.get("status") in {"failed", "partial_failure", "needs_input"} else None,
                    )
                except Exception:
                    # Delivery history must not hide a completed send or response.
                    pass
        with StageTimer("last_qa_persistence"):
            try:
                _last_qa_save(self.last_qa_store, request, bundled.last_qa_state)
            except Exception:
                if not _bundled_has_irreversible_effects(bundled):
                    raise
                bundled = replace(
                    bundled,
                    warnings=list(dict.fromkeys(
                        [*bundled.warnings, "post_effect_last_qa_persistence_failed"]
                    )),
                )
        with StageTimer("chat_output"):
            try:
                self.chat_output.emit(bundled)
            except Exception:
                if not _bundled_has_irreversible_effects(bundled):
                    raise
                bundled = replace(
                    bundled,
                    warnings=list(dict.fromkeys(
                        [*bundled.warnings, "post_effect_chat_output_failed"]
                    )),
                )
        trace = current_trace()
        return replace(bundled, trace_summary=trace.summary() if trace else None)
