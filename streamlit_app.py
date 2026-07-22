from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from html import escape
import json
import logging
from pathlib import Path
from typing import Callable
from uuid import uuid4

from assistant_rag.artifacts import artifact_mime_type, resolve_generated_artifacts
from assistant_rag.contracts import ChatRequest, ResponseType
from assistant_rag.llm import LLMTask
from assistant_rag.platform import GmailSender
from assistant_rag.observability import new_request_id, start_trace, trace_summary_asdict
from assistant_rag.reminder_reply import (
    build_reminder_reply_context_index,
    build_reminder_reply_last_qa,
    build_reminder_reply_metadata,
    reminder_notification_key,
    reminder_reply_conversation_id,
    save_reminder_reply_last_qa,
)
from assistant_rag.production_factory import (
    build_production_runtime,
)
from assistant_rag.request_lifecycle import (
    ChatRequestExecution,
    ChatRequestLifecycleExecutor,
    RequestLifecycleConflict,
)
from assistant_rag.settings import ProductionSettings


logger = logging.getLogger(__name__)


def _initialise_chat_state(st: object) -> None:
    """Keep UI-only chat state separate from the assistant's persisted history."""
    session_state = st.session_state
    if "chat_messages" not in session_state:
        session_state.chat_messages = []
    if "chat_session_summaries" not in session_state:
        session_state.chat_session_summaries = []
    if "chat_view_id" not in session_state:
        session_state.chat_view_id = uuid4().hex


def _refresh_durable_conversation_summaries(
    st: object,
    *,
    repository: object,
    user_id: str,
) -> None:
    """Hydrate the sidebar from SQL without crossing the authenticated user."""

    loader = getattr(repository, "list_conversations", None)
    if not callable(loader):
        return
    try:
        durable = loader(user_id=user_id, limit=20)
    except Exception:
        logger.exception("durable conversation listing failed")
        return
    existing = {
        str(item.get("view_id") or ""): item
        for item in st.session_state.get("chat_session_summaries", [])
        if isinstance(item, dict)
    }
    summaries: list[dict[str, object]] = []
    for item in durable:
        if not isinstance(item, dict):
            continue
        conversation_id = str(item.get("conversation_id") or "").strip()
        if not conversation_id:
            continue
        cached = existing.get(conversation_id, {})
        summaries.append(
            {
                "view_id": conversation_id,
                "created_at": str(
                    item.get("updated_at") or item.get("created_at") or ""
                ),
                "summary": str(
                    item.get("summary") or item.get("title") or "Conversation"
                ),
                "messages": list(cached.get("messages") or []),
                "latest_hop_id": item.get("latest_hop_id"),
            }
        )
    st.session_state.chat_session_summaries = summaries


def _load_durable_chat_messages(
    *,
    repository: object,
    user_id: str,
    conversation_id: str,
) -> list[dict[str, object]]:
    loader = getattr(repository, "load_conversation", None)
    if not callable(loader):
        return []
    try:
        conversation = loader(
            user_id=user_id,
            conversation_id=conversation_id,
        )
    except Exception:
        logger.exception("durable conversation loading failed")
        return []
    if not isinstance(conversation, dict):
        return []
    return [
        dict(message)
        for message in conversation.get("messages", [])
        if isinstance(message, dict)
    ]


def _render_gmail_credential_check(st: object) -> None:
    """Render a fragment so credential checks do not rerun the full app."""
    @st.fragment
    def credential_check() -> None:
        if st.button(
            "Check Gmail credentials",
            help="Checks Gmail sending access only; it does not send mail or save a draft.",
            key="check_gmail_credentials",
        ):
            valid, message = GmailSender().validate_smtp_credentials(
                st.session_state.get("gmail_username", ""),
                st.session_state.get("gmail_app_password", ""),
            )
            st.session_state.gmail_credential_check_result = {"valid": valid, "message": message}
        result = st.session_state.get("gmail_credential_check_result")
        if result:
            (st.success if result["valid"] else st.error)(result["message"])

    credential_check()


def _clear_gmail_credential_check_result(st: object) -> None:
    st.session_state.pop("gmail_credential_check_result", None)


def _render_artifact_downloads(st: object, artifacts: list[dict[str, object]]) -> None:
    """Offer every generated file or visibly report why it is unavailable."""
    downloadable, unavailable = resolve_generated_artifacts(artifacts)
    if not downloadable:
        for label in unavailable:
            st.error(f"Generated attachment unavailable: {label}.")
        return
    st.caption("Generated files")
    for artifact in downloadable:
        path = Path(str(artifact["storage_path"]))
        filename = str(artifact["filename"])
        try:
            data = path.read_bytes()
        except OSError:
            st.error(f"Generated attachment unavailable: {filename}.")
            continue
        artifact_key = str(artifact.get("artifact_id") or "").strip() or sha256(
            str(path).encode("utf-8")
        ).hexdigest()[:16]
        st.download_button(
            label=f"Download {filename}",
            data=data,
            file_name=filename,
            mime=artifact_mime_type(filename),
            key=f"artifact-download-{artifact_key}",
        )
    for label in unavailable:
        st.error(f"Generated attachment unavailable: {label}.")


def _execute_user_request(
    *,
    pipeline: object,
    repository: object,
    request: ChatRequest,
    before_pipeline: Callable[[ChatRequest], None] | None = None,
) -> ChatRequestExecution:
    """Run the same confirmation/idempotency lifecycle as the HTTP API."""

    request_id = new_request_id(request.idempotency_key)
    start_trace(request_id)
    return ChatRequestLifecycleExecutor(
        pipeline=pipeline,
        repository=repository,
    ).execute(
        request,
        fallback_request_id=request_id,
        before_pipeline=before_pipeline,
    )


def _execute_user_request_safely(
    *,
    pipeline: object,
    repository: object,
    request: ChatRequest,
    before_pipeline: Callable[[ChatRequest], None] | None = None,
) -> tuple[ChatRequestExecution | None, dict[str, object] | None]:
    """Keep a terminal lifecycle failure from replacing the Streamlit session.

    The production pipeline remains responsible for its own typed fallbacks.
    This is only the outer UI boundary for an exception that escaped that
    contract.  It deliberately returns a terminal statement instead of another
    clarification question, so a backend outage cannot create a question loop.
    """

    try:
        return (
            _execute_user_request(
                pipeline=pipeline,
                repository=repository,
                request=request,
                before_pipeline=before_pipeline,
            ),
            None,
        )
    except RequestLifecycleConflict as exc:
        logger.warning(
            "streamlit request lifecycle conflict",
            extra={"payload": {"user_id": request.user_id, "reason": str(exc)}},
        )
        text = (
            "This request is already being processed or conflicts with an earlier "
            "request, so it was not run again."
        )
    except Exception:
        logger.exception(
            "streamlit request execution failed",
            extra={"payload": {"user_id": request.user_id}},
        )
        text = (
            "I could not complete this request safely. The current conversation "
            "is still available, and you can retry when the service is ready."
        )
    return None, {
        "role": "assistant",
        "content": text,
        "artifacts": [],
        "pending_confirmations": [],
        "response_type": ResponseType.ERROR.value,
    }


def _chat_message_from_execution(execution: ChatRequestExecution) -> dict[str, object]:
    response = execution.response
    payload = execution.payload
    if response is not None:
        platform_payload = response.platform_payload
        return {
            "role": "assistant",
            "content": response.final_chat_text,
            "artifacts": list(platform_payload.get("artifacts") or []),
            "pending_confirmations": list(response.actions_pending_confirmation),
            "response_type": response.response_type.value,
            "conversation_topic_id": response.conversation_topic_id,
            "conversation_hop_id": response.conversation_hop_id,
        }
    platform_payload = payload.get("platform_payload") or {}
    return {
        "role": "assistant",
        "content": str(payload.get("final_chat_text") or "Request already completed."),
        "artifacts": list(platform_payload.get("artifacts") or []),
        "pending_confirmations": list(
            payload.get("actions_pending_confirmation") or []
        ),
        "response_type": str(payload.get("response_type") or "normal"),
        "conversation_topic_id": payload.get("conversation_topic_id"),
        "conversation_hop_id": payload.get("conversation_hop_id"),
    }


def _latest_visible_hop_id(messages: list[dict[str, object]]) -> str | None:
    for message in reversed(messages):
        hop_id = str(message.get("conversation_hop_id") or "").strip()
        if message.get("role") == "assistant" and hop_id:
            return hop_id
    return None


def _render_confirmation_controls(
    st: object,
    *,
    message: dict[str, object],
    pipeline: object,
    repository: object,
    user_id: str,
    gmail_username: str,
    gmail_app_password: str,
) -> None:
    """Render durable, exactly-once confirmation controls for one answer."""

    confirmations = list(message.get("pending_confirmations") or [])
    if not confirmations:
        return
    st.caption("This change is waiting for your confirmation.")
    for confirmation in confirmations:
        if not isinstance(confirmation, dict):
            continue
        token = str(confirmation.get("confirmation_token") or "").strip()
        if not token:
            continue
        action_type = str(confirmation.get("action_type") or "change")
        if st.button(
            f"Confirm {action_type.replace('_', ' ')}",
            key=f"confirm-action-{token}",
            type="primary",
        ):
            try:
                execution = _execute_user_request(
                    pipeline=pipeline,
                    repository=repository,
                    request=ChatRequest(
                        user_id=user_id,
                        raw_query=str(
                            message.get("request_query")
                            or "Confirm the pending action."
                        ),
                        conversation_id=st.session_state.get("chat_view_id"),
                        parent_hop_id=str(
                            message.get("conversation_hop_id") or ""
                        ).strip()
                        or None,
                        confirmation_token=token,
                        idempotency_key=f"streamlit-confirm:{token}",
                        platform_context={
                            "gmail_username": gmail_username,
                            "gmail_app_password": gmail_app_password,
                        },
                    ),
                )
            except RequestLifecycleConflict as exc:
                st.error(f"This change could not be confirmed: {exc}")
                continue
            except Exception:
                logger.exception("streamlit pending confirmation execution failed")
                st.error(
                    "The confirmed change could not be completed. It remains pending; "
                    "please try again."
                )
                continue

            confirmed_message = _chat_message_from_execution(execution)
            confirmation_failed = (
                confirmed_message.get("response_type") == ResponseType.ERROR.value
            )
            if not confirmation_failed:
                message["pending_confirmations"] = []
            st.session_state.chat_messages.extend(
                [
                    {"role": "user", "content": "Confirm the pending change."},
                    confirmed_message,
                ]
            )
            if confirmation_failed:
                st.session_state.confirmation_error_notice = (
                    "The confirmed change was not completed. It remains pending; "
                    "please try again."
                )
            else:
                st.session_state.confirmation_notice = (
                    "The confirmed request was processed exactly once."
                )
            st.rerun()


def _session_transcript(messages: list[dict[str, str]], *, max_chars: int = 12_000) -> str:
    """Build a bounded, model-safe transcript from the visible UI conversation."""
    transcript = "\n\n".join(
        f"{message.get('role', 'assistant').upper()}: {message.get('content', '').strip()}"
        for message in messages
        if message.get("content", "").strip()
    )
    return transcript[-max_chars:]


def _summarize_chat_session(pipeline: object, messages: list[dict[str, str]]) -> str:
    """Use the configured pipeline LLM, without creating a database record."""
    transcript = _session_transcript(messages)
    if not transcript:
        return "No messages were exchanged in this chat."

    llm = getattr(getattr(pipeline, "platform_selector", None), "llm", None)
    if llm is None:
        return "Chat closed. A summary could not be generated because no LLM is configured."

    try:
        summary = llm.chat(
            task=LLMTask.WRITING,
            system_prompt=(
                "Summarize this chat session for the user in at most five concise bullets. "
                "Include key decisions, requested actions, and unresolved items. "
                "Do not invent facts and do not mention database sessions or system internals."
            ),
            user_prompt=json.dumps({"chat_transcript": transcript}, ensure_ascii=False),
        ).strip()
        if summary:
            return summary
    except Exception:
        # Starting a clean visual chat must remain available even if the model
        # service is temporarily unavailable. No summary is persisted anywhere.
        pass
    return "Chat closed. The LLM summary was unavailable for this session."


def _catch_up_reminders(st: object) -> dict[str, int] | None:
    """Create durable UI notifications for all due users after a UI start."""
    try:
        summary = st.session_state.runtime.reminder_autoscan.catch_up_due(
            now_value=datetime.now(timezone.utc).isoformat(), batch_size=100
        )
        st.session_state.last_reminder_autoscan = summary
        st.session_state.pop("reminder_autoscan_error", None)
        return summary
    except Exception:
        # The durable worker may be running concurrently. Do not surface raw
        # backend errors or block chat; retain a safe retry notice instead.
        logger.exception("streamlit reminder autoscan failed")
        st.session_state.reminder_autoscan_error = True
        return None


def _format_notification_time(notification: dict[str, object]) -> str:
    value = notification.get("fire_time") or notification.get("reminder_time") or notification.get("created_at")
    if not value:
        return "Time unavailable"
    try:
        timestamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        return timestamp.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    except ValueError:
        return str(value)


def _render_reminder_notifications(
    st: object, *, repository: object, pipeline: object, user_id: str,
) -> None:
    """Render every durable notification, nearest at the sidebar top."""
    notifications = repository.list_notifications(user_id=user_id)
    unread_count = sum(item.get("ui_status") == "unread" for item in notifications)
    st.subheader(f"Reminders · {unread_count} unread")
    if not notifications:
        st.caption("No due reminder notifications.")
        return

    # Keep the state treatment unambiguous: read notifications use the red
    # treatment requested for this UI, while notifications not yet read use a
    # deliberately quiet grey.  Only fixed class names are injected; all
    # database-backed reminder text is escaped below.
    st.markdown(
        """
        <style>
        .reminder-notification-header {
            border-radius: 10px;
            margin: 0.15rem 0 0.55rem;
            padding: 0.7rem 0.8rem;
        }
        .reminder-notification-read {
            background: linear-gradient(135deg, #7f1d1d, #b91c1c);
            border: 1px solid #ef4444;
            color: #fff7ed;
        }
        .reminder-notification-unread {
            background: linear-gradient(135deg, #374151, #4b5563);
            border: 1px solid #6b7280;
            color: #f9fafb;
        }
        .reminder-notification-state {
            font-size: 0.7rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            opacity: 0.92;
        }
        .reminder-notification-title {
            font-size: 1rem;
            font-weight: 700;
            line-height: 1.35;
            margin-top: 0.15rem;
        }
        .reminder-supporting-question {
            background: #f8fafc;
            border-left: 4px solid #64748b;
            border-radius: 5px;
            color: #1e293b;
            margin: 0.45rem 0;
            padding: 0.55rem 0.65rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    bulk_read, bulk_unread = st.columns(2)
    if bulk_read.button("👁 Mark all read", type="primary", use_container_width=True):
        repository.update_all_notification_ui_status(user_id=user_id, ui_status="read")
        st.rerun()
    if bulk_unread.button("◉ Mark all unread", use_container_width=True):
        repository.update_all_notification_ui_status(user_id=user_id, ui_status="unread")
        st.rerun()

    context_index = build_reminder_reply_context_index(
        repository=repository, user_id=user_id, notifications=notifications
    )

    # The repository orders fire time descending: nearest due notification is
    # at the top, while increasingly older/farther overdue items flow down.
    with st.container(height=420):
        for notification in notifications:
            unread = notification.get("ui_status") == "unread"
            subject = str(notification.get("subject") or "Reminder")
            summary = str(notification.get("reminder_summary") or "")
            context = context_index.get(
                reminder_notification_key(
                    str(notification["reminder_id"]), str(notification["notification_id"])
                ),
                {},
            )
            # Normalize legacy JSON/plain-text question shapes into the exact
            # question the reply pipeline receives, so the UI never displays
            # an opaque serialized payload to the user.
            reply_metadata = build_reminder_reply_metadata(
                reminder_id=str(notification["reminder_id"]),
                notification_id=str(notification["notification_id"]),
                context=context,
            ) if context else {}
            supporting_question = reply_metadata.get("supporting_question")
            read = not unread
            status_text = "ALREADY READ" if read else "NOT READ YET"
            state_class = "reminder-notification-read" if read else "reminder-notification-unread"

            with st.container(border=True):
                st.markdown(
                    f"""
                    <div class="reminder-notification-header {state_class}">
                      <div class="reminder-notification-state">REMINDER NOTIFICATION · {status_text}</div>
                      <div class="reminder-notification-title">{escape(subject)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.caption(f"Notification time: {_format_notification_time(notification)}")
                if summary:
                    st.markdown(f"**Reminder details:** {summary}")
                if supporting_question:
                    st.markdown(
                        "<div class=\"reminder-supporting-question\">"
                        "<strong>Supporting question</strong><br>"
                        f"{escape(str(supporting_question))}</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.caption("Supporting question: none for this reminder.")

                toggle_column, _ = st.columns((1, 1))
                toggle_label = "👁 Mark as read" if unread else "◉ Mark as unread"
                target_status = "read" if unread else "unread"
                if toggle_column.button(
                    toggle_label,
                    key=f"notification-state-{notification['notification_id']}",
                    use_container_width=True,
                ):
                    repository.update_notification_ui_status(
                        user_id=user_id,
                        notification_id=str(notification["notification_id"]),
                        ui_status=target_status,
                    )
                    st.rerun()

                with st.form(key=f"notification-reply-form-{notification['notification_id']}", clear_on_submit=True):
                    reply_text = st.text_area(
                        "Reply or comment for the assistant",
                        key=f"notification-reply-text-{notification['notification_id']}",
                        placeholder="Answer the supporting question, add a comment, or ask for help…",
                    )
                    submitted = st.form_submit_button("Send to chatbot")
                if submitted:
                    reply = reply_text.strip()
                    if not reply:
                        st.warning("Enter a reply or comment first.")
                    elif not context or not context.get("source_hop_id"):
                        st.warning("This reminder’s source conversation is unavailable. Nothing was sent.")
                    else:
                        # The hash lookup restores the source hop's user query,
                        # response, and supporting questions before normal routing.
                        reply_last_qa = build_reminder_reply_last_qa(
                            context,
                            reminder_id=str(notification["reminder_id"]),
                            notification_id=str(notification["notification_id"]),
                        )
                        source_conversation_id = reminder_reply_conversation_id(
                            context
                        )
                        reply_fingerprint = sha256(
                            (
                                f"{notification['notification_id']}\0{reply}"
                            ).encode("utf-8")
                        ).hexdigest()
                        execution, failure_message = _execute_user_request_safely(
                            pipeline=pipeline,
                            repository=repository,
                            request=ChatRequest(
                                user_id=user_id,
                                raw_query=reply,
                                conversation_id=source_conversation_id,
                                reminder_id=str(notification["reminder_id"]),
                                notification_id=str(notification["notification_id"]),
                                reply_text=reply,
                                parent_hop_id=context.get("source_hop_id"),
                                idempotency_key=(
                                    f"streamlit-reminder-reply:"
                                    f"{notification['notification_id']}:"
                                    f"{reply_fingerprint}"
                                ),
                                metadata=reply_metadata,
                                platform_context={
                                    "gmail_username": st.session_state.get("gmail_username", ""),
                                    "gmail_app_password": st.session_state.get("gmail_app_password", ""),
                                },
                            ),
                            # Hydrate the reminder's source Last-QA only after a
                            # fresh/failed-retry idempotency claim. Replays and
                            # conflicts must never rewind the user's latest state.
                            before_pipeline=lambda _request: save_reminder_reply_last_qa(
                                pipeline.last_qa_store,
                                user_id=user_id,
                                state=reply_last_qa,
                                conversation_id=source_conversation_id,
                            ),
                        )
                        if execution is None:
                            assert failure_message is not None
                            failure_message["request_query"] = reply
                            st.session_state.chat_messages.extend([
                                {"role": "user", "content": reply},
                                failure_message,
                            ])
                            st.error(str(failure_message["content"]))
                            # The notification remains unread and the stable
                            # idempotency key makes a later retry safe.
                            continue
                        response_message = _chat_message_from_execution(execution)
                        response_message["request_query"] = reply
                        if source_conversation_id:
                            st.session_state.chat_view_id = source_conversation_id
                            restored_messages = _load_durable_chat_messages(
                                repository=repository,
                                user_id=user_id,
                                conversation_id=source_conversation_id,
                            )
                        else:
                            restored_messages = []
                        if restored_messages:
                            st.session_state.chat_messages = restored_messages
                        else:
                            st.session_state.chat_messages.extend([
                                {"role": "user", "content": reply},
                                response_message,
                            ])
                        if (
                            response_message.get("response_type")
                            == ResponseType.ERROR.value
                        ):
                            st.session_state.reminder_reply_error_notice = (
                                "Your reminder reply could not be completed. The notification "
                                "was left unread so you can retry."
                            )
                        else:
                            # Only a successful pipeline result explicitly acknowledges
                            # the notification. Error responses must remain retryable.
                            repository.update_notification_ui_status(
                                user_id=user_id,
                                notification_id=str(notification["notification_id"]),
                                ui_status="read",
                            )
                            st.session_state.reminder_reply_notice = (
                                "Your reminder reply was sent through the main chatbot."
                            )
                        st.rerun()
            st.divider()


def _render_reminder_notifications_safely(
    st: object,
    *,
    repository: object,
    pipeline: object,
    user_id: str,
) -> None:
    """Keep a reminder-panel outage isolated from the main chat surface."""

    try:
        _render_reminder_notifications(
            st,
            repository=repository,
            pipeline=pipeline,
            user_id=user_id,
        )
    except Exception:
        logger.exception(
            "streamlit reminder panel failed",
            extra={"payload": {"user_id": user_id}},
        )
        st.warning(
            "Reminder notifications are temporarily unavailable. Chat remains available."
        )


def main() -> None:
    try:
        import streamlit as st
    except ImportError as exc:
        raise RuntimeError("Install streamlit to run the assistant UI") from exc

    st.set_page_config(page_title="SQL-First RAG Assistant", layout="wide")
    st.title("SQL-First RAG Assistant")
    settings = ProductionSettings.from_env()
    if "runtime" not in st.session_state:
        st.session_state.runtime = build_production_runtime(settings)
    st.session_state.pipeline = st.session_state.runtime.pipeline
    st.session_state.repository = st.session_state.runtime.repository

    _initialise_chat_state(st)
    # A Streamlit process/session can start after reminders became due. Drain
    # the global durable backlog once immediately; duplicate database rows are
    # prevented by the reminder/fire-time uniqueness contract.
    if "startup_reminder_catchup_done" not in st.session_state:
        _catch_up_reminders(st)
        st.session_state.startup_reminder_catchup_done = True

    with st.sidebar:
        st.subheader("Identity")
        entered_user_id = st.text_input(
            "User ID",
            value="",
            placeholder="default_user",
            help="Leave empty to use default_user.",
        )
        user_id = entered_user_id.strip() or "default_user"
        active_chat_user_id = st.session_state.get("chat_user_id")
        if active_chat_user_id is None:
            st.session_state.chat_user_id = user_id
        elif active_chat_user_id != user_id:
            # Never retain visible messages, confirmations, conversation
            # cursors, or Gmail credentials across a user identity change.
            st.session_state.chat_user_id = user_id
            st.session_state.chat_messages = []
            st.session_state.chat_session_summaries = []
            st.session_state.chat_view_id = uuid4().hex
            st.session_state.gmail_username = ""
            st.session_state.gmail_app_password = ""
            st.session_state.pop("gmail_credential_check_result", None)

        _refresh_durable_conversation_summaries(
            st,
            repository=st.session_state.repository,
            user_id=user_id,
        )

        if st.button("Refresh reminder notifications"):
            _catch_up_reminders(st)
            st.rerun()
        autoscan_summary = st.session_state.get("last_reminder_autoscan")
        if autoscan_summary and any(
            autoscan_summary.get(key, 0)
            for key in (
                "planned",
                "needs_review",
                "supporting_planned",
                "supporting_needs_review",
                "notified",
            )
        ):
            st.caption(
                "Autoscan catch-up: "
                f"{autoscan_summary.get('notified', 0)} notified, "
                f"{autoscan_summary.get('planned', 0)} timed plans finalized, "
                f"{autoscan_summary.get('needs_review', 0)} future timings need review, "
                f"{autoscan_summary.get('supporting_planned', 0)} supporting "
                "decisions finalized, "
                f"{autoscan_summary.get('supporting_questions_created', 0)} "
                "supporting questions created, "
                f"{autoscan_summary.get('supporting_needs_review', 0)} "
                "supporting decisions need review."
            )
        if st.session_state.get("reminder_autoscan_error"):
            st.warning("Reminder catch-up is temporarily unavailable. Use refresh to retry; chat remains available.")

        _render_reminder_notifications_safely(
            st,
            repository=st.session_state.repository,
            pipeline=st.session_state.pipeline,
            user_id=user_id,
        )

        st.divider()
        st.subheader("Gmail & Debug")
        gmail_username = st.text_input(
            "Gmail username",
            value="",
            key="gmail_username",
            on_change=_clear_gmail_credential_check_result,
            args=(st,),
        )
        gmail_app_password = st.text_input(
            "Gmail app password",
            value="",
            type="password",
            key="gmail_app_password",
            on_change=_clear_gmail_credential_check_result,
            args=(st,),
        )
        show_pipeline_trace = st.checkbox(
            "Show real pipeline trace",
            value=False,
            help="Show the exact production stages and latency for this request.",
        )
        _render_gmail_credential_check(st)

        st.divider()
        if st.button("New chat", type="primary", help="Save this conversation and start a separate conversation cursor."):
            with st.spinner("Summarizing this chat session…"):
                summary = _summarize_chat_session(
                    st.session_state.pipeline,
                    st.session_state.chat_messages,
                )
            st.session_state.chat_session_summaries.append({
                "view_id": st.session_state.chat_view_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "summary": summary,
                "messages": [dict(message) for message in st.session_state.chat_messages],
                "latest_hop_id": _latest_visible_hop_id(
                    st.session_state.chat_messages
                ),
            })
            st.session_state.chat_messages = []
            st.session_state.chat_view_id = uuid4().hex
            st.session_state.new_chat_notice = (
                "Started a separate conversation. The previous conversation "
                "can be continued from its saved summary."
            )
            st.rerun()

        summaries = st.session_state.chat_session_summaries
        if summaries:
            with st.expander("Previous chat summaries", expanded=False):
                visible_summaries = [
                    item
                    for item in summaries
                    if item.get("view_id") != st.session_state.chat_view_id
                ]
                for item in reversed(visible_summaries[-5:]):
                    st.caption(item["created_at"])
                    st.write(item["summary"])
                    if st.button(
                        "Continue this conversation",
                        key=f"continue-chat-{item['view_id']}",
                    ):
                        current_messages = [
                            dict(message)
                            for message in st.session_state.chat_messages
                        ]
                        if current_messages:
                            st.session_state.chat_session_summaries.append(
                                {
                                    "view_id": st.session_state.chat_view_id,
                                    "created_at": datetime.now(
                                        timezone.utc
                                    ).isoformat(),
                                    "summary": "Paused conversation.",
                                    "messages": current_messages,
                                    "latest_hop_id": _latest_visible_hop_id(
                                        current_messages
                                    ),
                                }
                            )
                        st.session_state.chat_session_summaries = [
                            saved
                            for saved in st.session_state.chat_session_summaries
                            if saved is not item
                        ]
                        st.session_state.chat_view_id = item["view_id"]
                        durable_messages = _load_durable_chat_messages(
                            repository=st.session_state.repository,
                            user_id=user_id,
                            conversation_id=str(item["view_id"]),
                        )
                        st.session_state.chat_messages = durable_messages or [
                            dict(message)
                            for message in item.get("messages", [])
                        ]
                        st.session_state.new_chat_notice = (
                            "Restored the selected conversation and its exact "
                            "execution cursor."
                        )
                        st.rerun()

    notice = st.session_state.pop("new_chat_notice", None)
    if notice:
        st.success(notice)
    reminder_reply_notice = st.session_state.pop("reminder_reply_notice", None)
    if reminder_reply_notice:
        st.success(reminder_reply_notice)
    confirmation_notice = st.session_state.pop("confirmation_notice", None)
    if confirmation_notice:
        st.success(confirmation_notice)
    confirmation_error_notice = st.session_state.pop(
        "confirmation_error_notice", None
    )
    if confirmation_error_notice:
        st.error(confirmation_error_notice)
    reminder_reply_error_notice = st.session_state.pop(
        "reminder_reply_error_notice", None
    )
    if reminder_reply_error_notice:
        st.error(reminder_reply_error_notice)

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message["role"] == "assistant":
                _render_artifact_downloads(st, list(message.get("artifacts") or []))
                _render_confirmation_controls(
                    st,
                    message=message,
                    pipeline=st.session_state.pipeline,
                    repository=st.session_state.repository,
                    user_id=user_id,
                    gmail_username=gmail_username,
                    gmail_app_password=gmail_app_password,
                )

    query = st.chat_input("Ask the assistant")
    if query:
        st.session_state.chat_messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                # Streamlit uses the same production assembly and request
                # lifecycle as interactive debug; tracing only observes it.
                request = ChatRequest(
                    user_id=user_id,
                    raw_query=query,
                    conversation_id=st.session_state.chat_view_id,
                    parent_hop_id=_latest_visible_hop_id(
                        st.session_state.chat_messages
                    ),
                    idempotency_key=(
                        f"streamlit-chat:{st.session_state.chat_view_id}:"
                        f"{uuid4().hex}"
                    ),
                    platform_context={
                        "gmail_username": gmail_username,
                        "gmail_app_password": gmail_app_password,
                    },
                )
                execution, failure_message = _execute_user_request_safely(
                    pipeline=st.session_state.pipeline,
                    repository=st.session_state.repository,
                    request=request,
                )
                if execution is None:
                    assert failure_message is not None
                    failure_message["request_query"] = query
                    st.session_state.chat_messages.append(failure_message)
                    st.error(str(failure_message["content"]))
                    return
                response = execution.response
            response_message = _chat_message_from_execution(execution)
            response_message["request_query"] = query
            st.session_state.chat_messages.append(response_message)
            platform_payload = (
                response.platform_payload
                if response is not None
                else dict(execution.payload.get("platform_payload") or {})
            )
            st.write(response_message["content"])
            response_artifacts = list(response_message.get("artifacts") or [])
            _render_artifact_downloads(st, response_artifacts)
            _render_confirmation_controls(
                st,
                message=response_message,
                pipeline=st.session_state.pipeline,
                repository=st.session_state.repository,
                user_id=user_id,
                gmail_username=gmail_username,
                gmail_app_password=gmail_app_password,
            )
            if show_pipeline_trace:
                with st.expander("Production pipeline trace", expanded=False):
                    st.json(
                        (trace_summary_asdict(response.trace_summary) or {})
                        if response is not None
                        else {}
                    )
        logger.info(
            "streamlit_real_pipeline_request_completed",
            extra={"payload": {
                "response_type": response_message["response_type"],
                "trace_stage_count": (
                    len(response.trace_summary.stages)
                    if response is not None and response.trace_summary
                    else 0
                ),
                "conversation_hop_id": (
                    response.conversation_hop_id
                    if response is not None
                    else execution.payload.get("conversation_hop_id")
                ),
                "replayed": execution.replayed,
            }},
        )
        delivery = platform_payload.get("delivery", {})
        if delivery.get("channel") not in (None, "none"):
            st.caption(f"Delivery: {delivery.get('channel')} — {delivery.get('status')}")
            if delivery.get("notice"):
                st.info(str(delivery["notice"]))
            message = platform_payload.get("draft", {})
            if message:
                with st.expander(
                    f"{str(delivery.get('channel')).title()} delivery details",
                    expanded=delivery.get("status") in {"draft_ready", "draft_saved", "failed", "partial_failure", "needs_input"},
                ):
                    recipients = message.get("recipients") or [message.get("recipient")]
                    st.write(f"To: {', '.join(str(item) for item in recipients if item)}")
                    st.write(f"Subject: {message.get('subject') or '(none)'}")
                    st.write(f"Mode: {message.get('mode') or 'unknown'}")
                    st.text(message.get("body") or "")
                    attachments = message.get("attachments") or []
                    if attachments:
                        st.write(
                            "Attachments: "
                            + ", ".join(str(item.get("filename")) for item in attachments if item.get("filename"))
                        )


if __name__ == "__main__":
    main()
