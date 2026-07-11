from __future__ import annotations

from datetime import datetime, timezone
from html import escape
import json
import logging
from uuid import uuid4

from assistant_rag.contracts import ChatRequest
from assistant_rag.llm import LLMTask
from assistant_rag.autoscan import ReminderAutoscan
from assistant_rag.platform import GmailSender
from assistant_rag.observability import new_request_id, start_trace, trace_summary_asdict
from assistant_rag.reminder_reply import (
    build_reminder_reply_context_index,
    build_reminder_reply_last_qa,
    build_reminder_reply_metadata,
    reminder_notification_key,
)
from assistant_rag.production_factory import (
    build_production_pipeline,
    build_production_repository,
    build_reminder_timing_planner,
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


def _catch_up_reminders(st: object, settings: ProductionSettings) -> dict[str, int] | None:
    """Create durable UI notifications for all due users after a UI start."""
    if "reminder_autoscan" not in st.session_state:
        st.session_state.reminder_autoscan = ReminderAutoscan(
            st.session_state.repository,
            timing_planner=build_reminder_timing_planner(settings),
        )
    try:
        summary = st.session_state.reminder_autoscan.catch_up_due(
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
                        pipeline.last_qa_store.save(user_id, build_reminder_reply_last_qa(context))
                        start_trace(new_request_id())
                        response = pipeline.handle(
                            ChatRequest(
                                user_id=user_id,
                                raw_query=reply,
                                parent_hop_id=context.get("source_hop_id"),
                                metadata=reply_metadata,
                                platform_context={
                                    "gmail_username": st.session_state.get("gmail_username", ""),
                                    "gmail_app_password": st.session_state.get("gmail_app_password", ""),
                                },
                            ),
                            repository,
                        )
                        st.session_state.chat_messages.extend([
                            {"role": "user", "content": reply},
                            {"role": "assistant", "content": response.final_chat_text},
                        ])
                        # A successful reply is an explicit acknowledgement.
                        repository.update_notification_ui_status(
                            user_id=user_id,
                            notification_id=str(notification["notification_id"]),
                            ui_status="read",
                        )
                        st.session_state.reminder_reply_notice = "Your reminder reply was sent through the main chatbot."
                        st.rerun()
            st.divider()


def main() -> None:
    try:
        import streamlit as st
    except ImportError as exc:
        raise RuntimeError("Install streamlit to run the assistant UI") from exc

    st.set_page_config(page_title="SQL-First RAG Assistant", layout="wide")
    st.title("SQL-First RAG Assistant")
    settings = ProductionSettings.from_env()
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = build_production_pipeline(settings)
    
    if "repository" not in st.session_state:
        st.session_state.repository = build_production_repository(settings)

    _initialise_chat_state(st)
    # A Streamlit process/session can start after reminders became due. Drain
    # the global durable backlog once immediately; duplicate database rows are
    # prevented by the reminder/fire-time uniqueness contract.
    if "startup_reminder_catchup_done" not in st.session_state:
        _catch_up_reminders(st, settings)
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

        if st.button("Refresh reminder notifications"):
            _catch_up_reminders(st, settings)
            st.rerun()
        autoscan_summary = st.session_state.get("last_reminder_autoscan")
        if autoscan_summary and any(autoscan_summary.get(key, 0) for key in ("planned", "needs_review", "notified")):
            st.caption(
                "Autoscan catch-up: "
                f"{autoscan_summary.get('notified', 0)} notified, "
                f"{autoscan_summary.get('planned', 0)} timed plans finalized, "
                f"{autoscan_summary.get('needs_review', 0)} future timings need review."
            )
        if st.session_state.get("reminder_autoscan_error"):
            st.warning("Reminder catch-up is temporarily unavailable. Use refresh to retry; chat remains available.")

        _render_reminder_notifications(
            st,
            repository=st.session_state.repository,
            pipeline=st.session_state.pipeline,
            user_id=user_id,
        )

        st.divider()
        st.subheader("Gmail & Debug")
        gmail_username = st.text_input("Gmail username", value="", key="gmail_username")
        gmail_app_password = st.text_input("Gmail app password", value="", type="password", key="gmail_app_password")
        show_pipeline_trace = st.checkbox(
            "Show real pipeline trace",
            value=False,
            help="Show the exact production stages and latency for this request.",
        )
        if st.button("Check Gmail credentials"):
            valid, message = GmailSender().validate_credentials(gmail_username, gmail_app_password)
            (st.success if valid else st.error)(message)

        st.divider()
        if st.button("New chat", type="primary", help="Summarize and clear this visible chat. Your database conversation history is unchanged."):
            with st.spinner("Summarizing this chat session…"):
                summary = _summarize_chat_session(
                    st.session_state.pipeline,
                    st.session_state.chat_messages,
                )
            st.session_state.chat_session_summaries.append({
                "view_id": st.session_state.chat_view_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "summary": summary,
            })
            # This is only a new visual Streamlit session. The user_id and every
            # database conversation/topic/hop identifier remain exactly as-is.
            st.session_state.chat_messages = []
            st.session_state.chat_view_id = uuid4().hex
            st.session_state.new_chat_notice = "Started a new chat view. The previous view was summarized locally."
            st.rerun()

        summaries = st.session_state.chat_session_summaries
        if summaries:
            with st.expander("Previous chat summaries", expanded=False):
                for item in reversed(summaries[-5:]):
                    st.caption(item["created_at"])
                    st.write(item["summary"])

    notice = st.session_state.pop("new_chat_notice", None)
    if notice:
        st.success(notice)
    reminder_reply_notice = st.session_state.pop("reminder_reply_notice", None)
    if reminder_reply_notice:
        st.success(reminder_reply_notice)

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    query = st.chat_input("Ask the assistant")
    if query:
        st.session_state.chat_messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                # Streamlit uses the same production assembly and request
                # lifecycle as interactive debug; tracing only observes it.
                start_trace(new_request_id())
                response = st.session_state.pipeline.handle(
                    ChatRequest(
                        user_id=user_id,
                        raw_query=query,
                        # Do not pass the UI-only chat_view_id or its summary.
                        # Every visual chat deliberately shares the same
                        # application-level user/conversation history.
                        platform_context={
                            "gmail_username": gmail_username,
                            "gmail_app_password": gmail_app_password,
                        },
                    ),
                    st.session_state.repository,
                )
            st.write(response.final_chat_text)
            if show_pipeline_trace:
                with st.expander("Production pipeline trace", expanded=False):
                    st.json(trace_summary_asdict(response.trace_summary) or {})
        st.session_state.chat_messages.append({"role": "assistant", "content": response.final_chat_text})
        logger.info(
            "streamlit_real_pipeline_request_completed",
            extra={"payload": {
                "response_type": response.response_type.value,
                "trace_stage_count": len(response.trace_summary.stages) if response.trace_summary else 0,
                "conversation_hop_id": response.conversation_hop_id,
            }},
        )
        delivery = response.platform_payload.get("delivery", {})
        if delivery.get("channel") not in (None, "none"):
            st.caption(f"Delivery: {delivery.get('channel')} — {delivery.get('status')}")


if __name__ == "__main__":
    main()
