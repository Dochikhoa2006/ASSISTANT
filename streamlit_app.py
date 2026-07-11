from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from uuid import uuid4

from assistant_rag.contracts import ChatRequest
from assistant_rag.llm import LLMTask
from assistant_rag.autoscan import ReminderAutoscan
from assistant_rag.platform import GmailSender
from assistant_rag.observability import new_request_id, start_trace, trace_summary_asdict
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


def _render_reminder_notifications(st: object, *, repository: object, user_id: str) -> None:
    """Render every durable notification, nearest at the sidebar top."""
    notifications = repository.list_notifications(user_id=user_id)
    unread_count = sum(item.get("ui_status") == "unread" for item in notifications)
    st.subheader(f"Reminders · {unread_count} unread")
    if not notifications:
        st.caption("No due reminder notifications.")
        return

    # The repository orders fire time descending: nearest due notification is
    # at the top, while increasingly older/farther overdue items flow down.
    with st.container(height=420):
        for notification in notifications:
            unread = notification.get("ui_status") == "unread"
            subject = str(notification.get("subject") or "Reminder")
            summary = str(notification.get("reminder_summary") or "")
            prefix = "🔔" if unread else "✓"
            st.markdown(f"{prefix} **{subject}**")
            st.caption(_format_notification_time(notification))
            if summary:
                st.caption(summary)
            if unread and st.button("Mark read", key=f"notification-read-{notification['notification_id']}"):
                repository.update_notification_ui_status(
                    user_id=user_id,
                    notification_id=str(notification["notification_id"]),
                    ui_status="read",
                )
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
            st, repository=st.session_state.repository, user_id=user_id
        )

        st.divider()
        st.subheader("Gmail & Debug")
        gmail_username = st.text_input("Gmail username", value="")
        gmail_app_password = st.text_input("Gmail app password", value="", type="password")
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
