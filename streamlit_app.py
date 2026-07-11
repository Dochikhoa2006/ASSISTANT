from __future__ import annotations

from datetime import datetime, timezone
import json
from uuid import uuid4

from assistant_rag.contracts import ChatRequest
from assistant_rag.llm import LLMTask
from assistant_rag.platform import GmailSender
from assistant_rag.production_factory import build_production_pipeline, build_production_repository
from assistant_rag.settings import ProductionSettings


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

    with st.sidebar:
        st.subheader("Identity & Gmail")
        entered_user_id = st.text_input(
            "User ID",
            value="",
            placeholder="default_user",
            help="Leave empty to use default_user.",
        )
        user_id = entered_user_id.strip() or "default_user"
        gmail_username = st.text_input("Gmail username", value="")
        gmail_app_password = st.text_input("Gmail app password", value="", type="password")
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
        st.session_state.chat_messages.append({"role": "assistant", "content": response.final_chat_text})
        delivery = response.platform_payload.get("delivery", {})
        if delivery.get("channel") not in (None, "none"):
            st.caption(f"Delivery: {delivery.get('channel')} — {delivery.get('status')}")


if __name__ == "__main__":
    main()
