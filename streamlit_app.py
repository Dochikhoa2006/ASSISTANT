from __future__ import annotations

from assistant_rag.contracts import ChatRequest
from assistant_rag.platform import GmailSender
from assistant_rag.production_factory import build_production_pipeline, build_production_repository
from assistant_rag.settings import ProductionSettings


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
    query = st.chat_input("Ask the assistant")
    if query:
        response = st.session_state.pipeline.handle(
            ChatRequest(
                user_id=user_id,
                raw_query=query,
                platform_context={
                    "gmail_username": gmail_username,
                    "gmail_app_password": gmail_app_password,
                },
            ),
            st.session_state.repository
        )
        st.chat_message("assistant").write(response.final_chat_text)
        delivery = response.platform_payload.get("delivery", {})
        if delivery.get("channel") not in (None, "none"):
            st.caption(f"Delivery: {delivery.get('channel')} — {delivery.get('status')}")


if __name__ == "__main__":
    main()
