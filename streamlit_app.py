from __future__ import annotations

from assistant_rag.contracts import ChatRequest
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
        
    user_id = st.text_input("User ID", value="streamlit-user")
    gmail_username = st.text_input("Gmail username", value="")
    # Gmail app password should be loaded securely from secrets or OAuth in production
    query = st.chat_input("Ask the assistant")
    if query:
        response = st.session_state.pipeline.handle(
            ChatRequest(
                user_id=user_id,
                raw_query=query,
                platform_context={
                    "gmail_username": gmail_username,
                },
            ),
            st.session_state.repository
        )
        st.chat_message("assistant").write(response.final_chat_text)


if __name__ == "__main__":
    main()

