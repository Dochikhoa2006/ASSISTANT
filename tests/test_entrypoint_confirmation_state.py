from __future__ import annotations

from types import SimpleNamespace

import pytest

from assistant_rag.contracts import (
    BundledResponse,
    ChatRequest,
    LastQAState,
    ResponseType,
)
from assistant_rag.request_lifecycle import ChatRequestExecution
import streamlit_app


class SessionState(dict):
    def __getattr__(self, name: str):
        return self[name]

    def __setattr__(self, name: str, value: object) -> None:
        self[name] = value


class ConfirmationUI:
    def __init__(self) -> None:
        self.session_state = SessionState(chat_messages=[])
        self.errors: list[str] = []
        self.rerun_count = 0

    def caption(self, _message: str) -> None:
        pass

    def button(self, *_args: object, **_kwargs: object) -> bool:
        return True

    def error(self, message: str) -> None:
        self.errors.append(message)

    def rerun(self) -> None:
        self.rerun_count += 1


def _execution(response_type: ResponseType) -> ChatRequestExecution:
    text = "Completed." if response_type is not ResponseType.ERROR else "Failed safely."
    response = BundledResponse(
        final_chat_text=text,
        response_type=response_type,
        last_qa_state=LastQAState(
            last_user_query="confirm",
            last_response=text,
            response_type=response_type,
        ),
    )
    request = ChatRequest(user_id="user-1", raw_query="confirm")
    return ChatRequestExecution(
        request=request,
        request_id="request-1",
        response=response,
        payload={},
        replayed=False,
        is_mutation=True,
    )


@pytest.mark.parametrize(
    ("response_type", "pending_count", "notice_key"),
    [
        (ResponseType.KNOWLEDGE_ACTION, 0, "confirmation_notice"),
        (ResponseType.ERROR, 1, "confirmation_error_notice"),
    ],
)
def test_streamlit_confirmation_state_matches_terminal_result(
    monkeypatch: pytest.MonkeyPatch,
    response_type: ResponseType,
    pending_count: int,
    notice_key: str,
) -> None:
    st = ConfirmationUI()
    message = {
        "role": "assistant",
        "content": "Please confirm.",
        "request_query": "Change the stored fact.",
        "pending_confirmations": [
            {
                "confirmation_token": "token-1",
                "action_type": "knowledge_mutation",
            }
        ],
    }
    monkeypatch.setattr(
        streamlit_app,
        "_execute_user_request",
        lambda **_kwargs: _execution(response_type),
    )

    streamlit_app._render_confirmation_controls(
        st,
        message=message,
        pipeline=SimpleNamespace(),
        repository=SimpleNamespace(),
        user_id="user-1",
        gmail_username="",
        gmail_app_password="",
    )

    assert len(message["pending_confirmations"]) == pending_count
    assert notice_key in st.session_state
    other_notice = (
        "confirmation_error_notice"
        if notice_key == "confirmation_notice"
        else "confirmation_notice"
    )
    assert other_notice not in st.session_state
    assert st.rerun_count == 1
    assert st.session_state.chat_messages[-1]["response_type"] == response_type.value

