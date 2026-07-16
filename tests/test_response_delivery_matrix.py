from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import pytest

import assistant_rag.branches as branches_module
from assistant_rag.branches import GeneralResponseBranch
from assistant_rag.bundler import ResponseBundler
from assistant_rag.config import GeneralPurposeConfig
from assistant_rag.content_composer import ContentToolRegistry, DeterministicContentComposer
from assistant_rag.context_filter import ApprovedContext
from assistant_rag.contracts import (
    ApprovedConversationContext,
    BranchResult,
    BundledResponse,
    ChatRequest,
    ContentComposerInput,
    ContentComposerResult,
    ContentToolResult,
    ExpectedResponseType,
    GeneralSubBranch,
    GeneralSubBranchDecision,
    GeneratedQuestion,
    HopWrite,
    HumanSupportingDecision,
    Intent,
    LastQAState,
    PersistenceMode,
    PipelineContext,
    QuestionSource,
    RepositoryActionResult,
    ResponseType,
    SubBranchPromptContext,
)
from assistant_rag.platform import PlatformSelector


_FILE_TOOLS = ("generate_pdf", "generate_excel", "generate_pptx")
_RAW_QUERY_SENTINEL = "RAW_SENTINEL audit-only ingress text."


@dataclass
class RecordingContentTool:
    name: str
    calls: list[ContentComposerInput] = field(default_factory=list)

    @property
    def description(self) -> str:
        return self.name

    def can_handle(self, *_args: Any, **_kwargs: Any) -> bool:
        raise AssertionError("The deterministic composer must not scan can_handle().")

    def execute(
        self,
        composer_input: ContentComposerInput,
        _config: GeneralPurposeConfig,
    ) -> ContentToolResult:
        self.calls.append(composer_input)
        return ContentToolResult(
            tool_name=self.name,
            output_text=f"output:{self.name}",
            confidence=1.0,
            fallback_used=False,
            reason_summary=f"executed:{self.name}",
            artifact={"tool_name": self.name} if self.name in _FILE_TOOLS else None,
        )


def _composer_input(query: str, sub_branch: GeneralSubBranch) -> ContentComposerInput:
    persistence_mode = (
        PersistenceMode.CREATE_NEW_TOPIC
        if sub_branch is GeneralSubBranch.NEW_CONVERSATION_TOPIC
        else PersistenceMode.APPEND_TO_EXISTING_TOPIC
    )
    return ContentComposerInput(
        user_id="matrix-user",
        raw_user_query=query,
        rewritten_query=query,
        sub_branch=sub_branch,
        persistence_mode=persistence_mode,
        approved_conversation_history=[{"role": "user", "content": "prior"}],
        human_supporting_questions=[],
        reminder_supporting_questions=[],
        extracted_expected_response_types=[],
        approved_knowledge_evidence=[],
        approved_reminder_context=[],
        metadata={},
        platform_context={},
        sub_branch_prompt_context=SubBranchPromptContext(
            sub_branch=sub_branch,
            persistence_mode=persistence_mode,
            chat_history_role="matrix history role",
            response_goal="matrix response goal",
            database_update_mode="matrix persistence",
            allowed_database_updates=("conversation_hop_append",),
            prohibited_database_updates=("knowledge_mutation",),
        ),
        sub_branch_supporting_prompt=f"Policy for {sub_branch.value}",
    )


@pytest.mark.parametrize("sub_branch", tuple(GeneralSubBranch))
@pytest.mark.parametrize(
    ("outcome", "query", "composer_enabled", "expected_file_tool"),
    (
        ("none", "Explain the quarterly review.", True, None),
        ("pdf", "Create a PDF report for the quarterly review.", True, "generate_pdf"),
        ("xlsx", "Generate an Excel workbook for the quarterly review.", True, "generate_excel"),
        ("pptx", "Build a PowerPoint presentation for the quarterly review.", True, "generate_pptx"),
        (
            "ambiguous",
            "Create an Excel workbook and a PowerPoint presentation.",
            True,
            None,
        ),
        ("disabled", "Create an Excel workbook for the quarterly review.", False, None),
    ),
)
def test_content_composer_sub_branch_and_file_outcome_cross_product(
    sub_branch: GeneralSubBranch,
    outcome: str,
    query: str,
    composer_enabled: bool,
    expected_file_tool: str | None,
) -> None:
    config = GeneralPurposeConfig(content_composer_enabled=composer_enabled)
    tools = {
        name: RecordingContentTool(name)
        for name in ("answer_generation", *_FILE_TOOLS)
    }
    composer = DeterministicContentComposer(
        registry=ContentToolRegistry(tools=list(tools.values()), config=config)
    )

    result = composer.compose(_composer_input(query, sub_branch), config)

    expected_tools = (
        ("answer_generation", expected_file_tool)
        if expected_file_tool is not None
        else ("answer_generation",)
    )
    assert result.used_tool_names == expected_tools
    assert len(tools["answer_generation"].calls) == 1
    assert sum(len(tools[name].calls) for name in _FILE_TOOLS) <= 1
    assert all(
        item.sub_branch is sub_branch
        for tool in tools.values()
        for item in tool.calls
    )
    assert len(result.artifacts) == (1 if expected_file_tool else 0)
    trace = json.loads(result.tool_trace_summary)
    assert trace["executed_tools"] == list(expected_tools)
    if outcome == "ambiguous":
        assert trace["decision"] == "ambiguous_file_types"
        assert trace["selected_file_tool"] is None
    elif outcome == "disabled":
        assert trace["file_route_status"] == "disabled"
        assert trace["selected_file_tool"] is None


@dataclass
class RecordingComposer:
    inputs: list[ContentComposerInput] = field(default_factory=list)

    def compose(
        self,
        composer_input: ContentComposerInput,
        _config: GeneralPurposeConfig,
    ) -> ContentComposerResult:
        self.inputs.append(composer_input)
        return ContentComposerResult(
            final_response_text=f"answer:{composer_input.sub_branch.value}",
            tool_trace_summary="matrix composer",
            used_tool_names=("answer_generation",),
            confidence=1.0,
            fallback_used=False,
            reason_summary="matrix answer",
            content_warnings=(),
        )


@dataclass
class FixedSubBranchDetector:
    decision: GeneralSubBranchDecision
    calls: int = 0

    def detect(self, *_args: Any, **_kwargs: Any) -> GeneralSubBranchDecision:
        self.calls += 1
        return self.decision


@dataclass
class RecordingGeneralHITL:
    calls: int = 0

    def evaluate(self, **_kwargs: Any) -> HumanSupportingDecision:
        self.calls += 1
        return HumanSupportingDecision(
            should_ask=True,
            question="Would a worked example help?",
            confidence=0.95,
            question_source=QuestionSource.HUMAN_SUPPORTING_QUESTION,
            expected_response_type=ExpectedResponseType.YES_NO_ANSWER,
            reason_summary="matrix HITL",
            risk_flags=(),
        )


class EmptyContextFilter:
    def filter(self, **_kwargs: Any) -> ApprovedContext:
        return ApprovedContext(
            knowledge_evidence=[],
            reminder_context=[],
            approved_conversation_history=[],
            rejected_knowledge_ids=[],
            rejected_reminder_ids=[],
            rejected_conversation_ids=[],
        )


@dataclass
class RecordingRepository:
    ensure_topic_calls: int = 0
    append_calls: list[dict[str, Any]] = field(default_factory=list)

    @contextmanager
    def transaction(self) -> Iterator[object]:
        yield object()

    def ensure_topic(self, _cursor: object, *, user_id: str, title: str) -> str:
        assert user_id == "matrix-user"
        assert title == "General Conversation"
        self.ensure_topic_calls += 1
        return "topic-created"

    def append_conversation_hop(self, _cursor: object, **kwargs: Any) -> HopWrite:
        self.append_calls.append(dict(kwargs))
        return HopWrite(
            topic_id=str(kwargs["topic_id"]),
            hop_id=f"written-{len(self.append_calls)}",
            previous_hop_id=kwargs.get("parent_hop_id"),
            outbox_job_id=f"outbox-{len(self.append_calls)}",
        )


def _approved_conversation_context(
    *,
    history: list[dict[str, Any]],
    topic_id: str | None,
    hop_id: str | None,
) -> ApprovedConversationContext:
    return ApprovedConversationContext(
        approved_conversation_history=history,
        human_supporting_questions=[],
        reminder_supporting_questions=[],
        clarification_question_context=None,
        extracted_expected_response_types=[],
        conversation_retrieval_ran=bool(history),
        conversation_context_status="approved" if history else "empty",
        approved_conversation_count=len(history),
        _internal_selected_topic_candidates=[topic_id] if topic_id else [],
        _internal_selected_hop_candidates=[hop_id] if hop_id else [],
    )


def _branch_case(
    sub_branch: GeneralSubBranch,
) -> tuple[PipelineContext, GeneralSubBranchDecision, str, str | None, int]:
    if sub_branch is GeneralSubBranch.SUPPORT_QUESTION_ANSWER:
        topic_id, hop_id = "topic-support", "hop-support"
        history = [{"role": "assistant", "content": "prior answer"}]
        last_qa = LastQAState(
            last_user_query="prior",
            last_response="prior answer",
            response_type=ResponseType.NORMAL,
            supporting_questions=[
                GeneratedQuestion(
                    text="Which audience?",
                    source=QuestionSource.HUMAN_SUPPORTING_QUESTION,
                    purpose="optional_context",
                    confidence=1.0,
                )
            ],
            linked_topic_id=topic_id,
            linked_hop_id=hop_id,
        )
        decision = GeneralSubBranchDecision(
            sub_branch=sub_branch,
            confidence=1.0,
            persistence_mode=PersistenceMode.APPEND_TO_EXISTING_TOPIC,
            selected_topic_id=topic_id,
            selected_hop_id=hop_id,
            selected_parent_hop_id=hop_id,
            reason_summary="matrix support rule",
        )
        expected_ensure_calls = 0
    elif sub_branch is GeneralSubBranch.CONVERSATION_FOLLOW_UP:
        topic_id, hop_id = "topic-follow", "hop-follow"
        history = [{"role": "user", "content": "continue the prior topic"}]
        last_qa = None
        decision = GeneralSubBranchDecision(
            sub_branch=sub_branch,
            confidence=1.0,
            persistence_mode=PersistenceMode.APPEND_TO_EXISTING_TOPIC,
            selected_topic_id=topic_id,
            selected_hop_id=hop_id,
            selected_parent_hop_id=hop_id,
            reason_summary="matrix follow-up rule",
        )
        expected_ensure_calls = 0
    else:
        topic_id, hop_id = "topic-created", None
        history = []
        last_qa = None
        decision = GeneralSubBranchDecision(
            sub_branch=sub_branch,
            confidence=1.0,
            persistence_mode=PersistenceMode.CREATE_NEW_TOPIC,
            reason_summary="matrix new-topic rule",
        )
        expected_ensure_calls = 1

    context = PipelineContext(
        request=ChatRequest(user_id="matrix-user", raw_query="Matrix branch request"),
        rewritten_query="Matrix branch request",
        last_qa_state=last_qa,
        conversation_results=[],
        intent=Intent.GENERAL_RESPONSE,
        chat_history=history,
        conversation_retrieval=bool(history),
        chat_history_source="conversation_retrieval" if history else "last_qa",
        approved_conversation_context=_approved_conversation_context(
            history=history,
            topic_id=topic_id,
            hop_id=hop_id,
        ),
    )
    return context, decision, topic_id, hop_id, expected_ensure_calls


@pytest.mark.parametrize("sub_branch", tuple(GeneralSubBranch))
@pytest.mark.parametrize("hitl_enabled", (False, True))
def test_general_response_branch_sub_branch_and_hitl_cross_product(
    monkeypatch: pytest.MonkeyPatch,
    sub_branch: GeneralSubBranch,
    hitl_enabled: bool,
) -> None:
    monkeypatch.setattr(branches_module, "retrieve_knowledge", lambda **_kwargs: [])
    monkeypatch.setattr(branches_module, "retrieve_reminder_candidates", lambda **_kwargs: [])
    context, decision, expected_topic, expected_parent, expected_ensure_calls = _branch_case(
        sub_branch
    )
    detector = FixedSubBranchDetector(decision)
    composer = RecordingComposer()
    hitl = RecordingGeneralHITL()
    repository = RecordingRepository()
    general_config = GeneralPurposeConfig(
        hitl_supporting_question_enabled=hitl_enabled,
    )
    branch = GeneralResponseBranch(
        retriever=object(),  # retrieval adapters are replaced at their local boundaries
        config=SimpleNamespace(
            retrieval=SimpleNamespace(
                general_response_reminder_statuses=("scheduled", "notified"),
                general_response_reminder_limit=4,
            )
        ),
        context_filter=EmptyContextFilter(),
        sub_branch_detector=detector,
        content_composer=composer,
        general_hitl_strategy=hitl if hitl_enabled else None,
        general_purpose_config=general_config,
    )

    result = branch.execute(context, repository)  # type: ignore[arg-type]

    assert result.response_type is ResponseType.NORMAL
    assert result.normal_response_text == f"answer:{sub_branch.value}"
    assert result.linked_topic_id == expected_topic
    assert len(composer.inputs) == 1
    composed_input = composer.inputs[0]
    assert composed_input.sub_branch is sub_branch
    assert composed_input.approved_conversation_history == context.chat_history
    assert (
        composed_input.sub_branch_prompt_context.chat_history_role
        in composed_input.sub_branch_supporting_prompt
    )
    assert repository.ensure_topic_calls == expected_ensure_calls
    assert len(repository.append_calls) == 1
    append_call = repository.append_calls[0]
    assert append_call["topic_id"] == expected_topic
    assert append_call["parent_hop_id"] == expected_parent
    assert append_call["entities"]["sub_branch"] == sub_branch.value
    assert append_call["entities"]["used_tools"] == ["answer_generation"]
    assert hitl.calls == (1 if hitl_enabled else 0)
    assert len(result.human_supporting_questions) == (1 if hitl_enabled else 0)
    assert result.human_in_the_loop_result == (
        {"triggered": True, "confidence": 0.95, "question_count": 1}
        if hitl_enabled
        else {
            "triggered": False,
            "confidence": 1.0,
            "question_count": 0,
            "reason": "disabled_by_general_purpose_config",
        }
    )


def test_general_branch_keeps_created_artifact_visible_when_hop_write_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(branches_module, "retrieve_knowledge", lambda **_kwargs: [])
    monkeypatch.setattr(
        branches_module,
        "retrieve_reminder_candidates",
        lambda **_kwargs: [],
    )
    context, decision, *_ = _branch_case(GeneralSubBranch.NEW_CONVERSATION_TOPIC)
    artifact_path = tmp_path / "transaction-safe-workbook.xlsx"
    artifact_path.write_bytes(b"generated-before-hop-write")
    artifact = {
        "artifact_id": "artifact-before-hop-failure",
        "filename": artifact_path.name,
        "storage_path": str(artifact_path),
        "file_type": "xlsx",
        "status": "created",
    }

    class ArtifactComposer:
        def compose(self, *_args: Any, **_kwargs: Any) -> ContentComposerResult:
            return ContentComposerResult(
                final_response_text="The workbook is ready.",
                tool_trace_summary="artifact before failed hop",
                used_tool_names=("answer_generation", "generate_excel"),
                confidence=1.0,
                fallback_used=False,
                reason_summary="artifact created",
                content_warnings=(),
                artifacts=(artifact,),
            )

    class FailingRepository:
        @contextmanager
        def transaction(self) -> Iterator[object]:
            raise RuntimeError("forced hop failure")
            yield object()  # pragma: no cover

    branch = GeneralResponseBranch(
        retriever=object(),
        config=SimpleNamespace(
            retrieval=SimpleNamespace(
                general_response_reminder_statuses=("scheduled", "notified"),
                general_response_reminder_limit=4,
            )
        ),
        context_filter=EmptyContextFilter(),
        sub_branch_detector=FixedSubBranchDetector(decision),
        content_composer=ArtifactComposer(),
        general_hitl_strategy=None,
        general_purpose_config=GeneralPurposeConfig(),
    )

    result = branch.execute(context, FailingRepository())  # type: ignore[arg-type]

    assert result.response_type is ResponseType.ERROR
    assert result.platform_payload == {"artifacts": [artifact]}
    assert artifact_path.read_bytes() == b"generated-before-hop-write"


@pytest.mark.parametrize("response_type", tuple(ResponseType))
def test_response_bundler_all_response_type_shapes(response_type: ResponseType) -> None:
    request = ChatRequest(
        user_id="matrix-user",
        raw_query="raw request",
        metadata={"warnings": ["request warning"]},
    )
    kwargs: dict[str, Any] = {
        "response_type": response_type,
        "linked_topic_id": "topic-1",
        "linked_hop_id": "hop-1",
        "platform_payload": {"seed": "preserved"},
        "warnings": ["branch warning"],
        "database_write_result": {"conversation_hop_id": "audit-hop"},
        "indexing_job_result": {"conversation_hop_job_id": "outbox-1"},
    }
    if response_type is ResponseType.CLARIFICATION:
        kwargs["clarification_question"] = GeneratedQuestion(
            text="Which record?",
            source=QuestionSource.CLARIFICATION_QUESTION,
            purpose="resolve_missing_info",
            confidence=1.0,
        )
        expected_text = "Clarification question: Which record?"
    elif response_type is ResponseType.KNOWLEDGE_ACTION:
        kwargs["knowledge_operation_results"] = [
            RepositoryActionResult(
                action_id="knowledge-action",
                action_type="modify",
                status="committed",
                domain_entity_type="knowledge_chunk",
                domain_entity_id="knowledge-1",
                user_safe_summary="Knowledge updated.",
            )
        ]
        expected_text = "Knowledge updated."
    elif response_type is ResponseType.REMINDER_ACTION:
        kwargs["reminder_operation_results"] = [
            RepositoryActionResult(
                action_id="reminder-action",
                action_type="turn_off",
                status="committed",
                domain_entity_type="reminder",
                domain_entity_id="reminder-1",
                user_safe_summary="Reminder turned off.",
            )
        ]
        expected_text = "Reminder turned off."
    elif response_type is ResponseType.ERROR:
        kwargs["fallback_or_error_message"] = "The operation failed safely."
        expected_text = "The operation failed safely."
    else:
        expected_text = {
            ResponseType.NORMAL: "Normal answer.",
            ResponseType.REMINDER_REPLY: "Reminder reply recorded.",
            ResponseType.SAFE_NOOP: "No safe action was taken.",
        }[response_type]
        kwargs["normal_response_text"] = expected_text

    bundled = ResponseBundler().bundle(
        request=request,
        rewritten_query="rewritten request",
        branch_result=BranchResult(**kwargs),
    )

    assert bundled.response_type is response_type
    assert bundled.final_chat_text == expected_text
    assert bundled.last_qa_state.response_type is response_type
    assert bundled.last_qa_state.last_user_query == "rewritten request"
    assert bundled.last_qa_state.last_response == expected_text
    assert bundled.platform_payload == {"seed": "preserved"}
    assert bundled.conversation_topic_id == "topic-1"
    assert bundled.conversation_hop_id == "hop-1"
    assert bundled.warnings == ["request warning", "branch warning"]
    assert bundled.persistence_instructions["audit_hop_id"] == "audit-hop"
    expected_committed_count = int(
        response_type in {ResponseType.KNOWLEDGE_ACTION, ResponseType.REMINDER_ACTION}
    )
    assert len(bundled.actions_committed) == expected_committed_count


class ScriptedPlatformLLM:
    def __init__(self, channel: str, extraction: dict[str, Any]) -> None:
        self.channel = channel
        self.extraction = extraction

    def generate_json(self, **kwargs: Any) -> dict[str, Any]:
        if "platform selector" in str(kwargs["system_prompt"]):
            return {"channel": self.channel, "confidence": 1.0}
        return dict(self.extraction)


@dataclass
class ScriptedSender:
    send_result: dict[str, Any] = field(
        default_factory=lambda: {"status": "sent", "provider": "mock"}
    )
    raise_on_send: bool = False
    send_calls: list[tuple[dict[str, Any], dict[str, Any]]] = field(default_factory=list)
    draft_calls: list[tuple[dict[str, Any], dict[str, Any]]] = field(default_factory=list)

    def send(self, payload: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        self.send_calls.append((dict(payload), dict(context)))
        if self.raise_on_send:
            raise RuntimeError("mock delivery failure")
        return dict(self.send_result)

    def create_draft(
        self,
        payload: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        self.draft_calls.append((dict(payload), dict(context)))
        return {"status": "draft_saved", "provider": "gmail"}


def _bundled_platform_response(
    rewritten_query: str = "request",
) -> BundledResponse:
    return BundledResponse(
        final_chat_text="Subject: Project update\n\nHello team,\n\nThe update is ready.",
        response_type=ResponseType.NORMAL,
        last_qa_state=LastQAState(
            last_user_query=rewritten_query,
            last_response="response",
            response_type=ResponseType.NORMAL,
        ),
        platform_payload={"seed": "preserved"},
    )


def _platform_extraction(
    *,
    recipient: str = "alex@example.com",
    mode: str = "send",
) -> dict[str, Any]:
    return {
        "recipients": [recipient] if recipient else [],
        "subject": "Project update",
        "body": "Hello team,\n\nThe update is ready.",
        "mode": mode,
    }


def test_platform_none_route_is_explicit_safe_passthrough() -> None:
    rewritten_query = "REWRITTEN_SENTINEL Explain the project update."
    result = PlatformSelector(llm=None).select(
        _bundled_platform_response(rewritten_query),
        ChatRequest(user_id="matrix-user", raw_query=_RAW_QUERY_SENTINEL),
    )

    assert result["platform_selection"] == {
        "channel": "none",
        "confidence": 0.0,
        "source": "safe_fallback_no_llm",
    }
    assert result["delivery"] == {"channel": "none", "status": "not_requested"}
    assert result["text"].startswith("Subject: Project update")
    assert result["seed"] == "preserved"


def test_platform_formatter_receives_rewritten_compatibility_request() -> None:
    rewritten_query = (
        "REWRITTEN_SENTINEL Draft an email to alex@example.com about the update."
    )
    observed_requests: list[ChatRequest] = []

    class RecordingFormatter:
        def format(
            self, _response: BundledResponse, request: ChatRequest
        ) -> dict[str, Any]:
            observed_requests.append(request)
            return {}

    selector = PlatformSelector(llm=None)
    selector.register("gmail", RecordingFormatter())
    selector.select(
        _bundled_platform_response(rewritten_query),
        ChatRequest(user_id="matrix-user", raw_query=_RAW_QUERY_SENTINEL),
    )

    assert len(observed_requests) == 1
    assert observed_requests[0].raw_query == rewritten_query
    assert _RAW_QUERY_SENTINEL not in observed_requests[0].raw_query


@pytest.mark.parametrize(
    ("case", "query", "context", "sender", "expected_status"),
    (
        (
            "local_draft",
            "Draft an email to alex@example.com about the project update.",
            {},
            ScriptedSender(),
            "draft_ready",
        ),
        (
            "saved_draft",
            "Save a Gmail draft and write an email to alex@example.com.",
            {"gmail_username": "sender@example.com", "gmail_app_password": "secret"},
            ScriptedSender(),
            "draft_saved",
        ),
        (
            "send",
            "Send an email to alex@example.com about the project update.",
            {"gmail_username": "sender@example.com", "gmail_app_password": "secret"},
            ScriptedSender(send_result={"status": "sent", "provider": "gmail"}),
            "sent",
        ),
        (
            "needs_input",
            "Send an email to alex@example.com about the project update.",
            {},
            ScriptedSender(),
            "needs_input",
        ),
        (
            "failure",
            "Send an email to alex@example.com about the project update.",
            {"gmail_username": "sender@example.com", "gmail_app_password": "secret"},
            ScriptedSender(raise_on_send=True),
            "failed",
        ),
        (
            "partial_failure",
            "Send an email to alex@example.com and pat@example.com about the update.",
            {"gmail_username": "sender@example.com", "gmail_app_password": "secret"},
            ScriptedSender(
                send_result={
                    "status": "partial_failure",
                    "provider": "gmail",
                    "delivered_recipients": ["alex@example.com"],
                    "refused_recipients": ["pat@example.com"],
                }
            ),
            "partial_failure",
        ),
    ),
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_platform_gmail_status_matrix(
    case: str,
    query: str,
    context: dict[str, str],
    sender: ScriptedSender,
    expected_status: str,
) -> None:
    recipients = (
        ["alex@example.com", "pat@example.com"]
        if case == "partial_failure"
        else ["alex@example.com"]
    )
    llm = ScriptedPlatformLLM(
        "gmail",
        {
            **_platform_extraction(mode="draft" if "draft" in case else "send"),
            "recipients": recipients,
        },
    )
    result = PlatformSelector(llm=llm, senders={"gmail": sender}).select(
        _bundled_platform_response(query),
        ChatRequest(
            user_id="matrix-user",
            raw_query=_RAW_QUERY_SENTINEL,
            platform_context=context,
        ),
    )

    assert result["platform_selection"]["channel"] == "gmail"
    assert result["delivery"]["channel"] == "gmail"
    assert result["delivery"]["status"] == expected_status
    assert result["seed"] == "preserved"
    assert "secret" not in json.dumps(result)
    if case == "local_draft":
        assert sender.send_calls == []
        assert sender.draft_calls == []
    elif case == "saved_draft":
        assert sender.send_calls == []
        assert len(sender.draft_calls) == 1
    elif case == "needs_input":
        assert sender.send_calls == []
        assert "Gmail username" in result["delivery"]["question"]
        assert "Gmail app password" in result["delivery"]["question"]
    else:
        assert len(sender.send_calls) == 1
    if case in {"failure", "partial_failure"}:
        assert result["delivery"]["question"]
    if case == "partial_failure":
        assert result["delivery"]["delivered_recipients"] == ["alex@example.com"]
        assert result["delivery"]["refused_recipients"] == ["pat@example.com"]


@pytest.mark.parametrize(
    ("channel", "recipient", "platform_context"),
    (
        ("telegram", "-100123456", {"telegram_bot_token": "telegram-secret"}),
        (
            "zalo",
            "84987654321",
            {"zalo_access_token": "zalo-secret", "zalo_api_url": "https://zalo.test"},
        ),
    ),
)
def test_platform_non_email_channel_send_matrix(
    channel: str,
    recipient: str,
    platform_context: dict[str, str],
) -> None:
    sender = ScriptedSender()
    rewritten_query = f"Send this update through {channel} to {recipient}."
    selector = PlatformSelector(
        llm=ScriptedPlatformLLM(
            channel,
            _platform_extraction(recipient=recipient, mode="send"),
        ),
        senders={channel: sender},
    )

    result = selector.select(
        _bundled_platform_response(rewritten_query),
        ChatRequest(
            user_id="matrix-user",
            raw_query=_RAW_QUERY_SENTINEL,
            platform_context=platform_context,
        ),
    )

    assert result["platform_selection"]["channel"] == channel
    assert result["delivery"] == {
        "channel": channel,
        "status": "sent",
        "recipient": recipient,
        "recipients": [recipient],
        "provider": channel,
    }
    assert len(sender.send_calls) == 1
    sent_payload, sent_context = sender.send_calls[0]
    assert sent_payload["recipient"] == recipient
    assert sent_context == platform_context
    assert not any(secret in json.dumps(result) for secret in platform_context.values())
