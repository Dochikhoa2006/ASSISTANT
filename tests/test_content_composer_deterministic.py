from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from assistant_rag.config import GeneralPurposeConfig
from assistant_rag.content_composer import (
    AnswerGenerationTool,
    ContentToolRegistry,
    DeterministicContentComposer,
    GenerateExcelTool,
    GeneratePDFTool,
    GeneratePPTXTool,
    classify_file_creation_request,
)
from assistant_rag.contracts import (
    ContentComposerInput,
    ContentToolResult,
    GeneralSubBranch,
    PersistenceMode,
    SubBranchPromptContext,
)
from assistant_rag.llm import LLMTask
from assistant_rag.prompts import DEFAULT_PROMPT_REGISTRY


@dataclass
class CountingTool:
    name: str
    calls: int = 0
    inputs: list[ContentComposerInput] = field(default_factory=list)

    @property
    def description(self) -> str:
        return self.name

    def can_handle(self, *_args: object, **_kwargs: object) -> bool:
        raise AssertionError("deterministic composer must not scan tool can_handle methods")

    def execute(
        self,
        composer_input: ContentComposerInput,
        *_args: object,
        **_kwargs: object,
    ) -> ContentToolResult:
        self.calls += 1
        self.inputs.append(composer_input)
        is_file_tool = self.name in {"generate_pdf", "generate_excel", "generate_pptx"}
        return ContentToolResult(
            tool_name=self.name,
            output_text=f"ran {self.name}",
            confidence=1.0,
            fallback_used=False,
            reason_summary="counted",
            artifact={"tool_name": self.name} if is_file_tool else None,
        )


def _composer() -> tuple[DeterministicContentComposer, dict[str, CountingTool], GeneralPurposeConfig]:
    config = GeneralPurposeConfig()
    tools = {
        name: CountingTool(name)
        for name in ("answer_generation", "generate_pdf", "generate_excel", "generate_pptx")
    }
    registry = ContentToolRegistry(tools=list(tools.values()), config=config)
    return DeterministicContentComposer(registry=registry), tools, config


def _input(raw_query: str, rewritten_query: str | None = None) -> ContentComposerInput:
    return ContentComposerInput(
        user_id="test-user",
        raw_user_query=raw_query,
        rewritten_query=rewritten_query if rewritten_query is not None else raw_query,
        sub_branch=GeneralSubBranch.NEW_CONVERSATION_TOPIC,
        persistence_mode=PersistenceMode.CREATE_NEW_TOPIC,
        approved_conversation_history=[],
        human_supporting_questions=[],
        reminder_supporting_questions=[],
        extracted_expected_response_types=[],
        approved_knowledge_evidence=[],
        approved_reminder_context=[],
        metadata={},
        platform_context={},
        sub_branch_prompt_context=SubBranchPromptContext(
            sub_branch=GeneralSubBranch.NEW_CONVERSATION_TOPIC,
            persistence_mode=PersistenceMode.CREATE_NEW_TOPIC,
            chat_history_role="none",
            response_goal="answer directly",
            database_update_mode="create",
            allowed_database_updates=("conversation_hop_append",),
            prohibited_database_updates=("knowledge_mutation",),
        ),
        sub_branch_supporting_prompt="Answer directly.",
    )


def _compose(raw_query: str, rewritten_query: str | None = None):
    composer, tools, config = _composer()
    result = composer.compose(_input(raw_query, rewritten_query), config)
    return result, tools


@pytest.mark.parametrize(
    ("query", "expected_tool"),
    (
        ("Create a formal report for the board.", "generate_pdf"),
        ("Generate an Excel file to track expenses.", "generate_excel"),
        ("Build a PowerPoint file for the launch.", "generate_pptx"),
        ("Make a roadmap presentation for the team.", "generate_pptx"),
        ("Export the data as an .xlsx file.", "generate_excel"),
    ),
)
def test_explicit_single_type_executes_answer_then_exactly_one_file_tool(query: str, expected_tool: str) -> None:
    result, tools = _compose(query)

    assert result.used_tool_names == ("answer_generation", expected_tool)
    assert len(result.artifacts) == 1
    assert sum(tools[name].calls for name in ("generate_pdf", "generate_excel", "generate_pptx")) == 1
    assert tools[expected_tool].calls == 1
    assert tools["answer_generation"].calls == 1
    assert result.final_response_text == f"ran answer_generation\n\nran {expected_tool}"


@pytest.mark.parametrize(
    ("query", "reason"),
    (
        ("What is an Excel workbook?", "missing_verb_keyword"),
        ("Please create something useful.", "missing_file_keyword"),
        ("Create a report and a PowerPoint presentation.", "ambiguous_file_types"),
    ),
)
def test_missing_or_ambiguous_signals_fail_closed_to_general_answer(query: str, reason: str) -> None:
    result, tools = _compose(query)

    assert result.used_tool_names == ("answer_generation",)
    assert not result.artifacts
    assert all(tools[name].calls == 0 for name in ("generate_pdf", "generate_excel", "generate_pptx"))
    assert tools["answer_generation"].calls == 1
    assert reason in result.tool_trace_summary


def test_rewritten_query_cannot_inject_file_creation_signals() -> None:
    result, tools = _compose(
        "Can you help me with quarterly planning?",
        rewritten_query="Create an Excel spreadsheet for quarterly planning.",
    )

    assert result.used_tool_names == ("answer_generation",)
    assert all(tools[name].calls == 0 for name in ("generate_pdf", "generate_excel", "generate_pptx"))


def test_keyword_matching_uses_boundaries_instead_of_substrings() -> None:
    decision = classify_file_creation_request(
        "Explain how information is represented.",
        GeneralPurposeConfig(),
    )

    assert "form" not in decision.matched_verb_keywords
    assert decision.selected_tool_name is None


def test_file_tool_cannot_be_used_as_signal_free_default_or_fallback() -> None:
    config = GeneralPurposeConfig(
        content_composer_default_tool="generate_excel",
        content_composer_fallback_tool="generate_pptx",
    )
    tools = {
        name: CountingTool(name)
        for name in ("answer_generation", "generate_pdf", "generate_excel", "generate_pptx")
    }
    composer = DeterministicContentComposer(
        registry=ContentToolRegistry(tools=list(tools.values()), config=config)
    )

    result = composer.compose(_input("Hello"), config)

    assert result.used_tool_names == ("answer_generation",)
    assert all(tools[name].calls == 0 for name in ("generate_pdf", "generate_excel", "generate_pptx"))


def test_compound_email_and_excel_request_has_disjoint_tool_responsibilities() -> None:
    query = (
        "Write an email to the finance team explaining the Q3 review, and attach an "
        "Excel budget tracker with columns for owner, forecast, and actuals."
    )

    result, tools = _compose(query)

    assert result.used_tool_names == ("answer_generation", "generate_excel")
    answer_input = tools["answer_generation"].inputs[0]
    excel_input = tools["generate_excel"].inputs[0]
    assert answer_input.raw_user_query == "Write an email to the finance team explaining the Q3 review"
    assert answer_input.rewritten_query == answer_input.raw_user_query
    answer_scope = answer_input.metadata["content_composition_scope"]
    assert answer_scope["answer_request_scope"] == answer_input.raw_user_query
    assert answer_scope["assigned_file_tool"] == "generate_excel"
    assert "email" in answer_scope["answer_generation_responsibility"]
    assert "file_request_scope" not in answer_scope
    assert excel_input.raw_user_query == query  # raw input is retained for authorization only
    assert excel_input.rewritten_query == (
        "attach an Excel budget tracker with columns for owner, forecast, and actuals"
    )
    assert "Write an email" not in excel_input.rewritten_query
    assert "finance team" not in excel_input.rewritten_query
    file_scope = excel_input.metadata["content_composition_scope"]
    assert "surrounding email" in file_scope["file_tool_responsibility"]
    assert "answer_request_scope" not in file_scope


def test_file_first_then_email_request_removes_trailing_email_from_file_scope() -> None:
    query = (
        "Create an Excel budget tracker with forecast and actual columns, then write "
        "an email to the finance team summarizing the handoff."
    )

    result, tools = _compose(query)

    assert result.used_tool_names == ("answer_generation", "generate_excel")
    answer_input = tools["answer_generation"].inputs[0]
    excel_input = tools["generate_excel"].inputs[0]
    assert excel_input.rewritten_query == (
        "Create an Excel budget tracker with forecast and actual columns"
    )
    assert answer_input.raw_user_query == (
        "write an email to the finance team summarizing the handoff"
    )
    assert "email" not in excel_input.rewritten_query


def test_attaching_clause_without_conjunction_is_still_isolated() -> None:
    query = (
        "Write an email to operations attaching an Excel tracker with owner and "
        "status columns."
    )

    result, tools = _compose(query)

    assert result.used_tool_names == ("answer_generation", "generate_excel")
    assert tools["answer_generation"].inputs[0].rewritten_query == (
        "Write an email to operations"
    )
    assert tools["generate_excel"].inputs[0].rewritten_query == (
        "an Excel tracker with owner and status columns"
    )


def test_disabling_optional_composer_still_executes_answer_generation() -> None:
    composer, tools, _config = _composer()
    config = GeneralPurposeConfig(content_composer_enabled=False)

    result = composer.compose(_input("Create an Excel expense tracker."), config)

    assert result.used_tool_names == ("answer_generation",)
    assert tools["answer_generation"].calls == 1
    assert all(tools[name].calls == 0 for name in ("generate_pdf", "generate_excel", "generate_pptx"))
    scope = tools["answer_generation"].inputs[0].metadata["content_composition_scope"]
    assert scope["answer_request_scope"] == "Create an Excel expense tracker."
    assert "file_request_scope" not in scope
    assert scope["assigned_file_tool"] is None
    assert "complete user-facing answer" in scope["answer_generation_responsibility"]
    assert '"file_route_status": "disabled"' in result.tool_trace_summary


def test_missing_answer_tool_prevents_file_execution() -> None:
    config = GeneralPurposeConfig()
    excel = CountingTool("generate_excel")
    composer = DeterministicContentComposer(
        registry=ContentToolRegistry(tools=[excel], config=config)
    )

    result = composer.compose(_input("Create an Excel expense tracker."), config)

    assert result.used_tool_names == ()
    assert excel.calls == 0
    assert result.content_warnings == ("answer_tool_unavailable",)


def test_real_tool_prompts_keep_email_copy_out_of_excel_planner() -> None:
    class RecordingLLM:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def chat(self, **kwargs: object) -> str:
            self.calls.append(kwargs)
            return "EMAIL COPY" if kwargs.get("task") is LLMTask.ANSWER else "WORKBOOK PLAN"

    query = (
        "Write an email to finance about the handoff, and attach an Excel tracker "
        "with owner, due date, and status columns."
    )
    config = GeneralPurposeConfig()
    llm = RecordingLLM()
    composer = DeterministicContentComposer(
        registry=ContentToolRegistry(
            tools=[
                AnswerGenerationTool(llm=llm, prompt_registry=DEFAULT_PROMPT_REGISTRY),
                GenerateExcelTool(llm=llm, prompt_registry=DEFAULT_PROMPT_REGISTRY),
            ],
            config=config,
        )
    )

    result = composer.compose(_input(query), config)

    assert result.used_tool_names == ("answer_generation", "generate_excel")
    assert result.final_response_text == "EMAIL COPY\n\nWORKBOOK PLAN"
    assert [call["task"] for call in llm.calls] == [LLMTask.ANSWER, LLMTask.WRITING]
    answer_prompt = str(llm.calls[0]["user_prompt"])
    file_prompt = str(llm.calls[1]["user_prompt"])
    assert "Write an email to finance" in answer_prompt
    assert "answer_generation_responsibility" in answer_prompt
    assert "owner, due date" not in answer_prompt
    assert "Excel tracker" in file_prompt
    assert "file_tool_responsibility" in file_prompt
    assert "Write an email" not in file_prompt
    assert "finance" not in file_prompt


@pytest.mark.parametrize(
    "tool_class",
    (GeneratePDFTool, GenerateExcelTool, GeneratePPTXTool),
)
def test_direct_file_tool_execution_is_rejected_without_authorized_intent(tool_class: type) -> None:
    class ExplodingLLM:
        def chat(self, **_kwargs: object) -> str:
            raise AssertionError("unauthorized direct execution reached the planning LLM")

    tool = tool_class(llm=ExplodingLLM(), prompt_registry=SimpleNamespace())
    result = tool.execute(
        SimpleNamespace(raw_user_query="Tell me what this file type is."),
        GeneralPurposeConfig(),
    )

    assert result.artifact is None
    assert result.reason_summary == "file_tool_rejected:missing_verb_keyword"
    assert result.warnings == ("file_creation_intent_not_authorized",)
