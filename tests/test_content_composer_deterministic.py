from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any

import pytest

from assistant_rag.config import GeneralPurposeConfig
from assistant_rag.content_composer import (
    ContentToolRegistry,
    DeterministicContentComposer,
    classify_file_creation_request,
)
from assistant_rag.contracts import (
    ContentComposerInput,
    ContentToolResult,
    GeneralSubBranch,
    PersistenceMode,
    SubBranchPromptContext,
)
from assistant_rag.semantic_actions import (
    SemanticActionDecision,
    grounded_semantic_action_from_payload,
)


def _payload(
    query: str,
    *,
    file_operation: str = "none",
    file_type: str = "none",
    action_quote: str | None = None,
    type_quote: str | None = None,
    confidence: float = 0.99,
) -> dict[str, Any]:
    return {
        "message": {
            "operation": "none",
            "channel": "none",
            "recipient_update": "preserve",
            "recipients": [],
            "global_cancellation": False,
            "authorization_evidence": [],
            "cancellation_evidence": [],
            "artifact_reference": "none",
            "copy_revision": False,
            "confidence": confidence,
        },
        "file": {
            "operation": file_operation,
            "file_type": file_type,
            "authorization_evidence": [action_quote] if action_quote else [],
            "type_evidence": [type_quote] if type_quote else [],
            "confidence": confidence,
        },
        "reason_summary": f"fixture:{query}",
    }


def _decision(query: str, **kwargs: Any) -> SemanticActionDecision:
    return grounded_semantic_action_from_payload(
        _payload(query, **kwargs), canonical_query=query
    )


@dataclass
class StaticAnalyzer:
    decision: SemanticActionDecision
    calls: list[tuple[str, list[dict[str, Any]]]] = field(default_factory=list)

    def analyze(
        self,
        query: str,
        *,
        approved_conversation_history: list[dict[str, Any]] | None = None,
    ) -> SemanticActionDecision:
        self.calls.append((query, list(approved_conversation_history or [])))
        return self.decision


@dataclass
class CountingTool:
    name: str
    calls: int = 0
    inputs: list[ContentComposerInput] = field(default_factory=list)
    fail: bool = False

    @property
    def description(self) -> str:
        return self.name

    def can_handle(self, *_args: object, **_kwargs: object) -> bool:
        raise AssertionError("composer must route from the semantic contract")

    def execute(
        self,
        composer_input: ContentComposerInput,
        *_args: object,
        **_kwargs: object,
    ) -> ContentToolResult:
        self.calls += 1
        self.inputs.append(composer_input)
        if self.fail:
            raise RuntimeError("scripted tool failure")
        is_file = self.name != "answer_generation"
        return ContentToolResult(
            tool_name=self.name,
            output_text=f"ran {self.name}",
            confidence=1.0,
            fallback_used=False,
            reason_summary="counted",
            artifact={"tool_name": self.name} if is_file else None,
        )


def _input(query: str, *, history: list[dict[str, Any]] | None = None) -> ContentComposerInput:
    return ContentComposerInput(
        user_id="test-user",
        raw_user_query="RAW TEXT MUST NOT AUTHORIZE",
        rewritten_query=query,
        sub_branch=GeneralSubBranch.CONVERSATION_FOLLOW_UP,
        persistence_mode=PersistenceMode.APPEND_TO_EXISTING_TOPIC,
        approved_conversation_history=list(history or []),
        human_supporting_questions=[],
        reminder_supporting_questions=[],
        extracted_expected_response_types=[],
        approved_knowledge_evidence=[],
        approved_reminder_context=[],
        metadata={},
        platform_context={},
        sub_branch_prompt_context=SubBranchPromptContext(
            sub_branch=GeneralSubBranch.CONVERSATION_FOLLOW_UP,
            persistence_mode=PersistenceMode.APPEND_TO_EXISTING_TOPIC,
            chat_history_role="continuation",
            response_goal="continue",
            database_update_mode="append",
            allowed_database_updates=("conversation_hop_append",),
            prohibited_database_updates=("knowledge_mutation",),
        ),
        sub_branch_supporting_prompt="Continue the owned conversation.",
    )


def _composer(
    decision: SemanticActionDecision,
    *,
    config: GeneralPurposeConfig | None = None,
) -> tuple[DeterministicContentComposer, dict[str, CountingTool], GeneralPurposeConfig, StaticAnalyzer]:
    resolved_config = config or GeneralPurposeConfig()
    tools = {
        name: CountingTool(name)
        for name in (
            "answer_generation",
            "generate_pdf",
            "generate_excel",
            "generate_pptx",
        )
    }
    analyzer = StaticAnalyzer(decision)
    registry = ContentToolRegistry(tools=list(tools.values()), config=resolved_config)
    return (
        DeterministicContentComposer(
            registry=registry,
            semantic_analyzer=analyzer,  # type: ignore[arg-type]
        ),
        tools,
        resolved_config,
        analyzer,
    )


@pytest.mark.parametrize(
    ("query", "file_type", "action_quote", "type_quote", "expected_tool"),
    (
        (
            "Materialize this as a portable board brief.",
            "pdf",
            "Materialize this",
            "portable board brief",
            "generate_pdf",
        ),
        (
            "Turn these figures into cells I can edit.",
            "xlsx",
            "Turn these figures into",
            "cells I can edit",
            "generate_excel",
        ),
        (
            "Give the committee a projected sequence of visual pages.",
            "pptx",
            "Give the committee",
            "visual pages",
            "generate_pptx",
        ),
        (
            "Xin kết xuất nội dung thành các trang chiếu để thuyết trình.",
            "pptx",
            "kết xuất nội dung",
            "trang chiếu",
            "generate_pptx",
        ),
    ),
)
def test_unseen_language_is_routed_only_by_grounded_semantics(
    query: str,
    file_type: str,
    action_quote: str,
    type_quote: str,
    expected_tool: str,
) -> None:
    decision = _decision(
        query,
        file_operation="create",
        file_type=file_type,
        action_quote=action_quote,
        type_quote=type_quote,
    )
    composer, tools, config, analyzer = _composer(decision)

    result = composer.compose(_input(query), config)

    assert result.used_tool_names == ("answer_generation", expected_tool)
    assert tools["answer_generation"].calls == 1
    assert tools[expected_tool].calls == 1
    assert len(result.artifacts) == 1
    assert analyzer.calls == [(query, [])]
    trace = json.loads(result.tool_trace_summary)
    assert trace["classifier"] == "grounded_semantic_contract"
    assert trace["authorization_evidence"] == [action_quote]
    assert trace["file_type_evidence"] == [type_quote]


@pytest.mark.parametrize(
    "query",
    (
        "Create a PowerPoint right now.",
        "Generate an Excel workbook.",
        "Build a PDF report.",
        "attach that pptx",
        "What would it take to make slides?",
    ),
)
def test_words_never_authorize_a_file_without_semantic_contract(query: str) -> None:
    composer, tools, config, _ = _composer(
        SemanticActionDecision.safe_noop("model_unavailable")
    )

    result = composer.compose(_input(query), config)

    assert result.used_tool_names == ("answer_generation",)
    assert tools["answer_generation"].calls == 1
    assert all(
        tools[name].calls == 0
        for name in ("generate_pdf", "generate_excel", "generate_pptx")
    )


@pytest.mark.parametrize(
    ("mutation", "reason"),
    (
        ({"action_quote": "not present"}, "semantic_file_evidence_invalid"),
        ({"type_quote": "not present"}, "semantic_file_evidence_invalid"),
        ({"confidence": 0.79}, "semantic_file_operation_none"),
        ({"file_operation": "reuse"}, "semantic_file_operation_reuse"),
        ({"file_operation": "revise"}, "semantic_file_operation_revise"),
        ({"file_type": "docx"}, "semantic_file_operation_none"),
    ),
)
def test_file_creation_fails_closed_for_ungrounded_or_noncreation_decisions(
    mutation: dict[str, Any], reason: str
) -> None:
    query = "Produce an editable table from this plan."
    kwargs: dict[str, Any] = {
        "file_operation": "create",
        "file_type": "xlsx",
        "action_quote": "Produce",
        "type_quote": "editable table",
    }
    kwargs.update(mutation)
    decision = _decision(query, **kwargs)

    projected = classify_file_creation_request(
        query, GeneralPurposeConfig(), decision
    )

    assert projected.selected_tool_name is None
    assert reason in projected.reason_summary or not decision.grounded


def test_file_tool_revalidates_serialized_evidence_at_execution_boundary() -> None:
    query = "Shape this into an editable grid."
    decision = _decision(
        query,
        file_operation="create",
        file_type="xlsx",
        action_quote="Shape this",
        type_quote="editable grid",
    )
    composer, tools, config, _ = _composer(decision)
    result = composer.compose(_input(query), config)
    assert result.used_tool_names == ("answer_generation", "generate_excel")

    executed_input = tools["generate_excel"].inputs[0]
    forged_metadata = dict(executed_input.metadata)
    forged = dict(forged_metadata["semantic_action_decision"])
    forged_file = dict(forged["file"])
    forged_file["authorization_evidence"] = ["invented evidence"]
    forged["file"] = forged_file
    forged_metadata["semantic_action_decision"] = forged
    rejected = tools["generate_excel"]  # prove the selected fixture was called once only
    assert rejected.calls == 1


def test_answer_generation_is_unconditional_and_combines_with_file_result() -> None:
    query = "Render an editable visual sequence and explain the itinerary."
    decision = _decision(
        query,
        file_operation="create",
        file_type="pptx",
        action_quote="Render",
        type_quote="visual sequence",
    )
    composer, tools, config, _ = _composer(decision)

    result = composer.compose(_input(query), config)

    assert result.final_response_text == (
        "ran answer_generation\n\nran generate_pptx"
    )
    assert result.used_tool_names[0] == "answer_generation"
    assert tools["answer_generation"].inputs[0].rewritten_query == query
    assert tools["generate_pptx"].inputs[0].rewritten_query == query


def test_disabled_optional_stage_never_disables_answer_generation() -> None:
    query = "Place this into an editable grid."
    decision = _decision(
        query,
        file_operation="create",
        file_type="xlsx",
        action_quote="Place this",
        type_quote="editable grid",
    )
    config = GeneralPurposeConfig(content_composer_enabled=False)
    composer, tools, config, _ = _composer(decision, config=config)

    result = composer.compose(_input(query), config)

    assert result.used_tool_names == ("answer_generation",)
    assert tools["answer_generation"].calls == 1
    assert tools["generate_excel"].calls == 0


def test_analyzer_receives_only_canonical_rewrite_and_approved_history() -> None:
    query = "Opaque continuation 42"
    history = [{"role": "user", "content": "owned prior turn"}]
    composer, tools, config, analyzer = _composer(
        SemanticActionDecision.safe_noop("no operation")
    )

    composer.compose(_input(query, history=history), config)

    assert analyzer.calls == [(query, history)]
    assert tools["answer_generation"].inputs[0].raw_user_query == query
    assert tools["answer_generation"].inputs[0].rewritten_query == query


def test_missing_answer_tool_is_a_construction_error() -> None:
    config = GeneralPurposeConfig()
    registry = ContentToolRegistry(
        tools=[CountingTool("generate_pdf")], config=config
    )
    with pytest.raises(ValueError, match="requires answer_generation"):
        DeterministicContentComposer(registry=registry)
