"""Content composer and tool abstraction for the General Response branch."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from .artifacts import ArtifactGenerator
from .config import GeneralPurposeConfig
from .contracts import ContentComposerResult, ContentToolResult, ContentComposerInput
from .llm import LLMClient, LLMTask
from .prompts import (
    PromptContext,
    PromptRegistry,
)
from .semantic_actions import (
    SemanticActionAnalyzer,
    SemanticActionDecision,
    semantic_action_from_internal_payload,
)
from .answer_grounding import generate_answer


_DOCUMENT_TOOL_NAME = "generate_pdf"
_EXCEL_TOOL_NAME = "generate_excel"
_POWERPOINT_TOOL_NAME = "generate_pptx"
_CANONICAL_REWRITTEN_QUERY_KEY = "canonical_rewritten_query"
_SEMANTIC_ACTION_DECISION_KEY = "semantic_action_decision"


@dataclass(frozen=True)
class FileCreationDecision:
    """Auditable projection of one grounded file-action contract."""

    selected_tool_name: str | None
    authorization_evidence: tuple[str, ...]
    file_type_evidence: tuple[str, ...]
    matched_file_types: tuple[str, ...]
    reason_summary: str


@dataclass(frozen=True)
class ContentCompositionScope:
    """Deterministic ownership boundary between prose and file generation."""

    answer_request_scope: str
    # This remains explicit in trace/prompt metadata so runtime verification can
    # prove the unconditional general-purpose answer-stage invariant.
    answer_generation_required: bool
    file_request_scope: str | None
    answer_generation_responsibility: str
    file_tool_responsibility: str | None


def classify_file_creation_request(
    rewritten_query: str,
    config: GeneralPurposeConfig,
    semantic_action_decision: SemanticActionDecision | None = None,
) -> FileCreationDecision:
    """Project a grounded semantic contract onto the optional file-tool route.

    ``config`` remains in the signature for compatibility with tool protocols. It
    does not contribute language signals. Without a model-produced, query-grounded
    decision, optional file creation fails closed to the mandatory answer stage.
    """

    del rewritten_query, config
    semantic = semantic_action_decision
    if semantic is None or not semantic.grounded:
        return FileCreationDecision(
            selected_tool_name=None,
            authorization_evidence=(),
            file_type_evidence=(),
            matched_file_types=(),
            reason_summary="semantic_decision_unavailable",
        )
    file_decision = semantic.file
    if not file_decision.authorizes_creation:
        return FileCreationDecision(
            selected_tool_name=None,
            authorization_evidence=file_decision.authorization_evidence,
            file_type_evidence=file_decision.type_evidence,
            matched_file_types=(
                (file_decision.file_type,)
                if file_decision.file_type != "none"
                else ()
            ),
            reason_summary=f"semantic_file_operation_{file_decision.operation}",
        )
    tool_by_type = {
        "pdf": _DOCUMENT_TOOL_NAME,
        "xlsx": _EXCEL_TOOL_NAME,
        "pptx": _POWERPOINT_TOOL_NAME,
    }
    selected_tool = tool_by_type.get(file_decision.file_type)
    if selected_tool is None:
        return FileCreationDecision(
            selected_tool_name=None,
            authorization_evidence=file_decision.authorization_evidence,
            file_type_evidence=file_decision.type_evidence,
            matched_file_types=(file_decision.file_type,),
            reason_summary="semantic_file_type_unsupported",
        )
    return FileCreationDecision(
        selected_tool_name=selected_tool,
        authorization_evidence=file_decision.authorization_evidence,
        file_type_evidence=file_decision.type_evidence,
        matched_file_types=(file_decision.file_type,),
        reason_summary=f"semantic_selected_{file_decision.file_type}",
    )


def _file_request_scope(composer_input: ContentComposerInput) -> str:
    metadata = getattr(composer_input, "metadata", {}) or {}
    scope = metadata.get("content_composition_scope", {})
    if isinstance(scope, dict):
        file_scope = scope.get("file_request_scope")
        if isinstance(file_scope, str) and file_scope.strip():
            return file_scope.strip()
    return composer_input.rewritten_query


def _canonical_rewritten_query(composer_input: ContentComposerInput) -> str:
    metadata = getattr(composer_input, "metadata", {}) or {}
    value = metadata.get(_CANONICAL_REWRITTEN_QUERY_KEY)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return composer_input.rewritten_query.strip()


def _composition_scope_payload(composer_input: ContentComposerInput) -> dict[str, Any]:
    metadata = getattr(composer_input, "metadata", {}) or {}
    scope = metadata.get("content_composition_scope", {})
    return scope if isinstance(scope, dict) else {}


def _file_scope_payload(composer_input: ContentComposerInput) -> dict[str, Any]:
    scope = _composition_scope_payload(composer_input)
    return {
        key: scope[key]
        for key in ("file_request_scope", "file_tool_responsibility")
        if scope.get(key) is not None
    }


def _unauthorized_file_tool_result(
    tool_name: str,
    composer_input: ContentComposerInput,
    config: GeneralPurposeConfig,
) -> ContentToolResult | None:
    canonical_query = _canonical_rewritten_query(composer_input)
    semantic = semantic_action_from_internal_payload(
        (getattr(composer_input, "metadata", {}) or {}).get(
            _SEMANTIC_ACTION_DECISION_KEY
        ),
        canonical_query=canonical_query,
    )
    decision = classify_file_creation_request(
        canonical_query,
        config,
        semantic,
    )
    if decision.selected_tool_name == tool_name:
        return None
    return ContentToolResult(
        tool_name=tool_name,
        output_text="",
        confidence=0.0,
        fallback_used=True,
        reason_summary=f"file_tool_rejected:{decision.reason_summary}",
        warnings=("file_creation_intent_not_authorized",),
    )


class ContentTool(Protocol):
    @property
    def name(self) -> str: ...
    @property
    def description(self) -> str: ...
    def can_handle(self, composer_input: ContentComposerInput, config: GeneralPurposeConfig) -> bool: ...
    def execute(self, composer_input: ContentComposerInput, config: GeneralPurposeConfig) -> ContentToolResult: ...


def _artifact_fallback_result(
    *,
    tool_name: str,
    file_type: str,
    filename: str,
    label: str,
    composer_input: ContentComposerInput,
    config: GeneralPurposeConfig,
    error: Exception,
) -> ContentToolResult:
    """Create a usable file even when the optional content-planning LLM fails."""
    if composer_input.repository is None:
        return ContentToolResult(
            tool_name=tool_name,
            output_text="",
            confidence=0.0,
            fallback_used=True,
            reason_summary=str(error),
            warnings=("artifact_plan_fallback",),
        )
    try:
        file_scope = _file_request_scope(composer_input)
        artifact = ArtifactGenerator(
            composer_input.repository,
            storage_dir=config.artifact_storage_dir,
            download_base_url=config.artifact_download_base_url,
        ).generate(
            user_id=composer_input.user_id,
            file_type=file_type,
            filename=filename,
            content=f"{label}\n\nRequest: {file_scope}",
            metadata={"tool_name": tool_name, "plan_summary": file_scope[:500]},
        )
    except Exception as artifact_error:
        return ContentToolResult(
            tool_name=tool_name,
            output_text="",
            confidence=0.0,
            fallback_used=True,
            reason_summary=str(artifact_error),
            warnings=("artifact_generation_unavailable",),
        )
    return ContentToolResult(
        tool_name=tool_name,
        output_text=f"Created {label.casefold()}: {artifact['filename']}.",
        confidence=0.55,
        fallback_used=True,
        reason_summary=f"{file_type}_file_created_with_fallback_plan: {type(error).__name__}",
        artifact=artifact,
        warnings=("artifact_plan_fallback",),
    )


class AnswerGenerationTool:
    @property
    def name(self) -> str: return "answer_generation"
    @property
    def description(self) -> str: return "Standard LLM text generation."

    def __init__(self, llm: LLMClient, prompt_registry: PromptRegistry) -> None:
        self.llm = llm
        self.prompt_registry = prompt_registry

    def can_handle(self, composer_input: ContentComposerInput, config: GeneralPurposeConfig) -> bool:
        return True

    def execute(self, composer_input: ContentComposerInput, config: GeneralPurposeConfig) -> ContentToolResult:
        try:
            generation = generate_answer(
                llm=self.llm,
                prompt_registry=self.prompt_registry,
                prompt_context=PromptContext(
                    stage="answer_generation",
                    user_id=composer_input.user_id,
                    rewritten_query=composer_input.rewritten_query,
                    metadata=composer_input.metadata,
                    platform_context=composer_input.platform_context,
                    extra={
                        "approved_conversation_history": composer_input.approved_conversation_history,
                        "approved_knowledge_evidence": composer_input.approved_knowledge_evidence,
                        "approved_knowledge_records": composer_input.approved_knowledge_records,
                        "approved_reminder_context": composer_input.approved_reminder_context,
                        "merged_supporting_detail": composer_input.merged_supporting_detail,
                        "sub_branch_supporting_prompt": composer_input.sub_branch_supporting_prompt,
                        "human_supporting_questions": [q.text for q in composer_input.human_supporting_questions],
                        "reminder_supporting_questions": [q.text for q in composer_input.reminder_supporting_questions],
                        "extracted_expected_response_types": [t.value for t in composer_input.extracted_expected_response_types],
                        "content_composition_scope": _composition_scope_payload(composer_input),
                    },
                ),
                approved_knowledge_records=composer_input.approved_knowledge_records,
                approved_knowledge_evidence=composer_input.approved_knowledge_evidence,
            )
            output = generation.text.strip()
            if not generation.model_succeeded:
                has_approved_knowledge = bool(
                    composer_input.approved_knowledge_records
                    or composer_input.approved_knowledge_evidence
                )
                return ContentToolResult(
                    tool_name=self.name,
                    output_text=output,
                    confidence=0.0,
                    fallback_used=True,
                    reason_summary=(
                        "answer_generation_used_grounded_fallback"
                        if has_approved_knowledge
                        else "answer_generation_failed"
                    ),
                    warnings=(
                        ("answer_grounded_evidence_fallback",)
                        if has_approved_knowledge
                        else ("answer_model_unavailable",)
                    ),
                )
            return ContentToolResult(
                tool_name=self.name,
                output_text=output,
                confidence=1.0,
                fallback_used=False,
                reason_summary="Answer generated.",
            )
        except Exception as e:
            return ContentToolResult(
                tool_name=self.name,
                output_text=self.prompt_registry.message("answer_model_unavailable"),
                confidence=0.0,
                fallback_used=True,
                reason_summary=f"answer_generation_failed: {e}",
                warnings=("answer_model_unavailable",),
            )


class GenerateExcelTool:
    @property
    def name(self) -> str: return "generate_excel"
    @property
    def description(self) -> str:
        return (
            "Plans a professional Excel workbook structure from user requirements. "
            "Produces a detailed text plan including: sheet layout, data table structure "
            "(headers, row types, column types), summary sheet content, dashboard chart plan "
            "(bar/line charts, KPI metrics), conditional formatting rules, "
            "and audience/purpose theming. "
            "Best for: spreadsheet planning, workbook design, data table structuring, "
            "KPI dashboard layout, financial model structure, tracker design."
        )

    def __init__(self, llm: LLMClient, prompt_registry: PromptRegistry, config: GeneralPurposeConfig | None = None) -> None:
        self.llm = llm
        self.prompt_registry = prompt_registry
        self.config = config

    def can_handle(self, composer_input: ContentComposerInput, config: GeneralPurposeConfig) -> bool:
        return classify_file_creation_request(
            _canonical_rewritten_query(composer_input),
            config,
        ).selected_tool_name == self.name

    def execute(self, composer_input: ContentComposerInput, config: GeneralPurposeConfig) -> ContentToolResult:
        rejected = _unauthorized_file_tool_result(self.name, composer_input, config)
        if rejected is not None:
            return rejected
        try:
            excel_planning_context = (
                "You are planning a professional Excel workbook. "
                "Structure the plan with: "
                "1) Summary sheet (title, subtitle, audience, purpose, key messages) "
                "2) Data sheets (one per data table: headers, typed rows, filters, "
                "   conditional color-scale formatting on numeric columns) "
                "3) Dashboard sheet (bar/line charts from numeric columns, KPI metrics: "
                "   record count, totals, averages) "
                "4) Theming (audience-appropriate color palette, auto-sized columns, "
                "   freeze panes, print layout) "
                "Present the complete workbook plan as structured text. "
                "Generate only content that belongs inside the workbook. Never draft "
                "a surrounding email, chat message, cover note, greeting, sign-off, or "
                "delivery instructions. "
                "Do not claim any file was created. Do not reference any filename or path."
            )
            plan_text = str(
                self.llm.chat(
                    task=LLMTask.WRITING,
                    system_prompt=self.prompt_registry.system(
                        "content_tool_answer_generation"
                    ),
                    user_prompt=self.prompt_registry.user(
                        PromptContext(
                            stage="content_tool_answer_generation",
                            rewritten_query=composer_input.rewritten_query,
                            extra={
                                "planning_context": excel_planning_context,
                                "content_composition_scope": _file_scope_payload(
                                    composer_input
                                ),
                                "approved_conversation_history": composer_input.approved_conversation_history,
                                "approved_knowledge_evidence": composer_input.approved_knowledge_evidence,
                                "approved_knowledge_records": composer_input.approved_knowledge_records,
                                "approved_reminder_context": composer_input.approved_reminder_context,
                            },
                        )
                    ),
                )
                or ""
            ).strip()
            if not plan_text:
                raise ValueError("generate_excel writer returned blank output")
            artifact = None
            output_text = plan_text
            if composer_input.repository is not None:
                artifact = ArtifactGenerator(
                    composer_input.repository,
                    storage_dir=config.artifact_storage_dir,
                    download_base_url=config.artifact_download_base_url,
                ).generate(
                    user_id=composer_input.user_id,
                    file_type="xlsx",
                    filename="generated_workbook.xlsx",
                    content=plan_text,
                    metadata={"tool_name": self.name, "plan_summary": plan_text[:500]},
                )
                output_text = f"Created Excel workbook: {artifact['filename']}."
            return ContentToolResult(
                tool_name=self.name,
                output_text=output_text,
                confidence=0.80,
                fallback_used=False,
                reason_summary="excel_file_created" if artifact else "excel_plan_generated",
                artifact=artifact,
            )
        except Exception as e:
            return _artifact_fallback_result(
                tool_name=self.name,
                file_type="xlsx",
                filename="generated_workbook.xlsx",
                label="Excel workbook",
                composer_input=composer_input,
                config=config,
                error=e,
            )


class GeneratePDFTool:
    @property
    def name(self) -> str: return "generate_pdf"
    @property
    def description(self) -> str:
        return (
            "Plans a professional PDF report structure from user requirements. "
            "Produces a detailed text plan including: cover page design, "
            "executive summary (key messages), per-section layout "
            "(section title, body text, bullet points, data table), "
            "appendix structure, assumptions, and audience/purpose theming. "
            "Best for: report planning, formal document structuring, "
            "business proposal layout, memo design, white paper outlining."
        )

    def __init__(self, llm: LLMClient, prompt_registry: PromptRegistry, config: GeneralPurposeConfig | None = None) -> None:
        self.llm = llm
        self.prompt_registry = prompt_registry
        self.config = config

    def can_handle(self, composer_input: ContentComposerInput, config: GeneralPurposeConfig) -> bool:
        return classify_file_creation_request(
            _canonical_rewritten_query(composer_input),
            config,
        ).selected_tool_name == self.name

    def execute(self, composer_input: ContentComposerInput, config: GeneralPurposeConfig) -> ContentToolResult:
        rejected = _unauthorized_file_tool_result(self.name, composer_input, config)
        if rejected is not None:
            return rejected
        try:
            pdf_planning_context = (
                "You are planning a professional PDF report. "
                "Structure the plan with: "
                "1) Cover page (title, subtitle, audience, purpose) "
                "2) Executive summary (first 3 key messages, assumptions) "
                "3) Structured sections (each: title, body paragraph, bullet points up to 8, "
                "   optional data table with headers and rows up to 30) "
                "4) Appendix (raw data tables if applicable) "
                "5) Theming (audience-appropriate: color headers, section hierarchy, page layout) "
                "Present the complete report plan as structured text. "
                "Generate only content that belongs inside the document. Never draft "
                "a surrounding email, chat message, cover note, greeting, sign-off, or "
                "delivery instructions. "
                "Do not claim any file was created. Do not reference any filename or path."
            )
            plan_text = str(
                self.llm.chat(
                    task=LLMTask.WRITING,
                    system_prompt=self.prompt_registry.system(
                        "content_tool_answer_generation"
                    ),
                    user_prompt=self.prompt_registry.user(
                        PromptContext(
                            stage="content_tool_answer_generation",
                            rewritten_query=composer_input.rewritten_query,
                            extra={
                                "planning_context": pdf_planning_context,
                                "content_composition_scope": _file_scope_payload(
                                    composer_input
                                ),
                                "approved_conversation_history": composer_input.approved_conversation_history,
                                "approved_knowledge_evidence": composer_input.approved_knowledge_evidence,
                                "approved_knowledge_records": composer_input.approved_knowledge_records,
                                "approved_reminder_context": composer_input.approved_reminder_context,
                            },
                        )
                    ),
                )
                or ""
            ).strip()
            if not plan_text:
                raise ValueError("generate_pdf writer returned blank output")
            artifact = None
            output_text = plan_text
            if composer_input.repository is not None:
                artifact = ArtifactGenerator(
                    composer_input.repository,
                    storage_dir=config.artifact_storage_dir,
                    download_base_url=config.artifact_download_base_url,
                ).generate(
                    user_id=composer_input.user_id,
                    file_type="pdf",
                    filename="generated_report.pdf",
                    content=plan_text,
                    metadata={"tool_name": self.name, "plan_summary": plan_text[:500]},
                )
                output_text = f"Created PDF document: {artifact['filename']}."
            return ContentToolResult(
                tool_name=self.name,
                output_text=output_text,
                confidence=0.80,
                fallback_used=False,
                reason_summary="pdf_file_created" if artifact else "pdf_plan_generated",
                artifact=artifact,
            )
        except Exception as e:
            return _artifact_fallback_result(
                tool_name=self.name,
                file_type="pdf",
                filename="generated_report.pdf",
                label="PDF document",
                composer_input=composer_input,
                config=config,
                error=e,
            )


class GeneratePPTXTool:
    @property
    def name(self) -> str: return "generate_pptx"
    @property
    def description(self) -> str:
        return (
            "Plans a professional PowerPoint presentation structure from user requirements. "
            "Produces a detailed text plan including: title slide, executive summary slide "
            "(key message cards), per-section slides dispatched by layout type "
            "(content, table, chart, timeline, matrix/risk, process/roadmap), "
            "speaker notes per slide, chart data plan (categories + series), "
            "and audience/purpose theming (Aptos font, color palette). "
            "Best for: presentation planning, slide deck structuring, pitch deck design, "
            "slideshow outlining, roadmap visualization planning."
        )

    def __init__(self, llm: LLMClient, prompt_registry: PromptRegistry, config: GeneralPurposeConfig | None = None) -> None:
        self.llm = llm
        self.prompt_registry = prompt_registry
        self.config = config

    def can_handle(self, composer_input: ContentComposerInput, config: GeneralPurposeConfig) -> bool:
        return classify_file_creation_request(
            _canonical_rewritten_query(composer_input),
            config,
        ).selected_tool_name == self.name

    def execute(self, composer_input: ContentComposerInput, config: GeneralPurposeConfig) -> ContentToolResult:
        rejected = _unauthorized_file_tool_result(self.name, composer_input, config)
        if rejected is not None:
            return rejected
        try:
            pptx_planning_context = (
                "You are planning a professional PowerPoint presentation. "
                "Structure the plan with: "
                "1) Title slide (title, subtitle, audience footer) "
                "2) Executive summary slide (key message cards, up to 4) "
                "3) Per-section slides, each with a layout type: "
                "   - content: title + bullets (up to 6) + callout takeaway "
                "   - table: title + headers + rows (up to 8 rows, 5 columns) "
                "   - chart: title + chart type (bar/line) + categories + series data + takeaway "
                "   - timeline: title + milestone items (up to 5) "
                "   - matrix: title + 2x2 priority grid (Priority/Monitor/Selective/Defer) "
                "   - process: title + process steps (up to 5) as connected boxes "
                "4) Speaker notes for every slide "
                "5) Theming (Aptos font, audience-appropriate primary/accent/neutral palette) "
                "Present the complete presentation plan as structured text. "
                "Generate only content that belongs inside the presentation. Never "
                "draft a surrounding email, chat message, cover note, greeting, "
                "sign-off, or delivery instructions. "
                "Do not claim any file was created. Do not reference any filename or path."
            )
            plan_text = str(
                self.llm.chat(
                    task=LLMTask.WRITING,
                    system_prompt=self.prompt_registry.system(
                        "content_tool_answer_generation"
                    ),
                    user_prompt=self.prompt_registry.user(
                        PromptContext(
                            stage="content_tool_answer_generation",
                            rewritten_query=composer_input.rewritten_query,
                            extra={
                                "planning_context": pptx_planning_context,
                                "content_composition_scope": _file_scope_payload(
                                    composer_input
                                ),
                                "approved_conversation_history": composer_input.approved_conversation_history,
                                "approved_knowledge_evidence": composer_input.approved_knowledge_evidence,
                                "approved_knowledge_records": composer_input.approved_knowledge_records,
                                "approved_reminder_context": composer_input.approved_reminder_context,
                            },
                        )
                    ),
                )
                or ""
            ).strip()
            if not plan_text:
                raise ValueError("generate_pptx writer returned blank output")
            artifact = None
            output_text = plan_text
            if composer_input.repository is not None:
                artifact = ArtifactGenerator(
                    composer_input.repository,
                    storage_dir=config.artifact_storage_dir,
                    download_base_url=config.artifact_download_base_url,
                ).generate(
                    user_id=composer_input.user_id,
                    file_type="pptx",
                    filename="generated_presentation.pptx",
                    content=plan_text,
                    metadata={"tool_name": self.name, "plan_summary": plan_text[:500]},
                )
                output_text = f"Created PowerPoint presentation: {artifact['filename']}."
            return ContentToolResult(
                tool_name=self.name,
                output_text=output_text,
                confidence=0.80,
                fallback_used=False,
                reason_summary="pptx_file_created" if artifact else "pptx_plan_generated",
                artifact=artifact,
            )
        except Exception as e:
            return _artifact_fallback_result(
                tool_name=self.name,
                file_type="pptx",
                filename="generated_presentation.pptx",
                label="PowerPoint presentation",
                composer_input=composer_input,
                config=config,
                error=e,
            )


@dataclass
class ContentToolRegistry:
    tools: list[ContentTool]
    config: GeneralPurposeConfig

    def __post_init__(self) -> None:
        names = [tool.name for tool in self.tools]
        if len(names) != len(set(names)):
            raise ValueError("ContentToolRegistry does not allow duplicate tool names.")
        self._tools: dict[str, ContentTool] = {tool.name: tool for tool in self.tools}

    def register(self, tool: ContentTool) -> None:
        if tool.name == "answer_generation" and tool.name in self._tools:
            raise ValueError("The mandatory answer_generation tool cannot be replaced.")
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> ContentTool | None:
        return self._tools.get(name)

    def get_all_tool_descriptions(self) -> dict[str, str]:
        return {tool.name: tool.description for tool in self._tools.values()}


@dataclass
class DeterministicContentComposer:
    """Execute the mandatory answer and any semantically authorized file tool."""

    registry: ContentToolRegistry
    semantic_analyzer: SemanticActionAnalyzer | None = None
    _answer_tool: ContentTool = field(init=False, repr=False)

    def __post_init__(self) -> None:
        answer_tool = self.registry.get_tool("answer_generation")
        if answer_tool is None:
            raise ValueError(
                "DeterministicContentComposer requires answer_generation for every "
                "general-purpose request."
            )
        # Pin the construction-time registration so later optional-tool updates
        # cannot replace or remove the mandatory answer stage.
        self._answer_tool = answer_tool

    def compose(self, composer_input: ContentComposerInput, config: GeneralPurposeConfig) -> ContentComposerResult:
        canonical_query = str(composer_input.rewritten_query or "").strip()
        canonical_metadata = (
            dict(composer_input.metadata)
            if isinstance(composer_input.metadata, dict)
            else {}
        )
        canonical_metadata[_CANONICAL_REWRITTEN_QUERY_KEY] = canonical_query
        composer_input = replace(
            composer_input,
            # Compatibility field only: it must never retain the pre-rewrite input.
            raw_user_query=canonical_query,
            rewritten_query=canonical_query,
            metadata=canonical_metadata,
        )
        semantic_decision = semantic_action_from_internal_payload(
            canonical_metadata.get(_SEMANTIC_ACTION_DECISION_KEY),
            canonical_query=canonical_query,
        )
        if self.semantic_analyzer is not None:
            semantic_decision = self.semantic_analyzer.analyze(
                canonical_query,
                approved_conversation_history=(
                    composer_input.approved_conversation_history
                ),
            )
        canonical_metadata[_SEMANTIC_ACTION_DECISION_KEY] = (
            semantic_decision.to_payload()
        )
        composer_input = replace(composer_input, metadata=canonical_metadata)
        decision = FileCreationDecision(
            selected_tool_name=None,
            authorization_evidence=(),
            file_type_evidence=(),
            matched_file_types=(),
            reason_summary="routing_failed_answer_only",
        )
        composition_scope = ContentCompositionScope(
            answer_request_scope=canonical_query,
            answer_generation_required=True,
            file_request_scope=None,
            answer_generation_responsibility=(
                "The optional file router was unavailable. Produce the complete "
                "user-facing answer without claiming that a file or side effect "
                "was created."
            ),
            file_tool_responsibility=None,
        )
        desired_file_tool_name: str | None = None
        file_tool: ContentTool | None = None
        file_route_status = "not_requested"
        routing_warning: str | None = None
        try:
            decision = classify_file_creation_request(
                canonical_query,
                config,
                semantic_decision,
            )
            desired_file_tool_name = decision.selected_tool_name
            if desired_file_tool_name is not None:
                if not config.content_composer_enabled:
                    file_route_status = "disabled"
                elif desired_file_tool_name not in config.content_composer_allowed_tools:
                    file_route_status = "not_allowed"
                else:
                    file_tool = self.registry.get_tool(desired_file_tool_name)
                    file_route_status = "ready" if file_tool is not None else "unavailable"

            # If the optional file stage cannot actually run, answer_generation owns the
            # complete request so no requested content is silently dropped.
            scope_decision = (
                decision
                if file_tool is not None
                else replace(decision, selected_tool_name=None)
            )
            if scope_decision.selected_tool_name is None:
                composition_scope = ContentCompositionScope(
                    answer_request_scope=canonical_query,
                    answer_generation_required=True,
                    file_request_scope=None,
                    answer_generation_responsibility=(
                        "Produce the complete user-facing answer. Do not claim a "
                        "file or external side effect was completed unless its "
                        "validated tool result is present."
                    ),
                    file_tool_responsibility=None,
                )
            else:
                composition_scope = ContentCompositionScope(
                    answer_request_scope=canonical_query,
                    answer_generation_required=True,
                    file_request_scope=canonical_query,
                    answer_generation_responsibility=(
                        "Produce all requested prose and message copy from the "
                        "complete request. The file tool separately owns file "
                        "creation; do not fabricate its completion result."
                    ),
                    file_tool_responsibility=(
                        "Create exactly the semantically authorized file from the "
                        "complete request. Do not compose or send an external message."
                    ),
                )
        except Exception as exc:
            # Deterministic file routing is optional. A malformed runtime policy
            # must degrade locally to the mandatory answer stage instead of
            # escaping to the branch/router fallback pipeline.
            file_tool = None
            desired_file_tool_name = None
            file_route_status = "routing_failed"
            routing_warning = f"content_routing_failed:{type(exc).__name__}"
        answer_metadata = dict(composer_input.metadata)
        # The answer and file planners receive the same canonical request. Their
        # typed responsibilities prevent content loss in compound, unseen turns.
        answer_metadata.pop(_CANONICAL_REWRITTEN_QUERY_KEY, None)
        answer_metadata["content_composition_scope"] = {
            "answer_request_scope": composition_scope.answer_request_scope,
            "answer_generation_required": composition_scope.answer_generation_required,
            "answer_generation_responsibility": composition_scope.answer_generation_responsibility,
            "assigned_file_tool": desired_file_tool_name if file_tool is not None else None,
        }
        answer_input = replace(
            composer_input,
            raw_user_query=composition_scope.answer_request_scope,
            rewritten_query=composition_scope.answer_request_scope,
            metadata=answer_metadata,
        )

        used_tool_names: list[str] = []
        results: list[ContentToolResult] = []
        try:
            answer_result = self._answer_tool.execute(answer_input, config)
        except Exception as exc:
            answer_result = ContentToolResult(
                tool_name="answer_generation",
                output_text="The answer model is temporarily unavailable.",
                confidence=0.0,
                fallback_used=True,
                reason_summary=f"answer_generation_failed:{type(exc).__name__}",
                warnings=("answer_model_unavailable",),
            )
        used_tool_names.append("answer_generation")
        results.append(answer_result)
        selected_file_tool_name: str | None = None
        if file_tool is not None and desired_file_tool_name is not None:
            # Each Microsoft tool independently revalidates the serialized,
            # grounded semantic authorization before producing an artifact.
            file_metadata = dict(composer_input.metadata)
            file_metadata["content_composition_scope"] = {
                "file_request_scope": composition_scope.file_request_scope,
                "file_tool_responsibility": composition_scope.file_tool_responsibility,
            }
            file_input = replace(
                composer_input,
                raw_user_query=composition_scope.file_request_scope or "",
                rewritten_query=composition_scope.file_request_scope or "",
                metadata=file_metadata,
            )
            try:
                file_result = file_tool.execute(file_input, config)
            except Exception as exc:
                file_result = ContentToolResult(
                    tool_name=desired_file_tool_name,
                    output_text="",
                    confidence=0.0,
                    fallback_used=True,
                    reason_summary=(
                        f"microsoft_file_tool_failed:{type(exc).__name__}"
                    ),
                    warnings=("microsoft_file_tool_unavailable",),
                )
            results.append(file_result)
            used_tool_names.append(desired_file_tool_name)
            selected_file_tool_name = desired_file_tool_name
            file_route_status = "executed"

        normalized_outputs = [
            str(result.output_text or "").strip()
            for result in results
        ]
        output_parts = [output for output in normalized_outputs if output]
        artifacts = tuple(
            result.artifact
            for result in results
            if result.artifact is not None
        )
        warnings = tuple(
            dict.fromkeys(
                [
                    *(warning for result in results for warning in result.warnings),
                    *([routing_warning] if routing_warning else []),
                ]
            )
        )
        route_fallback = file_route_status in {
            "not_allowed",
            "unavailable",
            "scope_projection_failed",
            "routing_failed",
        }
        stage_outcomes = {
            result.tool_name: {
                "attempted": True,
                # A fallback may still contribute safe text or a basic artifact,
                # but it is not proof that the requested model stage succeeded.
                "succeeded": bool(
                    not result.fallback_used
                    and (normalized_output or result.artifact is not None)
                ),
                "fallback_used": result.fallback_used,
                "output_contributed": bool(normalized_output),
                "artifact_contributed": result.artifact is not None,
                "warnings": list(result.warnings),
            }
            for result, normalized_output in zip(results, normalized_outputs)
        }
        trace = {
            "classifier": "grounded_semantic_contract",
            "decision": decision.reason_summary,
            "authorization_evidence": decision.authorization_evidence,
            "file_type_evidence": decision.file_type_evidence,
            "matched_file_types": decision.matched_file_types,
            "selected_file_tool": selected_file_tool_name,
            "file_route_status": file_route_status,
            "executed_tools": used_tool_names,
            "attempted_tools": used_tool_names,
            "successful_tools": [
                name
                for name, outcome in stage_outcomes.items()
                if outcome["succeeded"]
            ],
            "output_contributing_tools": [
                name
                for name, outcome in stage_outcomes.items()
                if outcome["output_contributed"] or outcome["artifact_contributed"]
            ],
            "stage_outcomes": stage_outcomes,
            "answer_request_scope": composition_scope.answer_request_scope,
            "answer_generation_required": composition_scope.answer_generation_required,
            "file_request_scope": composition_scope.file_request_scope,
        }
        return ContentComposerResult(
            final_response_text="\n\n".join(output_parts),
            tool_trace_summary=json.dumps(trace, sort_keys=True),
            used_tool_names=tuple(used_tool_names),
            confidence=min(result.confidence for result in results),
            fallback_used=any(result.fallback_used for result in results) or route_fallback,
            reason_summary="; ".join(result.reason_summary for result in results),
            content_warnings=warnings,
            artifacts=artifacts,
            answer_response_text=normalized_outputs[0],
            semantic_action_decision=semantic_decision.to_payload(),
        )
