"""Content composer and tool abstraction for the General Response branch."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Protocol

from .artifacts import ArtifactGenerator
from .config import GeneralPurposeConfig
from .contracts import ContentComposerResult, ContentToolResult, ContentComposerInput
from .llm import LLMClient, LLMTask
from .prompts import (
    CONTENT_COMPOSER_REACT_SCHEMA,
    GENERATE_EXCEL_PLANNER_SCHEMA,
    GENERATE_PDF_PLANNER_SCHEMA,
    GENERATE_PPTX_PLANNER_SCHEMA,
    PromptContext,
    PromptRegistry,
)

logger = logging.getLogger(__name__)


class ContentTool(Protocol):
    @property
    def name(self) -> str: ...
    @property
    def description(self) -> str: ...
    def can_handle(self, composer_input: ContentComposerInput, config: GeneralPurposeConfig) -> bool: ...
    def execute(self, composer_input: ContentComposerInput, config: GeneralPurposeConfig) -> ContentToolResult: ...


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
            output = self.llm.chat(
                task=LLMTask.ANSWER,
                system_prompt=self.prompt_registry.system("answer_generation"),
                user_prompt=self.prompt_registry.user(
                    PromptContext(stage="answer_generation", rewritten_query=composer_input.rewritten_query)
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
        query_lower = composer_input.rewritten_query.casefold()
        return any(k in query_lower for k in config.excel_tool_signal_keywords)

    def execute(self, composer_input: ContentComposerInput, config: GeneralPurposeConfig) -> ContentToolResult:
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
                "Do not claim any file was created. Do not reference any filename or path."
            )
            plan_text = self.llm.chat(
                task=LLMTask.WRITING,
                system_prompt=self.prompt_registry.system("content_tool_answer_generation"),
                user_prompt=self.prompt_registry.user(
                    PromptContext(stage="content_tool_answer_generation", rewritten_query=composer_input.rewritten_query, extra={"planning_context": excel_planning_context})
                ),
            )
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
            return ContentToolResult(
                tool_name=self.name,
                output_text="",
                confidence=0.0,
                fallback_used=True,
                reason_summary=str(e),
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
        query_lower = composer_input.rewritten_query.casefold()
        return any(k in query_lower for k in config.pdf_tool_signal_keywords)

    def execute(self, composer_input: ContentComposerInput, config: GeneralPurposeConfig) -> ContentToolResult:
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
                "Do not claim any file was created. Do not reference any filename or path."
            )
            plan_text = self.llm.chat(
                task=LLMTask.WRITING,
                system_prompt=self.prompt_registry.system("content_tool_answer_generation"),
                user_prompt=self.prompt_registry.user(
                    PromptContext(stage="content_tool_answer_generation", rewritten_query=composer_input.rewritten_query, extra={"planning_context": pdf_planning_context})
                ),
            )
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
            return ContentToolResult(
                tool_name=self.name,
                output_text="",
                confidence=0.0,
                fallback_used=True,
                reason_summary=str(e),
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
        query_lower = composer_input.rewritten_query.casefold()
        return any(k in query_lower for k in config.pptx_tool_signal_keywords)

    def execute(self, composer_input: ContentComposerInput, config: GeneralPurposeConfig) -> ContentToolResult:
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
                "Do not claim any file was created. Do not reference any filename or path."
            )
            plan_text = self.llm.chat(
                task=LLMTask.WRITING,
                system_prompt=self.prompt_registry.system("content_tool_answer_generation"),
                user_prompt=self.prompt_registry.user(
                    PromptContext(stage="content_tool_answer_generation", rewritten_query=composer_input.rewritten_query, extra={"planning_context": pptx_planning_context})
                ),
            )
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
            return ContentToolResult(
                tool_name=self.name,
                output_text="",
                confidence=0.0,
                fallback_used=True,
                reason_summary=str(e),
            )


@dataclass
class ContentToolRegistry:
    tools: list[ContentTool]
    config: GeneralPurposeConfig

    def __post_init__(self) -> None:
        self._tools: dict[str, ContentTool] = {tool.name: tool for tool in self.tools}

    def register(self, tool: ContentTool) -> None:
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> ContentTool | None:
        return self._tools.get(name)

    def get_all_tool_descriptions(self) -> dict[str, str]:
        return {tool.name: tool.description for tool in self._tools.values()}


@dataclass
class ReActContentComposer:
    registry: ContentToolRegistry
    llm: LLMClient
    prompt_registry: PromptRegistry
    config: GeneralPurposeConfig

    def compose(self, composer_input: ContentComposerInput, config: GeneralPurposeConfig) -> ContentComposerResult:
        artifacts: list[dict[str, Any]] = []
        if not config.content_composer_enabled:
            tool = self.registry.get_tool(config.content_composer_default_tool)
            if not tool:
                return ContentComposerResult(
                    final_response_text="System error: Default tool not registered.",
                    tool_trace_summary="Composer disabled, default tool missing.",
                    used_tool_names=(),
                    confidence=0.0,
                    fallback_used=True,
                    reason_summary="Disabled by config",
                    content_warnings=("System disabled",),
                )
            result = tool.execute(composer_input, config)
            if result.artifact:
                artifacts.append(result.artifact)
            return ContentComposerResult(
                final_response_text=result.output_text,
                tool_trace_summary=f"Composer disabled, used default tool: {tool.name}",
                used_tool_names=(tool.name,),
                confidence=result.confidence,
                fallback_used=result.fallback_used,
                reason_summary=result.reason_summary,
                content_warnings=result.warnings,
                artifacts=tuple(artifacts),
            )
            
        # Short-circuit if only one tool is registered
        if len(self.registry._tools) == 1:
            tool = list(self.registry._tools.values())[0]
            result = tool.execute(composer_input, config)
            if result.artifact:
                artifacts.append(result.artifact)
            return ContentComposerResult(
                final_response_text=result.output_text,
                tool_trace_summary=f"Single tool short-circuit: {tool.name}",
                used_tool_names=(tool.name,),
                confidence=result.confidence,
                fallback_used=result.fallback_used,
                reason_summary=result.reason_summary,
                content_warnings=result.warnings,
                artifacts=tuple(artifacts),
            )

        tool_trace: list[dict[str, Any]] = []
        used_tool_names: list[str] = []
        content_warnings: list[str] = []
        final_answer = ""
        is_final = False

        for _ in range(config.content_composer_max_iterations):
            try:
                payload = self.llm.generate_json(
                    task=LLMTask.CONTENT_COMPOSER_REACT,
                    system_prompt=self.prompt_registry.system("content_composer_react"),
                    user_prompt=self.prompt_registry.user(
                        PromptContext(
                            stage="content_composer_react",
                            rewritten_query=composer_input.rewritten_query,
                            extra={
                                "available_tools": self.registry.get_all_tool_descriptions(),
                                "tool_trace": tool_trace,
                                "approved_conversation_history": composer_input.approved_conversation_history,
                                "sub_branch_supporting_prompt": composer_input.sub_branch_supporting_prompt,
                                "human_supporting_questions": [q.text for q in composer_input.human_supporting_questions],
                                "reminder_supporting_questions": [q.text for q in composer_input.reminder_supporting_questions],
                                "extracted_expected_response_types": [t.value for t in composer_input.extracted_expected_response_types],
                            },
                        )
                    ),
                    schema=CONTENT_COMPOSER_REACT_SCHEMA,
                )
            except Exception as e:
                logger.debug("ReAct loop failed: %s", e)
                break

            thought = payload.get("thought", "")
            tool_name = str(payload.get("tool_name", config.content_composer_fallback_tool))
            is_final = bool(payload.get("is_final_answer", False))
            
            if tool_name == "answer_generation":
                is_final = True
            
            tool = self.registry.get_tool(tool_name)
            if not tool:
                tool = self.registry.get_tool(config.content_composer_fallback_tool)
                if tool:
                    tool_name = tool.name
            
            if tool:
                if tool_name not in used_tool_names:
                    used_tool_names.append(tool_name)
                result = tool.execute(composer_input, config)
                content_warnings.extend(result.warnings)
                if result.artifact:
                    artifacts.append(result.artifact)
                observation = f"Tool {tool_name} returned: {result.reason_summary}. Preview: {result.output_text[:100]}"
                tool_trace.append({
                    "thought": thought,
                    "tool": tool_name,
                    "result_summary": result.reason_summary,
                    "output_preview": result.output_text[:100],
                    "observation": observation,
                })
                
                if is_final:
                    final_answer = result.output_text
                    break
            else:
                break

        if not is_final:
            tool = self.registry.get_tool(config.content_composer_fallback_tool)
            if tool:
                result = tool.execute(composer_input, config)
                content_warnings.extend(result.warnings)
                if result.artifact:
                    artifacts.append(result.artifact)
                final_answer = result.output_text
                if tool.name not in used_tool_names:
                    used_tool_names.append(tool.name)

        return ContentComposerResult(
            final_response_text=final_answer,
            tool_trace_summary=json.dumps(tool_trace),
            used_tool_names=tuple(used_tool_names),
            confidence=1.0,
            fallback_used=not is_final,
            reason_summary="ReAct loop completed.",
            content_warnings=tuple(dict.fromkeys(content_warnings)),
            artifacts=tuple(artifacts),
        )
