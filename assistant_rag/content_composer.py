"""Content composer and tool abstraction for the General Response branch."""

from __future__ import annotations

import json
from functools import lru_cache
import re
import unicodedata
from dataclasses import dataclass, replace
from typing import Any, Protocol

from .artifacts import ArtifactGenerator
from .config import GeneralPurposeConfig
from .contracts import ContentComposerResult, ContentToolResult, ContentComposerInput
from .llm import LLMClient, LLMTask
from .prompts import (
    PromptContext,
    PromptRegistry,
)


_DOCUMENT_TOOL_NAME = "generate_pdf"
_EXCEL_TOOL_NAME = "generate_excel"
_POWERPOINT_TOOL_NAME = "generate_pptx"
_GENERIC_FILE_KEYWORDS = frozenset(
    {"file", "artifact", "attachment", "downloadable file", "editable file", "template"}
)


@dataclass(frozen=True)
class FileCreationDecision:
    """Auditable result of the raw-query-only file-intent classifier."""

    selected_tool_name: str | None
    matched_verb_keywords: tuple[str, ...]
    matched_file_keywords: tuple[str, ...]
    matched_file_types: tuple[str, ...]
    reason_summary: str


@dataclass(frozen=True)
class ContentCompositionScope:
    """Deterministic ownership boundary between prose and file generation."""

    answer_request_scope: str
    file_request_scope: str | None
    answer_generation_responsibility: str
    file_tool_responsibility: str | None


@dataclass(frozen=True)
class _KeywordMatch:
    file_type: str
    keyword: str
    start: int
    end: int


def _normalize_for_matching(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold()


@lru_cache(maxsize=1024)
def _keyword_pattern(keyword: str) -> re.Pattern[str]:
    normalized = " ".join(_normalize_for_matching(keyword).split())
    if not normalized:
        return re.compile(r"(?!x)x")
    body = re.escape(normalized).replace(r"\ ", r"\s+")
    prefix = r"(?<!\w)" if normalized[0].isalnum() or normalized[0] == "_" else ""
    suffix = r"(?!\w)" if normalized[-1].isalnum() or normalized[-1] == "_" else ""
    return re.compile(f"{prefix}{body}{suffix}")


def _matches(text: str, keywords: tuple[str, ...]) -> tuple[tuple[str, int, int], ...]:
    found: list[tuple[str, int, int]] = []
    for keyword in keywords:
        for match in _keyword_pattern(keyword).finditer(text):
            found.append((keyword, match.start(), match.end()))
    return tuple(found)


def classify_file_creation_request(
    raw_user_query: str,
    config: GeneralPurposeConfig,
) -> FileCreationDecision:
    """Select one file tool only when both required keyword classes are explicit.

    The classifier never reads rewritten text or model output. Independently matched
    file types fail closed. A longer cross-type phrase suppresses a keyword contained
    inside it so requests such as ``create an Excel file`` are not made ambiguous by
    the generic document keyword ``file``.
    """

    text = _normalize_for_matching(raw_user_query)
    verb_matches = _matches(text, config.file_creation_verb_keywords)
    matched_verbs = tuple(dict.fromkeys(match[0] for match in verb_matches))

    file_groups = (
        ("document", _DOCUMENT_TOOL_NAME, config.document_tool_signal_keywords),
        ("excel", _EXCEL_TOOL_NAME, config.excel_tool_signal_keywords),
        ("powerpoint", _POWERPOINT_TOOL_NAME, config.pptx_tool_signal_keywords),
    )
    all_file_matches = [
        _KeywordMatch(file_type=file_type, keyword=keyword, start=start, end=end)
        for file_type, _tool_name, keywords in file_groups
        for keyword, start, end in _matches(text, keywords)
    ]
    effective_file_matches = [
        candidate
        for candidate in all_file_matches
        if not any(
            other.file_type != candidate.file_type
            and other.start <= candidate.start
            and other.end >= candidate.end
            and (other.end - other.start) > (candidate.end - candidate.start)
            for other in all_file_matches
        )
    ]
    if any(match.file_type in {"excel", "powerpoint"} for match in effective_file_matches):
        # Generic artifact nouns establish that a file is wanted but do not override
        # an explicit Excel or PowerPoint type elsewhere in the same request.
        effective_file_matches = [
            match
            for match in effective_file_matches
            if not (match.file_type == "document" and match.keyword in _GENERIC_FILE_KEYWORDS)
        ]
    matched_file_types = tuple(
        file_type
        for file_type, _tool_name, _keywords in file_groups
        if any(match.file_type == file_type for match in effective_file_matches)
    )
    matched_file_keywords = tuple(
        dict.fromkeys(
            match.keyword
            for match in sorted(effective_file_matches, key=lambda item: (item.start, item.end, item.keyword))
        )
    )

    if not matched_verbs:
        return FileCreationDecision(
            selected_tool_name=None,
            matched_verb_keywords=(),
            matched_file_keywords=matched_file_keywords,
            matched_file_types=matched_file_types,
            reason_summary="missing_verb_keyword",
        )
    if not matched_file_types:
        return FileCreationDecision(
            selected_tool_name=None,
            matched_verb_keywords=matched_verbs,
            matched_file_keywords=(),
            matched_file_types=(),
            reason_summary="missing_file_keyword",
        )
    if len(matched_file_types) != 1:
        return FileCreationDecision(
            selected_tool_name=None,
            matched_verb_keywords=matched_verbs,
            matched_file_keywords=matched_file_keywords,
            matched_file_types=matched_file_types,
            reason_summary="ambiguous_file_types",
        )

    selected_type = matched_file_types[0]
    selected_tool_name = next(
        tool_name for file_type, tool_name, _keywords in file_groups if file_type == selected_type
    )
    return FileCreationDecision(
        selected_tool_name=selected_tool_name,
        matched_verb_keywords=matched_verbs,
        matched_file_keywords=matched_file_keywords,
        matched_file_types=matched_file_types,
        reason_summary=f"selected_{selected_type}_tool",
    )


_FILE_SCOPE_BOUNDARY_PATTERN = re.compile(
    r"(?:[;.!?]\s+|\n+|,\s*(?:and|then)\s+|\b(?:and|then)\s+|"
    r"\b(?:attach|attaching|include|including)\s+(?=(?:an?\s+)?(?:attachment|file|document|excel|spreadsheet|"
    r"workbook|powerpoint|presentation|deck|report)\b)|"
    r"\bwith\s+(?=(?:an?\s+)?(?:attachment|file|document|excel|spreadsheet|"
    r"workbook|powerpoint|presentation|deck|report)\b))",
    re.IGNORECASE,
)
_NON_FILE_DELIVERABLE_AFTER_BOUNDARY = re.compile(
    r"^(?:please\s+)?(?:also\s+)?(?:"
    r"(?:write|draft|compose|prepare)\s+(?:an?\s+)?(?:email|e-mail|message|chat message|cover note|reply)\b|"
    r"(?:send|email)\s+(?:it|this|that|the\s+(?:file|document|workbook|presentation|attachment))\b|"
    r"(?:an?\s+)?(?:email|e-mail|message|chat message|cover note|reply)\s+to\b"
    r")",
    re.IGNORECASE,
)


def _selected_file_keywords(
    decision: FileCreationDecision,
    config: GeneralPurposeConfig,
) -> tuple[str, ...]:
    if decision.selected_tool_name == _DOCUMENT_TOOL_NAME:
        return config.document_tool_signal_keywords
    if decision.selected_tool_name == _EXCEL_TOOL_NAME:
        return config.excel_tool_signal_keywords
    if decision.selected_tool_name == _POWERPOINT_TOOL_NAME:
        return config.pptx_tool_signal_keywords
    return ()


def _derive_composition_scope(
    raw_user_query: str,
    decision: FileCreationDecision,
    config: GeneralPurposeConfig,
) -> ContentCompositionScope:
    """Project a compound request into non-file prose and file-only scopes.

    Routing still uses the complete raw query.  This projection happens only after
    authorization and limits what the optional file planner receives.  The first
    file keyword anchors the file clause; common compound-request boundaries keep an
    email/message clause outside the attachment's content.
    """

    if decision.selected_tool_name is None:
        return ContentCompositionScope(
            answer_request_scope=raw_user_query.strip(),
            file_request_scope=None,
            answer_generation_responsibility=(
                "No Microsoft file tool is assigned. Produce the complete user-facing "
                "answer or requested prose from this scope. Do not claim that a file "
                "or any other side effect was created."
            ),
            file_tool_responsibility=None,
        )

    answer_responsibility = (
        "Always produce the user-facing non-file deliverable. If the request asks for "
        "an email, message, cover note, explanation, or other prose outside an "
        "attachment, compose that prose here. Do not generate the attachment's "
        "internal document sections, workbook rows, or presentation slides, and do "
        "not claim that a file was created."
    )

    normalized = _normalize_for_matching(raw_user_query)
    file_matches = _matches(normalized, _selected_file_keywords(decision, config))
    if not file_matches:
        # Protected by the classifier, but fail closed to an answer-only scope if the
        # configured keyword policy changes between classification and projection.
        return ContentCompositionScope(
            answer_request_scope=raw_user_query.strip(),
            file_request_scope=None,
            answer_generation_responsibility=answer_responsibility,
            file_tool_responsibility=None,
        )

    first_file_start = min(start for _keyword, start, _end in file_matches)
    first_file_end = min(
        end for _keyword, start, end in file_matches if start == first_file_start
    )
    boundaries = tuple(_FILE_SCOPE_BOUNDARY_PATTERN.finditer(raw_user_query))
    prior_boundaries = [boundary for boundary in boundaries if boundary.end() <= first_file_start]
    prior_boundary = prior_boundaries[-1] if prior_boundaries else None
    if prior_boundary is not None and len(prior_boundaries) >= 2:
        previous_boundary = prior_boundaries[-2]
        boundary_text = prior_boundary.group().strip().casefold()
        between_boundaries = raw_user_query[previous_boundary.end():prior_boundary.start()]
        if (
            boundary_text.startswith(("attach", "include"))
            and not between_boundaries.strip()
        ):
            # Prefer "and/then" so the file scope retains its explicit attach/include
            # verb while neither connector leaks into the answer projection.
            prior_boundary = previous_boundary
    file_start = prior_boundary.end() if prior_boundary else 0
    answer_prefix_end = prior_boundary.start() if prior_boundary else 0
    file_end = len(raw_user_query)
    answer_suffix_start = len(raw_user_query)
    for boundary in boundaries:
        if boundary.start() < first_file_end:
            continue
        following_text = raw_user_query[boundary.end():].lstrip()
        if _NON_FILE_DELIVERABLE_AFTER_BOUNDARY.match(following_text):
            file_end = boundary.start()
            answer_suffix_start = boundary.end()
            break

    file_scope = raw_user_query[file_start:file_end].strip(" \t\r\n,;.-")
    answer_parts = (
        raw_user_query[:answer_prefix_end].strip(" \t\r\n,;.-"),
        raw_user_query[answer_suffix_start:].strip(" \t\r\n,;.-"),
    )
    answer_scope = " ".join(part for part in answer_parts if part).strip()
    if not answer_scope:
        file_label = {
            _DOCUMENT_TOOL_NAME: "document",
            _EXCEL_TOOL_NAME: "Excel workbook",
            _POWERPOINT_TOOL_NAME: "PowerPoint presentation",
        }.get(decision.selected_tool_name, "file")
        answer_scope = (
            f"Provide only a concise user-facing handoff for the separately generated "
            f"{file_label}; do not generate any content that belongs inside it."
        )

    return ContentCompositionScope(
        answer_request_scope=answer_scope,
        file_request_scope=file_scope or raw_user_query.strip(),
        answer_generation_responsibility=answer_responsibility,
        file_tool_responsibility=(
            "Construct only content that belongs inside the requested file. Exclude "
            "any surrounding email, chat message, cover note, recipient greeting or "
            "sign-off, and delivery instructions unless the file clause explicitly "
            "states that such content belongs inside the file."
        ),
    )


def _file_request_scope(composer_input: ContentComposerInput) -> str:
    metadata = getattr(composer_input, "metadata", {}) or {}
    scope = metadata.get("content_composition_scope", {})
    if isinstance(scope, dict):
        file_scope = scope.get("file_request_scope")
        if isinstance(file_scope, str) and file_scope.strip():
            return file_scope.strip()
    return composer_input.raw_user_query


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
    decision = classify_file_creation_request(composer_input.raw_user_query, config)
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
            output = self.llm.chat(
                task=LLMTask.ANSWER,
                system_prompt=self.prompt_registry.system("answer_generation"),
                user_prompt=self.prompt_registry.user(
                    PromptContext(
                        stage="answer_generation",
                        user_id=composer_input.user_id,
                        raw_query=composer_input.raw_user_query,
                        rewritten_query=composer_input.rewritten_query,
                        metadata=composer_input.metadata,
                        platform_context=composer_input.platform_context,
                        extra={
                            "approved_conversation_history": composer_input.approved_conversation_history,
                            "approved_knowledge_evidence": composer_input.approved_knowledge_evidence,
                            "approved_reminder_context": composer_input.approved_reminder_context,
                            "merged_supporting_detail": composer_input.merged_supporting_detail,
                            "sub_branch_supporting_prompt": composer_input.sub_branch_supporting_prompt,
                            "human_supporting_questions": [q.text for q in composer_input.human_supporting_questions],
                            "reminder_supporting_questions": [q.text for q in composer_input.reminder_supporting_questions],
                            "extracted_expected_response_types": [t.value for t in composer_input.extracted_expected_response_types],
                            "content_composition_scope": _composition_scope_payload(composer_input),
                        },
                    )
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
        return classify_file_creation_request(composer_input.raw_user_query, config).selected_tool_name == self.name

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
            plan_text = self.llm.chat(
                task=LLMTask.WRITING,
                system_prompt=self.prompt_registry.system("content_tool_answer_generation"),
                user_prompt=self.prompt_registry.user(
                    PromptContext(
                        stage="content_tool_answer_generation",
                        rewritten_query=composer_input.rewritten_query,
                        extra={
                            "planning_context": excel_planning_context,
                            "content_composition_scope": _file_scope_payload(composer_input),
                        },
                    )
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
        return classify_file_creation_request(composer_input.raw_user_query, config).selected_tool_name == self.name

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
            plan_text = self.llm.chat(
                task=LLMTask.WRITING,
                system_prompt=self.prompt_registry.system("content_tool_answer_generation"),
                user_prompt=self.prompt_registry.user(
                    PromptContext(
                        stage="content_tool_answer_generation",
                        rewritten_query=composer_input.rewritten_query,
                        extra={
                            "planning_context": pdf_planning_context,
                            "content_composition_scope": _file_scope_payload(composer_input),
                        },
                    )
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
        return classify_file_creation_request(composer_input.raw_user_query, config).selected_tool_name == self.name

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
            plan_text = self.llm.chat(
                task=LLMTask.WRITING,
                system_prompt=self.prompt_registry.system("content_tool_answer_generation"),
                user_prompt=self.prompt_registry.user(
                    PromptContext(
                        stage="content_tool_answer_generation",
                        rewritten_query=composer_input.rewritten_query,
                        extra={
                            "planning_context": pptx_planning_context,
                            "content_composition_scope": _file_scope_payload(composer_input),
                        },
                    )
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
        self._tools: dict[str, ContentTool] = {tool.name: tool for tool in self.tools}

    def register(self, tool: ContentTool) -> None:
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> ContentTool | None:
        return self._tools.get(name)

    def get_all_tool_descriptions(self) -> dict[str, str]:
        return {tool.name: tool.description for tool in self._tools.values()}


@dataclass
class DeterministicContentComposer:
    """Always execute answer generation, then at most one authorized file tool."""

    registry: ContentToolRegistry

    def compose(self, composer_input: ContentComposerInput, config: GeneralPurposeConfig) -> ContentComposerResult:
        decision = classify_file_creation_request(composer_input.raw_user_query, config)
        desired_file_tool_name = decision.selected_tool_name
        file_tool: ContentTool | None = None
        file_route_status = "not_requested"
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
        composition_scope = _derive_composition_scope(
            composer_input.raw_user_query,
            scope_decision,
            config,
        )
        if file_tool is not None and composition_scope.file_request_scope is None:
            file_tool = None
            file_route_status = "scope_projection_failed"
            composition_scope = _derive_composition_scope(
                composer_input.raw_user_query,
                replace(decision, selected_tool_name=None),
                config,
            )
        answer_metadata = dict(composer_input.metadata)
        answer_metadata["content_composition_scope"] = {
            "answer_request_scope": composition_scope.answer_request_scope,
            "answer_generation_responsibility": composition_scope.answer_generation_responsibility,
            "assigned_file_tool": desired_file_tool_name if file_tool is not None else None,
        }
        answer_input = replace(
            composer_input,
            raw_user_query=composition_scope.answer_request_scope,
            rewritten_query=composition_scope.answer_request_scope,
            metadata=answer_metadata,
        )

        answer_tool = self.registry.get_tool("answer_generation")
        if answer_tool is None:
            trace = {
                "classifier": "deterministic_keyword_pipeline",
                "decision": decision.reason_summary,
                "matched_verbs": decision.matched_verb_keywords,
                "matched_file_keywords": decision.matched_file_keywords,
                "matched_file_types": decision.matched_file_types,
                "selected_file_tool": None,
                "executed_tools": (),
                "pipeline_status": "answer_tool_unavailable",
            }
            return ContentComposerResult(
                final_response_text="System error: General answer tool not registered.",
                tool_trace_summary=json.dumps(trace, sort_keys=True),
                used_tool_names=(),
                confidence=0.0,
                fallback_used=True,
                reason_summary="deterministic_tool_unavailable",
                content_warnings=("answer_tool_unavailable",),
            )

        # Stage one is unconditional.  Configuration may disable file creation, but
        # it cannot disable the user-facing answer-generation responsibility.
        answer_result = answer_tool.execute(answer_input, config)
        used_tool_names = ["answer_generation"]
        results = [answer_result]
        selected_file_tool_name: str | None = None
        if file_tool is not None and desired_file_tool_name is not None:
            # The raw query is retained solely so each Microsoft tool can
            # independently re-check deterministic authorization.  Only the
            # file clause reaches its content-planning prompt and fallback.
            file_metadata = dict(composer_input.metadata)
            file_metadata["content_composition_scope"] = {
                "file_request_scope": composition_scope.file_request_scope,
                "file_tool_responsibility": composition_scope.file_tool_responsibility,
            }
            file_input = replace(
                composer_input,
                rewritten_query=composition_scope.file_request_scope or "",
                metadata=file_metadata,
            )
            file_result = file_tool.execute(file_input, config)
            results.append(file_result)
            used_tool_names.append(desired_file_tool_name)
            selected_file_tool_name = desired_file_tool_name
            file_route_status = "executed"

        output_parts = [result.output_text.strip() for result in results if result.output_text.strip()]
        artifacts = tuple(
            result.artifact
            for result in results[1:]
            if result.artifact is not None
        )
        warnings = tuple(
            dict.fromkeys(warning for result in results for warning in result.warnings)
        )
        route_fallback = file_route_status in {
            "not_allowed",
            "unavailable",
            "scope_projection_failed",
        }
        trace = {
            "classifier": "deterministic_keyword_pipeline",
            "decision": decision.reason_summary,
            "matched_verbs": decision.matched_verb_keywords,
            "matched_file_keywords": decision.matched_file_keywords,
            "matched_file_types": decision.matched_file_types,
            "selected_file_tool": selected_file_tool_name,
            "file_route_status": file_route_status,
            "executed_tools": used_tool_names,
            "answer_request_scope": composition_scope.answer_request_scope,
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
        )
