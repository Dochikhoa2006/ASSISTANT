"""Response Bundler.

All branch outputs pass through this module before Chat Output.
"""

from __future__ import annotations

from typing import Any

from .contracts import BranchResult, BundledResponse, ChatRequest, GeneratedQuestion, LastQAState, ResponseType
from .prompts import DEFAULT_PROMPT_REGISTRY, PromptRegistry


def _question_text(question: GeneratedQuestion | dict[str, Any] | str | Any) -> str:
    if isinstance(question, GeneratedQuestion):
        return question.text.strip()
    if isinstance(question, dict):
        return str(question.get("text") or question.get("question_text") or "").strip()
    return str(getattr(question, "text", question) or "").strip()


def _question_line(label: str, question: GeneratedQuestion | dict[str, Any] | str | Any) -> str:
    text = _question_text(question)
    return f"{label}: {text}" if text else ""


def _append_unique_line(parts: list[str], seen: set[str], line: str) -> None:
    normalized = " ".join(line.casefold().split())
    if not normalized or normalized in seen:
        return
    seen.add(normalized)
    parts.append(line)


class ResponseBundler:
    def __init__(self, prompt_registry: PromptRegistry = DEFAULT_PROMPT_REGISTRY) -> None:
        self.prompt_registry = prompt_registry

    def bundle(
        self,
        *,
        request: ChatRequest,
        rewritten_query: str,
        branch_result: BranchResult,
    ) -> BundledResponse:
        parts: list[str] = []
        seen_parts: set[str] = set()
        operation_summaries = [
            getattr(result, "user_safe_summary", None) or getattr(result, "user_facing_summary", None)
            for result in (
                branch_result.knowledge_operation_results
                + branch_result.reminder_operation_results
            )
        ]
        for summary in operation_summaries:
            if summary:
                _append_unique_line(parts, seen_parts, str(summary).strip())
        if branch_result.normal_response_text and not operation_summaries:
            _append_unique_line(parts, seen_parts, branch_result.normal_response_text.strip())
        if branch_result.clarification_question:
            _append_unique_line(
                parts,
                seen_parts,
                _question_line("Clarification question", branch_result.clarification_question),
            )
        if branch_result.fallback_or_error_message:
            _append_unique_line(parts, seen_parts, branch_result.fallback_or_error_message.strip())
        for question in branch_result.human_supporting_questions:
            _append_unique_line(
                parts,
                seen_parts,
                _question_line("Supporting question", question),
            )
        if branch_result.reminder_supporting_question:
            _append_unique_line(
                parts,
                seen_parts,
                _question_line("Reminder supporting question", branch_result.reminder_supporting_question),
            )
            
        final_text = "\n".join(part.strip() for part in parts if part.strip())
        if not final_text:
            final_text = self.prompt_registry.message("bundler_empty")

        committed_actions = []
        for result in branch_result.knowledge_operation_results + branch_result.reminder_operation_results:
            if getattr(result, "status", None) == "committed":
                committed_actions.append(
                    {
                        "action_type": getattr(result, "action_type", None),
                        "domain_entity_type": getattr(result, "domain_entity_type", None),
                        "domain_entity_id": getattr(result, "domain_entity_id", None),
                        "summary": getattr(result, "user_safe_summary", None),
                    }
                )

        last_qa_state = LastQAState(
            last_user_query=rewritten_query or request.raw_query,
            last_response=final_text,
            response_type=branch_result.response_type,
            supporting_questions=list(branch_result.human_supporting_questions),
            clarification_question=branch_result.clarification_question,
            reminder_supporting_question=branch_result.reminder_supporting_question,
            linked_topic_id=branch_result.linked_topic_id,
            linked_hop_id=branch_result.linked_hop_id,
        )
        return BundledResponse(
            final_chat_text=final_text,
            response_type=branch_result.response_type,
            last_qa_state=last_qa_state,
            platform_payload=dict(branch_result.platform_payload),
            conversation_topic_id=branch_result.linked_topic_id,
            conversation_hop_id=branch_result.linked_hop_id,
            actions_committed=committed_actions,
            actions_pending_confirmation=list(branch_result.actions_pending_confirmation),
            warnings=list(request.metadata.get("warnings", [])) + list(branch_result.warnings),
            persistence_instructions={
                "audit_hop_id": branch_result.database_write_result.get("conversation_hop_id"),
                "outbox_job_ids": branch_result.indexing_job_result,
                "database_write_result": branch_result.database_write_result,
                "indexing_job_result": branch_result.indexing_job_result,
            },
        )


class ChatOutput:
    def emit(self, bundled_response: BundledResponse) -> str:
        return bundled_response.final_chat_text
