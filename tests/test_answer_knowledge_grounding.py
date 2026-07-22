from __future__ import annotations

from contextlib import contextmanager
import json
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

import assistant_rag.branches as branches_module
from assistant_rag.answer_grounding import generate_answer
from assistant_rag.branches import GeneralResponseBranch
from assistant_rag.config import GeneralPurposeConfig
from assistant_rag.content_composer import AnswerGenerationTool, GeneratePPTXTool
from assistant_rag.context_filter import ApprovedContext
from assistant_rag.contracts import (
    ChatRequest,
    ContentComposerInput,
    ContentComposerResult,
    GeneralSubBranch,
    HopWrite,
    Intent,
    PersistenceMode,
    PipelineContext,
    ResponseType,
    SubBranchPromptContext,
)
from assistant_rag.llm import LLMTask
from assistant_rag.prompts import DEFAULT_PROMPT_REGISTRY, PromptContext
from assistant_rag.semantic_actions import grounded_semantic_action_from_payload


EVIDENCE = {
    "candidate_key": "owned-sql-candidate",
    "text": "Project Juniper uses the copper release channel.",
    "version": 3,
    "validation_status": "sql_validated",
}


class _StructuredLLM:
    def __init__(self, payload: dict[str, Any], *, enforce_invariant: bool = True):
        self.payload = payload
        self.enforce_invariant = enforce_invariant
        self.calls: list[dict[str, Any]] = []

    def generate_json(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(dict(kwargs))
        if self.enforce_invariant:
            kwargs["invariant_validator"](self.payload)
        return dict(self.payload)


def _prompt_context() -> PromptContext:
    return PromptContext(
        stage="answer_generation",
        user_id="grounding-user",
        rewritten_query="Which release channel does Juniper use?",
        extra={"approved_knowledge_records": [EVIDENCE]},
    )


def test_grounded_answer_accepts_only_supplied_candidate_and_verbatim_support() -> None:
    llm = _StructuredLLM(
        {
            "answer_text": EVIDENCE["text"],
            "evidence_references": [
                {
                    "candidate_key": EVIDENCE["candidate_key"],
                    "verbatim_support": "copper release channel",
                }
            ],
        }
    )

    outcome = generate_answer(
        llm=llm,  # type: ignore[arg-type]
        prompt_registry=DEFAULT_PROMPT_REGISTRY,
        prompt_context=_prompt_context(),
        approved_knowledge_records=[EVIDENCE],
    )

    assert outcome.model_succeeded is True
    assert outcome.text == EVIDENCE["text"]
    assert outcome.grounded_candidate_keys == (EVIDENCE["candidate_key"],)
    prompt = json.loads(llm.calls[0]["user_prompt"].removeprefix("Runtime context:\n"))
    assert prompt["extra"]["approved_knowledge_records"] == [EVIDENCE]


def test_non_grounded_answer_is_replaced_by_exact_approved_sql_evidence() -> None:
    llm = _StructuredLLM(
        {
            "answer_text": "I do not have enough context to provide that result.",
            "evidence_references": [],
        }
    )

    outcome = generate_answer(
        llm=llm,  # type: ignore[arg-type]
        prompt_registry=DEFAULT_PROMPT_REGISTRY,
        prompt_context=_prompt_context(),
        approved_knowledge_records=[EVIDENCE],
    )

    assert outcome.model_succeeded is False
    assert outcome.fallback_used is True
    assert outcome.text == EVIDENCE["text"]


def test_invented_evidence_reference_cannot_authorize_answer() -> None:
    llm = _StructuredLLM(
        {
            "answer_text": "Juniper uses the copper release channel.",
            "evidence_references": [
                {
                    "candidate_key": "another-users-candidate",
                    "verbatim_support": "copper release channel",
                }
            ],
        }
    )

    outcome = generate_answer(
        llm=llm,  # type: ignore[arg-type]
        prompt_registry=DEFAULT_PROMPT_REGISTRY,
        prompt_context=_prompt_context(),
        approved_knowledge_records=[EVIDENCE],
    )

    assert outcome.model_succeeded is False
    assert outcome.text == EVIDENCE["text"]


def test_grounding_rejects_negated_claim_even_when_full_evidence_is_quoted() -> None:
    llm = _StructuredLLM(
        {
            "answer_text": f"It is false that {EVIDENCE['text']}",
            "evidence_references": [
                {
                    "candidate_key": EVIDENCE["candidate_key"],
                    "verbatim_support": EVIDENCE["text"],
                }
            ],
        }
    )

    outcome = generate_answer(
        llm=llm,  # type: ignore[arg-type]
        prompt_registry=DEFAULT_PROMPT_REGISTRY,
        prompt_context=_prompt_context(),
        approved_knowledge_records=[EVIDENCE],
    )

    assert outcome.model_succeeded is False
    assert outcome.fallback_used is True
    assert outcome.text == EVIDENCE["text"]


def test_grounding_rejects_refusal_that_merely_appends_the_evidence_quote() -> None:
    llm = _StructuredLLM(
        {
            "answer_text": (
                "I cannot answer the question. The available record says: "
                f"{EVIDENCE['text']}"
            ),
            "evidence_references": [
                {
                    "candidate_key": EVIDENCE["candidate_key"],
                    "verbatim_support": EVIDENCE["text"],
                }
            ],
        }
    )

    outcome = generate_answer(
        llm=llm,  # type: ignore[arg-type]
        prompt_registry=DEFAULT_PROMPT_REGISTRY,
        prompt_context=_prompt_context(),
        approved_knowledge_records=[EVIDENCE],
    )

    assert outcome.model_succeeded is False
    assert outcome.fallback_used is True
    assert outcome.text == EVIDENCE["text"]


def test_grounding_rejects_an_extra_personal_claim_without_approved_support() -> None:
    llm = _StructuredLLM(
        {
            "answer_text": (
                f"{EVIDENCE['text']} "
                "The user's private access tier is platinum."
            ),
            "evidence_references": [
                {
                    "candidate_key": EVIDENCE["candidate_key"],
                    "verbatim_support": EVIDENCE["text"],
                }
            ],
        }
    )

    outcome = generate_answer(
        llm=llm,  # type: ignore[arg-type]
        prompt_registry=DEFAULT_PROMPT_REGISTRY,
        prompt_context=_prompt_context(),
        approved_knowledge_records=[EVIDENCE],
    )

    assert outcome.model_succeeded is False
    assert outcome.fallback_used is True
    assert outcome.text == EVIDENCE["text"]


def test_grounding_cannot_invert_a_negative_record_by_extracting_its_inner_clause() -> None:
    negative_record = {
        **EVIDENCE,
        "text": (
            "It is false that Project Juniper uses the copper release channel."
        ),
    }
    llm = _StructuredLLM(
        {
            "answer_text": "Project Juniper uses the copper release channel.",
            "evidence_references": [
                {
                    "candidate_key": negative_record["candidate_key"],
                    "verbatim_support": (
                        "Project Juniper uses the copper release channel"
                    ),
                }
            ],
        }
    )

    outcome = generate_answer(
        llm=llm,  # type: ignore[arg-type]
        prompt_registry=DEFAULT_PROMPT_REGISTRY,
        prompt_context=replace(
            _prompt_context(),
            extra={"approved_knowledge_records": [negative_record]},
        ),
        approved_knowledge_records=[negative_record],
    )

    assert outcome.model_succeeded is False
    assert outcome.fallback_used is True
    assert outcome.text == negative_record["text"]


def test_terminal_fallback_selects_the_query_relevant_approved_record_only() -> None:
    relevant = {**EVIDENCE, "rerank_score": 0.97}
    unrelated = {
        "candidate_key": "owned-unrelated-candidate",
        "text": "The bicycle inventory is audited on Thursdays.",
        "version": 2,
        "validation_status": "sql_validated",
        "rerank_score": 0.21,
    }
    llm = _StructuredLLM(
        {
            "answer_text": "Unsupported output.",
            "evidence_references": [],
        }
    )
    prompt_context = replace(
        _prompt_context(),
        extra={"approved_knowledge_records": [unrelated, relevant]},
    )

    outcome = generate_answer(
        llm=llm,  # type: ignore[arg-type]
        prompt_registry=DEFAULT_PROMPT_REGISTRY,
        prompt_context=prompt_context,
        approved_knowledge_records=[unrelated, relevant],
    )

    assert outcome.model_succeeded is False
    assert outcome.fallback_used is True
    assert outcome.text == relevant["text"]
    assert outcome.grounded_candidate_keys == (relevant["candidate_key"],)


def _composer_input() -> ContentComposerInput:
    return ContentComposerInput(
        user_id="grounding-user",
        raw_user_query="Which release channel does Juniper use?",
        rewritten_query="Which release channel does Juniper use?",
        sub_branch=GeneralSubBranch.NEW_CONVERSATION_TOPIC,
        persistence_mode=PersistenceMode.CREATE_NEW_TOPIC,
        approved_conversation_history=[],
        human_supporting_questions=[],
        reminder_supporting_questions=[],
        extracted_expected_response_types=[],
        approved_knowledge_evidence=[EVIDENCE["text"]],
        approved_reminder_context=[],
        metadata={},
        platform_context={},
        sub_branch_prompt_context=SubBranchPromptContext(
            sub_branch=GeneralSubBranch.NEW_CONVERSATION_TOPIC,
            persistence_mode=PersistenceMode.CREATE_NEW_TOPIC,
            chat_history_role="new request",
            response_goal="answer directly",
            database_update_mode="append audit hop",
            allowed_database_updates=("conversation_hop_append",),
            prohibited_database_updates=("knowledge_mutation",),
        ),
        sub_branch_supporting_prompt="Answer directly.",
        approved_knowledge_records=[EVIDENCE],
    )


def test_answer_tool_preserves_grounded_fallback_instead_of_generic_unavailable_text() -> None:
    llm = _StructuredLLM(
        {
            "answer_text": "Unrelated output.",
            "evidence_references": [],
        }
    )

    result = AnswerGenerationTool(
        llm=llm,  # type: ignore[arg-type]
        prompt_registry=DEFAULT_PROMPT_REGISTRY,
    ).execute(_composer_input(), object())  # type: ignore[arg-type]

    assert result.output_text == EVIDENCE["text"]
    assert result.fallback_used is True
    assert result.warnings == ("answer_grounded_evidence_fallback",)
    assert llm.calls[0]["task"] is LLMTask.ANSWER


def test_file_writer_receives_the_same_approved_knowledge_records() -> None:
    query = (
        "Create a PowerPoint presentation using my saved Project Juniper "
        "release details."
    )
    semantic = grounded_semantic_action_from_payload(
        {
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
                "confidence": 0.99,
            },
            "file": {
                "operation": "create",
                "file_type": "pptx",
                "authorization_evidence": ["Create"],
                "type_evidence": ["PowerPoint presentation"],
                "confidence": 0.99,
            },
            "reason_summary": "test fixture",
        },
        canonical_query=query,
    )

    class _WriterLLM:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def chat(self, **kwargs: Any) -> str:
            self.calls.append(dict(kwargs))
            return "Presentation plan grounded in the approved details."

    llm = _WriterLLM()
    composer_input = replace(
        _composer_input(),
        raw_user_query=query,
        rewritten_query=query,
        metadata={"semantic_action_decision": semantic.to_payload()},
    )
    result = GeneratePPTXTool(
        llm=llm,  # type: ignore[arg-type]
        prompt_registry=DEFAULT_PROMPT_REGISTRY,
        config=GeneralPurposeConfig(),
    ).execute(composer_input, GeneralPurposeConfig())

    assert result.fallback_used is False
    prompt = json.loads(
        llm.calls[0]["user_prompt"].removeprefix("Runtime context:\n")
    )
    assert prompt["extra"]["approved_knowledge_records"] == [EVIDENCE]


def test_injected_legacy_composer_cannot_bypass_branch_grounding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(branches_module, "retrieve_knowledge", lambda **_kwargs: [])
    monkeypatch.setattr(
        branches_module,
        "retrieve_reminder_candidates",
        lambda **_kwargs: [],
    )

    class _ApprovedKnowledgeFilter:
        def filter(self, **_kwargs: Any) -> ApprovedContext:
            return ApprovedContext(
                knowledge_evidence=[EVIDENCE["text"]],
                reminder_context=[],
                approved_conversation_history=[],
                rejected_knowledge_ids=[],
                rejected_reminder_ids=[],
                rejected_conversation_ids=[],
                knowledge_records=[EVIDENCE],
            )

    class _LegacyComposer:
        def compose(self, *_args: Any, **_kwargs: Any) -> ContentComposerResult:
            return ContentComposerResult(
                final_response_text=(
                    f"{EVIDENCE['text']}\n\n"
                    "The user's private access tier is platinum."
                ),
                tool_trace_summary="legacy trace without grounded stage outcomes",
                used_tool_names=("answer_generation",),
                confidence=1.0,
                fallback_used=False,
                reason_summary="legacy composer self-reported success",
                content_warnings=(),
                answer_response_text=EVIDENCE["text"],
            )

    class _Repository:
        @contextmanager
        def transaction(self):  # type: ignore[no-untyped-def]
            yield object()

        def ensure_topic(
            self,
            _cursor: object,
            *,
            user_id: str,
            title: str,
        ) -> str:
            assert user_id == "grounding-user"
            assert title
            return "grounded-topic"

        def append_conversation_hop(
            self,
            _cursor: object,
            **kwargs: Any,
        ) -> HopWrite:
            return HopWrite(
                topic_id=str(kwargs["topic_id"]),
                hop_id="grounded-hop",
                previous_hop_id=kwargs.get("parent_hop_id"),
                outbox_job_id="grounded-outbox",
            )

    llm = _StructuredLLM(
        {
            "answer_text": EVIDENCE["text"],
            "evidence_references": [
                {
                    "candidate_key": EVIDENCE["candidate_key"],
                    "verbatim_support": EVIDENCE["text"],
                }
            ],
        }
    )
    query = "Which release channel does Juniper use?"
    context = PipelineContext(
        request=ChatRequest(user_id="grounding-user", raw_query=query),
        rewritten_query=query,
        last_qa_state=None,
        conversation_results=[],
        intent=Intent.GENERAL_RESPONSE,
    )
    branch = GeneralResponseBranch(
        retriever=object(),  # type: ignore[arg-type]
        config=SimpleNamespace(
            retrieval=SimpleNamespace(
                general_response_reminder_statuses=("scheduled", "notified"),
                general_response_reminder_limit=4,
            )
        ),  # type: ignore[arg-type]
        context_filter=_ApprovedKnowledgeFilter(),  # type: ignore[arg-type]
        content_composer=_LegacyComposer(),
        general_purpose_config=GeneralPurposeConfig(),
        llm=llm,  # type: ignore[arg-type]
    )

    result = branch.execute(context, _Repository())  # type: ignore[arg-type]

    assert result.response_type is ResponseType.NORMAL
    assert result.normal_response_text == EVIDENCE["text"]
    assert len(llm.calls) == 0
    assert "answer_generation_recovered" not in result.warnings
