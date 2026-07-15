from __future__ import annotations

from dataclasses import replace
from itertools import product

import pytest

from assistant_rag.config import GeneralPurposeConfig
from assistant_rag.contracts import (
    ApprovedConversationContext,
    ChatRequest,
    ExpectedResponseType,
    GeneratedQuestion,
    GeneralSubBranch,
    Intent,
    LastQAState,
    PersistenceMode,
    PipelineContext,
    QuestionSource,
    ResponseType,
)
from assistant_rag.general_sub_branch import GeneralSubBranchDetector
from assistant_rag.prompts import DEFAULT_PROMPT_REGISTRY


class FailingLLM:
    def chat(self, **_kwargs: object) -> str:
        raise AssertionError("deterministic general sub-branch detection must not call the LLM")

    def generate_json(self, **_kwargs: object) -> dict[str, object]:
        raise AssertionError("deterministic general sub-branch detection must not call the LLM")


def _question() -> GeneratedQuestion:
    return GeneratedQuestion(
        text="Which environment?",
        source=QuestionSource.HUMAN_SUPPORTING_QUESTION,
        purpose="optional_context",
        confidence=1.0,
        expected_response_type=ExpectedResponseType.FREE_TEXT_ANSWER,
    )


def _last_qa(*, supporting_questions: bool = True) -> LastQAState:
    return LastQAState(
        last_user_query="Describe Project Atlas",
        last_response="Atlas needs QA signoff.",
        response_type=ResponseType.NORMAL,
        supporting_questions=[_question()] if supporting_questions else [],
        linked_topic_id="topic-linked",
        linked_hop_id="hop-linked",
    )


def _approved_context(
    *,
    history: bool = True,
    topic_candidates: tuple[str, ...] = ("topic-linked", "topic-later"),
    hop_candidates: tuple[str, ...] = ("hop-linked", "hop-later"),
) -> ApprovedConversationContext:
    return ApprovedConversationContext(
        approved_conversation_history=(
            [{"topic_id": "topic-linked", "hop_id": "hop-linked", "text": "Atlas context"}]
            if history
            else []
        ),
        human_supporting_questions=[],
        reminder_supporting_questions=[],
        clarification_question_context=None,
        extracted_expected_response_types=[],
        conversation_retrieval_ran=True,
        conversation_context_status="approved" if history else "empty",
        approved_conversation_count=1 if history else 0,
        _internal_selected_topic_candidates=list(topic_candidates),
        _internal_selected_hop_candidates=list(hop_candidates),
    )


def _context(
    *,
    last_qa_state: LastQAState | None = None,
    approved_context: ApprovedConversationContext | None = None,
) -> PipelineContext:
    return PipelineContext(
        request=ChatRequest(user_id="test-user", raw_query="Continue Project Atlas"),
        rewritten_query="Continue Project Atlas",
        last_qa_state=last_qa_state,
        conversation_results=[],
        intent=Intent.GENERAL_RESPONSE,
        approved_conversation_context=approved_context,
    )


def _detector() -> GeneralSubBranchDetector:
    return GeneralSubBranchDetector(
        llm=FailingLLM(),  # type: ignore[arg-type]
        prompt_registry=DEFAULT_PROMPT_REGISTRY,
    )


def _assert_unchanged_defaults(decision: object) -> None:
    assert getattr(decision, "selected_candidate_ref") is None
    assert getattr(decision, "risk_flags") == ()
    assert getattr(decision, "missing_context") == ()


@pytest.mark.parametrize(
    (
        "last_qa_exists",
        "supporting_questions_exist",
        "approved_context_exists",
        "hop_candidates_exist",
        "approved_history_exists",
    ),
    tuple(product((False, True), repeat=5)),
)
def test_complete_deterministic_rule_truth_table(
    last_qa_exists: bool,
    supporting_questions_exist: bool,
    approved_context_exists: bool,
    hop_candidates_exist: bool,
    approved_history_exists: bool,
) -> None:
    last_qa_state = (
        _last_qa(supporting_questions=supporting_questions_exist)
        if last_qa_exists
        else None
    )
    approved_context = (
        _approved_context(
            history=approved_history_exists,
            topic_candidates=("topic-first",) if hop_candidates_exist else (),
            hop_candidates=("hop-first",) if hop_candidates_exist else (),
        )
        if approved_context_exists
        else None
    )

    decision = _detector().detect(
        _context(
            last_qa_state=last_qa_state,
            approved_context=approved_context,
        ),
        GeneralPurposeConfig(),
    )

    support_rule = (
        last_qa_exists
        and supporting_questions_exist
        and approved_context_exists
        and hop_candidates_exist
    )
    if support_rule:
        expected_branch = GeneralSubBranch.SUPPORT_QUESTION_ANSWER
        expected_topic_id = "topic-linked"
        expected_hop_id = "hop-linked"
        expected_mode = PersistenceMode.APPEND_TO_EXISTING_TOPIC
    elif approved_context_exists and approved_history_exists:
        expected_branch = GeneralSubBranch.CONVERSATION_FOLLOW_UP
        expected_topic_id = "topic-first" if hop_candidates_exist else None
        expected_hop_id = "hop-first" if hop_candidates_exist else None
        expected_mode = PersistenceMode.APPEND_TO_EXISTING_TOPIC
    else:
        expected_branch = GeneralSubBranch.NEW_CONVERSATION_TOPIC
        expected_topic_id = None
        expected_hop_id = None
        expected_mode = PersistenceMode.CREATE_NEW_TOPIC

    assert decision.sub_branch is expected_branch
    assert decision.persistence_mode is expected_mode
    assert decision.selected_topic_id == expected_topic_id
    assert decision.selected_hop_id == expected_hop_id
    assert decision.confidence == 1.0
    assert decision.reason_summary == (
        f"Deterministic rule fired: {expected_branch.name}."
    )
    _assert_unchanged_defaults(decision)


def test_support_question_rule_has_precedence_and_uses_linked_last_qa_ids() -> None:
    decision = _detector().detect(
        _context(last_qa_state=_last_qa(), approved_context=_approved_context()),
        GeneralPurposeConfig(),
        merged_supporting_detail="This value must not affect deterministic routing.",
    )

    assert decision.sub_branch is GeneralSubBranch.SUPPORT_QUESTION_ANSWER
    assert decision.persistence_mode is PersistenceMode.APPEND_TO_EXISTING_TOPIC
    assert decision.selected_topic_id == "topic-linked"
    assert decision.selected_hop_id == "hop-linked"
    assert decision.selected_parent_hop_id == "hop-linked"
    assert decision.confidence == 1.0
    assert decision.reason_summary == "Deterministic rule fired: SUPPORT_QUESTION_ANSWER."
    _assert_unchanged_defaults(decision)


@pytest.mark.parametrize("missing_requirement", ("last_qa", "supporting_questions", "approved_context", "hop_candidates"))
def test_support_question_rule_requires_every_explicit_requirement(missing_requirement: str) -> None:
    last_qa_state = None if missing_requirement == "last_qa" else _last_qa(
        supporting_questions=missing_requirement != "supporting_questions"
    )
    approved_context = None if missing_requirement == "approved_context" else _approved_context(
        hop_candidates=() if missing_requirement == "hop_candidates" else ("hop-linked",)
    )

    decision = _detector().detect(
        _context(last_qa_state=last_qa_state, approved_context=approved_context),
        GeneralPurposeConfig(),
    )

    expected_sub_branch = (
        GeneralSubBranch.NEW_CONVERSATION_TOPIC
        if missing_requirement == "approved_context"
        else GeneralSubBranch.CONVERSATION_FOLLOW_UP
    )
    expected_persistence_mode = (
        PersistenceMode.CREATE_NEW_TOPIC
        if missing_requirement == "approved_context"
        else PersistenceMode.APPEND_TO_EXISTING_TOPIC
    )
    assert decision.sub_branch is expected_sub_branch
    assert decision.persistence_mode is expected_persistence_mode
    assert decision.confidence == 1.0
    assert decision.reason_summary == f"Deterministic rule fired: {expected_sub_branch.name}."
    _assert_unchanged_defaults(decision)


def test_conversation_follow_up_selects_first_approved_candidates() -> None:
    decision = _detector().detect(
        _context(
            approved_context=_approved_context(
                topic_candidates=("topic-first", "topic-second"),
                hop_candidates=("hop-first", "hop-second"),
            )
        ),
        GeneralPurposeConfig(),
    )

    assert decision.sub_branch is GeneralSubBranch.CONVERSATION_FOLLOW_UP
    assert decision.persistence_mode is PersistenceMode.APPEND_TO_EXISTING_TOPIC
    assert decision.selected_topic_id == "topic-first"
    assert decision.selected_hop_id == "hop-first"
    assert decision.selected_parent_hop_id == "hop-first"
    assert decision.confidence == 1.0
    assert decision.reason_summary == "Deterministic rule fired: CONVERSATION_FOLLOW_UP."
    _assert_unchanged_defaults(decision)


@pytest.mark.parametrize(
    "approved_context",
    (
        None,
        _approved_context(history=False),
    ),
)
def test_new_conversation_is_the_fallback(
    approved_context: ApprovedConversationContext | None,
) -> None:
    decision = _detector().detect(
        _context(approved_context=approved_context),
        GeneralPurposeConfig(),
    )

    assert decision.sub_branch is GeneralSubBranch.NEW_CONVERSATION_TOPIC
    assert decision.persistence_mode is PersistenceMode.CREATE_NEW_TOPIC
    assert decision.selected_topic_id is None
    assert decision.selected_hop_id is None
    assert decision.selected_parent_hop_id is None
    assert decision.confidence == 1.0
    assert decision.reason_summary == "Deterministic rule fired: NEW_CONVERSATION_TOPIC."
    _assert_unchanged_defaults(decision)


def test_disabled_guard_keeps_configured_fallback_path_and_never_calls_llm() -> None:
    config = replace(
        GeneralPurposeConfig(),
        general_sub_branch_detector_enabled=False,
        general_sub_branch_fallback_mode=GeneralSubBranch.CONVERSATION_FOLLOW_UP.value,
    )

    decision = _detector().detect(
        _context(last_qa_state=_last_qa(), approved_context=_approved_context()),
        config,
    )

    assert decision.sub_branch is GeneralSubBranch.CONVERSATION_FOLLOW_UP
    assert decision.persistence_mode is PersistenceMode.CREATE_NEW_TOPIC
    assert decision.selected_topic_id is None
    assert decision.selected_hop_id is None
    assert decision.selected_parent_hop_id is None
    assert decision.confidence == 1.0
    assert decision.reason_summary == "Detector disabled by config."
    _assert_unchanged_defaults(decision)
