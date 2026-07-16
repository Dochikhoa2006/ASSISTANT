from __future__ import annotations

from dataclasses import replace
from itertools import product

import pytest

from assistant_rag.branches import SUB_BRANCH_PROMPT_POLICIES
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
from assistant_rag.general_sub_branch import (
    GeneralPersistencePlanBuilder,
    GeneralSubBranchDetector,
    GeneralSubBranchValidator,
)
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
    top_score: float | None = 0.95,
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
        top_hop_rerank_score=top_score,
        _internal_selected_topic_candidates=list(topic_candidates),
        _internal_selected_hop_candidates=list(hop_candidates),
    )


def _context(
    *,
    last_qa_state: LastQAState | None = None,
    approved_context: ApprovedConversationContext | None = None,
    resolution_confidence: float = 1.0,
) -> PipelineContext:
    return PipelineContext(
        request=ChatRequest(user_id="test-user", raw_query="Continue Project Atlas"),
        rewritten_query="Continue Project Atlas",
        last_qa_state=last_qa_state,
        conversation_results=[],
        intent=Intent.GENERAL_RESPONSE,
        last_qa_trace={"resolution_confidence": resolution_confidence},
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


@pytest.mark.parametrize(
    ("resolution_confidence", "expected"),
    (
        (0.899, GeneralSubBranch.CONVERSATION_FOLLOW_UP),
        (0.90, GeneralSubBranch.SUPPORT_QUESTION_ANSWER),
        (1.0, GeneralSubBranch.SUPPORT_QUESTION_ANSWER),
    ),
)
def test_support_question_resolution_confidence_gate_is_inclusive(
    resolution_confidence: float,
    expected: GeneralSubBranch,
) -> None:
    decision = _detector().detect(
        _context(
            last_qa_state=_last_qa(),
            approved_context=_approved_context(),
            resolution_confidence=resolution_confidence,
        ),
        GeneralPurposeConfig(support_question_resolution_min_confidence=0.90),
    )

    assert decision.sub_branch is expected
    assert decision.confidence == 1.0


@pytest.mark.parametrize(
    ("top_score", "expected"),
    (
        (None, GeneralSubBranch.NEW_CONVERSATION_TOPIC),
        (0.649, GeneralSubBranch.NEW_CONVERSATION_TOPIC),
        (0.65, GeneralSubBranch.CONVERSATION_FOLLOW_UP),
        (0.99, GeneralSubBranch.CONVERSATION_FOLLOW_UP),
    ),
)
def test_conversation_follow_up_top_hop_score_gate_is_inclusive(
    top_score: float | None,
    expected: GeneralSubBranch,
) -> None:
    decision = _detector().detect(
        _context(approved_context=_approved_context(top_score=top_score)),
        GeneralPurposeConfig(conversation_followup_min_score=0.65),
    )

    assert decision.sub_branch is expected
    assert decision.confidence == 1.0


def test_support_validator_forces_new_topic_when_linked_ids_are_missing() -> None:
    state = _last_qa()
    state.linked_hop_id = None
    context = _context(last_qa_state=state, approved_context=_approved_context())
    unsafe = replace(
        _detector().detect(context, GeneralPurposeConfig()),
        sub_branch=GeneralSubBranch.SUPPORT_QUESTION_ANSWER,
        persistence_mode=PersistenceMode.APPEND_TO_EXISTING_TOPIC,
        selected_topic_id="topic-linked",
        selected_hop_id="hop-from-decision",
    )

    decision = GeneralSubBranchValidator().validate(
        unsafe, context, GeneralPurposeConfig()
    )

    assert decision.sub_branch is GeneralSubBranch.NEW_CONVERSATION_TOPIC
    assert decision.persistence_mode is PersistenceMode.CREATE_NEW_TOPIC
    assert decision.selected_topic_id is None
    assert decision.selected_hop_id is None


def test_follow_up_validator_and_plan_enforce_candidate_identity() -> None:
    context = _context(approved_context=_approved_context())
    unsafe = replace(
        _detector().detect(context, GeneralPurposeConfig()),
        selected_hop_id=None,
    )
    validated = GeneralSubBranchValidator().validate(
        unsafe, context, GeneralPurposeConfig()
    )
    plan = GeneralPersistencePlanBuilder().build_plan(
        validated, context, GeneralPurposeConfig()
    )

    assert validated.sub_branch is GeneralSubBranch.NEW_CONVERSATION_TOPIC
    assert plan.sub_branch is GeneralSubBranch.NEW_CONVERSATION_TOPIC
    assert plan.persistence_mode is PersistenceMode.CREATE_NEW_TOPIC
    assert plan.topic_id is None
    assert plan.previous_hop_id is None


def test_sub_branch_prompt_policies_match_the_persistence_contract() -> None:
    assert SUB_BRANCH_PROMPT_POLICIES[GeneralSubBranch.SUPPORT_QUESTION_ANSWER][
        "chat_history_role"
    ] == "User is answering a prior supporting question."
    assert SUB_BRANCH_PROMPT_POLICIES[GeneralSubBranch.SUPPORT_QUESTION_ANSWER][
        "response_goal"
    ] == "Enhance/refine prior answer using current reply."
    assert SUB_BRANCH_PROMPT_POLICIES[GeneralSubBranch.CONVERSATION_FOLLOW_UP][
        "chat_history_role"
    ] == "User is continuing an approved existing conversation."
    assert SUB_BRANCH_PROMPT_POLICIES[GeneralSubBranch.CONVERSATION_FOLLOW_UP][
        "response_goal"
    ] == "Use chat history to stay on topic."
    assert SUB_BRANCH_PROMPT_POLICIES[GeneralSubBranch.NEW_CONVERSATION_TOPIC][
        "chat_history_role"
    ] == "User is starting fresh."
    assert SUB_BRANCH_PROMPT_POLICIES[GeneralSubBranch.NEW_CONVERSATION_TOPIC][
        "response_goal"
    ] == "Answer directly without forcing old context."
