from __future__ import annotations

from pathlib import Path
from typing import Any
import zipfile

import pytest

from assistant_rag.config import GeneralPurposeConfig
from assistant_rag.content_composer import (
    AnswerGenerationTool,
    ContentToolRegistry,
    DeterministicContentComposer,
    GenerateExcelTool,
    GeneratePDFTool,
    GeneratePPTXTool,
)
from assistant_rag.contracts import (
    BundledResponse,
    ChatRequest,
    ContentComposerInput,
    GeneralSubBranch,
    LastQAState,
    PersistenceMode,
    ResponseType,
    SubBranchPromptContext,
)
from assistant_rag.database import SQLiteRepository
from assistant_rag.llm import LLMTask
from assistant_rag.platform import PlatformSelector
from assistant_rag.prompts import DEFAULT_PROMPT_REGISTRY


class RecordingComposerLLM:
    def __init__(self, *, answer_text: str, file_plan: str) -> None:
        self.answer_text = answer_text
        self.file_plan = file_plan
        self.calls: list[dict[str, Any]] = []

    def chat(self, **kwargs: Any) -> str:
        self.calls.append(dict(kwargs))
        if kwargs["task"] is LLMTask.ANSWER:
            return self.answer_text
        if kwargs["task"] is LLMTask.WRITING:
            return self.file_plan
        raise AssertionError(f"Unexpected composer task: {kwargs['task']}")


class ScriptedPlatformLLM:
    def __init__(self, *, body: str, mode: str) -> None:
        self.body = body
        self.mode = mode
        self.calls: list[dict[str, Any]] = []

    def generate_json(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(dict(kwargs))
        return {
            "recipients": ["finance@example.com"],
            "subject": "Q3 handoff",
            "body": self.body,
            "mode": self.mode,
        }


class RecordingGmailSender:
    def __init__(self) -> None:
        self.calls: list[tuple[dict[str, Any], dict[str, Any]]] = []

    def send(
        self,
        payload: dict[str, Any],
        platform_context: dict[str, Any],
    ) -> dict[str, Any]:
        self.calls.append((dict(payload), dict(platform_context)))
        return {
            "status": "sent",
            "provider": "gmail",
            "recipient": payload["recipient"],
        }


def _composer_input(
    query: str,
    repository: SQLiteRepository,
) -> ContentComposerInput:
    return ContentComposerInput(
        user_id="artifact-integration-user",
        raw_user_query=query,
        rewritten_query=query,
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
        repository=repository,
    )


def _real_composer(
    tmp_path: Path,
    llm: RecordingComposerLLM,
) -> tuple[DeterministicContentComposer, GeneralPurposeConfig, SQLiteRepository]:
    repository = SQLiteRepository.persistent(
        str(tmp_path / "assistant.sqlite3"),
        enable_wal=False,
        busy_timeout_ms=5_000,
    )
    repository.initialize_schema()
    config = GeneralPurposeConfig(
        artifact_storage_dir=str(tmp_path / "artifacts"),
        artifact_download_base_url="/artifacts",
    )
    registry = ContentToolRegistry(
        tools=[
            AnswerGenerationTool(llm=llm, prompt_registry=DEFAULT_PROMPT_REGISTRY),
            GeneratePDFTool(llm=llm, prompt_registry=DEFAULT_PROMPT_REGISTRY),
            GenerateExcelTool(llm=llm, prompt_registry=DEFAULT_PROMPT_REGISTRY),
            GeneratePPTXTool(llm=llm, prompt_registry=DEFAULT_PROMPT_REGISTRY),
        ],
        config=config,
    )
    return DeterministicContentComposer(registry=registry), config, repository


def _assert_valid_artifact(path: Path, file_type: str) -> None:
    payload = path.read_bytes()
    if file_type == "pdf":
        assert payload.startswith(b"%PDF-1.4")
        assert payload.rstrip().endswith(b"%%EOF")
        return

    assert payload.startswith(b"PK")
    assert zipfile.is_zipfile(path)
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
        members = set(archive.namelist())
    if file_type == "xlsx":
        assert {
            "[Content_Types].xml",
            "_rels/.rels",
            "xl/workbook.xml",
            "xl/worksheets/sheet1.xml",
        } <= members
    elif file_type == "pptx":
        assert {
            "[Content_Types].xml",
            "_rels/.rels",
            "ppt/presentation.xml",
            "ppt/slides/slide1.xml",
        } <= members
    else:  # pragma: no cover - protects the parameterized test contract
        raise AssertionError(f"Unexpected file type: {file_type}")


@pytest.mark.parametrize(
    ("query", "expected_tool", "file_type"),
    (
        (
            "Create a PDF file with a concise workplace safety overview.",
            "generate_pdf",
            "pdf",
        ),
        (
            "Generate an Excel workbook with owner, forecast, and actual columns.",
            "generate_excel",
            "xlsx",
        ),
        (
            "Build a PowerPoint presentation with launch milestones and risks.",
            "generate_pptx",
            "pptx",
        ),
    ),
)
def test_real_artifact_matrix_runs_answer_first_and_materializes_one_valid_file(
    tmp_path: Path,
    query: str,
    expected_tool: str,
    file_type: str,
) -> None:
    answer_text = "I prepared the requested file and kept this handoff concise."
    llm = RecordingComposerLLM(
        answer_text=answer_text,
        file_plan="Title\nOwner,Forecast,Actual\nOperations,100,90",
    )
    composer, config, repository = _real_composer(tmp_path, llm)

    result = composer.compose(_composer_input(query, repository), config)

    assert [call["task"] for call in llm.calls] == [LLMTask.ANSWER, LLMTask.WRITING]
    assert result.used_tool_names == ("answer_generation", expected_tool)
    assert result.final_response_text.startswith(f"{answer_text}\n\nCreated ")
    assert len(result.artifacts) == 1

    artifact = result.artifacts[0]
    assert artifact["file_type"] == file_type
    assert artifact["status"] == "created"
    artifact_path = Path(artifact["storage_path"])
    assert artifact_path.parent == tmp_path / "artifacts" / "artifact-integration-user"
    assert artifact_path.suffix == f".{file_type}"
    _assert_valid_artifact(artifact_path, file_type)

    stored = repository.list_generated_artifacts(user_id="artifact-integration-user")
    assert len(stored) == 1
    assert stored[0]["artifact_id"] == artifact["artifact_id"]
    assert stored[0]["storage_path"] == str(artifact_path)
    assert repository.table_count("generated_artifacts") == 1
    repository.connection.close()


def test_compound_email_and_attachment_keep_real_outputs_disjoint(
    tmp_path: Path,
) -> None:
    query = (
        "Write an email to finance@example.com explaining the Q3 handoff, and attach "
        "an Excel budget tracker with owner, forecast, and actual columns."
    )
    email_copy = (
        "Subject: Q3 handoff\n\nDear Finance team,\n\n"
        "The Q3 tracker is attached for review.\n\nRegards,\nOperations"
    )
    workbook_plan = "Owner,Forecast,Actual\nOperations,100,90"
    llm = RecordingComposerLLM(answer_text=email_copy, file_plan=workbook_plan)
    composer, config, repository = _real_composer(tmp_path, llm)

    result = composer.compose(_composer_input(query, repository), config)

    assert result.used_tool_names == ("answer_generation", "generate_excel")
    assert [call["task"] for call in llm.calls] == [LLMTask.ANSWER, LLMTask.WRITING]
    answer_prompt = str(llm.calls[0]["user_prompt"])
    file_prompt = str(llm.calls[1]["user_prompt"])
    assert "Write an email to finance@example.com explaining the Q3 handoff" in answer_prompt
    assert "owner, forecast, and actual" not in answer_prompt
    assert "Excel budget tracker with owner, forecast, and actual columns" in file_prompt
    assert "finance@example.com" not in file_prompt
    assert "surrounding email" in file_prompt

    assert len(result.artifacts) == 1
    artifact_path = Path(result.artifacts[0]["storage_path"])
    with zipfile.ZipFile(artifact_path) as archive:
        sheet_xml = archive.read("xl/worksheets/sheet1.xml").decode("utf-8")
    assert all(value in sheet_xml for value in ("Owner", "Forecast", "Actual"))
    assert "Dear Finance" not in sheet_xml
    assert "Q3 handoff" not in sheet_xml
    assert result.final_response_text.startswith(email_copy)
    assert workbook_plan not in result.final_response_text
    assert repository.table_count("generated_artifacts") == 1

    draft_query = (
        "Draft an email about the Excel workbook to finance@example.com and "
        "operations@example.com; do not send it."
    )
    draft_bundled = BundledResponse(
        final_chat_text=result.final_response_text,
        response_type=ResponseType.NORMAL,
        last_qa_state=LastQAState(
            last_user_query=draft_query,
            last_response=result.final_response_text,
            response_type=ResponseType.NORMAL,
        ),
        platform_payload={"artifacts": list(result.artifacts)},
    )

    draft_sender = RecordingGmailSender()
    draft_llm = ScriptedPlatformLLM(body=email_copy, mode="draft")
    draft = PlatformSelector(
        llm=draft_llm,
        senders={"gmail": draft_sender},
    ).select(
        draft_bundled,
        ChatRequest(
            user_id="artifact-integration-user",
            raw_query="RAW_DELIVERY_SENTINEL must not control the draft",
        ),
    )

    assert draft["delivery"]["status"] == "draft_ready"
    assert draft["delivery"]["recipients"] == [
        "finance@example.com",
        "operations@example.com",
    ]
    assert draft["draft"]["body"] == email_copy
    assert len(draft["draft"]["attachments"]) == 1
    assert draft["draft"]["attachments"][0]["filename"] == result.artifacts[0]["filename"]
    assert "storage_path" not in draft["draft"]["attachments"][0]
    assert draft_sender.calls == []

    send_sender = RecordingGmailSender()
    send_llm = ScriptedPlatformLLM(body=email_copy, mode="send")
    send_query = (
        "Email the Excel workbook to finance@example.com and "
        "operations@example.com now."
    )
    send_bundled = BundledResponse(
        final_chat_text=result.final_response_text,
        response_type=ResponseType.NORMAL,
        last_qa_state=LastQAState(
            last_user_query=send_query,
            last_response=result.final_response_text,
            response_type=ResponseType.NORMAL,
        ),
        platform_payload={"artifacts": list(result.artifacts)},
    )
    sent = PlatformSelector(
        llm=send_llm,
        senders={"gmail": send_sender},
    ).select(
        send_bundled,
        ChatRequest(
            user_id="artifact-integration-user",
            raw_query="RAW_DELIVERY_SENTINEL must not authorize sending",
            platform_context={
                "gmail_username": "sender@example.com",
                "gmail_app_password": "test-app-password",
            },
        ),
    )

    assert sent["delivery"]["status"] == "sent"
    assert sent["draft"]["body"] == email_copy
    assert len(send_sender.calls) == 1
    sent_payload, sent_context = send_sender.calls[0]
    assert sent_payload["body"] == email_copy
    assert sent_payload["recipients"] == [
        "finance@example.com",
        "operations@example.com",
    ]
    assert len(sent_payload["attachments"]) == 1
    assert sent_payload["attachments"][0]["storage_path"] == str(artifact_path)
    assert sent_context == {
        "gmail_username": "sender@example.com",
        "gmail_app_password": "test-app-password",
    }
    repository.connection.close()
