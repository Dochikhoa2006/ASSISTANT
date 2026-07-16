from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from assistant_rag.contracts import (
    BundledResponse,
    ChatRequest,
    LastQAState,
    ResponseType,
)
from assistant_rag.request_lifecycle import (
    ChatRequestExecution,
    bundled_response_payload,
)
import streamlit_app


@dataclass
class DownloadCapture:
    captions: list[str] = field(default_factory=list)
    downloads: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def caption(self, message: object) -> None:
        self.captions.append(str(message))

    def download_button(self, **kwargs: Any) -> None:
        self.downloads.append(dict(kwargs))

    def error(self, message: object) -> None:
        self.errors.append(str(message))


def _artifacts(tmp_path: Path) -> list[dict[str, object]]:
    definitions = (
        (
            "artifact-pdf",
            "quarterly-report.pdf",
            b"%PDF-1.4\nstreamlit artifact test\n%%EOF\n",
        ),
        (
            "artifact-xlsx",
            "budget-workbook.xlsx",
            b"PK\x03\x04xlsx-streamlit-artifact-test",
        ),
        (
            "artifact-pptx",
            "launch-presentation.pptx",
            b"PK\x03\x04pptx-streamlit-artifact-test",
        ),
    )
    artifacts: list[dict[str, object]] = []
    for artifact_id, filename, payload in definitions:
        path = tmp_path / filename
        path.write_bytes(payload)
        artifacts.append(
            {
                "artifact_id": artifact_id,
                "filename": filename,
                "storage_path": str(path),
                "file_type": path.suffix.removeprefix("."),
            }
        )
    return artifacts


def _response(artifacts: list[dict[str, object]]) -> BundledResponse:
    text = "The requested Microsoft files are ready."
    return BundledResponse(
        final_chat_text=text,
        response_type=ResponseType.NORMAL,
        last_qa_state=LastQAState(
            last_user_query="Create the requested files.",
            last_response=text,
            response_type=ResponseType.NORMAL,
        ),
        platform_payload={"artifacts": artifacts},
    )


def _execution(
    response: BundledResponse,
    *,
    replayed: bool,
) -> ChatRequestExecution:
    request = ChatRequest(
        user_id="streamlit-artifact-user",
        raw_query="Create the requested Microsoft file.",
        idempotency_key="streamlit-artifact-key",
    )
    return ChatRequestExecution(
        request=request,
        request_id="artifact-request-id",
        response=None if replayed else response,
        payload=bundled_response_payload(response) if replayed else {},
        replayed=replayed,
        is_mutation=False,
    )


def test_render_artifact_downloads_preserves_exact_files_mimes_and_stable_keys(
    tmp_path: Path,
) -> None:
    artifacts = _artifacts(tmp_path)
    first_render = DownloadCapture()
    second_render = DownloadCapture()

    streamlit_app._render_artifact_downloads(first_render, artifacts)
    streamlit_app._render_artifact_downloads(second_render, artifacts)

    assert first_render.captions == ["Generated files"]
    assert first_render.errors == []
    assert [item["label"] for item in first_render.downloads] == [
        "Download quarterly-report.pdf",
        "Download budget-workbook.xlsx",
        "Download launch-presentation.pptx",
    ]
    assert [item["data"] for item in first_render.downloads] == [
        Path(str(artifact["storage_path"])).read_bytes() for artifact in artifacts
    ]
    assert [item["file_name"] for item in first_render.downloads] == [
        "quarterly-report.pdf",
        "budget-workbook.xlsx",
        "launch-presentation.pptx",
    ]
    assert [item["mime"] for item in first_render.downloads] == [
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ]
    expected_keys = [
        "artifact-download-artifact-pdf",
        "artifact-download-artifact-xlsx",
        "artifact-download-artifact-pptx",
    ]
    assert [item["key"] for item in first_render.downloads] == expected_keys
    assert [item["key"] for item in second_render.downloads] == expected_keys
    assert len(set(expected_keys)) == len(expected_keys)


@pytest.mark.parametrize("replayed", (False, True), ids=("fresh", "idempotent-replay"))
def test_chat_message_retains_and_renders_every_artifact_for_fresh_and_replay(
    tmp_path: Path,
    replayed: bool,
) -> None:
    artifacts = _artifacts(tmp_path)
    execution = _execution(_response(artifacts), replayed=replayed)

    message = streamlit_app._chat_message_from_execution(execution)

    assert message["role"] == "assistant"
    assert message["content"] == "The requested Microsoft files are ready."
    assert message["response_type"] == ResponseType.NORMAL.value
    assert message["artifacts"] == artifacts

    rendered = DownloadCapture()
    streamlit_app._render_artifact_downloads(
        rendered,
        list(message["artifacts"]),
    )
    assert [item["file_name"] for item in rendered.downloads] == [
        "quarterly-report.pdf",
        "budget-workbook.xlsx",
        "launch-presentation.pptx",
    ]
    assert rendered.errors == []


def test_unavailable_artifact_is_reported_without_hiding_valid_download(
    tmp_path: Path,
) -> None:
    valid = _artifacts(tmp_path)[0]
    missing = {
        "artifact_id": "artifact-missing",
        "filename": "missing-workbook.xlsx",
        "storage_path": str(tmp_path / "missing-workbook.xlsx"),
    }
    rendered = DownloadCapture()

    streamlit_app._render_artifact_downloads(rendered, [missing, valid])

    assert [item["file_name"] for item in rendered.downloads] == [
        "quarterly-report.pdf"
    ]
    assert len(rendered.errors) == 1
    assert "missing-workbook.xlsx" in rendered.errors[0]


@pytest.mark.parametrize(
    "artifact",
    (
        {},
        {"artifact_id": "missing-filename", "storage_path": "unused.pdf"},
        {"artifact_id": "missing-path", "filename": "unavailable.pdf"},
    ),
    ids=("empty-metadata", "missing-filename", "missing-storage-path"),
)
def test_invalid_artifact_metadata_is_visibly_reported(
    artifact: dict[str, object],
) -> None:
    rendered = DownloadCapture()

    streamlit_app._render_artifact_downloads(rendered, [artifact])

    assert rendered.downloads == []
    assert len(rendered.errors) == 1
    assert "unavailable" in rendered.errors[0].casefold()
