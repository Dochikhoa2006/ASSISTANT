"""Generated artifact storage and simple file generators."""

from __future__ import annotations

import csv
import html
import json
import re
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import ArtifactFileType
from .metrics import GLOBAL_METRICS
from .repository import AssistantRepository


def safe_filename(name: str, extension: str) -> str:
    stem = Path(name).stem or "artifact"
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._") or "artifact"
    return f"{stem}.{extension}"


def _write_minimal_xlsx(path: Path, rows: list[list[str]]) -> None:
    rows = rows or [["Generated Content"], ["No structured rows were provided."]]
    sheet_rows = []
    for r_idx, row in enumerate(rows, start=1):
        cells = []
        for c_idx, value in enumerate(row, start=1):
            col = chr(ord("A") + c_idx - 1)
            cells.append(
                f'<c r="{col}{r_idx}" t="inlineStr"><is><t>{html.escape(str(value))}</t></is></c>'
            )
        sheet_rows.append(f'<row r="{r_idx}">{"".join(cells)}</row>')
    files = {
        "[Content_Types].xml": """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
</Types>""",
        "_rels/.rels": """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>""",
        "xl/workbook.xml": """<?xml version="1.0" encoding="UTF-8"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
<sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets></workbook>""",
        "xl/_rels/workbook.xml.rels": """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
</Relationships>""",
        "xl/worksheets/sheet1.xml": f"""<?xml version="1.0" encoding="UTF-8"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>{''.join(sheet_rows)}</sheetData></worksheet>""",
    }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)


def _write_minimal_pdf(path: Path, text: str) -> None:
    safe_text = re.sub(r"[()\\]", " ", text)[:3000] or "Generated document"
    stream = f"BT /F1 12 Tf 72 720 Td ({safe_text}) Tj ET"
    objects = [
        "1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj",
        "2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj",
        "3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj",
        "4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj",
        f"5 0 obj << /Length {len(stream)} >> stream\n{stream}\nendstream endobj",
    ]
    content = "%PDF-1.4\n"
    offsets = []
    for obj in objects:
        offsets.append(len(content.encode("latin-1")))
        content += obj + "\n"
    xref_at = len(content.encode("latin-1"))
    content += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n"
    for offset in offsets:
        content += f"{offset:010d} 00000 n \n"
    content += f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n"
    path.write_bytes(content.encode("latin-1", errors="replace"))


def _write_minimal_pptx(path: Path, title: str, bullets: list[str]) -> None:
    bullet_xml = "".join(
        f"<a:p><a:r><a:t>{html.escape(bullet)}</a:t></a:r></a:p>" for bullet in (bullets or ["Generated slide"])
    )
    files = {
        "[Content_Types].xml": """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>
<Override PartName="/ppt/slides/slide1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>
</Types>""",
        "_rels/.rels": """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
</Relationships>""",
        "ppt/presentation.xml": """<?xml version="1.0" encoding="UTF-8"?>
<p:presentation xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><p:sldIdLst><p:sldId id="256" r:id="rId1"/></p:sldIdLst></p:presentation>""",
        "ppt/_rels/presentation.xml.rels": """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide1.xml"/>
</Relationships>""",
        "ppt/slides/slide1.xml": f"""<?xml version="1.0" encoding="UTF-8"?>
<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><p:cSld><p:spTree><p:sp><p:txBody><a:bodyPr/><a:lstStyle/><a:p><a:r><a:t>{html.escape(title)}</a:t></a:r></a:p>{bullet_xml}</p:txBody></p:sp></p:spTree></p:cSld></p:sld>""",
    }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)


@dataclass
class ArtifactGenerator:
    repository: AssistantRepository
    storage_dir: str = "assistant_data/artifacts"
    download_base_url: str = "/artifacts"

    def generate(
        self,
        *,
        user_id: str,
        file_type: str,
        filename: str,
        content: str,
        conversation_hop_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            file_type = file_type.casefold()
            if file_type not in {item.value for item in ArtifactFileType}:
                raise ValueError("Unsupported artifact file type")
            base_name = safe_filename(filename, file_type)
            safe_name = f"{uuid.uuid4().hex[:12]}_{base_name}"
            user_dir = Path(self.storage_dir) / re.sub(r"[^A-Za-z0-9_.-]+", "_", user_id)
            user_dir.mkdir(parents=True, exist_ok=True)
            path = user_dir / safe_name
            if file_type == "xlsx":
                rows = [line.split(",") for line in content.splitlines() if line.strip()]
                _write_minimal_xlsx(path, rows)
            elif file_type == "pdf":
                _write_minimal_pdf(path, content)
            elif file_type == "pptx":
                lines = [line.strip("- ") for line in content.splitlines() if line.strip()]
                _write_minimal_pptx(path, lines[0] if lines else "Generated Presentation", lines[1:])
            elif file_type == "csv":
                with path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle)
                    for line in content.splitlines() or ["Generated Content"]:
                        writer.writerow([line])
            else:
                path.write_text(content or "Generated content", encoding="utf-8")
            if not path.exists():
                raise RuntimeError("Artifact file generation failed")
            storage_url = f"{self.download_base_url.rstrip('/')}/{safe_name}"
            artifact = self.repository.create_generated_artifact(
                user_id=user_id,
                conversation_hop_id=conversation_hop_id,
                file_type=file_type,
                filename=safe_name,
                storage_path=str(path),
                storage_url=storage_url,
                metadata=metadata or {"plan_summary": content[:500]},
            )
            GLOBAL_METRICS.increment("artifact_generation_success_total", file_type=file_type)
            return artifact
        except Exception:
            GLOBAL_METRICS.increment("artifact_generation_failure_total", file_type=str(file_type))
            raise
