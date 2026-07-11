import re
import json
from typing import Any

def _repair_json(text: str) -> str:
    # Remove trailing commas
    text = re.sub(r',\s*([}\]])', r'\1', text)
    # Fix python booleans/None
    text = re.sub(r'\bTrue\b', 'true', text)
    text = re.sub(r'\bFalse\b', 'false', text)
    text = re.sub(r'\bNone\b', 'null', text)
    return text

def advanced_parse_json(raw: str) -> dict[str, Any]:
    stripped = raw.strip()
    if not stripped:
        raise ValueError("Expected JSON object, got empty LLM response")
    
    # Remove markdown code blocks
    if "```" in stripped:
        parts = stripped.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                stripped = part
                break
                
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as exc:
        repaired = _repair_json(stripped)
        try:
            parsed = json.loads(repaired)
        except json.JSONDecodeError:
            extracted = _extract_json_object(stripped)
            if extracted is None:
                preview = stripped[:240].replace("\n", "\\n")
                raise ValueError(f"Expected JSON object, got non-JSON response: {preview!r}") from exc
            
            repaired_extracted = _repair_json(extracted)
            try:
                parsed = json.loads(repaired_extracted)
            except json.JSONDecodeError as extracted_exc:
                preview = extracted[:240].replace("\n", "\\n")
                raise ValueError(f"Extracted invalid JSON object from LLM response: {preview!r}") from extracted_exc
                
    if not isinstance(parsed, dict):
        if isinstance(parsed, str) and "{" in parsed:
            return advanced_parse_json(parsed)
        raise ValueError("Expected JSON object")
    return parsed

def _extract_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape = False
    for idx, char in enumerate(text[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:idx + 1]

    # If it didn't close, try to auto-close it
    if depth > 0:
        if in_string:
            return text[start:] + '"' + "}" * depth
        return text[start:] + "}" * depth

    return None
