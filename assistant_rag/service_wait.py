"""Wait for runtime services needed by debug and Streamlit entrypoints."""

from __future__ import annotations

import json
import time
from urllib import error, request

from .llm import uses_onnx_runtime
from .settings import ProductionSettings


def main() -> int:
    settings = ProductionSettings.from_env()
    timeout_seconds = settings.service_wait.timeout_seconds
    opensearch_url = settings.opensearch.url.rstrip("/")
    chroma_host = settings.chroma.host
    chroma_port = settings.chroma.port
    if not chroma_host or chroma_port is None:
        raise RuntimeError("ChromaDB host and port must be configured for service wait.")
    ollama_url = settings.ollama.base_url.rstrip("/")

    wait_for_any(
        "OpenSearch",
        [opensearch_url],
        timeout_seconds=timeout_seconds,
        poll_interval_seconds=settings.service_wait.poll_interval_seconds,
        probe_timeout_seconds=settings.service_wait.probe_timeout_seconds,
    )
    wait_for_any(
        "ChromaDB",
        [
            f"http://{chroma_host}:{chroma_port}/api/v2/heartbeat",
            f"http://{chroma_host}:{chroma_port}/api/v1/heartbeat",
            f"http://{chroma_host}:{chroma_port}",
        ],
        timeout_seconds=timeout_seconds,
        poll_interval_seconds=settings.service_wait.poll_interval_seconds,
        probe_timeout_seconds=settings.service_wait.probe_timeout_seconds,
    )
    wait_for_any(
        "Ollama",
        [f"{ollama_url}/api/tags"],
        timeout_seconds=timeout_seconds,
        poll_interval_seconds=settings.service_wait.poll_interval_seconds,
        probe_timeout_seconds=settings.service_wait.probe_timeout_seconds,
    )

    maybe_pull_ollama_models(ollama_url, settings)
    print("Runtime services are reachable.")
    return 0


def wait_for_any(
    name: str,
    urls: list[str],
    *,
    timeout_seconds: int,
    poll_interval_seconds: int,
    probe_timeout_seconds: int,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        for url in urls:
            try:
                fetch_json(url, timeout_seconds=probe_timeout_seconds)
                print(f"{name} is reachable at {url}")
                return
            except Exception as exc:
                last_error = exc
        time.sleep(poll_interval_seconds)
    raise RuntimeError(f"{name} did not become reachable: {last_error}")


def maybe_pull_ollama_models(ollama_url: str, settings: ProductionSettings) -> None:
    models = configured_ollama_models(settings)
    installed = installed_ollama_models(
        ollama_url,
        timeout_seconds=settings.service_wait.probe_timeout_seconds,
    )
    missing = [model for model in models if model not in installed]
    if not missing:
        print("Configured Ollama models are already installed.")
        return
    if not settings.service_wait.auto_pull_ollama_models:
        print(
            "Ollama is running, but these configured models are not installed: "
            + ", ".join(missing)
        )
        print("Set ASSISTANT_AUTO_PULL_OLLAMA_MODELS=1 before ./run to pull them automatically.")
        return
    for model in missing:
        print(f"Pulling Ollama model: {model}")
        post_json(f"{ollama_url}/api/pull", {"name": model, "stream": False})


def configured_ollama_models(settings: ProductionSettings) -> list[str]:
    import dataclasses
    values: list[str | None] = []
    for field in dataclasses.fields(settings.ollama):
        if field.name.startswith("model_"):
            val = getattr(settings.ollama, field.name)
            # ONNX/Hugging Face models are loaded by ONNX Runtime, not Ollama.
            if val and not uses_onnx_runtime(val):
                values.append(val)

    unique: list[str] = []
    for value in values:
        if value and value not in unique:
            unique.append(value)
    return unique


def installed_ollama_models(ollama_url: str, *, timeout_seconds: int = 5) -> set[str]:
    payload = fetch_json(f"{ollama_url}/api/tags", timeout_seconds=timeout_seconds)
    return {str(model["name"]) for model in payload.get("models", [])}


def fetch_json(url: str, *, timeout_seconds: int = 5) -> dict[str, object]:
    req = request.Request(url, method="GET")
    try:
        with request.urlopen(req, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
    except error.URLError as exc:
        raise ConnectionError(f"GET {url} failed: {exc}") from exc
    return json.loads(body) if body.strip() else {}


def post_json(url: str, payload: dict[str, object]) -> dict[str, object]:
    req = request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=None) as response:
            body = response.read().decode("utf-8")
    except error.URLError as exc:
        raise ConnectionError(f"POST {url} failed: {exc}") from exc
    return json.loads(body) if body.strip() else {}


if __name__ == "__main__":
    raise SystemExit(main())
