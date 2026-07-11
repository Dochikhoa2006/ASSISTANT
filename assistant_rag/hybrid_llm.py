"""Hybrid LLM Client routing between Ollama and ONNX."""

from __future__ import annotations

import logging
from typing import Any
from .llm import LLMTask, OllamaLLMClient, uses_onnx_runtime
from .onnx_llm import ONNXLLMClient

logger = logging.getLogger(__name__)

class HybridLLMClient:
    """Routes LLM requests to either Ollama or ONNX engines depending on the model name."""
    def __init__(self, ollama_client: OllamaLLMClient, onnx_client: ONNXLLMClient):
        self.ollama_client = ollama_client
        self.onnx_client = onnx_client
        self.settings = ollama_client.settings
        self.router = ollama_client.router

    @property
    def last_error_by_task(self) -> dict[LLMTask, str]:
        errors: dict[LLMTask, str] = {}
        errors.update(self.ollama_client.last_error_by_task)
        errors.update(self.onnx_client.last_error_by_task)
        return errors

    def generate_json(
        self,
        *,
        task: LLMTask,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        model_name = self.ollama_client.router.model_for_task(task)
        if uses_onnx_runtime(model_name):
            payload = self.onnx_client.generate_json(
                task=task,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema=schema
            )
            onnx_error = self.onnx_client.last_error_by_task.get(task)
            if not onnx_error:
                return payload

            fallback_model = self.ollama_client._fallback_model_for_task(task)
            if not fallback_model or fallback_model == model_name or uses_onnx_runtime(fallback_model):
                return payload

            logger.debug(
                "ONNX structured generation failed for task %s; retrying once with configured Ollama fallback %s: %s",
                task.value,
                fallback_model,
                onnx_error,
            )
            fallback_payload = self.ollama_client.generate_json(
                task=task,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema=schema,
                model_override=fallback_model,
                fallback_for=model_name,
            )
            # A successful fallback is a recovered condition, not an active
            # user-facing error. The original failure remains in debug logs.
            if task not in self.ollama_client.last_error_by_task:
                self.onnx_client.last_error_by_task.pop(task, None)
            return fallback_payload
        return self.ollama_client.generate_json(
            task=task,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=schema
        )

    def chat(self, *, task: LLMTask, system_prompt: str, user_prompt: str) -> str:
        model_name = self.ollama_client.router.model_for_task(task)
        if uses_onnx_runtime(model_name):
            try:
                return self.onnx_client.chat(
                    task=task,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )
            except Exception:
                fallback_model = self.ollama_client._fallback_model_for_task(task)
                if not fallback_model or fallback_model == model_name or uses_onnx_runtime(fallback_model):
                    raise
                response = self.ollama_client._chat_raw(
                    task=task,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    format_schema=None,
                    model_override=fallback_model,
                    fallback_for=model_name,
                )
                self.onnx_client.last_error_by_task.pop(task, None)
                return response
        return self.ollama_client.chat(
            task=task,
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )
