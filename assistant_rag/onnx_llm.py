"""ONNX Runtime GenAI LLM integration."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import replace
from typing import Any

import onnxruntime_genai as og
from huggingface_hub import snapshot_download

from .llm import (
    LLMTask,
    ModelDecision,
    OllamaModelRouter,
    _structured_attempt_plan,
    _structured_attempt_prompt,
    llm_trace_stage_name_for_prompt,
    normalize_structured_output,
    parse_json_object,
    structured_fallback_payload,
    _log_structured_fallback,
    validate_json_schema,
)
from .metrics import GLOBAL_METRICS
from .observability import StageTimer, current_trace

logger = logging.getLogger(__name__)


def _runtime_context_from_prompt(user_prompt: str) -> dict[str, Any]:
    prefix = "Runtime context:\n"
    if not user_prompt.startswith(prefix):
        return {}
    try:
        payload = json.loads(user_prompt[len(prefix):])
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _adaptive_max_new_tokens(task: LLMTask, user_prompt: str, configured: int) -> int:
    if task is not LLMTask.ANSWER or configured <= 0:
        return configured

    context = _runtime_context_from_prompt(user_prompt)
    query = str(context.get("rewritten_query") or "").casefold()
    extra = context.get("extra") if isinstance(context.get("extra"), dict) else {}
    has_approved_context = bool(
        extra.get("approved_conversation_history")
        or extra.get("approved_knowledge_evidence")
        or extra.get("approved_reminder_context")
    )

    explicit_long_signals = (
        "exhaustive", "comprehensive", "very detailed", "full specification",
        "long-form", "write a long", "1000 words", "1500 words", "full code",
        "complete implementation",
    )
    if any(signal in query for signal in explicit_long_signals):
        return min(configured, 896)

    complex_signals = (
        "deep", "architecture", "implementation", "debug", "latency",
        "optimize", "tradeoff", "compare", "analysis", "analyze", "design",
        "migration", "security", "sql", "api", "code", "proof",
        "step by step solution",
    )
    if any(signal in query for signal in complex_signals):
        return min(configured, 640)

    if has_approved_context:
        return min(configured, 512)

    teaching_signals = (
        "explain", "example", "what is", "how does", "teach", "show me",
        "binary search", "python", "basics",
    )
    if any(signal in query for signal in teaching_signals):
        return min(configured, 320)

    if len(query) <= 180:
        return min(configured, 320)
    return min(configured, 448)


def _repetition_stop_reason(text: str) -> str | None:
    words = re.findall(r"\w+", text.casefold())
    if len(words) < 180:
        return None
    for size in (18, 14, 10):
        if len(words) < size * 3:
            continue
        tail = words[-size:]
        if words[-size * 2:-size] == tail and words[-size * 3:-size * 2] == tail:
            return f"repeated_{size}_word_tail"

    lines = [
        re.sub(r"\s+", " ", line.strip().casefold())
        for line in text.splitlines()
        if len(line.strip()) >= 35
    ]
    if len(lines) >= 4 and lines[-1] in lines[-4:-1]:
        return "repeated_line_tail"
    return None


def _sentence_boundary_stop_reason(text: str, generated_tokens: int, max_new_tokens: int) -> str | None:
    stripped = text.rstrip()
    if not stripped or stripped[-1] not in ".!?":
        return None
    if max_new_tokens <= 320:
        min_tokens = 140
    elif max_new_tokens <= 448:
        min_tokens = 220
    elif max_new_tokens <= 512:
        min_tokens = 280
    elif max_new_tokens <= 640:
        min_tokens = 360
    else:
        min_tokens = int(max_new_tokens * 0.70)
    if generated_tokens < min_tokens:
        return None
    return "complete_sentence_after_budget_floor"


def _is_non_retryable_model_error(exc: Exception) -> bool:
    """Return true for model-resolution failures a JSON retry cannot repair."""
    message = str(exc).casefold()
    return any(
        marker in message
        for marker in (
            "repository not found",
            "revision not found",
            "404 client error",
            "401 client error",
            "403 client error",
            "gated repo",
        )
    )


class ONNXLLMClient:
    def __init__(self, router: OllamaModelRouter, cache_dir: str = ".onnx_models", *, preload: bool = False):
        self.router = router
        self.cache_dir = cache_dir
        self._models: dict[str, og.Model] = {}
        self._tokenizers: dict[str, og.Tokenizer] = {}
        self.last_error_by_task: dict[LLMTask, str] = {}
        if preload:
            self.preload_models()

    def preload_models(self) -> list[str]:
        """Load every configured ONNX model into memory."""
        import dataclasses
        settings = self.router.settings
        loaded: list[str] = []
        for field in dataclasses.fields(settings):
            if field.name.startswith("model_"):
                model_name = getattr(settings, field.name)
                if model_name and "onnx" in model_name.lower():
                    self._get_model_and_tokenizer(model_name)
                    if model_name not in loaded:
                        loaded.append(model_name)
        return loaded

    def _get_model_and_tokenizer(self, model_name: str) -> tuple[og.Model, og.Tokenizer]:
        if model_name not in self._models:
            # ONNX model repos have multiple hardware variants. We only want the CPU variant for Mac.
            variant_folder = "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4"
            model_path = os.path.join(self.cache_dir, model_name.replace("/", "_"))
            exact_path = os.path.join(model_path, variant_folder)
            if not os.path.exists(os.path.join(exact_path, "model.onnx.data")):
                logger.info("Downloading ONNX model %s variant %s from Hugging Face...", model_name, variant_folder)
                model_path = snapshot_download(
                    model_name,
                    local_dir=model_path,
                    allow_patterns=[f"{variant_folder}/*"],
                )
                exact_path = os.path.join(model_path, variant_folder)
            # The Model class requires the exact directory containing the .onnx files.
            logger.info("Loading ONNX model into memory from %s", exact_path)
            self._models[model_name] = og.Model(exact_path)
            self._tokenizers[model_name] = og.Tokenizer(self._models[model_name])
        return self._models[model_name], self._tokenizers[model_name]

    def _generate(self, task: LLMTask, system_prompt: str, user_prompt: str, decision: ModelDecision) -> str:
        model, tokenizer = self._get_model_and_tokenizer(decision.model)

        messages = json.dumps(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            separators=(",", ":"),
        )
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        tokens = tokenizer.encode(prompt)
        params = og.GeneratorParams(model)

        configured_max_new_tokens = decision.num_predict or 512
        max_new_tokens = _adaptive_max_new_tokens(task, user_prompt, configured_max_new_tokens)
        max_length = len(tokens) + max_new_tokens
        if decision.num_ctx:
            if len(tokens) >= decision.num_ctx:
                raise ValueError(
                    f"ONNX prompt for task {task.value} has {len(tokens)} tokens, "
                    f"exceeding configured context window {decision.num_ctx}"
                )
            max_length = min(max_length, decision.num_ctx)

        params.set_search_options(
            max_length=max_length,
            temperature=decision.temperature,
            do_sample=decision.temperature > 0.0,
        )
        stream = tokenizer.create_stream()
        generated_text: list[str] = []

        with StageTimer(
            llm_trace_stage_name_for_prompt(
                task,
                user_prompt,
                engine="onnx",
            )
        ):
            generator = og.Generator(model, params)
            generator.append_tokens(tokens)
            stop_reason: str | None = None
            while not generator.is_done():
                generator.generate_next_token()
                next_tokens = generator.get_next_tokens()
                if len(next_tokens):
                    generated_text.append(stream.decode(int(next_tokens[0])))
                generated_count = generator.token_count() - len(tokens)
                if task is LLMTask.ANSWER and generated_count > 0 and generated_count % 24 == 0:
                    partial_text = "".join(generated_text)
                    stop_reason = _repetition_stop_reason(partial_text)
                    if not stop_reason:
                        stop_reason = _sentence_boundary_stop_reason(
                            partial_text,
                            generated_count,
                            max_new_tokens,
                        )
                    if stop_reason:
                        break
            trace = current_trace()
            if trace and trace._active_timers:
                timer = trace._active_timers[-1]
                timer.metadata["model"] = decision.model
                timer.metadata["input_count"] = len(tokens)
                timer.metadata["output_count"] = max(0, generator.token_count() - len(tokens))
                timer.metadata["engine"] = "onnxruntime_genai"
                timer.metadata["max_length"] = max_length
                timer.metadata["num_predict"] = max_new_tokens
                timer.metadata["configured_num_predict"] = configured_max_new_tokens
                if stop_reason:
                    timer.metadata["stop_reason"] = stop_reason

        return "".join(generated_text).strip()

    def chat(self, *, task: LLMTask, system_prompt: str, user_prompt: str) -> str:
        decision = self.router.decision_for_task(task)
        try:
            response = self._generate(task, system_prompt, user_prompt, decision)
        except Exception as exc:
            GLOBAL_METRICS.increment("llm_failures_total", task=task.value)
            self.last_error_by_task[task] = str(exc)
            raise
        self.last_error_by_task.pop(task, None)
        return response

    def generate_json(
        self,
        *,
        task: LLMTask,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any],
        model_override: str | None = None,
        fallback_for: str | None = None,
    ) -> dict[str, Any]:
        del fallback_for
        decision = self.router.decision_for_task(task)
        if model_override:
            decision = replace(
                decision,
                model=model_override,
                reason_summary=(
                    f"Explicit model override for {task.value}"
                ),
            )
        retry_count = getattr(self.router.settings, f"json_retry_count_{task.value}", self.router.settings.structured_retry_count)
        total_attempts = retry_count + 1
        attempt_plan = _structured_attempt_plan(total_attempts)
        attempt_errors: list[str] = []
        last_error: Exception | None = None

        for attempt_index, attempt_mode in enumerate(attempt_plan, start=1):
            try:
                attempt_prompt = _structured_attempt_prompt(
                    user_prompt=user_prompt,
                    schema=schema,
                    mode=attempt_mode,
                )
                response = self._generate(task, system_prompt, attempt_prompt, decision)
                payload = parse_json_object(response)
                payload, normalized = normalize_structured_output(payload, schema)
                validate_json_schema(payload, schema)
                trace = current_trace()
                if trace and trace._active_timers:
                    timer = trace._active_timers[-1]
                    timer.metadata["structured_attempts"] = attempt_index
                    timer.metadata["structured_configured_attempts"] = total_attempts
                    timer.metadata["attempt_mode"] = attempt_mode
                    if normalized:
                        timer.metadata["structured_output_normalized"] = True
                self.last_error_by_task.pop(task, None)
                return payload
            except Exception as exc:
                GLOBAL_METRICS.increment("llm_json_parse_failures_total", task=task.value)
                last_error = exc
                attempt_errors.append(f"attempt {attempt_index}/{len(attempt_plan)} [{attempt_mode}]: {type(exc).__name__}: {exc}")
                self.last_error_by_task[task] = attempt_errors[-1]
                logger.debug(
                    "ONNX JSON generation failed on attempt %d/%d for task %s (mode: %s): %s",
                    attempt_index,
                    len(attempt_plan),
                    task.value,
                    attempt_mode,
                    exc,
                )
                if _is_non_retryable_model_error(exc):
                    logger.debug(
                        "Stopping ONNX JSON retries for task %s because model resolution cannot succeed: %s",
                        task.value,
                        exc,
                    )
                    break

        GLOBAL_METRICS.increment("llm_failures_total", task=task.value)
        payload = structured_fallback_payload(
            task=task,
            schema=schema,
            user_prompt=user_prompt,
            error=last_error,
        )
        validate_json_schema(payload, schema)
        _log_structured_fallback(
            engine="ONNX",
            task=task,
            attempt_count=len(attempt_plan),
            attempt_errors=attempt_errors,
        )
        self.last_error_by_task[task] = f"structured_fallback_used_after_{len(attempt_errors)}_failures: {last_error}"
        return payload
