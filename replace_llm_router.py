import sys

def replace_in_file():
    with open("assistant_rag/llm.py", "r") as f:
        lines = f.readlines()
        
    # Find OllamaModelRouter
    router_start = -1
    for i, line in enumerate(lines):
        if line.startswith("class OllamaModelRouter:"):
            router_start = i
            break
            
    router_end = -1
    for i in range(router_start + 1, len(lines)):
        if line.startswith("@dataclass") or line.startswith("class "):
            router_end = i
            break
            
    if router_start == -1 or router_end == -1:
        print("Could not find OllamaModelRouter")
        sys.exit(1)
        
    router_def = """class OllamaModelRouter:
    settings: OllamaSettings

    def decision_for_task(self, task: LLMTask) -> ModelDecision:
        return ModelDecision(
            task=task,
            model=self.model_for_task(task),
            temperature=self.temperature_for_task(task),
            timeout_seconds=self.timeout_for_task(task),
            num_ctx=self.num_ctx_for_task(task),
            num_predict=self.num_predict_for_task(task),
            reason_summary=f"Task-specific configuration for {task.value}",
        )

    def model_for_task(self, task: LLMTask) -> str:
        return getattr(self.settings, f"model_{task.value}")

    def temperature_for_task(self, task: LLMTask) -> float:
        return getattr(self.settings, f"temperature_{task.value}")

    def num_ctx_for_task(self, task: LLMTask) -> int | None:
        return getattr(self.settings, f"num_ctx_{task.value}")

    def num_predict_for_task(self, task: LLMTask) -> int | None:
        return getattr(self.settings, f"num_predict_{task.value}")

    def timeout_for_task(self, task: LLMTask) -> float:
        return getattr(self.settings, f"timeout_{task.value}")


"""

    new_lines = lines[:router_start] + [router_def] + lines[router_end:]
    
    # Now replace generate_json in OllamaLLMClient
    gen_start = -1
    for i, line in enumerate(new_lines):
        if line.startswith("    def generate_json("):
            gen_start = i
            break
            
    gen_end = -1
    for i in range(gen_start + 1, len(new_lines)):
        if line.startswith("    def chat("):
            if new_lines[i].startswith("    def chat("):
                gen_end = i
                break
                
    if gen_start == -1 or gen_end == -1:
        print("Could not find generate_json")
        sys.exit(1)
        
    gen_def = """    def generate_json(
        self,
        *,
        task: LLMTask,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        with StageTimer(f"llm_{task.value}"):
            last_error: Exception | None = None
            retry_count = getattr(self.settings, f"json_retry_count_{task.value}", self.settings.structured_retry_count)
            for _ in range(retry_count + 1):
                try:
                    raw = self._chat_raw(
                        task=task,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        format_schema=schema,
                    )
                    payload = parse_json_object(raw)
                    validate_json_schema(payload, schema)
                    self.last_error_by_task.pop(task, None)
                    return payload
                except Exception as exc:
                    GLOBAL_METRICS.increment("llm_json_parse_failures_total", task=task.value)
                    last_error = exc
                    self.last_error_by_task[task] = str(exc)
            GLOBAL_METRICS.increment("llm_failures_total", task=task.value)
            raise ValueError(f"Ollama structured output failed: {last_error}")

"""
    
    final_lines = new_lines[:gen_start] + [gen_def] + new_lines[gen_end:]
    
    with open("assistant_rag/llm.py", "w") as f:
        f.write("".join(final_lines))

replace_in_file()
