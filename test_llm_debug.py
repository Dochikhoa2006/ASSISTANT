import json
from assistant_rag.llm import OllamaLLMClient, OllamaModelRouter, LLMTask
from assistant_rag.settings import OllamaSettings

settings = OllamaSettings()
router = OllamaModelRouter(settings)
client = OllamaLLMClient(settings, router)

task = LLMTask.ANSWER
system_prompt = "You are a helpful assistant."
user_prompt = "Hello, testing 1 2 3!"
decision = router.decision_for_task(task)

body = {
    "model": decision.model,
    "messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ],
    "stream": False,
    "keep_alive": settings.keep_alive,
    "options": {
        "temperature": decision.temperature,
    },
}
if decision.num_ctx:
    body["options"]["num_ctx"] = decision.num_ctx
if decision.num_predict:
    body["options"]["num_predict"] = decision.num_predict

print(f"Request body: {json.dumps(body)}")
try:
    payload = client._request_json("/api/chat", body, timeout=decision.timeout_seconds)
    print(f"Raw payload keys: {list(payload.keys())}")
    print(f"Message obj: {payload.get('message')}")
    print(f"Content: '{payload.get('message', {}).get('content', '')}'")
except Exception as e:
    print(f"Error: {e}")
