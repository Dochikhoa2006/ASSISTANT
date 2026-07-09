from assistant_rag.llm import OllamaLLMClient, OllamaModelRouter, LLMTask
from assistant_rag.settings import OllamaSettings

settings = OllamaSettings()
router = OllamaModelRouter(settings)
client = OllamaLLMClient(settings, router)

task = LLMTask.ANSWER
system_prompt = "You are a helpful assistant."
user_prompt = "Hello, testing 1 2 3!"

print(f"Testing {task}...")
try:
    response = client.chat(task=task, system_prompt=system_prompt, user_prompt=user_prompt)
    print(f"Response: '{response}'")
    print(f"Length: {len(response)}")
except Exception as e:
    print(f"Error: {e}")
