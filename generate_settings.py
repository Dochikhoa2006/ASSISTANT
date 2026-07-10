def generate():
    tasks = [
        ("query_rewrite", "qwen3.5:0.8b", 12.0, 1024, 96, 0.0, None),
        ("last_qa", "qwen3.5:2b", 18.0, 1536, 160, 0.0, 1),
        ("intent", "qwen3.5:2b", 18.0, 2048, 96, 0.0, None),
        ("action_extraction", "qwen3.5:4b", 30.0, 3072, 512, 0.0, None),
        ("generate_clarification", "qwen3.5:2b", 15.0, 2048, 96, 0.15, 1),
        ("generate_human_supporting", "qwen3.5:2b", 15.0, 2048, 128, 0.25, 1),
        ("generate_reminder_supporting", "qwen3.5:2b", 15.0, 2048, 128, 0.2, 1),
        ("clarification_merge", "qwen3.5:4b", 24.0, 3072, 256, 0.0, 1),
        ("answer", "qwen3.5:9b", 75.0, 8192, 1024, 0.22, None),
        ("writing", "qwen3.5:9b", 90.0, 8192, 1536, 0.38, None),
        ("risky_action", "qwen3.5:9b", 35.0, 4096, 384, 0.0, 1),
        ("retrieval_validation", "qwen3.5:9b", 35.0, 4096, 384, 0.0, None),
        ("general_sub_branch_detection", "qwen3.5:2b", 12.0, 1536, 96, 0.0, None),
        ("content_composer_react", "qwen3.5:2b", 15.0, 2048, 128, 0.0, None),
        ("action_planning", "qwen3.5:4b", 35.0, 4096, 512, 0.0, None),
    ]

    print("@dataclass(frozen=True)")
    print("class OllamaSettings:")
    print('    base_url: str = "http://localhost:11434"')
    print('    structured_retry_count: int = 1')
    print('    keep_alive: int | str = "30m"')
    print()

    for task in tasks:
        t, model, timeout, ctx, predict, temp, retry = task
        print(f"    # Task: {t.upper()}")
        print(f'    model_{t}: str = "{model}"')
        print(f"    timeout_{t}: float = {timeout}")
        print(f"    num_ctx_{t}: int = {ctx}")
        if predict is not None:
            print(f"    num_predict_{t}: int | None = {predict}")
        else:
            print(f"    num_predict_{t}: int | None = None")
        print(f"    temperature_{t}: float = {temp}")
        if retry is not None:
            print(f"    json_retry_count_{t}: int = {retry}")
        print()

    print("=" * 80)
    print("from_env")
    print("=" * 80)

    print("            ollama=OllamaSettings(")
    print('                base_url=os.getenv("OLLAMA_BASE_URL", OllamaSettings.base_url),')
    print('                structured_retry_count=_get_int("OLLAMA_STRUCTURED_RETRY_COUNT", OllamaSettings.structured_retry_count),')
    print('                keep_alive=-1 if os.getenv("OLLAMA_KEEP_ALIVE", str(OllamaSettings.keep_alive)) == "-1" else os.getenv("OLLAMA_KEEP_ALIVE", OllamaSettings.keep_alive),')
    for task in tasks:
        t, model, timeout, ctx, predict, temp, retry = task
        T = t.upper()
        print(f'                model_{t}=os.getenv("OLLAMA_{T}_MODEL", OllamaSettings.model_{t}),')
        print(f'                timeout_{t}=_get_float("OLLAMA_{T}_TIMEOUT", OllamaSettings.timeout_{t}),')
        print(f'                num_ctx_{t}=_get_int("OLLAMA_{T}_NUM_CTX", OllamaSettings.num_ctx_{t}),')
        print(f'                num_predict_{t}=_get_int("OLLAMA_{T}_NUM_PREDICT", OllamaSettings.num_predict_{t}) if os.getenv("OLLAMA_{T}_NUM_PREDICT") else OllamaSettings.num_predict_{t},')
        print(f'                temperature_{t}=_get_float("OLLAMA_{T}_TEMPERATURE", OllamaSettings.temperature_{t}),')
        if retry is not None:
            print(f'                json_retry_count_{t}=_get_int("OLLAMA_{T}_JSON_RETRY_COUNT", OllamaSettings.json_retry_count_{t}),')
    print("            ),")

generate()
