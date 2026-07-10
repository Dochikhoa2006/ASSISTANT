import sys
import logging
from assistant_rag.settings import ProductionSettings
from assistant_rag.production_factory import build_production_pipeline, build_production_repository
from assistant_rag.contracts import ChatRequest
from assistant_rag.observability import start_trace, new_request_id

# Configure minimal clean logging to console
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def run_test_query(pipeline, repository, query: str, user_id: str = "test-user-1"):
    print("\n" + "="*80)
    print(f"USER QUERY: '{query}'")
    print("="*80)
    
    # 1. Formulate the request
    request = ChatRequest(
        raw_query=query,
        user_id=user_id
    )
    try:
        start_trace(new_request_id())
        response = pipeline.handle(request, repository)
            
        print(f"RESPONSE TYPE: {response.response_type.value}")
        print(f"PERSISTED IN TOPIC: {response.conversation_topic_id}")
        print(f"PERSISTED IN HOP: {response.conversation_hop_id}")
        if response.warnings:
            print(f"WARNINGS: {response.warnings}")
            
        print("\nASSISTANT RESPONSE:")
        print(response.final_chat_text)
        
    except Exception as e:
        print(f"Execution Error: {e}", file=sys.stderr)
    print("="*80)

def main():
    try:
        settings = ProductionSettings()
        print("Active System Models Configuration:")
        print(f"  - Fast Model:               {settings.ollama.fast_model}")
        print(f"  - Balanced Model:           {settings.ollama.balanced_model}")
        print(f"  - Accurate Model:           {settings.ollama.accurate_model}")
        print(f"  - Writing Model:            {settings.ollama.writing_model}")
        print(f"  - Intent Classifier Model:  {settings.ollama.intent_model or settings.ollama.balanced_model}")
        print(f"  - Last-QA Resolver Model:   {settings.ollama.last_qa_model or settings.ollama.fast_model}")
        print(f"  - Reranker Model:           {settings.reranker.model_name}")
        print("\nInitializing assistant pipeline... (this might take a moment to load/warm up models)")
        # Ensure we connect to sqlite in workspace and use configured routing
        pipeline = build_production_pipeline(settings)
        repository = build_production_repository(settings)
        print("Pipeline successfully initialized.\n")
    except Exception as e:
        print(f"Failed to initialize pipeline: {e}", file=sys.stderr)
        sys.exit(1)
        
    # Predefined sample questions
    sample_queries = [
        "Hello, who are you and what can you do?",
        "Add a note to my python study log that lists are ordered and mutable.",
        "Remind me to review SQL triggers tomorrow at 9 AM."
    ]
    
    print("Running predefined sample queries:")
    for query in sample_queries:
        run_test_query(pipeline, repository, query)
        
    # Optional Interactive Mode
    print("\nPredefined tests finished. Entering interactive CLI mode.")
    print("Type your questions below. Type 'exit' or 'quit' to end.\n")
    
    while True:
        try:
            query = input("Ask Assistant > ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit"):
                print("Goodbye!")
                break
            run_test_query(pipeline, repository, query)
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

if __name__ == "__main__":
    main()
