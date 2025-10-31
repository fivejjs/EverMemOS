import os
from dotenv import load_dotenv

load_dotenv()

class ExperimentConfig:
    experiment_name: str = "locomo_evaluation_event_log2"
    mode: str = "cot"
    datase_path: str = "data/locomo10.json"
    use_emb: bool = True
    use_reranker: bool = True
    num_conv: int = 10
    # embedding_config: dict = {
    #     "model_name": "Qwen3-Embedding-4B",
    #     "base_url": "http://0.0.0.0:11000/v1/embeddings",
    # }
    # reranker_config: dict = {
    #     "model_name": "Qwen3-Reranker-4B",
    #     "base_url": "http://0.0.0.0:12000/v1/score",
    # }
    reranker_instruction: str = "Given a user's question and a text passage, determine if the passage contains specific information that directly answers the question. A relevant passage should provide a clear and precise answer, not just be on the same topic."
    llm_service: str = "openai" # openai, gemini, vllm
    # experiment_name: str = "locomo_evaluation_nemori"
    llm_config: dict = {
        "openai": {
            "llm_provider": "openai",
            "model": "gpt-4.1-mini-2025-04-14",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": os.getenv("LLM_API_KEY"),
            "temperature": 0,
            "max_tokens": 16384,
        },
        "vllm": {
            "llm_provider": "openai",
            "model": "Qwen3-30B",
            "base_url": "http://0.0.0.0:8000/v1",
            "api_key": "123",
            "temperature": 0,
            "max_tokens": 20000,
        }
    }
    max_retries: int = 5
    max_concurrent_requests: int = 10