try:
    from pydantic_settings import BaseSettings
except ImportError:
    try:
        from pydantic import BaseSettings
    except ImportError:
        from pydantic.v1 import BaseSettings


class Settings(BaseSettings):
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "openai/gpt-oss-20b"

    # --- conversation memory / context management ---
    MAX_CHAT_HISTORY: int = 6
    SUMMARY_TRIGGER_TURNS: int = 12
    SUMMARY_KEEP_RECENT: int = 6

    # --- tool orchestration ---
    MAX_TOOL_CALLS_PER_TURN: int = 5

    # --- reliability ---
    MAX_LLM_RETRIES: int = 3
    RETRY_BASE_DELAY_SECONDS: float = 0.5

    # --- semantic tool retrieval (Qdrant + BGE-M3) ---
    # Empty QDRANT_URL runs an embedded/on-disk Qdrant instance -- fine for
    # local dev; production should point this at a real Qdrant server
    # (self-hosted or Qdrant Cloud) so the index survives restarts and is
    # shared across app workers instead of living in one process.
    QDRANT_URL: str = ""
    QDRANT_API_KEY: str = ""
    QDRANT_LOCAL_PATH: str = "./qdrant_data"
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-m3"
    EMBEDDING_DIM: int = 1024
    RERANKER_MODEL_NAME: str = "BAAI/bge-reranker-v2-m3"
    RETRIEVAL_TOP_K: int = 15   # candidates pulled from Qdrant before rerank
    RERANK_TOP_K: int = 7       # final tools bound to the LLM after rerank
    # Below this rerank confidence (0-1, sigmoid-normalized), the request is
    # treated as too vague to route -- ask the user to clarify instead of
    # guessing a default tool set.
    CLARIFICATION_CONFIDENCE_THRESHOLD: float = 0.35

    class Config:
        env_file = ".env"


settings = Settings()