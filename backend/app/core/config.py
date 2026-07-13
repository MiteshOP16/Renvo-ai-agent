from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.3-70b-versatile"

    # --- conversation memory / context management ---
    # how many raw recent turns get sent to the LLM verbatim
    MAX_CHAT_HISTORY: int = 6
    # once stored raw history exceeds this many turns, the oldest ones get
    # folded into the running summary and pruned (hierarchical summarization)
    SUMMARY_TRIGGER_TURNS: int = 12
    # how many of the most recent turns survive pruning (must be >= MAX_CHAT_HISTORY)
    SUMMARY_KEEP_RECENT: int = 6

    # --- tool orchestration ---
    MAX_TOOL_CALLS_PER_TURN: int = 4

    # --- reliability ---
    MAX_LLM_RETRIES: int = 3
    RETRY_BASE_DELAY_SECONDS: float = 0.5

    class Config:
        env_file = ".env"


settings = Settings()