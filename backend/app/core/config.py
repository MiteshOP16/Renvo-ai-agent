from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.3-70b-versatile"

    # --- conversation memory / context management ---
    MAX_CHAT_HISTORY: int = 6
    SUMMARY_TRIGGER_TURNS: int = 12
    SUMMARY_KEEP_RECENT: int = 6

    # --- tool orchestration ---
    # bumped from 4 -> 5: read-only analysis tools (e.g. "check outliers,
    # then fix them") can now combine with one mutating action in a turn,
    # plus retry headroom if a call fails validation.
    MAX_TOOL_CALLS_PER_TURN: int = 5

    # --- reliability ---
    MAX_LLM_RETRIES: int = 3
    RETRY_BASE_DELAY_SECONDS: float = 0.5

    class Config:
        env_file = ".env"


settings = Settings()