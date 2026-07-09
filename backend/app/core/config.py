from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    MAX_CHAT_HISTORY: int = 8
    MAX_TOOL_CALLS_PER_TURN: int = 3

    class Config:
        env_file = ".env"


settings = Settings()
