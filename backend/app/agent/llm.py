from langchain_groq import ChatGroq

from app.core.config import settings
from app.tools.definitions import ALL_TOOLS


def get_llm_with_tools(tools: list | None = None):
    """Fresh ChatGroq client bound to the given tools via LangChain's
    tool_bind. Defaults to ALL_TOOLS if none are passed, but callers should
    normally pass a routed subset (see agent/tool_router.py) -- binding all
    ~25 tools on every call is what triggers Groq's malformed tool-call
    failures in practice."""
    base_llm = ChatGroq(api_key=settings.GROQ_API_KEY, model=settings.GROQ_MODEL, temperature=0)
    return base_llm.bind_tools(tools if tools is not None else ALL_TOOLS)


def get_plain_llm():
    """Fresh ChatGroq client with NO tools bound, for cheap tool-free calls
    like conversation summarization."""
    return ChatGroq(api_key=settings.GROQ_API_KEY, model=settings.GROQ_MODEL, temperature=0)