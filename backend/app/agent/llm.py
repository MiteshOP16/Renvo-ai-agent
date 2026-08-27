from __future__ import annotations

from langchain_groq import ChatGroq

from app.core.config import settings
from app.tools.definitions import ALL_TOOLS


def get_llm_with_tools(tools: list | None = None):
    """Fresh ChatGroq client bound to the given tools via LangChain's
    tool_bind. Callers should normally pass a routed/retrieved subset (see
    agent/tool_router.py) rather than the default full ALL_TOOLS."""
    base_llm = ChatGroq(api_key=settings.GROQ_API_KEY, model=settings.GROQ_MODEL, temperature=0)
    return base_llm.bind_tools(tools if tools is not None else ALL_TOOLS)


def get_plain_llm():
    """Fresh ChatGroq client with NO tools bound, for cheap tool-free calls
    like conversation summarization or clarifying questions."""
    return ChatGroq(api_key=settings.GROQ_API_KEY, model=settings.GROQ_MODEL, temperature=0)