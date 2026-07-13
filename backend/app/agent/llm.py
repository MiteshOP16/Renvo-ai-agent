from langchain_groq import ChatGroq

from app.core.config import settings
from app.tools.definitions import ALL_TOOLS


def get_llm_with_tools():
    """Fresh ChatGroq client bound to all registered tools via LangChain's
    tool_bind. Used for the main decision step -- the model picks which
    tool (if any) to call; nothing about tool selection is hardcoded here."""
    base_llm = ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model=settings.GROQ_MODEL,
        temperature=0,
    )
    return base_llm.bind_tools(ALL_TOOLS)


def get_plain_llm():
    """Fresh ChatGroq client with NO tools bound. Used for cheap, tool-free
    calls like conversation summarization, where sending the full tool
    schema set would waste tokens for no benefit."""
    return ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model=settings.GROQ_MODEL,
        temperature=0,
    )