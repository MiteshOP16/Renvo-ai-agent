from langchain_groq import ChatGroq

from app.core.config import settings
from app.tools.definitions import ALL_TOOLS


def get_llm_with_tools():
    """Fresh ChatGroq client bound to all 5 tools via LangChain's tool_bind."""
    base_llm = ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model=settings.GROQ_MODEL,
        temperature=0,
    )
    return base_llm.bind_tools(ALL_TOOLS)
