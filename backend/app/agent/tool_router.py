"""
Tool routing via semantic retrieval + rerank (replaces the earlier
keyword-based version).

Two-stage pipeline per turn:
  1. RETRIEVE -- embed the turn's text (BGE-M3) and search Qdrant for the
     top-N most similar tool descriptions (fast, cheap, slightly imprecise
     since query and candidates are scored independently).
  2. RE-RANK -- score that shortlist with a cross-encoder reranker
     (BAAI/bge-reranker-v2-m3), which jointly scores (query, tool) pairs for
     much better precision. Only ever runs on the small shortlist, not the
     whole tool catalog, so the extra precision is cheap.

This is what keeps Groq's per-call token usage small and roughly constant
as the tool catalog grows: instead of binding every registered tool's full
JSON schema on every call, we bind only the handful (RERANK_TOP_K) that are
actually relevant to what the user just asked.

WHEN THE REQUEST IS TOO VAGUE TO ROUTE:
If the top reranked score falls below CLARIFICATION_CONFIDENCE_THRESHOLD,
route_tools() reports needs_clarification=True instead of falling back to a
guessed default tool set. The caller (agent_decide_node in graph.py) uses
this to skip tool binding entirely for that turn and ask the user a direct
clarifying question instead -- deliberately not guessing which of several
plausible tools they meant. This is a design choice: guessing wrong on a
mutating action is worse than asking one extra question.
"""

from app.agent.embeddings import embed_text
from app.agent.reranker import rerank
from app.agent.vector_store import (
    ensure_collection,
    get_qdrant_client,
    index_is_populated,
    search_tools,
    upsert_tools,
)
from app.core.config import settings
from app.tools.analysis_implementations import ANALYSIS_TOOL_EXECUTORS
from app.tools.definitions import ALL_TOOLS

_TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}


def _tool_type(name: str) -> str:
    return "analysis" if name in ANALYSIS_TOOL_EXECUTORS else "mutating"


def _tool_text(tool) -> str:
    # name + description together give the embedding/reranker the fullest
    # signal about what the tool does and when to use it.
    return f"{tool.name}: {tool.description}"


def build_tool_index(force_rebuild: bool = False) -> int:
    """Embed every registered tool's description and (re)populate the
    Qdrant collection. Idempotent -- safe to call on every app startup;
    only does real work the first time or when force_rebuild=True (e.g.
    after adding/editing tools). Returns the number of tools indexed."""
    client = get_qdrant_client()
    ensure_collection(client)

    if index_is_populated(client) and not force_rebuild:
        return 0

    texts = [_tool_text(t) for t in ALL_TOOLS]
    vectors = embed_text_batch(texts)
    records = [
        {"id": i, "name": t.name, "vector": vec, "type": _tool_type(t.name), "text": text}
        for i, (t, vec, text) in enumerate(zip(ALL_TOOLS, vectors, texts))
    ]
    upsert_tools(client, records)
    return len(records)


def embed_text_batch(texts: list[str]) -> list[list[float]]:
    """Thin wrapper so build_tool_index can batch-embed via embeddings.py
    without importing embed_texts directly at module load time (keeps this
    module's import graph simple and easy to monkeypatch in tests)."""
    from app.agent.embeddings import embed_texts

    return embed_texts(texts)


def route_tools(context_text: str) -> dict:
    """Returns {"tools": [...], "confidence": float, "needs_clarification": bool}.
    `tools` is a list of LangChain tool objects ready to pass to bind_tools;
    empty when needs_clarification is True."""
    client = get_qdrant_client()
    query_vector = embed_text(context_text)

    retrieved = search_tools(client, query_vector, top_k=settings.RETRIEVAL_TOP_K)
    if not retrieved:
        return {"tools": [], "confidence": 0.0, "needs_clarification": True}

    try:
        reranked = rerank(context_text, retrieved, top_k=settings.RERANK_TOP_K)
        top_score = reranked[0]["rerank_score"] if reranked else 0.0
    except Exception:
        # Reranker unavailable at request time -- degrade to the retrieval
        # stage's own similarity ranking rather than failing the whole
        # routing decision. Coarser (embedding similarity alone is less
        # precise than a cross-encoder rerank -- see reranker.py) but keeps
        # the app functional instead of bouncing every request to a
        # clarifying question or the full-tool-set fallback.
        reranked = sorted(retrieved, key=lambda r: r["score"], reverse=True)[: settings.RERANK_TOP_K]
        top_score = reranked[0]["score"] if reranked else 0.0

    tools = [_TOOLS_BY_NAME[r["name"]] for r in reranked if r["name"] in _TOOLS_BY_NAME]

    return {
        "tools": tools,
        "confidence": top_score,
        "needs_clarification": top_score < settings.CLARIFICATION_CONFIDENCE_THRESHOLD,
    }


def accumulate_turn_text(messages) -> str:
    """Concatenate the text content of this turn's messages (human + any
    tool results seen so far) into one string to embed/rerank against --
    catches both the user's original request and follow-up context, e.g. a
    column-not-found suggestion from an earlier tool call this same turn."""
    parts = []
    for m in messages:
        content = getattr(m, "content", None)
        if isinstance(content, str) and content:
            parts.append(content)
    return "\n".join(parts)