import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router

logger = logging.getLogger("datacleanai")

app = FastAPI(title="Data Cleaning Agent", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.on_event("startup")
def check_tool_retrieval_health():
    """Eagerly load the embedding + reranker models and build/verify the
    Qdrant tool index at startup, so a broken dependency install (e.g. an
    incompatible transformers version -- see requirements.txt for two real
    examples of this) surfaces immediately in the startup logs, instead of
    silently degrading to the full-tool-set fallback on a user's first
    prompt and only showing up as an [ERROR] line in the activity log.

    This check is diagnostic only -- if it fails, the app still starts.
    The agent graph's own per-request fallback (see agent/graph.py) already
    handles retrieval being unavailable at request time; this just makes
    that situation loud and visible at boot instead of quiet and delayed."""
    try:
        from app.agent.embeddings import get_embedding_model

        get_embedding_model()
        logger.info("Embedding model (BGE-M3) loaded OK.")
    except Exception as e:
        logger.error(
            "STARTUP HEALTH CHECK FAILED: embedding model could not be loaded. "
            "Tool retrieval will fall back to binding the full tool set on every "
            "request until this is fixed. Error: %s", e,
        )
        return  # no point testing the reranker/index if embeddings are already broken

    try:
        from app.agent.reranker import get_reranker

        get_reranker()
        logger.info("Reranker model (bge-reranker-v2-m3) loaded OK.")
    except Exception as e:
        # Deliberately NOT `return`-ing here. Building the tool index only
        # needs the embedding model (already confirmed working above), and
        # route_tools() itself degrades gracefully without a working
        # reranker -- it falls back to embedding-similarity ranking alone
        # (see agent/tool_router.py). A reranker-only failure should not
        # block retrieval from working at all; it should just make it
        # slightly less precise until fixed. (An earlier version of this
        # function DID return here, which meant a broken reranker silently
        # prevented the tool index from ever being built -- every request
        # then queried an empty Qdrant collection, got zero candidates back,
        # and every single prompt looked like "confidence 0.0, ask for
        # clarification" regardless of what was actually typed. If you saw
        # that symptom before this fix, restarting now will also
        # self-heal it: build_tool_index() below checks the actual point
        # count, not just whether the collection exists, so it will notice
        # the index is empty and populate it properly this time.)
        logger.warning(
            "STARTUP HEALTH CHECK WARNING: reranker model could not be loaded. "
            "Tool retrieval will still work using embedding-similarity ranking alone "
            "(slightly less precise than with reranking) until this is fixed. Error: %s", e,
        )

    try:
        from app.agent.tool_router import build_tool_index

        n = build_tool_index()
        if n:
            logger.info("Tool index built: %d tool(s) indexed.", n)
        else:
            logger.info("Tool index already populated -- skipped rebuild.")
    except Exception as e:
        logger.error("STARTUP HEALTH CHECK FAILED: could not build/verify the Qdrant tool index: %s", e)


@app.get("/")
def root():
    return {"status": "ok", "service": "Data Cleaning Agent API"}