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
def on_startup():
    """Verify TF-IDF tool router is ready (instantaneous, no model loading)."""
    try:
        from app.agent.tool_router import build_tool_index
        n = build_tool_index()
        logger.info("TF-IDF tool router ready: %d tools indexed.", n)
    except Exception as e:
        logger.error("Tool router startup failed: %s", e)


@app.get("/")
def root():
    return {"status": "ok", "service": "Data Cleaning Agent API"}