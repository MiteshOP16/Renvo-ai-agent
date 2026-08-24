"""
Qdrant-backed vector store for tool descriptions.

Stores one point per registered tool (its name + description embedded as a
1024-dim BGE-M3 vector) and exposes similarity search over them. This is
the "retrieve" half of the retrieve-then-rerank pipeline used to decide
which tools to bind to the LLM each turn (see tool_router.py) -- instead of
sending Groq the full JSON schema for every tool the app knows about, we
send it only the handful that are actually relevant to what the user asked.

Runs against a real Qdrant server (self-hosted or Qdrant Cloud) when
QDRANT_URL is set in settings, or an embedded on-disk instance otherwise --
the embedded mode needs no separate server process and is fine for local
development, but production should point QDRANT_URL at a real server so the
index survives restarts and is shared across app workers instead of living
inside one process's disk.
"""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.core.config import settings

COLLECTION_NAME = "tool_descriptions"


def get_qdrant_client() -> QdrantClient:
    if settings.QDRANT_URL:
        return QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY or None)
    return QdrantClient(path=settings.QDRANT_LOCAL_PATH)


def ensure_collection(client: QdrantClient) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=settings.EMBEDDING_DIM, distance=Distance.COSINE),
        )


def index_is_populated(client: QdrantClient) -> bool:
    try:
        info = client.get_collection(COLLECTION_NAME)
        return (info.points_count or 0) > 0
    except Exception:
        return False


def upsert_tools(client: QdrantClient, tool_records: list[dict]) -> None:
    """tool_records: [{"id": int, "name": str, "vector": list[float],
    "type": "mutating"|"analysis", "text": str}, ...]"""
    points = [
        PointStruct(
            id=rec["id"],
            vector=rec["vector"],
            payload={"name": rec["name"], "type": rec["type"], "text": rec["text"]},
        )
        for rec in tool_records
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)


def search_tools(client: QdrantClient, query_vector: list[float], top_k: int) -> list[dict]:
    """Returns the top_k most similar tools, each as
    {"name": str, "type": str, "text": str, "score": float} (score = cosine
    similarity, higher is better). This is the fast/imprecise first stage --
    callers should rerank the result before trusting the ordering."""
    hits = client.query_points(
        collection_name=COLLECTION_NAME, query=query_vector, limit=top_k
    ).points
    return [
        {"name": h.payload["name"], "type": h.payload["type"], "text": h.payload["text"], "score": h.score}
        for h in hits
    ]


def reset_collection(client: QdrantClient) -> None:
    """Drop and recreate the collection -- use when tool descriptions have
    changed and the index needs a full rebuild, not incremental upserts."""
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    ensure_collection(client)