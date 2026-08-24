"""
BGE-M3 dense embedding wrapper (1024-dim), loaded via sentence-transformers.

Deliberately does NOT use the FlagEmbedding package. FlagEmbedding's
__init__.py eagerly imports its ENTIRE codebase on `from FlagEmbedding
import ...` -- including decoder-only Gemma reranker code this project
never uses -- so any breaking change anywhere in transformers' Gemma2
modeling internals (which HF has repeatedly refactored: 4.48, 4.52, and
5.0 each broke a different symbol FlagEmbedding relies on) takes down the
import for everyone, even users who only wanted a totally unrelated model.

sentence-transformers instead loads only the specific model requested, via
the standard AutoModel/AutoTokenizer API -- and BGE-M3 is architecturally
just XLM-RoBERTa, one of the most stable, long-supported model families in
transformers. BAAI's own bge-m3 model card explicitly documents
sentence-transformers as a supported loading path, so this isn't a hack --
it's the officially sanctioned alternative to FlagEmbedding.

The model is loaded lazily and cached as a module-level singleton so the
~2GB weights are loaded into memory once per process, not once per call.
"""

from app.core.config import settings

_model = None


def get_embedding_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
    return _model


def embed_text(text: str) -> list[float]:
    return embed_texts([text])[0]


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_embedding_model()
    # normalize_embeddings=True makes cosine similarity equivalent to a dot
    # product, matching the Distance.COSINE config on the Qdrant collection.
    vectors = model.encode(texts, normalize_embeddings=True)
    return vectors.tolist()