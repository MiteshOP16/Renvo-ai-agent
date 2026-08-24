"""
Cross-encoder reranker (BAAI/bge-reranker-v2-m3), loaded via
sentence-transformers' CrossEncoder -- same rationale as embeddings.py:
avoids FlagEmbedding's fragile whole-package import in favor of loading
only this specific model through the standard, stable transformers API.
This is also BAAI's own documented alternative loading path for this model,
not an unofficial workaround.

The embedding-based retrieval step (vector_store.py) is fast but scores the
query and each tool independently, which occasionally ranks a
superficially-similar-but-wrong tool above the actually-correct one. This
cross-encoder jointly scores each (query, candidate) pair instead, which is
meaningfully more precise -- it can afford to be slower per-pair because it
only ever runs on the small shortlist retrieval already narrowed down to,
never the whole tool catalog.

activation_fn=torch.nn.Sigmoid() squashes the model's raw logits into
[0, 1] "relevance probability" scores -- this is the officially documented
way to get normalized scores from a sentence-transformers CrossEncoder (raw,
un-squashed logits otherwise range roughly -10 to 10), and gives us a clean,
comparable confidence value to threshold on for the "too vague to route,
ask for clarification" decision in tool_router.py.
"""

from app.core.config import settings

_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        import inspect

        import torch
        from sentence_transformers import CrossEncoder

        # The kwarg name for "squash raw logits into [0,1] via sigmoid" was
        # renamed across sentence-transformers versions --
        # default_activation_function (older, e.g. 3.x) vs activation_fn
        # (newer, post the Cross Encoder API rewrite). Detect which one the
        # installed version actually accepts via introspection BEFORE
        # attempting the (large, slow-to-download) model load, rather than
        # guessing and risking a failed multi-minute download-then-retry.
        init_params = inspect.signature(CrossEncoder.__init__).parameters
        if "activation_fn" in init_params:
            _reranker = CrossEncoder(settings.RERANKER_MODEL_NAME, activation_fn=torch.nn.Sigmoid())
        elif "default_activation_function" in init_params:
            _reranker = CrossEncoder(settings.RERANKER_MODEL_NAME, default_activation_function=torch.nn.Sigmoid())
        else:
            # Neither kwarg exists in this version -- load without forcing
            # sigmoid and normalize the raw logits ourselves in rerank().
            _reranker = CrossEncoder(settings.RERANKER_MODEL_NAME)
    return _reranker


def rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    """candidates: [{"name": ..., "text": ..., ...}, ...]. Returns the same
    dicts sorted by rerank score (highest first, in [0, 1]), trimmed to
    top_k, each with a "rerank_score" key added."""
    if not candidates:
        return []

    model = get_reranker()
    pairs = [[query, c["text"]] for c in candidates]
    scores = model.predict(pairs)

    # If get_reranker() fell back to no activation function (very old/very
    # new sentence-transformers with neither known kwarg), scores are raw
    # logits, not [0,1] -- apply sigmoid here so downstream confidence
    # thresholding (see tool_router.py) always gets a comparable value.
    scores = [float(s) for s in scores]
    if scores and (min(scores) < 0 or max(scores) > 1):
        import math

        scores = [1 / (1 + math.exp(-s)) for s in scores]

    for c, s in zip(candidates, scores):
        c["rerank_score"] = s

    candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
    return candidates[:top_k]