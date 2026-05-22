"""Optional cross-encoder reranking for retrieved chunks."""

from __future__ import annotations

import logging

from config import RERANK_BATCH_SIZE, RERANK_DEVICE, RERANK_ENABLED, RERANK_MODEL

logger = logging.getLogger(__name__)

_reranker_model = None
_reranker_unavailable = False


def get_reranker_model():
    global _reranker_model, _reranker_unavailable
    if not RERANK_ENABLED or _reranker_unavailable:
        return None
    if _reranker_model is not None:
        return _reranker_model

    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        _reranker_unavailable = True
        logger.warning(
            "Reranker enabled but sentence-transformers is not installed. "
            "Continuing without reranking."
        )
        return None

    try:
        _reranker_model = CrossEncoder(RERANK_MODEL, device=RERANK_DEVICE)
        logger.info("Loaded reranker model=%s device=%s", RERANK_MODEL, RERANK_DEVICE)
    except Exception as exc:
        _reranker_unavailable = True
        logger.warning("Failed to load reranker (%s). Continuing without reranking.", exc)
        return None
    return _reranker_model


def rerank_chunks(query: str, chunks: list[dict], top_k: int) -> list[dict]:
    model = get_reranker_model()
    if model is None or not chunks:
        return chunks[:top_k]

    pairs = [(query, chunk.get("text", "")) for chunk in chunks]
    try:
        scores = model.predict(
            pairs,
            batch_size=RERANK_BATCH_SIZE,
            show_progress_bar=False,
        )
    except Exception as exc:
        logger.warning("Reranking failed (%s). Falling back to vector ranking.", exc)
        return chunks[:top_k]

    for idx, chunk in enumerate(chunks):
        chunk["rerank_score"] = float(scores[idx])

    ranked = sorted(chunks, key=lambda item: item["rerank_score"], reverse=True)
    logger.info("Reranking complete: kept top %s of %s chunks", min(top_k, len(ranked)), len(ranked))
    return ranked[:top_k]
