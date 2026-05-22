"""Shared Qdrant client and low-level vector / scroll operations."""

from __future__ import annotations

import logging

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny

from config import EMBED_MODEL, EMBEDDINGS_URL, QDRANT_API_KEY, QDRANT_COLLECTION, QDRANT_URL
from llm.embeddings import fetch_embeddings_http
from llm.llm_client import get_openai_client

logger = logging.getLogger(__name__)

_qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

TIER1_REC_CHUNK_TYPES = frozenset({"recommendation", "practice_point", "rec", "pp"})
TIER2_NARRATIVE_CHUNK_TYPES = frozenset({"synopsis", "supportive_text", "rationale"})


def get_qdrant() -> QdrantClient:
    return _qdrant


def embed_query(query: str) -> list[float]:
    if EMBEDDINGS_URL:
        return fetch_embeddings_http([query])[0]

    client = get_openai_client()
    if client is None:
        raise RuntimeError(
            "No embedding provider configured. Set EMBEDDINGS_URL or OPENAI_API_KEY."
        )
    return client.embeddings.create(input=query, model=EMBED_MODEL).data[0].embedding


def expected_vector_dim() -> int | None:
    info = _qdrant.get_collection(QDRANT_COLLECTION)
    vectors = info.config.params.vectors
    if hasattr(vectors, "size"):
        return vectors.size
    if isinstance(vectors, dict):
        first = next(iter(vectors.values()), None)
        return getattr(first, "size", None)
    return None


def query_chunks(
    embedding: list[float],
    qdrant_filter: Filter,
    top_k: int,
) -> list[dict]:
    response = _qdrant.query_points(
        collection_name=QDRANT_COLLECTION,
        query=embedding,
        query_filter=qdrant_filter,
        limit=top_k,
    )
    points = getattr(response, "points", response)
    results = []
    for p in points:
        if not getattr(p, "payload", None):
            continue
        pl = p.payload
        kdigo_grade = pl.get("kdigo_grade") or ""
        legacy_grade = pl.get("grade") or ""
        results.append({
            "text": pl.get("text", ""),
            "chunk_type": pl.get("chunk_type", "unknown"),
            "rec_id": pl.get("rec_id", ""),
            "section": pl.get("section_id", ""),
            "guideline": pl.get("guideline", "") or "unknown",
            "grade": kdigo_grade or legacy_grade,
            "cor": pl.get("cor", "") or "",
            "loe": pl.get("loe", "") or "",
            "kdigo_grade": kdigo_grade or legacy_grade,
            "score": getattr(p, "score", None),
        })
    return results


def fetch_tier2_narrative_for_recs(rec_ids: list[str]) -> list[dict]:
    if not rec_ids:
        return []

    narrative_filter = Filter(
        must=[
            FieldCondition(
                key="chunk_type",
                match=MatchAny(any=list(TIER2_NARRATIVE_CHUNK_TYPES)),
            ),
            FieldCondition(key="parent_rec_id", match=MatchAny(any=rec_ids)),
        ]
    )
    response = _qdrant.scroll(
        collection_name=QDRANT_COLLECTION,
        scroll_filter=narrative_filter,
        limit=len(rec_ids) * 4,
        with_payload=True,
    )
    points = response[0] if response else []
    out: list[dict] = []
    for p in points:
        pl = getattr(p, "payload", None) or {}
        out.append({
            "text": pl.get("text", ""),
            "parent_rec_id": pl.get("parent_rec_id", "") or "",
            "guideline": pl.get("guideline", "") or "unknown",
            "chunk_type": pl.get("chunk_type", "unknown"),
        })
    return out
