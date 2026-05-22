"""Guideline chunk retrieval orchestration (tier-1 vector search + tier-2 narrative)."""

from __future__ import annotations

import logging
from typing import Any, NamedTuple

from config import RERANK_TOP_K, TOP_K_CHUNKS
from llm.evidence_format import pack_evidence_block
from pipeline.logging_utils import log_retrieved_chunks
from retrieval.qdrant_client import (
    TIER1_REC_CHUNK_TYPES,
    embed_query,
    expected_vector_dim,
    fetch_tier2_narrative_for_recs,
    query_chunks,
)
from retrieval.qdrant_filters import build_qdrant_filter
from retrieval.query_decompose import (
    infer_comorbidity_tags_from_query,
    infer_disease_tags_from_query,
    map_stage_to_payload_stage,
)
from retrieval.rerank import rerank_chunks

logger = logging.getLogger(__name__)


class RetrievalOutcome(NamedTuple):
    """Packed evidence strings for the LLM prompt plus score/metadata for logging."""

    chunks: list[str]
    records: list[dict[str, Any]]


def format_tier1_source_line(row: dict) -> str:
    grade_bits = []
    if row.get("cor") or row.get("loe"):
        grade_bits.append(f"COR={row.get('cor') or '—'}/LOE={row.get('loe') or '—'}")
    if row.get("kdigo_grade"):
        grade_bits.append(f"KDIGO={row.get('kdigo_grade')}")
    grade_str = ",".join(grade_bits) if grade_bits else "—"
    return " | ".join(
        [
            f"guideline={row.get('guideline') or 'unknown'}",
            f"rec_id={row.get('rec_id') or '—'}",
            f"section={row.get('section') or '—'}",
            f"type={row.get('chunk_type', '')}",
            f"grade={grade_str}",
        ]
    )


def retrieve_guideline_chunks(
    query: str,
    stage: str = "",
    top_k: int = TOP_K_CHUNKS,
    payload_filters: dict | None = None,
) -> RetrievalOutcome:
    if payload_filters is None:
        payload_filters = {
            "disease_tags": infer_disease_tags_from_query(query),
            "ckd_stage_tags": map_stage_to_payload_stage(stage),
            "chunk_types": ["recommendation", "practice_point", "rationale"],
            "drug_tags": [],
            "comorbidity_tags": infer_comorbidity_tags_from_query(query),
        }

    logger.info(
        "Retrieval started: disease_tags=%s stage=%s top_k=%s query=%s",
        payload_filters.get("disease_tags"),
        stage,
        top_k,
        query,
    )

    embedding = embed_query(query)
    logger.info("Embedding generated (dim=%s)", len(embedding))

    dim = expected_vector_dim()
    if dim is not None and len(embedding) != dim:
        raise RuntimeError(
            f"Embedding dimension mismatch: collection expects {dim}, "
            f"query has {len(embedding)}. Use same provider/model as ingestion."
        )

    qdrant_filter = build_qdrant_filter(
        payload_filters, apply_disease_tags=True, apply_stage=True
    )
    tier1_results = query_chunks(embedding, qdrant_filter, top_k=top_k)
    logger.info("Tier-1 search returned %s chunks", len(tier1_results))

    has_stage_boost = bool(
        payload_filters.get("ckd_stage_tags") or payload_filters.get("bp_stage_tags")
    )
    if not tier1_results and has_stage_boost:
        logger.info(
            "No results with stage boost (ckd=%s bp=%s). Retrying without stage filter.",
            payload_filters.get("ckd_stage_tags"),
            payload_filters.get("bp_stage_tags"),
        )
        qdrant_filter_ns = build_qdrant_filter(
            payload_filters,
            apply_disease_tags=True,
            apply_stage=False,
            apply_bp_stage=False,
        )
        tier1_results = query_chunks(embedding, qdrant_filter_ns, top_k=top_k)
        logger.info("Fallback search (no stage) returned %s chunks", len(tier1_results))

    if not tier1_results and payload_filters.get("disease_tags"):
        logger.info(
            "No results with disease_tags=%s. Retrying without disease filter.",
            payload_filters["disease_tags"],
        )
        qdrant_filter_nd = build_qdrant_filter(
            payload_filters, apply_disease_tags=False, apply_stage=False
        )
        tier1_results = query_chunks(embedding, qdrant_filter_nd, top_k=top_k)
        logger.info("Fallback search (no disease) returned %s chunks", len(tier1_results))

    rerank_keep_k = min(RERANK_TOP_K, top_k)
    tier1_results = rerank_chunks(query=query, chunks=tier1_results, top_k=rerank_keep_k)

    rec_ids = [
        r["rec_id"]
        for r in tier1_results
        if r["chunk_type"] in TIER1_REC_CHUNK_TYPES and r["rec_id"]
    ]
    tier2_rows = fetch_tier2_narrative_for_recs(rec_ids)
    logger.info(
        "Tier-2 fetch: %s rec_ids → %s narrative chunks",
        len(rec_ids),
        len(tier2_rows),
    )

    all_chunks: list[str] = []
    records: list[dict[str, Any]] = []
    ref = 1
    for r in tier1_results:
        text = r.get("text", "")
        all_chunks.append(
            pack_evidence_block(ref, format_tier1_source_line(r), text)
        )
        records.append({
            "evidence_index": ref,
            "tier": "tier1",
            "guideline": r.get("guideline"),
            "rec_id": r.get("rec_id"),
            "section": r.get("section"),
            "chunk_type": r.get("chunk_type"),
            "vector_score": r.get("score"),
            "rerank_score": r.get("rerank_score"),
            "text_preview": text[:400],
        })
        ref += 1
    for row in tier2_rows:
        text = row.get("text", "")
        src = " | ".join(
            [
                f"guideline={row.get('guideline') or 'unknown'}",
                f"parent_rec_id={row.get('parent_rec_id') or '—'}",
                f"type={row.get('chunk_type', '')}",
            ]
        )
        all_chunks.append(pack_evidence_block(ref, src, text))
        records.append({
            "evidence_index": ref,
            "tier": "tier2",
            "guideline": row.get("guideline"),
            "parent_rec_id": row.get("parent_rec_id"),
            "chunk_type": row.get("chunk_type"),
            "vector_score": None,
            "rerank_score": None,
            "text_preview": text[:400],
        })
        ref += 1

    logger.info(
        "Retrieval complete: %s total packed chunks (tier1=%s tier2=%s)",
        len(all_chunks),
        len(tier1_results),
        len(tier2_rows),
    )
    log_retrieved_chunks(records, query=query)
    return RetrievalOutcome(chunks=all_chunks, records=records)
