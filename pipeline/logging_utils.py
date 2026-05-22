"""Structured debug logging for retrieval chunks and LLM prompts."""

from __future__ import annotations

import json
import logging
from typing import Any

from config import LOG_LLM_PROMPTS, LOG_PROMPT_MAX_CHARS, LOG_RETRIEVED_CHUNKS

logger = logging.getLogger(__name__)


def log_retrieved_chunks(
    records: list[dict[str, Any]],
    *,
    query: str = "",
    route: str = "",
) -> None:
    """Log each retrieved chunk with vector/rerank scores and metadata."""
    if not LOG_RETRIEVED_CHUNKS or not records:
        return

    header = f"RETRIEVED CHUNKS ({len(records)} items)"
    if route:
        header += f" route={route}"
    logger.info("%s", header)
    if query:
        logger.info("  semantic_query: %s", query[:500])

    for rec in records:
        idx = rec.get("evidence_index", "?")
        tier = rec.get("tier", "tier1")
        guideline = rec.get("guideline", "unknown")
        rec_id = rec.get("rec_id") or rec.get("parent_rec_id") or "—"
        chunk_type = rec.get("chunk_type", "")
        section = rec.get("section", "")

        vector = rec.get("vector_score")
        rerank = rec.get("rerank_score")
        score_parts = []
        if vector is not None:
            score_parts.append(f"vector={vector:.4f}" if isinstance(vector, float) else f"vector={vector}")
        if rerank is not None:
            score_parts.append(f"rerank={rerank:.4f}" if isinstance(rerank, float) else f"rerank={rerank}")
        scores_str = ", ".join(score_parts) if score_parts else "scores=n/a"

        preview = (rec.get("text_preview") or "")[:300]
        logger.info(
            "  [%s] %s | %s | guideline=%s rec_id=%s section=%s type=%s | %s",
            idx,
            tier,
            scores_str,
            guideline,
            rec_id,
            section,
            chunk_type,
            preview.replace("\n", " ")[:300],
        )


def log_prompt(label: str, prompt: str, *, extra: dict[str, Any] | None = None) -> None:
    """Log a full or truncated LLM prompt for debugging."""
    if not LOG_LLM_PROMPTS:
        return

    text = prompt or ""
    truncated = False
    if LOG_PROMPT_MAX_CHARS > 0 and len(text) > LOG_PROMPT_MAX_CHARS:
        text = text[:LOG_PROMPT_MAX_CHARS] + f"\n... [truncated, total {len(prompt)} chars]"
        truncated = True

    logger.info("=" * 60)
    logger.info("LLM PROMPT: %s (chars=%s%s)", label, len(prompt), " truncated" if truncated else "")
    if extra:
        logger.info("  meta: %s", json.dumps(extra, default=str))
    logger.info("-" * 60)
    logger.info("%s", text)
    logger.info("=" * 60)
