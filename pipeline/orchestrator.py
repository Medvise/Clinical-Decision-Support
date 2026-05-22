import logging
import time
import uuid

from api.schema import (
    SCHEMA_VERSION,
    CDSSResponse,
    ConfidenceLevel,
    ResponseMeta,
    RetrievalMeta,
)
from api.summary_mapper import to_patient_summary
from config import LLM_MODEL
from llm.evidence_format import parse_citations
from pipeline.query_graph import CDSS_GRAPH

logger = logging.getLogger(__name__)

_VALID_CONFIDENCE = frozenset({"High", "Medium", "Low"})


def _normalize_confidence(value: str | None) -> ConfidenceLevel:
    if value in _VALID_CONFIDENCE:
        return value  # type: ignore[return-value]
    if value:
        for level in _VALID_CONFIDENCE:
            if level.lower() in value.lower():
                return level  # type: ignore[return-value]
    return "Medium"


def _build_retrieval_meta(state: dict) -> RetrievalMeta | None:
    spec = state.get("retrieval_spec")
    if not spec and state.get("route") != "patient_specific":
        return None
    if spec:
        return RetrievalMeta(
            chunk_count=len(state.get("chunks") or []),
            disease_tags=list(spec.get("disease_tags") or []),
            ckd_stage_tags=list(spec.get("ckd_stage_tags") or []),
            bp_stage_tags=list(spec.get("bp_stage_tags") or []),
            semantic_query=spec.get("semantic_query"),
            tagger_rationale=spec.get("rationale"),
        )
    return RetrievalMeta(chunk_count=len(state.get("chunks") or []))


def run_cdss_pipeline(
    patient_id: str,
    query: str,
    evidence_mode: str = "rag",
    request_id: str | None = None,
) -> dict:
    """
    Run the LangGraph-orchestrated CDSS pipeline and return a v1 CDSSResponse dict.
    """
    req_id = request_id or str(uuid.uuid4())
    started = time.perf_counter()
    logger.info("CDSS pipeline started request_id=%s patient_id=%s", req_id, patient_id)

    state = CDSS_GRAPH.invoke(
        {
            "patient_id": patient_id,
            "query": query,
            "evidence_mode": evidence_mode,
        }
    )
    latency_ms = (time.perf_counter() - started) * 1000

    logger.info(
        "CDSS graph completed request_id=%s route=%s chunks=%s latency_ms=%.2f",
        req_id,
        state.get("route"),
        len(state.get("chunks", [])),
        latency_ms,
    )

    result = dict(state["result"])
    chunks = state.get("chunks") or []

    patient_summary = None
    if state.get("route") == "patient_specific" and state.get("summary"):
        patient_summary = to_patient_summary(state["summary"])

    citations = parse_citations(result.get("citations", []), chunks)

    response = CDSSResponse(
        schema_version=SCHEMA_VERSION,
        recommendation=result.get("recommendation", ""),
        reasoning=result.get("reasoning", ""),
        citations=citations,
        patient_summary=patient_summary,
        llm_summary=result.get("llm_summary"),
        confidence=_normalize_confidence(result.get("confidence")),
        meta=ResponseMeta(
            request_id=req_id,
            route=state.get("route", "general_guideline"),
            route_reason=state.get("route_reason"),
            evidence_mode=state.get("evidence_mode", "rag"),  # type: ignore[arg-type]
            model=LLM_MODEL,
            latency_ms=latency_ms,
            retrieval=_build_retrieval_meta(state),
        ),
    )

    logger.info("CDSS pipeline completed request_id=%s", req_id)
    return response.model_dump(mode="json")
