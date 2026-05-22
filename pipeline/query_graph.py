import logging
import json
import re
import time
from typing import Any, Literal
from typing_extensions import TypedDict

from langgraph.graph import END, StateGraph

from data.patient_fetcher import fetch_patient_from_gold
from llm.evidence_format import enrich_citations_with_evidence
from llm.llm_client import call_llm
from llm.prompt_builder import (
    assemble_general_prompt,
    assemble_general_synthesis_prompt,
    assemble_patient_synthesis_prompt,
    assemble_prompt,
)
from llm.retrieval_tagger import extract_retrieval_tags
from pipeline.logging_utils import log_prompt
from retrieval.retriever import (
    infer_comorbidity_tags_from_query,
    infer_disease_tags_from_query,
    retrieve_guideline_chunks,
)
from rules.patient_summary import build_patient_summary

logger = logging.getLogger(__name__)

Route = Literal["patient_specific", "general_guideline", "clarification_needed"]


EvidenceMode = Literal["rag", "llm_synthesis"]


class CDSSState(TypedDict, total=False):
    patient_id: str
    query: str
    evidence_mode: EvidenceMode
    route: Route
    route_reason: str
    summary: dict[str, Any] | None
    retrieval_spec: dict[str, Any] | None
    chunk_records: list[dict[str, Any]] | None
    chunks: list[str]
    prompt: str
    result: dict[str, Any]


_PATIENT_HINT_PATTERNS = [
    r"\bthis patient\b",
    # r"\bfor (him|her|them)\b",
    # r"\bmy patient\b",
    # r"\bcurrent medications?\b",
    # r"\bwhat should we do now\b",
    # r"\bcan (he|she|they) (start|continue|stop)\b",
]

_GENERAL_HINT_PATTERNS = [
    r"\bwhat is\b",
    r"\bdefine\b",
    r"\bguideline\b",
    r"\bkdigo\b",
    r"\bnice\b",
    r"\boverview\b",
    r"\bexplain\b",
    r"\bdifference between\b",
]


def _classify_route(patient_id: str, query: str) -> tuple[Route, str]:
    q = (query or "").strip()
    if not q:
        return "patient_specific", "empty query defaulted to patient-specific flow"

    patient_score = sum(
        1 for p in _PATIENT_HINT_PATTERNS if re.search(p, q, flags=re.IGNORECASE)
    )
    general_score = sum(
        1 for p in _GENERAL_HINT_PATTERNS if re.search(p, q, flags=re.IGNORECASE)
    )

    if not patient_id and patient_score > 0:
        return (
            "clarification_needed",
            f"patient cues={patient_score} with missing patient_id; clarification required",
        )
    if not patient_id:
        return "general_guideline", "patient_id missing; inferred general-guideline flow"

    if patient_score > general_score:
        return "patient_specific", f"patient cues={patient_score}, general cues={general_score}"
    if general_score > patient_score:
        return "general_guideline", f"general cues={general_score}, patient cues={patient_score}"

    # Conservative tie-breaker: in this API patient_id is usually present, so default to patient flow.
    return "patient_specific", f"tie score={patient_score}; default patient-specific route"


def route_query(state: CDSSState) -> CDSSState:
    started = time.perf_counter()
    route, reason = _classify_route(state.get("patient_id", ""), state.get("query", ""))
    latency_ms = (time.perf_counter() - started) * 1000
    logger.info("Query routed to %s (%s) latency_ms=%.2f", route, reason, latency_ms)
    return {"route": route, "route_reason": reason}


def build_clarification_result(state: CDSSState) -> CDSSState:
    started = time.perf_counter()
    query = state.get("query", "")
    result = {
        "recommendation": (
            "I need patient context to answer this as a patient-specific medication decision. "
            "Please provide a patient_id (uniqueempi) and optionally the latest labs/medications."
        ),
        "reasoning": (
            "Your question appears patient-specific, but no patient_id was supplied, "
            "so patient summary and stage-aware guideline retrieval cannot be performed safely."
        ),
        "citations": [],
        "confidence": "Low",
    }
    latency_ms = (time.perf_counter() - started) * 1000
    logger.info("Clarification response built latency_ms=%.2f", latency_ms)
    return {"result": result}


def build_patient_context(state: CDSSState) -> CDSSState:
    started = time.perf_counter()
    patient_id = state["patient_id"]
    patient = fetch_patient_from_gold(patient_id)
    summary = build_patient_summary(patient)
    logger.info(
        "Patient summary JSON for patient_id=%s: %s",
        patient_id,
        json.dumps(summary, default=str),
    )
    latency_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "Patient context ready for patient_id=%s (stage=%s acr=%s) latency_ms=%.2f",
        patient_id,
        summary.get("stage"),
        summary.get("acr_category"),
        latency_ms,
    )
    return {"summary": summary}


def extract_retrieval_tags_node(state: CDSSState) -> CDSSState:
    started = time.perf_counter()
    summary = state.get("summary") or {}
    query = state.get("query", "")
    spec = extract_retrieval_tags(summary, query)
    latency_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "Retrieval tags ready disease=%s latency_ms=%.2f",
        spec.disease_tags,
        latency_ms,
    )
    return {"retrieval_spec": spec.model_dump()}


def skip_guideline_retrieval(state: CDSSState) -> CDSSState:
    logger.info("Skipping Qdrant retrieval (evidence_mode=llm_synthesis)")
    return {"chunks": []}


def retrieve_patient_chunks(state: CDSSState) -> CDSSState:
    started = time.perf_counter()
    summary = state["summary"] or {}
    spec = state.get("retrieval_spec") or {}
    semantic_query = spec.get("semantic_query") or state.get("query", "")
    payload_filters = {
        "disease_tags": spec.get("disease_tags") or [],
        "ckd_stage_tags": spec.get("ckd_stage_tags") or [],
        "bp_stage_tags": spec.get("bp_stage_tags") or [],
        "chunk_types": spec.get("chunk_types") or [
            "recommendation",
            "practice_point",
            "rationale",
        ],
        "drug_tags": spec.get("drug_tags") or [],
        "comorbidity_tags": spec.get("comorbidity_tags") or [],
    }
    outcome = retrieve_guideline_chunks(
        query=semantic_query,
        stage=summary.get("stage") or summary.get("ckd_stage", ""),
        payload_filters=payload_filters,
    )
    latency_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "Patient-specific retrieval returned %s chunks latency_ms=%.2f",
        len(outcome.chunks),
        latency_ms,
    )
    return {"chunks": outcome.chunks, "chunk_records": outcome.records}


def retrieve_general_chunks(state: CDSSState) -> CDSSState:
    started = time.perf_counter()
    query = state["query"]
    if state.get("evidence_mode") == "llm_synthesis":
        logger.info("Skipping general Qdrant retrieval (evidence_mode=llm_synthesis)")
        return {"chunks": []}
    disease_tags = infer_disease_tags_from_query(query)
    outcome = retrieve_guideline_chunks(
        query=f"clinical guideline evidence {query}",
        payload_filters={
            "disease_tags": disease_tags,
            "ckd_stage_tags": [],
            "bp_stage_tags": [],
            "chunk_types": [
                "recommendation",
                "practice_point",
                "rationale",
                "synopsis",
                "supportive_text",
                "section_summary",
                "table",
                "rec",
                "pp",
                "summary",
            ],
            "drug_tags": [],
            "comorbidity_tags": infer_comorbidity_tags_from_query(query),
        },
    )
    latency_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "General-guideline retrieval returned %s chunks latency_ms=%.2f",
        len(outcome.chunks),
        latency_ms,
    )
    return {"chunks": outcome.chunks, "chunk_records": outcome.records}


def build_prompt(state: CDSSState) -> CDSSState:
    started = time.perf_counter()
    route = state["route"]
    chunks = state.get("chunks", [])
    query = state["query"]
    ev_mode = state.get("evidence_mode", "rag")
    if route == "patient_specific":
        if ev_mode == "llm_synthesis":
            prompt = assemble_patient_synthesis_prompt(state["summary"] or {}, query)
        else:
            prompt = assemble_prompt(state["summary"] or {}, chunks, query)
    else:
        if ev_mode == "llm_synthesis":
            prompt = assemble_general_synthesis_prompt(query)
        else:
            prompt = assemble_general_prompt(chunks, query)
    label = f"answer_{route}_{ev_mode}"
    log_prompt(
        label,
        prompt,
        extra={
            "route": route,
            "evidence_mode": ev_mode,
            "chunk_count": len(chunks),
        },
    )
    latency_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "Prompt assembled for route=%s evidence_mode=%s latency_ms=%.2f",
        route,
        ev_mode,
        latency_ms,
    )
    return {"prompt": prompt}


def call_model(state: CDSSState) -> CDSSState:
    started = time.perf_counter()
    synthesis = state.get("evidence_mode") == "llm_synthesis"
    result = call_llm(state["prompt"], synthesis=synthesis)
    latency_ms = (time.perf_counter() - started) * 1000
    logger.info("LLM call completed for route=%s latency_ms=%.2f", state["route"], latency_ms)
    chunks = state.get("chunks", [])
    result = enrich_citations_with_evidence(result, chunks)
    return {"result": result}


def _after_patient_context(state: CDSSState) -> str:
    if state.get("evidence_mode") == "llm_synthesis":
        return "skip_guideline_retrieval"
    return "extract_retrieval_tags"


def _next_after_route(state: CDSSState) -> str:
    if state["route"] == "patient_specific":
        return "build_patient_context"
    if state["route"] == "general_guideline":
        return "retrieve_general_chunks"
    return "build_clarification_result"


def build_cdss_graph():
    graph = StateGraph(CDSSState)
    graph.add_node("route_query", route_query)
    graph.add_node("build_patient_context", build_patient_context)
    graph.add_node("extract_retrieval_tags", extract_retrieval_tags_node)
    graph.add_node("retrieve_patient_chunks", retrieve_patient_chunks)
    graph.add_node("skip_guideline_retrieval", skip_guideline_retrieval)
    graph.add_node("retrieve_general_chunks", retrieve_general_chunks)
    graph.add_node("build_clarification_result", build_clarification_result)
    graph.add_node("build_prompt", build_prompt)
    graph.add_node("call_model", call_model)

    graph.set_entry_point("route_query")
    graph.add_conditional_edges("route_query", _next_after_route)
    graph.add_conditional_edges(
        "build_patient_context",
        _after_patient_context,
        {
            "extract_retrieval_tags": "extract_retrieval_tags",
            "skip_guideline_retrieval": "skip_guideline_retrieval",
        },
    )
    graph.add_edge("extract_retrieval_tags", "retrieve_patient_chunks")
    graph.add_edge("retrieve_patient_chunks", "build_prompt")
    graph.add_edge("skip_guideline_retrieval", "build_prompt")
    graph.add_edge("retrieve_general_chunks", "build_prompt")
    graph.add_edge("build_clarification_result", END)
    graph.add_edge("build_prompt", "call_model")
    graph.add_edge("call_model", END)
    return graph.compile()


CDSS_GRAPH = build_cdss_graph()
