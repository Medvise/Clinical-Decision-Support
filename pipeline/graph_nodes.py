"""LangGraph node implementations for the CDSS pipeline."""

from __future__ import annotations

import json
import logging
import time

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
from pipeline.graph_state import CDSSState
from pipeline.logging_utils import log_prompt
from retrieval.retriever import (
    infer_comorbidity_tags_from_query,
    infer_disease_tags_from_query,
    retrieve_guideline_chunks,
)
from rules.patient_summary import build_patient_summary

logger = logging.getLogger(__name__)


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


def after_patient_context(state: CDSSState) -> str:
    if state.get("evidence_mode") == "llm_synthesis":
        return "skip_guideline_retrieval"
    return "extract_retrieval_tags"
