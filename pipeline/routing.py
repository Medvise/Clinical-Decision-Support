"""Query routing and clarification responses."""

from __future__ import annotations

import logging
import re
import time

from api.schema import RouteType
from pipeline.graph_state import CDSSState

logger = logging.getLogger(__name__)

_PATIENT_HINT_PATTERNS = [
    r"\bthis patient\b",
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


def classify_route(patient_id: str, query: str) -> tuple[RouteType, str]:
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

    return "patient_specific", f"tie score={patient_score}; default patient-specific route"


def route_query(state: CDSSState) -> CDSSState:
    started = time.perf_counter()
    route, reason = classify_route(state.get("patient_id", ""), state.get("query", ""))
    latency_ms = (time.perf_counter() - started) * 1000
    logger.info("Query routed to %s (%s) latency_ms=%.2f", route, reason, latency_ms)
    return {"route": route, "route_reason": reason}


def build_clarification_result(state: CDSSState) -> CDSSState:
    started = time.perf_counter()
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


def next_after_route(state: CDSSState) -> str:
    if state["route"] == "patient_specific":
        return "build_patient_context"
    if state["route"] == "general_guideline":
        return "retrieve_general_chunks"
    return "build_clarification_result"
