"""Rule-based query decomposition and tag inference for retrieval."""

from __future__ import annotations

import logging

from retrieval.tags_vocab import (
    DRUG_FLAG_KEYWORDS,
    QUERY_COMORBIDITY_KEYWORDS,
    QUERY_DISEASE_KEYWORDS,
)

logger = logging.getLogger(__name__)


def detect_query_intent(user_query: str) -> str:
    q = (user_query or "").lower()
    if any(t in q for t in ("medication", "medicine", "drug", "currently on", "current med", "prescri")):
        return "medication_list"
    if any(t in q for t in ("dose", "dosage", "adjust", "titrate", "up-titrate", "down-titrate")):
        return "dosing"
    if any(t in q for t in ("why", "reason", "because", "rationale")):
        return "reasoning"
    return "treatment_recommendation"


def extract_drug_classes_from_flags(flags: list[str]) -> list[str]:
    combined = " ".join(flags).lower()
    return [
        cls
        for cls, kws in DRUG_FLAG_KEYWORDS.items()
        if any(kw in combined for kw in kws)
    ]


def map_stage_to_payload_stage(stage: str | None) -> list[str]:
    """Map patient CKD stage label to payload ckd_stage_tags values."""
    if not stage:
        return []
    mapped = {
        "stage 1": ["G1"],
        "stage 2": ["G2"],
        "stage 3a": ["G3a", "G3"],
        "stage 3b": ["G3b", "G3"],
        "stage 4": ["G4"],
        "stage 5": ["G5"],
    }.get(stage.strip().lower())
    return mapped or []


def normalize_disease_tag(label: str) -> str:
    lowered = (label or "").strip().lower()
    if lowered in {"hypertension", "htn"}:
        return "hypertension"
    if lowered in {"ckd", "chronic kidney disease"}:
        return "CKD"
    if lowered in {"achd", "congenital heart disease"}:
        return "ACHD"
    return label.strip()


def infer_disease_tags_from_query(user_query: str) -> list[str]:
    q = (user_query or "").lower()
    tags: list[str] = []
    for tag, keywords in QUERY_DISEASE_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            tags.append(tag)
    return tags


def infer_comorbidity_tags_from_query(user_query: str) -> list[str]:
    q = (user_query or "").lower()
    tags: list[str] = []
    for tag, keywords in QUERY_COMORBIDITY_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            tags.append(tag)
    return tags


def infer_disease_tags(patient_summary: dict, user_query: str) -> list[str]:
    """Merge patient-derived disease_tags with query hints."""
    rc = patient_summary.get("retrieval_context") or {}
    tags: list[str] = []
    for t in rc.get("disease_tags") or []:
        norm = normalize_disease_tag(t)
        if norm and norm not in tags:
            tags.append(norm)

    for t in infer_disease_tags_from_query(user_query):
        if t not in tags:
            tags.append(t)

    return tags


def infer_comorbidity_tags(patient_summary: dict, user_query: str) -> list[str]:
    rc = patient_summary.get("retrieval_context") or {}
    tags: list[str] = list(rc.get("comorbidity_tags") or [])
    for t in infer_comorbidity_tags_from_query(user_query):
        if t not in tags:
            tags.append(t)
    flags_text = " ".join(patient_summary.get("flags") or []).lower()
    if "diabetes" in flags_text or "diabetic" in flags_text:
        if "diabetes" not in tags:
            tags.append("diabetes")
    if any(kw in flags_text for kw in ("heart failure", "hfref", "hfpef")):
        if "heart_failure" not in tags:
            tags.append("heart_failure")
    return tags


def decompose_query(
    patient_summary: dict,
    user_query: str,
) -> tuple[str, dict, dict]:
    intent = detect_query_intent(user_query)
    rc = patient_summary.get("retrieval_context") or {}
    stage = patient_summary.get("stage") or patient_summary.get("ckd_stage") or rc.get("stage", "")
    acr_cat = patient_summary.get("acr_category") or rc.get("acr_category", "")
    flags = patient_summary.get("flags", [])
    kv = patient_summary.get("key_values", {})
    drug_classes = extract_drug_classes_from_flags(flags)
    disease_tags = infer_disease_tags(patient_summary, user_query)

    parts: list[str] = list(disease_tags) if disease_tags else []
    if "CKD" in disease_tags:
        if stage and stage not in ("Unknown", "Not CKD"):
            parts.append(stage)
        if acr_cat and acr_cat != "Unknown":
            parts.append(f"albuminuria {acr_cat}")

    egfr = kv.get("egfr")
    urine_acr = kv.get("urine_acr")
    potassium = kv.get("potassium")
    systolic_bp = kv.get("systolic_bp")
    diastolic_bp = kv.get("diastolic_bp")
    if egfr is not None:
        parts.append(f"eGFR {egfr}")
    if urine_acr is not None:
        parts.append(f"ACR {urine_acr}")
    if potassium is not None:
        parts.append(f"potassium {potassium}")
    if systolic_bp is not None:
        parts.append(f"systolic BP {systolic_bp}")
    if diastolic_bp is not None:
        parts.append(f"diastolic BP {diastolic_bp}")
    if systolic_bp is not None and diastolic_bp is not None:
        parts.append(f"blood pressure {systolic_bp}/{diastolic_bp}")

    if intent == "medication_list":
        parts += ["medication", "prescribing", "renoprotective therapy"]
        parts += drug_classes
        chunk_types = ["recommendation", "practice_point"]

    elif intent == "dosing":
        parts += ["renal dose adjustment", "dosing guidance"]
        parts += drug_classes
        chunk_types = ["recommendation", "practice_point", "rationale"]

    elif intent == "reasoning":
        parts += ["rationale", "evidence", "certainty of evidence"]
        chunk_types = ["rationale", "synopsis", "supportive_text"]

    else:
        parts += ["treatment recommendation", "management"]
        parts += drug_classes
        chunk_types = ["recommendation", "practice_point", "rationale"]

    if user_query:
        parts.append(user_query.strip())

    comorbidity_tags = infer_comorbidity_tags(patient_summary, user_query)
    semantic_query = " ".join(parts)

    payload_filters = {
        "disease_tags": disease_tags,
        "ckd_stage_tags": map_stage_to_payload_stage(stage),
        "chunk_types": chunk_types,
        "drug_tags": drug_classes,
        "comorbidity_tags": comorbidity_tags,
    }

    td = patient_summary.get("test_dates", {})
    patient_context = {
        "stage": stage,
        "acr_category": acr_cat,
        "flags": flags,
        "labs": kv,
        "test_dates": td,
        "medications": patient_summary.get("medications", []),
        "icd_codes": patient_summary.get("icd_codes", []),
    }

    logger.info(
        "Query decomposed — intent=%s semantic_query=%r "
        "disease_tags=%s ckd_stage_tags=%s chunk_types=%s drug_tags=%s comorbidity_tags=%s",
        intent,
        semantic_query,
        payload_filters["disease_tags"],
        payload_filters["ckd_stage_tags"],
        payload_filters["chunk_types"],
        payload_filters["drug_tags"],
        payload_filters["comorbidity_tags"],
    )

    return semantic_query, payload_filters, patient_context
