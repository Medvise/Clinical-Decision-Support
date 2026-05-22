"""LLM pass to derive Qdrant retrieval filters from patient context and query."""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field, field_validator

from config import LLM_MODEL, RETRIEVAL_TAG_MODEL
from llm.llm_client import RETRIEVAL_TAGGER_SYSTEM_PROMPT, call_llm_json, get_openai_client
from llm.prompt_builder import _patient_context_block
from pipeline.logging_utils import log_prompt
from retrieval.retriever import decompose_query
from retrieval.tags_vocab import (
    ALLOWED_BP_STAGE_TAGS,
    ALLOWED_CKD_STAGE_TAGS,
    ALLOWED_COMORBIDITY_TAGS,
    ALLOWED_DISEASE_TAGS,
    ALLOWED_DRUG_TAGS,
    DEFAULT_CHUNK_TYPES,
)
from rules.disease_inference import infer_patient_disease_tags, primary_disease_tag

logger = logging.getLogger(__name__)


class RetrievalTagSpec(BaseModel):
    disease_tags: list[str] = Field(default_factory=list)
    ckd_stage_tags: list[str] = Field(default_factory=list)
    bp_stage_tags: list[str] = Field(default_factory=list)
    comorbidity_tags: list[str] = Field(default_factory=list)
    drug_tags: list[str] = Field(default_factory=list)
    chunk_types: list[str] = Field(default_factory=lambda: list(DEFAULT_CHUNK_TYPES))
    semantic_query: str = ""
    rationale: str | None = None

    @field_validator("disease_tags", mode="before")
    @classmethod
    def _norm_disease(cls, v):
        if not v:
            return []
        out: list[str] = []
        for item in v:
            tag = str(item).strip()
            if tag.lower() in {"htn"}:
                tag = "hypertension"
            if tag in ALLOWED_DISEASE_TAGS and tag not in out:
                out.append(tag)
        return out[:5]


def _cap_filter_list(items: list[str], allowed: frozenset[str], limit: int = 8) -> list[str]:
    out: list[str] = []
    for item in items:
        tag = str(item).strip()
        if tag in allowed and tag not in out:
            out.append(tag)
    return out[:limit]


def _validate_spec(spec: RetrievalTagSpec) -> RetrievalTagSpec:
    return RetrievalTagSpec(
        disease_tags=spec.disease_tags,
        ckd_stage_tags=_cap_filter_list(spec.ckd_stage_tags, ALLOWED_CKD_STAGE_TAGS),
        bp_stage_tags=_cap_filter_list(spec.bp_stage_tags, ALLOWED_BP_STAGE_TAGS),
        comorbidity_tags=_cap_filter_list(spec.comorbidity_tags, ALLOWED_COMORBIDITY_TAGS),
        drug_tags=_cap_filter_list(spec.drug_tags, ALLOWED_DRUG_TAGS),
        chunk_types=[
            t for t in (spec.chunk_types or DEFAULT_CHUNK_TYPES) if str(t).strip()
        ][:12] or list(DEFAULT_CHUNK_TYPES),
        semantic_query=(spec.semantic_query or "").strip(),
        rationale=spec.rationale,
    )


def _baseline_disease_tags(summary: dict, user_query: str) -> list[str]:
    rc = summary.get("retrieval_context") or {}
    tags = list(rc.get("disease_tags") or [])
    if not tags:
        kv = summary.get("key_values") or {}
        tags = infer_patient_disease_tags(
            summary.get("icd_codes") or [],
            kv,
            summary.get("flags") or [],
        )
    from retrieval.retriever import infer_disease_tags_from_query

    for t in infer_disease_tags_from_query(user_query):
        if t not in tags:
            tags.append(t)
    return tags


def _fallback_spec(summary: dict, user_query: str) -> RetrievalTagSpec:
    semantic_query, payload_filters, _ = decompose_query(summary, user_query)
    disease_tags = payload_filters.get("disease_tags") or _baseline_disease_tags(summary, user_query)
    return RetrievalTagSpec(
        disease_tags=disease_tags,
        ckd_stage_tags=list(payload_filters.get("ckd_stage_tags") or []),
        bp_stage_tags=list(payload_filters.get("bp_stage_tags") or []),
        comorbidity_tags=list(payload_filters.get("comorbidity_tags") or []),
        drug_tags=list(payload_filters.get("drug_tags") or []),
        chunk_types=list(payload_filters.get("chunk_types") or DEFAULT_CHUNK_TYPES),
        semantic_query=semantic_query,
        rationale="fallback:decompose_query",
    )


def _build_tagger_prompt(summary: dict, user_query: str) -> str:
    patient_block = _patient_context_block(summary)
    rc = summary.get("retrieval_context") or {}
    return f"""

        ## PERSONA AND OBJECTIVE ##    
        You are a medical expert who can read the medical records and extract the disease tags, comorbidity tags, and other relevant information. 
        You extract structured retrieval filters for a clinical guideline search engine.

        Allowed values (use ONLY these exact strings):
        - disease_tags: CKD, hypertension, ACHD
        - ckd_stage_tags: G1, G2, G3, G3a, G3b, G4, G5, A1, A2, A3
        - bp_stage_tags: normal, elevated, stage1, stage2, severe
        - comorbidity_tags: diabetes, heart_failure, coronary, pregnancy, atrial_fibrillation, stroke, obesity, PAH, cyanosis
        - drug_tags: ACEi, ARB, SGLT2i, MRA, CCB, thiazide, betablocker, GLP1, statin
        - chunk_types: recommendation, practice_point, rationale, synopsis, supportive_text

        ## RULES ##
        1. Set disease_tags from ICD codes, labs, BP, flags, and the clinical question.
        2. If CKD is present, set ckd_stage_tags from eGFR/stage in context (e.g. G3b).
        3. If hypertension is present, set bp_stage_tags from BP values (e.g. stage2).
        4. Leave ckd_stage_tags empty if not CKD; leave bp_stage_tags empty if not hypertension.
        5. semantic_query: one line optimized for semantic search (diseases, stage, key labs, question intent).
        6. Do not invent data not supported by the patient context or question.

        Baseline disease_tags from rules (may refine): {rc.get("disease_tags", [])}
        Primary disease hint: {rc.get("disease") or primary_disease_tag(rc.get("disease_tags") or [])}

        {patient_block}

        ## CLINICAL QUESTION ##
        {user_query}

        ## RESPONSE FORMAT ##
        Respond ONLY with JSON:
        {{
        "disease_tags": [],
        "ckd_stage_tags": [],
        "bp_stage_tags": [],
        "comorbidity_tags": [],
        "drug_tags": [],
        "chunk_types": ["recommendation", "practice_point", "rationale"],
        "semantic_query": "",
        "rationale": "brief note"
        }}
"""


def extract_retrieval_tags(patient_summary: dict, user_query: str) -> RetrievalTagSpec:
    """LLM-derived retrieval filters with validation and decompose_query fallback."""
    if get_openai_client() is None:
        logger.warning("OPENAI_API_KEY missing; using decompose_query fallback for retrieval tags")
        return _validate_spec(_fallback_spec(patient_summary, user_query))

    prompt = _build_tagger_prompt(patient_summary, user_query)
    log_prompt(
        "retrieval_tagger",
        prompt,
        extra={"patient_id": patient_summary.get("patient_id"), "query": user_query[:200]},
    )
    model = RETRIEVAL_TAG_MODEL or LLM_MODEL
    try:
        parsed = call_llm_json(
            prompt,
            model=model,
            system=RETRIEVAL_TAGGER_SYSTEM_PROMPT,
            temperature=0,
            max_tokens=800,
        )
        spec = _validate_spec(RetrievalTagSpec.model_validate(parsed))
    except Exception as exc:
        logger.warning("Retrieval tagger failed (%s); using decompose_query fallback", exc)
        return _validate_spec(_fallback_spec(patient_summary, user_query))

    if not spec.disease_tags:
        spec = spec.model_copy(update={"disease_tags": _baseline_disease_tags(patient_summary, user_query)})

    if not spec.semantic_query.strip():
        sq, _, _ = decompose_query(patient_summary, user_query)
        spec = spec.model_copy(update={"semantic_query": sq})

    logger.info(
        "Retrieval tags: disease=%s ckd_stage=%s bp_stage=%s query=%r",
        spec.disease_tags,
        spec.ckd_stage_tags,
        spec.bp_stage_tags,
        spec.semantic_query[:120],
    )
    return spec
