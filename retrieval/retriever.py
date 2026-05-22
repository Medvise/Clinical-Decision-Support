# retrieval/retriever.py
import logging
import requests
from typing import Any, NamedTuple

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny
from openai import OpenAI

from llm.evidence_format import pack_evidence_block
from pipeline.logging_utils import log_retrieved_chunks
from config import (
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION,
    OPENAI_API_KEY,
    EMBED_MODEL,
    EMBEDDINGS_URL,
    TOP_K_CHUNKS,
    EMBEDDINGS_TIMEOUT_SECONDS,
    RERANK_ENABLED,
    RERANK_MODEL,
    RERANK_DEVICE,
    RERANK_BATCH_SIZE,
    RERANK_TOP_K,
)

_qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
_openai = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
logger = logging.getLogger(__name__)
_reranker_model = None
_reranker_unavailable = False

TIER1_REC_CHUNK_TYPES = frozenset({"recommendation", "practice_point", "rec", "pp"})
TIER2_NARRATIVE_CHUNK_TYPES = frozenset({"synopsis", "supportive_text", "rationale"})


class RetrievalOutcome(NamedTuple):
    """Packed evidence strings for the LLM prompt plus score/metadata for logging."""

    chunks: list[str]
    records: list[dict[str, Any]]


def _detect_query_intent(user_query: str) -> str:
    q = (user_query or "").lower()
    if any(t in q for t in ("medication", "medicine", "drug", "currently on", "current med", "prescri")):
        return "medication_list"
    if any(t in q for t in ("dose", "dosage", "adjust", "titrate", "up-titrate", "down-titrate")):
        return "dosing"
    if any(t in q for t in ("why", "reason", "because", "rationale")):
        return "reasoning"
    return "treatment_recommendation"


_DRUG_CLASS_KEYWORDS: dict[str, list[str]] = {
    "ACEi": ["ramipril", "lisinopril", "enalapril", "perindopril",
             "captopril", "fosinopril", "ace inhibitor", "acei"],
    "ARB": ["losartan", "valsartan", "candesartan", "irbesartan",
            "telmisartan", "arb"],
    "SGLT2i": ["dapagliflozin", "empagliflozin", "canagliflozin",
               "sotagliflozin", "sglt2"],
    "MRA": ["spironolactone", "eplerenone", "finerenone", "mra"],
    "statin": ["atorvastatin", "rosuvastatin", "simvastatin", "statin"],
}


def _extract_drug_classes_from_flags(flags: list[str]) -> list[str]:
    combined = " ".join(flags).lower()
    return [cls for cls, kws in _DRUG_CLASS_KEYWORDS.items()
            if any(kw in combined for kw in kws)]


def _map_stage_to_payload_stage(stage: str | None) -> list[str]:
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


_QUERY_DISEASE_KEYWORDS: dict[str, list[str]] = {
    "CKD": [
        "ckd", "chronic kidney", "kidney disease", "egfr", "albuminuria",
        "kdigo", "nice", "nephro",
    ],
    "hypertension": [
        "hypertension", "blood pressure", " bp ", "htn", "antihypertensive",
    ],
    "ACHD": [
        "achd", "congenital heart", "fontan", "cyanotic heart",
    ],
}

_COMORBIDITY_QUERY_KEYWORDS: dict[str, list[str]] = {
    "diabetes": ["diabetes", "diabetic", "glycemic", "hba1c", "t2d"],
    "heart_failure": ["heart failure", "hfref", "hfpef", "bnp", "nt-probnp"],
    "pregnancy": ["pregnant", "pregnancy", "gestational", "preeclampsia"],
    "coronary": ["coronary", "cad", "acs", "myocardial infarction"],
}


def _normalize_disease_tag(label: str) -> str:
    lowered = (label or "").strip().lower()
    if lowered in {"hypertension", "htn"}:
        return "hypertension"
    if lowered in {"ckd", "chronic kidney disease"}:
        return "CKD"
    if lowered in {"achd", "congenital heart disease"}:
        return "ACHD"
    return label.strip()


def _infer_disease_tags_from_query(user_query: str) -> list[str]:
    q = (user_query or "").lower()
    tags: list[str] = []
    for tag, keywords in _QUERY_DISEASE_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            tags.append(tag)
    return tags


def _infer_comorbidity_tags_from_query(user_query: str) -> list[str]:
    q = (user_query or "").lower()
    tags: list[str] = []
    for tag, keywords in _COMORBIDITY_QUERY_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            tags.append(tag)
    return tags


def _infer_disease_tags(patient_summary: dict, user_query: str) -> list[str]:
    """
    Merge patient-derived disease_tags (from rules.disease_inference) with query hints.
    """
    rc = patient_summary.get("retrieval_context") or {}
    tags: list[str] = []
    for t in rc.get("disease_tags") or []:
        norm = _normalize_disease_tag(t)
        if norm and norm not in tags:
            tags.append(norm)

    for t in _infer_disease_tags_from_query(user_query):
        if t not in tags:
            tags.append(t)

    return tags


def _infer_comorbidity_tags(patient_summary: dict, user_query: str) -> list[str]:
    rc = patient_summary.get("retrieval_context") or {}
    tags: list[str] = list(rc.get("comorbidity_tags") or [])
    for t in _infer_comorbidity_tags_from_query(user_query):
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
    intent = _detect_query_intent(user_query)
    rc = patient_summary.get("retrieval_context") or {}
    stage = patient_summary.get("stage") or patient_summary.get("ckd_stage") or rc.get("stage", "")
    acr_cat = patient_summary.get("acr_category") or rc.get("acr_category", "")
    flags = patient_summary.get("flags", [])
    kv = patient_summary.get("key_values", {})
    drug_classes = _extract_drug_classes_from_flags(flags)
    disease_tags = _infer_disease_tags(patient_summary, user_query)

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

    comorbidity_tags = _infer_comorbidity_tags(patient_summary, user_query)

    semantic_query = " ".join(parts)

    payload_filters = {
        "disease_tags": disease_tags,
        "ckd_stage_tags": _map_stage_to_payload_stage(stage),
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


def build_clinical_query(patient: dict, user_query: str = "") -> str:
    semantic_query, _, _ = decompose_query(patient, user_query)
    return semantic_query


def infer_disease_tags_from_query(user_query: str) -> list[str]:
    """Public helper for general-guideline routes."""
    return _infer_disease_tags_from_query(user_query)


def infer_comorbidity_tags_from_query(user_query: str) -> list[str]:
    """Public helper for general-guideline routes."""
    return _infer_comorbidity_tags_from_query(user_query)


def _build_qdrant_filter(
    payload_filters: dict,
    *,
    apply_disease_tags: bool = True,
    apply_stage: bool = True,
    apply_bp_stage: bool = True,
) -> Filter:
    must: list = []

    chunk_types = payload_filters.get("chunk_types", [])
    if chunk_types:
        must.append(
            FieldCondition(key="chunk_type", match=MatchAny(any=chunk_types))
        )

    if apply_disease_tags:
        disease_tags = payload_filters.get("disease_tags") or []
        if disease_tags:
            must.append(
                FieldCondition(
                    key="disease_tags",
                    match=MatchAny(any=disease_tags),
                )
            )

    should: list = []

    if apply_stage:
        for stage_tag in payload_filters.get("ckd_stage_tags") or []:
            should.append(
                FieldCondition(
                    key="ckd_stage_tags",
                    match=MatchValue(value=stage_tag),
                )
            )

    if apply_bp_stage:
        bp_tags = payload_filters.get("bp_stage_tags") or []
        if bp_tags:
            should.append(
                FieldCondition(
                    key="bp_stage_tags",
                    match=MatchAny(any=bp_tags),
                )
            )

    for tag in payload_filters.get("drug_tags", []):
        should.append(
            FieldCondition(key="drug_tags", match=MatchValue(value=tag))
        )

    comorbidity_tags = payload_filters.get("comorbidity_tags") or []
    if comorbidity_tags:
        should.append(
            FieldCondition(
                key="comorbidity_tags",
                match=MatchAny(any=comorbidity_tags),
            )
        )

    return Filter(must=must, should=should if should else None)


def _embed_query(query: str) -> list[float]:
    if EMBEDDINGS_URL:
        resp = requests.post(
            EMBEDDINGS_URL,
            json={"sentences": [query]},
            timeout=(10, EMBEDDINGS_TIMEOUT_SECONDS),
        )
        try:
            resp.raise_for_status()
        except requests.HTTPError as exc:
            body = (resp.text or "").strip()[:500]
            raise RuntimeError(
                f"Embeddings request failed ({resp.status_code}) for {EMBEDDINGS_URL}. "
                f"Response body: {body or '<empty>'}"
            ) from exc
        embeddings = resp.json().get("embeddings", [])
        if not embeddings:
            raise RuntimeError("Embeddings response missing 'embeddings' values.")
        return embeddings[0]

    if _openai is None:
        raise RuntimeError(
            "No embedding provider configured. Set EMBEDDINGS_URL or OPENAI_API_KEY."
        )
    return _openai.embeddings.create(input=query, model=EMBED_MODEL).data[0].embedding


def _expected_vector_dim() -> int | None:
    info = _qdrant.get_collection(QDRANT_COLLECTION)
    vectors = info.config.params.vectors
    if hasattr(vectors, "size"):
        return vectors.size
    if isinstance(vectors, dict):
        first = next(iter(vectors.values()), None)
        return getattr(first, "size", None)
    return None


def _get_reranker_model():
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


def _rerank_chunks(query: str, chunks: list[dict], top_k: int) -> list[dict]:
    model = _get_reranker_model()
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


def _query_chunks(
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


def _format_tier1_source_line(row: dict) -> str:
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


def _fetch_tier2_narrative_for_recs(rec_ids: list[str]) -> list[dict]:
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


def retrieve_guideline_chunks(
    query: str,
    stage: str = "",
    top_k: int = TOP_K_CHUNKS,
    payload_filters: dict | None = None,
) -> RetrievalOutcome:
    if payload_filters is None:
        payload_filters = {
            "disease_tags": _infer_disease_tags_from_query(query),
            "ckd_stage_tags": _map_stage_to_payload_stage(stage),
            "chunk_types": ["recommendation", "practice_point", "rationale"],
            "drug_tags": [],
            "comorbidity_tags": _infer_comorbidity_tags_from_query(query),
        }

    logger.info(
        "Retrieval started: disease_tags=%s stage=%s top_k=%s query=%s",
        payload_filters.get("disease_tags"),
        stage,
        top_k,
        query,
    )

    embedding = _embed_query(query)
    logger.info("Embedding generated (dim=%s)", len(embedding))

    expected_dim = _expected_vector_dim()
    if expected_dim is not None and len(embedding) != expected_dim:
        raise RuntimeError(
            f"Embedding dimension mismatch: collection expects {expected_dim}, "
            f"query has {len(embedding)}. Use same provider/model as ingestion."
        )

    qdrant_filter = _build_qdrant_filter(
        payload_filters, apply_disease_tags=True, apply_stage=True
    )
    tier1_results = _query_chunks(embedding, qdrant_filter, top_k=top_k)
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
        qdrant_filter_ns = _build_qdrant_filter(
            payload_filters,
            apply_disease_tags=True,
            apply_stage=False,
            apply_bp_stage=False,
        )
        tier1_results = _query_chunks(embedding, qdrant_filter_ns, top_k=top_k)
        logger.info("Fallback search (no stage) returned %s chunks", len(tier1_results))

    if not tier1_results and payload_filters.get("disease_tags"):
        logger.info(
            "No results with disease_tags=%s. Retrying without disease filter.",
            payload_filters["disease_tags"],
        )
        qdrant_filter_nd = _build_qdrant_filter(
            payload_filters, apply_disease_tags=False, apply_stage=False
        )
        tier1_results = _query_chunks(embedding, qdrant_filter_nd, top_k=top_k)
        logger.info("Fallback search (no disease) returned %s chunks", len(tier1_results))

    rerank_keep_k = min(RERANK_TOP_K, top_k)
    tier1_results = _rerank_chunks(query=query, chunks=tier1_results, top_k=rerank_keep_k)

    rec_ids = [
        r["rec_id"]
        for r in tier1_results
        if r["chunk_type"] in TIER1_REC_CHUNK_TYPES and r["rec_id"]
    ]
    tier2_rows = _fetch_tier2_narrative_for_recs(rec_ids)
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
            pack_evidence_block(ref, _format_tier1_source_line(r), text)
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
