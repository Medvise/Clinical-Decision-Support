"""Infer guideline disease_tags from ICD codes, labs, and clinical flags."""

from __future__ import annotations

from rules.ckd_classifier import is_ckd_confirmed
from rules.utils import safe_float

# Aligns with Qdrant payload disease_tags.
_ICD_PREFIXES: dict[str, tuple[str, ...]] = {
    "CKD": (
        "N181", "N182", "N183", "N184", "N185", "N186", "N189", "N18",
    ),
    "hypertension": (
        "I10", "I11", "I12", "I13", "I15", "I16",
    ),
    "ACHD": (
        "Q20", "Q21", "Q22", "Q23", "Q24", "Q25", "Q26", "Q27", "Q28",
    ),
    "diabetes": (
        "E10", "E11", "E13", "E14",
    ),
    "stroke": (
        "I60", "I61", "I62", "I63", "I64", "I65", "I66", "I67", "I68", "I69", "G45",
    ),
    "dyslipidemia": (
        "E78",
    ),
}

_COMORBIDITY_ICD_PREFIXES: dict[str, tuple[str, ...]] = {
    "diabetes": ("E10", "E11", "E13", "E14"),
    "heart_failure": ("I50",),
    "coronary": ("I21", "I22", "I23", "I24", "I25"),
    "pregnancy": ("O09", "O10", "O11", "O12", "O13", "O14", "O15", "O16", "Z33", "Z34"),
    "atrial_fibrillation": ("I48",),
}

CKD_LAB_EGFR_THRESHOLD = 90
HTN_SBP_THRESHOLD = 120
HTN_DBP_THRESHOLD = 80

DISEASE_PRIORITY = ("CKD", "ACHD", "hypertension", "diabetes", "stroke", "dyslipidemia")


def _normalize_icd(code: str) -> str:
    return (code or "").strip().upper().replace(".", "")


def _icd_matches_prefix(code: str, prefix: str) -> bool:
    norm = _normalize_icd(code)
    pref = prefix.upper().replace(".", "")
    return norm.startswith(pref)


def _has_icd_prefix(icd_codes: list[str], prefixes: tuple[str, ...]) -> bool:
    return any(
        _icd_matches_prefix(code, pref)
        for code in icd_codes
        for pref in prefixes
    )


def infer_patient_disease_tags(
    icd_codes: list[str],
    key_values: dict | None = None,
    flags: list[str] | None = None,
) -> list[str]:
    """
    Derive Qdrant disease_tags from patient data (no hardcoded primary disease).

    CKD: N18.x ICDs, is_ckd_confirmed(), or labs (eGFR < 60 / ACR ≥ 30 with eGFR).
    Hypertension: I10–I16 ICDs, or BP thresholds, or BP-related flags.
    ACHD: Q20–Q28 congenital heart ICDs.
    diabetes/stroke/dyslipidemia: inferred from ICD coding and clinical flags.
    """
    key_values = key_values or {}
    flags = flags or []
    codes = [str(c).strip() for c in icd_codes if c and str(c).strip()]
    tags: list[str] = []

    egfr = safe_float(key_values.get("egfr"))
    urine_acr = safe_float(key_values.get("urine_acr"))
    systolic = safe_float(key_values.get("systolic_bp"))
    diastolic = safe_float(key_values.get("diastolic_bp"))

    ckd_by_icd = is_ckd_confirmed(codes) or _has_icd_prefix(codes, _ICD_PREFIXES["CKD"])
    ckd_by_labs = (egfr is not None and egfr < CKD_LAB_EGFR_THRESHOLD) or (
        egfr is not None and urine_acr is not None and urine_acr >= 30
    )
    if ckd_by_icd or ckd_by_labs:
        tags.append("CKD")

    if _has_icd_prefix(codes, _ICD_PREFIXES["hypertension"]):
        tags.append("hypertension")
    elif systolic is not None and systolic >= HTN_SBP_THRESHOLD:
        tags.append("hypertension")
    elif diastolic is not None and diastolic >= HTN_DBP_THRESHOLD:
        tags.append("hypertension")

    flags_text = " ".join(flags).lower()
    if "hypertension" not in tags and any(
        kw in flags_text
        for kw in ("uncontrolled bp", "suboptimal bp", "hypertension", " htn", "htn ")
    ):
        tags.append("hypertension")

    if _has_icd_prefix(codes, _ICD_PREFIXES["ACHD"]):
        tags.append("ACHD")

    if _has_icd_prefix(codes, _ICD_PREFIXES["diabetes"]):
        tags.append("diabetes")
    if _has_icd_prefix(codes, _ICD_PREFIXES["stroke"]):
        tags.append("stroke")
    if _has_icd_prefix(codes, _ICD_PREFIXES["dyslipidemia"]):
        tags.append("dyslipidemia")

    if "diabetes" not in tags and any(
        kw in flags_text for kw in ("diabetes", "hyperglycemia", "hba1c", "glycemic")
    ):
        tags.append("diabetes")
    if "stroke" not in tags and any(
        kw in flags_text for kw in ("stroke", "tia", "cerebrovascular")
    ):
        tags.append("stroke")

    return tags


def infer_patient_comorbidity_tags(icd_codes: list[str]) -> list[str]:
    """Comorbidity payload tags for optional Qdrant should-boost filters."""
    codes = [str(c).strip() for c in icd_codes if c and str(c).strip()]
    tags: list[str] = []
    for tag, prefixes in _COMORBIDITY_ICD_PREFIXES.items():
        if _has_icd_prefix(codes, prefixes):
            tags.append(tag)
    return tags


def primary_disease_tag(disease_tags: list[str]) -> str | None:
    """Most specific primary label for legacy `disease` field."""
    for disease in DISEASE_PRIORITY:
        if disease in disease_tags:
            return disease
    return disease_tags[0] if disease_tags else None
