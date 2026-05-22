"""Keyword tagging + cross-guideline disease hints for unified chunks."""

from __future__ import annotations

import re

DRUG_KEYWORDS: dict[str, list[str]] = {
    "ACEi": [
        "ACE inhibitor", "ACEi", "lisinopril", "enalapril", "ramipril",
        "perindopril", "captopril", "fosinopril",
    ],
    "ARB": [
        "ARB", "angiotensin receptor blocker", "losartan", "valsartan",
        "candesartan", "irbesartan", "telmisartan",
    ],
    "SGLT2i": ["SGLT2", "dapagliflozin", "empagliflozin", "canagliflozin", "sotagliflozin"],
    "MRA": ["MRA", "mineralocorticoid", "spironolactone", "eplerenone", "finerenone"],
    "CCB": ["calcium channel blocker", "CCB", "amlodipine", "nifedipine"],
    "thiazide": ["thiazide", "hydrochlorothiazide", "chlorthalidone", "indapamide"],
    "betablocker": ["beta-blocker", "metoprolol", "carvedilol", "bisoprolol"],
    "GLP1": ["GLP-1", "semaglutide", "liraglutide", "dulaglutide"],
    "statin": ["statin", "atorvastatin", "rosuvastatin", "simvastatin"],
}

COMORBIDITY_KEYWORDS: dict[str, list[str]] = {
    "diabetes": ["diabetes", "T2D", "T1D", "glycemic", "HbA1c"],
    "CKD": ["CKD", "chronic kidney disease", "eGFR", "albuminuria", "proteinuria"],
    "atrial_fibrillation": ["atrial fibrillation", "AF ", "AFib", " AF,"],
    "heart_failure": ["heart failure", "HFrEF", "HFpEF", "ejection fraction"],
    "stroke": ["stroke", "cerebrovascular", "TIA"],
    "pregnancy": ["pregnant", "pregnancy", "gestational", "preeclampsia"],
    "obesity": ["obesity", "obese", "BMI", "overweight"],
    "coronary": ["coronary", "CAD", "angina", "myocardial infarction", "ACS"],
    "PAH": ["pulmonary hypertension", "PAH", "pulmonary arterial"],
    "cyanosis": ["cyanosis", "cyanotic", "oxygen saturation"],
}

# (guideline, section prefix or exact id) -> extra disease_tags (merged with primary disease)
_CROSS_SECTION_DISEASE_TAGS: list[tuple[str, str, list[str]]] = [
    ("HTN", "5.3.1", ["hypertension", "diabetes"]),
    ("HTN", "5.3.8", ["hypertension", "CKD"]),
    ("HTN", "5.3.4", ["hypertension", "heart_failure"]),
    ("HTN", "5.5", ["hypertension", "pregnancy"]),
    ("KDIGO", "3.4", ["CKD", "hypertension"]),
    ("KDIGO", "3.5", ["CKD", "diabetes"]),
    ("KDIGO", "3.15", ["CKD", "coronary", "heart_failure"]),
    ("ACHD", "3.6", ["ACHD", "heart_failure"]),
    ("ACHD", "3.8", ["ACHD", "pregnancy"]),
]

_STAGE_CKDG: list[tuple[str, str]] = [
    (r"\bG1\b|\bstage\s*1\b", "G1"),
    (r"\bG2\b|\bstage\s*2\b", "G2"),
    (r"\bG3a\b", "G3a"),
    (r"\bG3b\b", "G3b"),
    (r"\bG3\b(?![ab])|\bstage\s*3", "G3"),
    (r"\bG4\b|\bstage\s*4\b", "G4"),
    (r"\bG5\b|\bstage\s*5\b", "G5"),
    (r"\bA1\b|\bA2\b|\bA3\b", ""),  # albuminuria categories captured below
]

_ALBUMINURIA_TAGS = [
    (r"\bA1\b", "A1"),
    (r"\bA2\b", "A2"),
    (r"\bA3\b", "A3"),
]

_BP_STAGE_PATTERNS: list[tuple[str, str]] = [
    (r"\bstage\s*1\b|\bstage\s*one\b", "stage1"),
    (r"\bstage\s*2\b|\bstage\s*two\b", "stage2"),
    (r"\belevated\b(?!\s+risk)", "elevated"),
]

_INTERVENTION_PATTERNS: list[tuple[str, str]] = [
    (r"\bclosure\b", "closure"),
    (r"\bsurgery\b|\bsurgical\b", "surgery"),
    (r"\bcatheterization\b|\bcatheter\b", "catheterization"),
    (r"\btransplant\b", "transplant"),
]


def _norm_section_key(section_id: str) -> str:
    return section_id.strip()


def _matches_section_hint(section_id: str, hint: str) -> bool:
    sid = _norm_section_key(section_id)
    if not sid:
        return False
    return sid == hint or sid.startswith(hint + ".")


def _extract_drug_tags(text: str) -> list[str]:
    tl = text.lower()
    out: list[str] = []
    for tag, keywords in DRUG_KEYWORDS.items():
        if any(kw.lower() in tl for kw in keywords):
            out.append(tag)
    return out


def _extract_comorbidity_tags(text: str) -> list[str]:
    tl = text.lower()
    out: list[str] = []
    for tag, keywords in COMORBIDITY_KEYWORDS.items():
        if any(kw.lower() in tl for kw in keywords):
            out.append(tag)
    return out


def _extract_ckd_stage_tags(text: str) -> list[str]:
    tl = text.lower()
    found: list[str] = []
    for pattern, tag in _STAGE_CKDG:
        if not tag:
            continue
        if re.search(pattern, tl, re.IGNORECASE):
            if tag not in found:
                found.append(tag)
    for pattern, tag in _ALBUMINURIA_TAGS:
        if re.search(pattern, text):
            if tag not in found:
                found.append(tag)
    return found


def _extract_bp_stage_tags(text: str) -> list[str]:
    tl = text.lower()
    found: list[str] = []
    for pattern, tag in _BP_STAGE_PATTERNS:
        if re.search(pattern, tl, re.IGNORECASE):
            if tag not in found:
                found.append(tag)
    return found


def _extract_intervention_tags(text: str) -> list[str]:
    tl = text.lower()
    found: list[str] = []
    for pattern, tag in _INTERVENTION_PATTERNS:
        if re.search(pattern, tl, re.IGNORECASE):
            if tag not in found:
                found.append(tag)
    return found


def _primary_disease_tag(disease: str) -> str:
    if disease.lower() in {"hypertension"}:
        return "hypertension"
    return disease


def enrich_chunk(chunk: dict) -> dict:
    """Mutates chunk in place; fills keyword tags and disease_tags."""
    text = chunk.get("text") or ""
    guideline = chunk.get("guideline") or ""
    disease = chunk.get("disease") or ""

    chunk["drug_tags"] = _extract_drug_tags(text)
    chunk["comorbidity_tags"] = _extract_comorbidity_tags(text)

    if guideline in {"KDIGO", "NICE"}:
        chunk["ckd_stage_tags"] = _extract_ckd_stage_tags(text)
        chunk["bp_stage_tags"] = []
        chunk["intervention_tags"] = []
    elif guideline == "HTN":
        chunk["bp_stage_tags"] = _extract_bp_stage_tags(text)
        chunk["ckd_stage_tags"] = []
        chunk["intervention_tags"] = []
    elif guideline == "ACHD":
        chunk["intervention_tags"] = _extract_intervention_tags(text)
        chunk["ckd_stage_tags"] = []
        chunk["bp_stage_tags"] = []
    else:
        chunk["ckd_stage_tags"] = _extract_ckd_stage_tags(text)
        chunk["bp_stage_tags"] = _extract_bp_stage_tags(text)
        chunk["intervention_tags"] = _extract_intervention_tags(text)

    primary = _primary_disease_tag(disease)
    chunk["disease_tags"] = [primary] if primary else []

    section_id = chunk.get("section_id") or ""
    for gl, hint, extra in _CROSS_SECTION_DISEASE_TAGS:
        if gl != guideline:
            continue
        if _matches_section_hint(section_id, hint):
            for t in extra:
                if t not in chunk["disease_tags"]:
                    chunk["disease_tags"].append(t)

    return chunk


def enrich_chunks(chunks: list[dict]) -> list[dict]:
    for c in chunks:
        enrich_chunk(c)
    return chunks
