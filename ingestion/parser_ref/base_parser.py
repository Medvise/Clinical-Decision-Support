"""
Shared dataclass, tag extraction, and text utilities for guideline parsers.

Canonical copy lives in `ingestion.parser_ref` (imported by `ingestion.script`).
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import List


@dataclass
class Chunk:
    text: str
    chunk_type: str
    guideline: str
    disease: str
    source_doc: str

    section_id: str = ""
    chapter_id: str = ""
    section_title: str = ""

    rec_id: str = ""
    parent_rec_id: str = ""

    cor: str = ""
    loe: str = ""
    kdigo_grade: str = ""

    drug_tags: List[str] = field(default_factory=list)
    comorbidity_tags: List[str] = field(default_factory=list)
    disease_tags: List[str] = field(default_factory=list)
    bp_stage_tags: List[str] = field(default_factory=list)
    ckd_stage_tags: List[str] = field(default_factory=list)
    intervention_tags: List[str] = field(default_factory=list)

    def to_payload(self) -> dict:
        return {
            "text": self.text,
            "chunk_type": self.chunk_type,
            "guideline": self.guideline,
            "disease": self.disease,
            "source_doc": self.source_doc,
            "section_id": self.section_id,
            "chapter_id": self.chapter_id,
            "section_title": self.section_title,
            "rec_id": self.rec_id,
            "parent_rec_id": self.parent_rec_id,
            "cor": self.cor,
            "loe": self.loe,
            "kdigo_grade": self.kdigo_grade,
            "drug_tags": self.drug_tags,
            "comorbidity_tags": self.comorbidity_tags,
            "disease_tags": self.disease_tags,
            "bp_stage_tags": self.bp_stage_tags,
            "ckd_stage_tags": self.ckd_stage_tags,
            "intervention_tags": self.intervention_tags,
        }

    def point_id(self) -> str:
        return str(uuid.uuid4())


DRUG_VOCAB: dict[str, list[str]] = {
    "ACEi": [
        "ace inhibitor", "acei", "lisinopril", "enalapril", "ramipril",
        "perindopril", "captopril", "benazepril", "quinapril",
    ],
    "ARB": [
        "arb", "angiotensin receptor blocker", "losartan", "valsartan",
        "candesartan", "irbesartan", "olmesartan", "telmisartan",
    ],
    "SGLT2i": [
        "sglt2", "dapagliflozin", "empagliflozin", "canagliflozin",
        "ertugliflozin",
    ],
    "MRA": [
        "mra", "mineralocorticoid receptor antagonist", "spironolactone",
        "eplerenone", "finerenone",
    ],
    "CCB": [
        "calcium channel blocker", "ccb", "amlodipine", "nifedipine",
        "felodipine", "diltiazem", "verapamil",
    ],
    "thiazide": [
        "thiazide", "hydrochlorothiazide", "hctz", "chlorthalidone",
        "indapamide",
    ],
    "betablocker": [
        "beta-blocker", "beta blocker", "metoprolol", "carvedilol",
        "bisoprolol", "atenolol", "nebivolol",
    ],
    "GLP1": [
        "glp-1", "glp1", "semaglutide", "liraglutide", "dulaglutide",
        "exenatide", "tirzepatide",
    ],
    "statin": [
        "statin", "atorvastatin", "rosuvastatin", "simvastatin",
        "pravastatin", "lovastatin",
    ],
    "diuretic": ["diuretic", "furosemide", "torsemide", "bumetanide"],
    "anticoag": [
        "anticoagulant", "warfarin", "apixaban", "rivaroxaban",
        "dabigatran", "edoxaban", "heparin",
    ],
    "antidiab": [
        "metformin", "insulin", "sulfonylurea", "glipizide", "glimepiride",
        "sitagliptin", "dpp-4",
    ],
}

COMORBIDITY_VOCAB: dict[str, list[str]] = {
    "diabetes": [
        "diabetes", "t2d", "t1d", "glycemic", "hba1c",
        "hyperglycemia", "hypoglycemia",
    ],
    "CKD": [
        "ckd", "chronic kidney disease", "egfr", "albuminuria",
        "proteinuria", "renal insufficiency", "kidney disease",
    ],
    "atrial_fibrillation": ["atrial fibrillation", "afib", "af ", "a-fib"],
    "heart_failure": [
        "heart failure", "hfref", "hfpef", "hfmref",
        "ejection fraction", "cardiac failure", "lvef",
    ],
    "stroke": ["stroke", "cerebrovascular", "tia", "transient ischemic"],
    "pregnancy": [
        "pregnant", "pregnancy", "gestational", "preeclampsia",
        "eclampsia", "obstetric",
    ],
    "obesity": ["obesity", "obese", "bmi", "overweight", "adiposity"],
    "coronary": [
        "coronary", "cad", "angina", "myocardial infarction",
        "acs", "acute coronary", "pcsk9", "atherosclerosis",
    ],
    "PAH": [
        "pulmonary hypertension", "pah", "pulmonary arterial",
        "pulmonary vascular",
    ],
    "cyanosis": ["cyanosis", "cyanotic", "oxygen saturation", "hypoxemia"],
    "anemia": [
        "anemia", "anaemia", "hemoglobin", "erythropoiesis",
        "esa", "iron deficiency",
    ],
    "metabolic_acidosis": ["metabolic acidosis", "bicarbonate", "serum bicarbonate"],
    "hyperkalemia": ["hyperkalemia", "potassium", "hyperkalaemia"],
    "dyslipidemia": [
        "dyslipidemia", "hyperlipidemia", "ldl", "hdl",
        "cholesterol", "triglyceride",
    ],
}

BP_STAGE_VOCAB: dict[str, list[str]] = {
    "normal": ["normal bp", "normal blood pressure", "<120"],
    "elevated": [
        "elevated bp", "elevated blood pressure", "120 to 129", "prehypertension",
    ],
    "stage1": ["stage 1 hypertension", "stage 1 htn", "130 to 139", "80 to 89"],
    "stage2": ["stage 2 hypertension", "stage 2 htn", "≥140", "≥90"],
    "severe": [
        "severe hypertension", ">180", "hypertensive emergency",
        "hypertensive urgency",
    ],
}

CKD_STAGE_VOCAB: dict[str, list[str]] = {
    "G1": ["g1", "gfr category 1", "gfr ≥90", "stage 1 ckd"],
    "G2": ["g2", "gfr category 2", "60 to 89", "stage 2 ckd"],
    "G3a": ["g3a", "45 to 59"],
    "G3b": ["g3b", "30 to 44"],
    "G3": ["g3 ", "gfr category 3", "stage 3 ckd"],
    "G4": ["g4", "gfr category 4", "15 to 29", "stage 4 ckd"],
    "G5": [
        "g5", "gfr category 5", "gfr <15", "stage 5 ckd", "esrd", "esrf",
        "kidney failure", "dialysis",
    ],
    "A1": ["a1", "acr <30", "normal to mildly increased"],
    "A2": ["a2", "acr 30", "moderately increased"],
    "A3": ["a3", "acr >300", "severely increased", "nephrotic"],
}

INTERVENTION_VOCAB: dict[str, list[str]] = {
    "closure": ["closure", "device closure", "transcatheter closure"],
    "surgery": [
        "surgery", "surgical repair", "open heart", "sternotomy",
        "cardiopulmonary bypass",
    ],
    "catheterization": [
        "catheterization", "catheter", "percutaneous", "intervention",
        "cath lab",
    ],
    "transplant": [
        "transplant", "transplantation", "heart transplant",
        "cardiac transplant",
    ],
    "pacemaker": [
        "pacemaker", "icd", "crt", "defibrillator", "ep study",
        "electrophysiology",
    ],
    "valvuloplasty": ["valvuloplasty", "balloon dilation", "valvotomy"],
    "fontan": ["fontan", "fontan circulation", "single ventricle"],
}

CROSS_DISEASE_MAP: dict[str, tuple[str, list[str]]] = {
    "HTN_CKD": ("HTN", ["ckd", "chronic kidney", "egfr", "albuminuria"]),
    "HTN_diabetes": ("HTN", ["diabetes", "t2d", "glycemic"]),
    "HTN_HF": ("HTN", ["heart failure", "hfref", "hfpef"]),
    "HTN_pregnancy": ("HTN", ["pregnant", "pregnancy", "gestational"]),
    "KDIGO_HTN": ("KDIGO", ["hypertension", "blood pressure", "bp control"]),
    "KDIGO_diabetes": ("KDIGO", ["diabetes", "glycemic", "hba1c"]),
    "KDIGO_CVD": ("KDIGO", ["cardiovascular", "coronary", "heart failure"]),
    "ACHD_HF": ("ACHD", ["heart failure", "hfref", "hfpef", "ejection fraction"]),
    "ACHD_pregnancy": ("ACHD", ["pregnant", "pregnancy", "obstetric"]),
}


def _match_vocab(text: str, vocab: dict[str, list[str]]) -> list[str]:
    text_lower = text.lower()
    matched = []
    for tag, keywords in vocab.items():
        if any(kw in text_lower for kw in keywords):
            matched.append(tag)
    return matched


def enrich_tags(chunk: Chunk) -> Chunk:
    text = chunk.text

    chunk.drug_tags = _match_vocab(text, DRUG_VOCAB)
    chunk.comorbidity_tags = _match_vocab(text, COMORBIDITY_VOCAB)
    chunk.bp_stage_tags = _match_vocab(text, BP_STAGE_VOCAB)
    chunk.ckd_stage_tags = _match_vocab(text, CKD_STAGE_VOCAB)
    chunk.intervention_tags = _match_vocab(text, INTERVENTION_VOCAB)

    base = {
        "HTN": ["hypertension"],
        "ACHD": ["ACHD"],
        "KDIGO": ["CKD"],
        "NICE": ["CKD"],
    }.get(chunk.guideline, [])

    gl_for_cross = "KDIGO" if chunk.guideline == "NICE" else chunk.guideline
    cross: list[str] = []
    for key, (gl, kws) in CROSS_DISEASE_MAP.items():
        if gl == gl_for_cross:
            text_lower = text.lower()
            if any(kw in text_lower for kw in kws):
                secondary = key.split("_", 1)[1].lower()
                cross.append(secondary)

    chunk.disease_tags = list(dict.fromkeys(base + cross))
    return chunk


_PAGE_NOISE = re.compile(
    r"Downloaded from http://ahajournals\.org[^\n]*\n"
    r"|CLINICAL STATEMENTS\s*\n\s*AND GUIDELINES\s*\n"
    r"|(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2},?\s+\d{4}\s*\x08?\s*\n"
    r"|Circulation\.\s+\d{4};[\d:e–\-\.]+\s+DOI:[^\n]+\n"
    r"|[A-Za-z ]+\s+\d{4}\s+(?:High Blood Pressure|ACHD)\s+Guideline\s*\n"
    r"|(?:Jones|Gurvitz) et al\s*\n",
    re.IGNORECASE,
)

_KDIGO_NOISE = re.compile(
    r"Kidney International\s*\(\d{4}\)\s*\d+[^\n]*\n"
    r"|www\.kidney-international\.org\s*\n"
    r"|chapter\s+\d+\s*\nwww\.kidney",
    re.IGNORECASE,
)


def clean_text(raw: str, guideline: str) -> str:
    if guideline in ("HTN", "ACHD"):
        raw = _PAGE_NOISE.sub("\n", raw)
    else:
        raw = _KDIGO_NOISE.sub("\n", raw)

    raw = re.sub(r"(\w)-\s*\n\s+(\w)", r"\1\2", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw)

    return raw.strip()


def clean_prose(text: str) -> str:
    text = re.sub(r"(?<![.\:;!?])\n(?!\n)", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def min_length_ok(text: str, chunk_type: str) -> bool:
    thresholds = {
        "recommendation": 40,
        "practice_point": 40,
        "table": 80,
        "synopsis": 60,
        "supportive_text": 60,
        "rationale": 100,
        "section_summary": 100,
    }
    return len(text.strip()) >= thresholds.get(chunk_type, 40)
