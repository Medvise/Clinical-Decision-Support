"""
Shared dataclass, tag extraction, and text utilities for guideline parsers.

Canonical copy lives in `ingestion.parser_ref` (imported by `ingestion.script`).
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import List

from retrieval.tags_vocab import (
    BP_STAGE_VOCAB,
    CKD_STAGE_VOCAB,
    COMORBIDITY_VOCAB,
    CROSS_DISEASE_MAP,
    DRUG_VOCAB,
    INTERVENTION_VOCAB,
)


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
        "ADA": ["diabetes"],
        "Diabetes": ["diabetes"],
        "LIPID": ["dyslipidemia"],
        "STROKE": ["stroke"],
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
    r"|Stroke\.\s+\d{4};[\d:e–\-\.]+\s+DOI:[^\n]+\n"
    r"|[A-Za-z ]+\s+\d{4}\s+(?:High Blood Pressure|ACHD|Acute Ischemic Stroke)\s+Guideline\s*\n"
    r"|(?:Jones|Gurvitz|Prabhakaran) et al[^\n]*\n",
    re.IGNORECASE,
)

_KDIGO_NOISE = re.compile(
    r"Kidney International\s*\(\d{4}\)\s*\d+[^\n]*\n"
    r"|www\.kidney-international\.org\s*\n"
    r"|chapter\s+\d+\s*\nwww\.kidney",
    re.IGNORECASE,
)

_ADA_NOISE = re.compile(
    r"Diabetes Care\s+\d{4};\d+[^\n]*\n"
    r"|\d{4}\.\s+Diabetes Care\s+\d{4}[^\n]*\n"
    r"|Available from https?://[^\n]+\n"
    r"|Accessed\s+\d{1,2}\s+[A-Za-z]+\s+\d{4}[^\n]*\n"
    r"|Downloaded from[^\n]+\n"
    r"|N Engl J Med\s+\d{4}[^\n]*\n"
    r"|https?://doi\.org/[^\n]+\n"
    r"|Professional Practice Committee[^\n]*\n",
    re.IGNORECASE,
)


def clean_text(raw: str, guideline: str) -> str:
    if guideline in ("HTN", "ACHD", "LIPID", "STROKE"):
        raw = _PAGE_NOISE.sub("\n", raw)
    elif guideline in ("ADA", "Diabetes"):
        raw = _ADA_NOISE.sub("\n", raw)
    else:
        raw = _KDIGO_NOISE.sub("\n", raw)

    # Join standard end-of-line hyphens: 'hyper-\ntension' → 'hypertension'
    raw = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", raw)
    # Join spaced end-of-line hyphens: 'hypo -\nglycemia' → 'hypoglycemia'  [ADA PDF quirk]
    raw = re.sub(r"(\w)\s+-\s*\n\s*(\w)", r"\1\2", raw)
    # Collapse standalone hyphen lines: 'manage\n-\nment' → 'management'  [ADA PDF quirk]
    raw = re.sub(r"(\w)\n-\n(\w)", r"\1\2", raw)
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