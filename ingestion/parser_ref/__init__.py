"""
Canonical guideline PDF → Chunk parsing.

Used by `ingestion.script` as `ingestion.parser_ref`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from .ada_parser import parse as parse_ada
from .aha_parser import parse as parse_aha
from .kdigo_parser import parse as parse_kdigo

ParserFamily = Literal["aha", "kdigo", "ada"]


def detect_guideline_meta(
    pdf_path: Path,
) -> tuple[str, str, ParserFamily]:
    """
    Returns (guideline_code, primary disease label, parser family).

    Filename hints control routing — name PDFs with KDIGO, NICE, HTN/HYPERTENSION,
    ACHD, DIABETES, STROKE, or CHOLESTEROL/LIPID in the stem.
    """
    stem = pdf_path.stem.upper().replace(" ", "_")
    if stem in {"CKD", "CKD_GUIDELINE"} or stem.startswith("CKD_"):
        return "KDIGO", "CKD", "kdigo"
    if "KDIGO" in stem:
        return "KDIGO", "CKD", "kdigo"
    if "NICE" in stem:
        return "NICE", "CKD", "kdigo"
    if "HTN" in stem or "HYPERTENSION" in stem:
        return "HTN", "hypertension", "aha"
    if (
        "ACHD" in stem
        or "CONGENITAL" in stem
        or "CHD" in stem
        or "HEART_FAILURE" in stem
        or "HEARTFAILURE" in stem
    ):
        return "ACHD", "ACHD", "aha"
    if "DIABETES" in stem or "ADA" in stem:
        return "Diabetes", "diabetes", "ada"
    if "STROKE" in stem:
        return "STROKE", "stroke", "aha"
    if any(
        token in stem
        for token in ("CHOLESTEROL", "CHOLESTROL", "LIPID", "DYSLIPID", "DYSPLIPID")
    ):
        return "LIPID", "dyslipidemia", "aha"
    raise ValueError(
        f"Cannot detect guideline from filename {pdf_path.name!r}. "
        "Include KDIGO, NICE, HTN, HYPERTENSION, ACHD, DIABETES, STROKE, or CHOLESTEROL/LIPID."
    )


def parse_pdf(pdf_path: Path, raw_text: str) -> list[dict]:
    guideline, disease, family = detect_guideline_meta(pdf_path)
    source_doc = pdf_path.stem
    if family == "kdigo":
        chunks = parse_kdigo(
            raw_text,
            guideline=guideline,
            disease=disease,
            source_doc=source_doc,
        )
    elif family == "ada":
        chunks = parse_ada(
            raw_text,
            guideline=guideline,
            disease=disease,
            source_doc=source_doc,
        )
    else:
        # family == "aha"  →  handles HTN, ACHD, LIPID, and STROKE
        chunks = parse_aha(
            raw_text,
            guideline=guideline,
            disease=disease,
            source_doc=source_doc,
        )
    return [c.to_payload() for c in chunks]


__all__ = [
    "detect_guideline_meta",
    "parse_pdf",
    "parse_aha",
    "parse_kdigo",
    "parse_ada",
    "ParserFamily",
]