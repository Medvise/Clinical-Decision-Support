"""
Canonical guideline PDF → Chunk parsing.

Used by `ingestion.script` as `ingestion.parser_ref`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from .aha_parser import parse as parse_aha
from .kdigo_parser import parse as parse_kdigo


def detect_guideline_meta(
    pdf_path: Path,
) -> tuple[str, str, Literal["aha", "kdigo"]]:
    """
    Returns (guideline_code, primary disease label, parser family).

    Filename hints control routing — name PDFs with KDIGO, NICE, HTN/HYPERTENSION, ACHD, etc.
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
    raise ValueError(
        f"Cannot detect guideline from filename {pdf_path.name!r}. "
        "Include KDIGO, NICE, HTN, HYPERTENSION, or ACHD in the stem."
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
    else:
        chunks = parse_aha(
            raw_text,
            guideline=guideline,
            disease=disease,
            source_doc=source_doc,
        )
    return [c.to_payload() for c in chunks]


__all__ = ["detect_guideline_meta", "parse_pdf", "parse_aha", "parse_kdigo"]
