"""
kdigo_parser.py
Parser for KDIGO 2024 CKD Clinical Practice Guideline.

Root causes fixed vs original:
  1. Rationale pattern now matches "Rationale" as a STANDALONE LINE
     (line.strip() == "Rationale"), not via regex that requires \\n prefix.
     pypdf puts "Rationale" on its own line with no preceding \\n in
     the same string segment, so the original multiline pattern fired
     only ~10% of the time → only 12 rationale chunks instead of ~50.
  2. Rec + PP boundaries now built from a sorted position list so that
     a Recommendation line immediately followed by a Practice Point line
     terminates the Recommendation body correctly.
  3. KDIGO grade is extracted from anywhere within the full rec body
     (the grade "(1B)" often appears at the end of a multi-line rec),
     not just from the header line.
  4. Section summaries capped at depth-2 section headings ("N.N"),
     not sub-subsections, matching AHA parser behavior.
"""

from __future__ import annotations
import re
from typing import List, Tuple, Optional

from .base_parser import Chunk, clean_text, enrich_tags, min_length_ok


# ─────────────────────────────────────────────────────────────────────────────
# Compiled patterns
# ─────────────────────────────────────────────────────────────────────────────

# Chapter heading: strictly "Chapter N: Title" — requires colon separator + title starting with capital
# Avoids matching inline prose references like "see chapter 3 for..."
_CHAPTER_LINE = re.compile(
    r'^Chapter\s+(?P<num>\d+)\s*:\s*(?P<title>[A-Z][^\n]{10,120})$',
    re.MULTILINE | re.IGNORECASE
)

# Section heading at depth 1–2: "3.6 Renin-angiotensin…"
# Must start at beginning of line; NOT sub-subsections like "3.6.1"
_SECTION_D1_D2 = re.compile(
    r'^(?P<sec_id>\d+\.\d+)\s+(?P<title>[A-Z][^\n]{3,100})$',
    re.MULTILINE
)

# Any section heading (for context lookup)
_SECTION_ANY = re.compile(
    r'^(?P<sec_id>\d+(?:\.\d+){1,3})\s+(?P<title>[A-Z][^\n]{3,100})$',
    re.MULTILINE
)

# Recommendation line start: "Recommendation 3.6.1: …"
_REC_START = re.compile(
    r'^Recommendation\s+(?P<id>\d+(?:\.\d+){1,3}):\s*(?P<text>.+)',
    re.MULTILINE | re.IGNORECASE
)

# Practice Point line start: "Practice Point 3.2.1: …"
_PP_START = re.compile(
    r'^Practice\s+Point\s+(?P<id>\d+(?:\.\d+){1,3}):\s*(?P<text>.+)',
    re.MULTILINE | re.IGNORECASE
)

# KDIGO grade: (1A), (2B), (1C), (2D) — may appear anywhere in body
_GRADE_INLINE = re.compile(r'\((?P<grade>[12][ABCD])\)')

# Table heading: "Table N | title" or "Table N. title"
_TABLE_HEADER = re.compile(
    r'^Table\s+(?P<num>\d+)\s*[|\.]\s*(?P<title>[^\n]{5,150})$',
    re.MULTILINE | re.IGNORECASE
)

# "Rationale" as a standalone line (exact match after strip)
# We use line-by-line logic, not a regex, because pypdf outputs it as a bare word.


# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

def _find_chapter_section(text: str, pos: int) -> Tuple[str, str, str]:
    """Return (chapter_id, section_id, section_title) for the given position."""
    chapter_id = ""
    for m in _CHAPTER_LINE.finditer(text):
        if m.start() > pos:
            break
        chapter_id = f"chapter_{m.group('num')}"

    sec_id, sec_title = "", ""
    for m in _SECTION_ANY.finditer(text):
        if m.start() > pos:
            break
        sec_id = m.group("sec_id")
        sec_title = m.group("title").strip()

    return chapter_id, sec_id, sec_title


def _extract_grade(text: str) -> str:
    m = _GRADE_INLINE.search(text)
    return m.group("grade") if m else ""


# ─────────────────────────────────────────────────────────────────────────────
# Build a sorted boundary list for Rec / PP / Rationale
# ─────────────────────────────────────────────────────────────────────────────

def _build_boundary_list(text: str) -> List[dict]:
    """
    Return a list of dicts sorted by start position, each with keys:
        kind  : "rec" | "pp" | "rationale" | "chapter" | "eof"
        start : int
        id_   : recommendation/PP id (for rec/pp); "" otherwise
        text  : matched text of the header line
        pos   : start position in full text
    """
    boundaries = []

    for m in _REC_START.finditer(text):
        boundaries.append({
            "kind": "rec",
            "start": m.start(),
            "id_": m.group("id"),
            "header_text": m.group(0),
            "body_start": m.end(),
        })

    for m in _PP_START.finditer(text):
        boundaries.append({
            "kind": "pp",
            "start": m.start(),
            "id_": m.group("id"),
            "header_text": m.group(0),
            "body_start": m.end(),
        })

    # Rationale: detect as standalone line using character-accurate positions
    pos = 0
    for line in text.splitlines(keepends=True):
        if line.strip() == "Rationale":
            boundaries.append({
                "kind": "rationale",
                "start": pos,
                "id_": "",
                "header_text": "Rationale",
                "body_start": pos + len(line),
            })
        pos += len(line)

    for m in _CHAPTER_LINE.finditer(text):
        boundaries.append({
            "kind": "chapter",
            "start": m.start(),
            "id_": f"chapter_{m.group('num')}",
            "header_text": m.group(0),
            "body_start": m.end(),
        })

    # Sentinel
    boundaries.append({"kind": "eof", "start": len(text), "id_": "",
                        "header_text": "", "body_start": len(text)})

    boundaries.sort(key=lambda x: x["start"])
    return boundaries


# ─────────────────────────────────────────────────────────────────────────────
# Rationale: find the closest preceding Rec or PP id
# ─────────────────────────────────────────────────────────────────────────────

def _preceding_rec_id(boundaries: List[dict], current_idx: int, guideline: str) -> str:
    gl = guideline.upper()
    for i in range(current_idx - 1, -1, -1):
        if boundaries[i]["kind"] in ("rec", "pp"):
            return f"{gl}_{boundaries[i]['kind'].upper()}_{boundaries[i]['id_']}"
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# Table parser
# ─────────────────────────────────────────────────────────────────────────────

def _parse_tables(text: str, guideline: str, disease: str, source_doc: str) -> List[Chunk]:
    chunks: List[Chunk] = []
    matches = list(_TABLE_HEADER.finditer(text))
    for idx, m in enumerate(matches):
        body_start = m.end()
        body_end   = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[body_start:min(body_start + 1500, body_end)].strip()
        if not min_length_ok(body, "table"):
            continue
        ch_id, sec_id, sec_title = _find_chapter_section(text, m.start())
        full_text = f"Table {m.group('num')} | {m.group('title').strip()}\n{body}"
        c = Chunk(
            text=full_text,
            chunk_type="table",
            guideline=guideline,
            disease=disease,
            source_doc=source_doc,
            chapter_id=ch_id,
            section_id=sec_id,
            section_title=sec_title,
        )
        chunks.append(enrich_tags(c))
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Section summary parser — depth-2 sections only ("N.N")
# ─────────────────────────────────────────────────────────────────────────────

def _parse_section_summaries(text: str, guideline: str, disease: str, source_doc: str) -> List[Chunk]:
    chunks: List[Chunk] = []
    all_matches = list(_SECTION_D1_D2.finditer(text))
    # Dedup by sec_id — keep last match (body, not TOC repetition)
    seen: dict = {}
    for m in all_matches:
        seen[m.group("sec_id")] = m
    matches = sorted(seen.values(), key=lambda m: m.start())
    for idx, m in enumerate(matches):
        body_start = m.end()
        body_end   = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        opening = text[body_start:body_start + 800].strip()
        if not min_length_ok(opening, "section_summary"):
            continue
        ch_id, _, _ = _find_chapter_section(text, m.start())
        full_text = (
            f"[{m.group('sec_id')} — {m.group('title').strip()}]\n\n"
            f"{opening}"
        )
        c = Chunk(
            text=full_text,
            chunk_type="section_summary",
            guideline=guideline,
            disease=disease,
            source_doc=source_doc,
            chapter_id=ch_id,
            section_id=m.group("sec_id"),
            section_title=m.group("title").strip(),
        )
        chunks.append(enrich_tags(c))
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Chapter summary parser — one per chapter (Tier-3)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_chapter_summaries(text: str, guideline: str, disease: str, source_doc: str) -> List[Chunk]:
    chunks: List[Chunk] = []
    all_matches = list(_CHAPTER_LINE.finditer(text))
    # Dedup by chapter number — keep last occurrence
    seen: dict = {}
    for m in all_matches:
        seen[m.group("num")] = m
    matches = sorted(seen.values(), key=lambda m: m.start())
    for idx, m in enumerate(matches):
        body_start = m.end()
        body_end   = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        opening = text[body_start:body_start + 800].strip()
        if not min_length_ok(opening, "section_summary"):
            continue
        ch_id = f"chapter_{m.group('num')}"
        full_text = (
            f"[Chapter {m.group('num')} — {m.group('title').strip()}]\n\n"
            f"{opening}"
        )
        c = Chunk(
            text=full_text,
            chunk_type="section_summary",
            guideline=guideline,
            disease=disease,
            source_doc=source_doc,
            chapter_id=ch_id,
            section_id="",
            section_title=m.group("title").strip(),
        )
        chunks.append(enrich_tags(c))
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse(raw_text: str, guideline: str = "KDIGO",
          disease: str = "CKD", source_doc: str = "KDIGO-2024-CKD") -> List[Chunk]:
    """
    Parse KDIGO CKD guideline PDF text into chunks.

    Returns
    -------
    List[Chunk]
    """
    gl = guideline.upper()
    text = clean_text(raw_text, guideline)
    chunks: List[Chunk] = []

    boundaries = _build_boundary_list(text)

    for i, bnd in enumerate(boundaries):
        kind = bnd["kind"]
        if kind not in ("rec", "pp", "rationale"):
            continue

        body_start = bnd["body_start"]
        body_end   = boundaries[i + 1]["start"]  # next boundary always exists (sentinel)
        body = text[body_start:body_end].strip()

        ch_id, sec_id, sec_title = _find_chapter_section(text, bnd["start"])

        # ── Tier-1: Recommendation ───────────────────────────────────────────
        if kind == "rec":
            raw_id  = bnd["id_"]
            rec_id  = f"{gl}_REC_{raw_id}"
            grade   = _extract_grade(body) or _extract_grade(bnd["header_text"])
            # Compose full text: header line + body
            header_line = bnd["header_text"].strip()
            full_text = (
                f"[Chapter {ch_id} | {sec_id} — {sec_title}]\n"
                f"{header_line}\n"
                f"{body}"
            )
            if min_length_ok(body, "recommendation"):
                c = Chunk(
                    text=full_text,
                    chunk_type="recommendation",
                    guideline=guideline,
                    disease=disease,
                    source_doc=source_doc,
                    chapter_id=ch_id,
                    section_id=sec_id,
                    section_title=sec_title,
                    rec_id=rec_id,
                    kdigo_grade=grade,
                )
                chunks.append(enrich_tags(c))

        # ── Tier-1: Practice Point ───────────────────────────────────────────
        elif kind == "pp":
            raw_id = bnd["id_"]
            rec_id = f"{gl}_PP_{raw_id}"
            header_line = bnd["header_text"].strip()
            full_text = (
                f"[Chapter {ch_id} | {sec_id} — {sec_title}]\n"
                f"{header_line}\n"
                f"{body}"
            )
            if min_length_ok(body, "practice_point"):
                c = Chunk(
                    text=full_text,
                    chunk_type="practice_point",
                    guideline=guideline,
                    disease=disease,
                    source_doc=source_doc,
                    chapter_id=ch_id,
                    section_id=sec_id,
                    section_title=sec_title,
                    rec_id=rec_id,
                )
                chunks.append(enrich_tags(c))

        # ── Tier-2: Rationale ────────────────────────────────────────────────
        elif kind == "rationale":
            parent = _preceding_rec_id(boundaries, i, guideline)
            full_text = (
                f"[Chapter {ch_id} | {sec_id} — {sec_title}]\n"
                f"Rationale\n\n"
                f"{body}"
            )
            if min_length_ok(body, "rationale"):
                c = Chunk(
                    text=full_text,
                    chunk_type="rationale",
                    guideline=guideline,
                    disease=disease,
                    source_doc=source_doc,
                    chapter_id=ch_id,
                    section_id=sec_id,
                    section_title=sec_title,
                    parent_rec_id=parent,
                )
                chunks.append(enrich_tags(c))

    # ── Tier-1: Tables ───────────────────────────────────────────────────────
    chunks.extend(_parse_tables(text, guideline, disease, source_doc))

    # ── Tier-3: Section summaries (depth-2) + Chapter summaries ─────────────
    chunks.extend(_parse_section_summaries(text, guideline, disease, source_doc))
    chunks.extend(_parse_chapter_summaries(text, guideline, disease, source_doc))

    return chunks
