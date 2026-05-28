"""
ada_parser.py
Parser for the ADA Standards of Care in Diabetes (2026 edition).

Document structure
──────────────────
The ADA PDF is structured across three layers for each section:

  1. TOC entry  — brief one-liner in the table of contents
  2. Change-notes page — summary of revisions, begins with
       "N. Section Title\n(https://doi.org/...)"
  3. Full content page — starts with
       "N. Section Title: Standards of Care in Diabetes—YYYY\n
        Diabetes Care YYYY;49(Suppl. 1):S...|..."
     This is the ONLY layer we parse for recommendations.

  • Within each section, subsection headings appear in ALL-CAPS on their own line,
    followed by a "Recommendations" label line, then numbered rec lines.
  • Recommendation lines: "<N.M[a-z]?> <text> <grade>"
    where grade is one of A | B | C | E at end of sentence.
  • Due to PDF column layout, line breaks appear mid-sentence and hyphens are
    split across lines (both 'hypo -\nglycemia' and 'manage\n-\nment' forms).
    These are normalised by clean_text() in base_parser before parsing.
  • References appear at end of each section as numbered footnotes.

Key implementation note
───────────────────────
Section boundaries are detected on the RAW (uncleaned) text using the
'Diabetes Care YYYY;49(Suppl. 1):S<page>' DOI line as an anchor — this line
reliably appears only on the real content page, not the TOC or change-notes
pages.  The noise cleaner in base_parser strips these DOI lines, so section
detection MUST happen before clean_text() is called.  Each section body is
then cleaned individually before rec/prose parsing.

Tier mapping to unified schema
──────────────────────────────
  Tier 1 – recommendation   : individual numbered ADA recommendation sentences
  Tier 1 – table            : Table N.N blocks
  Tier 2 – supportive_text  : prose paragraphs following recommendations
  Tier 3 – section_summary  : one per subsection, first prose paragraph (~450 chars)

Chunk fields used
──────────────────
  chunk_type     recommendation | table | supportive_text | section_summary
  guideline      "Diabetes"
  disease        "diabetes"
  source_doc     PDF stem (e.g. "Diabetes")
  section_id     "9"
  chapter_id     "chapter_9"
  section_title  "Pharmacologic Approaches to Glycemic Treatment"
  rec_id         "Diabetes_9_9a"
  parent_rec_id  "" for tier-1; rec_id of nearest tier-1 for tier-2
  kdigo_grade    ADA grade: "A" | "B" | "C" | "E"
  cor, loe       empty (not applicable)
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from typing import List, Tuple

from .base_parser import Chunk, clean_prose, clean_text, enrich_tags, min_length_ok

# ─────────────────────────────────────────────────────────────────────────────
# ADA canonical section titles  (sec number → title)
# ─────────────────────────────────────────────────────────────────────────────

SECTION_TITLES: dict[str, str] = {
    "1": "Improving Care and Promoting Health in Populations",
    "2": "Diagnosis and Classification of Diabetes",
    "3": "Prevention or Delay of Diabetes and Associated Comorbidities",
    "4": "Comprehensive Medical Evaluation and Assessment of Comorbidities",
    "5": "Facilitating Positive Health Behaviors and Well-being to Improve Health Outcomes",
    "6": "Glycemic Goals Hypoglycemia and Hyperglycemic Crises",
    "7": "Diabetes Technology",
    "8": "Obesity and Weight Management for the Prevention and Treatment of Diabetes",
    "9": "Pharmacologic Approaches to Glycemic Treatment",
    "10": "Cardiovascular Disease and Risk Management",
    "11": "Chronic Kidney Disease and Risk Management",
    "12": "Retinopathy Neuropathy and Foot Care",
    "13": "Older Adults",
    "14": "Children and Adolescents",
    "15": "Management of Diabetes in Pregnancy",
    "16": "Diabetes Care in the Hospital",
    "17": "Diabetes Advocacy",
}

# ─────────────────────────────────────────────────────────────────────────────
# Compiled patterns
# ─────────────────────────────────────────────────────────────────────────────

# Marker that reliably identifies the REAL section content page (on raw text only —
# base_parser's clean_text strips these lines, so use before cleaning):
#   "Diabetes Care YYYY;49(Suppl. 1):S<page>"
_SECTION_DOI_LINE = re.compile(
    r"^Diabetes Care \d{4};49\(Suppl\.\s*1\):S(\d+)",
    re.IGNORECASE,
)

# Section number line: "9. Pharmacologic Approaches to"
# Title may be truncated at line break — canonical title comes from SECTION_TITLES dict.
_SECTION_NUMBER_LINE = re.compile(r"^(\d{1,2})\.\s+(.{3,80}?)(?::\s*Standards.*)?$")

# ALL-CAPS subsection heading (≥8 chars of uppercase letters/punctuation)
_SUBSECTION_ALLCAPS = re.compile(r"^[A-Z][A-Z\s\-/,&'()]{7,}$")

# "Recommendations" label line preceding a rec block
_RECOMMENDATIONS_LABEL = re.compile(r"^Recommendations?\s*$", re.IGNORECASE)

# Full single-line recommendation with grade at end
_REC_FULL = re.compile(
    r"^(\d{1,2}\.\d{1,3}[a-z]?)\s+(.+?)(?:\s+|(?<=[.;:!?]))([ABCE])$",
    re.DOTALL,
)

# Rec number at start of line (continuation follows on next lines)
_REC_START = re.compile(r"^(\d{1,2}\.\d{1,3}[a-z]?)\s+")

# Table header: "Table N.N"
_TABLE_HEADER = re.compile(r"^Table\s+\d+\.\d+", re.IGNORECASE)

# Reference section markers
_REFERENCES_HEADER = re.compile(r"^References\s*$", re.IGNORECASE)
_GRADE_AT_END = re.compile(r"(?:\s+|(?<=[.;:!?]))([ABCE])$")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _is_reference_block(line: str) -> bool:
    return bool(_REFERENCES_HEADER.match(line))


def _extract_grade(text: str) -> Tuple[str, str]:
    m = _GRADE_AT_END.search(text.strip())
    if m:
        return text[: m.start()].strip(), m.group(1)
    return text.strip(), ""


def _make_rec_id(guideline: str, section_id: str, rec_number: str) -> str:
    safe = rec_number.replace(".", "_")
    return f"{guideline}_{section_id}_{safe}"


def _rec_matches_section(rec_number: str, section_id: str) -> bool:
    return rec_number.split(".", 1)[0] == section_id


def _line_starts_section_rec(line: str, section_id: str) -> bool:
    m = _REC_START.match(line)
    return bool(m and _rec_matches_section(m.group(1), section_id))


# ─────────────────────────────────────────────────────────────────────────────
# Section boundary detection  (must run on RAW text, before noise cleaning)
# ─────────────────────────────────────────────────────────────────────────────

def _find_section_starts(lines: List[str]) -> List[Tuple[int, str, str]]:
    """
    Return (line_index, section_number, canonical_title) for each of the 17
    real content sections.

    Uses the 'Diabetes Care YYYY;49(Suppl. 1):S...' DOI line as the anchor —
    this line appears ONLY on the real content page, not in the TOC or the
    change-notes summary page.

    Looks back up to 6 lines from each DOI line for a line matching "N. title".
    The ceiling of 6 filters out false-positive DOI lines (Introduction/
    Methodology, Summary of Revisions, Disclosures, and in-text citations)
    whose nearest "N." line is always >6 lines away.

    IMPORTANT: call this on raw (uncleaned) text — base_parser.clean_text()
    strips the DOI lines, so they must be located before cleaning.
    """
    starts: List[Tuple[int, str, str]] = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not _SECTION_DOI_LINE.match(stripped):
            continue
        for back in range(1, 7):
            prev_idx = i - back
            if prev_idx < 0:
                break
            prev = lines[prev_idx].strip()
            m = _SECTION_NUMBER_LINE.match(prev)
            if m:
                sec_num = m.group(1)
                if sec_num in SECTION_TITLES:
                    canonical_title = SECTION_TITLES[sec_num]
                    starts.append((prev_idx, sec_num, canonical_title))
                break
    return starts


# ─────────────────────────────────────────────────────────────────────────────
# Section body parser
# ─────────────────────────────────────────────────────────────────────────────

def _parse_section(
    section_lines: List[str],
    section_id: str,
    section_title: str,
    *,
    guideline: str,
    disease: str,
    source_doc: str,
) -> List[Chunk]:
    """Parse one section's (already cleaned) lines into Chunks."""
    chunks: List[Chunk] = []
    chapter_id = f"chapter_{section_id}"
    current_subsection = section_title

    rec_buffer: List[str] = []
    rec_number = ""
    prose_buffer: List[str] = []
    table_buffer: List[str] = []
    in_table = False
    in_references = False
    last_rec_id = ""
    subsection_summary_emitted = False

    def flush_rec():
        nonlocal rec_buffer, rec_number, last_rec_id
        if not rec_buffer:
            return
        full_text = " ".join(rec_buffer).strip()
        body, grade = _extract_grade(full_text)
        if not grade:
            parts = full_text.rsplit(" ", 1)
            if len(parts) == 2 and parts[1] in ("A", "B", "C", "E"):
                body, grade = parts[0].strip(), parts[1]
        if not body:
            rec_buffer.clear()
            rec_number = ""
            return
        rid = _make_rec_id(guideline, section_id, rec_number) if rec_number else ""
        c = Chunk(
            text=clean_prose(body),
            chunk_type="recommendation",
            guideline=guideline,
            disease=disease,
            source_doc=source_doc,
            section_id=section_id,
            chapter_id=chapter_id,
            section_title=current_subsection,
            rec_id=rid,
            kdigo_grade=grade,
        )
        if min_length_ok(c.text, "recommendation"):
            chunks.append(enrich_tags(c))
            last_rec_id = rid
        rec_buffer.clear()
        rec_number = ""

    def flush_prose():
        nonlocal prose_buffer, subsection_summary_emitted
        if not prose_buffer:
            return
        full_text = clean_prose(" ".join(prose_buffer).strip())
        if not min_length_ok(full_text, "supportive_text"):
            prose_buffer.clear()
            return
        if not subsection_summary_emitted:
            summary_text = full_text[:450].rsplit(" ", 1)[0]
            if min_length_ok(summary_text, "section_summary"):
                sc = Chunk(
                    text=summary_text,
                    chunk_type="section_summary",
                    guideline=guideline,
                    disease=disease,
                    source_doc=source_doc,
                    section_id=section_id,
                    chapter_id=chapter_id,
                    section_title=current_subsection,
                )
                chunks.append(enrich_tags(sc))
            subsection_summary_emitted = True
        c = Chunk(
            text=full_text,
            chunk_type="supportive_text",
            guideline=guideline,
            disease=disease,
            source_doc=source_doc,
            section_id=section_id,
            chapter_id=chapter_id,
            section_title=current_subsection,
            parent_rec_id=last_rec_id,
        )
        chunks.append(enrich_tags(c))
        prose_buffer.clear()

    def flush_table():
        nonlocal table_buffer, in_table
        if not table_buffer:
            in_table = False
            return
        full_text = "\n".join(table_buffer).strip()
        if min_length_ok(full_text, "table"):
            c = Chunk(
                text=full_text,
                chunk_type="table",
                guideline=guideline,
                disease=disease,
                source_doc=source_doc,
                section_id=section_id,
                chapter_id=chapter_id,
                section_title=current_subsection,
                parent_rec_id=last_rec_id,
            )
            chunks.append(enrich_tags(c))
        table_buffer.clear()
        in_table = False

    for raw_line in section_lines:
        line = raw_line.strip()

        if not line:
            if prose_buffer:
                flush_prose()
            continue

        if _is_reference_block(line):
            in_references = True
        if in_references:
            flush_rec()
            flush_prose()
            flush_table()
            continue

        # ── Table handling ────────────────────────────────────────────────
        if not rec_buffer and _TABLE_HEADER.match(line):
            flush_rec()
            flush_prose()
            flush_table()
            in_table = True
            table_buffer.append(line)
            continue

        if in_table:
            if _SUBSECTION_ALLCAPS.match(line) or _line_starts_section_rec(line, section_id):
                flush_table()
            else:
                table_buffer.append(line)
                continue

        # ── ALL-CAPS subsection heading ───────────────────────────────────
        if _SUBSECTION_ALLCAPS.match(line) and not _REC_START.match(line):
            flush_rec()
            flush_prose()
            current_subsection = line.title()
            subsection_summary_emitted = False
            continue

        # ── "Recommendations" label line (skip — just marks a block) ──────
        if _RECOMMENDATIONS_LABEL.match(line):
            flush_rec()
            flush_prose()
            continue

        # ── Recommendation lines ──────────────────────────────────────────
        m_rec_full = _REC_FULL.match(line)
        m_rec_start = _REC_START.match(line)
        if m_rec_full and not _rec_matches_section(m_rec_full.group(1), section_id):
            m_rec_full = None
        if m_rec_start and not _rec_matches_section(m_rec_start.group(1), section_id):
            m_rec_start = None

        if m_rec_full:
            flush_rec()
            flush_prose()
            rec_number = m_rec_full.group(1)
            rec_body = m_rec_full.group(2).strip()
            grade = m_rec_full.group(3)
            rid = _make_rec_id(guideline, section_id, rec_number)
            c = Chunk(
                text=clean_prose(rec_body),
                chunk_type="recommendation",
                guideline=guideline,
                disease=disease,
                source_doc=source_doc,
                section_id=section_id,
                chapter_id=chapter_id,
                section_title=current_subsection,
                rec_id=rid,
                kdigo_grade=grade,
            )
            if min_length_ok(c.text, "recommendation"):
                chunks.append(enrich_tags(c))
                last_rec_id = rid
            rec_number = ""
            rec_buffer.clear()

        elif m_rec_start:
            flush_rec()
            flush_prose()
            rec_number = m_rec_start.group(1)
            rest = line[m_rec_start.end():].strip()
            rec_buffer = [rest] if rest else []

        elif rec_buffer:
            rec_buffer.append(line)
            if _GRADE_AT_END.search(line):
                flush_rec()

        else:
            # Prose — skip very short fragments (stray page-footer tokens)
            if len(line) >= 8:
                prose_buffer.append(line)

    flush_rec()
    flush_prose()
    flush_table()

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse(raw_text: str, guideline: str, disease: str, source_doc: str) -> List[Chunk]:
    """
    Parse ADA Standards of Care text into chunks.

    Parameters
    ----------
    raw_text   : full concatenated pypdf output (from ingestion.script)
    guideline  : "Diabetes"
    disease    : "diabetes"
    source_doc : PDF filename stem

    Returns
    -------
    List[Chunk]
    """
    # Step 1: detect section boundaries on RAW text (before noise cleaning),
    # because clean_text() strips the 'Diabetes Care ...' DOI anchor lines.
    raw_lines = raw_text.splitlines()
    section_starts = _find_section_starts(raw_lines)

    if not section_starts:
        raise ValueError(
            "No ADA section headers found — check PDF extraction and filename.\n"
            "Expected 'Diabetes Care YYYY;49(Suppl. 1):S...' DOI lines after each "
            "section title in the raw PDF text."
        )

    all_chunks: List[Chunk] = []

    for idx, (start_line, sec_num, sec_title) in enumerate(section_starts):
        end_line = (
            section_starts[idx + 1][0]
            if idx + 1 < len(section_starts)
            else len(raw_lines)
        )

        # Step 2: extract raw section slice, then clean it
        raw_section = "\n".join(raw_lines[start_line:end_line])
        cleaned_section = clean_text(raw_section, guideline)
        section_lines = cleaned_section.splitlines()

        all_chunks.extend(
            _parse_section(
                section_lines,
                sec_num,
                sec_title,
                guideline=guideline,
                disease=disease,
                source_doc=source_doc,
            )
        )

    return all_chunks


# ─────────────────────────────────────────────────────────────────────────────
# CLI smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from pathlib import Path
    from ingestion.script import extract_text

    pdf = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("ingestion/Diabetes.pdf")
    chunks = parse(
        extract_text(pdf),
        guideline="Diabetes",
        disease="diabetes",
        source_doc=pdf.stem,
    )
    ctr = Counter(c.chunk_type for c in chunks)
    grade_ctr = Counter(
        c.kdigo_grade for c in chunks if c.chunk_type == "recommendation"
    )

    print(f"\nTotal chunks : {len(chunks)}")
    print(f"By type      : {dict(ctr)}")
    print(f"Rec grades   : {dict(grade_ctr)}")

    print("\n─── Sample recommendations (section 9) ───")
    shown = 0
    for c in chunks:
        if c.section_id == "9" and c.chunk_type == "recommendation":
            print(f"  [{c.rec_id}] grade={c.kdigo_grade}  {c.text[:120]}")
            shown += 1
            if shown >= 5:
                break

    print("\n─── Sample supportive_text ───")
    shown = 0
    for c in chunks:
        if c.chunk_type == "supportive_text":
            print(f"  sec={c.section_id} parent={c.parent_rec_id}\n  {c.text[:200]}\n")
            shown += 1
            if shown >= 3:
                break
