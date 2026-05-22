"""
aha_parser.py
Parser for AHA/ACC-format guidelines (Hypertension 2025, ACHD 2026).

Fixes applied vs original approach:
  1. COR now parsed from TWO lines when split across newline ("3: No \\nBenefit C-LD")
  2. Each COR/LOE block is first extracted as a bounded scope (between
     "COR LOE Recommendation(s)" header and "Synopsis"), then individual
     rows are parsed within that scope → eliminates false Tier-2 matches
     from numbered lists elsewhere in the document
  3. Synopsis and Supportive Text are extracted ONLY within a known
     knowledge-chunk boundary, never globally
  4. Section summaries are generated ONLY for depth-1 and depth-2 section
     headings (e.g. "3." or "3.1."), not deeper subsections
"""

from __future__ import annotations
import re
from typing import List, Tuple, Optional

from .base_parser import Chunk, clean_text, enrich_tags, min_length_ok


# ─────────────────────────────────────────────────────────────────────────────
# Compiled patterns
# ─────────────────────────────────────────────────────────────────────────────

# Section heading at depth 1–2 only: "5." or "5.1." (not "5.1.2.")
# Used for Tier-3 section_summary
_SECTION_D1_D2 = re.compile(
    r'^(?P<sec_id>\d+(?:\.\d+)?)\.\s+(?P<title>[A-Z][^\n]{3,100})$',
    re.MULTILINE
)

# Section heading at any depth: used to detect section context for each chunk
_SECTION_ANY = re.compile(
    r'^(?P<sec_id>\d+(?:\.\d+){0,3})\.\s+(?P<title>[A-Z][^\n]{3,100})$',
    re.MULTILINE
)

# "Recommendations for <topic>" subsection header — marks start of a knowledge chunk.
# Case-SENSITIVE: real headers start with capital "R". Lowercase occurrences in
# reference list citations (e.g. "recommendations for the validation of…") are excluded.
_REC_FOR_HEADER = re.compile(
    r'^Recommendation(?:s)? for (?P<topic>[^\n]{5,120})$',
    re.MULTILINE          # NO re.IGNORECASE
)

# "COR LOE Recommendation(s)" table header — marks start of rec rows
_COR_LOE_HEADER = re.compile(
    r'^COR\s+LOE\s+Recommendation(?:s)?$',
    re.MULTILINE | re.IGNORECASE
)

# "Synopsis" marker
_SYNOPSIS_MARKER = re.compile(r'^Synopsis$', re.MULTILINE)

# "Recommendation-Specific Supportive Text" marker
_RST_MARKER = re.compile(
    r'^Recommendation-Specific Supportive Text$',
    re.MULTILINE | re.IGNORECASE
)

# End-of-knowledge-chunk: next section heading OR next "Recommendations for"
_CHUNK_END = re.compile(
    r'(?=^\d+(?:\.\d+){0,3}\.\s+[A-Z]|^Recommendation(?:s)? for )',
    re.MULTILINE
)

# COR/LOE row: captures cor+loe on one line, then rec number + text
# Handles:
#   "1 B-NR\n 1. In adults…"
#   "2a C-EO\n 2. When measuring…"
#   "3: No \nBenefit C-LD\n 1. In adults…"   ← split COR across two lines
#   "3: Harm A\n 3. In adults…"
_COR_VALUE_LINE = re.compile(
    r'^(?P<cor>(?:1|2a|2b|3(?::\s*(?:No\s*)?(?:Benefit|Harm|No\s+Benefit)?)?))(?:\s+(?P<loe>[AB]|[ABC]-[A-Z]{1,3}))?$',
    re.IGNORECASE
)
_LOE_CONTINUATION = re.compile(
    r'^(?P<loe_cont>(?:Benefit|Harm|No\s+Benefit)?[\s\w]*[ABC]-[A-Z]{1,3})$',
    re.IGNORECASE
)
_REC_ROW_START = re.compile(r'^\s*(?P<num>\d+)\.\s+(?P<text>.+)')

# Grade in supportive text item: " N. text..."
_SUPPORT_ITEM = re.compile(r'^\s*(?P<num>\d+)\.\s+(?P<text>.+)')


# ─────────────────────────────────────────────────────────────────────────────
# Helper: find the enclosing section for a text position
# ─────────────────────────────────────────────────────────────────────────────

def _find_section(text: str, pos: int) -> Tuple[str, str]:
    """Return (section_id, section_title) of the nearest preceding section heading."""
    sec_id, sec_title = "", ""
    for m in _SECTION_ANY.finditer(text):
        if m.start() > pos:
            break
        sec_id = m.group("sec_id")
        sec_title = m.group("title").strip()
    return sec_id, sec_title


# ─────────────────────────────────────────────────────────────────────────────
# Helper: parse COR and LOE from consecutive raw lines
# ─────────────────────────────────────────────────────────────────────────────

def _parse_cor_loe_from_lines(lines: list[str], idx: int) -> Tuple[str, str, int]:
    """
    Starting at lines[idx], attempt to parse a COR+LOE pair.
    Returns (cor, loe, next_idx) where next_idx points past the consumed lines.

    Handles two cases:
      Case A (single line):  "1 B-NR"   →  cor="1",            loe="B-NR"
      Case B (split):        "3: No "   →  first line has cor fragment;
                             "Benefit C-LD" on next line completes cor+loe
    """
    line = lines[idx].strip()

    # Try single-line match first
    m = _COR_VALUE_LINE.match(line)
    if m:
        cor = _normalise_cor(m.group("cor"))
        loe = (m.group("loe") or "").strip()
        if not loe and idx + 1 < len(lines):
            # LOE may be on next line (rare but seen in split layout)
            next_line = lines[idx + 1].strip()
            loe_m = re.match(r'^([ABC]-[A-Z]{1,3})$', next_line, re.IGNORECASE)
            if loe_m:
                return cor, loe_m.group(1), idx + 2
        return cor, loe, idx + 1

    # Try split COR across two lines: "3: No " then "Benefit C-LD"
    if re.match(r'^3:\s*(?:No)?\s*$', line, re.IGNORECASE) and idx + 1 < len(lines):
        next_line = lines[idx + 1].strip()
        # next_line should be like "Benefit C-LD" or "Harm A"
        m2 = re.match(
            r'^(?P<cor_cont>(?:No\s+)?(?:Benefit|Harm))\s+(?P<loe>[ABC]-[A-Z]{1,3})$',
            next_line, re.IGNORECASE
        )
        if m2:
            cor = f"3: {m2.group('cor_cont').strip()}"
            return cor, m2.group("loe"), idx + 2

    return "", "", idx  # not a COR/LOE line


def _normalise_cor(raw: str) -> str:
    raw = raw.strip()
    if re.match(r'^1$', raw):            return "1"
    if re.match(r'^2a$', raw, re.I):    return "2a"
    if re.match(r'^2b$', raw, re.I):    return "2b"
    if re.match(r'^3$', raw):            return "3"
    if re.match(r'^3:\s*Harm', raw, re.I):        return "3: Harm"
    if re.match(r'^3:\s*No\s*Benefit', raw, re.I): return "3: No Benefit"
    if re.match(r'^3:\s*No\s*$', raw, re.I):       return "3: No"  # partial; resolved by caller
    return raw


# ─────────────────────────────────────────────────────────────────────────────
# Core parser: extract all knowledge chunks from a text block
# ─────────────────────────────────────────────────────────────────────────────

def _parse_knowledge_chunk(
    block_text: str,
    full_text: str,
    block_start: int,
    topic: str,
    guideline: str,
    disease: str,
    source_doc: str,
) -> List[Chunk]:
    """
    Parse one AHA/ACC knowledge chunk (the text between a 'Recommendations for…'
    header and the next heading or 'Recommendations for…').

    Returns Tier-1 (rec) + Tier-2 (synopsis, supportive_text) chunks.
    """
    chunks: List[Chunk] = []
    sec_id, sec_title = _find_section(full_text, block_start)
    lines = block_text.splitlines()

    # ── Locate key sub-blocks within the chunk ──────────────────────────────
    cor_loe_pos   = None  # line index of "COR LOE Recommendation(s)"
    synopsis_pos  = None  # line index of "Synopsis"
    rst_pos       = None  # line index of "Recommendation-Specific Supportive Text"

    for i, line in enumerate(lines):
        stripped = line.strip()
        if re.match(r'^COR\s+LOE\s+Recommendation', stripped, re.IGNORECASE):
            cor_loe_pos = i
        elif stripped == "Synopsis":
            synopsis_pos = i
        elif stripped == "Recommendation-Specific Supportive Text":
            rst_pos = i

    # Continuation blocks (page-split) may lack a COR LOE header but still
    # contain COR rows. Fall back to scanning from line 1.
    if cor_loe_pos is None:
        cor_loe_pos = 0

    # ── Tier-1: parse recommendation rows ───────────────────────────────────
    rec_ids_in_block: List[str] = []

    if True:
        # Rows live between COR LOE header and Synopsis (or RST, or end of block)
        row_end = synopsis_pos or rst_pos or len(lines)
        i = cor_loe_pos + 1
        while i < row_end:
            line = lines[i].strip()
            if not line:
                i += 1
                continue

            cor, loe, next_i = _parse_cor_loe_from_lines(lines, i)
            if cor and next_i > i:
                i = next_i
                # Gather recommendation text lines until next COR line or boundary
                rec_text_lines: List[str] = []
                while i < row_end:
                    peek = lines[i].strip()
                    # Stop if we hit another COR-like line
                    cor2, _, _ = _parse_cor_loe_from_lines(lines, i)
                    if cor2:
                        break
                    # Stop at boundary markers
                    if peek in ("Synopsis", "Recommendation-Specific Supportive Text"):
                        break
                    # Skip reference-only lines (e.g. "1,2" or "3–5,10")
                    if re.match(r'^[\d,–\-\s]+$', peek):
                        i += 1
                        continue
                    rec_text_lines.append(lines[i])
                    i += 1

                rec_text = " ".join(t.strip() for t in rec_text_lines).strip()
                # Derive rec number from leading "N. "
                m_num = _REC_ROW_START.match(rec_text)
                rec_num = m_num.group("num") if m_num else str(len(rec_ids_in_block) + 1)

                rec_id = f"{guideline}_{sec_id}_rec{rec_num}" if sec_id else \
                         f"{guideline}_rec{len(rec_ids_in_block)+1}"
                rec_ids_in_block.append(rec_id)

                full_chunk_text = (
                    f"[{sec_id} — {sec_title}]\n"
                    f"Recommendations for {topic}\n"
                    f"COR: {cor} | LOE: {loe}\n"
                    f"{rec_text}"
                )

                if min_length_ok(rec_text, "recommendation"):
                    c = Chunk(
                        text=full_chunk_text,
                        chunk_type="recommendation",
                        guideline=guideline,
                        disease=disease,
                        source_doc=source_doc,
                        section_id=sec_id,
                        section_title=sec_title,
                        rec_id=rec_id,
                        cor=cor,
                        loe=loe,
                    )
                    chunks.append(enrich_tags(c))
            else:
                i += 1

    # ── Tier-2a: Synopsis ────────────────────────────────────────────────────
    if synopsis_pos is not None:
        syn_end = rst_pos or len(lines)
        syn_lines = lines[synopsis_pos + 1: syn_end]
        # Remove trailing blank/noise lines
        syn_text = "\n".join(l for l in syn_lines if l.strip()).strip()

        if min_length_ok(syn_text, "synopsis") and rec_ids_in_block:
            full_syn_text = (
                f"[{sec_id} — {sec_title}]\n"
                f"Synopsis — {topic}\n\n"
                f"{syn_text}"
            )
            c = Chunk(
                text=full_syn_text,
                chunk_type="synopsis",
                guideline=guideline,
                disease=disease,
                source_doc=source_doc,
                section_id=sec_id,
                section_title=sec_title,
                parent_rec_id=",".join(rec_ids_in_block),
            )
            chunks.append(enrich_tags(c))

    # ── Tier-2b: Recommendation-Specific Supportive Text items ──────────────
    if rst_pos is not None:
        rst_lines = lines[rst_pos + 1:]
        # Build items: each starts with a line matching " N. text..."
        # We accumulate lines until the next " M. " or end
        current_num: Optional[str] = None
        current_lines: List[str] = []

        def _flush_support_item(num, item_lines, sec_id, sec_title, topic,
                                 guideline, disease, source_doc, rec_ids_in_block, chunks):
            item_text = " ".join(l.strip() for l in item_lines if l.strip()).strip()
            if not min_length_ok(item_text, "supportive_text"):
                return
            # Try to map this item number to a rec_id
            parent = ""
            if rec_ids_in_block:
                try:
                    idx = int(num) - 1
                    parent = rec_ids_in_block[idx] if 0 <= idx < len(rec_ids_in_block) \
                             else rec_ids_in_block[-1]
                except (ValueError, IndexError):
                    parent = rec_ids_in_block[0]

            full_text = (
                f"[{sec_id} — {sec_title}]\n"
                f"Recommendation-Specific Supportive Text ({topic}), item {num}:\n\n"
                f"{item_text}"
            )
            c = Chunk(
                text=full_text,
                chunk_type="supportive_text",
                guideline=guideline,
                disease=disease,
                source_doc=source_doc,
                section_id=sec_id,
                section_title=sec_title,
                parent_rec_id=parent,
            )
            chunks.append(enrich_tags(c))

        for line in rst_lines:
            m = _SUPPORT_ITEM.match(line)
            if m:
                if current_num is not None:
                    _flush_support_item(current_num, current_lines, sec_id,
                                        sec_title, topic, guideline, disease,
                                        source_doc, rec_ids_in_block, chunks)
                current_num = m.group("num")
                current_lines = [m.group("text")]
            else:
                if current_num is not None:
                    stripped = line.strip()
                    # Stop accumulating on noise markers
                    if re.match(r'^\d+(?:\.\d+){0,3}\.\s+[A-Z]', stripped):
                        _flush_support_item(current_num, current_lines, sec_id,
                                            sec_title, topic, guideline, disease,
                                            source_doc, rec_ids_in_block, chunks)
                        current_num = None
                        current_lines = []
                    else:
                        current_lines.append(line)

        if current_num is not None:
            _flush_support_item(current_num, current_lines, sec_id,
                                sec_title, topic, guideline, disease,
                                source_doc, rec_ids_in_block, chunks)

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Table parser
# ─────────────────────────────────────────────────────────────────────────────

_TABLE_HEADER = re.compile(
    r'^Table\s+(?P<num>\d+)[.\s]+(?P<title>[^\n]{5,150})$',
    re.MULTILINE | re.IGNORECASE
)

def _parse_tables(text: str, guideline: str, disease: str, source_doc: str) -> List[Chunk]:
    chunks: List[Chunk] = []
    matches = list(_TABLE_HEADER.finditer(text))
    for idx, m in enumerate(matches):
        body_start = m.end()
        body_end   = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        # Cap table body at 1500 chars to avoid swallowing adjacent prose
        body = text[body_start:min(body_start + 1500, body_end)].strip()
        if not min_length_ok(body, "table"):
            continue
        sec_id, sec_title = _find_section(text, m.start())
        full_text = f"Table {m.group('num')}. {m.group('title').strip()}\n{body}"
        c = Chunk(
            text=full_text,
            chunk_type="table",
            guideline=guideline,
            disease=disease,
            source_doc=source_doc,
            section_id=sec_id,
            section_title=sec_title,
        )
        chunks.append(enrich_tags(c))
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Section summary parser (Tier-3)  — depth 1 and 2 ONLY
# ─────────────────────────────────────────────────────────────────────────────

_SECTION_D1_D2_RE = re.compile(
    r'^(?P<sec_id>\d+(?:\.\d+)?)\.\s+(?P<title>[A-Z][^\n]{3,100})$',
    re.MULTILINE
)

def _parse_section_summaries(text: str, guideline: str, disease: str, source_doc: str) -> List[Chunk]:
    chunks: List[Chunk] = []
    all_matches = list(_SECTION_D1_D2_RE.finditer(text))
    # Deduplicate by sec_id keeping the LAST occurrence (body always after TOC)
    seen: dict = {}
    for m in all_matches:
        seen[m.group("sec_id")] = m
    matches = sorted(seen.values(), key=lambda m: m.start())
    for idx, m in enumerate(matches):
        body_start = m.end()
        body_end   = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        # Use only the first 800 chars of the opening prose
        opening = text[body_start:body_start + 800].strip()
        # Remove leading "Recommendations for…" lines — we want only prose
        opening = re.sub(r'^Recommendation(?:s)? for[^\n]+\n?', '', opening, flags=re.IGNORECASE)
        opening = opening.strip()
        if not min_length_ok(opening, "section_summary"):
            continue
        full_text = (
            f"[Section {m.group('sec_id')} — {m.group('title').strip()}]\n\n"
            f"{opening}"
        )
        c = Chunk(
            text=full_text,
            chunk_type="section_summary",
            guideline=guideline,
            disease=disease,
            source_doc=source_doc,
            section_id=m.group("sec_id"),
            section_title=m.group("title").strip(),
        )
        chunks.append(enrich_tags(c))
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse(raw_text: str, guideline: str, disease: str, source_doc: str) -> List[Chunk]:
    """
    Parse an AHA/ACC guideline PDF text into chunks.

    Parameters
    ----------
    raw_text   : full concatenated pypdf output
    guideline  : "HTN" or "ACHD"
    disease    : "hypertension" or "ACHD"
    source_doc : PDF filename stem

    Returns
    -------
    List[Chunk]
    """
    text = clean_text(raw_text, guideline)

    # Strip content after the true REFERENCES/bibliography section.
    # Use the LAST standalone "REFERENCES" heading confirmed by a following
    # numbered item line — earlier occurrences may be table column headers.
    _ref_matches = list(re.finditer(r'^REFERENCES\s*$', text, re.MULTILINE | re.IGNORECASE))
    for _rm in reversed(_ref_matches):
        _window = text[_rm.start(): _rm.start() + 400]
        if re.search(r'\n\s*1\.\s+[A-Z]', _window):
            text = text[:_rm.start()]
            break

    chunks: List[Chunk] = []

    # Split the document at each "Recommendations for …" header
    rec_for_matches = list(_REC_FOR_HEADER.finditer(text))

    for idx, m in enumerate(rec_for_matches):
        topic = m.group("topic").strip()
        block_start = m.start()
        # Block ends at the next "Recommendations for" or major section heading
        # Boundary is the next "Recommendations for" block only.
        # Using section headings as a secondary boundary cuts blocks too short
        # because TOC entries appear immediately after rec_for headers.
        if idx + 1 < len(rec_for_matches):
            block_end = rec_for_matches[idx + 1].start()
        else:
            block_end = len(text)
        block_text = text[block_start:block_end]

        chunks.extend(
            _parse_knowledge_chunk(
                block_text=block_text,
                full_text=text,
                block_start=block_start,
                topic=topic,
                guideline=guideline,
                disease=disease,
                source_doc=source_doc,
            )
        )

    # ── Tier-1: Tables ───────────────────────────────────────────────────────
    chunks.extend(_parse_tables(text, guideline, disease, source_doc))

    # ── Tier-3: Section summaries (depth 1–2 only) ───────────────────────────
    chunks.extend(_parse_section_summaries(text, guideline, disease, source_doc))

    return chunks
