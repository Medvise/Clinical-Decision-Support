"""Pack/unpack guideline evidence blocks; parse LLM citations into structured form."""

from __future__ import annotations

import re
from typing import Any

from api.schema import Citation


def pack_evidence_block(ref: int, source_line: str, body: str) -> str:
    """Wrap one chunk so we can parse ref, provenance, and full text later."""
    return (
        f"<<<CHUNK_REF:{ref}>>>\n"
        f"SOURCE_LINE: {source_line}\n"
        f"BEGIN_GUIDELINE_TEXT\n{body}\nEND_GUIDELINE_TEXT"
    )


def parse_evidence_blocks(chunks: list[str]) -> dict[int, dict[str, str]]:
    """
    Map evidence index (1-based) -> {source_line, full_text}.
    Skips strings that are not packed evidence blocks.
    """
    out: dict[int, dict[str, str]] = {}
    for raw in chunks:
        if not isinstance(raw, str) or not raw.startswith("<<<CHUNK_REF:"):
            continue
        try:
            head, _, tail = raw.partition(">>>\nSOURCE_LINE: ")
            if not tail:
                continue
            ref_part = head.removeprefix("<<<CHUNK_REF:")
            ref = int(ref_part)
            mid, _, body = tail.partition("\nBEGIN_GUIDELINE_TEXT\n")
            if not body:
                continue
            if body.endswith("\nEND_GUIDELINE_TEXT"):
                body = body[: -len("\nEND_GUIDELINE_TEXT")]
            out[ref] = {"source_line": mid.strip(), "full_text": body.strip()}
        except (ValueError, AttributeError):
            continue
    return out


_CITATION_REF = re.compile(r"^\s*\[(\d+)\]")
_CITATION_PARSED = re.compile(
    r'^\s*\[(\d+)\]\s*\(SOURCE_LINE:\s*(.*?)\)\s*["\u201c](.+?)["\u201d]\s*$',
    re.DOTALL,
)


def _guess_guideline_family(source_line: str | None, excerpt: str) -> str | None:
    combined = f"{source_line or ''} {excerpt}".lower()
    for name in ("kdigo", "nice", "acc", "aha", "esc", "htn", "achd"):
        if name in combined:
            return name.upper() if name in ("kdigo", "nice", "esc") else name
    return None


def parse_citation_string(cit: str, blocks: dict[int, dict[str, str]] | None = None) -> Citation:
    """Parse one citation string into a Citation model."""
    blocks = blocks or {}
    if not isinstance(cit, str):
        return Citation(excerpt=str(cit))

    text = cit.strip()
    m = _CITATION_PARSED.match(text)
    if m:
        ref = int(m.group(1))
        source_line = m.group(2).strip()
        excerpt = m.group(3).strip()
        block = blocks.get(ref, {})
        return Citation(
            index=ref,
            source_line=source_line or block.get("source_line"),
            excerpt=excerpt,
            full_text=block.get("full_text"),
            guideline_family=_guess_guideline_family(source_line, excerpt),
        )

    m_ref = _CITATION_REF.match(text)
    if m_ref:
        ref = int(m_ref.group(1))
        block = blocks.get(ref, {})
        if block:
            return Citation(
                index=ref,
                source_line=block.get("source_line"),
                excerpt=text,
                full_text=block.get("full_text"),
                guideline_family=_guess_guideline_family(block.get("source_line"), text),
            )
        return Citation(index=ref, excerpt=text)

    return Citation(
        excerpt=text,
        guideline_family=_guess_guideline_family(None, text),
    )


def parse_citations(raw_citations: list, chunks: list[str]) -> list[Citation]:
    blocks = parse_evidence_blocks(chunks)
    out: list[Citation] = []
    for cit in raw_citations or []:
        if isinstance(cit, Citation):
            out.append(cit)
        elif isinstance(cit, dict):
            out.append(Citation.model_validate(cit))
        else:
            out.append(parse_citation_string(str(cit), blocks))
    return out


def enrich_citations_with_evidence(result: dict[str, Any], chunks: list[str]) -> dict[str, Any]:
    """
    Attach full_text from retrieved chunks to citation dicts or strings.
    Returns result with citations as list (strings or dicts) for downstream parsing.
    """
    if not result or not isinstance(result.get("citations"), list):
        return result

    index = parse_evidence_blocks(chunks)
    if not index:
        return result

    new_citations: list = []
    for cit in result["citations"]:
        if isinstance(cit, dict):
            ref_m = cit.get("index")
            if ref_m is None and cit.get("excerpt"):
                m = _CITATION_REF.match(str(cit["excerpt"]))
                ref_m = int(m.group(1)) if m else None
            if ref_m and index.get(ref_m):
                block = index[ref_m]
                cit = {
                    **cit,
                    "full_text": block.get("full_text"),
                    "source_line": cit.get("source_line") or block.get("source_line"),
                }
            new_citations.append(cit)
            continue

        if not isinstance(cit, str):
            new_citations.append(cit)
            continue

        m = _CITATION_REF.match(cit)
        if not m:
            new_citations.append(cit)
            continue
        ref = int(m.group(1))
        block = index.get(ref)
        if not block:
            new_citations.append(cit)
            continue
        full = (
            f"{cit.strip()}\n\n"
            f"--- Full guideline text (evidence [{ref}]) ---\n"
            f"SOURCE: {block['source_line']}\n\n"
            f"{block['full_text']}"
        )
        new_citations.append(full)

    return {**result, "citations": new_citations}
