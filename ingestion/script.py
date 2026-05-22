#!/usr/bin/env python3
"""
Multi-guideline ingestion → unified Qdrant payload (HTN, ACHD, KDIGO/NICE).

PDF → ingestion.parser_ref (detect_guideline, kdigo/aha parsers) → embeddings → Qdrant

Usage:
  python -m ingestion.script path/to/KDIGO.pdf path/to/HTN.pdf ...
  # or: INGEST_PDF_PATHS="a.pdf,b.pdf" python -m ingestion.script

Environment: QDRANT_URL, QDRANT_COLLECTION, EMBEDDINGS_URL (see config.py defaults).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from pathlib import Path

import requests
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException
from qdrant_client.models import Distance, PointStruct, VectorParams

# Project root on sys.path for `python ingestion/script.py`
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import (  # noqa: E402
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION,
    EMBEDDINGS_URL,
    EMBEDDINGS_TIMEOUT_SECONDS,
)
from ingestion.parser_ref import parse_pdf  # noqa: E402


EMBEDDING_BATCH_SIZE = 50
UPSERT_BATCH_SIZE = 25
QDRANT_TIMEOUT_SECONDS = int(os.getenv("QDRANT_TIMEOUT_SECONDS", "180"))


def extract_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    parts: list[str] = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            parts.append(page_text)
    return "\n".join(parts)


def get_embeddings(texts: list[str]) -> list[list[float]]:
    resp = requests.post(
        EMBEDDINGS_URL,
        json={"sentences": texts},
        timeout=(10, EMBEDDINGS_TIMEOUT_SECONDS),
    )
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        body = (resp.text or "").strip()
        if len(body) > 500:
            body = body[:500] + "…"
        raise RuntimeError(
            f"Embeddings request failed ({resp.status_code}) for {EMBEDDINGS_URL}. "
            f"Response body: {body or '<empty>'}"
        ) from e
    return resp.json()["embeddings"]


def create_collection(client: QdrantClient, collection_name: str, vector_size: int) -> None:
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        print(f"Created collection: {collection_name}")


def process_pdf(pdf_path: Path) -> list[dict]:
    print(f"Processing {pdf_path.name}")
    raw = extract_text(pdf_path)
    chunks = parse_pdf(pdf_path, raw)
    tier1 = sum(1 for c in chunks if c["chunk_type"] in {"recommendation", "practice_point", "table"})
    tier2 = sum(
        1 for c in chunks if c["chunk_type"] in {"rationale", "synopsis", "supportive_text"}
    )
    tier3 = sum(1 for c in chunks if c["chunk_type"] == "section_summary")
    print(
        f"  {pdf_path.name}: tier1={tier1} | tier2={tier2} | tier3={tier3} | total={len(chunks)}"
    )
    return chunks


def _payload_from_chunk(chunk: dict) -> dict:
    return {
        "text": chunk["text"],
        "chunk_type": chunk["chunk_type"],
        "guideline": chunk["guideline"],
        "disease": chunk["disease"],
        "source_doc": chunk["source_doc"],
        "section_id": chunk["section_id"],
        "chapter_id": chunk["chapter_id"],
        "section_title": chunk["section_title"],
        "rec_id": chunk["rec_id"],
        "parent_rec_id": chunk["parent_rec_id"],
        "cor": chunk["cor"],
        "loe": chunk["loe"],
        "kdigo_grade": chunk["kdigo_grade"],
        "drug_tags": chunk["drug_tags"],
        "comorbidity_tags": chunk["comorbidity_tags"],
        "bp_stage_tags": chunk["bp_stage_tags"],
        "ckd_stage_tags": chunk["ckd_stage_tags"],
        "intervention_tags": chunk["intervention_tags"],
        "disease_tags": chunk["disease_tags"],
    }


def _resolve_pdf_paths(argv_paths: list[str]) -> list[Path]:
    env_paths = os.getenv("INGEST_PDF_PATHS", "").strip()
    if argv_paths:
        return [Path(p).expanduser().resolve() for p in argv_paths]
    if env_paths:
        return [Path(p.strip()).expanduser().resolve() for p in env_paths.split(",") if p.strip()]
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest clinical guideline PDFs into Qdrant.")
    parser.add_argument(
        "pdfs",
        nargs="*",
        help="PDF files (or set INGEST_PDF_PATHS=comma-separated list)",
    )
    args = parser.parse_args()

    missing = [
        name
        for name, value in {
            "QDRANT_URL": QDRANT_URL,
            "QDRANT_COLLECTION": QDRANT_COLLECTION,
            "EMBEDDINGS_URL": EMBEDDINGS_URL,
        }.items()
        if not value
    ]
    if missing:
        raise RuntimeError(
            "Missing required configuration: "
            + ", ".join(missing)
            + ". Set in `.env` or environment (see config.py)."
        )

    pdf_files = _resolve_pdf_paths(args.pdfs)
    if not pdf_files:
        raise SystemExit(
            "No PDFs specified. Pass paths as arguments or set INGEST_PDF_PATHS."
        )

    for p in pdf_files:
        if not p.exists():
            raise FileNotFoundError(p)

    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=QDRANT_TIMEOUT_SECONDS,
    )
    all_chunks: list[dict] = []
    for pdf in pdf_files:
        all_chunks.extend(process_pdf(pdf))

    if not all_chunks:
        raise ValueError("No chunks extracted from PDFs. Check PDF text extraction.")

    first_vector = get_embeddings([all_chunks[0]["text"]])[0]
    vector_size = len(first_vector)
    create_collection(client, QDRANT_COLLECTION, vector_size)

    total = len(all_chunks)
    for start in range(0, total, EMBEDDING_BATCH_SIZE):
        batch = all_chunks[start : start + EMBEDDING_BATCH_SIZE]
        texts = [c["text"] for c in batch]
        embeddings = get_embeddings(texts)

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embeddings[i],
                payload=_payload_from_chunk(chunk),
            )
            for i, chunk in enumerate(batch)
        ]

        for pstart in range(0, len(points), UPSERT_BATCH_SIZE):
            sub_batch = points[pstart : pstart + UPSERT_BATCH_SIZE]
            for attempt in range(1, 4):
                try:
                    client.upsert(
                        collection_name=QDRANT_COLLECTION,
                        points=sub_batch,
                        wait=True,
                        timeout=QDRANT_TIMEOUT_SECONDS,
                    )
                    break
                except ResponseHandlingException:
                    if attempt >= 3:
                        raise
                    time.sleep(1.5 * attempt)

        print(f"Upserted {start + len(batch)} / {total}")
        time.sleep(0.1)

    print(f"Done: indexed into {QDRANT_COLLECTION}")


if __name__ == "__main__":
    main()
