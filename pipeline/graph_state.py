"""LangGraph state definition for the CDSS pipeline."""

from __future__ import annotations

from typing import Any

from typing_extensions import TypedDict

from api.schema import EvidenceMode, RouteType


class CDSSState(TypedDict, total=False):
    patient_id: str
    query: str
    evidence_mode: EvidenceMode
    route: RouteType
    route_reason: str
    summary: dict[str, Any] | None
    retrieval_spec: dict[str, Any] | None
    chunk_records: list[dict[str, Any]] | None
    chunks: list[str]
    prompt: str
    result: dict[str, Any]
