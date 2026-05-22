"""Build Qdrant Filter objects from retrieval payload_filters."""

from __future__ import annotations

from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue


def build_qdrant_filter(
    payload_filters: dict,
    *,
    apply_disease_tags: bool = True,
    apply_stage: bool = True,
    apply_bp_stage: bool = True,
) -> Filter:
    must: list = []

    chunk_types = payload_filters.get("chunk_types", [])
    if chunk_types:
        must.append(
            FieldCondition(key="chunk_type", match=MatchAny(any=chunk_types))
        )

    if apply_disease_tags:
        disease_tags = payload_filters.get("disease_tags") or []
        if disease_tags:
            must.append(
                FieldCondition(
                    key="disease_tags",
                    match=MatchAny(any=disease_tags),
                )
            )

    should: list = []

    if apply_stage:
        for stage_tag in payload_filters.get("ckd_stage_tags") or []:
            should.append(
                FieldCondition(
                    key="ckd_stage_tags",
                    match=MatchValue(value=stage_tag),
                )
            )

    if apply_bp_stage:
        bp_tags = payload_filters.get("bp_stage_tags") or []
        if bp_tags:
            should.append(
                FieldCondition(
                    key="bp_stage_tags",
                    match=MatchAny(any=bp_tags),
                )
            )

    for tag in payload_filters.get("drug_tags", []):
        should.append(
            FieldCondition(key="drug_tags", match=MatchValue(value=tag))
        )

    comorbidity_tags = payload_filters.get("comorbidity_tags") or []
    if comorbidity_tags:
        should.append(
            FieldCondition(
                key="comorbidity_tags",
                match=MatchAny(any=comorbidity_tags),
            )
        )

    return Filter(must=must, should=should if should else None)
