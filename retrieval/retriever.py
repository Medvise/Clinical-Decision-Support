"""
Public retrieval API.

Implementation is split across query_decompose, qdrant_filters, qdrant_client,
rerank, and search modules; this module re-exports the stable surface.
"""

from retrieval.query_decompose import (
    decompose_query,
    infer_comorbidity_tags_from_query,
    infer_disease_tags_from_query,
)
from retrieval.search import RetrievalOutcome, retrieve_guideline_chunks

__all__ = [
    "RetrievalOutcome",
    "decompose_query",
    "infer_comorbidity_tags_from_query",
    "infer_disease_tags_from_query",
    "retrieve_guideline_chunks",
]
