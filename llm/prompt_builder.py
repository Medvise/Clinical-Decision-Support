"""
Public prompt builder API.

Implementation lives under llm.prompts; this module re-exports for backward compatibility.
"""

from llm.prompts.assemble import (
    assemble_general_prompt,
    assemble_general_synthesis_prompt,
    assemble_patient_synthesis_prompt,
    assemble_prompt,
)
from llm.prompts.context import patient_context_block

# Backward-compatible alias used by retrieval_tagger
_patient_context_block = patient_context_block

__all__ = [
    "assemble_general_prompt",
    "assemble_general_synthesis_prompt",
    "assemble_patient_synthesis_prompt",
    "assemble_prompt",
    "patient_context_block",
    "_patient_context_block",
]
