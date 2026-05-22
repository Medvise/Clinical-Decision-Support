from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.0.0"

EvidenceMode = Literal["rag", "llm_synthesis"]
ConfidenceLevel = Literal["High", "Medium", "Low"]
RouteType = Literal["patient_specific", "general_guideline", "clarification_needed"]
FlagSeverity = Literal["info", "warning", "critical"]
DiseaseTag = Literal["CKD", "hypertension", "ACHD"]


class CDSSRequest(BaseModel):
    patient_id: str = ""
    query: str = Field(..., min_length=1, max_length=4000)
    evidence_mode: EvidenceMode = "rag"


class LabValue(BaseModel):
    value: float
    unit: str | None = None
    observed_at: date | None = None


class Demographics(BaseModel):
    gender: str | None = None
    dob: date | None = None
    age: int | None = None
    first_name: str | None = None
    last_name: str | None = None


class ClinicalFlag(BaseModel):
    code: str
    severity: FlagSeverity
    message: str


class PatientSummary(BaseModel):
    patient_id: str | None = None
    primary_disease: DiseaseTag | None = None
    disease_tags: list[str] = []
    comorbidity_tags: list[str] = []
    stage: str | None = None
    acr_category: str | None = None
    demographics: Demographics | None = None
    key_values: dict[str, LabValue] = {}
    flags: list[ClinicalFlag] = []
    medications: list[str] = []
    icd_codes: list[str] = []
    clinical_observations: list[dict] = []


class Citation(BaseModel):
    index: int | None = None
    source_line: str | None = None
    excerpt: str
    full_text: str | None = None
    guideline_family: str | None = None


class RetrievalMeta(BaseModel):
    chunk_count: int = 0
    disease_tags: list[str] = []
    ckd_stage_tags: list[str] = []
    bp_stage_tags: list[str] = []
    semantic_query: str | None = None
    tagger_rationale: str | None = None


class ResponseMeta(BaseModel):
    schema_version: str = SCHEMA_VERSION
    request_id: str
    route: RouteType
    route_reason: str | None = None
    evidence_mode: EvidenceMode
    model: str
    latency_ms: float | None = None
    retrieval: RetrievalMeta | None = None


class CDSSResponse(BaseModel):
    schema_version: str = SCHEMA_VERSION
    recommendation: str
    reasoning: str
    citations: list[Citation]
    patient_summary: PatientSummary | None = None
    llm_summary: str | None = None
    confidence: ConfidenceLevel
    meta: ResponseMeta


class ErrorResponse(BaseModel):
    schema_version: str = SCHEMA_VERSION
    request_id: str
    code: str
    message: str
