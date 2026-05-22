"""Map server-built patient summary dicts to API PatientSummary models."""

from __future__ import annotations

import re
from datetime import date, datetime

from api.schema import ClinicalFlag, Demographics, FlagSeverity, LabValue, PatientSummary


def _parse_date(value) -> date | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    raw = str(value).strip()
    if not raw:
        return None
    for fmt in ("%Y%m%d", "%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    return None


def _flag_code(message: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", message.lower()).strip("_")
    return (slug[:48] or "clinical_flag")


def _flag_severity(message: str) -> FlagSeverity:
    lower = message.lower()
    if any(k in lower for k in ("critical", "urgent", "emergency", "severe", "danger")):
        return "critical"
    if any(k in lower for k in ("hyperkalemia", "uncontrolled", "missing", "gap", "elevated")):
        return "warning"
    return "info"


def _to_clinical_flags(flags: list) -> list[ClinicalFlag]:
    out: list[ClinicalFlag] = []
    for item in flags or []:
        if isinstance(item, ClinicalFlag):
            out.append(item)
            continue
        if isinstance(item, dict):
            out.append(
                ClinicalFlag(
                    code=item.get("code") or _flag_code(str(item.get("message", ""))),
                    severity=item.get("severity") or _flag_severity(str(item.get("message", ""))),
                    message=str(item.get("message", "")),
                )
            )
            continue
        msg = str(item).strip()
        if msg:
            out.append(ClinicalFlag(code=_flag_code(msg), severity=_flag_severity(msg), message=msg))
    return out


def _to_key_values(raw: dict | None) -> dict[str, LabValue]:
    if not raw:
        return {}
    out: dict[str, LabValue] = {}
    for key, val in raw.items():
        if val is None:
            continue
        if isinstance(val, LabValue):
            out[key] = val
        elif isinstance(val, dict) and "value" in val:
            out[key] = LabValue(**{k: v for k, v in val.items() if k in LabValue.model_fields})
        else:
            try:
                out[key] = LabValue(value=float(val))
            except (TypeError, ValueError):
                continue
    return out


def _to_demographics(raw: dict | None) -> Demographics | None:
    if not raw:
        return None
    return Demographics(
        gender=raw.get("gender"),
        dob=_parse_date(raw.get("dob") or raw.get("memdob")),
        age=raw.get("age"),
        first_name=raw.get("first_name"),
        last_name=raw.get("last_name"),
    )


def _normalize_primary(disease: str | None) -> str | None:
    if disease in ("CKD", "hypertension", "ACHD"):
        return disease
    return None


def to_patient_summary(raw: dict) -> PatientSummary:
    rc = raw.get("retrieval_context") or {}
    disease_tags = list(rc.get("disease_tags") or [])
    primary = _normalize_primary(rc.get("disease"))
    if primary and primary not in disease_tags:
        disease_tags = [primary] + [t for t in disease_tags if t != primary]

    stage = raw.get("stage") or raw.get("ckd_stage") or rc.get("stage")

    return PatientSummary(
        patient_id=raw.get("patient_id"),
        primary_disease=primary,
        disease_tags=disease_tags,
        comorbidity_tags=list(rc.get("comorbidity_tags") or []),
        stage=stage if stage not in (None, "", "Unknown", "Not CKD") else stage,
        acr_category=raw.get("acr_category") or rc.get("acr_category"),
        demographics=_to_demographics(raw.get("demographics")),
        key_values=_to_key_values(raw.get("key_values")),
        flags=_to_clinical_flags(raw.get("flags") or []),
        medications=list(raw.get("medications") or []),
        icd_codes=list(raw.get("icd_codes") or []),
        clinical_observations=list(raw.get("clinical_observations") or []),
    )
