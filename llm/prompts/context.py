"""Patient context formatting for LLM prompts."""

from rules.utils import format_yyyymmdd_display


def format_prompt_date(date_value) -> str:
    if not date_value:
        return "Unknown"
    return format_yyyymmdd_display(date_value) or "Unknown"


def format_observation_line(obs: dict) -> str:
    value = obs.get("value")
    unit = obs.get("unit")
    if unit:
        value_text = f"{value} {unit}"
    else:
        value_text = str(value)
    if obs.get("date"):
        return f"- {obs['label']}: {value_text} (date: {format_prompt_date(obs['date'])})"
    return f"- {obs['label']}: {value_text}"


def patient_context_block(summary: dict) -> str:
    """Shared patient narrative for RAG and LLM-synthesis prompts."""
    demographics = summary.get("demographics", {})
    observations = summary.get("clinical_observations", [])
    meds = summary.get("medications") or []
    icd_codes = summary.get("icd_codes") or []
    flags = summary.get("flags") or []

    lines = ["PATIENT CONTEXT:"]
    rc = summary.get("retrieval_context") or {}
    disease_tags = rc.get("disease_tags") or []
    primary = rc.get("disease")
    if primary or disease_tags:
        tag_text = ", ".join(disease_tags) if disease_tags else str(primary)
        lines.append(f"- Primary disease: {primary or 'unknown'}")
        if disease_tags:
            lines.append(f"- Disease tags: {tag_text}")
    comorbidities = rc.get("comorbidity_tags") or []
    if comorbidities:
        lines.append(f"- Comorbidities: {', '.join(comorbidities)}")

    stage = summary.get("stage") or summary.get("ckd_stage")
    acr_category = summary.get("acr_category")
    if stage or acr_category:
        stage_parts = []
        if stage:
            stage_parts.append(f"Stage: {stage}")
        if acr_category:
            stage_parts.append(f"ACR category: {acr_category}")
        lines.append(f"- {', '.join(stage_parts)}")

    first_name = demographics.get("first_name")
    last_name = demographics.get("last_name")
    patient_name = f"{first_name or ''} {last_name or ''}".strip()
    if patient_name:
        lines.append(f"- Patient name: {patient_name}")

    if demographics.get("gender") is not None:
        lines.append(f"- Gender: {demographics['gender']}")
    if demographics.get("age") is not None:
        lines.append(f"- Age: {demographics['age']}")

    if observations:
        lines.append("- Available clinical data:")
        lines.extend(format_observation_line(obs) for obs in observations)

    if meds:
        lines.append(f"- Medications: {meds}")
    if icd_codes:
        lines.append(f"- ICD codes: {icd_codes}")
    if flags:
        lines.append(f"- Clinical flags: {flags}")

    return "\n".join(lines)
