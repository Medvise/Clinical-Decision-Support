import logging

import gradio as gr

from pipeline.orchestrator import run_cdss_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _fmt_value(value):
    return "N/A" if value in (None, "", []) else str(value)


def _extract_patient_name(patient_summary: dict) -> str:
    if not patient_summary:
        return "Unknown"
    demographics = patient_summary.get("demographics", {}) or {}
    if isinstance(demographics, dict):
        first_name = demographics.get("first_name") or ""
        last_name = demographics.get("last_name") or ""
        full_name = f"{str(first_name).strip()} {str(last_name).strip()}".strip()
        if full_name:
            return full_name
    return "Unknown"


def _build_medical_data_text(patient_summary: dict) -> str:
    if not patient_summary:
        return ""
    lines: list[str] = []

    primary = patient_summary.get("primary_disease")
    tags = patient_summary.get("disease_tags") or []
    stage = patient_summary.get("stage")
    if primary or tags:
        lines.append(f"Disease: {primary or '—'} | Tags: {', '.join(tags) or '—'}")
    if stage:
        lines.append(f"Stage: {stage}")

    key_values = patient_summary.get("key_values") or {}
    if key_values:
        lines.append("Key values:")
        for key, lab in key_values.items():
            if isinstance(lab, dict):
                val = lab.get("value")
                unit = lab.get("unit")
                text = f"{val} {unit}".strip() if unit else str(val)
            else:
                text = str(lab)
            lines.append(f"  - {key}: {text}")

    for obs in patient_summary.get("clinical_observations", []) or []:
        label = obs.get("label", "Unknown")
        value = _fmt_value(obs.get("value"))
        unit = obs.get("unit")
        if unit:
            value = f"{value} {unit}"
        date_value = obs.get("date")
        if date_value:
            lines.append(f"{label}: {value} (date: {date_value})")
        else:
            lines.append(f"{label}: {value}")

    meds = patient_summary.get("medications") or []
    if meds:
        lines.append(f"Medications: {', '.join(str(m) for m in meds)}")

    return "\n".join(lines)


def _format_citation(citation) -> str:
    if isinstance(citation, dict):
        idx = citation.get("index")
        prefix = f"[{idx}] " if idx is not None else ""
        source = citation.get("source_line") or ""
        excerpt = citation.get("excerpt") or ""
        if source:
            return f"{prefix}({source}) {excerpt}"
        return f"{prefix}{excerpt}"
    return str(citation)


def run_cdss_demo(patient_id: str, query: str, evidence_mode: str):
    cleaned_patient_id = (patient_id or "").strip()
    cleaned_query = (query or "").strip()

    if not cleaned_query:
        return (
            "",
            "",
            "Please enter a clinical query.",
            "",
            "",
        )

    try:
        result = run_cdss_pipeline(
            patient_id=cleaned_patient_id,
            query=cleaned_query,
            evidence_mode=evidence_mode or "rag",
        )
    except Exception as exc:
        logger.exception("Gradio CDSS demo failed")
        return (
            "",
            "",
            f"Error while running CDSS: {exc}",
            "",
            "",
        )

    reasoning = result.get("reasoning", "")
    patient_summary = result.get("patient_summary") or {}
    patient_name = _extract_patient_name(patient_summary)
    medical_data = _build_medical_data_text(patient_summary)
    recommendation = result.get("recommendation", "")
    citations = result.get("citations", [])
    citations_text = (
        "\n".join(f"- {_format_citation(c)}" for c in citations)
        if citations
        else "No citations returned."
    )

    meta = result.get("meta") or {}
    retrieval = meta.get("retrieval")
    if retrieval and retrieval.get("disease_tags"):
        medical_data = (
            f"Retrieval tags: {retrieval.get('disease_tags')}\n"
            + (medical_data or "")
        ).strip()

    return patient_name, medical_data, reasoning, recommendation, citations_text


with gr.Blocks(title="CDSS Demo") as demo:
    gr.Markdown("# CDSS Demo")
    gr.Markdown(
        "Multi-disease clinical decision support (CKD, hypertension, ACHD). "
        "Enter an optional patient ID and a clinical query. "
        "Choose **RAG** for indexed guideline retrieval, or **LLM synthesis** to skip retrieval."
    )

    evidence_mode_input = gr.Radio(
        choices=[
            ("RAG (guideline chunks)", "rag"),
            ("LLM synthesis (no retrieval)", "llm_synthesis"),
        ],
        value="rag",
        label="Evidence mode",
    )

    with gr.Row():
        patient_id_input = gr.Textbox(
            label="Patient ID (optional)",
            placeholder="e.g., 123456",
        )
        query_input = gr.Textbox(
            label="Clinical Query",
            placeholder="e.g., Blood pressure management for this patient with CKD and hypertension",
            lines=3,
        )

    submit_button = gr.Button("Run CDSS")

    patient_name_output = gr.Textbox(label="Patient Name", lines=1, interactive=False)
    medical_data_output = gr.Textbox(label="Medical Data", lines=8, interactive=False)
    reasoning_output = gr.Textbox(label="Reasoning", lines=8, interactive=False)
    recommendation_output = gr.Textbox(label="Recommendations", lines=6, interactive=False)
    citations_output = gr.Textbox(label="Citations", lines=8, interactive=False)

    submit_button.click(
        fn=run_cdss_demo,
        inputs=[patient_id_input, query_input, evidence_mode_input],
        outputs=[
            patient_name_output,
            medical_data_output,
            reasoning_output,
            recommendation_output,
            citations_output,
        ],
    )

    query_input.submit(
        fn=run_cdss_demo,
        inputs=[patient_id_input, query_input, evidence_mode_input],
        outputs=[
            patient_name_output,
            medical_data_output,
            reasoning_output,
            recommendation_output,
            citations_output,
        ],
    )


if __name__ == "__main__":
    demo.launch()
