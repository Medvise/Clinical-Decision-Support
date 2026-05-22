def _format_prompt_date(date_value) -> str:
    if not date_value:
        return "Unknown"

    normalized = str(date_value).strip()
    if len(normalized) == 8 and normalized.isdigit():
        return f"{normalized[:4]}/{normalized[4:6]}/{normalized[6:]}"
    return normalized


def _format_observation_line(obs: dict) -> str:
    value = obs.get("value")
    unit = obs.get("unit")
    if unit:
        value_text = f"{value} {unit}"
    else:
        value_text = str(value)
    if obs.get("date"):
        return f"- {obs['label']}: {value_text} (date: {_format_prompt_date(obs['date'])})"
    return f"- {obs['label']}: {value_text}"


def _patient_context_block(summary: dict) -> str:
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
        lines.extend(_format_observation_line(obs) for obs in observations)

    if meds:
        lines.append(f"- Medications: {meds}")
    if icd_codes:
        lines.append(f"- ICD codes: {icd_codes}")
    if flags:
        lines.append(f"- Clinical flags: {flags}")

    return "\n".join(lines)


# LLM answer JSON — patient_summary is attached server-side, not by the model.
_LLM_ANSWER_JSON_SCHEMA = """{
  "recommendation": "...",
  "reasoning": "...",
  "citations": [],
  "confidence": "High | Medium | Low",
  "llm_summary": "optional one-sentence recap of how you interpreted the patient"
}"""


def assemble_patient_synthesis_prompt(summary: dict, query: str) -> str:
    """
    Patient-specific prompt without retrieved guideline chunks.
    Citations are attributive (major guideline family + summary), not evidence-indexed.
    """
    patient_block = _patient_context_block(summary)

    return f"""
## PERSONA ##
You are a clinical decision support system. No indexed guideline passages are provided in this request.

## OBJECTIVE ##
Answer the CLINICAL QUESTION using the patient context below and sound general medical practice.
Name the most relevant major guideline source when you rely on it (e.g. KDIGO, ACC/AHA hypertension, ACHD guidelines, ADA).
If you are uncertain about wording, strength of recommendation, or edition, say so explicitly.
Do not default to treatment advice when the question asks for current-state facts (medications, stage, labs).

## MODE DISCLAIMER ##
State briefly in your reasoning that this answer is not grounded in locally retrieved guideline text.

## PATIENT CONTEXT ##
{patient_block}

## CLINICAL QUESTION ##
{query}

## ANSWER STYLE RULES ##
- Keep the response human-friendly and clinically clear.
- If the user asks about current medications, list them from patient context.
- If data are missing, say so. Provide treatment advice only when the question asks for management or next steps.

CITATION RULES (no evidence blocks in this mode):
- citations must be plain strings such as: "KDIGO CKD guideline (general knowledge): <short paraphrase or topic>".
- Do not use [1] style evidence indices. Do not claim verbatim quotes from guidelines.

Respond ONLY with this JSON — no prose outside the JSON:
{_LLM_ANSWER_JSON_SCHEMA}

Do NOT include patient_summary in your JSON; clinical facts are handled server-side.
"""


def assemble_prompt(
    summary: dict,
    chunks:  list[str],
    query:   str
) -> str:
    """
    Combine patient summary, retrieved guideline chunks,
    and the clinical question into a single LLM prompt.
    """
    patient_block = _patient_context_block(summary)
    evidence = "\n\n".join(
        f"### Evidence block {i + 1} ###\n{chunk}"
        for i, chunk in enumerate(chunks)
    )

    return f"""
            ## PERSONA ##    
            You are a clinical decision support system specialising in various medical conditions.

            ## OBJECTIVE ##
            Your primary responsibility is to answer the CLINICAL QUESTION directly and accurately using patient context and guideline evidence (in case guideline evidence is not available, use your knowledge and expertise to answer the question with proper citations and references).
            Do not default to treatment advice when the question asks for current-state facts (for example: current medications, current stage, latest lab values).

            ## GUIDELINE EVIDENCE ##
            1. Do NOT invent medical facts.
            2. Use ONLY the guideline evidence provided below. 
            3. Always return valid JSON matching the schema specified.
            4. If guideline evidence is not available, use your knowledge and expertise to answer the question with proper citations and references.

            ## PATIENT CONTEXT ##
            {patient_block}

            ## GUIDELINE EVIDENCE ##
            {evidence}

            ## CLINICAL QUESTION ## :
            {query}

            ## ANSWER STYLE RULES ## :
            - Keep the response human-friendly and conversational.
            - If multiple diseases are present, provide recommendations for each disease separately with proper citations and references.
            - If the user asks about current medications, recommendation must directly list the patient's current medications from patient context.
            - If requested data is missing or uncertain, say so explicitly.
            - Provide treatment advice only when the user asks for recommendation/management/next step.

            CITATION RULES:
            - Every citation MUST start with the evidence index matching the block number (e.g., [1] for "### Evidence block 1 ###").
            - Each citation MUST include a contiguous verbatim excerpt from that block's guideline text (the prose between BEGIN_GUIDELINE_TEXT and END_GUIDELINE_TEXT). Use at least 120 characters when the source text is longer; if the source is shorter, quote it in full.
            - Also include the SOURCE_LINE from that same block verbatim in parentheses after the index, e.g. [1] (SOURCE_LINE: …) "…excerpt…".
            - Do NOT output generic placeholders like "guideline source 1".
            - If evidence is insufficient, set confidence to "Low" and state that explicitly.

            ## Reasoning style ## :
            1. Understand the clinical question followed by the patient context and guideline evidence.
            2. Analyse the lab values and determine stage of disease if applicable.
            3. Provide a step-by-step reasoning referencing patient values and guideline evidence.
            4. If the user asks about current medications, recommendation must directly list the patient's current medications from patient context.
            5. If requested data is missing or uncertain, say so explicitly.
            6. Provide treatment advice only when the user asks for recommendation/management/next step.

            ## RECOMMENDATION STYLE RULES ## :
            - Provide recommendations for each disease separately with proper citations and references.
            - Answer should be in detail, covering all the information provided in the patient context and guideline evidence.
            - This section will be shown in the UI to the user so it should be in as much detail as possible, for example, explain the recommendation in detail, include the rationale, and include the citations and references.

            Respond ONLY with this JSON — no prose outside the JSON:
            {_LLM_ANSWER_JSON_SCHEMA}

            For citations use [n] (SOURCE_LINE: …) "verbatim excerpt" format when evidence blocks are present.
            Do NOT include patient_summary in your JSON.
"""


def assemble_general_synthesis_prompt(query: str) -> str:
    """
    Guideline-style question without retrieved chunks; attributive citations only.
    """
    return f"""
## PERSONA ##
You are a clinical decision support assistant. No indexed guideline passages are provided.

## OBJECTIVE ##
Answer the CLINICAL QUESTION using sound general medical knowledge.
Name the most relevant major guideline source when applicable (e.g. KDIGO, ACC/AHA/HFSA, ESC).
If uncertain about edition or exact wording, say so. Do not fabricate verbatim guideline quotes.

## MODE DISCLAIMER ##
State briefly in your reasoning that this answer is not grounded in locally retrieved guideline text.

## CLINICAL QUESTION ##
{query}

## ANSWER STYLE RULES ##
- Keep the response concise and clinically clear.
- If the question is vignette-style, answer the clinical principle asked; do not invent patient-specific data not stated in the question.

CITATION RULES (no evidence blocks in this mode):
- citations must be plain strings such as: "ACC/AHA/HFSA heart failure guideline (general knowledge): <topic>".
- Do not use [1] style evidence indices.

Respond ONLY with this JSON — no prose outside the JSON:
{_LLM_ANSWER_JSON_SCHEMA}
"""


def assemble_general_prompt(chunks: list[str], query: str) -> str:
    """
    Build a guideline-only prompt for non-patient-specific questions.
    """
    evidence = "\n\n".join(
        f"### Evidence block {i + 1} ###\n{chunk}"
        for i, chunk in enumerate(chunks)
    )

    return f"""
                ## PERSONA ##
                You are a clinical decision support assistant specialising in multiple disease guidelines.

                ## OBJECTIVE ##
                Answer the question using guideline evidence only.
                Do not claim patient-specific advice because no patient context is being used.

                ## GUIDELINE EVIDENCE RULES ##
                1. Use ONLY the evidence chunks below.
                2. Do NOT invent medical facts.
                3. If evidence is incomplete, say so and set confidence to "Low" or "Medium".
                4. Always return valid JSON matching the schema specified.

                ## GUIDELINE EVIDENCE ##
                {evidence}

                ## CLINICAL QUESTION ##
                {query}

                ## ANSWER/RECOMMENDATION STYLE RULES ## :
                - Keep the response human-friendly and conversational.
                - If multiple diseases are present, provide recommendations for each disease separately with proper citations and references.
                - If the user asks about current medications, recommendation must directly list the patient's current medications from patient context.
                - If requested data is missing or uncertain, say so explicitly.
                - Provide treatment advice only when the user asks for recommendation/management/next step.
                - If question asks "what does guideline say", summarize recommendations faithfully.
                - Include uncertainty limits where evidence is sparse.
                - Answer should be in detail, covering all the information provided in the guideline evidence.
                - This section will be shown in the UI to the user so it should be in as much detail as possible, for example, explain the recommendation in detail, include the rationale, and include the citations and references.

                CITATION RULES:
                - Every citation MUST start with the evidence index (e.g., [1] for "### Evidence block 1 ###").
                - Include the SOURCE_LINE from that block verbatim in parentheses, then a contiguous verbatim excerpt (≥120 chars when possible) from between BEGIN_GUIDELINE_TEXT and END_GUIDELINE_TEXT.
                - Do NOT output generic placeholders.

                Respond ONLY with this JSON — no prose outside the JSON:
                {_LLM_ANSWER_JSON_SCHEMA}

                For citations use [n] (SOURCE_LINE: …) "verbatim excerpt" format.
                Do NOT include patient_summary in your JSON.
"""