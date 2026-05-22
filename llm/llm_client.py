import json
import logging
import re
from openai import OpenAI
from config import OPENAI_API_KEY, LLM_MODEL

_openai = OpenAI(api_key=OPENAI_API_KEY)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a clinical decision support assistant for multiple disease and medical conditions "
    "(e.g. CKD, hypertension, heart failure, diabetes, etc.). "
    "Always respond with valid JSON only — no markdown, no prose outside JSON."
)

SYSTEM_PROMPT_SYNTHESIS = (
    "You are a clinical decision support assistant for cardiovascular and kidney care. "
    "No indexed guideline text is supplied in this mode: use general and latest medical knowledge and "
    "attribute statements to the appropriate major guideline source when possible "
    "(e.g. KDIGO, ADA, ACC/AHA, ESC). Do not fabricate verbatim guideline quotes. "
    "Always respond with valid JSON only — no markdown, no prose outside JSON."
)


def _extract_json_object(text: str) -> str:
    """Return the outermost JSON object substring from model output."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
        if fence_match:
            cleaned = fence_match.group(1).strip()
        else:
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE).strip()

    try:
        json.loads(cleaned)
        return cleaned
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        return cleaned[start : end + 1]
    return cleaned


def _parse_llm_json(raw: str) -> dict:
    payload = _extract_json_object(raw)
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        logger.error(
            "LLM JSON parse failed at line %s col %s (chars=%s)",
            exc.lineno,
            exc.colno,
            len(payload),
        )
        raise ValueError(
            "Model returned invalid JSON. Try again or shorten the clinical question."
        ) from exc


def call_llm(prompt: str, *, synthesis: bool = False) -> dict:
    """
    Send the assembled prompt to the configured chat model.
    When synthesis=True, use a broader system prompt (no retrieved guideline text).
    Returns parsed JSON response dict.
    """
    logger.info("LLM request started (model=%s synthesis=%s)", LLM_MODEL, synthesis)
    system = SYSTEM_PROMPT_SYNTHESIS if synthesis else SYSTEM_PROMPT
    response = _openai.chat.completions.create(
        model    = LLM_MODEL,
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt}
        ],
        temperature = 0.1,   # low temp for consistent clinical output
        max_tokens  = 2500,
        response_format = {"type": "json_object"},
    )

    raw = response.choices[0].message.content.strip()
    logger.info("LLM response received (chars=%s)", len(raw))

    parsed = _parse_llm_json(raw)
    logger.info("LLM response JSON parsed")
    return parsed