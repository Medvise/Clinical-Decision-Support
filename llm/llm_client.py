import json
import logging
import re

from openai import OpenAI

from config import LLM_MODEL, OPENAI_API_KEY

logger = logging.getLogger(__name__)

_client: OpenAI | None = None

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

RETRIEVAL_TAGGER_SYSTEM_PROMPT = (
    "You output JSON only for clinical guideline retrieval metadata. "
    "Use allowed tag vocabularies exactly."
)


def get_openai_client() -> OpenAI | None:
    """Return a shared OpenAI client, or None if OPENAI_API_KEY is unset."""
    global _client
    if not OPENAI_API_KEY:
        return None
    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


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


def parse_llm_json(raw: str) -> dict:
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


# Backward-compatible alias
_parse_llm_json = parse_llm_json


def call_llm_json(
    prompt: str,
    *,
    model: str | None = None,
    system: str = SYSTEM_PROMPT,
    temperature: float = 0.1,
    max_tokens: int = 2500,
) -> dict:
    """
    Chat completion with JSON response_format; returns parsed dict.
    Raises if OPENAI_API_KEY is missing or the model returns invalid JSON.
    """
    client = get_openai_client()
    if client is None:
        raise RuntimeError("OPENAI_API_KEY is not configured.")

    use_model = model or LLM_MODEL
    logger.info("LLM JSON request started (model=%s)", use_model)
    response = client.chat.completions.create(
        model=use_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content.strip()
    logger.info("LLM JSON response received (chars=%s)", len(raw))
    parsed = parse_llm_json(raw)
    logger.info("LLM JSON response parsed")
    return parsed


def call_llm(prompt: str, *, synthesis: bool = False) -> dict:
    """
    Send the assembled prompt to the configured chat model.
    When synthesis=True, use a broader system prompt (no retrieved guideline text).
    Returns parsed JSON response dict.
    """
    logger.info("LLM request started (model=%s synthesis=%s)", LLM_MODEL, synthesis)
    system = SYSTEM_PROMPT_SYNTHESIS if synthesis else SYSTEM_PROMPT
    return call_llm_json(
        prompt,
        model=LLM_MODEL,
        system=system,
        temperature=0.1,
        max_tokens=2500,
    )
