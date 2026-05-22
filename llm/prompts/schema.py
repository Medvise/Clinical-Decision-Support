"""Shared JSON schema snippets for LLM answer prompts."""

LLM_ANSWER_JSON_SCHEMA = """{
  "recommendation": "...",
  "reasoning": "...",
  "citations": [],
  "confidence": "High | Medium | Low",
  "llm_summary": "optional one-sentence recap of how you interpreted the patient"
}"""
