import os
from dotenv import load_dotenv

load_dotenv()


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

# Databricks
DATABRICKS_HOST      = os.getenv("DATABRICKS_HOST")
DATABRICKS_HTTP_PATH = os.getenv("DATABRICKS_HTTP_PATH")
DATABRICKS_TOKEN     = os.getenv("DATABRICKS_TOKEN")

# Qdrant
QDRANT_URL     = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = "CDSS_V1_MultiDisease"

# OpenAI
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
EMBED_MODEL      = "text-embedding-3-small"
LLM_MODEL        = os.getenv("LLM_MODEL", "gpt-4o")
RETRIEVAL_TAG_MODEL = os.getenv("RETRIEVAL_TAG_MODEL")  # optional; defaults to LLM_MODEL
EMBEDDINGS_URL   = os.getenv("EMBEDDINGS_URL")
EMBEDDINGS_TIMEOUT_SECONDS = int(os.getenv("EMBEDDINGS_TIMEOUT_SECONDS", "300"))

# App
TOP_K_CHUNKS = int(os.getenv("TOP_K_CHUNKS", "15"))

# Retrieval reranker (optional)
RERANK_ENABLED = _as_bool(os.getenv("RERANK_ENABLED"), default=False)
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-large")
RERANK_DEVICE = os.getenv("RERANK_DEVICE", "cpu")
RERANK_BATCH_SIZE = int(os.getenv("RERANK_BATCH_SIZE", "16"))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "5"))

# Debug logging
LOG_RETRIEVED_CHUNKS = _as_bool(os.getenv("LOG_RETRIEVED_CHUNKS"), default=True)
LOG_LLM_PROMPTS = _as_bool(os.getenv("LOG_LLM_PROMPTS"), default=True)
# 0 = log full prompt; set e.g. 12000 to truncate very long prompts
LOG_PROMPT_MAX_CHARS = int(os.getenv("LOG_PROMPT_MAX_CHARS", "0"))