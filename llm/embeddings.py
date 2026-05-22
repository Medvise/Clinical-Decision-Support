"""HTTP embeddings client shared by retrieval and ingestion."""

from __future__ import annotations

import requests

from config import EMBEDDINGS_TIMEOUT_SECONDS, EMBEDDINGS_URL


def fetch_embeddings_http(texts: list[str]) -> list[list[float]]:
    """
    POST sentences to EMBEDDINGS_URL and return embedding vectors.
    Requires EMBEDDINGS_URL to be configured.
    """
    if not EMBEDDINGS_URL:
        raise RuntimeError("EMBEDDINGS_URL is not configured.")

    resp = requests.post(
        EMBEDDINGS_URL,
        json={"sentences": texts},
        timeout=(10, EMBEDDINGS_TIMEOUT_SECONDS),
    )
    try:
        resp.raise_for_status()
    except requests.HTTPError as exc:
        body = (resp.text or "").strip()[:500]
        raise RuntimeError(
            f"Embeddings request failed ({resp.status_code}) for {EMBEDDINGS_URL}. "
            f"Response body: {body or '<empty>'}"
        ) from exc

    embeddings = resp.json().get("embeddings", [])
    if not embeddings:
        raise RuntimeError("Embeddings response missing 'embeddings' values.")
    return embeddings
