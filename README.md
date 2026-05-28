# Clinical Decision Support System (CDSS)

Multi-disease clinical decision support for **CKD**, **hypertension**, **ACHD/heart failure**, **diabetes**, **stroke**, and **dyslipidemia**, combining patient data from Databricks with guideline retrieval from Qdrant and LLM-generated recommendations with citations.

**Repository:** [Medvise/Clinical-Decision-Support](https://github.com/Medvise/Clinical-Decision-Support)

## Overview

The system answers clinician questions in two modes:

| Route | When | Behavior |
|-------|------|----------|
| **Patient-specific** | `patient_id` provided and query is about this patient | Loads labs, meds, and stage from Databricks → builds patient summary → tags retrieval → fetches filtered guideline chunks from Qdrant → LLM answer with citations |
| **General guideline** | No `patient_id` or query is definitional/overview | Retrieves guideline chunks from Qdrant (disease inferred from query) → LLM answer |
| **Clarification** | Patient-specific phrasing but no `patient_id` | Returns a safe prompt to supply patient context |

Evidence modes:

- **`rag`** (default) — retrieve guideline chunks and ground the LLM on them.
- **`llm_synthesis`** — skip retrieval; model answers from internal knowledge only.

Supported disease tags: `CKD`, `hypertension`, `ACHD`, `diabetes`, `stroke`, `dyslipidemia`.

## Architecture

```
                    ┌─────────────────┐
  Clinician query   │   FastAPI /     │
  + patient_id  ──► │   Gradio demo   │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  LangGraph      │
                    │  query_graph    │
                    └────────┬────────┘
           ┌─────────────────┼─────────────────┐
           ▼                 ▼                 ▼
   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
   │ Databricks   │  │ Qdrant       │  │ OpenAI LLM   │
   │ (patient     │  │ (guideline   │  │ (recommend-  │
   │  gold layer) │  │  chunks)     │  │  ation +     │
   └──────────────┘  └──────────────┘  │  citations)  │
                                       └──────────────┘
```

**Ingestion (offline):** guideline PDFs → parsers (`ingestion/parser_ref/`) → embeddings → Qdrant collection.

## Project structure

```
api/                 # FastAPI app, request/response schemas
data/                # Databricks client, patient_fetcher
data/queries/        # Databricks SQL (patient gold)
ingestion/           # PDF ingestion script and guideline parsers
llm/                 # LLM client, retrieval tagger, citations, prompt_builder facade
llm/prompts/         # Prompt context, schema, assemble helpers
pipeline/            # graph_state, routing, graph_nodes, query_graph, logging
retrieval/           # tags_vocab, query_decompose, qdrant_*, search, rerank
rules/               # CKD staging, patient summary, disease inference
gradio_demo.py       # Interactive UI
config.py            # Environment-driven settings
```

## Prerequisites

- Python 3.11+ (3.14 used in local `cdss_env`)
- Access to:
  - **Databricks** SQL warehouse (patient gold tables)
  - **Qdrant** with an ingested guideline collection
  - **OpenAI** API (or compatible endpoint for chat)
  - **Embeddings service** (for ingestion and query embedding; see `EMBEDDINGS_URL`)

## Setup

### 1. Clone and virtual environment

```bash
git clone https://github.com/Medvise/Clinical-Decision-Support.git
cd Clinical-Decision-Support

python -m venv cdss_env
source cdss_env/bin/activate   # Windows: cdss_env\Scripts\activate
```

### 2. Install dependencies

There is no `requirements.txt` in the repo yet. Install from imports:

```bash
pip install \
  python-dotenv fastapi uvicorn pydantic pytest httpx gradio \
  langgraph openai qdrant-client requests pypdf cryptography \
  databricks-sql-connector

# Optional: cross-encoder reranking (when RERANK_ENABLED=true)
pip install sentence-transformers
```

### 3. Environment variables

Copy the example file and fill in your values:

```bash
cp .env.example .env
```

See [`.env.example`](.env.example) for all supported variables.

### 4. Guideline PDFs (local only)

PDFs are **not** stored in git. Place guideline files under `ingestion/` (filenames should include routing hints such as `KDIGO`, `NICE`, `HTN`, `HYPERTENSION`, `ACHD`, `DIABETES`, `STROKE`, `CHOLESTEROL`, or `LIPID`), then ingest:

```bash
python -m ingestion.script \
  ingestion/CKD.pdf \
  ingestion/Hypertension.pdf \
  ingestion/HeartFailure.pdf \
  ingestion/Diabetes.pdf \
  ingestion/Stroke.pdf
```

Or:

```bash
export INGEST_PDF_PATHS="ingestion/CKD.pdf,ingestion/Hypertension.pdf,ingestion/HeartFailure.pdf,ingestion/Diabetes.pdf,ingestion/Stroke.pdf"
python -m ingestion.script
```

## Running the application

From the project root (with venv active):

**API server:**

```bash
uvicorn api.main:app --reload --port 8000
```

Check the API is up (one line):

```bash
curl -sf http://localhost:8000/health
```

Or run the health test: `pytest tests/test_api_health.py -q`

- Health: `GET http://localhost:8000/health`
- Query: `POST http://localhost:8000/v1/cdss/query`

**Gradio demo:**

```bash
python gradio_demo.py
```

## API usage

### `POST /v1/cdss/query`

**Request body:**

```json
{
  "patient_id": "12345678",
  "query": "Should we start an SGLT2 inhibitor for this patient?",
  "evidence_mode": "rag"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `patient_id` | string | Databricks `uniqueempi`; empty for general-guideline questions |
| `query` | string | Clinical question (1–4000 chars) |
| `evidence_mode` | `"rag"` \| `"llm_synthesis"` | Whether to retrieve guideline chunks |

**Example (patient-specific):**

```bash
curl -s -X POST http://localhost:8000/v1/cdss/query \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "YOUR_UNIQUEEMPI",
    "query": "What BP target and agents are appropriate for this patient with CKD G3a?",
    "evidence_mode": "rag"
  }' | jq .
```

**Example (general guideline):**

```bash
curl -s -X POST http://localhost:8000/v1/cdss/query \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "",
    "query": "What does KDIGO recommend for ACEi in CKD?",
    "evidence_mode": "rag"
  }' | jq .
```

**Response** includes `recommendation`, `reasoning`, `citations`, optional `patient_summary`, `confidence`, and `meta` (route, latency, retrieval tags).

Legacy endpoint `POST /cdss/query` is deprecated; use `/v1/cdss/query`.

## Configuration highlights

| Variable | Purpose |
|----------|---------|
| `DATABRICKS_HOST`, `DATABRICKS_HTTP_PATH`, `DATABRICKS_TOKEN` | Patient data warehouse |
| `QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_COLLECTION` | Guideline vector store |
| `OPENAI_API_KEY`, `LLM_MODEL` | Chat completions |
| `EMBEDDINGS_URL` | Embedding service for retrieval and ingestion |
| `TOP_K_CHUNKS` | Max chunks passed to the LLM (default `15`) |
| `RERANK_ENABLED`, `RERANK_MODEL`, `RERANK_TOP_K` | Optional cross-encoder reranking |
| `LOG_RETRIEVED_CHUNKS`, `LOG_LLM_PROMPTS` | Debug logging |

## Internal documentation

Detailed design notes (`PIPELINE_AND_CHUNKING.md`, `KDIGO_CHUNKING_AND_QDRANT.md`, etc.) are kept locally and are not versioned in this repository. Ask the team for copies if you need ingestion or chunking specifics.

## License

Internal Medvise project. All rights reserved unless otherwise specified by the organization.
