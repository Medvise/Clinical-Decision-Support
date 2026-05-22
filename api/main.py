# api/main.py
import logging
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from api.schema import CDSSRequest, CDSSResponse, ErrorResponse, SCHEMA_VERSION
from pipeline.orchestrator import run_cdss_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="CDSS API", version="1.0.0")


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


def _error_response(request: Request, status: int, code: str, message: str) -> JSONResponse:
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    body = ErrorResponse(
        schema_version=SCHEMA_VERSION,
        request_id=request_id,
        code=code,
        message=message,
    )
    return JSONResponse(status_code=status, content=body.model_dump(mode="json"))


@app.post("/v1/cdss/query", response_model=CDSSResponse)
def cdss_query_v1(request: CDSSRequest, http_request: Request):
    """Multi-disease CDSS endpoint (v1)."""
    request_id = getattr(http_request.state, "request_id", str(uuid.uuid4()))
    try:
        logger.info(
            "API request request_id=%s patient_id=%s query=%s",
            request_id,
            request.patient_id,
            request.query[:120],
        )
        result = run_cdss_pipeline(
            patient_id=request.patient_id,
            query=request.query,
            evidence_mode=request.evidence_mode,
            request_id=request_id,
        )
        logger.info("API request completed request_id=%s", request_id)
        return result
    except ValueError as e:
        logger.warning("API 404 request_id=%s: %s", request_id, e)
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception:
        logger.exception("API 500 request_id=%s", request_id)
        return _error_response(
            http_request,
            500,
            "internal_error",
            "An internal error occurred while processing the request.",
        )


@app.post("/cdss/query", response_model=CDSSResponse, deprecated=True)
def cdss_query_legacy(request: CDSSRequest, http_request: Request):
    """Deprecated alias for /v1/cdss/query."""
    return cdss_query_v1(request, http_request)


@app.get("/health")
def health():
    return {"status": "ok", "schema_version": SCHEMA_VERSION}
