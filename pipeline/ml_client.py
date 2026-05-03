"""
HTTP client for ML service calls.

All ML service communication goes through this module.
Consistent timeout, retry, and error logging for every call.
"""
from __future__ import annotations

import httpx

from config import settings
from core.exceptions import MLFaceCheckFailedError, MLServiceUnreachableError
from core.logging import get_logger

log = get_logger(__name__)

_BASE = settings.ML_SERVICE_URL
_TIMEOUT = 30.0


async def face_check(candidate_id: str, image_base64: str) -> dict:
    """
    POST /face-check to ML service.
    Returns {is_unique: bool, matched_candidate_id: str|null}.
    Raises MLServiceUnreachableError or MLFaceCheckFailedError.
    """
    url = f"{_BASE}/face-check"
    payload = {"candidate_id": candidate_id, "image": image_base64}
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.post(url, json=payload)
    except httpx.TimeoutException as exc:
        log.error("ML /face-check timed out", candidate_id=candidate_id)
        raise MLServiceUnreachableError(url) from exc
    except httpx.RequestError as exc:
        log.error("ML /face-check request error", candidate_id=candidate_id, exc_info=True)
        raise MLServiceUnreachableError(url) from exc

    if resp.status_code >= 500:
        log.error("ML /face-check 5xx", status_code=resp.status_code)
        raise MLServiceUnreachableError(url)

    try:
        data = resp.json()
    except Exception:
        raise MLFaceCheckFailedError("Invalid JSON response")

    log.info("ML face check result", candidate_id=candidate_id, is_unique=data.get("is_unique"))
    return data


async def get_pipeline_status(candidate_id: str) -> dict:
    """GET /status/{candidate_id} — returns {"status": "PROCESSING"|"DONE"|"ERROR"}"""
    url = f"{_BASE}/status/{candidate_id}"
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(url)
    except (httpx.TimeoutException, httpx.RequestError) as exc:
        log.warning("ML /status request failed", candidate_id=candidate_id, exc_info=True)
        raise MLServiceUnreachableError(url) from exc

    if resp.status_code == 404:
        return {"status": "NOT_FOUND"}

    try:
        return resp.json()
    except Exception:
        return {"status": "ERROR"}


async def get_verdict(candidate_id: str) -> dict:
    """GET /verdict/{candidate_id} — returns full verdict payload."""
    url = f"{_BASE}/verdict/{candidate_id}"
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(url)
    except (httpx.TimeoutException, httpx.RequestError) as exc:
        log.error("ML /verdict request failed", candidate_id=candidate_id, exc_info=True)
        raise MLServiceUnreachableError(url) from exc

    if resp.status_code == 404:
        log.warning("ML verdict not found", candidate_id=candidate_id)
        return {}

    try:
        return resp.json()
    except Exception:
        return {}


async def trigger_pipeline(candidate_id: str) -> bool:
    """POST /admin/trigger/{candidate_id} — re-trigger ML pipeline (admin only)."""
    url = f"{_BASE}/admin/trigger/{candidate_id}"
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.post(url)
        return resp.status_code < 400
    except Exception:
        log.error("ML admin trigger failed", candidate_id=candidate_id, exc_info=True)
        return False


async def health_check() -> bool:
    """Return True if ML service responds to GET /health."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{_BASE}/health")
        return resp.status_code < 400
    except Exception:
        return False
