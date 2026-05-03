"""
Degree image upload — synchronous ML service check.
"""
from __future__ import annotations

import httpx

from config import settings
from core.exceptions import MLServiceUnreachableError
from core.logging import get_logger

log = get_logger(__name__)


async def check_degree_with_ml(
    candidate_id: str,
    image_bytes: bytes,
    filename: str,
) -> dict:
    """
    POST image to ML service POST /degree/{candidate_id}.
    Returns the full response payload or raises on failure.
    """
    url = f"{settings.ML_SERVICE_URL}/degree/{candidate_id}"
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                url,
                files={"file": (filename, image_bytes, "image/jpeg")},
            )
    except httpx.TimeoutException as exc:
        log.error("ML service degree check timed out", candidate_id=candidate_id)
        raise MLServiceUnreachableError(url) from exc
    except httpx.RequestError as exc:
        log.error("ML service degree check request failed", candidate_id=candidate_id, exc_info=True)
        raise MLServiceUnreachableError(url) from exc

    if resp.status_code >= 500:
        log.error(
            "ML service returned 5xx for degree check",
            candidate_id=candidate_id,
            status_code=resp.status_code,
        )
        raise MLServiceUnreachableError(url)

    try:
        data = resp.json()
    except Exception:
        log.error("ML service degree response is not JSON", candidate_id=candidate_id)
        raise MLServiceUnreachableError(url)

    log.info(
        "Degree check completed",
        candidate_id=candidate_id,
        is_valid_doc=data.get("is_valid_doc"),
    )
    return data
