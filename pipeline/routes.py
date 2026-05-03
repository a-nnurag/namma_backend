"""
Application status and result routes:
  GET /application/status/{id}         — poll status (served from Redis cache)
  GET /application/status-stream/{id}  — SSE stream; server pushes when ML finishes
  GET /application/result/{id}         — full verdict (only when DONE)
"""
from __future__ import annotations

import json
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from auth.middleware import get_registered_user
from cache.redis_client import get_ml_status, subscribe_ml_result
from core.logging import get_logger
from db.adapter import DatabaseAdapter
from db.crud import get_application_by_id
from db.session import get_db

log = get_logger(__name__)
router = APIRouter(prefix="/application", tags=["application"])

_TERMINAL_STATUSES = {"READY", "SKILL_TRAINING", "MANUAL_VERIFICATION", "UNSKILLED", "FRAUD"}


@router.get("/status/{application_id}", status_code=200)
async def get_application_status(
    application_id: str,
    payload: Annotated[dict, Depends(get_registered_user)],
    db: Annotated[DatabaseAdapter, Depends(get_db)],
) -> dict:
    user_id = payload["sub"]

    # Redis cache first
    cached = await get_ml_status(application_id)
    if cached:
        return {"application_id": application_id, "source": "cache", **cached}

    # DB fallback
    app = await get_application_by_id(db, application_id)
    if app is None or str(app.user_id) != user_id:
        raise HTTPException(status_code=404, detail="Application not found")

    return {
        "application_id": application_id,
        "status": app.status,
        "source": "db",
        "cooldown_until": app.cooldown_until.isoformat() if app.cooldown_until else None,
    }


@router.get("/application/status-stream/{application_id}")
async def stream_application_status(
    application_id: str,
    payload: Annotated[dict, Depends(get_registered_user)],
    db: Annotated[DatabaseAdapter, Depends(get_db)],
) -> StreamingResponse:
    """
    SSE endpoint.  Sends the current status immediately, then blocks until the
    ML service publishes a terminal status via Redis pub/sub.  A heartbeat
    comment (': heartbeat') is emitted every 20 s to keep proxies alive.

    The client should close the connection once it receives a terminal status.
    """
    user_id = payload["sub"]

    app = await get_application_by_id(db, application_id)
    if app is None or str(app.user_id) != user_id:
        raise HTTPException(status_code=404, detail="Application not found")

    async def event_stream():
        # Always emit current status first so the client has something immediately
        current = {"status": app.status, "application_id": application_id}
        yield f"data: {json.dumps(current)}\n\n"

        if app.status in _TERMINAL_STATUSES:
            return  # Already done — one event, then close

        log.info(
            "SSE stream opened for ML result",
            application_id=application_id,
            user_id=user_id,
        )

        async for event in subscribe_ml_result(application_id):
            if event.get("_heartbeat"):
                yield ": heartbeat\n\n"
            else:
                yield f"data: {json.dumps(event)}\n\n"
                log.info(
                    "SSE stream delivered ML result",
                    application_id=application_id,
                    status=event.get("status"),
                )
                return

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable Nginx buffering
            "Connection": "keep-alive",
        },
    )


@router.get("/result/{application_id}", status_code=200)
async def get_application_result(
    application_id: str,
    payload: Annotated[dict, Depends(get_registered_user)],
    db: Annotated[DatabaseAdapter, Depends(get_db)],
) -> dict:
    user_id = payload["sub"]
    app = await get_application_by_id(db, application_id)
    if app is None or str(app.user_id) != user_id:
        raise HTTPException(status_code=404, detail="Application not found")

    if app.status in ("DOCUMENTS_PENDING", "INTERVIEW_PENDING", "ML_PROCESSING", "INTERVIEW_DONE"):
        raise HTTPException(
            status_code=202,
            detail=f"Result not ready yet. Current status: {app.status}",
        )

    verdict = app.verdict
    if verdict is None:
        raise HTTPException(status_code=404, detail="Verdict not found")

    return {
        "application_id": application_id,
        "status": app.status,
        "verdict": verdict.verdict,
        "composite_score": verdict.composite_score,
        "fraud_score": verdict.fraud_score,
        "domain_score": verdict.domain_score,
        "communication_score": verdict.communication_score,
        "skill_confidence": verdict.skill_confidence,
        "cooldown_until": app.cooldown_until.isoformat() if app.cooldown_until else None,
    }
