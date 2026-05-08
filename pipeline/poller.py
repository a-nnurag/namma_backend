"""
Background poller — checks ML service every 30s for PROCESSING applications.

When ML returns status=DONE:
  1. Fetch full verdict via GET /verdict/{candidate_id}
  2. Store verdict in ml_verdicts table
  3. Update candidate_applications.status = verdict value
  4. If verdict = FRAUD → deactivate user
  5. Update Redis ml_status cache
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from cache.redis_client import set_ml_status
from config import settings
from core.logging import get_logger
from db.crud import (
    deactivate_user,
    get_last_session_for_application,
    get_processing_applications,
    get_user_by_id,
    update_application_status,
    upsert_ml_verdict,
)
from db.session import get_db_context
from pipeline.ml_client import get_pipeline_status, get_verdict

log = get_logger(__name__)

_POLL_INTERVAL = settings.ML_POLL_INTERVAL_SECONDS


async def _process_one(candidate_id: str, application_id: str, session_id: str | None) -> None:
    try:
        status_data = await get_pipeline_status(candidate_id)
    except Exception:
        log.warning("Poller: ML status check failed", candidate_id=candidate_id, exc_info=True)
        return

    ml_status = status_data.get("status", "PROCESSING")

    # Update Redis cache regardless
    await set_ml_status(application_id, {"status": ml_status})

    if ml_status != "DONE":
        log.debug("Poller: ML not done yet", candidate_id=candidate_id, status=ml_status)
        return

    # Fetch full verdict
    try:
        verdict_data = await get_verdict(candidate_id)
    except Exception:
        log.error("Poller: verdict fetch failed", candidate_id=candidate_id, exc_info=True)
        return

    if not verdict_data:
        log.warning("Poller: empty verdict returned", candidate_id=candidate_id)
        return

    verdict_value = verdict_data.get("verdict", "MANUAL_VERIFICATION")

    async with get_db_context() as db:
        from db.models import CandidateApplication
        import uuid

        # Upsert ML verdict record
        await upsert_ml_verdict(
            db,
            application_id=uuid.UUID(application_id),
            session_id=uuid.UUID(session_id) if session_id else None,
            verdict_data=verdict_data,
        )

        # Update application status
        from db.crud import get_application_by_id
        app = await get_application_by_id(db, uuid.UUID(application_id))
        if app:
            await update_application_status(db, app, verdict_value)

            # FRAUD → deactivate user
            if verdict_value == "FRAUD":
                user = await get_user_by_id(db, app.user_id)
                if user:
                    await deactivate_user(db, user)
                    log.warning(
                        "User deactivated due to FRAUD verdict",
                        candidate_id=candidate_id,
                        application_id=application_id,
                    )

    # Update Redis with final verdict
    await set_ml_status(application_id, {"status": "DONE", "verdict": verdict_value})

    log.info(
        "Poller: verdict stored",
        candidate_id=candidate_id,
        application_id=application_id,
        verdict=verdict_value,
    )


async def poll_once() -> None:
    """Single polling pass — called every POLL_INTERVAL seconds."""
    async with get_db_context() as db:
        processing = await get_processing_applications(db)

    if not processing:
        log.debug("Poller: no applications in ML_PROCESSING")
        return

    log.info("Poller: checking applications", count=len(processing))

    # Load last session IDs explicitly — async session cannot lazy-load relationships
    tasks = []
    for app in processing:
        last_session = await get_last_session_for_application(db, app.id)
        tasks.append(
            _process_one(
                str(app.user_id),
                str(app.id),
                str(last_session.id) if last_session else None,
            )
        )
    await asyncio.gather(*tasks, return_exceptions=True)


async def start_poller() -> None:
    """Infinite polling loop. Run as a background asyncio task via lifespan."""
    log.info("ML status poller started", interval_seconds=_POLL_INTERVAL)
    while True:
        try:
            await poll_once()
        except Exception:
            log.error("Poller iteration failed unexpectedly", exc_info=True)
        await asyncio.sleep(_POLL_INTERVAL)
