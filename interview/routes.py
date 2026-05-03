"""
Interview routes:
  POST /interview/start                  — create session, get session_id
  WS   /interview/ws/{session_id}        — chatbot audio WebSocket
  WS   /interview/video-ws/{session_id}  — video frame stream WebSocket
  POST /interview/end/{session_id}       — end interview, trigger Kafka send
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect

from auth.middleware import get_registered_user
from core.exceptions import SessionAlreadyEndedError, SessionNotFoundError
from core.logging import get_logger
from db.adapter import DatabaseAdapter
from db.crud import (
    complete_interview_session,
    create_interview_session,
    get_application_by_id,
    get_documents_for_application,
    get_session_by_id,
    set_application_cooldown,
    update_application_status,
)
from db.session import get_db
from interview.chatbot import (
    create_chatbot_session,
    get_chatbot_session,
    remove_chatbot_session,
)
from interview.media_buffer import get_or_create_buffer
from pipeline.kafka_producer import send_interview_media_to_kafka

log = get_logger(__name__)
router = APIRouter(prefix="/interview", tags=["interview"])


_SUPPORTED_LANGUAGES = {"kn", "hi", "en", "te", "ta"}


@router.post("/start", status_code=201)
async def start_interview(
    payload: Annotated[dict, Depends(get_registered_user)],
    db: Annotated[DatabaseAdapter, Depends(get_db)],
    application_id: str = None,
    language: str = "kn",
) -> dict:
    user_id = payload["sub"]

    if application_id is None:
        raise HTTPException(status_code=400, detail="application_id is required")

    if language not in _SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language '{language}'. Supported: {sorted(_SUPPORTED_LANGUAGES)}",
        )

    app = await get_application_by_id(db, application_id)
    if app is None or str(app.user_id) != user_id:
        raise HTTPException(status_code=404, detail="Application not found")

    if app.status not in ("DOCUMENTS_PENDING", "INTERVIEW_PENDING"):
        raise HTTPException(
            status_code=409,
            detail=f"Cannot start interview from status: {app.status}",
        )

    # Get skill name for system prompt
    from db.crud import get_skill_by_id
    skill = await get_skill_by_id(db, app.skill_id)
    skill_name = skill.name if skill else "labour"

    # Create DB session record (stores language for Kafka meta later)
    session = await create_interview_session(db, app.id, language=language)

    # Create in-memory chatbot session with language
    create_chatbot_session(str(session.id), skill_name, str(app.id), language=language)

    await update_application_status(db, app, "INTERVIEW_PENDING")

    log.info(
        "Interview session started",
        user_id=user_id,
        application_id=str(app.id),
        session_id=str(session.id),
        skill=skill_name,
        language=language,
    )

    return {
        "session_id": str(session.id),
        "skill_name": skill_name,
        "language": language,
        "status": "ACTIVE",
    }


@router.websocket("/ws/{session_id}")
async def interview_audio_ws(websocket: WebSocket, session_id: str) -> None:
    chatbot = get_chatbot_session(session_id)
    if chatbot is None:
        await websocket.close(code=4004, reason="Session not found")
        return

    await chatbot.handle_audio_websocket(websocket)


@router.websocket("/video-ws/{session_id}")
async def interview_video_ws(websocket: WebSocket, session_id: str) -> None:
    chatbot = get_chatbot_session(session_id)
    if chatbot is None:
        await websocket.close(code=4004, reason="Session not found")
        return

    await chatbot.handle_video_websocket(websocket)


@router.post("/end/{session_id}", status_code=200)
async def end_interview(
    session_id: str,
    payload: Annotated[dict, Depends(get_registered_user)],
    db: Annotated[DatabaseAdapter, Depends(get_db)],
) -> dict:
    user_id = payload["sub"]

    db_session = await get_session_by_id(db, session_id)
    if db_session is None:
        raise HTTPException(status_code=404, detail="Interview session not found")

    app = await get_application_by_id(db, db_session.application_id)
    if app is None or str(app.user_id) != user_id:
        raise HTTPException(status_code=403, detail="Forbidden")

    if db_session.status != "ACTIVE":
        raise HTTPException(
            status_code=409,
            detail=f"Session already {db_session.status}",
        )

    chatbot = get_chatbot_session(session_id)
    transcript = chatbot.get_transcript() if chatbot else []
    language   = chatbot.language if chatbot else db_session.language

    # Save transcript + complete session
    db_session = await complete_interview_session(db, db_session, transcript)

    # Set 60-day cooldown immediately
    app = await set_application_cooldown(db, app)

    # Check documents to set has_degree / has_workexp accurately
    from db.crud import get_documents_for_application
    docs = await get_documents_for_application(db, str(app.id))
    doc_types = {d.doc_type for d in docs}
    has_degree  = "DEGREE" in doc_types
    has_workexp = "WORKEXP_VIDEO" in doc_types

    # Send media to Kafka
    buf = get_or_create_buffer(session_id)
    try:
        await send_interview_media_to_kafka(
            candidate_id=user_id,
            session_id=session_id,
            application_id=str(app.id),
            skill_name=app.skill.name if app.skill else "unknown",
            audio_buffer=buf.audio,
            video_buffer=buf.video,
            has_degree=has_degree,
            has_workexp=has_workexp,
            language=language,
        )
        await update_application_status(db, app, "ML_PROCESSING")
        log.info(
            "Interview ended — media sent to Kafka",
            user_id=user_id,
            session_id=session_id,
            application_id=str(app.id),
        )
    except Exception as exc:
        log.error(
            "Kafka send failed after interview end",
            session_id=session_id,
            exc_info=True,
        )
        # Don't fail the HTTP response — Kafka failure is retried separately

    # Cleanup in-memory resources
    remove_chatbot_session(session_id)

    return {
        "status": "completed",
        "session_id": session_id,
        "cooldown_until": app.cooldown_until.isoformat() if app.cooldown_until else None,
        "question_count": len([m for m in transcript if m.get("role") == "assistant"]),
    }
