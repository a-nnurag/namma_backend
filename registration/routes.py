"""
Registration routes:
  POST /register/face     — face video upload, embedding + hash
  POST /register/aadhaar  — Aadhaar number, API verify, duplicate check
  GET  /register/status   — which step is complete
"""
from __future__ import annotations

import re
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field, field_validator

from auth.aadhaar_verifier import AadhaarVerifierBackend
from auth.jwt import create_and_cache_token
from auth.middleware import get_current_user
from config import settings
from core.exceptions import (
    AadhaarAPIFailedError,
    AadhaarAlreadyRegisteredError,
    AadhaarInvalidFormatError,
    AadhaarVerificationFailedError,
    FaceAlreadyRegisteredError,
    FaceEmbeddingFailedError,
    FaceNotDetectedError,
    FaceNotYetRegisteredError,
    FaceVideoInvalidFormatError,
    FaceVideoTooLargeError,
)
from core.logging import get_logger
from db.adapter import DatabaseAdapter
from db.crud import get_user_by_aadhaar, get_user_by_id, update_user_aadhaar
from db.session import get_db
from registration.face import register_face
from registration.face_hash_strategy import FaceHashStrategy

log = get_logger(__name__)
router = APIRouter(prefix="/register", tags=["registration"])

_AADHAAR_RE = re.compile(r"^\d{12}$")


def _validate_aadhaar_format(aadhaar: str) -> bool:
    if not _AADHAAR_RE.match(aadhaar):
        return False
    if len(set(aadhaar)) == 1:
        return False
    return True


# Dependencies injected via main.py state
def _get_hash_strategy(request) -> FaceHashStrategy:  # type: ignore[no-untyped-def]
    return request.app.state.face_hash_strategy


def _get_aadhaar_verifier(request) -> AadhaarVerifierBackend:  # type: ignore[no-untyped-def]
    return request.app.state.aadhaar_verifier


@router.post("/face", status_code=200)
async def register_face_route(
    payload: Annotated[dict, Depends(get_current_user)],
    db: Annotated[DatabaseAdapter, Depends(get_db)],
    video: UploadFile = File(..., description="Short 3-5 second face video (max 10MB)"),
) -> dict:
    from fastapi import Request

    user_id = payload["sub"]
    user = await get_user_by_id(db, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    if user.face_registered:
        log.info("Face already registered — skipping", user_id=user_id)
        return {"status": "already_registered"}

    video_bytes = await video.read()
    filename = video.filename or "face_video.mp4"

    from fastapi import Request
    import inspect

    # Resolve hash strategy from app state
    # (FastAPI doesn't inject Request automatically into non-route deps)
    # We accept a slight coupling here by importing app state singleton
    from main import app as _app  # type: ignore[import]
    hash_strategy: FaceHashStrategy = _app.state.face_hash_strategy

    try:
        result = await register_face(db, user, video_bytes, filename, hash_strategy)
    except FaceVideoTooLargeError as exc:
        raise HTTPException(status_code=413, detail=exc.message)
    except FaceVideoInvalidFormatError as exc:
        raise HTTPException(status_code=400, detail=exc.message)
    except FaceNotDetectedError as exc:
        raise HTTPException(status_code=422, detail=exc.message)
    except FaceAlreadyRegisteredError as exc:
        raise HTTPException(status_code=409, detail=exc.message)
    except FaceEmbeddingFailedError as exc:
        log.error("Face embedding failed", user_id=user_id, error_code=exc.error_code)
        raise HTTPException(status_code=500, detail="Face processing failed. Try again.")

    log.info("Face registration successful", user_id=user_id)
    return {"status": "face_registered", "embedding_dim": result["embedding_dim"]}


class AadhaarRequest(BaseModel):
    aadhaar_number: str = Field(..., min_length=12, max_length=12)

    @field_validator("aadhaar_number")
    @classmethod
    def validate(cls, v: str) -> str:
        if not _AADHAAR_RE.match(v) or len(set(v)) == 1:
            raise ValueError("Invalid Aadhaar number format")
        return v


@router.post("/aadhaar", status_code=200)
async def register_aadhaar(
    body: AadhaarRequest,
    payload: Annotated[dict, Depends(get_current_user)],
    db: Annotated[DatabaseAdapter, Depends(get_db)],
) -> dict:
    user_id = payload["sub"]
    user = await get_user_by_id(db, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    if not user.face_registered:
        raise HTTPException(
            status_code=400,
            detail="Face must be registered before Aadhaar verification.",
        )

    if user.is_registered:
        new_token = await create_and_cache_token(
            user_id=str(user.id),
            phone=user.phone,
            role="user",
            is_registered=True,
        )
        return {"status": "already_registered", "token": new_token}

    # DB duplicate check first (fast, no API call)
    duplicate = await get_user_by_aadhaar(db, body.aadhaar_number)
    if duplicate and duplicate.id != user.id:
        raise HTTPException(status_code=409, detail="Aadhaar number already registered.")

    # Aadhaar API verification via strategy
    from main import app as _app  # type: ignore[import]
    verifier: AadhaarVerifierBackend = _app.state.aadhaar_verifier

    try:
        is_valid = await verifier.verify(body.aadhaar_number)
    except AadhaarAPIFailedError as exc:
        log.error("Aadhaar API failed", user_id=user_id, error_code=exc.error_code)
        raise HTTPException(
            status_code=503,
            detail="Aadhaar verification service unavailable. Try again later.",
        )

    if not is_valid:
        raise HTTPException(status_code=422, detail="Aadhaar number could not be verified.")

    user = await update_user_aadhaar(db, user, body.aadhaar_number)

    new_token = await create_and_cache_token(
        user_id=str(user.id),
        phone=user.phone,
        role="user",
        is_registered=True,
    )

    log.info("Aadhaar registration complete", user_id=user_id)
    return {"status": "registered", "token": new_token}


@router.get("/status", status_code=200)
async def registration_status(
    payload: Annotated[dict, Depends(get_current_user)],
    db: Annotated[DatabaseAdapter, Depends(get_db)],
) -> dict:
    user_id = payload["sub"]
    user = await get_user_by_id(db, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    if user.is_registered:
        pending_step = None
    elif not user.face_registered:
        pending_step = "face"
    else:
        pending_step = "aadhaar"

    return {
        "phone_verified": True,
        "face_registered": user.face_registered,
        "aadhaar_verified": user.aadhaar_verified,
        "is_registered": user.is_registered,
        "pending_step": pending_step,
    }
