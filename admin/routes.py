"""
Admin routes (requires admin JWT):
  GET  /admin/candidates              — list all candidates (paginated)
  GET  /admin/candidate/{user_id}     — full candidate detail
  PUT  /admin/verdict/{app_id}/override — override ML verdict
  GET  /admin/stats                   — dashboard stats
  POST /admin/users                   — create officer (super_admin only)
"""
from __future__ import annotations

from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from auth.middleware import get_admin_user, get_super_admin_user
from core.logging import get_logger
from db.adapter import DatabaseAdapter
from db.crud import (
    create_admin_user,
    get_application_by_id,
    get_applications_by_user,
    get_documents_for_application,
    get_session_by_id,
    get_user_by_id,
    list_candidates_paginated,
    update_application_status,
    upsert_ml_verdict,
)
from db.session import get_db

log = get_logger(__name__)
router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/candidates", status_code=200)
async def list_candidates(
    payload: Annotated[dict, Depends(get_admin_user)],
    db: Annotated[DatabaseAdapter, Depends(get_db)],
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    skill: Optional[str] = None,
    status: Optional[str] = None,
) -> dict:
    users, total = await list_candidates_paginated(db, page, page_size, skill, status)
    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "candidates": [
            {
                "user_id": str(u.id),
                "phone": u.phone,
                "face_registered": u.face_registered,
                "is_registered": u.is_registered,
                "is_active": u.is_active,
                "created_at": u.created_at.isoformat(),
            }
            for u in users
        ],
    }


@router.get("/candidate/{user_id}", status_code=200)
async def get_candidate_detail(
    user_id: str,
    payload: Annotated[dict, Depends(get_admin_user)],
    db: Annotated[DatabaseAdapter, Depends(get_db)],
) -> dict:
    user = await get_user_by_id(db, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    apps = await get_applications_by_user(db, user.id)

    applications_detail = []
    for app in apps:
        docs = await get_documents_for_application(db, app.id)
        verdict_data = None
        if app.verdict:
            v = app.verdict
            verdict_data = {
                "verdict": v.verdict,
                "composite_score": v.composite_score,
                "fraud_score": v.fraud_score,
                "fraud_level": v.fraud_level,
                "domain_score": v.domain_score,
                "communication_score": v.communication_score,
                "skill_confidence": v.skill_confidence,
            }

        sessions = []
        for sess in app.sessions:
            sessions.append({
                "session_id": str(sess.id),
                "status": sess.status,
                "started_at": sess.started_at.isoformat(),
                "ended_at": sess.ended_at.isoformat() if sess.ended_at else None,
                "transcript": sess.transcript,
            })

        applications_detail.append({
            "application_id": str(app.id),
            "skill_id": str(app.skill_id),
            "attempt_number": app.attempt_number,
            "status": app.status,
            "cooldown_until": app.cooldown_until.isoformat() if app.cooldown_until else None,
            "created_at": app.created_at.isoformat(),
            "documents": [
                {"doc_type": d.doc_type, "uploaded_at": d.uploaded_at.isoformat()}
                for d in docs
            ],
            "sessions": sessions,
            "verdict": verdict_data,
        })

    # Mask Aadhaar — show only last 4 digits
    aadhaar_masked = None
    if user.aadhaar_number:
        aadhaar_masked = "XXXXXXXX" + user.aadhaar_number[-4:]

    return {
        "user_id": str(user.id),
        "phone": user.phone,
        "aadhaar_masked": aadhaar_masked,
        "aadhaar_verified": user.aadhaar_verified,
        "face_registered": user.face_registered,
        "is_registered": user.is_registered,
        "is_active": user.is_active,
        "created_at": user.created_at.isoformat(),
        "applications": applications_detail,
    }


class OverrideVerdictRequest(BaseModel):
    verdict: str
    reason: str


@router.put("/verdict/{application_id}/override", status_code=200)
async def override_verdict(
    application_id: str,
    body: OverrideVerdictRequest,
    payload: Annotated[dict, Depends(get_admin_user)],
    db: Annotated[DatabaseAdapter, Depends(get_db)],
) -> dict:
    valid_verdicts = {"READY", "SKILL_TRAINING", "MANUAL_VERIFICATION", "UNSKILLED", "FRAUD"}
    if body.verdict not in valid_verdicts:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid verdict. Must be one of: {', '.join(valid_verdicts)}",
        )

    app = await get_application_by_id(db, application_id)
    if app is None:
        raise HTTPException(status_code=404, detail="Application not found")

    import uuid
    await upsert_ml_verdict(
        db,
        application_id=uuid.UUID(application_id),
        session_id=None,
        verdict_data={
            "verdict": body.verdict,
            "override_reason": body.reason,
            "overridden_by": payload.get("sub"),
        },
    )
    await update_application_status(db, app, body.verdict)

    log.info(
        "Verdict overridden by admin",
        application_id=application_id,
        new_verdict=body.verdict,
        admin_id=payload.get("sub"),
        reason=body.reason,
    )

    return {"status": "overridden", "verdict": body.verdict}


@router.get("/stats", status_code=200)
async def get_stats(
    payload: Annotated[dict, Depends(get_admin_user)],
    db: Annotated[DatabaseAdapter, Depends(get_db)],
) -> dict:
    from sqlalchemy import func, select
    from db.models import CandidateApplication, User

    verdict_stmt = select(
        CandidateApplication.status, func.count()
    ).group_by(CandidateApplication.status)
    result = await db.execute_query(verdict_stmt)
    counts_by_status = {row[0]: row[1] for row in result.fetchall()}

    user_count_stmt = select(func.count()).select_from(User)
    user_result = await db.execute_query(user_count_stmt)
    total_users = user_result.scalar() or 0

    return {
        "total_users": total_users,
        "applications_by_status": counts_by_status,
    }


class CreateOfficerRequest(BaseModel):
    phone: str
    name: str
    role: str = "OFFICER"


@router.post("/users", status_code=201)
async def create_officer(
    body: CreateOfficerRequest,
    payload: Annotated[dict, Depends(get_super_admin_user)],
    db: Annotated[DatabaseAdapter, Depends(get_db)],
) -> dict:
    if body.role not in ("OFFICER", "SUPER_ADMIN"):
        raise HTTPException(status_code=400, detail="Role must be OFFICER or SUPER_ADMIN")

    admin = await create_admin_user(db, body.phone, body.name, body.role)
    log.info("Admin officer created", admin_id=str(admin.id), phone=body.phone, role=body.role)
    return {"admin_id": str(admin.id), "phone": admin.phone, "role": admin.role}
