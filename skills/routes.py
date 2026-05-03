"""
Skill routes:
  GET  /skills                 — list all 6 skills (public)
  POST /skills/choose          — select skill for application
  GET  /skills/my-applications — list user's applications with status
"""
from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from auth.middleware import get_registered_user
from core.exceptions import InCooldownError, MaxAttemptsReachedError, UserInactiveError
from core.logging import get_logger
from db.adapter import DatabaseAdapter
from db.crud import (
    create_application,
    get_applications_by_user,
    get_skill_by_id,
    get_user_by_id,
    list_skills,
)
from db.session import get_db
from skills.cooldown import check_skill_eligibility

log = get_logger(__name__)
router = APIRouter(prefix="/skills", tags=["skills"])


@router.get("", status_code=200)
async def get_skills(db: Annotated[DatabaseAdapter, Depends(get_db)]) -> dict:
    skills = await list_skills(db)
    return {
        "skills": [
            {"id": str(s.id), "name": s.name, "description": s.description}
            for s in skills
        ]
    }


class ChooseSkillRequest(BaseModel):
    skill_id: UUID


@router.post("/choose", status_code=201)
async def choose_skill(
    body: ChooseSkillRequest,
    payload: Annotated[dict, Depends(get_registered_user)],
    db: Annotated[DatabaseAdapter, Depends(get_db)],
) -> dict:
    user_id = payload["sub"]
    user = await get_user_by_id(db, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    try:
        next_attempt = await check_skill_eligibility(db, user, body.skill_id)
    except UserInactiveError as exc:
        raise HTTPException(status_code=403, detail=exc.message)
    except MaxAttemptsReachedError as exc:
        raise HTTPException(status_code=409, detail=exc.message)
    except InCooldownError as exc:
        raise HTTPException(
            status_code=429,
            detail=exc.message,
            headers={"X-Cooldown-Until": exc.detail.get("cooldown_until", "")},
        )
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    application = await create_application(db, user.id, body.skill_id, next_attempt)

    skill = await get_skill_by_id(db, body.skill_id)
    log.info(
        "Application created",
        user_id=user_id,
        skill_name=skill.name if skill else "?",
        attempt=next_attempt,
        application_id=str(application.id),
    )

    return {
        "application_id": str(application.id),
        "skill_id": str(body.skill_id),
        "attempt_number": next_attempt,
        "status": application.status,
    }


@router.get("/my-applications", status_code=200)
async def my_applications(
    payload: Annotated[dict, Depends(get_registered_user)],
    db: Annotated[DatabaseAdapter, Depends(get_db)],
) -> dict:
    user_id = payload["sub"]
    apps = await get_applications_by_user(db, user_id)
    return {
        "applications": [
            {
                "application_id": str(a.id),
                "skill_id": str(a.skill_id),
                "attempt_number": a.attempt_number,
                "status": a.status,
                "cooldown_until": a.cooldown_until.isoformat() if a.cooldown_until else None,
                "created_at": a.created_at.isoformat(),
            }
            for a in apps
        ]
    }
