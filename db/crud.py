"""
All database operations for NammaKelsa.

Routes and services call ONLY these functions — never raw SQLAlchemy in routes.
Every function accepts a DatabaseAdapter, so tests can inject a mock adapter.
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any
from uuid import UUID

from sqlalchemy import select, func

from core.logging import get_logger
from db.adapter import DatabaseAdapter
from db.models import (
    AdminUser,
    CandidateApplication,
    Document,
    InterviewSession,
    MLVerdict,
    Skill,
    User,
)

log = get_logger(__name__)


# ── Users ──────────────────────────────────────────────────────────────────────

async def get_user_by_id(db: DatabaseAdapter, user_id: UUID) -> User | None:
    return await db.get(User, user_id)


async def get_user_by_phone(db: DatabaseAdapter, phone: str) -> User | None:
    return await db.get_by(User, phone=phone)


async def get_user_by_aadhaar(db: DatabaseAdapter, aadhaar_number: str) -> User | None:
    return await db.get_by(User, aadhaar_number=aadhaar_number)


async def get_user_by_face_hash(db: DatabaseAdapter, face_hash: str) -> User | None:
    return await db.get_by(User, face_hash=face_hash)


async def create_user(db: DatabaseAdapter, phone: str) -> User:
    user = User(phone=phone)
    return await db.create(user)


async def update_user_face(
    db: DatabaseAdapter,
    user: User,
    face_hash: str,
    face_embedding_id: str | None,
) -> User:
    return await db.update(
        user,
        face_hash=face_hash,
        face_embedding_id=face_embedding_id,
        face_registered=True,
    )


async def update_user_aadhaar(
    db: DatabaseAdapter,
    user: User,
    aadhaar_number: str,
) -> User:
    return await db.update(
        user,
        aadhaar_number=aadhaar_number,
        aadhaar_verified=True,
        is_registered=True,
    )


async def deactivate_user(db: DatabaseAdapter, user: User) -> User:
    log.warning("Deactivating user due to FRAUD verdict", user_id=str(user.id))
    return await db.update(user, is_active=False)


# ── Skills ──────────────────────────────────────────────────────────────────────

async def list_skills(db: DatabaseAdapter) -> list[Skill]:
    return await db.list_by(Skill)


async def get_skill_by_id(db: DatabaseAdapter, skill_id: UUID) -> Skill | None:
    return await db.get(Skill, skill_id)


async def get_skill_by_name(db: DatabaseAdapter, name: str) -> Skill | None:
    return await db.get_by(Skill, name=name)


async def seed_skills(db: DatabaseAdapter) -> None:
    skill_names = [
        ("electrician", "Electrical installation and maintenance work"),
        ("plumber", "Plumbing installation and repair"),
        ("welder", "Metal welding — MIG, TIG, Arc, Gas"),
        ("carpenter", "Woodwork, furniture, and structural carpentry"),
        ("mason", "Brickwork, plastering, tiling, and masonry"),
        ("labour", "General construction site labour"),
    ]
    for name, description in skill_names:
        exists = await db.exists(Skill, name=name)
        if not exists:
            await db.create(Skill(name=name, description=description))
            log.info("Skill seeded", skill_name=name)


# ── Applications ───────────────────────────────────────────────────────────────

async def get_application_by_id(
    db: DatabaseAdapter, application_id: UUID
) -> CandidateApplication | None:
    return await db.get(CandidateApplication, application_id)


async def get_applications_by_user(
    db: DatabaseAdapter, user_id: UUID
) -> list[CandidateApplication]:
    return await db.list_by(CandidateApplication, user_id=user_id)


async def get_applications_by_user_and_skill(
    db: DatabaseAdapter, user_id: UUID, skill_id: UUID
) -> list[CandidateApplication]:
    stmt = (
        select(CandidateApplication)
        .where(
            CandidateApplication.user_id == user_id,
            CandidateApplication.skill_id == skill_id,
        )
        .order_by(CandidateApplication.created_at.asc())
    )
    result = await db.execute_query(stmt)
    return list(result.scalars().all())


async def get_latest_application_for_skill(
    db: DatabaseAdapter, user_id: UUID, skill_id: UUID
) -> CandidateApplication | None:
    stmt = (
        select(CandidateApplication)
        .where(
            CandidateApplication.user_id == user_id,
            CandidateApplication.skill_id == skill_id,
        )
        .order_by(CandidateApplication.created_at.desc())
        .limit(1)
    )
    result = await db.execute_query(stmt)
    return result.scalars().first()


async def count_attempts(db: DatabaseAdapter, user_id: UUID, skill_id: UUID) -> int:
    stmt = select(func.count()).where(
        CandidateApplication.user_id == user_id,
        CandidateApplication.skill_id == skill_id,
    )
    result = await db.execute_query(stmt)
    return result.scalar() or 0


async def create_application(
    db: DatabaseAdapter,
    user_id: UUID,
    skill_id: UUID,
    attempt_number: int,
) -> CandidateApplication:
    app = CandidateApplication(
        user_id=user_id,
        skill_id=skill_id,
        attempt_number=attempt_number,
        status="DOCUMENTS_PENDING",
    )
    return await db.create(app)


async def update_application_status(
    db: DatabaseAdapter, application: CandidateApplication, status: str
) -> CandidateApplication:
    log.info(
        "Application status updated",
        application_id=str(application.id),
        old_status=application.status,
        new_status=status,
    )
    return await db.update(application, status=status)


async def set_application_cooldown(
    db: DatabaseAdapter,
    application: CandidateApplication,
    cooldown_days: int = 60,
) -> CandidateApplication:
    cooldown_until = datetime.now(tz=timezone.utc) + timedelta(days=cooldown_days)
    return await db.update(
        application,
        cooldown_until=cooldown_until,
        status="INTERVIEW_DONE",
    )


async def get_processing_applications(
    db: DatabaseAdapter,
) -> list[CandidateApplication]:
    """Return all applications currently being processed by ML service."""
    return await db.list_by(CandidateApplication, status="ML_PROCESSING")


# ── Interview Sessions ─────────────────────────────────────────────────────────

async def get_session_by_id(
    db: DatabaseAdapter, session_id: UUID
) -> InterviewSession | None:
    return await db.get(InterviewSession, session_id)


async def get_active_session_for_application(
    db: DatabaseAdapter, application_id: UUID
) -> InterviewSession | None:
    return await db.get_by(
        InterviewSession, application_id=application_id, status="ACTIVE"
    )


async def create_interview_session(
    db: DatabaseAdapter,
    application_id: UUID,
    language: str = "kn",
) -> InterviewSession:
    session = InterviewSession(application_id=application_id, status="ACTIVE", language=language)
    return await db.create(session)


async def complete_interview_session(
    db: DatabaseAdapter,
    session: InterviewSession,
    transcript: list[dict],
) -> InterviewSession:
    return await db.update(
        session,
        status="COMPLETED",
        ended_at=datetime.now(tz=timezone.utc),
        transcript=transcript,
    )


async def abandon_interview_session(
    db: DatabaseAdapter, session: InterviewSession
) -> InterviewSession:
    return await db.update(
        session,
        status="ABANDONED",
        ended_at=datetime.now(tz=timezone.utc),
    )


# ── Documents ──────────────────────────────────────────────────────────────────

async def create_document(
    db: DatabaseAdapter,
    user_id: UUID,
    application_id: UUID,
    doc_type: str,
    file_path: str,
) -> Document:
    doc = Document(
        user_id=user_id,
        application_id=application_id,
        doc_type=doc_type,
        file_path=file_path,
    )
    return await db.create(doc)


async def get_documents_for_application(
    db: DatabaseAdapter, application_id: UUID
) -> list[Document]:
    return await db.list_by(Document, application_id=application_id)


# ── ML Verdicts ────────────────────────────────────────────────────────────────

async def upsert_ml_verdict(
    db: DatabaseAdapter,
    application_id: UUID,
    session_id: UUID | None,
    verdict_data: dict,
) -> MLVerdict:
    existing = await db.get_by(MLVerdict, application_id=application_id)
    if existing:
        return await db.update(
            existing,
            session_id=session_id,
            verdict=verdict_data.get("verdict"),
            composite_score=verdict_data.get("composite_score"),
            fraud_score=verdict_data.get("fraud_score"),
            fraud_level=verdict_data.get("fraud_level"),
            domain_score=verdict_data.get("domain_score"),
            communication_score=verdict_data.get("communication_score"),
            skill_confidence=verdict_data.get("skill_confidence"),
            full_result=verdict_data,
            fetched_at=datetime.now(tz=timezone.utc),
        )

    verdict = MLVerdict(
        application_id=application_id,
        session_id=session_id,
        verdict=verdict_data.get("verdict"),
        composite_score=verdict_data.get("composite_score"),
        fraud_score=verdict_data.get("fraud_score"),
        fraud_level=verdict_data.get("fraud_level"),
        domain_score=verdict_data.get("domain_score"),
        communication_score=verdict_data.get("communication_score"),
        skill_confidence=verdict_data.get("skill_confidence"),
        full_result=verdict_data,
    )
    return await db.create(verdict)


# ── Admin ──────────────────────────────────────────────────────────────────────

async def get_admin_by_phone(db: DatabaseAdapter, phone: str) -> AdminUser | None:
    return await db.get_by(AdminUser, phone=phone)


async def create_admin_user(
    db: DatabaseAdapter, phone: str, name: str, role: str = "OFFICER"
) -> AdminUser:
    admin = AdminUser(phone=phone, name=name, role=role)
    return await db.create(admin)


async def list_candidates_paginated(
    db: DatabaseAdapter,
    page: int = 1,
    page_size: int = 50,
    skill_name: str | None = None,
    status: str | None = None,
) -> tuple[list[User], int]:
    from sqlalchemy import distinct

    stmt = select(User).join(CandidateApplication, CandidateApplication.user_id == User.id)

    if skill_name:
        stmt = stmt.join(Skill, Skill.id == CandidateApplication.skill_id).where(
            Skill.name == skill_name
        )
    if status:
        stmt = stmt.where(CandidateApplication.status == status)

    stmt = stmt.distinct().order_by(User.created_at.desc())

    count_stmt = select(func.count(distinct(User.id))).join(
        CandidateApplication, CandidateApplication.user_id == User.id
    )
    if skill_name:
        count_stmt = count_stmt.join(Skill, Skill.id == CandidateApplication.skill_id).where(
            Skill.name == skill_name
        )
    if status:
        count_stmt = count_stmt.where(CandidateApplication.status == status)

    total_result = await db.execute_query(count_stmt)
    total = total_result.scalar() or 0

    stmt = stmt.offset((page - 1) * page_size).limit(page_size)
    result = await db.execute_query(stmt)
    return list(result.scalars().all()), total
