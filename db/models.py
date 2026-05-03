"""
SQLAlchemy ORM models for NammaKelsa backend.

All tables use UUID primary keys and UTC timestamps.
pgvector extension required for face_embedding column on users table.
"""
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, relationship

try:
    from pgvector.sqlalchemy import Vector  # type: ignore[import]
    _HAS_PGVECTOR = True
except ImportError:
    _HAS_PGVECTOR = False
    Vector = None


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    phone = Column(String(15), unique=True, nullable=False, index=True)
    aadhaar_number = Column(String(12), unique=True, nullable=True)
    aadhaar_verified = Column(Boolean, default=False, nullable=False)
    face_hash = Column(String(64), unique=True, nullable=True)
    face_embedding_id = Column(String(100), nullable=True)
    face_registered = Column(Boolean, default=False, nullable=False)
    is_registered = Column(Boolean, default=False, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    preferred_language = Column(String(5), nullable=False, default="kn")  # kn | hi | en | te | ta
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)

    # pgvector column — only added if extension is available
    if _HAS_PGVECTOR:
        face_embedding = Column(Vector(512), nullable=True)

    applications = relationship("CandidateApplication", back_populates="user", lazy="select")
    documents = relationship("Document", back_populates="user", lazy="select")


class Skill(Base):
    __tablename__ = "skills"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(50), unique=True, nullable=False)
    description = Column(Text, nullable=True)

    applications = relationship("CandidateApplication", back_populates="skill", lazy="select")


class CandidateApplication(Base):
    __tablename__ = "candidate_applications"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    skill_id = Column(UUID(as_uuid=True), ForeignKey("skills.id"), nullable=False, index=True)
    attempt_number = Column(Integer, nullable=False)
    status = Column(String(30), nullable=False, default="DOCUMENTS_PENDING", index=True)
    cooldown_until = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at = Column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )

    __table_args__ = (
        UniqueConstraint("user_id", "skill_id", "attempt_number", name="uq_app_user_skill_attempt"),
        CheckConstraint("attempt_number BETWEEN 1 AND 3", name="chk_attempt_range"),
    )

    user = relationship("User", back_populates="applications")
    skill = relationship("Skill", back_populates="applications")
    sessions = relationship("InterviewSession", back_populates="application", lazy="select")
    documents = relationship("Document", back_populates="application", lazy="select")
    verdict = relationship(
        "MLVerdict", back_populates="application", uselist=False, lazy="select"
    )


class InterviewSession(Base):
    __tablename__ = "interview_sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    application_id = Column(
        UUID(as_uuid=True), ForeignKey("candidate_applications.id"), nullable=False, index=True
    )
    transcript = Column(JSONB, nullable=True)
    language = Column(String(5), nullable=False, default="kn")  # kn | hi | en | te | ta
    started_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    ended_at = Column(DateTime(timezone=True), nullable=True)
    status = Column(String(20), nullable=False, default="ACTIVE")  # ACTIVE | COMPLETED | ABANDONED

    application = relationship("CandidateApplication", back_populates="sessions")
    verdict = relationship("MLVerdict", back_populates="session", uselist=False, lazy="select")


class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    application_id = Column(
        UUID(as_uuid=True), ForeignKey("candidate_applications.id"), nullable=False, index=True
    )
    doc_type = Column(String(20), nullable=False)  # DEGREE | WORKEXP_VIDEO
    file_path = Column(Text, nullable=False)
    uploaded_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)

    user = relationship("User", back_populates="documents")
    application = relationship("CandidateApplication", back_populates="documents")


class MLVerdict(Base):
    __tablename__ = "ml_verdicts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    application_id = Column(
        UUID(as_uuid=True),
        ForeignKey("candidate_applications.id"),
        unique=True,
        nullable=False,
        index=True,
    )
    session_id = Column(
        UUID(as_uuid=True), ForeignKey("interview_sessions.id"), nullable=True
    )
    verdict = Column(
        String(30), nullable=True
    )  # READY | SKILL_TRAINING | MANUAL_VERIFICATION | UNSKILLED | FRAUD
    composite_score = Column(Float, nullable=True)
    fraud_score = Column(Float, nullable=True)
    fraud_level = Column(String(20), nullable=True)
    domain_score = Column(Float, nullable=True)
    communication_score = Column(Float, nullable=True)
    skill_confidence = Column(Float, nullable=True)
    full_result = Column(JSONB, nullable=True)
    fetched_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)

    application = relationship("CandidateApplication", back_populates="verdict")
    session = relationship("InterviewSession", back_populates="verdict")


class AdminUser(Base):
    __tablename__ = "admin_users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    phone = Column(String(15), unique=True, nullable=False)
    name = Column(String(100), nullable=True)
    role = Column(String(20), nullable=False, default="OFFICER")  # OFFICER | SUPER_ADMIN
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
