"""
Admin auth routes: OTP-based login for officers and super-admins.

  POST /admin/auth/send-otp   — sends OTP only if phone exists in admin_users table
  POST /admin/auth/verify-otp — verifies OTP and returns JWT with role (officer | super_admin)

Uses a namespaced Redis key ("admin:<phone>") to avoid collision with citizen OTPs.
"""
from __future__ import annotations

import re
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from auth.jwt import create_and_cache_token
from auth.otp import generate_otp, send_otp_via_twilio
from cache.redis_client import (
    check_otp_rate_limit,
    delete_otp,
    get_otp_data,
    increment_otp_attempts,
    store_otp,
)
from core.exceptions import OTPSendFailedError
from core.logging import get_logger
from db.adapter import DatabaseAdapter
from db.crud import get_admin_by_phone
from db.session import get_db

log = get_logger(__name__)
router = APIRouter(prefix="/admin/auth", tags=["admin-auth"])

_PHONE_RE = re.compile(r"^\+91[6-9]\d{9}$")
OTP_MAX_WRONG = 3
_ADMIN_OTP_NS = "admin"  # Redis key prefix: "admin:<phone>"


class SendOTPRequest(BaseModel):
    phone: str = Field(..., description="Phone in +91XXXXXXXXXX format")

    @field_validator("phone")
    @classmethod
    def validate_phone(cls, v: str) -> str:
        if not _PHONE_RE.match(v):
            raise ValueError("Phone must be in format +91XXXXXXXXXX (Indian mobile)")
        return v


class VerifyOTPRequest(BaseModel):
    phone: str
    otp: str = Field(..., min_length=6, max_length=6, pattern=r"^\d{6}$")


def _otp_key(phone: str) -> str:
    return f"{_ADMIN_OTP_NS}:{phone}"


@router.post("/send-otp", status_code=200)
async def admin_send_otp(
    body: SendOTPRequest,
    db: Annotated[DatabaseAdapter, Depends(get_db)],
) -> dict:
    admin = await get_admin_by_phone(db, body.phone)

    # Always return success to avoid phone enumeration; OTP only actually stored for valid admins
    if admin is None or not admin.is_active:
        log.warning("OTP request for unknown/inactive admin phone", phone=body.phone)
        return {"message": "OTP sent", "expires_in": 600}

    allowed = await check_otp_rate_limit(_otp_key(body.phone))
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many OTP requests. Try again in an hour.",
        )

    otp = generate_otp()
    await store_otp(_otp_key(body.phone), otp)

    try:
        await send_otp_via_twilio(body.phone, otp)
    except OTPSendFailedError as exc:
        log.error("Admin OTP send failed", phone=body.phone, error_code=exc.error_code)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not send OTP. Please try again.",
        )

    log.info("Admin OTP sent", phone=body.phone, role=admin.role)
    return {"message": "OTP sent", "expires_in": 600}


@router.post("/verify-otp", status_code=200)
async def admin_verify_otp(
    body: VerifyOTPRequest,
    db: Annotated[DatabaseAdapter, Depends(get_db)],
) -> dict:
    admin = await get_admin_by_phone(db, body.phone)
    if admin is None or not admin.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized.")

    otp_data = await get_otp_data(_otp_key(body.phone))
    if otp_data is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="OTP expired or not found. Request a new OTP.",
        )

    if otp_data["attempts"] >= OTP_MAX_WRONG:
        await delete_otp(_otp_key(body.phone))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Too many wrong attempts. Request a new OTP.",
        )

    if otp_data["otp"] != body.otp:
        attempts = await increment_otp_attempts(_otp_key(body.phone))
        remaining = OTP_MAX_WRONG - attempts
        log.warning("Wrong admin OTP", phone=body.phone, attempts=attempts)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid OTP. {remaining} attempt(s) remaining.",
        )

    await delete_otp(_otp_key(body.phone))

    role = "super_admin" if admin.role == "SUPER_ADMIN" else "officer"

    token = await create_and_cache_token(
        user_id=str(admin.id),
        phone=admin.phone,
        role=role,
        is_registered=True,
        is_active=admin.is_active,
    )

    log.info("Admin authenticated", phone=body.phone, role=role, admin_id=str(admin.id))
    return {"token": token, "role": role, "name": admin.name}
